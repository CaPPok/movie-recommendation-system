"""Turn runtime interaction events into a single preference score per movie.

Offline training only ever sees ratings, because that is all the dataset holds.
A live system sees clicks, watches, likes, dislikes, shares and comments as
well, and those carry very different amounts of evidence: finishing a film says
far more than opening its detail page. This module converts a mixed event stream
into one comparable number per movie so the ranking layer does not have to know
about event types.

Three rules shape the design.

Recency matters. Someone who loved thrillers three years ago and romance since
should get romance today, so every event decays exponentially with age instead of
counting forever.

Absence is not rejection. A movie a user never touched is unknown, never
disliked. Only an explicit negative signal produces a negative score.

Effort is not the same as approval. Sharing and commenting cost the user far
more than a click, which is why they weigh heavily — but a comment can be
hostile, so its weight follows the sentiment backend attaches to it and is
allowed to come out negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

# Event types the system actually supports end to end. Adding one here is not
# enough on its own: the data dictionary, DynamoDB schema, backend API, frontend
# tracking and validation rules all have to agree, or the model will weight a
# signal nobody is sending.
SUPPORTED_EVENT_TYPES = (
    "click",
    "watch",
    "complete",
    "like",
    "dislike",
    "rating",
    "share",
    "comment",
)

# Why an event did not count. Reported per reason rather than as one total,
# because the fixes differ: a missing comment sentiment is a backend feature that
# has not shipped, while a malformed value is a bug.
IGNORED_MISSING_SENTIMENT = "missing_sentiment"
IGNORED_MALFORMED_VALUE = "malformed_value"
IGNORED_BELOW_THRESHOLD = "below_threshold"
IGNORED_NO_WEIGHT = "no_weight_configured"
IGNORED_UNSUPPORTED_TYPE = "unsupported_event_type"
IGNORED_BAD_MOVIE_ID = "unusable_movie_id"

# Sentiment labels accepted on a `comment` event, for backends whose sentiment
# service returns a class rather than a number. A numeric `value` works too.
# There is no fallback for an absent value: see `comment_sentiment`.
COMMENT_SENTIMENT_LABELS = {
    "positive": 1.0,
    "pos": 1.0,
    "neutral": 0.0,
    "mixed": 0.0,
    "negative": -1.0,
    "neg": -1.0,
}


@dataclass
class InteractionProfile:
    """Per-movie preference scores derived from one user's recent events."""

    scores: dict[int, float] = field(default_factory=dict)
    disliked: set[int] = field(default_factory=set)
    ignored_events: int = 0
    unsupported_types: set[str] = field(default_factory=set)
    counted_events: int = 0
    capped_events: int = 0
    # Why events were dropped, keyed by the IGNORED_* reasons above. Backend logs
    # this: a rising `missing_sentiment` count means comments are being collected
    # but never classified, so none of them reach the model.
    ignored_by_reason: dict[str, int] = field(default_factory=dict)
    # movie_id -> event_type -> how many events of that type were counted.
    # Backend persists this alongside the score so a profile can be explained
    # ("scored 21.4: two watches and a share") without replaying the raw stream.
    events_by_movie: dict[int, dict[str, int]] = field(default_factory=dict)

    def positive_movie_ids(self, limit: int | None = None) -> list[int]:
        """Movies with a net positive score, strongest first."""
        ranked = sorted(
            ((movie_id, score) for movie_id, score in self.scores.items() if score > 0),
            key=lambda item: (-item[1], item[0]),
        )
        ids = [movie_id for movie_id, _ in ranked]
        return ids[:limit] if limit else ids

    def normalised_weights(self) -> dict[int, float]:
        """Positive scores rescaled to 0-1 for use as blending weights."""
        positives = {
            movie_id: score for movie_id, score in self.scores.items() if score > 0
        }
        if not positives:
            return {}
        highest = max(positives.values())
        return {movie_id: score / highest for movie_id, score in positives.items()}


def _as_float(value: Any) -> float | None:
    """Numeric value of a payload field, or None when backend sent something else.

    A malformed field must not raise: the event stream arrives from the network
    and one bad record cannot be allowed to fail an entire recommendation
    request. The event is dropped and counted as ignored instead.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # drop NaN


def comment_sentiment(value: Any) -> float | None:
    """Sentiment of a comment in [-1, 1], or None when it cannot be determined.

    There is deliberately no default. A comment whose sentiment is unknown is
    unusable, not mildly positive: treating it as positive would score "worst
    film I have ever seen" as interest and recommend more of the same. Returning
    None drops the event instead, which is the only safe reading of an opinion
    nobody has classified.
    """
    if isinstance(value, str):
        sentiment = COMMENT_SENTIMENT_LABELS.get(value.strip().lower())
        if sentiment is None:
            sentiment = _as_float(value)
    else:
        sentiment = _as_float(value)
    if sentiment is None:
        return None
    return max(-1.0, min(1.0, sentiment))


def _event_weight(
    event_type: str,
    value: Any,
    settings: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    """Weight for one event as `(weight, ignore_reason)`.

    A weight of None means the event does not count, and the reason says why so
    the caller can report it. "Backend never classified this comment" and
    "backend sent a malformed number" need different fixes.
    """
    weights: Mapping[str, Any] = settings["weights"]

    if event_type == "rating":
        # A rating carries its own magnitude, so a fixed weight would throw away
        # the difference between one star and five.
        rating = _as_float(value)
        if rating is None:
            return None, IGNORED_MALFORMED_VALUE
        neutral = float(settings["rating_neutral_point"])
        scale = float(settings["rating_scale"])
        return (rating - neutral) * scale, None

    if event_type == "comment":
        # Effort and approval are different things. Writing a comment costs real
        # effort, so there is an engagement component; but only sentiment says
        # whether the effort was praise or a complaint, and a complaint is strong
        # evidence *against* recommending more of the same. Without sentiment
        # there is no way to tell the two apart, so the event is dropped.
        sentiment = comment_sentiment(value)
        if sentiment is None:
            return None, IGNORED_MISSING_SENTIMENT
        engagement = float(settings["comment_engagement_weight"])
        scale = float(settings["comment_sentiment_scale"])
        return engagement + scale * sentiment, None

    if event_type == "watch":
        # A few seconds of playback is not evidence of interest; only count a
        # watch once it passes the configured share of the runtime.
        threshold = float(settings["watch_progress_threshold"])
        progress = _as_float(value)
        if progress is None:
            return None, IGNORED_MALFORMED_VALUE
        if progress < threshold:
            return None, IGNORED_BELOW_THRESHOLD

    # `share` needs no special case: the act itself is the signal and its
    # `value` field, if backend sends one at all, only names the channel.
    weight = weights.get(event_type)
    if weight is None:
        return None, IGNORED_NO_WEIGHT
    return float(weight), None


def _decay_factor(
    timestamp: Any, now: datetime, half_life_days: float
) -> float:
    """Exponential decay so old preferences fade instead of accumulating forever."""
    if half_life_days <= 0 or timestamp is None:
        return 1.0
    try:
        stamp = (
            timestamp
            if isinstance(timestamp, datetime)
            else datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        )
    except (TypeError, ValueError):
        return 1.0
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    age_days = max((now - stamp).total_seconds() / 86_400.0, 0.0)
    return float(0.5 ** (age_days / half_life_days))


def build_interaction_profile(
    events: Iterable[Mapping[str, Any]],
    settings: Mapping[str, Any],
    now: datetime | None = None,
) -> InteractionProfile:
    """Collapse an event stream into one score per movie.

    Events are summed rather than replaced, so three clicks outweigh one, but the
    decay keeps a long-dormant burst from dominating today's recommendations.

    Summing without limit is safe for the passive signals but not for the
    deliberate ones: sharing a film to ten friends or leaving eight comments on
    it says roughly the same thing as doing it once, while an unbounded sum
    would let a single title crowd out the rest of the profile. Types listed in
    `max_events_per_movie` therefore stop counting past their cap. Events are
    expected newest-first, so the cap keeps the most recent occurrences.
    """
    now = now or datetime.now(timezone.utc)
    half_life = float(settings["half_life_days"])
    dislike_threshold = float(settings["dislike_score_threshold"])
    caps: Mapping[str, Any] = settings.get("max_events_per_movie") or {}

    profile = InteractionProfile()

    def ignore(reason: str) -> None:
        profile.ignored_events += 1
        profile.ignored_by_reason[reason] = profile.ignored_by_reason.get(reason, 0) + 1

    for event in events:
        if not isinstance(event, Mapping):
            ignore(IGNORED_BAD_MOVIE_ID)
            continue
        raw_id = event.get("movie_id")
        event_type = str(event.get("event_type") or "").strip().lower()
        try:
            movie_id = int(raw_id)
        except (TypeError, ValueError):
            ignore(IGNORED_BAD_MOVIE_ID)
            continue
        if event_type not in SUPPORTED_EVENT_TYPES:
            profile.unsupported_types.add(event_type or "<empty>")
            ignore(IGNORED_UNSUPPORTED_TYPE)
            continue

        weight, reason = _event_weight(event_type, event.get("value"), settings)
        if weight is None:
            ignore(reason or IGNORED_NO_WEIGHT)
            continue

        seen = profile.events_by_movie.setdefault(movie_id, {})
        cap = caps.get(event_type)
        if cap is not None and seen.get(event_type, 0) >= int(cap):
            profile.capped_events += 1
            continue

        decayed = weight * _decay_factor(event.get("timestamp"), now, half_life)
        profile.scores[movie_id] = profile.scores.get(movie_id, 0.0) + decayed
        seen[event_type] = seen.get(event_type, 0) + 1
        profile.counted_events += 1

    profile.disliked = {
        movie_id
        for movie_id, score in profile.scores.items()
        if score <= dislike_threshold
    }
    return profile
