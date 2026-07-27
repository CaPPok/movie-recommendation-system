"""Turn runtime interaction events into a single preference score per movie.

Offline training only ever sees ratings, because that is all the dataset holds.
A live system sees clicks, watches, likes and dislikes as well, and those carry
very different amounts of evidence: finishing a film says far more than opening
its detail page. This module converts a mixed event stream into one comparable
number per movie so the ranking layer does not have to know about event types.

Two rules shape the design.

Recency matters. Someone who loved thrillers three years ago and romance since
should get romance today, so every event decays exponentially with age instead of
counting forever.

Absence is not rejection. A movie a user never touched is unknown, never
disliked. Only an explicit negative signal produces a negative score.
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
)


@dataclass
class InteractionProfile:
    """Per-movie preference scores derived from one user's recent events."""

    scores: dict[int, float] = field(default_factory=dict)
    disliked: set[int] = field(default_factory=set)
    ignored_events: int = 0
    unsupported_types: set[str] = field(default_factory=set)

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


def _event_weight(
    event_type: str,
    value: Any,
    settings: Mapping[str, Any],
) -> float | None:
    """Weight for one event, or None when it should not count at all."""
    weights: Mapping[str, Any] = settings["weights"]

    if event_type == "rating":
        # A rating carries its own magnitude, so a fixed weight would throw away
        # the difference between one star and five.
        if value is None:
            return None
        neutral = float(settings["rating_neutral_point"])
        scale = float(settings["rating_scale"])
        return (float(value) - neutral) * scale

    if event_type == "watch":
        # A few seconds of playback is not evidence of interest; only count a
        # watch once it passes the configured share of the runtime.
        threshold = float(settings["watch_progress_threshold"])
        if value is None or float(value) < threshold:
            return None

    weight = weights.get(event_type)
    return None if weight is None else float(weight)


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
    """
    now = now or datetime.now(timezone.utc)
    half_life = float(settings["half_life_days"])
    dislike_threshold = float(settings["dislike_score_threshold"])

    profile = InteractionProfile()
    for event in events:
        if not isinstance(event, Mapping):
            profile.ignored_events += 1
            continue
        raw_id = event.get("movie_id")
        event_type = str(event.get("event_type") or "").strip().lower()
        try:
            movie_id = int(raw_id)
        except (TypeError, ValueError):
            profile.ignored_events += 1
            continue
        if event_type not in SUPPORTED_EVENT_TYPES:
            profile.unsupported_types.add(event_type or "<empty>")
            profile.ignored_events += 1
            continue

        weight = _event_weight(event_type, event.get("value"), settings)
        if weight is None:
            profile.ignored_events += 1
            continue

        decayed = weight * _decay_factor(event.get("timestamp"), now, half_life)
        profile.scores[movie_id] = profile.scores.get(movie_id, 0.0) + decayed

    profile.disliked = {
        movie_id
        for movie_id, score in profile.scores.items()
        if score <= dislike_threshold
    }
    return profile
