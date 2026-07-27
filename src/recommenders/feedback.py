"""Backend-callable scoring of a raw interaction event stream.

`POST /model/recommend` already accepts `recent_interactions` and scores them
internally, but backend needs the same numbers for two things the recommend call
cannot answer:

* it has to persist a user preference profile in DynamoDB, so the next request
  does not have to resend a year of history;
* it has to fill `valid_interaction_count_90d` and choose a `scenario_hint`
  before it calls the model at all, and those depend on exactly which event
  types count — a rule that lives in `configs/model_serving.yaml`, not in the
  API layer.

Reimplementing either on the backend side would let the two definitions drift,
and a drifted `valid_interaction_count_90d` silently routes users to the wrong
model. So the rules are served from here instead.

This module loads no artifacts and touches no disk: it is a pure function over
the JSON body, cheap enough to call on every write of an interaction event.

Usage from backend:

    from src.data.config import load_config
    from src.recommenders.feedback import score_interaction_events

    model_config = load_config("configs/model_serving.yaml")   # once, at start-up
    result = score_interaction_events(request_body, model_config)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from src.models.interaction_weights import (
    SUPPORTED_EVENT_TYPES,
    build_interaction_profile,
    comment_sentiment,
)

SCENARIO_GUEST = "guest"
SCENARIO_ONBOARDING = "onboarding_user"
SCENARIO_RETURNING = "returning_user"


class InteractionPayloadError(ValueError):
    """Raised for a malformed payload; like the recommend contract, a 4xx."""


def _parse_timestamp(value: Any) -> datetime | None:
    """Best-effort ISO-8601 parse; unparseable stamps are treated as missing."""
    if value is None:
        return None
    if isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _int_set(values: Any) -> set[int]:
    """Integer movie ids from a possibly untyped array; unusable entries dropped."""
    result: set[int] = set()
    for value in values or []:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _require_events(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    events = payload.get("events")
    if events is None:
        events = payload.get("recent_interactions")
    if events is None:
        raise InteractionPayloadError("payload must contain an 'events' array")
    if isinstance(events, Mapping) or not isinstance(events, Sequence):
        raise InteractionPayloadError("'events' must be an array of objects")
    return list(events)


def count_valid_interactions(
    events: Iterable[Mapping[str, Any]],
    model_config: Mapping[str, Any],
    now: datetime | None = None,
) -> int:
    """Count events that qualify a user as having usable history.

    This is the number backend sends as `valid_interaction_count_90d`. Three
    filters apply: the event type must be in `scenario.valid_event_types`, the
    event must fall inside `scenario.interaction_recency_days`, and the event has
    to be one the scorer would actually count — a `watch` below the progress
    threshold is autoplay, and a `comment` with no sentiment produces no signal.

    That last filter matters more than it looks. Counting events the model then
    ignores would promote a user to `returning_user` on the strength of history
    that contributes nothing, and the collaborative model would have nothing to
    rank for them.

    An event with no timestamp is counted. Backend stamps every write, so a
    missing timestamp means an import or a bug, and dropping those rows would
    quietly downgrade real users to onboarding.
    """
    scenario = model_config["scenario"]
    interactions = model_config["interactions"]
    valid_types = {str(value).lower() for value in scenario["valid_event_types"]}
    window_days = float(scenario["interaction_recency_days"])
    threshold = float(interactions["watch_progress_threshold"])

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days) if window_days > 0 else None

    total = 0
    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_type = str(event.get("event_type") or "").strip().lower()
        if event_type not in valid_types:
            continue
        if event_type == "watch":
            try:
                progress = float(event.get("value"))
            except (TypeError, ValueError):
                continue
            if progress < threshold:
                continue
        if event_type == "comment" and comment_sentiment(event.get("value")) is None:
            continue
        stamp = _parse_timestamp(event.get("timestamp"))
        if cutoff is not None and stamp is not None and stamp < cutoff:
            continue
        total += 1
    return total


def suggest_scenario_hint(
    user_id: Any,
    onboarding_completed: bool,
    valid_interaction_count: int,
    model_config: Mapping[str, Any],
) -> str:
    """Backend's decision table from spec section 10.2, in code.

    The model may still downgrade the result — that is `scenario_applied`, and
    it is the engine's call, not this function's.
    """
    if user_id in (None, ""):
        return SCENARIO_GUEST
    if not onboarding_completed:
        return SCENARIO_GUEST
    minimum = int(model_config["scenario"]["min_interactions_for_cf"])
    if valid_interaction_count < minimum:
        return SCENARIO_ONBOARDING
    return SCENARIO_RETURNING


def score_interaction_events(
    payload: Mapping[str, Any],
    model_config: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Score one user's event stream and return a JSON-serialisable profile.

    Expected body (all fields other than `events` optional)::

        {
          "user_id": 12,
          "as_of": "2026-07-27T10:00:00Z",
          "onboarding_completed": true,
          "events": [
            {"movie_id": 862, "event_type": "share",   "timestamp": "..."},
            {"movie_id": 862, "event_type": "comment", "value": -0.8,
             "timestamp": "..."},
            {"movie_id": 550, "event_type": "watch",   "value": 0.82,
             "timestamp": "..."}
          ]
        }

    `as_of` exists so a batch job can rescore a historical stream and get the
    same answer twice; without it the decay is measured from wall-clock now and
    the result changes between runs.
    """
    if not isinstance(payload, Mapping):
        raise InteractionPayloadError("payload must be a JSON object")

    events = _require_events(payload)
    interactions = model_config["interactions"]
    max_events = int(interactions["max_recent_events"])
    # Truncating rather than rejecting: a user with a long history is not an
    # error, and the events are newest-first, so the cut drops the least
    # relevant ones.
    truncated = max(len(events) - max_events, 0)
    events = events[:max_events]

    as_of = _parse_timestamp(payload.get("as_of")) or now or datetime.now(timezone.utc)
    profile = build_interaction_profile(events, interactions, now=as_of)

    weights = profile.normalised_weights()
    movie_scores = [
        {
            "movie_id": movie_id,
            "score": round(score, 6),
            "normalised_weight": round(weights.get(movie_id, 0.0), 6),
            "disliked": movie_id in profile.disliked,
            "events_counted": sum(profile.events_by_movie.get(movie_id, {}).values()),
            "event_types": sorted(profile.events_by_movie.get(movie_id, {})),
        }
        for movie_id, score in sorted(
            profile.scores.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    valid_count = count_valid_interactions(events, model_config, now=as_of)
    scenario_hint = suggest_scenario_hint(
        payload.get("user_id"),
        bool(payload.get("onboarding_completed")),
        valid_count,
        model_config,
    )

    return {
        "user_id": payload.get("user_id"),
        "scored_at": as_of.isoformat(timespec="seconds"),
        "half_life_days": float(interactions["half_life_days"]),
        "events_received": len(events) + truncated,
        "events_truncated": truncated,
        "events_counted": profile.counted_events,
        "events_ignored": profile.ignored_events,
        "events_capped": profile.capped_events,
        # Keyed by reason so backend can tell a shipping gap from a bug. A rising
        # `missing_sentiment` means comments are being collected but never
        # classified, so none of them reach the model.
        "events_ignored_by_reason": dict(profile.ignored_by_reason),
        # Reported rather than rejected: an unknown type is usually frontend
        # shipping a new event ahead of the model, and the request still has to
        # succeed. It belongs in backend logs so the mismatch gets noticed.
        "unsupported_event_types": sorted(profile.unsupported_types),
        "supported_event_types": list(SUPPORTED_EVENT_TYPES),
        "movie_scores": movie_scores,
        "positive_movie_ids": profile.positive_movie_ids(),
        "disliked_movie_ids": sorted(profile.disliked),
        "valid_interaction_count": valid_count,
        "suggested_scenario_hint": scenario_hint,
    }


def build_recommend_request(
    payload: Mapping[str, Any],
    model_config: Mapping[str, Any],
    limit: int | None = None,
    selected_movie_ids: Sequence[int] | None = None,
    selected_genres: Sequence[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Turn a stored event stream into a ready-to-send `/model/recommend` body.

    Convenience for the common backend path: read the user's events out of
    DynamoDB, hand them here, post the result to the engine. Disliked movies are
    pre-filled into `exclude_movie_ids` so the exclusion survives even if the
    stream is later trimmed below the point where the dislike is still visible.
    """
    scored = score_interaction_events(payload, model_config, now=now)
    events = _require_events(payload)
    hybrid = model_config["hybrid"]
    return {
        "user_id": payload.get("user_id"),
        "scenario_hint": scored["suggested_scenario_hint"],
        "onboarding_completed": bool(payload.get("onboarding_completed")),
        "valid_interaction_count_90d": scored["valid_interaction_count"],
        "selected_movie_ids": list(selected_movie_ids or []),
        "selected_genres": list(selected_genres or []),
        "recent_interactions": list(events)[
            : int(hybrid["max_recent_interactions"])
        ],
        "exclude_movie_ids": sorted(
            set(scored["disliked_movie_ids"]) | _int_set(payload.get("exclude_movie_ids"))
        ),
        "limit": int(limit or model_config["candidates"]["api_default_limit"]),
    }
