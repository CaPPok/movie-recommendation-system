"""Runtime event scoring: the share and comment rules backend depends on."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.data.config import load_config
from src.models.interaction_weights import (
    IGNORED_MALFORMED_VALUE,
    IGNORED_MISSING_SENTIMENT,
    SUPPORTED_EVENT_TYPES,
    build_interaction_profile,
    comment_sentiment,
)
from src.recommenders.feedback import (
    InteractionPayloadError,
    build_recommend_request,
    count_valid_interactions,
    score_interaction_events,
    suggest_scenario_hint,
)

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def model_config() -> dict:
    return load_config("configs/model_serving.yaml")


@pytest.fixture(scope="module")
def settings(model_config: dict) -> dict:
    return model_config["interactions"]


def _event(movie_id: int, event_type: str, value=None, minutes_ago: int = 0) -> dict:
    event = {
        "movie_id": movie_id,
        "event_type": event_type,
        "timestamp": (NOW - timedelta(minutes=minutes_ago)).isoformat(),
    }
    if value is not None:
        event["value"] = value
    return event


def test_share_and_comment_are_supported(settings: dict) -> None:
    assert "share" in SUPPORTED_EVENT_TYPES
    assert "comment" in SUPPORTED_EVENT_TYPES
    # A comment needs sentiment to score; a share does not.
    profile = build_interaction_profile(
        [_event(1, "share"), _event(2, "comment", 0.6)], settings, now=NOW
    )
    assert profile.ignored_events == 0
    assert profile.counted_events == 2
    assert set(profile.scores) == {1, 2}


def test_share_weighs_as_much_as_a_like(settings: dict) -> None:
    share = build_interaction_profile([_event(1, "share")], settings, now=NOW)
    like = build_interaction_profile([_event(1, "like")], settings, now=NOW)
    assert share.scores[1] == pytest.approx(like.scores[1])


def test_comment_sentiment_moves_the_score_in_both_directions(settings: dict) -> None:
    praise = build_interaction_profile(
        [_event(1, "comment", 1.0)], settings, now=NOW
    ).scores[1]
    silent = build_interaction_profile(
        [_event(1, "comment", 0.0)], settings, now=NOW
    ).scores[1]
    complaint = build_interaction_profile(
        [_event(1, "comment", -1.0)], settings, now=NOW
    ).scores[1]

    assert praise > silent > 0 > complaint
    # A hostile comment must not seed "more films like this one".
    assert complaint < 0


def test_comment_accepts_sentiment_labels(settings: dict) -> None:
    numeric = build_interaction_profile(
        [_event(1, "comment", -1.0)], settings, now=NOW
    ).scores[1]
    labelled = build_interaction_profile(
        [_event(1, "comment", "NEGATIVE")], settings, now=NOW
    ).scores[1]
    assert labelled == pytest.approx(numeric)


def test_comment_without_sentiment_is_dropped_not_defaulted(settings: dict) -> None:
    """Scoring an unclassified opinion as positive is worse than ignoring it.

    A default above neutral would read "worst film I have ever seen" as interest
    and recommend more of the same, which is a wrong-signed signal rather than a
    missing one.
    """
    profile = build_interaction_profile([_event(1, "comment")], settings, now=NOW)
    assert profile.scores == {}
    assert profile.counted_events == 0
    assert profile.ignored_events == 1
    assert profile.ignored_by_reason == {IGNORED_MISSING_SENTIMENT: 1}


def test_unparseable_comment_sentiment_is_also_dropped(settings: dict) -> None:
    profile = build_interaction_profile(
        [_event(1, "comment", "rất hay"), _event(2, "comment", None)],
        settings,
        now=NOW,
    )
    assert profile.scores == {}
    assert profile.ignored_by_reason[IGNORED_MISSING_SENTIMENT] == 2


def test_comment_sentiment_helper_returns_none_when_unknown() -> None:
    assert comment_sentiment(None) is None
    assert comment_sentiment("không rõ") is None
    assert comment_sentiment(0.5) == pytest.approx(0.5)
    assert comment_sentiment("negative") == pytest.approx(-1.0)
    # Clamped, so a miscalibrated service cannot manufacture weight.
    assert comment_sentiment(9.0) == pytest.approx(1.0)


def test_share_still_counts_without_any_value(settings: dict) -> None:
    """Unlike a comment, the act of sharing is itself the signal."""
    profile = build_interaction_profile([_event(1, "share")], settings, now=NOW)
    assert profile.counted_events == 1
    assert profile.scores[1] == pytest.approx(float(settings["weights"]["share"]))


def test_comment_sentiment_is_clamped(settings: dict) -> None:
    """A miscalibrated sentiment service cannot manufacture unbounded weight."""
    clamped = build_interaction_profile(
        [_event(1, "comment", 50.0)], settings, now=NOW
    ).scores[1]
    maximum = build_interaction_profile(
        [_event(1, "comment", 1.0)], settings, now=NOW
    ).scores[1]
    assert clamped == pytest.approx(maximum)


def test_repeated_shares_stop_counting_at_the_cap(settings: dict) -> None:
    cap = int(settings["max_events_per_movie"]["share"])
    events = [_event(1, "share", minutes_ago=index) for index in range(cap + 4)]
    profile = build_interaction_profile(events, settings, now=NOW)

    single = build_interaction_profile([_event(1, "share")], settings, now=NOW).scores[1]
    assert profile.capped_events == 4
    assert profile.events_by_movie[1]["share"] == cap
    assert profile.scores[1] < single * (cap + 1)


def test_uncapped_types_still_accumulate(settings: dict) -> None:
    """The cap is deliberately per-type; clicks and watches are unchanged."""
    events = [_event(1, "click", minutes_ago=index) for index in range(6)]
    profile = build_interaction_profile(events, settings, now=NOW)
    assert profile.capped_events == 0
    assert profile.counted_events == 6


def test_malformed_value_is_ignored_not_raised(settings: dict) -> None:
    """One bad record from the network must not fail the whole request."""
    profile = build_interaction_profile(
        [_event(1, "rating", "four and a half"), _event(2, "share")],
        settings,
        now=NOW,
    )
    assert profile.ignored_events == 1
    assert profile.ignored_by_reason == {IGNORED_MALFORMED_VALUE: 1}
    assert set(profile.scores) == {2}


def test_valid_count_skips_comments_with_no_sentiment(model_config: dict) -> None:
    """Otherwise a user is promoted to returning_user on events that score zero.

    The collaborative model would then have nothing to rank for them.
    """
    events = [
        _event(1, "comment"),          # no sentiment: no signal, must not count
        _event(2, "comment", 0.6),     # classified: counts
        _event(3, "share"),            # counts
    ]
    assert count_valid_interactions(events, model_config, now=NOW) == 2


def test_negative_comment_lands_in_the_dislike_set(settings: dict) -> None:
    profile = build_interaction_profile(
        [_event(1, "comment", -1.0)], settings, now=NOW
    )
    assert 1 in profile.disliked
    assert 1 not in profile.positive_movie_ids()


def test_score_interaction_events_reports_unsupported_types(
    model_config: dict,
) -> None:
    payload = {
        "user_id": 7,
        "as_of": NOW.isoformat(),
        "events": [_event(1, "share"), _event(2, "add_to_watchlist")],
    }
    result = score_interaction_events(payload, model_config)
    assert result["unsupported_event_types"] == ["add_to_watchlist"]
    assert result["events_ignored"] == 1
    assert result["events_counted"] == 1
    assert [item["movie_id"] for item in result["movie_scores"]] == [1]


def test_score_interaction_events_is_reproducible_with_as_of(
    model_config: dict,
) -> None:
    payload = {
        "user_id": 7,
        "as_of": NOW.isoformat(),
        "events": [_event(1, "share", minutes_ago=60 * 24 * 30)],
    }
    first = score_interaction_events(payload, model_config)
    second = score_interaction_events(payload, model_config)
    assert first == second


def test_missing_events_array_is_a_client_error(model_config: dict) -> None:
    with pytest.raises(InteractionPayloadError):
        score_interaction_events({"user_id": 7}, model_config)
    with pytest.raises(InteractionPayloadError):
        score_interaction_events({"events": {"movie_id": 1}}, model_config)


def test_valid_interaction_count_excludes_clicks_and_stale_events(
    model_config: dict,
) -> None:
    window = int(model_config["scenario"]["interaction_recency_days"])
    events = [
        _event(1, "share"),
        _event(2, "comment", 0.5),
        _event(3, "click"),
        _event(4, "watch", 0.1),
        _event(5, "watch", 0.9),
        _event(6, "rating", 5.0, minutes_ago=60 * 24 * (window + 1)),
    ]
    # share, comment, watch(0.9) count; click, short watch and the stale rating
    # do not.
    assert count_valid_interactions(events, model_config, now=NOW) == 3


def test_suggested_scenario_hint_follows_the_backend_decision_table(
    model_config: dict,
) -> None:
    minimum = int(model_config["scenario"]["min_interactions_for_cf"])
    assert suggest_scenario_hint(None, True, 100, model_config) == "guest"
    assert suggest_scenario_hint(7, False, 100, model_config) == "guest"
    assert suggest_scenario_hint(7, True, minimum - 1, model_config) == "onboarding_user"
    assert suggest_scenario_hint(7, True, minimum, model_config) == "returning_user"


def test_build_recommend_request_matches_the_engine_contract(
    model_config: dict,
) -> None:
    payload = {
        "user_id": 7,
        "onboarding_completed": True,
        "as_of": NOW.isoformat(),
        "events": [
            _event(1, "share"),
            _event(2, "comment", -1.0),
            _event(3, "watch", 0.9),
            _event(4, "rating", 5.0),
            _event(5, "like"),
        ],
        "exclude_movie_ids": [99],
    }
    request = build_recommend_request(payload, model_config, limit=10)

    assert request["scenario_hint"] == "returning_user"
    assert request["valid_interaction_count_90d"] == 5
    assert request["limit"] == 10
    # The disliked movie is carried into the exclusion list, not left for the
    # engine to rediscover from a stream that may later be trimmed.
    assert 2 in request["exclude_movie_ids"]
    assert 99 in request["exclude_movie_ids"]
    assert set(request) == {
        "user_id",
        "scenario_hint",
        "onboarding_completed",
        "valid_interaction_count_90d",
        "selected_movie_ids",
        "selected_genres",
        "recent_interactions",
        "exclude_movie_ids",
        "limit",
    }
