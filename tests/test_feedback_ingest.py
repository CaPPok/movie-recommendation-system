"""Event-to-training-row conversion and the promotion gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import retrain
from src.data.config import load_config
from src.data.feedback_ingest import (
    EVENT_INTERACTION_TYPE,
    events_to_interactions,
    merge_into_interactions,
    score_to_rating,
)

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_config() -> dict:
    return load_config(REPOSITORY_ROOT / "configs/model_serving.yaml")


@pytest.fixture(scope="module")
def aws_config() -> dict:
    return load_config(REPOSITORY_ROOT / "configs/aws.yaml")


@pytest.fixture(scope="module")
def retraining(aws_config: dict) -> dict:
    return aws_config["retraining"]


def _event(user_id: int, movie_id: int, event_type: str, value=None, days_ago: int = 0):
    event = {
        "user_id": user_id,
        "movie_id": movie_id,
        "event_type": event_type,
        "timestamp": (NOW - timedelta(days=days_ago)).isoformat(),
    }
    if value is not None:
        event["value"] = value
    return event


def test_score_to_rating_anchors(retraining: dict) -> None:
    anchors = retraining["event_rating"]
    assert score_to_rating(0.0, retraining) == pytest.approx(anchors["neutral_rating"])
    assert score_to_rating(15.0, retraining) == pytest.approx(anchors["maximum_rating"])
    assert score_to_rating(-15.0, retraining) == pytest.approx(anchors["minimum_rating"])
    # Beyond the anchors the mapping saturates instead of running off the scale.
    assert score_to_rating(500.0, retraining) == pytest.approx(anchors["maximum_rating"])
    assert score_to_rating(-500.0, retraining) == pytest.approx(anchors["minimum_rating"])


def test_score_to_rating_is_monotonic(retraining: dict) -> None:
    values = [score_to_rating(score, retraining) for score in range(-20, 21)]
    assert values == sorted(values)


def test_share_becomes_a_positive_training_signal(
    model_config: dict, retraining: dict
) -> None:
    frame, summary = events_to_interactions(
        [_event(1, 862, "share")], model_config, retraining
    )
    assert summary["rows"] == 1
    row = frame.iloc[0]
    assert row["interaction_type"] == EVENT_INTERACTION_TYPE
    # Above positive_rating_threshold, so it enters the ALS confidence matrix.
    threshold = float(model_config["collaborative"]["positive_rating_threshold"])
    assert row["interaction_value"] >= threshold


def test_hostile_comment_becomes_a_negative_training_signal(
    model_config: dict, retraining: dict
) -> None:
    frame, _ = events_to_interactions(
        [_event(1, 550, "comment", -1.0)], model_config, retraining
    )
    negative = float(model_config["collaborative"]["negative_rating_threshold"])
    assert frame.iloc[0]["interaction_value"] <= negative


def test_a_bare_click_stays_neutral_and_is_excluded_from_training(
    model_config: dict, retraining: dict
) -> None:
    frame, _ = events_to_interactions(
        [_event(1, 13, "click")], model_config, retraining
    )
    settings = model_config["collaborative"]
    value = frame.iloc[0]["interaction_value"]
    assert float(settings["negative_rating_threshold"]) < value
    assert value < float(settings["positive_rating_threshold"])


def test_ingest_ignores_decay_so_two_runs_agree(
    model_config: dict, retraining: dict
) -> None:
    """Training rows must not depend on when the job happened to run."""
    recent, _ = events_to_interactions(
        [_event(1, 862, "share", days_ago=0)], model_config, retraining
    )
    old, _ = events_to_interactions(
        [_event(1, 862, "share", days_ago=400)], model_config, retraining
    )
    assert recent.iloc[0]["interaction_value"] == old.iloc[0]["interaction_value"]


def test_repeated_shares_are_capped_before_training(
    model_config: dict, retraining: dict
) -> None:
    once, _ = events_to_interactions(
        [_event(1, 862, "share")], model_config, retraining
    )
    many, summary = events_to_interactions(
        [_event(1, 862, "share", days_ago=index) for index in range(12)],
        model_config,
        retraining,
    )
    assert summary["events_capped"] == 12 - int(
        model_config["interactions"]["max_events_per_movie"]["share"]
    )
    # Both saturate at the top of the scale; the point is that the cap ran.
    assert many.iloc[0]["interaction_value"] >= once.iloc[0]["interaction_value"]


def test_events_are_aggregated_per_user_and_movie(
    model_config: dict, retraining: dict
) -> None:
    events = [
        _event(1, 862, "watch", 0.9),
        _event(1, 862, "share"),
        _event(1, 550, "like"),
        _event(2, 862, "click"),
    ]
    frame, summary = events_to_interactions(events, model_config, retraining)
    assert summary["rows"] == 3
    assert summary["users"] == 2
    assert len(frame[(frame["user_id"] == 1) & (frame["movie_id"] == 862)]) == 1


def test_events_without_user_or_timestamp_are_counted_not_crashed(
    model_config: dict, retraining: dict
) -> None:
    events = [
        {"movie_id": 862, "event_type": "share"},  # no user_id
        {"user_id": 3, "movie_id": 550, "event_type": "share"},  # no timestamp
        _event(4, 13, "share"),
    ]
    frame, summary = events_to_interactions(events, model_config, retraining)
    assert summary["events_without_user_id"] == 1
    assert summary["events_without_timestamp"] == 1
    # Only the fully-formed event can be placed chronologically.
    assert summary["rows"] == 1
    assert int(frame.iloc[0]["user_id"]) == 4


def test_unsupported_types_are_reported(model_config: dict, retraining: dict) -> None:
    _, summary = events_to_interactions(
        [_event(1, 862, "add_to_watchlist")], model_config, retraining
    )
    assert summary["unsupported_event_types"] == ["add_to_watchlist"]
    assert summary["rows"] == 0


# ----------------------------------------------------------------------
# Merge into the canonical interaction table
# ----------------------------------------------------------------------


def _interactions_frame(rows: list[tuple[int, int, float, str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "movie_id",
            "interaction_value",
            "interaction_type",
            "timestamp",
        ],
    ).astype({"user_id": "int64", "movie_id": "int64", "interaction_value": "float32"})
    frame["interaction_type"] = frame["interaction_type"].astype("string")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


@pytest.fixture()
def merge_project(tmp_path: Path) -> dict:
    """A minimal project tree holding only what the merge touches."""
    features = tmp_path / "data" / "features"
    features.mkdir(parents=True)
    existing = _interactions_frame(
        [
            (1, 10, 4.0, "rating", "2020-01-01T00:00:00Z"),
            (1, 11, 3.0, "rating", "2020-01-02T00:00:00Z"),
            (2, 10, 5.0, "rating", "2020-01-03T00:00:00Z"),
            (4, 12, 2.0, "rating", "2020-01-04T00:00:00Z"),
        ]
    )
    existing.to_parquet(features / "user_item_interactions.parquet", index=False)
    return {
        "_project_root": tmp_path,
        "outputs": {"features_dir": "data/features"},
    }


def _merged(config: dict) -> pd.DataFrame:
    return pd.read_parquet(
        Path(config["_project_root"]) / "data/features/user_item_interactions.parquet"
    )


def test_merge_keeps_rows_grouped_and_ordered_by_user(merge_project: dict) -> None:
    """The chronological splitter refuses to run on an ungrouped table."""
    new_rows = _interactions_frame(
        [
            (3, 20, 5.0, "event", "2026-07-01T00:00:00Z"),
            (1, 21, 5.0, "event", "2026-07-02T00:00:00Z"),
            (9, 22, 5.0, "event", "2026-07-03T00:00:00Z"),
        ]
    )
    merge_into_interactions(merge_project, new_rows)
    result = _merged(merge_project)

    users = result["user_id"].tolist()
    assert users == sorted(users)
    for user_id in result["user_id"].unique():
        positions = result.index[result["user_id"] == user_id].tolist()
        assert positions == list(range(positions[0], positions[-1] + 1))


def test_merge_adds_new_users_and_new_pairs(merge_project: dict) -> None:
    new_rows = _interactions_frame(
        [
            (1, 21, 5.0, "event", "2026-07-02T00:00:00Z"),
            (9, 22, 5.0, "event", "2026-07-03T00:00:00Z"),
        ]
    )
    summary = merge_into_interactions(merge_project, new_rows)
    result = _merged(merge_project)

    assert summary["merged_rows"] == 2
    assert summary["output_rows"] == summary["source_rows"] + 2
    assert 9 in set(result["user_id"])
    assert (result["interaction_type"] == "event").sum() == 2


def test_explicit_rating_wins_over_a_derived_event(merge_project: dict) -> None:
    """A film the user already rated keeps the rating, not the event score."""
    new_rows = _interactions_frame(
        [(1, 10, 5.0, "event", "2026-07-02T00:00:00Z")]
    )
    summary = merge_into_interactions(
        merge_project, new_rows, conflict_policy="prefer_rating"
    )
    result = _merged(merge_project)

    assert summary["replaced_rows"] == 1
    assert summary["merged_rows"] == 0
    row = result[(result["user_id"] == 1) & (result["movie_id"] == 10)].iloc[0]
    assert row["interaction_type"] == "rating"
    assert row["interaction_value"] == pytest.approx(4.0)


def test_prefer_event_policy_replaces_the_rating(merge_project: dict) -> None:
    new_rows = _interactions_frame(
        [(1, 10, 5.0, "event", "2026-07-02T00:00:00Z")]
    )
    merge_into_interactions(merge_project, new_rows, conflict_policy="prefer_event")
    result = _merged(merge_project)

    row = result[(result["user_id"] == 1) & (result["movie_id"] == 10)].iloc[0]
    assert row["interaction_type"] == "event"
    assert len(result[(result["user_id"] == 1) & (result["movie_id"] == 10)]) == 1


def test_merge_with_no_new_rows_is_a_no_op(merge_project: dict) -> None:
    before = _merged(merge_project)
    summary = merge_into_interactions(merge_project, before.iloc[0:0])
    assert summary["changed"] is False
    pd.testing.assert_frame_equal(before, _merged(merge_project))


def test_unknown_conflict_policy_is_rejected(merge_project: dict) -> None:
    with pytest.raises(ValueError, match="event_conflict_policy"):
        merge_into_interactions(merge_project, pd.DataFrame(), conflict_policy="newest")


# ----------------------------------------------------------------------
# Promotion gate
# ----------------------------------------------------------------------


def test_mismatched_evaluation_sample_is_flagged(promotion: dict) -> None:
    """A metric from 1,500 users cannot be compared with one from 5,000."""
    previous = {"hit_rate_at_10": 0.110, "users_scored": 5000}
    decision = retrain.evaluate_promotion(
        _payload(0.104, 0.05, users=1484), previous, promotion
    )
    assert "protocol_mismatch" in decision
    assert decision["protocol_mismatch"]["previous_users_scored"] == 5000
    assert decision["protocol_mismatch"]["candidate_users_scored"] == 1484


def test_matching_evaluation_sample_is_not_flagged(promotion: dict) -> None:
    previous = {"hit_rate_at_10": 0.110, "users_scored": 5000}
    decision = retrain.evaluate_promotion(
        _payload(0.109, 0.05, users=4950), previous, promotion
    )
    assert "protocol_mismatch" not in decision
    assert decision["promote"] is True


def _payload(candidate: float, popularity: float, users: int = 5000) -> dict:
    return {
        "models": [
            {
                "model": "popularity_train",
                "users_scored": users,
                "metrics": {"hit_rate_at_10": popularity},
            },
            {
                "model": "collaborative_als",
                "users_scored": users,
                "metrics": {"hit_rate_at_10": candidate},
            },
        ]
    }


@pytest.fixture()
def promotion(aws_config: dict) -> dict:
    return aws_config["retraining"]["promotion"]


def test_first_promotion_needs_only_the_baseline(promotion: dict) -> None:
    decision = retrain.evaluate_promotion(_payload(0.11, 0.05), {}, promotion)
    assert decision["promote"] is True


def test_model_that_loses_to_popularity_is_rejected(promotion: dict) -> None:
    decision = retrain.evaluate_promotion(_payload(0.04, 0.05), {}, promotion)
    assert decision["promote"] is False
    assert "beats_popularity" in decision["failed_checks"]


def test_small_regression_is_tolerated(promotion: dict) -> None:
    previous = {"hit_rate_at_10": 0.110}
    decision = retrain.evaluate_promotion(
        _payload(0.107, 0.05), previous, promotion
    )
    assert decision["promote"] is True


def test_large_regression_blocks_promotion(promotion: dict) -> None:
    previous = {"hit_rate_at_10": 0.110}
    decision = retrain.evaluate_promotion(_payload(0.080, 0.05), previous, promotion)
    assert decision["promote"] is False
    assert "no_regression" in decision["failed_checks"]


def test_too_few_scored_users_blocks_promotion(promotion: dict) -> None:
    decision = retrain.evaluate_promotion(_payload(0.5, 0.05, users=10), {}, promotion)
    assert decision["promote"] is False
    assert "enough_users" in decision["failed_checks"]


def test_missing_candidate_blocks_promotion(promotion: dict) -> None:
    payload = {"models": [{"model": "popularity_train", "users_scored": 5000,
                           "metrics": {"hit_rate_at_10": 0.05}}]}
    decision = retrain.evaluate_promotion(payload, {}, promotion)
    assert decision["promote"] is False
