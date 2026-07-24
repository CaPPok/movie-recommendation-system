from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import _names_from_records, normalize_text
from src.data.profiling import _sniff_csv
from src.data.serving_export import to_logical_value
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.guest import build_top_rated_rankings, weighted_rating


def test_schema_detection(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("id,name\n1,Alpha\n", encoding="utf-8")
    detected = _sniff_csv(path)
    assert detected["encoding"] == "utf-8"
    assert detected["delimiter"] == ","


def test_missing_value_and_genre_normalization() -> None:
    assert normalize_text("  hello   world ") == "hello world"
    assert normalize_text(" NULL ") is None
    assert _names_from_records("[{'name': ' Drama '}, {'name': 'Drama'}]") == [
        "Drama"
    ]


def test_duplicate_handling_and_id_mapping(sample_project: tuple[Path, dict]) -> None:
    root, _ = sample_project
    movies = pd.read_parquet(root / "data/processed/movies_clean.parquet")
    mapping = pd.read_parquet(root / "data/processed/id_mapping_clean.parquet")
    rejected = pd.read_parquet(root / "data/interim/rejected_movies.parquet")
    assert len(movies) == 6
    assert movies["movie_id"].is_unique
    assert mapping["movielens_movie_id"].is_unique
    assert set(mapping["movie_id"]) == set(movies["movie_id"])
    assert set(rejected["rejection_reason"]) == {
        "exact_duplicate_row",
        "invalid_movie_id",
        "empty_title",
    }


def test_rating_validation_and_movie_reference(
    sample_project: tuple[Path, dict],
) -> None:
    root, _ = sample_project
    ratings = pd.read_parquet(root / "data/processed/ratings_clean.parquet")
    rejected = pd.read_parquet(root / "data/interim/rejected_ratings.parquet")
    assert ratings["rating"].between(0.5, 5.0).all()
    assert not ratings.duplicated(["user_id", "movie_id"]).any()
    assert "missing_tmdb_mapping" in set(rejected["rejection_reason"])


def test_weighted_rating_calculation() -> None:
    score = weighted_rating(4.5, 100.0, 3.5, 50.0)
    assert score == pytest.approx((100 / 150) * 4.5 + (50 / 150) * 3.5)


def test_top_rated_ranking_is_unique(sample_project: tuple[Path, dict]) -> None:
    root, _ = sample_project
    ranking = pd.read_parquet(root / "data/serving/top_rated_all.parquet")
    assert ranking["movie_id"].is_unique
    assert ranking["rank"].tolist() == list(range(1, len(ranking) + 1))
    assert not any("user" in column.lower() for column in ranking.columns)


def test_content_input_validation(sample_project: tuple[Path, dict]) -> None:
    _, config = sample_project
    recommender = ContentBasedRecommender(config)
    with pytest.raises(TypeError):
        recommender.recommend("10", [], 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        recommender.recommend([], [], 0)


def test_selected_movie_exclusion_and_uniqueness(
    sample_project: tuple[Path, dict],
) -> None:
    _, config = sample_project
    result = ContentBasedRecommender(config).recommend([10], ["Drama"], 4)
    ids = [record["movie_id"] for record in result.recommendations]
    assert 10 not in ids
    assert len(ids) == len(set(ids))


def test_onboarding_fallback(sample_project: tuple[Path, dict]) -> None:
    _, config = sample_project
    result = ContentBasedRecommender(config).recommend(
        ["bad", 9999], ["unknown"], 3
    )
    assert result.fallback_used
    assert 0 < len(result.recommendations) <= 3
    assert result.warnings


def test_interaction_table_construction(sample_project: tuple[Path, dict]) -> None:
    root, _ = sample_project
    interactions = pd.read_parquet(
        root / "data/features/user_item_interactions.parquet"
    )
    ratings = pd.read_parquet(root / "data/processed/ratings_clean.parquet")
    assert len(interactions) == len(ratings)
    assert set(interactions["interaction_type"]) == {"rating"}
    assert interactions["interaction_value"].tolist() == ratings["rating"].tolist()


def test_chronological_splitting(sample_project: tuple[Path, dict]) -> None:
    root, _ = sample_project
    train = pd.read_parquet(root / "data/splits/interactions_train.parquet")
    validation = pd.read_parquet(
        root / "data/splits/interactions_validation.parquet"
    )
    test = pd.read_parquet(root / "data/splits/interactions_test.parquet")
    for user_id in test["user_id"].unique():
        assert train.loc[train.user_id == user_id, "timestamp"].max() <= validation.loc[
            validation.user_id == user_id, "timestamp"
        ].min()
        assert validation.loc[
            validation.user_id == user_id, "timestamp"
        ].max() <= test.loc[test.user_id == user_id, "timestamp"].min()
    assert 3 not in set(test["user_id"])


def test_split_leakage_prevention(sample_project: tuple[Path, dict]) -> None:
    root, _ = sample_project
    splits = [
        pd.read_parquet(root / f"data/splits/interactions_{name}.parquet")
        for name in ("train", "validation", "test")
    ]
    pair_sets = [
        set(zip(frame.user_id, frame.movie_id, strict=True)) for frame in splits
    ]
    assert pair_sets[0].isdisjoint(pair_sets[1])
    assert pair_sets[0].isdisjoint(pair_sets[2])
    assert pair_sets[1].isdisjoint(pair_sets[2])


def test_serving_json_serialization() -> None:
    value = to_logical_value(
        {
            "s": "x",
            "n": np.int64(2),
            "b": np.bool_(True),
            "l": [np.float32(1.5), None],
            "m": {"null": pd.NA},
        }
    )
    assert value == {
        "s": "x",
        "n": 2,
        "b": True,
        "l": [1.5, None],
        "m": {"null": None},
    }


def test_deterministic_ranking_output(sample_project: tuple[Path, dict]) -> None:
    root, config = sample_project
    path = root / "data/serving/top_rated_all.parquet"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    ratings = pd.read_parquet(root / "data/processed/ratings_clean.parquet")
    stats = (
        ratings.groupby("movie_id", as_index=False)["rating"]
        .agg(rating_count="count", rating_sum="sum")
    )
    stats["average_rating"] = stats["rating_sum"] / stats["rating_count"]
    build_top_rated_rankings(config, stats)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after


def test_integration_pipeline_outputs(sample_project: tuple[Path, dict]) -> None:
    root, _ = sample_project
    required = [
        "reports/profiling/raw_profile.json",
        "reports/validation/final_validation.json",
        "data/processed/movies_clean.parquet",
        "data/features/movie_content_features.parquet",
        "data/features/user_item_interactions.parquet",
        "data/splits/interactions_train.parquet",
        "data/serving/movies_serving.jsonl",
    ]
    assert all((root / value).exists() for value in required)
    final = (root / "reports/validation/final_validation.json").read_text(
        encoding="utf-8"
    )
    assert '"overall_status": "FAIL"' not in final
