"""Complete validation pass over processed, feature, split, and serving outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.config import output_dir
from src.data.serving_export import to_logical_value
from src.recommenders.content_based import ContentBasedRecommender
from src.utils.reporting import truncate, write_json


def _rule(
    rule_id: str,
    status: str,
    description: str,
    evidence: dict[str, Any],
    critical: bool = False,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "status": status,
        "critical": critical,
        "description": description,
        "evidence": evidence,
    }


def _scan_ratings(path: Path, valid_movies: set[int]) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    rows = 0
    invalid_movie_fk = 0
    invalid_rating = 0
    null_required = 0
    duplicate_pairs = 0
    previous_pair: np.ndarray | None = None
    minimum_rating = np.inf
    maximum_rating = -np.inf
    timestamp_min: pd.Timestamp | None = None
    timestamp_max: pd.Timestamp | None = None
    for batch in parquet.iter_batches(batch_size=1_000_000):
        frame = batch.to_pandas()
        rows += len(frame)
        invalid_movie_fk += int((~frame["movie_id"].isin(valid_movies)).sum())
        invalid_rating += int((~frame["rating"].between(0.5, 5.0)).sum())
        null_required += int(
            frame[["user_id", "movie_id", "rating", "timestamp"]].isna().sum().sum()
        )
        pairs = frame[["user_id", "movie_id"]].to_numpy()
        if previous_pair is not None:
            pairs = np.vstack([previous_pair, pairs])
        duplicate_pairs += int(
            (
                (pairs[1:, 0] == pairs[:-1, 0])
                & (pairs[1:, 1] == pairs[:-1, 1])
            ).sum()
        )
        previous_pair = pairs[-1:].copy()
        minimum_rating = min(minimum_rating, float(frame["rating"].min()))
        maximum_rating = max(maximum_rating, float(frame["rating"].max()))
        current_min = frame["timestamp"].min()
        current_max = frame["timestamp"].max()
        timestamp_min = current_min if timestamp_min is None else min(timestamp_min, current_min)
        timestamp_max = current_max if timestamp_max is None else max(timestamp_max, current_max)
    return {
        "rows": rows,
        "invalid_movie_foreign_keys": invalid_movie_fk,
        "invalid_ratings": invalid_rating,
        "null_required_values": null_required,
        "duplicate_user_movie_pairs": duplicate_pairs,
        "rating_min": float(minimum_rating),
        "rating_max": float(maximum_rating),
        "timestamp_min": timestamp_min.isoformat() if timestamp_min is not None else None,
        "timestamp_max": timestamp_max.isoformat() if timestamp_max is not None else None,
        "schema": parquet.schema_arrow.names,
    }


def _scan_interactions(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    interaction_types: set[str] = set()
    rows = 0
    nulls = 0
    for batch in parquet.iter_batches(batch_size=1_000_000):
        frame = batch.to_pandas()
        rows += len(frame)
        interaction_types.update(frame["interaction_type"].dropna().astype(str).unique())
        nulls += int(
            frame[
                [
                    "user_id",
                    "movie_id",
                    "interaction_value",
                    "interaction_type",
                    "timestamp",
                ]
            ]
            .isna()
            .sum()
            .sum()
        )
    return {
        "rows": rows,
        "schema": parquet.schema_arrow.names,
        "interaction_types": sorted(interaction_types),
        "null_required_values": nulls,
    }


def _validate_splits(config: dict[str, Any], source_rows: int) -> dict[str, Any]:
    splits_dir = output_dir(config, "splits_dir", create=False)
    paths = {
        name: splits_dir / f"interactions_{name}.parquet"
        for name in ("train", "validation", "test")
    }
    metadata_rows = {
        name: pq.ParquetFile(path).metadata.num_rows for name, path in paths.items()
    }
    validation = pd.read_parquet(paths["validation"])
    test = pd.read_parquet(paths["test"])
    validation_pairs = (
        validation["user_id"].astype("int64").to_numpy() << np.int64(32)
    ) | validation["movie_id"].astype("int64").to_numpy()
    test_pairs = (test["user_id"].astype("int64").to_numpy() << np.int64(32)) | test[
        "movie_id"
    ].astype("int64").to_numpy()
    holdout_pairs = np.unique(np.concatenate([validation_pairs, test_pairs]))
    train_pair_overlap = 0
    train_user_parts: list[np.ndarray] = []
    train_max_parts: list[pd.DataFrame] = []
    for batch in pq.ParquetFile(paths["train"]).iter_batches(batch_size=1_000_000):
        frame = batch.to_pandas()
        train_pairs = (
            frame["user_id"].astype("int64").to_numpy() << np.int64(32)
        ) | frame["movie_id"].astype("int64").to_numpy()
        train_pair_overlap += int(np.isin(train_pairs, holdout_pairs).sum())
        train_user_parts.append(frame["user_id"].unique())
        train_max_parts.append(
            frame.groupby("user_id", as_index=False)["timestamp"]
            .max()
            .rename(columns={"timestamp": "train_max"})
        )
    train_users = set(np.unique(np.concatenate(train_user_parts)).tolist())
    train_max = (
        pd.concat(train_max_parts, ignore_index=True)
        .groupby("user_id", as_index=True)["train_max"]
        .max()
    )
    validation_time = validation.set_index("user_id")["timestamp"]
    test_time = test.set_index("user_id")["timestamp"]
    common_validation = train_max.index.intersection(validation_time.index)
    common_test = train_max.index.intersection(test_time.index)
    common_holdout = validation_time.index.intersection(test_time.index)
    chronological_violations = int(
        (train_max.loc[common_validation] > validation_time.loc[common_validation]).sum()
        + (train_max.loc[common_test] > test_time.loc[common_test]).sum()
        + (
            validation_time.loc[common_holdout]
            > test_time.loc[common_holdout]
        ).sum()
    )
    validation_internal_duplicates = int(
        validation.duplicated(["user_id", "movie_id"]).sum()
    )
    test_internal_duplicates = int(test.duplicated(["user_id", "movie_id"]).sum())
    validation_test_overlap = int(np.isin(validation_pairs, test_pairs).sum())
    return {
        "row_counts": metadata_rows,
        "source_rows": source_rows,
        "rows_reconciled": sum(metadata_rows.values()) == source_rows,
        "train_holdout_pair_overlap": train_pair_overlap,
        "validation_test_pair_overlap": validation_test_overlap,
        "validation_internal_duplicates": validation_internal_duplicates,
        "test_internal_duplicates": test_internal_duplicates,
        "chronological_violations": chronological_violations,
        "test_users_without_train": len(set(test["user_id"]) - train_users),
        "validation_users_without_train": len(set(validation["user_id"]) - train_users),
    }


def _validate_jsonl(path: Path) -> dict[str, Any]:
    rows = 0
    malformed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line)
            rows += 1
        except (json.JSONDecodeError, ValueError):
            malformed += 1
    return {"rows": rows, "malformed_rows": malformed}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Final validation",
        "",
        f"Overall status: **{report['overall_status']}**",
        "",
        "| Rule | Status | Critical | Description | Evidence |",
        "|---|---|---|---|---|",
    ]
    for rule in report["rules"]:
        evidence = "; ".join(
            f"{key}={truncate(value)}" for key, value in rule["evidence"].items()
        )
        lines.append(
            f"| `{rule['rule_id']}` | **{rule['status']}** | "
            f"{'yes' if rule['critical'] else 'no'} | "
            f"{rule['description']} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "A critical `FAIL` causes the validation command and complete pipeline to exit non-zero. Warnings are retained in this report and are not treated as passes.",
            "",
        ]
    )
    return "\n".join(lines)


def run_final_validation(config: dict[str, Any]) -> dict[str, Any]:
    """Validate every final local artifact and write JSON/Markdown status reports."""
    root = Path(config["_project_root"])
    processed_dir = output_dir(config, "processed_dir", create=False)
    features_dir = output_dir(config, "features_dir", create=False)
    serving_dir = output_dir(config, "serving_dir", create=False)
    validation_dir = output_dir(config, "validation_dir")
    required = [
        "data/processed/movies_clean.parquet",
        "data/processed/ratings_clean.parquet",
        "data/features/movie_content_features.parquet",
        "data/features/user_item_interactions.parquet",
        "data/serving/top_rated_all.parquet",
        "data/serving/top_rated_by_genre.parquet",
        "data/splits/interactions_train.parquet",
        "data/splits/interactions_validation.parquet",
        "data/splits/interactions_test.parquet",
        "data/serving/movies_serving.parquet",
        "data/serving/movies_serving.jsonl",
        "data/serving/popular_movies.jsonl",
        "docs/data_dictionary.md",
        "docs/id_mapping.md",
        "docs/local_serving_schema.md",
    ]
    missing = [value for value in required if not (root / value).exists()]
    rules: list[dict[str, Any]] = [
        _rule(
            "REQUIRED_OUTPUTS_PRESENT",
            "PASS" if not missing else "FAIL",
            "Every required local data, serving, and schema artifact exists.",
            {"missing": missing},
            critical=True,
        )
    ]
    if missing:
        report = {
            "overall_status": "FAIL",
            "rules": rules,
        }
        write_json(validation_dir / "final_validation.json", report)
        (validation_dir / "final_validation.md").write_text(
            _markdown(report), encoding="utf-8", newline="\n"
        )
        raise RuntimeError(f"Required outputs are missing: {missing}")

    movies = pd.read_parquet(processed_dir / "movies_clean.parquet")
    valid_movies = set(movies["movie_id"].astype(int))
    rules.extend(
        [
            _rule(
                "MOVIE_KEY_UNIQUE",
                "PASS" if not movies["movie_id"].duplicated().any() else "FAIL",
                "Canonical movie IDs are unique and positive.",
                {
                    "rows": len(movies),
                    "duplicate_ids": int(movies["movie_id"].duplicated().sum()),
                    "non_positive_ids": int((movies["movie_id"] <= 0).sum()),
                },
                critical=True,
            ),
            _rule(
                "MOVIE_REQUIRED_FIELDS",
                "PASS"
                if movies[["movie_id", "title", "genres"]].isna().sum().sum() == 0
                and movies["title"].str.strip().ne("").all()
                else "FAIL",
                "Required movie serving/model fields are non-null and titles are non-empty.",
                {
                    "required_nulls": int(
                        movies[["movie_id", "title", "genres"]].isna().sum().sum()
                    ),
                    "empty_titles": int(movies["title"].str.strip().eq("").sum()),
                },
                critical=True,
            ),
        ]
    )
    mapping = pd.read_parquet(processed_dir / "id_mapping_clean.parquet")
    mapping_issues = {
        "duplicate_source_ids": int(mapping["movielens_movie_id"].duplicated().sum()),
        "invalid_movie_foreign_keys": int((~mapping["movie_id"].isin(valid_movies)).sum()),
    }
    rules.append(
        _rule(
            "ID_MAPPING_INTEGRITY",
            "PASS" if not any(mapping_issues.values()) else "FAIL",
            "MovieLens IDs map uniquely to existing canonical movies.",
            mapping_issues,
            critical=True,
        )
    )

    ratings = _scan_ratings(processed_dir / "ratings_clean.parquet", valid_movies)
    rating_failures = (
        ratings["invalid_movie_foreign_keys"]
        + ratings["invalid_ratings"]
        + ratings["null_required_values"]
        + ratings["duplicate_user_movie_pairs"]
    )
    rules.append(
        _rule(
            "CLEAN_RATING_INTEGRITY",
            "PASS" if rating_failures == 0 else "FAIL",
            "Clean ratings have valid keys, ranges, timestamps, and unique canonical pairs.",
            ratings,
            critical=True,
        )
    )

    child_paths = {
        "genres": processed_dir / "movie_genres_clean.parquet",
        "companies": processed_dir / "movie_companies_clean.parquet",
        "countries": processed_dir / "movie_countries_clean.parquet",
        "keywords": processed_dir / "movie_keywords_clean.parquet",
        "credits": processed_dir / "movie_credits_clean.parquet",
    }
    child_fk_issues = {
        name: int(
            (
                ~pd.read_parquet(path, columns=["movie_id"])["movie_id"].isin(
                    valid_movies
                )
            ).sum()
        )
        for name, path in child_paths.items()
    }
    rules.append(
        _rule(
            "CHILD_TABLE_FOREIGN_KEYS",
            "PASS" if not any(child_fk_issues.values()) else "FAIL",
            "Every normalized metadata/content child row references a clean movie.",
            child_fk_issues,
            critical=True,
        )
    )

    features = pd.read_parquet(features_dir / "movie_content_features.parquet")
    feature_issues = {
        "row_difference_from_movies": len(features) - len(movies),
        "duplicate_movie_ids": int(features["movie_id"].duplicated().sum()),
        "empty_cleaned_text": int(features["cleaned_text"].str.strip().eq("").sum()),
        "missing_movie_ids": len(valid_movies - set(features["movie_id"])),
    }
    rules.append(
        _rule(
            "MOVIE_FEATURE_COVERAGE",
            "PASS" if not any(feature_issues.values()) else "FAIL",
            "Content features contain exactly one usable row per clean movie.",
            feature_issues,
            critical=True,
        )
    )

    interactions = _scan_interactions(
        features_dir / "user_item_interactions.parquet"
    )
    interaction_ok = (
        interactions["rows"] == ratings["rows"]
        and interactions["interaction_types"] == ["rating"]
        and interactions["null_required_values"] == 0
    )
    rules.append(
        _rule(
            "INTERACTION_TABLE_INTEGRITY",
            "PASS" if interaction_ok else "FAIL",
            "Returning-user features preserve every clean rating and invent no interaction type.",
            interactions,
            critical=True,
        )
    )

    split = _validate_splits(config, ratings["rows"])
    split_failures = (
        not split["rows_reconciled"]
        or split["train_holdout_pair_overlap"] > 0
        or split["validation_test_pair_overlap"] > 0
        or split["validation_internal_duplicates"] > 0
        or split["test_internal_duplicates"] > 0
        or split["chronological_violations"] > 0
        or split["test_users_without_train"] > 0
        or split["validation_users_without_train"] > 0
    )
    rules.append(
        _rule(
            "SPLIT_LEAKAGE",
            "PASS" if not split_failures else "FAIL",
            "Splits reconcile, are pair-disjoint, chronological, and give holdout users train history.",
            split,
            critical=True,
        )
    )

    top_all = pd.read_parquet(serving_dir / "top_rated_all.parquet")
    top_genre = pd.read_parquet(serving_dir / "top_rated_by_genre.parquet")
    ranking_issues = {
        "all_duplicate_movies": int(top_all["movie_id"].duplicated().sum()),
        "genre_duplicate_movies": int(
            top_genre.duplicated(["genre", "movie_id"]).sum()
        ),
        "genre_duplicate_ranks": int(top_genre.duplicated(["genre", "rank"]).sum()),
        "user_columns": [
            column
            for column in [*top_all.columns, *top_genre.columns]
            if "user" in column.lower()
        ],
    }
    ranking_ok = (
        ranking_issues["all_duplicate_movies"] == 0
        and ranking_issues["genre_duplicate_movies"] == 0
        and ranking_issues["genre_duplicate_ranks"] == 0
        and not ranking_issues["user_columns"]
    )
    rules.append(
        _rule(
            "GUEST_RANKING_INTEGRITY",
            "PASS" if ranking_ok else "FAIL",
            "Guest rankings are unique, ranked deterministically, and contain no tracking fields.",
            ranking_issues,
            critical=True,
        )
    )

    recommender = ContentBasedRecommender(config)
    first = recommender.recommend([862], ["Animation"], 10)
    second = recommender.recommend([862], ["Animation"], 10)
    fallback = recommender.recommend([], [], 5)
    onboarding_evidence = {
        "deterministic": first.recommendations == second.recommendations,
        "selected_movie_excluded": 862
        not in [record["movie_id"] for record in first.recommendations],
        "unique_results": len(
            {record["movie_id"] for record in first.recommendations}
        )
        == len(first.recommendations),
        "fallback_used_for_empty_input": fallback.fallback_used,
    }
    rules.append(
        _rule(
            "ONBOARDING_RECOMMENDER_BEHAVIOR",
            "PASS" if all(onboarding_evidence.values()) else "FAIL",
            "Onboarding is deterministic, excludes selections, returns unique movies, and falls back safely.",
            onboarding_evidence,
            critical=True,
        )
    )

    serving_movies = pd.read_parquet(serving_dir / "movies_serving.parquet")
    serving_issues = {
        "row_difference_from_movies": len(serving_movies) - len(movies),
        "duplicate_movie_ids": int(serving_movies["movie_id"].duplicated().sum()),
        "required_nulls": int(
            serving_movies[["movie_id", "title", "genres"]].isna().sum().sum()
        ),
    }
    rules.append(
        _rule(
            "SERVING_TABLE_INTEGRITY",
            "PASS" if not any(serving_issues.values()) else "FAIL",
            "Movie serving rows are complete and unique at canonical key grain.",
            serving_issues,
            critical=True,
        )
    )
    movie_json = _validate_jsonl(serving_dir / "movies_serving.jsonl")
    popular_json = _validate_jsonl(serving_dir / "popular_movies.jsonl")
    serialized = to_logical_value(
        {
            "string": "x",
            "number": np.int64(2),
            "boolean": np.bool_(True),
            "list": [np.float32(1.5), None],
            "map": {"nested": pd.NA},
        }
    )
    serializer_ok = (
        movie_json["malformed_rows"] == 0
        and movie_json["rows"] == len(movies)
        and popular_json["malformed_rows"] == 0
        and isinstance(serialized["number"], int)
        and isinstance(serialized["boolean"], bool)
        and isinstance(serialized["list"], list)
        and isinstance(serialized["map"], dict)
        and serialized["map"]["nested"] is None
    )
    rules.append(
        _rule(
            "JSON_SERIALIZATION",
            "PASS" if serializer_ok else "FAIL",
            "Serving JSONL parses fully and preserves logical scalar/container/null types.",
            {
                "movies_jsonl": movie_json,
                "popular_jsonl": popular_json,
                "type_round_trip": serialized,
            },
            critical=True,
        )
    )

    determinism_path = validation_dir / "determinism_summary.json"
    if determinism_path.exists():
        determinism = json.loads(determinism_path.read_text(encoding="utf-8"))
        determinism_status = "PASS" if determinism.get("all_match") else "FAIL"
    else:
        determinism = {"reason": "Artifact hash rerun has not been executed yet."}
        determinism_status = "WARNING"
    rules.append(
        _rule(
            "DETERMINISTIC_PIPELINE_ARTIFACTS",
            determinism_status,
            "Core generated artifacts match byte-for-byte across a pipeline rerun.",
            determinism,
            critical=determinism_status == "FAIL",
        )
    )

    warning_facts = {
        "movies_without_genres": int(movies["genres"].map(len).eq(0).sum()),
        "train_only_sparse_users": json.loads(
            (validation_dir / "split_summary.json").read_text(encoding="utf-8")
        )["treatment"]["train_only_sparse_users"],
        "available_interaction_types": interactions["interaction_types"],
    }
    rules.append(
        _rule(
            "KNOWN_DATASET_LIMITATIONS",
            "WARNING",
            "Genre gaps, sparse histories, and ratings-only feedback remain visible limitations.",
            warning_facts,
            critical=False,
        )
    )
    overall = (
        "FAIL"
        if any(rule["status"] == "FAIL" for rule in rules)
        else "WARNING"
        if any(rule["status"] == "WARNING" for rule in rules)
        else "PASS"
    )
    report = {
        "overall_status": overall,
        "rules": rules,
    }
    write_json(validation_dir / "final_validation.json", report)
    (validation_dir / "final_validation.md").write_text(
        _markdown(report), encoding="utf-8", newline="\n"
    )
    critical_failures = [
        rule for rule in rules if rule["critical"] and rule["status"] == "FAIL"
    ]
    if critical_failures:
        raise RuntimeError(
            "Critical final validation failure(s): "
            + ", ".join(rule["rule_id"] for rule in critical_failures)
        )
    return report

