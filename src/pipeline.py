"""Orchestration for the complete local-only data pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scripts.build_features import _example_markdown
from src.data.cleaning import run_cleaning
from src.data.config import output_dir
from src.data.final_validation import run_final_validation
from src.data.profiling import run_raw_profiling
from src.data.serving_export import build_serving_exports
from src.data.splitting import build_interaction_splits
from src.features.content import build_movie_content_features
from src.features.interactions import build_user_item_interactions
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.guest import build_top_rated_rankings
from src.utils.reporting import write_json


def run_pipeline(
    config: dict[str, Any],
    *,
    include_profiling: bool = True,
    include_validation: bool = True,
) -> dict[str, Any]:
    """Run phases in dependency order and stop on any critical failure."""
    results: dict[str, Any] = {}
    if include_profiling:
        run_raw_profiling(config)
        results["profiling"] = "completed"
        print("Phase B: raw profiling and validation completed")

    results["cleaning"] = run_cleaning(config)
    print("Phase C: cleaning and canonical tables completed")

    content = build_movie_content_features(config)
    interactions, movie_stats = build_user_item_interactions(config)
    rankings = build_top_rated_rankings(config, movie_stats)
    recommender = ContentBasedRecommender(config)
    movies = pd.read_parquet(
        output_dir(config, "processed_dir") / "movies_clean.parquet",
        columns=["movie_id", "title", "genres"],
    )
    (
        output_dir(config, "validation_dir")
        / "onboarding_recommendation_examples.md"
    ).write_text(
        _example_markdown(recommender, movies), encoding="utf-8", newline="\n"
    )
    results["features"] = {
        "content_features": content,
        "interactions": interactions,
        "top_rated": rankings,
    }
    write_json(
        output_dir(config, "validation_dir") / "feature_build_summary.json",
        results["features"],
    )
    print("Phase D: features and three recommendation scenarios completed")

    splits = build_interaction_splits(config)
    serving = build_serving_exports(config)
    write_json(
        output_dir(config, "validation_dir") / "serving_export_summary.json",
        serving,
    )
    results["splits"] = splits
    results["serving"] = serving
    print("Phase E: splits and local serving exports completed")

    if include_validation:
        results["validation"] = run_final_validation(config)
        print("Phase F: final validation completed")
    return results

