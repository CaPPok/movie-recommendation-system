"""CLI for Phase D content, guest, onboarding, and interaction datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from src.data.config import load_config, output_dir
from src.features.content import build_movie_content_features
from src.features.interactions import build_user_item_interactions
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.guest import build_top_rated_rankings
from src.utils.reporting import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    return parser.parse_args()


def _example_markdown(
    recommender: ContentBasedRecommender, movies: pd.DataFrame
) -> str:
    title_by_id = movies.set_index("movie_id")["title"].to_dict()
    genre_by_id = {
        int(row.movie_id): (
            row.genres.tolist() if hasattr(row.genres, "tolist") else list(row.genres)
        )
        for row in movies.itertuples(index=False)
    }
    cases = [
        {
            "name": "Movie-only: Toy Story",
            "movie_ids": [862],
            "genres": [],
        },
        {
            "name": "Genre-only: Drama and Thriller",
            "movie_ids": [],
            "genres": ["Drama", "Thriller"],
        },
        {
            "name": "Movies plus genres",
            "movie_ids": [862, 949],
            "genres": ["Animation", "Crime"],
        },
        {
            "name": "Unusable input fallback",
            "movie_ids": ["not-a-movie", -1],
            "genres": ["Not A Genre"],
        },
    ]
    lines = [
        "# Onboarding recommendation sanity checks",
        "",
        "These examples are local plausibility checks, not offline quality claims. Recommendations use only selected canonical movie IDs, selected genres, and cleaned movie content. Selected movies are excluded.",
    ]
    for case in cases:
        result = recommender.recommend(case["movie_ids"], case["genres"], top_k=5)
        selected = [
            f"{movie_id} ({title_by_id.get(movie_id, 'unknown')})"
            for movie_id in case["movie_ids"]
            if isinstance(movie_id, int) and movie_id in title_by_id
        ]
        selected_genres = {
            genre
            for movie_id in case["movie_ids"]
            if isinstance(movie_id, int)
            for genre in genre_by_id.get(movie_id, [])
        } | set(case["genres"])
        lines.extend(
            [
                "",
                f"## {case['name']}",
                "",
                f"- Selected movies: {selected or 'none'}",
                f"- Selected genres: {case['genres'] or 'none'}",
                f"- Fallback used: {result.fallback_used}",
            ]
        )
        if result.warnings:
            lines.append(f"- Warnings: {'; '.join(result.warnings)}")
        lines.extend(
            [
                "",
                "| Rank | Movie | Genres | Score | Why plausible |",
                "|---:|---|---|---:|---|",
            ]
        )
        for record in result.recommendations:
            overlap = sorted(selected_genres & set(record["genres"]))
            reason = (
                f"shared genres: {', '.join(overlap)}"
                if overlap
                else (
                    "metadata/text similarity to selected movies"
                    if not result.fallback_used
                    else "weighted top-rated fallback"
                )
            )
            lines.append(
                f"| {record['rank']} | {record['movie_id']} — {record['title']} | "
                f"{', '.join(record['genres']) or 'none'} | {record['score']:.6f} | "
                f"{reason} |"
            )
    lines.extend(
        [
            "",
            "Similarity alone does not establish recommendation quality. A future evaluation should use held-out returning-user interactions and suitable ranking metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = load_config(parse_args().config)
    content = build_movie_content_features(config)
    interactions, movie_stats = build_user_item_interactions(config)
    rankings = build_top_rated_rankings(config, movie_stats)
    recommender = ContentBasedRecommender(config)
    movies = pd.read_parquet(
        output_dir(config, "processed_dir") / "movies_clean.parquet",
        columns=["movie_id", "title", "genres"],
    )
    examples_path = (
        output_dir(config, "validation_dir")
        / "onboarding_recommendation_examples.md"
    )
    examples_path.write_text(
        _example_markdown(recommender, movies), encoding="utf-8", newline="\n"
    )
    summary = {
        "content_features": content,
        "interactions": interactions,
        "top_rated": rankings,
    }
    write_json(
        output_dir(config, "validation_dir") / "feature_build_summary.json",
        summary,
    )
    print(
        f"Content matrix: {content['matrix_shape'][0]:,} x "
        f"{content['matrix_shape'][1]:,}"
    )
    print(f"Interactions: {interactions['interactions']:,}")
    print(
        f"Guest rankings: {rankings['all_rows']:,} ALL + "
        f"{rankings['genre_rows']:,} genre rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

