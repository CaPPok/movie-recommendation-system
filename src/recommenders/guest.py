"""Scenario 1: deterministic weighted-rating recommendations without tracking."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.config import load_config, output_dir


def weighted_rating(
    average_rating: pd.Series | float,
    rating_count: pd.Series | float,
    global_average: float,
    minimum_count: float,
) -> pd.Series | float:
    """IMDb-style shrinkage of movie means toward the global mean."""
    return (
        (rating_count / (rating_count + minimum_count)) * average_rating
        + (minimum_count / (rating_count + minimum_count)) * global_average
    )


def _threshold_evaluation(
    counts: pd.Series, candidates: list[float]
) -> list[dict[str, Any]]:
    return [
        {
            "percentile": float(percentile),
            "minimum_rating_count": int(
                counts.quantile(percentile, interpolation="higher")
            ),
            "eligible_movies": int(
                (
                    counts
                    >= counts.quantile(percentile, interpolation="higher")
                ).sum()
            ),
        }
        for percentile in candidates
    ]


def build_top_rated_rankings(
    config: dict[str, Any], movie_stats: pd.DataFrame
) -> dict[str, Any]:
    """Build global and sufficiently supported genre rankings."""
    processed_dir = output_dir(config, "processed_dir")
    serving_dir = output_dir(config, "serving_dir")
    validation_dir = output_dir(config, "validation_dir")
    movies = pd.read_parquet(
        processed_dir / "movies_clean.parquet", columns=["movie_id", "genres"]
    )
    stats = movie_stats.merge(movies, on="movie_id", how="inner")
    global_average = float(
        stats["rating_sum"].sum() / stats["rating_count"].sum()
    )
    ranking_config = config["ranking"]
    candidates = [float(value) for value in ranking_config["percentile_candidates"]]
    selected = float(ranking_config["selected_percentile"])
    evaluations = _threshold_evaluation(stats["rating_count"], candidates)
    minimum_count = int(
        stats["rating_count"].quantile(selected, interpolation="higher")
    )
    eligible = stats.loc[stats["rating_count"] >= minimum_count].copy()
    eligible["score"] = weighted_rating(
        eligible["average_rating"],
        eligible["rating_count"],
        global_average,
        minimum_count,
    )
    eligible = eligible.sort_values(
        ["score", "rating_count", "average_rating", "movie_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).head(int(ranking_config["top_k_all"]))
    top_all = pd.DataFrame(
        {
            "ranking_type": "ALL",
            "genre": "ALL",
            "rank": range(1, len(eligible) + 1),
            "movie_id": eligible["movie_id"].astype("int64").to_numpy(),
            "score": eligible["score"].astype("float64").to_numpy(),
            "average_rating": eligible["average_rating"].astype("float64").to_numpy(),
            "rating_count": eligible["rating_count"].astype("int64").to_numpy(),
        }
    )

    exploded = stats.explode("genres").rename(columns={"genres": "genre"})
    exploded = exploded.loc[exploded["genre"].notna() & exploded["genre"].ne("")]
    genre_frames: list[pd.DataFrame] = []
    genre_summary: list[dict[str, Any]] = []
    minimum_candidates = int(ranking_config["minimum_genre_candidates"])
    for genre, group in exploded.groupby("genre", sort=True):
        genre_minimum = int(
            group["rating_count"].quantile(selected, interpolation="higher")
        )
        genre_eligible = group.loc[group["rating_count"] >= genre_minimum].copy()
        status = "included" if len(genre_eligible) >= minimum_candidates else "skipped"
        genre_summary.append(
            {
                "genre": str(genre),
                "rated_movies": len(group),
                "minimum_rating_count": genre_minimum,
                "eligible_movies": len(genre_eligible),
                "status": status,
            }
        )
        if status == "skipped":
            continue
        genre_eligible["score"] = weighted_rating(
            genre_eligible["average_rating"],
            genre_eligible["rating_count"],
            global_average,
            genre_minimum,
        )
        genre_eligible = genre_eligible.sort_values(
            ["score", "rating_count", "average_rating", "movie_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).head(int(ranking_config["top_k_per_genre"]))
        genre_frames.append(
            pd.DataFrame(
                {
                    "ranking_type": "GENRE",
                    "genre": str(genre),
                    "rank": range(1, len(genre_eligible) + 1),
                    "movie_id": genre_eligible["movie_id"].astype("int64").to_numpy(),
                    "score": genre_eligible["score"].astype("float64").to_numpy(),
                    "average_rating": genre_eligible["average_rating"]
                    .astype("float64")
                    .to_numpy(),
                    "rating_count": genre_eligible["rating_count"]
                    .astype("int64")
                    .to_numpy(),
                }
            )
        )
    top_genre = pd.concat(genre_frames, ignore_index=True)
    top_all.to_parquet(
        serving_dir / "top_rated_all.parquet", index=False, compression="snappy"
    )
    top_genre.to_parquet(
        serving_dir / "top_rated_by_genre.parquet", index=False, compression="snappy"
    )

    lines = [
        "# Guest top-rated ranking summary",
        "",
        "Guest access is not tracked. These are precomputed movie rankings with no user or session identifier.",
        "",
        f"Global mean rating `C`: {global_average:.6f}",
        "",
        "## Threshold candidates",
        "",
        "| Rating-count percentile | Minimum count (m) | Eligible movies |",
        "|---:|---:|---:|",
    ]
    for evaluation in evaluations:
        marker = " (selected)" if evaluation["percentile"] == selected else ""
        lines.append(
            f"| {evaluation['percentile']:.0%}{marker} | "
            f"{evaluation['minimum_rating_count']:,} | "
            f"{evaluation['eligible_movies']:,} |"
        )
    lines.extend(
        [
            "",
            f"The configured {selected:.0%} percentile was selected as a balance between vote reliability and catalog coverage; it is computed from the current clean rating counts, not hard-coded. The resulting global `m` is {minimum_count:,}.",
            "",
            "Score formula: `(v/(v+m))*R + (m/(v+m))*C`. Ties are resolved by rating count descending, raw mean descending, then `movie_id` ascending.",
            "",
            "## Genre coverage",
            "",
            "| Genre | Rated movies | m | Eligible | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in genre_summary:
        lines.append(
            f"| {item['genre']} | {item['rated_movies']:,} | "
            f"{item['minimum_rating_count']:,} | {item['eligible_movies']:,} | "
            f"{item['status']} |"
        )
    (validation_dir / "top_rated_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "global_average": global_average,
        "selected_percentile": selected,
        "minimum_rating_count": minimum_count,
        "threshold_candidates": evaluations,
        "all_rows": len(top_all),
        "genre_rows": len(top_genre),
        "genres_included": int((pd.Series([x["status"] for x in genre_summary]) == "included").sum()),
        "genres_skipped": [
            item["genre"] for item in genre_summary if item["status"] == "skipped"
        ],
    }


class GuestRecommender:
    """Read precomputed guest rankings; never accepts or stores a user ID."""

    def __init__(self, config: dict[str, Any]) -> None:
        serving_dir = output_dir(config, "serving_dir", create=False)
        self.all_ranking = pd.read_parquet(serving_dir / "top_rated_all.parquet")
        self.genre_ranking = pd.read_parquet(
            serving_dir / "top_rated_by_genre.parquet"
        )

    def get_guest_recommendations(
        self, genre: str | None = None, top_k: int = 20
    ) -> pd.DataFrame:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if genre is None or not str(genre).strip() or str(genre).upper() == "ALL":
            return self.all_ranking.head(top_k).copy()
        requested = str(genre).strip().casefold()
        known = {
            str(value).casefold(): str(value)
            for value in self.genre_ranking["genre"].unique()
        }
        if requested not in known:
            warnings.warn(
                f"Unknown or unsupported genre {genre!r}; using ALL ranking.",
                UserWarning,
                stacklevel=2,
            )
            return self.all_ranking.head(top_k).copy()
        return self.genre_ranking.loc[
            self.genre_ranking["genre"] == known[requested]
        ].head(top_k).copy()


def get_guest_recommendations(
    genre: str | None = None,
    top_k: int = 20,
    config_path: str | Path = "configs/data_pipeline.yaml",
) -> pd.DataFrame:
    """Convenience function matching the local guest-serving contract."""
    return GuestRecommender(load_config(config_path)).get_guest_recommendations(
        genre=genre, top_k=top_k
    )

