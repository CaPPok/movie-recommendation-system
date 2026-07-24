"""Ratings-only returning-user interaction feature table and report."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.config import output_dir
from src.utils.reporting import write_json


def _interaction_markdown(summary: dict[str, Any]) -> str:
    quantiles = summary["interactions_per_user"]
    movie_quantiles = summary["interactions_per_movie"]
    lines = [
        "# Returning-user interaction summary",
        "",
        "Only explicit ratings exist in the source dataset. Every row therefore has `interaction_type = rating`; no clicks, watches, likes, or completion events were invented.",
        "",
        f"- Users: {summary['users']:,}",
        f"- Movies with interactions: {summary['movies_with_interactions']:,}",
        f"- Clean movies without interactions: {summary['cold_start_movies']:,}",
        f"- Interactions: {summary['interactions']:,}",
        f"- Matrix sparsity: {summary['sparsity']:.8%}",
        f"- Timestamp coverage: {summary['timestamp_min']} to {summary['timestamp_max']}",
        f"- Users below configured minimum: {summary['users_below_minimum']:,}",
        "- Sparse users removed: 0",
        "",
        "## Interactions per user",
        "",
        "| min | p25 | median | p75 | p95 | max |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {quantiles['min']:.0f} | {quantiles['p25']:.0f} | {quantiles['median']:.0f} | {quantiles['p75']:.0f} | {quantiles['p95']:.0f} | {quantiles['max']:.0f} |",
        "",
        "## Interactions per movie",
        "",
        "| min | p25 | median | p75 | p95 | max |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {movie_quantiles['min']:.0f} | {movie_quantiles['p25']:.0f} | {movie_quantiles['median']:.0f} | {movie_quantiles['p75']:.0f} | {movie_quantiles['p95']:.0f} | {movie_quantiles['max']:.0f} |",
        "",
        "Minimum-interaction thresholds are reported but not applied during cleaning. Users with insufficient history remain in training-only during chronological splitting.",
        "",
    ]
    return "\n".join(lines)


def _distribution(values: pd.Series) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def build_user_item_interactions(
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Stream ratings into the canonical returning-user interaction schema."""
    processed_dir = output_dir(config, "processed_dir")
    features_dir = output_dir(config, "features_dir")
    validation_dir = output_dir(config, "validation_dir")
    source_path = processed_dir / "ratings_clean.parquet"
    target_path = features_dir / "user_item_interactions.parquet"
    temporary_path = target_path.with_name(f"{target_path.name}.tmp")
    source = pq.ParquetFile(source_path)
    writer: pq.ParquetWriter | None = None
    row_count = 0
    rating_counts: Counter[float] = Counter()
    movie_partials: list[pd.DataFrame] = []
    timestamp_min: pd.Timestamp | None = None
    timestamp_max: pd.Timestamp | None = None

    for batch in source.iter_batches(batch_size=1_000_000):
        frame = batch.to_pandas()
        output = pd.DataFrame(
            {
                "user_id": frame["user_id"].astype("int64"),
                "movie_id": frame["movie_id"].astype("int64"),
                "interaction_value": frame["rating"].astype("float32"),
                "interaction_type": pd.Series(
                    "rating", index=frame.index, dtype="string"
                ),
                "timestamp": frame["timestamp"],
            }
        )
        table = pa.Table.from_pandas(output, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                temporary_path, table.schema, compression="snappy", use_dictionary=True
            )
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        row_count += len(frame)
        rating_counts.update(frame["rating"].value_counts().to_dict())
        partial = frame.groupby("movie_id", sort=False)["rating"].agg(["count", "sum"])
        movie_partials.append(partial.reset_index())
        current_min = frame["timestamp"].min()
        current_max = frame["timestamp"].max()
        timestamp_min = current_min if timestamp_min is None else min(timestamp_min, current_min)
        timestamp_max = current_max if timestamp_max is None else max(timestamp_max, current_max)
    if writer is None:
        raise RuntimeError("No clean ratings were available for interaction features")
    writer.close()
    os.replace(temporary_path, target_path)

    movie_stats = (
        pd.concat(movie_partials, ignore_index=True)
        .groupby("movie_id", as_index=False, sort=True)
        .agg(rating_count=("count", "sum"), rating_sum=("sum", "sum"))
    )
    movie_stats["average_rating"] = (
        movie_stats["rating_sum"] / movie_stats["rating_count"]
    )
    users = pd.read_parquet(processed_dir / "users_clean.parquet")
    movies = pd.read_parquet(processed_dir / "movies_clean.parquet", columns=["movie_id"])
    minimum = int(config["interactions"]["minimum_user_interactions"])
    possible = len(users) * len(movie_stats)
    summary = {
        "users": len(users),
        "movies_with_interactions": len(movie_stats),
        "interactions": row_count,
        "sparsity": float(1 - row_count / possible),
        "interactions_per_user": _distribution(users["interaction_count"]),
        "interactions_per_movie": _distribution(movie_stats["rating_count"]),
        "cold_start_users": 0,
        "cold_start_movies": int(
            (~movies["movie_id"].isin(movie_stats["movie_id"])).sum()
        ),
        "users_below_minimum": int((users["interaction_count"] < minimum).sum()),
        "minimum_user_interactions_configured": minimum,
        "sparse_users_removed": 0,
        "rating_distribution": {
            str(key): int(value) for key, value in sorted(rating_counts.items())
        },
        "timestamp_min": timestamp_min.isoformat() if timestamp_min is not None else None,
        "timestamp_max": timestamp_max.isoformat() if timestamp_max is not None else None,
        "interaction_types": ["rating"],
    }
    write_json(validation_dir / "interactions_summary.json", summary)
    (validation_dir / "interactions_summary.md").write_text(
        _interaction_markdown(summary), encoding="utf-8", newline="\n"
    )
    return summary, movie_stats

