"""Local JSON/JSONL serialization for future backend ingestion.

This module intentionally has no cloud SDK import and performs no network calls.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def to_logical_value(value: Any) -> Any:
    """Convert cleaned Python/pandas values to JSON-safe logical values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): to_logical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [to_logical_value(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, (str, int)):
        return value
    return str(value)


def dataframe_records(frame: pd.DataFrame) -> Iterable[dict[str, Any]]:
    """Yield records without leaking pandas/numpy scalar types."""
    for record in frame.to_dict(orient="records"):
        yield {key: to_logical_value(value) for key, value in record.items()}


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> int:
    """Atomically write deterministic UTF-8 JSON Lines and return record count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = to_logical_value(record)
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            handle.write("\n")
            count += 1
    os.replace(temporary, path)
    return count


def build_serving_exports(config: dict[str, Any]) -> dict[str, Any]:
    """Build backend-focused local Parquet/JSONL and small schema examples."""
    from src.data.config import output_dir

    processed_dir = output_dir(config, "processed_dir")
    serving_dir = output_dir(config, "serving_dir")
    samples_dir = output_dir(config, "samples_dir")
    validation_dir = output_dir(config, "validation_dir")
    movies = pd.read_parquet(processed_dir / "movies_clean.parquet")
    serving = movies[
        [
            "movie_id",
            "title",
            "release_year",
            "genres",
            "overview",
            "poster_path",
            "vote_average",
            "vote_count",
            "popularity",
            "runtime",
            "original_language",
            "production_companies",
            "production_countries",
        ]
    ].rename(
        columns={
            "production_companies": "companies",
            "production_countries": "countries",
        }
    )
    serving = serving.sort_values("movie_id", kind="mergesort").reset_index(drop=True)
    parquet_path = serving_dir / "movies_serving.parquet"
    temporary_parquet = parquet_path.with_name(f"{parquet_path.name}.tmp")
    serving.to_parquet(temporary_parquet, index=False, compression="snappy")
    os.replace(temporary_parquet, parquet_path)
    movie_json_rows = write_jsonl(
        dataframe_records(serving), serving_dir / "movies_serving.jsonl"
    )

    all_ranking = pd.read_parquet(serving_dir / "top_rated_all.parquet")
    genre_ranking = pd.read_parquet(serving_dir / "top_rated_by_genre.parquet")
    interactions_summary = json.loads(
        (validation_dir / "interactions_summary.json").read_text(encoding="utf-8")
    )
    generated_at = interactions_summary["timestamp_max"]
    popular_records: list[dict[str, Any]] = []
    for (ranking_type, genre), group in pd.concat(
        [all_ranking, genre_ranking], ignore_index=True
    ).groupby(["ranking_type", "genre"], sort=True):
        ordered = group.sort_values("rank", kind="mergesort")
        popular_records.append(
            {
                "ranking_type": str(ranking_type),
                "genre": str(genre),
                "movie_ids": ordered["movie_id"].astype(int).tolist(),
                "scores": ordered["score"].astype(float).tolist(),
                "generated_at": generated_at,
            }
        )
    popular_rows = write_jsonl(
        popular_records, serving_dir / "popular_movies.jsonl"
    )

    first_batch = next(
        pq.ParquetFile(processed_dir / "ratings_clean.parquet").iter_batches(
            batch_size=3
        )
    )
    interactions = first_batch.to_pandas()
    examples = [
        {
            "user_id": int(row.user_id),
            "movie_id": int(row.movie_id),
            "interaction_type": "rating",
            "interaction_value": float(row.rating),
            "timestamp": row.timestamp.isoformat(),
            "session_id": None,
            "_note": "Schema example derived from an observed clean rating; session_id is unavailable.",
        }
        for row in interactions.itertuples(index=False)
    ]
    examples_path = samples_dir / "interaction_event_examples.json"
    temporary_examples = examples_path.with_name(f"{examples_path.name}.tmp")
    temporary_examples.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary_examples, examples_path)
    return {
        "movies_serving_rows": len(serving),
        "movies_jsonl_rows": movie_json_rows,
        "popular_ranking_records": popular_rows,
        "interaction_schema_examples": len(examples),
        "generated_at_policy": (
            "Uses the maximum clean interaction timestamp as a deterministic "
            "data-as-of marker."
        ),
        "excluded_fields": [
            "imdb_id",
            "original_title",
            "release_date",
            "cleaned_text",
            "keywords",
            "cast_names",
            "director_names",
            "embeddings",
            "sparse_vectors",
        ],
    }


# Imported late above to keep the public serializer independently reusable.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
