"""Modular cleaning for movie metadata, mappings, and full ratings."""

from __future__ import annotations

import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.config import input_path, output_dir
from src.utils.reporting import write_json


MISSING_TEXT = {"", "nan", "none", "null", "n/a", "na"}


def normalize_text(value: Any, *, lowercase: bool = False) -> str | None:
    """Normalize whitespace and configured missing-value representations."""
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text.lower() in MISSING_TEXT:
        return None
    return text.lower() if lowercase else text


def normalize_label(value: Any) -> str | None:
    """Normalize a human-readable categorical label without losing accents."""
    text = normalize_text(value)
    if text is None:
        return None
    return text


def parse_literal(value: Any, expected_type: type) -> Any:
    """Safely parse a Python-literal field and return an empty expected value on failure."""
    empty = [] if expected_type is list else {}
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return empty
    if isinstance(value, expected_type):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError, TypeError):
        return empty
    return parsed if isinstance(parsed, expected_type) else empty


def _deduplicate_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _names_from_records(value: Any) -> list[str]:
    names = []
    for record in parse_literal(value, list):
        if isinstance(record, dict):
            name = normalize_label(record.get("name"))
            if name:
                names.append(name)
    return _deduplicate_preserve_order(names)


def _company_records(value: Any) -> list[dict[str, Any]]:
    result = []
    seen: set[tuple[int | None, str]] = set()
    for record in parse_literal(value, list):
        if not isinstance(record, dict):
            continue
        name = normalize_label(record.get("name"))
        if not name:
            continue
        raw_id = pd.to_numeric(record.get("id"), errors="coerce")
        company_id = None if pd.isna(raw_id) else int(raw_id)
        key = (company_id, name.casefold())
        if key not in seen:
            seen.add(key)
            result.append({"company_id": company_id, "company": name})
    return result


def _country_records(value: Any) -> list[dict[str, str | None]]:
    result = []
    seen: set[tuple[str | None, str]] = set()
    for record in parse_literal(value, list):
        if not isinstance(record, dict):
            continue
        name = normalize_label(record.get("name"))
        code = normalize_text(record.get("iso_3166_1"))
        code = code.upper() if code else None
        if not name and not code:
            continue
        display = name or code or ""
        key = (code, display.casefold())
        if key not in seen:
            seen.add(key)
            result.append({"country_code": code, "country": display})
    return result


def _null_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {name: int(value) for name, value in frame.isna().sum().items()}


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="snappy")
    os.replace(temporary, path)


class _ParquetStream:
    """Write homogeneous DataFrame batches through an atomic temporary file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_name(f"{path.name}.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.temporary, table.schema, compression="snappy", use_dictionary=True
            )
        else:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.rows += len(frame)

    def close(self) -> None:
        if self.writer is None:
            raise RuntimeError(f"No rows were written to required output {self.path}")
        self.writer.close()
        os.replace(self.temporary, self.path)


def clean_movies(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean metadata and write canonical movie/child/rejection tables."""
    raw = pd.read_csv(input_path(config, "movies"), low_memory=False)
    raw["_source_row"] = np.arange(len(raw), dtype="int64")
    nulls_before = _null_counts(raw.drop(columns="_source_row"))
    rejected_parts: list[pd.DataFrame] = []

    def rejection_rows(
        frame: pd.DataFrame, reason: str, duplicate_of: pd.Series | None = None
    ) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                "source_row": frame["_source_row"].astype("int64"),
                "raw_id": frame["id"].astype("string"),
                "raw_title": frame["title"].astype("string"),
                "rejection_reason": reason,
                "duplicate_of_movie_id": pd.Series(
                    pd.NA, index=frame.index, dtype="Int64"
                ),
            }
        )
        if duplicate_of is not None:
            result["duplicate_of_movie_id"] = duplicate_of.astype("Int64")
        return result.reset_index(drop=True)

    exact_mask = raw.drop(columns="_source_row").duplicated(keep="first")
    if exact_mask.any():
        rejected_parts.append(rejection_rows(raw.loc[exact_mask], "exact_duplicate_row"))
    work = raw.loc[~exact_mask].copy()

    work["_movie_id"] = pd.to_numeric(work["id"], errors="coerce")
    invalid_id = work["_movie_id"].isna() | (work["_movie_id"] <= 0) | (
        work["_movie_id"] % 1 != 0
    )
    if invalid_id.any():
        rejected_parts.append(rejection_rows(work.loc[invalid_id], "invalid_movie_id"))
    work = work.loc[~invalid_id].copy()
    work["_movie_id"] = work["_movie_id"].astype("int64")

    normalized_titles = work["title"].map(normalize_text)
    empty_title = normalized_titles.isna()
    if empty_title.any():
        rejected_parts.append(rejection_rows(work.loc[empty_title], "empty_title"))
    work = work.loc[~empty_title].copy()
    work["_normalized_title"] = normalized_titles.loc[work.index]

    completeness_columns = [
        "title",
        "overview",
        "release_date",
        "genres",
        "poster_path",
        "original_language",
        "runtime",
        "vote_average",
        "vote_count",
    ]
    work["_completeness"] = work[completeness_columns].notna().sum(axis=1)
    work["_vote_count_numeric"] = pd.to_numeric(work["vote_count"], errors="coerce").fillna(-1)
    work = work.sort_values(
        ["_movie_id", "_completeness", "_vote_count_numeric", "_source_row"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    duplicate_id = work.duplicated("_movie_id", keep="first")
    if duplicate_id.any():
        rejected_parts.append(
            rejection_rows(
                work.loc[duplicate_id],
                "duplicate_movie_id",
                work.loc[duplicate_id, "_movie_id"],
            )
        )
    work = work.loc[~duplicate_id].sort_values("_movie_id", kind="mergesort").copy()

    release_date = pd.to_datetime(work["release_date"], errors="coerce")
    minimum_year = int(config["processing"]["minimum_release_year"])
    maximum_year = int(config["processing"]["maximum_release_year"])
    invalid_year = release_date.notna() & ~release_date.dt.year.between(
        minimum_year, maximum_year
    )
    release_date.loc[invalid_year] = pd.NaT

    runtime = pd.to_numeric(work["runtime"], errors="coerce")
    runtime.loc[runtime <= 0] = np.nan
    vote_average = pd.to_numeric(work["vote_average"], errors="coerce")
    vote_average.loc[~vote_average.between(0, 10)] = np.nan
    vote_count = pd.to_numeric(work["vote_count"], errors="coerce")
    vote_count.loc[vote_count < 0] = np.nan
    popularity = pd.to_numeric(work["popularity"], errors="coerce")
    popularity.loc[popularity < 0] = np.nan

    genre_lists = work["genres"].map(_names_from_records)
    company_records = work["production_companies"].map(_company_records)
    country_records = work["production_countries"].map(_country_records)

    movies = pd.DataFrame(
        {
            "movie_id": work["_movie_id"].astype("int64"),
            "imdb_id": work["imdb_id"].map(normalize_text).astype("string"),
            "title": work["_normalized_title"].astype("string"),
            "original_title": work["original_title"].map(normalize_text).astype("string"),
            "overview": work["overview"].map(normalize_text).astype("string"),
            "genres": genre_lists,
            "original_language": work["original_language"]
            .map(lambda value: normalize_text(value, lowercase=True))
            .astype("string"),
            "release_date": release_date,
            "release_year": release_date.dt.year.astype("Int16"),
            "runtime": runtime.astype("Float32"),
            "production_companies": company_records.map(
                lambda records: [record["company"] for record in records]
            ),
            "production_countries": country_records.map(
                lambda records: [record["country"] for record in records]
            ),
            "poster_path": work["poster_path"].map(normalize_text).astype("string"),
            "vote_average": vote_average.astype("Float32"),
            "vote_count": vote_count.round().astype("Int64"),
            "popularity": popularity.astype("Float32"),
        }
    ).reset_index(drop=True)

    genre_rows = [
        {"movie_id": int(movie_id), "genre": genre}
        for movie_id, genres in zip(movies["movie_id"], movies["genres"], strict=True)
        for genre in genres
    ]
    genres = pd.DataFrame(genre_rows, columns=["movie_id", "genre"]).drop_duplicates()
    genres = genres.sort_values(["movie_id", "genre"], kind="mergesort").reset_index(drop=True)

    company_rows = [
        {
            "movie_id": int(movie_id),
            "company_id": record["company_id"],
            "company": record["company"],
        }
        for movie_id, records in zip(
            work["_movie_id"], company_records, strict=True
        )
        for record in records
    ]
    companies = pd.DataFrame(
        company_rows, columns=["movie_id", "company_id", "company"]
    ).drop_duplicates()
    if not companies.empty:
        companies["company_id"] = companies["company_id"].astype("Int64")
        companies = companies.sort_values(
            ["movie_id", "company"], kind="mergesort"
        ).reset_index(drop=True)

    country_rows = [
        {
            "movie_id": int(movie_id),
            "country_code": record["country_code"],
            "country": record["country"],
        }
        for movie_id, records in zip(
            work["_movie_id"], country_records, strict=True
        )
        for record in records
    ]
    countries = pd.DataFrame(
        country_rows, columns=["movie_id", "country_code", "country"]
    ).drop_duplicates()
    if not countries.empty:
        countries = countries.sort_values(
            ["movie_id", "country"], kind="mergesort"
        ).reset_index(drop=True)

    rejected = pd.concat(rejected_parts, ignore_index=True)
    rejected = rejected.sort_values("source_row", kind="mergesort").reset_index(drop=True)
    processed_dir = output_dir(config, "processed_dir")
    interim_dir = output_dir(config, "interim_dir")
    _atomic_parquet(movies, processed_dir / "movies_clean.parquet")
    _atomic_parquet(genres, processed_dir / "movie_genres_clean.parquet")
    _atomic_parquet(companies, processed_dir / "movie_companies_clean.parquet")
    _atomic_parquet(countries, processed_dir / "movie_countries_clean.parquet")
    _atomic_parquet(rejected, interim_dir / "rejected_movies.parquet")

    summary = {
        "rows_before": len(raw),
        "rows_after": len(movies),
        "rows_removed": len(rejected),
        "rows_modified": len(movies),
        "duplicates_removed": int(exact_mask.sum() + duplicate_id.sum()),
        "null_values_before": nulls_before,
        "null_values_after": _null_counts(movies),
        "removal_reasons": {
            str(key): int(value)
            for key, value in rejected["rejection_reason"].value_counts().items()
        },
        "normalized_child_rows": {
            "movie_genres_clean": len(genres),
            "movie_companies_clean": len(companies),
            "movie_countries_clean": len(countries),
        },
        "warnings": {
            "runtime_over_configured_warning_max": int(
                (
                    movies["runtime"]
                    > float(config["processing"]["runtime_warning_max_minutes"])
                ).sum()
            ),
            "movies_without_genres": int(movies["genres"].map(len).eq(0).sum()),
        },
    }
    return movies, summary


def clean_content_tables(
    config: dict[str, Any], valid_movie_ids: set[int]
) -> dict[str, Any]:
    """Clean and merge duplicated keyword/credit records by canonical movie ID."""
    processed_dir = output_dir(config, "processed_dir")

    raw_keywords = pd.read_csv(input_path(config, "keywords"))
    keyword_map: dict[int, list[str]] = defaultdict(list)
    malformed_keywords = 0
    for row in raw_keywords.itertuples(index=False):
        movie_id = pd.to_numeric(row.id, errors="coerce")
        if pd.isna(movie_id) or int(movie_id) not in valid_movie_ids:
            continue
        parsed = parse_literal(row.keywords, list)
        if not isinstance(parsed, list):
            malformed_keywords += 1
            continue
        for record in parsed:
            if isinstance(record, dict):
                name = normalize_label(record.get("name"))
                if name:
                    keyword_map[int(movie_id)].append(name)
    keyword_rows = [
        {
            "movie_id": movie_id,
            "keywords": _deduplicate_preserve_order(values),
        }
        for movie_id, values in sorted(keyword_map.items())
    ]
    keywords = pd.DataFrame(keyword_rows, columns=["movie_id", "keywords"])
    _atomic_parquet(keywords, processed_dir / "movie_keywords_clean.parquet")

    raw_credits = pd.read_csv(input_path(config, "credits"), low_memory=False)
    cast_map: dict[int, list[str]] = defaultdict(list)
    director_map: dict[int, list[str]] = defaultdict(list)
    malformed_credits = 0
    for row in raw_credits.itertuples(index=False):
        movie_id = pd.to_numeric(row.id, errors="coerce")
        if pd.isna(movie_id) or int(movie_id) not in valid_movie_ids:
            continue
        cast_records = parse_literal(row.cast, list)
        crew_records = parse_literal(row.crew, list)
        if not isinstance(cast_records, list) or not isinstance(crew_records, list):
            malformed_credits += 1
            continue
        for record in cast_records:
            if isinstance(record, dict):
                name = normalize_label(record.get("name"))
                if name:
                    cast_map[int(movie_id)].append(name)
        for record in crew_records:
            if isinstance(record, dict) and normalize_text(
                record.get("job"), lowercase=True
            ) == "director":
                name = normalize_label(record.get("name"))
                if name:
                    director_map[int(movie_id)].append(name)
    credit_ids = sorted(set(cast_map) | set(director_map))
    credits = pd.DataFrame(
        [
            {
                "movie_id": movie_id,
                "cast_names": _deduplicate_preserve_order(cast_map[movie_id]),
                "director_names": _deduplicate_preserve_order(director_map[movie_id]),
            }
            for movie_id in credit_ids
        ],
        columns=["movie_id", "cast_names", "director_names"],
    )
    _atomic_parquet(credits, processed_dir / "movie_credits_clean.parquet")
    return {
        "keywords": {
            "rows_before": len(raw_keywords),
            "rows_after": len(keywords),
            "duplicate_source_rows_merged": int(
                len(raw_keywords) - raw_keywords["id"].nunique()
            ),
            "malformed_records": malformed_keywords,
        },
        "credits": {
            "rows_before": len(raw_credits),
            "rows_after": len(credits),
            "duplicate_source_rows_merged": int(
                len(raw_credits) - raw_credits["id"].nunique()
            ),
            "malformed_records": malformed_credits,
        },
    }


def clean_id_mapping(
    config: dict[str, Any], valid_movie_ids: set[int]
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Clean MovieLens-to-canonical mapping while retaining legitimate aliases."""
    raw = pd.read_csv(input_path(config, "links"))
    source_id = pd.to_numeric(raw["movieId"], errors="coerce")
    tmdb_id = pd.to_numeric(raw["tmdbId"], errors="coerce")
    valid_source = source_id.notna() & (source_id > 0) & (source_id % 1 == 0)
    mapped = tmdb_id.notna() & tmdb_id.astype("Int64").isin(valid_movie_ids)
    valid = valid_source & mapped
    imdb_numeric = pd.to_numeric(raw.loc[valid, "imdbId"], errors="coerce")
    imdb = imdb_numeric.map(
        lambda value: None if pd.isna(value) else f"tt{int(value):07d}"
    )
    mapping = pd.DataFrame(
        {
            "movielens_movie_id": source_id.loc[valid].astype("int64"),
            "movie_id": tmdb_id.loc[valid].astype("int64"),
            "imdb_id": imdb.astype("string"),
        }
    ).sort_values("movielens_movie_id", kind="mergesort")
    mapping = mapping.drop_duplicates("movielens_movie_id", keep="first").reset_index(
        drop=True
    )
    _atomic_parquet(
        mapping, output_dir(config, "processed_dir") / "id_mapping_clean.parquet"
    )

    statuses = pd.Series("mapped", index=raw.index, dtype="string")
    statuses.loc[~valid_source] = "invalid_movielens_movie_id"
    statuses.loc[valid_source & tmdb_id.isna()] = "missing_tmdb_mapping"
    statuses.loc[
        valid_source & tmdb_id.notna() & ~tmdb_id.astype("Int64").isin(valid_movie_ids)
    ] = "movie_metadata_missing"
    status_by_source = pd.Series(
        statuses.to_numpy(), index=source_id.astype("Int64"), dtype="string"
    )
    summary = {
        "rows_before": len(raw),
        "rows_after": len(mapping),
        "rows_removed": len(raw) - len(mapping),
        "rows_modified": len(mapping),
        "duplicates_removed": int(raw.duplicated("movieId").sum()),
        "removal_reasons": {
            str(key): int(value)
            for key, value in statuses.loc[statuses != "mapped"].value_counts().items()
        },
        "canonical_movie_ids_with_multiple_movielens_ids": int(
            (mapping.groupby("movie_id").size() > 1).sum()
        ),
        "duplicate_alias_mapping_rows": int(
            mapping.duplicated("movie_id", keep=False).sum()
        ),
    }
    return mapping, status_by_source, summary


def _prepare_rejected_ratings(
    source_rows: pd.Series,
    users: pd.Series,
    movies: pd.Series,
    ratings: pd.Series,
    timestamps: pd.Series,
    reasons: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row": source_rows.astype("int64"),
            "user_id_raw": users.astype("Int64"),
            "movielens_movie_id_raw": movies.astype("Int64"),
            "rating_raw": ratings.astype("Float64"),
            "timestamp_raw": timestamps.astype("Int64"),
            "rejection_reason": reasons.astype("string"),
        }
    ).reset_index(drop=True)


def clean_ratings(
    config: dict[str, Any],
    mapping: pd.DataFrame,
    status_by_source: pd.Series,
) -> dict[str, Any]:
    """Map and stream-clean full ratings without loading 26 million rows at once."""
    chunk_size = int(config["processing"]["chunk_size"])
    minimum_rating = float(config["processing"]["minimum_rating"])
    maximum_rating = float(config["processing"]["maximum_rating"])
    mapping_series = mapping.set_index("movielens_movie_id")["movie_id"]
    ratings_path = input_path(config, "ratings")
    processed_dir = output_dir(config, "processed_dir")
    interim_dir = output_dir(config, "interim_dir")
    ratings_writer = _ParquetStream(processed_dir / "ratings_clean.parquet")
    rejected_parts: list[pd.DataFrame] = []
    carry = pd.DataFrame()
    source_offset = 0
    rows_before = 0
    alias_duplicates = 0
    users_rows: list[dict[str, Any]] = []
    previous_raw_user: int | None = None
    nulls_before: dict[str, int] = defaultdict(int)

    def resolve_and_write(frame: pd.DataFrame) -> None:
        nonlocal alias_duplicates
        if frame.empty:
            return
        ordered = frame.sort_values(
            [
                "user_id",
                "movie_id",
                "timestamp_seconds",
                "movielens_movie_id",
                "source_row",
            ],
            kind="mergesort",
        )
        duplicate = ordered.duplicated(["user_id", "movie_id"], keep="last")
        if duplicate.any():
            removed = ordered.loc[duplicate]
            same_source = removed.duplicated(
                ["user_id", "movielens_movie_id"], keep=False
            )
            reasons = pd.Series(
                np.where(
                    same_source,
                    "duplicate_user_movie",
                    "duplicate_canonical_user_movie_alias",
                ),
                index=removed.index,
                dtype="string",
            )
            rejected_parts.append(
                _prepare_rejected_ratings(
                    removed["source_row"],
                    removed["user_id"],
                    removed["movielens_movie_id"],
                    removed["rating"],
                    removed["timestamp_seconds"],
                    reasons,
                )
            )
            alias_duplicates += int(duplicate.sum())
        kept = ordered.loc[~duplicate].copy()
        output = pd.DataFrame(
            {
                "user_id": kept["user_id"].astype("int64"),
                "movie_id": kept["movie_id"].astype("int64"),
                "rating": kept["rating"].astype("float32"),
                "timestamp": pd.to_datetime(
                    kept["timestamp_seconds"].astype("int64"), unit="s", utc=True
                ),
            }
        ).sort_values(["user_id", "movie_id"], kind="mergesort")
        ratings_writer.write(output)
        stats = output.groupby("user_id", sort=True).agg(
            interaction_count=("movie_id", "size"),
            first_interaction_timestamp=("timestamp", "min"),
            last_interaction_timestamp=("timestamp", "max"),
        )
        for user_id, row in stats.iterrows():
            users_rows.append(
                {
                    "user_id": int(user_id),
                    "interaction_count": int(row["interaction_count"]),
                    "first_interaction_timestamp": row["first_interaction_timestamp"],
                    "last_interaction_timestamp": row["last_interaction_timestamp"],
                }
            )

    for raw in pd.read_csv(ratings_path, chunksize=chunk_size):
        rows_before += len(raw)
        for column, count in raw.isna().sum().items():
            nulls_before[column] += int(count)
        raw_user = pd.to_numeric(raw["userId"], errors="coerce")
        raw_movie = pd.to_numeric(raw["movieId"], errors="coerce")
        raw_rating = pd.to_numeric(raw["rating"], errors="coerce")
        raw_timestamp = pd.to_numeric(raw["timestamp"], errors="coerce")
        source_rows = pd.Series(
            np.arange(source_offset, source_offset + len(raw), dtype="int64"),
            index=raw.index,
        )
        source_offset += len(raw)

        if previous_raw_user is not None and raw_user.dropna().iloc[0] < previous_raw_user:
            raise RuntimeError(
                "ratings.csv is not grouped by userId; streaming duplicate resolution is unsafe"
            )
        previous_raw_user = int(raw_user.dropna().iloc[-1])

        valid_user = raw_user.notna() & (raw_user > 0) & (raw_user % 1 == 0)
        valid_movie = raw_movie.notna() & (raw_movie > 0) & (raw_movie % 1 == 0)
        valid_rating = raw_rating.between(minimum_rating, maximum_rating)
        valid_timestamp = raw_timestamp.notna() & (raw_timestamp > 0)
        source_int = raw_movie.astype("Int64")
        status = source_int.map(status_by_source).astype("string")
        mapped_movie = source_int.map(mapping_series).astype("Int64")

        reasons = pd.Series(pd.NA, index=raw.index, dtype="string")
        reasons.loc[~valid_user] = "invalid_user_id"
        reasons.loc[reasons.isna() & ~valid_movie] = "invalid_movielens_movie_id"
        reasons.loc[reasons.isna() & ~valid_rating] = "invalid_rating"
        reasons.loc[reasons.isna() & ~valid_timestamp] = "invalid_timestamp"
        reasons.loc[reasons.isna() & status.isna()] = "missing_link_record"
        reasons.loc[reasons.isna() & (status == "missing_tmdb_mapping")] = (
            "missing_tmdb_mapping"
        )
        reasons.loc[reasons.isna() & (status == "movie_metadata_missing")] = (
            "movie_metadata_missing"
        )
        reasons.loc[reasons.isna() & mapped_movie.isna()] = "unresolved_movie_mapping"

        invalid = reasons.notna()
        if invalid.any():
            rejected_parts.append(
                _prepare_rejected_ratings(
                    source_rows.loc[invalid],
                    raw_user.loc[invalid],
                    raw_movie.loc[invalid],
                    raw_rating.loc[invalid],
                    raw_timestamp.loc[invalid],
                    reasons.loc[invalid],
                )
            )
        good = pd.DataFrame(
            {
                "source_row": source_rows.loc[~invalid].astype("int64"),
                "user_id": raw_user.loc[~invalid].astype("int64"),
                "movielens_movie_id": raw_movie.loc[~invalid].astype("int64"),
                "movie_id": mapped_movie.loc[~invalid].astype("int64"),
                "rating": raw_rating.loc[~invalid].astype("float64"),
                "timestamp_seconds": raw_timestamp.loc[~invalid].astype("int64"),
            }
        )
        if not carry.empty:
            good = pd.concat([carry, good], ignore_index=True)
        boundary_user = int(raw_user.dropna().iloc[-1])
        carry = good.loc[good["user_id"] == boundary_user].copy()
        resolve_and_write(good.loc[good["user_id"] != boundary_user])

    resolve_and_write(carry)
    ratings_writer.close()
    rejected = pd.concat(rejected_parts, ignore_index=True)
    rejected = rejected.sort_values("source_row", kind="mergesort").reset_index(drop=True)
    _atomic_parquet(rejected, interim_dir / "rejected_ratings.parquet")

    users = pd.DataFrame(users_rows).sort_values("user_id", kind="mergesort")
    if users["user_id"].duplicated().any():
        raise RuntimeError("User chunks overlapped during streaming ratings cleaning")
    users["interaction_count"] = users["interaction_count"].astype("int64")
    _atomic_parquet(users, processed_dir / "users_clean.parquet")

    reason_counts = {
        str(key): int(value)
        for key, value in rejected["rejection_reason"].value_counts().items()
    }
    return {
        "rows_before": rows_before,
        "rows_after": ratings_writer.rows,
        "rows_removed": len(rejected),
        "rows_modified": ratings_writer.rows,
        "duplicates_removed": alias_duplicates,
        "null_values_before": dict(nulls_before),
        "null_values_after": {
            "user_id": 0,
            "movie_id": 0,
            "rating": 0,
            "timestamp": 0,
        },
        "removal_reasons": reason_counts,
        "users_after": len(users),
        "minimum_user_interactions_observed": int(users["interaction_count"].min()),
        "users_below_configured_minimum": int(
            (
                users["interaction_count"]
                < int(config["interactions"]["minimum_user_interactions"])
            ).sum()
        ),
        "sparse_users_removed": 0,
        "duplicate_resolution": (
            "For duplicate canonical user/movie pairs created by MovieLens aliases, "
            "keep the latest timestamp; break ties by MovieLens ID then source row."
        ),
    }


def _cleaning_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Cleaning summary",
        "",
        f"Overall status: **{summary['overall_status']}**",
        "",
        "Raw files were read in place and were not modified. Every removed movie or rating row is represented in a Parquet rejection log.",
        "",
        "| Table | Rows before | Rows after | Removed | Modified | Duplicates removed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("movies", "id_mapping", "ratings"):
        item = summary[name]
        lines.append(
            f"| {name} | {item['rows_before']:,} | {item['rows_after']:,} | "
            f"{item['rows_removed']:,} | {item['rows_modified']:,} | "
            f"{item['duplicates_removed']:,} |"
        )
    lines.extend(["", "## Removal reasons", ""])
    for name in ("movies", "id_mapping", "ratings"):
        lines.append(f"### {name}")
        lines.append("")
        reasons = summary[name].get("removal_reasons", {})
        if not reasons:
            lines.append("- None.")
        else:
            for reason, count in reasons.items():
                lines.append(f"- `{reason}`: {count:,}")
        lines.append("")
    lines.extend(
        [
            "## Resolution rules",
            "",
            "- Canonical movie ID is the valid integer TMDB ID from metadata.",
            "- Exact metadata duplicates are removed first. Remaining duplicate TMDB IDs keep the most complete record, then the highest vote count, then earliest source row.",
            "- Empty-title movies are rejected because title is a required serving field.",
            "- Ratings with missing or metadata-less mappings are rejected and logged.",
            f"- {summary['ratings']['duplicate_resolution']}",
            "- Sparse users are retained in cleaned interactions and reported; holdout eligibility is handled later during splitting.",
            "",
            "## Warnings",
            "",
        ]
    )
    for warning in summary["warnings"]:
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _data_dictionary_markdown() -> str:
    columns = [
        ("movies_clean", "movie_id", "int64", "no", "Canonical TMDB movie ID", "movies_metadata.id", "Numeric, positive, deduplicated", "yes", "yes"),
        ("movies_clean", "imdb_id", "string", "yes", "IMDb traceability ID", "movies_metadata.imdb_id", "Trim; missing tokens to null", "no", "no"),
        ("movies_clean", "title", "string", "no", "Display title", "movies_metadata.title", "Trim/collapse whitespace; reject empty", "yes", "yes"),
        ("movies_clean", "original_title", "string", "yes", "Original-language title", "movies_metadata.original_title", "Trim/collapse whitespace", "yes", "yes"),
        ("movies_clean", "overview", "string", "yes", "Movie synopsis", "movies_metadata.overview", "Trim/collapse whitespace", "yes", "yes"),
        ("movies_clean", "genres", "list<string>", "no", "Normalized genre names", "movies_metadata.genres", "Safe literal parse; unique order", "yes", "yes"),
        ("movies_clean", "original_language", "string", "yes", "Lowercase source language code", "movies_metadata.original_language", "Trim and lowercase", "yes", "yes"),
        ("movies_clean", "release_date", "timestamp", "yes", "Parsed release date", "movies_metadata.release_date", "Coerce invalid/out-of-range values to null", "yes", "yes"),
        ("movies_clean", "release_year", "int16", "yes", "Release year", "movies_metadata.release_date", "Derive from valid release date", "yes", "yes"),
        ("movies_clean", "runtime", "float32", "yes", "Runtime in minutes", "movies_metadata.runtime", "Numeric; non-positive to null", "yes", "yes"),
        ("movies_clean", "production_companies", "list<string>", "no", "Production company names", "movies_metadata.production_companies", "Safe literal parse and normalize", "yes", "yes"),
        ("movies_clean", "production_countries", "list<string>", "no", "Production country names", "movies_metadata.production_countries", "Safe literal parse and normalize", "yes", "yes"),
        ("movies_clean", "poster_path", "string", "yes", "Relative poster path", "movies_metadata.poster_path", "Trim; missing tokens to null", "yes", "no"),
        ("movies_clean", "vote_average", "float32", "yes", "TMDB vote mean for descriptive metadata", "movies_metadata.vote_average", "Numeric in [0,10]", "yes", "yes"),
        ("movies_clean", "vote_count", "int64", "yes", "TMDB vote count", "movies_metadata.vote_count", "Non-negative integer", "yes", "yes"),
        ("movies_clean", "popularity", "float32", "yes", "TMDB popularity measure", "movies_metadata.popularity", "Non-negative numeric", "yes", "yes"),
        ("ratings_clean", "user_id", "int64", "no", "MovieLens user ID", "ratings.userId", "Positive integer", "no", "yes"),
        ("ratings_clean", "movie_id", "int64", "no", "Canonical TMDB movie ID", "ratings.movieId + links.tmdbId", "Map through cleaned ID mapping", "no", "yes"),
        ("ratings_clean", "rating", "float32", "no", "Explicit rating value", "ratings.rating", "Numeric in configured range", "no", "yes"),
        ("ratings_clean", "timestamp", "timestamp[UTC]", "no", "Interaction time", "ratings.timestamp", "Unix seconds to UTC", "no", "yes"),
        ("users_clean", "user_id", "int64", "no", "Observed rating user", "ratings.userId", "Distinct valid user", "no", "yes"),
        ("users_clean", "interaction_count", "int64", "no", "Clean rating count", "ratings rows", "Count after mapping/deduplication", "no", "yes"),
        ("users_clean", "first_interaction_timestamp", "timestamp[UTC]", "no", "Earliest rating time", "ratings.timestamp", "Per-user minimum", "no", "yes"),
        ("users_clean", "last_interaction_timestamp", "timestamp[UTC]", "no", "Latest rating time", "ratings.timestamp", "Per-user maximum", "no", "yes"),
        ("id_mapping_clean", "movielens_movie_id", "int64", "no", "Source MovieLens movie ID", "links.movieId", "Positive and unique", "no", "yes"),
        ("id_mapping_clean", "movie_id", "int64", "no", "Canonical TMDB movie ID", "links.tmdbId", "Must exist in movies_clean", "no", "yes"),
        ("id_mapping_clean", "imdb_id", "string", "yes", "IMDb traceability ID", "links.imdbId", "Prefix tt; pad to at least 7 digits", "no", "no"),
        ("movie_genres_clean", "movie_id", "int64", "no", "Canonical movie foreign key", "movies_metadata.id", "Reference movies_clean", "no", "yes"),
        ("movie_genres_clean", "genre", "string", "no", "Normalized genre", "movies_metadata.genres.name", "Trim and deduplicate", "yes", "yes"),
        ("movie_companies_clean", "movie_id", "int64", "no", "Canonical movie foreign key", "movies_metadata.id", "Reference movies_clean", "no", "yes"),
        ("movie_companies_clean", "company_id", "int64", "yes", "Source company identifier", "production_companies.id", "Numeric when present", "no", "no"),
        ("movie_companies_clean", "company", "string", "no", "Company name", "production_companies.name", "Trim and deduplicate", "yes", "yes"),
        ("movie_countries_clean", "movie_id", "int64", "no", "Canonical movie foreign key", "movies_metadata.id", "Reference movies_clean", "no", "yes"),
        ("movie_countries_clean", "country_code", "string", "yes", "ISO country code", "production_countries.iso_3166_1", "Trim and uppercase", "yes", "yes"),
        ("movie_countries_clean", "country", "string", "no", "Country name", "production_countries.name", "Trim and deduplicate", "yes", "yes"),
        ("movie_keywords_clean", "movie_id", "int64", "no", "Canonical movie foreign key", "keywords.id", "Reference movies_clean; merge duplicates", "no", "yes"),
        ("movie_keywords_clean", "keywords", "list<string>", "no", "Normalized keywords", "keywords.keywords.name", "Safe parse; unique order", "no", "yes"),
        ("movie_credits_clean", "movie_id", "int64", "no", "Canonical movie foreign key", "credits.id", "Reference movies_clean; merge duplicates", "no", "yes"),
        ("movie_credits_clean", "cast_names", "list<string>", "no", "Ordered cast names", "credits.cast.name", "Safe parse; unique order", "no", "yes"),
        ("movie_credits_clean", "director_names", "list<string>", "no", "Director names", "credits.crew", "Filter job=Director; unique order", "no", "yes"),
        ("movie_content_features", "movie_id", "int64", "no", "Canonical movie key", "movies_clean.movie_id", "One row per clean movie", "no", "yes"),
        ("movie_content_features", "cleaned_text", "string", "no", "TF-IDF source document", "clean metadata, keywords, credits", "Normalize text and add prefixed categorical tokens", "no", "yes"),
        ("movie_content_features", "genres", "list<string>", "no", "Normalized genres", "movies_clean.genres", "Pass through canonical list", "no", "yes"),
        ("movie_content_features", "keywords", "list<string>", "no", "Normalized keywords", "movie_keywords_clean.keywords", "Left join; missing to empty list", "no", "yes"),
        ("movie_content_features", "cast_names", "list<string>", "no", "Top ordered cast names", "movie_credits_clean.cast_names", "Limit to configured count", "no", "yes"),
        ("movie_content_features", "director_names", "list<string>", "no", "Director names", "movie_credits_clean.director_names", "Left join; missing to empty list", "no", "yes"),
        ("movie_content_features", "numeric metadata", "mixed", "yes", "Language, year, runtime, vote and popularity features", "movies_clean", "Typed canonical values", "no", "yes"),
        ("movie_content_features", "availability indicators", "bool", "no", "Feature presence flags", "movies_clean/content joins", "Boolean derivation", "no", "yes"),
        ("user_item_interactions", "user_id", "int64", "no", "Returning-user key", "ratings_clean.user_id", "Pass through", "no", "yes"),
        ("user_item_interactions", "movie_id", "int64", "no", "Canonical item key", "ratings_clean.movie_id", "Pass through", "no", "yes"),
        ("user_item_interactions", "interaction_value", "float32", "no", "Observed rating", "ratings_clean.rating", "Rename only", "no", "yes"),
        ("user_item_interactions", "interaction_type", "string", "no", "Observed signal type", "dataset capability", "Constant rating; no fabricated signals", "no", "yes"),
        ("user_item_interactions", "timestamp", "timestamp[UTC]", "no", "Observed rating time", "ratings_clean.timestamp", "Pass through", "no", "yes"),
        ("interaction splits", "all interaction columns", "canonical", "no", "Chronological train/validation/test partitions", "user_item_interactions", "Latest=test, penultimate=validation for eligible users", "no", "yes"),
        ("top_rated rankings", "ranking_type", "string", "no", "ALL or GENRE ranking", "derived", "Configured scenario label", "yes", "no"),
        ("top_rated rankings", "genre", "string", "no", "ALL or normalized genre", "movies_clean.genres", "Exploded normalized genre", "yes", "no"),
        ("top_rated rankings", "rank", "int64", "no", "One-based deterministic rank", "weighted score", "Score/count/mean/movie ID ordering", "yes", "no"),
        ("top_rated rankings", "movie_id", "int64", "no", "Canonical ranked movie", "ratings_clean + movies_clean", "Foreign-key join", "yes", "no"),
        ("top_rated rankings", "score", "float64", "no", "Weighted rating score", "rating aggregates", "IMDb-style shrinkage", "yes", "no"),
        ("top_rated rankings", "average_rating", "float64", "no", "Clean mean rating", "ratings_clean.rating", "Per-movie mean", "yes", "no"),
        ("top_rated rankings", "rating_count", "int64", "no", "Clean rating count", "ratings_clean rows", "Per-movie count", "yes", "no"),
        ("movies_serving", "movie_id", "int64", "no", "Backend movie key", "movies_clean.movie_id", "Pass through", "yes", "no"),
        ("movies_serving", "display metadata", "mixed", "partly", "Title, year, genres, overview, poster, vote, popularity, runtime, language, companies, countries", "movies_clean", "Backend allowlist and field renames", "yes", "no"),
    ]
    lines = [
        "# Data dictionary",
        "",
        "This dictionary covers canonical processed tables. Feature, split, and serving schemas are added by their respective pipeline phases.",
        "",
        "| Final table | Column | Type | Nullable | Description | Source | Transformation | Backend | Model |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in columns)
    return "\n".join(lines) + "\n"


def _id_mapping_markdown(
    movie_summary: dict[str, Any],
    mapping_summary: dict[str, Any],
    rating_summary: dict[str, Any],
) -> str:
    reasons = rating_summary["removal_reasons"]
    return f"""# ID mapping

## Canonical identifiers

- `movie_id` is the positive integer TMDB ID from `movies_metadata.id`.
- `user_id` is the positive integer MovieLens ID from `ratings.userId`.
- `movielens_movie_id` is retained in `id_mapping_clean.parquet` only for source-to-canonical translation.
- `imdb_id` is retained only for traceability and is not a model key.

## Mapping direction

`ratings.movieId` -> `links.movieId` -> `links.tmdbId` -> `movies_clean.movie_id`

The clean mapping has {mapping_summary['rows_after']:,} MovieLens-to-canonical rows. A MovieLens ID maps to at most one canonical ID. A canonical ID may have multiple source aliases: {mapping_summary['canonical_movie_ids_with_multiple_movielens_ids']:,} canonical movies have more than one MovieLens ID ({mapping_summary['duplicate_alias_mapping_rows']:,} participating mapping rows).

## Unmatched and duplicate records

- Missing TMDB mappings removed from the mapping: {mapping_summary['removal_reasons'].get('missing_tmdb_mapping', 0):,}.
- Mappings whose TMDB ID has no retained metadata: {mapping_summary['removal_reasons'].get('movie_metadata_missing', 0):,}.
- Rating rows rejected for missing TMDB mapping: {reasons.get('missing_tmdb_mapping', 0):,}.
- Rating rows rejected because metadata is missing: {reasons.get('movie_metadata_missing', 0):,}.
- Canonical alias rating duplicates removed: {rating_summary['duplicates_removed']:,}.
- Invalid/duplicate/empty-title movie rows rejected: {movie_summary['rows_removed']:,}.

## Resolution rules

Metadata exact duplicates are removed first. For remaining duplicate TMDB IDs, the record with the most populated model/serving fields wins; ties use higher vote count and then earlier source row. Multiple MovieLens aliases remain in the mapping because they are real source identifiers. If aliases produce more than one rating for the same canonical user/movie pair, the latest timestamp wins; ties use the larger MovieLens ID and then later source row. Every losing row is retained in `data/interim/rejected_ratings.parquet`.
"""


def run_cleaning(config: dict[str, Any]) -> dict[str, Any]:
    """Run Phase C and write all required processed/rejection/report outputs."""
    movies, movie_summary = clean_movies(config)
    valid_movie_ids = set(movies["movie_id"].astype(int))
    content_summary = clean_content_tables(config, valid_movie_ids)
    mapping, status_by_source, mapping_summary = clean_id_mapping(
        config, valid_movie_ids
    )
    rating_summary = clean_ratings(config, mapping, status_by_source)

    samples_dir = output_dir(config, "samples_dir")
    sample_rows = int(config["processing"]["sample_rows"])
    movies.head(sample_rows).to_csv(
        samples_dir / "movies_clean_sample.csv", index=False, encoding="utf-8"
    )
    ratings_sample = pd.read_parquet(
        output_dir(config, "processed_dir") / "ratings_clean.parquet"
    ).head(sample_rows)
    ratings_sample.to_csv(
        samples_dir / "ratings_clean_sample.csv", index=False, encoding="utf-8"
    )

    warnings = [
        (
            f"{movie_summary['warnings']['runtime_over_configured_warning_max']} "
            "movies exceed the configured runtime warning threshold and were retained."
        ),
        (
            f"{movie_summary['warnings']['movies_without_genres']} movies have no genre "
            "metadata and were retained with an empty list."
        ),
        (
            f"{rating_summary['users_below_configured_minimum']:,} users have fewer than "
            f"{config['interactions']['minimum_user_interactions']} clean interactions; "
            "none were removed."
        ),
        "Only explicit ratings are available as returning-user interactions.",
    ]
    summary = {
        "overall_status": "WARNING" if warnings else "PASS",
        "movies": movie_summary,
        "id_mapping": mapping_summary,
        "ratings": rating_summary,
        "content": content_summary,
        "warnings": warnings,
    }
    validation_dir = output_dir(config, "validation_dir")
    write_json(validation_dir / "cleaning_summary.json", summary)
    (validation_dir / "cleaning_summary.md").write_text(
        _cleaning_markdown(summary), encoding="utf-8", newline="\n"
    )
    docs_dir = Path(config["_project_root"]) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "data_dictionary.md").write_text(
        _data_dictionary_markdown(), encoding="utf-8", newline="\n"
    )
    (docs_dir / "id_mapping.md").write_text(
        _id_mapping_markdown(movie_summary, mapping_summary, rating_summary),
        encoding="utf-8",
        newline="\n",
    )
    return summary
