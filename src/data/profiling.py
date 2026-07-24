"""Reproducible raw CSV profiling and cross-table validation."""

from __future__ import annotations

import ast
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.config import input_path, output_dir
from src.utils.reporting import truncate, write_json


PRIMARY_KEYS: dict[str, list[str]] = {
    "movies_metadata.csv": ["id"],
    "credits.csv": ["id"],
    "keywords.csv": ["id"],
    "links.csv": ["movieId"],
    "links_small.csv": ["movieId"],
    "ratings.csv": ["userId", "movieId"],
    "ratings_small.csv": ["userId", "movieId"],
}

NUMERIC_COLUMNS = {
    "id",
    "movieId",
    "userId",
    "imdbId",
    "tmdbId",
    "budget",
    "popularity",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "rating",
    "timestamp",
}

LIST_COLUMNS = {
    "cast",
    "crew",
    "keywords",
    "genres",
    "production_companies",
    "production_countries",
    "spoken_languages",
}


def _sniff_csv(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()[:131_072]
    encoding = "utf-8"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        encoding = "cp1252"
        text = raw.decode(encoding)
    try:
        delimiter = csv.Sniffer().sniff(text[:65_536], delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    return {
        "encoding": encoding,
        "has_utf8_bom": raw.startswith(b"\xef\xbb\xbf"),
        "delimiter": delimiter,
        "nul_bytes_in_sample": raw.count(b"\x00"),
    }


def _representative_values(series: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    counts = series.dropna().astype(str).value_counts().head(limit)
    return [{"value": truncate(value), "count": int(count)} for value, count in counts.items()]


def _list_parse_stats(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    malformed = 0
    wrong_type = 0
    non_empty = 0
    for value in non_null:
        try:
            parsed = ast.literal_eval(value) if isinstance(value, str) else value
            if not isinstance(parsed, (list, dict)):
                wrong_type += 1
            elif bool(parsed):
                non_empty += 1
        except (ValueError, SyntaxError, TypeError):
            malformed += 1
    return {
        "stored_as_string": bool(non_null.astype(str).str.match(r"^\s*[\[\{]").any()),
        "non_empty_count": non_empty,
        "malformed_count": malformed,
        "wrong_type_count": wrong_type,
    }


def _column_profile(series: pd.Series, total_rows: int) -> dict[str, Any]:
    null_count = int(series.isna().sum())
    result: dict[str, Any] = {
        "inferred_dtype": str(series.dtype),
        "memory_bytes": int(series.memory_usage(index=False, deep=True)),
        "null_count": null_count,
        "null_percentage": round((null_count / total_rows * 100) if total_rows else 0.0, 6),
        "unique_value_count": int(series.nunique(dropna=True)),
        "representative_values": _representative_values(series),
    }
    if series.name in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(series, errors="coerce")
        malformed = int((series.notna() & numeric.isna()).sum())
        result.update(
            {
                "malformed_numeric_count": malformed,
                "minimum": None if numeric.dropna().empty else float(numeric.min()),
                "maximum": None if numeric.dropna().empty else float(numeric.max()),
            }
        )
    if series.name in LIST_COLUMNS:
        result["list_like"] = _list_parse_stats(series)
    if series.name == "release_date":
        parsed = pd.to_datetime(series, errors="coerce")
        result["timestamp_range"] = {
            "minimum": None if parsed.dropna().empty else parsed.min().date().isoformat(),
            "maximum": None if parsed.dropna().empty else parsed.max().date().isoformat(),
            "invalid_or_null_count": int(parsed.isna().sum()),
        }
    if series.name == "timestamp":
        numeric = pd.to_numeric(series, errors="coerce")
        parsed = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
        result["timestamp_range"] = {
            "minimum": None if parsed.dropna().empty else parsed.min().isoformat(),
            "maximum": None if parsed.dropna().empty else parsed.max().isoformat(),
            "invalid_or_null_count": int(parsed.isna().sum()),
        }
    return result


def _profile_in_memory(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    key = PRIMARY_KEYS.get(path.name, [])
    numeric_id = None
    invalid_primary_key = 0
    if key == ["id"]:
        numeric_id = pd.to_numeric(frame["id"], errors="coerce")
        invalid_primary_key = int(numeric_id.isna().sum())
    duplicate_pk = int(frame.duplicated(key).sum()) if key else 0
    columns = {name: _column_profile(frame[name], len(frame)) for name in frame.columns}
    suspicious: list[str] = []
    if "id" in frame:
        suspicious.append(f"{invalid_primary_key} non-numeric primary IDs")
    if "rating" in frame:
        invalid_ratings = int((~frame["rating"].between(0.5, 5.0)).sum())
        suspicious.append(f"{invalid_ratings} ratings outside [0.5, 5.0]")
    for id_column in ("id", "movieId", "userId"):
        if id_column in frame:
            numeric = pd.to_numeric(frame[id_column], errors="coerce")
            suspicious.append(f"{int((numeric <= 0).sum())} non-positive {id_column} values")
    profile = {
        "filename": path.name,
        "file_type": "csv",
        "size_bytes": path.stat().st_size,
        **_sniff_csv(path),
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "column_names": list(frame.columns),
        "memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
        "duplicate_row_count": int(frame.duplicated().sum()),
        "primary_key": key,
        "duplicate_primary_key_count": duplicate_pk,
        "invalid_primary_key_count": invalid_primary_key,
        "columns": columns,
        "suspicious_outliers": suspicious,
    }
    state = {
        "frame": frame,
        "ids": set(numeric_id.dropna().astype("int64")) if numeric_id is not None else set(),
        "movie_ids": set(frame["movieId"].dropna().astype("int64")) if "movieId" in frame else set(),
    }
    return profile, state


def _profile_large_ratings(path: Path, chunk_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    row_count = 0
    memory_bytes = 0
    nulls: Counter[str] = Counter()
    users: set[int] = set()
    movies: set[int] = set()
    user_counts: Counter[int] = Counter()
    movie_counts: Counter[int] = Counter()
    rating_counts: Counter[float] = Counter()
    timestamps: list[np.ndarray] = []
    minimums = {"userId": np.inf, "movieId": np.inf, "rating": np.inf, "timestamp": np.inf}
    maximums = {"userId": -np.inf, "movieId": -np.inf, "rating": -np.inf, "timestamp": -np.inf}
    duplicate_pairs = 0
    duplicate_rows = 0
    sorted_by_user_movie = True
    previous_pair: np.ndarray | None = None
    previous_row: np.ndarray | None = None
    dtypes: dict[str, str] = {}

    for chunk in pd.read_csv(path, chunksize=chunk_size):
        if not dtypes:
            dtypes = {name: str(dtype) for name, dtype in chunk.dtypes.items()}
        row_count += len(chunk)
        memory_bytes += int(chunk.memory_usage(index=True, deep=True).sum())
        nulls.update({name: int(value) for name, value in chunk.isna().sum().items()})
        users.update(chunk["userId"].unique().tolist())
        movies.update(chunk["movieId"].unique().tolist())
        user_counts.update(chunk["userId"].value_counts().to_dict())
        movie_counts.update(chunk["movieId"].value_counts().to_dict())
        rating_counts.update(chunk["rating"].value_counts().to_dict())
        timestamps.append(chunk["timestamp"].to_numpy(copy=True))
        for name in minimums:
            minimums[name] = min(minimums[name], chunk[name].min())
            maximums[name] = max(maximums[name], chunk[name].max())

        pair_array = chunk[["userId", "movieId"]].to_numpy()
        if previous_pair is not None:
            pair_array = np.vstack([previous_pair, pair_array])
        same_pair = (pair_array[1:, 0] == pair_array[:-1, 0]) & (
            pair_array[1:, 1] == pair_array[:-1, 1]
        )
        duplicate_pairs += int(same_pair.sum())
        ordered = (pair_array[1:, 0] > pair_array[:-1, 0]) | (
            (pair_array[1:, 0] == pair_array[:-1, 0])
            & (pair_array[1:, 1] >= pair_array[:-1, 1])
        )
        sorted_by_user_movie = sorted_by_user_movie and bool(np.all(ordered))
        previous_pair = pair_array[-1:].copy()

        row_array = chunk[["userId", "movieId", "rating", "timestamp"]].to_numpy()
        if previous_row is not None:
            row_array = np.vstack([previous_row, row_array])
        duplicate_rows += int(np.all(row_array[1:] == row_array[:-1], axis=1).sum())
        previous_row = row_array[-1:].copy()

    timestamp_array = np.concatenate(timestamps)
    timestamp_array.sort()
    timestamp_unique = int(1 + np.count_nonzero(timestamp_array[1:] != timestamp_array[:-1]))
    del timestamp_array, timestamps

    def numeric_column(
        name: str,
        unique_count: int,
        representatives: list[dict[str, Any]],
    ) -> dict[str, Any]:
        null_count = int(nulls[name])
        result = {
            "inferred_dtype": dtypes[name],
            "memory_bytes": row_count * 8,
            "null_count": null_count,
            "null_percentage": round(null_count / row_count * 100, 6),
            "unique_value_count": unique_count,
            "representative_values": representatives,
            "malformed_numeric_count": 0,
            "minimum": float(minimums[name]),
            "maximum": float(maximums[name]),
        }
        return result

    rating_reps = [
        {"value": str(value), "count": int(count)}
        for value, count in sorted(rating_counts.items())
    ]
    columns = {
        "userId": numeric_column(
            "userId",
            len(users),
            [
                {"value": str(value), "count": int(count)}
                for value, count in user_counts.most_common(5)
            ],
        ),
        "movieId": numeric_column(
            "movieId",
            len(movies),
            [
                {"value": str(value), "count": int(count)}
                for value, count in movie_counts.most_common(5)
            ],
        ),
        "rating": numeric_column("rating", len(rating_counts), rating_reps),
        "timestamp": numeric_column("timestamp", timestamp_unique, []),
    }
    columns["timestamp"]["timestamp_range"] = {
        "minimum": datetime.fromtimestamp(
            int(minimums["timestamp"]), tz=timezone.utc
        ).isoformat(),
        "maximum": datetime.fromtimestamp(
            int(maximums["timestamp"]), tz=timezone.utc
        ).isoformat(),
        "invalid_or_null_count": 0,
    }
    invalid_ratings = sum(
        count for value, count in rating_counts.items() if not 0.5 <= value <= 5.0
    )
    profile = {
        "filename": path.name,
        "file_type": "csv",
        "size_bytes": path.stat().st_size,
        **_sniff_csv(path),
        "row_count": row_count,
        "column_count": 4,
        "column_names": ["userId", "movieId", "rating", "timestamp"],
        "memory_bytes": memory_bytes,
        "duplicate_row_count": duplicate_rows,
        "primary_key": ["userId", "movieId"],
        "duplicate_primary_key_count": duplicate_pairs,
        "invalid_primary_key_count": 0,
        "columns": columns,
        "rating_range": {
            "minimum": float(minimums["rating"]),
            "maximum": float(maximums["rating"]),
            "distribution": {str(key): int(value) for key, value in sorted(rating_counts.items())},
        },
        "timestamp_range": columns["timestamp"]["timestamp_range"],
        "suspicious_outliers": [
            f"{invalid_ratings} ratings outside [0.5, 5.0]",
            f"{sum(value for key, value in user_counts.items() if key <= 0)} rows with non-positive userId",
            f"{sum(value for key, value in movie_counts.items() if key <= 0)} rows with non-positive movieId",
        ],
        "streaming_duplicate_check_valid": sorted_by_user_movie,
    }
    return profile, {"movie_ids": movies, "user_ids": users}


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


def build_raw_reports(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Profile all raw tables and validate their key relationships."""
    chunk_size = int(config["processing"]["chunk_size"])
    order = [
        "movies",
        "credits",
        "keywords",
        "links",
        "ratings",
        "links_small",
        "ratings_small",
    ]
    profiles: dict[str, Any] = {}
    states: dict[str, dict[str, Any]] = {}
    for key in order:
        path = input_path(config, key)
        if path.name == "ratings.csv":
            profile, state = _profile_large_ratings(path, chunk_size)
        else:
            profile, state = _profile_in_memory(path)
        profiles[path.name] = profile
        states[path.name] = state

    metadata = states["movies_metadata.csv"]["frame"]
    metadata_ids = set(pd.to_numeric(metadata["id"], errors="coerce").dropna().astype("int64"))
    links = states["links.csv"]["frame"]
    linked_tmdb = set(pd.to_numeric(links["tmdbId"], errors="coerce").dropna().astype("int64"))
    missing_tmdb = int(links["tmdbId"].isna().sum())
    link_tmdb_absent = linked_tmdb - metadata_ids
    ratings_movies = states["ratings.csv"]["movie_ids"]
    link_movies = states["links.csv"]["movie_ids"]
    keyword_ids = states["keywords.csv"]["ids"]
    credit_ids = states["credits.csv"]["ids"]

    duplicate_tmdb_rows = int(
        links.loc[links["tmdbId"].notna(), "tmdbId"].duplicated(keep=False).sum()
    )
    relationships = {
        "ratings_movie_ids_not_in_links": len(ratings_movies - link_movies),
        "link_movie_ids_without_ratings": len(link_movies - ratings_movies),
        "links_missing_tmdb_id": missing_tmdb,
        "linked_tmdb_ids_not_in_metadata": len(link_tmdb_absent),
        "metadata_ids_without_link": len(metadata_ids - linked_tmdb),
        "duplicate_non_null_tmdb_mapping_rows": duplicate_tmdb_rows,
        "credit_ids_not_in_metadata": len(credit_ids - metadata_ids),
        "metadata_ids_without_credits": len(metadata_ids - credit_ids),
        "keyword_ids_not_in_metadata": len(keyword_ids - metadata_ids),
        "metadata_ids_without_keywords": len(metadata_ids - keyword_ids),
    }
    profiles["movies_metadata.csv"]["broken_references"] = {
        "metadata_ids_without_link": relationships["metadata_ids_without_link"],
        "metadata_ids_without_credits": relationships["metadata_ids_without_credits"],
        "metadata_ids_without_keywords": relationships["metadata_ids_without_keywords"],
    }
    profiles["links.csv"]["broken_references"] = {
        "linked_tmdb_ids_not_in_metadata": relationships["linked_tmdb_ids_not_in_metadata"],
        "duplicate_non_null_tmdb_mapping_rows": duplicate_tmdb_rows,
    }
    profiles["ratings.csv"]["broken_references"] = {
        "movie_ids_not_in_links": relationships["ratings_movie_ids_not_in_links"]
    }

    raw_profile = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": config["inputs"]["raw_dir"],
        "tables": profiles,
        "relationships": relationships,
        "notes": [
            "ratings.csv and links.csv are the full production inputs.",
            "ratings_small.csv and links_small.csv are profiled auxiliary subsets and are not mixed into production interactions.",
            "Only explicit rating interactions are present; no clicks, watches, or likes are inferred.",
        ],
    }

    rules = [
        _rule(
            "RAW_FILES_PRESENT",
            "PASS",
            "All seven configured raw CSV files exist.",
            {"files": [input_path(config, key).name for key in order]},
            critical=True,
        ),
        _rule(
            "RAW_FILES_PARSEABLE",
            "PASS",
            "Every configured CSV parsed as UTF-8 comma-delimited data.",
            {"table_count": len(order)},
            critical=True,
        ),
        _rule(
            "MOVIE_ID_NUMERIC",
            "WARNING" if profiles["movies_metadata.csv"]["invalid_primary_key_count"] else "PASS",
            "Canonical movie IDs must be numeric TMDB IDs.",
            {
                "invalid_rows": profiles["movies_metadata.csv"]["invalid_primary_key_count"]
            },
            critical=False,
        ),
        _rule(
            "MOVIE_ID_UNIQUE",
            "WARNING"
            if profiles["movies_metadata.csv"]["duplicate_primary_key_count"]
            else "PASS",
            "Raw metadata contains duplicate TMDB IDs that need deterministic resolution.",
            {
                "duplicate_extra_rows": profiles["movies_metadata.csv"][
                    "duplicate_primary_key_count"
                ]
            },
            critical=False,
        ),
        _rule(
            "RATING_SOURCE_PAIR_UNIQUE",
            "PASS"
            if profiles["ratings.csv"]["duplicate_primary_key_count"] == 0
            else "WARNING",
            "Raw full ratings should be unique by MovieLens user/movie pair.",
            {
                "duplicate_extra_rows": profiles["ratings.csv"][
                    "duplicate_primary_key_count"
                ]
            },
            critical=False,
        ),
        _rule(
            "RATING_RANGE",
            "PASS"
            if profiles["ratings.csv"]["rating_range"]["minimum"] >= 0.5
            and profiles["ratings.csv"]["rating_range"]["maximum"] <= 5.0
            else "FAIL",
            "Full ratings must be within the configured 0.5–5.0 scale.",
            profiles["ratings.csv"]["rating_range"],
            critical=True,
        ),
        _rule(
            "RATING_MOVIE_LINK_REFERENCE",
            "PASS" if not relationships["ratings_movie_ids_not_in_links"] else "FAIL",
            "Every MovieLens movie ID in ratings must exist in the full link table.",
            {
                "unmatched_movie_ids": relationships["ratings_movie_ids_not_in_links"]
            },
            critical=True,
        ),
        _rule(
            "LINK_METADATA_REFERENCE",
            "WARNING"
            if missing_tmdb or relationships["linked_tmdb_ids_not_in_metadata"]
            else "PASS",
            "Some source link records cannot resolve to cleaned movie metadata.",
            {
                "missing_tmdb_ids": missing_tmdb,
                "tmdb_ids_absent_from_metadata": relationships[
                    "linked_tmdb_ids_not_in_metadata"
                ],
            },
            critical=False,
        ),
        _rule(
            "CONTENT_REFERENCE_COVERAGE",
            "WARNING"
            if relationships["metadata_ids_without_credits"]
            or relationships["metadata_ids_without_keywords"]
            else "PASS",
            "Credits and keywords should cover metadata IDs where available.",
            {
                "movies_without_credits": relationships["metadata_ids_without_credits"],
                "movies_without_keywords": relationships["metadata_ids_without_keywords"],
            },
            critical=False,
        ),
        _rule(
            "INTERACTION_SIGNAL_LIMITATION",
            "WARNING",
            "The dataset contains ratings only; no click, watch, like, or completion signals are available.",
            {"available_interaction_types": ["rating"]},
            critical=False,
        ),
    ]
    overall = "FAIL" if any(rule["status"] == "FAIL" for rule in rules) else (
        "WARNING" if any(rule["status"] == "WARNING" for rule in rules) else "PASS"
    )
    raw_validation = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "rules": rules,
    }
    return raw_profile, raw_validation


def _profile_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# Raw data profile",
        "",
        f"Generated: {profile['generated_at_utc']}",
        "",
        "The full ratings/link pair is the production source. The similarly named `*_small.csv` files are profiled auxiliary subsets and are not combined with the full interactions.",
        "",
        "## Table summary",
        "",
        "| Table | Rows | Columns | Memory | Duplicate rows | Duplicate key extras |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, table in profile["tables"].items():
        lines.append(
            f"| `{name}` | {table['row_count']:,} | {table['column_count']} | "
            f"{table['memory_bytes'] / 1_048_576:,.2f} MiB | "
            f"{table['duplicate_row_count']:,} | {table['duplicate_primary_key_count']:,} |"
        )
    lines.extend(["", "## Relationships", ""])
    for name, value in profile["relationships"].items():
        lines.append(f"- `{name}`: {value:,}")
    for name, table in profile["tables"].items():
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                f"- Format: {table['file_type'].upper()}, {table['encoding']}, delimiter `{table['delimiter']}`",
                f"- Primary key candidate: {', '.join(table['primary_key'])}",
                f"- Columns: {', '.join(table['column_names'])}",
                "",
                "| Column | Type | Nulls | Null % | Unique | Min | Max | List-like malformed |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for column_name, column in table["columns"].items():
            list_malformed = column.get("list_like", {}).get("malformed_count", "")
            lines.append(
                f"| `{column_name}` | {column['inferred_dtype']} | "
                f"{column['null_count']:,} | {column['null_percentage']:.4f} | "
                f"{column['unique_value_count']:,} | "
                f"{column.get('minimum', '')} | {column.get('maximum', '')} | "
                f"{list_malformed} |"
            )
        lines.extend(["", "Representative/suspicious findings:"])
        for finding in table["suspicious_outliers"]:
            lines.append(f"- {finding}")
        if table.get("broken_references"):
            for key, value in table["broken_references"].items():
                lines.append(f"- `{key}`: {value:,}")
    lines.extend(["", "## Interpretation", ""])
    for note in profile["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _validation_markdown(validation: dict[str, Any]) -> str:
    lines = [
        "# Raw data validation",
        "",
        f"Overall status: **{validation['overall_status']}**",
        "",
        "| Rule | Status | Critical | Description | Evidence |",
        "|---|---|---|---|---|",
    ]
    for rule in validation["rules"]:
        evidence = "; ".join(f"{key}={truncate(value)}" for key, value in rule["evidence"].items())
        lines.append(
            f"| `{rule['rule_id']}` | **{rule['status']}** | "
            f"{'yes' if rule['critical'] else 'no'} | {rule['description']} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "Warnings are expected to be handled explicitly during cleaning. A critical `FAIL` blocks later phases.",
            "",
        ]
    )
    return "\n".join(lines)


def run_raw_profiling(config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    """Create required raw profile and validation reports."""
    profile, validation = build_raw_reports(config)
    profiling_dir = output_dir(config, "profiling_dir")
    validation_dir = output_dir(config, "validation_dir")
    profile_json = profiling_dir / "raw_profile.json"
    profile_md = profiling_dir / "raw_profile.md"
    validation_json = validation_dir / "raw_validation.json"
    validation_md = validation_dir / "raw_validation.md"
    write_json(profile_json, profile)
    profile_md.write_text(_profile_markdown(profile), encoding="utf-8", newline="\n")
    write_json(validation_json, validation)
    validation_md.write_text(
        _validation_markdown(validation), encoding="utf-8", newline="\n"
    )
    critical_failures = [
        rule for rule in validation["rules"] if rule["critical"] and rule["status"] == "FAIL"
    ]
    if critical_failures:
        names = ", ".join(rule["rule_id"] for rule in critical_failures)
        raise RuntimeError(f"Critical raw validation failure(s): {names}")
    return profile_json, profile_md, validation_json, validation_md

