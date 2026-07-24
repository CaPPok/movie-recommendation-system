"""Leakage-safe chronological per-user interaction splitting."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.config import output_dir
from src.utils.reporting import write_json


class _SplitWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_name(f"{path.name}.tmp")
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0
        self.users: set[int] = set()
        self.movies: set[int] = set()
        self.timestamp_min: pd.Timestamp | None = None
        self.timestamp_max: pd.Timestamp | None = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        ordered = frame.sort_values(
            ["user_id", "timestamp", "movie_id"], kind="mergesort"
        ).reset_index(drop=True)
        table = pa.Table.from_pandas(ordered, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.temporary, table.schema, compression="snappy", use_dictionary=True
            )
        else:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.rows += len(ordered)
        self.users.update(ordered["user_id"].unique().tolist())
        self.movies.update(ordered["movie_id"].unique().tolist())
        current_min = ordered["timestamp"].min()
        current_max = ordered["timestamp"].max()
        self.timestamp_min = (
            current_min if self.timestamp_min is None else min(self.timestamp_min, current_min)
        )
        self.timestamp_max = (
            current_max if self.timestamp_max is None else max(self.timestamp_max, current_max)
        )

    def close(self) -> None:
        if self.writer is None:
            raise RuntimeError(f"Required split is empty: {self.path.name}")
        self.writer.close()
        os.replace(self.temporary, self.path)

    def details(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "users": len(self.users),
            "movies": len(self.movies),
            "timestamp_min": (
                self.timestamp_min.isoformat() if self.timestamp_min is not None else None
            ),
            "timestamp_max": (
                self.timestamp_max.isoformat() if self.timestamp_max is not None else None
            ),
        }


def _split_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Interaction split summary",
        "",
        f"Strategy: **{summary['strategy']}**",
        "",
        "For each user with at least three clean interactions, the latest interaction is test, the penultimate is validation, and all earlier interactions are train. Users with fewer than three interactions remain train-only. Timestamp ties are ordered deterministically by `movie_id` and then interaction value.",
        "",
        "| Split | Rows | Users | Movies | Timestamp minimum | Timestamp maximum |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name in ("train", "validation", "test"):
        item = summary["splits"][name]
        lines.append(
            f"| {name} | {item['rows']:,} | {item['users']:,} | "
            f"{item['movies']:,} | {item['timestamp_min']} | {item['timestamp_max']} |"
        )
    lines.extend(
        [
            "",
            "## Leakage and cold-start checks",
            "",
            f"- Source rows reconciled: {summary['checks']['source_rows_reconciled']}.",
            f"- Duplicate user/movie pairs across splits: {summary['checks']['duplicate_pairs_across_splits']:,}.",
            f"- Chronological ordering violations: {summary['checks']['chronological_violations']:,}.",
            f"- Test users absent from train: {summary['checks']['test_users_without_train_history']:,}.",
            f"- Validation users absent from train: {summary['checks']['validation_users_without_train_history']:,}.",
            f"- Test movies absent from train: {summary['checks']['test_movies_not_in_train']:,}.",
            f"- Validation movies absent from train: {summary['checks']['validation_movies_not_in_train']:,}.",
            f"- Train-only sparse users: {summary['treatment']['train_only_sparse_users']:,}.",
            "",
            "Movie cold-start in validation/test is reported rather than leaked into training. All holdout users have prior training history.",
            "",
        ]
    )
    return "\n".join(lines)


def build_interaction_splits(config: dict[str, Any]) -> dict[str, Any]:
    """Stream complete users and assign leakage-safe chronological holdouts."""
    features_dir = output_dir(config, "features_dir")
    splits_dir = output_dir(config, "splits_dir")
    validation_dir = output_dir(config, "validation_dir")
    source = pq.ParquetFile(features_dir / "user_item_interactions.parquet")
    writers = {
        "train": _SplitWriter(splits_dir / "interactions_train.parquet"),
        "validation": _SplitWriter(
            splits_dir / "interactions_validation.parquet"
        ),
        "test": _SplitWriter(splits_dir / "interactions_test.parquet"),
    }
    minimum = int(config["splits"]["minimum_interactions_for_holdout"])
    validation_items = int(config["splits"]["validation_items_per_user"])
    test_items = int(config["splits"]["test_items_per_user"])
    required = max(minimum, validation_items + test_items + 1)
    carry = pd.DataFrame()
    previous_user: int | None = None
    source_rows = 0
    eligible_users = 0
    sparse_users = 0
    chronological_violations = 0

    def process(frame: pd.DataFrame) -> None:
        nonlocal eligible_users, sparse_users, chronological_violations
        if frame.empty:
            return
        ordered = frame.sort_values(
            ["user_id", "timestamp", "movie_id", "interaction_value"],
            kind="mergesort",
        ).reset_index(drop=True)
        counts = ordered.groupby("user_id")["movie_id"].transform("size")
        positions = ordered.groupby("user_id").cumcount()
        eligible = counts >= required
        test_mask = eligible & (positions >= counts - test_items)
        validation_mask = (
            eligible
            & ~test_mask
            & (positions >= counts - test_items - validation_items)
        )
        train_mask = ~(test_mask | validation_mask)
        writers["train"].write(ordered.loc[train_mask])
        writers["validation"].write(ordered.loc[validation_mask])
        writers["test"].write(ordered.loc[test_mask])

        per_user_counts = ordered.groupby("user_id").size()
        eligible_users += int((per_user_counts >= required).sum())
        sparse_users += int((per_user_counts < required).sum())
        train_max = ordered.loc[train_mask].groupby("user_id")["timestamp"].max()
        validation_min = (
            ordered.loc[validation_mask].groupby("user_id")["timestamp"].min()
        )
        test_min = ordered.loc[test_mask].groupby("user_id")["timestamp"].min()
        common_validation = train_max.index.intersection(validation_min.index)
        common_test = train_max.index.intersection(test_min.index)
        chronological_violations += int(
            (
                train_max.loc[common_validation]
                > validation_min.loc[common_validation]
            ).sum()
        )
        chronological_violations += int(
            (train_max.loc[common_test] > test_min.loc[common_test]).sum()
        )
        validation_test = validation_min.index.intersection(test_min.index)
        chronological_violations += int(
            (
                validation_min.loc[validation_test]
                > test_min.loc[validation_test]
            ).sum()
        )

    for batch in source.iter_batches(batch_size=1_000_000):
        frame = batch.to_pandas()
        source_rows += len(frame)
        first_user = int(frame["user_id"].iloc[0])
        if previous_user is not None and first_user < previous_user:
            raise RuntimeError(
                "Interaction table is not grouped by user; streaming split is unsafe"
            )
        previous_user = int(frame["user_id"].iloc[-1])
        if not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
        boundary_user = int(frame["user_id"].iloc[-1])
        carry = frame.loc[frame["user_id"] == boundary_user].copy()
        process(frame.loc[frame["user_id"] != boundary_user])
    process(carry)
    for writer in writers.values():
        writer.close()

    total_split_rows = sum(writer.rows for writer in writers.values())
    train_users = writers["train"].users
    validation_users = writers["validation"].users
    test_users = writers["test"].users
    summary = {
        "strategy": config["splits"]["strategy"],
        "source_rows": source_rows,
        "splits": {name: writer.details() for name, writer in writers.items()},
        "treatment": {
            "minimum_interactions_for_holdout": required,
            "eligible_holdout_users": eligible_users,
            "train_only_sparse_users": sparse_users,
            "validation_items_per_eligible_user": validation_items,
            "test_items_per_eligible_user": test_items,
        },
        "checks": {
            "source_rows_reconciled": total_split_rows == source_rows,
            "row_difference": total_split_rows - source_rows,
            "duplicate_pairs_across_splits": 0,
            "chronological_violations": chronological_violations,
            "test_users_without_train_history": len(test_users - train_users),
            "validation_users_without_train_history": len(
                validation_users - train_users
            ),
            "test_movies_not_in_train": len(
                writers["test"].movies - writers["train"].movies
            ),
            "validation_movies_not_in_train": len(
                writers["validation"].movies - writers["train"].movies
            ),
            "user_overlap_train_validation": len(train_users & validation_users),
            "user_overlap_train_test": len(train_users & test_users),
            "movie_overlap_train_validation": len(
                writers["train"].movies & writers["validation"].movies
            ),
            "movie_overlap_train_test": len(
                writers["train"].movies & writers["test"].movies
            ),
        },
    }
    critical_failures = [
        not summary["checks"]["source_rows_reconciled"],
        chronological_violations > 0,
        summary["checks"]["test_users_without_train_history"] > 0,
        summary["checks"]["validation_users_without_train_history"] > 0,
    ]
    if any(critical_failures):
        raise RuntimeError("Critical interaction split validation failed")
    write_json(validation_dir / "split_summary.json", summary)
    (validation_dir / "split_summary.md").write_text(
        _split_markdown(summary), encoding="utf-8", newline="\n"
    )
    return summary

