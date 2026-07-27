"""Turn production interaction events into rows the trainer can learn from.

The feedback loop closes here. Backend writes `click`, `watch`, `complete`,
`like`, `dislike`, `rating`, `share` and `comment` events to DynamoDB; an export
drops them in S3 as JSONL; this module folds them into
`data/features/user_item_interactions.parquet`, which the existing chronological
splitter and ALS trainer already consume unchanged.

Three decisions shape the conversion, and all three exist to keep a retrained
model comparable with the one it replaces.

**Events become pseudo-ratings, not a second signal.** `build_training_matrix`
thresholds on the 0.5-5.0 scale. Rather than teach it a second scale, an
aggregated event score is projected onto the rating axis through three anchors,
so a share lands where a five-star rating lands and a hostile comment lands
where a one-star rating lands.

**Decay is off.** At serving time, decay answers "what does this user want
today". Training wants the observed history weighted the same way on every run.
With decay on, two runs over identical data would produce different training
sets, and no regression in the metrics could be attributed to anything.

**Explicit ratings win.** A user who both rated a film and interacted with it
produces two candidate rows; the rating is kept. It is the signal every
historical row already carries, and mixing a derived value into the same
(user, movie) cell would make the two eras incomparable.

Derived rows are marked `interaction_type = "event"`, so evaluation and reports
can always separate them from the `rating` rows that came out of the Kaggle
dataset. That distinction is required by `MODEL_DESIGN_SPEC.md` section 11.3.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.config import output_dir
from src.models.interaction_weights import build_interaction_profile

INTERACTIONS_FILE = "user_item_interactions.parquet"
EVENT_INTERACTION_TYPE = "event"
RATING_INTERACTION_TYPE = "rating"

# Row groups are read user-aligned, so a user's history is never split across
# two emitted chunks. Matches the batch size the splitter already uses.
MERGE_BATCH_ROWS = 1_000_000


def score_to_rating(score: float, retraining_config: Mapping[str, Any]) -> float:
    """Project an aggregated event score onto the 0.5-5.0 rating scale.

    Piecewise-linear through three anchors so the mapping stays monotonic and
    has no discontinuity at zero: a score of 0 is exactly neutral, and the two
    halves are scaled independently because the positive and negative event
    weights are not symmetric.
    """
    anchors = retraining_config["event_rating"]
    neutral = float(anchors["neutral_rating"])
    minimum = float(anchors["minimum_rating"])
    maximum = float(anchors["maximum_rating"])
    positive_anchor = float(anchors["positive_anchor_score"])
    negative_anchor = float(anchors["negative_anchor_score"])

    if score >= 0:
        span = positive_anchor if positive_anchor > 0 else 1.0
        rating = neutral + (maximum - neutral) * min(score / span, 1.0)
    else:
        span = abs(negative_anchor) if negative_anchor else 1.0
        rating = neutral - (neutral - minimum) * min(abs(score) / span, 1.0)
    return float(min(max(rating, minimum), maximum))


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(stamp) else stamp


def events_to_interactions(
    events: Iterable[Mapping[str, Any]],
    model_config: Mapping[str, Any],
    retraining_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate a flat event stream into one training row per (user, movie).

    Each record needs `user_id`, `movie_id`, `event_type`; `value` and
    `timestamp` follow the same rules as the serving contract in
    `docs/interaction_events_api.md`. Records missing a usable `user_id` are
    counted and dropped, because an interaction with nobody attached cannot
    train a collaborative model.
    """
    # Serving weights, serving caps, training decay. The caps stay on: someone
    # who shared one film forty times should not out-vote forty other users.
    settings = dict(model_config["interactions"])
    settings["half_life_days"] = float(retraining_config["half_life_days"])

    by_user: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    latest: dict[tuple[int, int], pd.Timestamp] = {}
    dropped_no_user = 0
    dropped_no_timestamp = 0
    received = 0

    for record in events:
        received += 1
        if not isinstance(record, Mapping):
            dropped_no_user += 1
            continue
        try:
            user_id = int(record["user_id"])
        except (KeyError, TypeError, ValueError):
            dropped_no_user += 1
            continue
        by_user[user_id].append(record)

        try:
            movie_id = int(record.get("movie_id"))
        except (TypeError, ValueError):
            continue
        stamp = _coerce_timestamp(record.get("timestamp"))
        if stamp is None:
            dropped_no_timestamp += 1
            continue
        key = (user_id, movie_id)
        if key not in latest or stamp > latest[key]:
            latest[key] = stamp

    rows: list[tuple[int, int, float, pd.Timestamp]] = []
    counted = 0
    ignored = 0
    capped = 0
    unsupported: set[str] = set()
    for user_id, user_events in by_user.items():
        # Newest first, so the per-movie caps drop the oldest repeats.
        ordered = sorted(
            user_events,
            key=lambda item: _coerce_timestamp(item.get("timestamp"))
            or pd.Timestamp.min.tz_localize("UTC"),
            reverse=True,
        )
        profile = build_interaction_profile(ordered, settings)
        counted += profile.counted_events
        ignored += profile.ignored_events
        capped += profile.capped_events
        unsupported |= profile.unsupported_types
        for movie_id, score in profile.scores.items():
            stamp = latest.get((user_id, movie_id))
            if stamp is None:
                # No usable timestamp anywhere for this pair. The chronological
                # splitter orders on it, so a row without one cannot be placed.
                continue
            rows.append((user_id, movie_id, score_to_rating(score, retraining_config), stamp))

    frame = pd.DataFrame(
        rows, columns=["user_id", "movie_id", "interaction_value", "timestamp"]
    )
    frame = frame.astype(
        {"user_id": "int64", "movie_id": "int64", "interaction_value": "float32"}
    )
    # Dtypes are pinned rather than inferred: the merge concatenates these rows
    # with the existing parquet, and a tz-naive or object-dtype timestamp column
    # would either fail the concat or silently widen the schema.
    frame["timestamp"] = pd.to_datetime(
        pd.Series(frame["timestamp"], dtype="object"), utc=True
    ).astype("datetime64[ns, UTC]")
    frame["interaction_type"] = pd.Series(
        EVENT_INTERACTION_TYPE, index=frame.index, dtype="string"
    )
    frame = frame[
        ["user_id", "movie_id", "interaction_value", "interaction_type", "timestamp"]
    ]

    summary = {
        "events_received": received,
        "events_counted": counted,
        "events_ignored": ignored,
        "events_capped": capped,
        "events_without_user_id": dropped_no_user,
        "events_without_timestamp": dropped_no_timestamp,
        "unsupported_event_types": sorted(unsupported),
        "users": int(frame["user_id"].nunique()) if not frame.empty else 0,
        "movies": int(frame["movie_id"].nunique()) if not frame.empty else 0,
        "rows": int(len(frame)),
        "half_life_days": float(settings["half_life_days"]),
    }
    return frame, summary


def restrict_to_catalogue(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, int]:
    """Drop rows for movies that are not in the clean catalogue.

    A `movie_id` the pipeline rejected has no content features and no serving
    record. Training on it produces an item factor the engine can never return,
    and the response assertion in `engine._build_response` would fail if it
    somehow did.
    """
    if frame.empty:
        return frame, 0
    processed_dir = output_dir(config, "processed_dir", create=False)
    catalogue = pd.read_parquet(
        processed_dir / "movies_clean.parquet", columns=["movie_id"]
    )["movie_id"].astype("int64")
    known = frame["movie_id"].isin(pd.Index(catalogue))
    return frame.loc[known].copy(), int((~known).sum())


def merge_into_interactions(
    config: Mapping[str, Any],
    new_rows: pd.DataFrame,
    conflict_policy: str = "prefer_rating",
) -> dict[str, Any]:
    """Fold derived rows into the canonical interaction table, in place.

    The table is 25M+ rows, so it is rewritten by streaming rather than loaded.
    Two invariants have to survive the rewrite, because the chronological
    splitter depends on both: rows stay grouped by `user_id` in ascending order,
    and every row of one user stays contiguous. The batch loop therefore carries
    the boundary user forward, exactly as `src/data/splitting.py` does, and only
    injects new rows once a user's existing rows are all in hand.
    """
    if conflict_policy not in {"prefer_rating", "prefer_event"}:
        raise ValueError(
            "event_conflict_policy must be 'prefer_rating' or 'prefer_event', "
            f"got {conflict_policy!r}"
        )

    features_dir = output_dir(config, "features_dir", create=False)
    target = features_dir / INTERACTIONS_FILE
    if not target.exists():
        raise FileNotFoundError(
            f"Chưa có {target}. Chạy `python scripts/run_data_pipeline.py` trước."
        )
    if new_rows.empty:
        return {
            "merged_rows": 0,
            "replaced_rows": 0,
            "source_rows": int(pq.ParquetFile(target).metadata.num_rows),
            "output_rows": int(pq.ParquetFile(target).metadata.num_rows),
            "changed": False,
        }

    pending = new_rows.sort_values(
        ["user_id", "timestamp", "movie_id"], kind="mergesort"
    ).reset_index(drop=True)
    source = pq.ParquetFile(target)
    source_rows = int(source.metadata.num_rows)
    temporary = target.with_name(f"{target.name}.tmp")
    writer: pq.ParquetWriter | None = None
    cursor = 0
    merged = 0
    replaced = 0
    output_rows = 0

    def emit(existing: pd.DataFrame) -> None:
        nonlocal writer, cursor, merged, replaced, output_rows
        if existing.empty:
            return
        highest_user = int(existing["user_id"].max())
        take = pending.loc[cursor:]
        take = take.loc[take["user_id"] <= highest_user]
        cursor += len(take)

        if not take.empty:
            if conflict_policy == "prefer_rating":
                occupied = set(
                    zip(
                        existing["user_id"].to_numpy(),
                        existing["movie_id"].to_numpy(),
                        strict=True,
                    )
                )
                keep = [
                    (int(user), int(movie)) not in occupied
                    for user, movie in zip(
                        take["user_id"].to_numpy(),
                        take["movie_id"].to_numpy(),
                        strict=True,
                    )
                ]
                replaced += int(len(take) - sum(keep))
                take = take.loc[np.asarray(keep)]
            else:
                pairs = set(
                    zip(
                        take["user_id"].to_numpy(),
                        take["movie_id"].to_numpy(),
                        strict=True,
                    )
                )
                before = len(existing)
                existing = existing.loc[
                    [
                        (int(user), int(movie)) not in pairs
                        for user, movie in zip(
                            existing["user_id"].to_numpy(),
                            existing["movie_id"].to_numpy(),
                            strict=True,
                        )
                    ]
                ]
                replaced += before - len(existing)

        combined = (
            pd.concat([existing, take], ignore_index=True) if not take.empty else existing
        )
        combined = combined.sort_values(
            ["user_id", "timestamp", "movie_id"], kind="mergesort"
        ).reset_index(drop=True)
        merged += len(take)
        output_rows += len(combined)

        table = pa.Table.from_pandas(combined, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                temporary, table.schema, compression="snappy", use_dictionary=True
            )
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)

    carry = pd.DataFrame()
    for batch in source.iter_batches(batch_size=MERGE_BATCH_ROWS):
        frame = batch.to_pandas()
        if not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
        boundary_user = int(frame["user_id"].iloc[-1])
        carry = frame.loc[frame["user_id"] == boundary_user].copy()
        emit(frame.loc[frame["user_id"] != boundary_user])
    emit(carry)

    # Users who appear only in the new events sort after every existing user and
    # were never reached by the loop above.
    leftover = pending.loc[cursor:]
    if not leftover.empty:
        emit_leftover = leftover.sort_values(
            ["user_id", "timestamp", "movie_id"], kind="mergesort"
        ).reset_index(drop=True)
        table = pa.Table.from_pandas(emit_leftover, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                temporary, table.schema, compression="snappy", use_dictionary=True
            )
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        merged += len(emit_leftover)
        output_rows += len(emit_leftover)

    if writer is None:
        raise RuntimeError("Interaction merge produced no output")
    writer.close()
    # The source and the destination are the same path, so the read handle has to
    # be released before the swap. On Windows os.replace fails outright while the
    # file is still open; on Linux it succeeds and leaves the old inode alive,
    # which is worse because the bug only shows up as stale memory-mapped reads.
    source.close()
    os.replace(temporary, target)

    return {
        "merged_rows": merged,
        "replaced_rows": replaced,
        "source_rows": source_rows,
        "output_rows": output_rows,
        "conflict_policy": conflict_policy,
        "changed": merged > 0 or replaced > 0,
    }
