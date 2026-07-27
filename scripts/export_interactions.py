"""Export interaction events from DynamoDB to JSONL for retraining.

Usage:
    python scripts/export_interactions.py --out data/events/2026-07-27.jsonl
    python scripts/export_interactions.py --upload
    python scripts/export_interactions.py --since 2026-07-01 --out events.jsonl

This is the first half of the feedback loop: backend writes events to the
`Interactions` table as users click, watch, share and comment; this script reads
them back out into the flat shape `retrain.py --events` consumes.

A full `Scan` is used because the table has no index on time and retraining
wants everything, not one user. That reads (and is billed for) the whole table.
Fine at demo scale; past a few hundred thousand events, replace this with a
DynamoDB point-in-time export to S3 and point `--events` at that prefix, which
costs a fraction of a Scan and does not consume read capacity.

Output is one JSON object per line:

    {"user_id": 12, "movie_id": 862, "event_type": "share",
     "value": null, "timestamp": "2026-07-27T11:00:00+00:00"}
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config
from src.models.interaction_weights import SUPPORTED_EVENT_TYPES


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws-config", default="configs/aws.yaml")
    parser.add_argument(
        "--out",
        default=None,
        help="Local JSONL path; defaults to data/events/<UTC date>.jsonl.",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Keep only events at or after this ISO-8601 instant.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Also upload the file to the configured S3 events prefix.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many items; 0 means no limit. For smoke tests.",
    )
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _plain(value: Any) -> Any:
    """DynamoDB returns Decimal for every number; JSON does not accept it."""
    if isinstance(value, Decimal):
        as_float = float(value)
        return int(as_float) if as_float.is_integer() else as_float
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _sort_key_timestamp(item: dict[str, Any], sort_key: str) -> str | None:
    """Recover the timestamp from the composite sort key when it is missing.

    Spec section 4.3 defines the sort key as `interaction_timestamp#movie_id`, so
    an item written by backend always carries the instant even if the optional
    `timestamp` attribute was left off.
    """
    raw = item.get(sort_key)
    if not isinstance(raw, str) or "#" not in raw:
        return None
    return raw.split("#", 1)[0] or None


def scan_events(
    table: Any, sort_key: str, page_size: int, limit: int
) -> Iterator[dict[str, Any]]:
    arguments: dict[str, Any] = {"Limit": page_size}
    seen = 0
    while True:
        response = table.scan(**arguments)
        for item in response.get("Items", []):
            item = {key: _plain(value) for key, value in item.items()}
            timestamp = item.get("timestamp") or _sort_key_timestamp(item, sort_key)
            yield {
                "user_id": item.get("user_id"),
                "movie_id": item.get("movie_id"),
                "event_type": item.get("event_type"),
                # `value` carries watch progress, star rating or comment
                # sentiment depending on the type; see
                # docs/interaction_events_api.md.
                "value": item.get("value"),
                "timestamp": timestamp,
            }
            seen += 1
            if limit and seen >= limit:
                return
        token = response.get("LastEvaluatedKey")
        if not token:
            return
        arguments["ExclusiveStartKey"] = token


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    aws_config = load_config(REPOSITORY_ROOT / arguments.aws_config)

    from src.aws import s3_sync

    settings = aws_config["dynamodb"]
    if str(settings["export_mode"]) != "scan":
        print(
            f"export_mode = {settings['export_mode']!r} chưa được hỗ trợ ở script "
            "này. Dùng DynamoDB point-in-time export rồi trỏ --events vào prefix "
            "S3 đó.",
            file=sys.stderr,
        )
        return 2

    since = None
    if arguments.since:
        since = datetime.fromisoformat(arguments.since.replace("Z", "+00:00"))
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

    destination = Path(
        arguments.out
        or REPOSITORY_ROOT / "data" / "events" / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        import boto3

        region = s3_sync.resolve_setting(aws_config, "region")
    except (ImportError, ValueError) as error:
        print(
            error
            if not isinstance(error, ImportError)
            else "boto3 chưa được cài. Chạy: pip install -r requirements-aws.txt",
            file=sys.stderr,
        )
        return 2

    resource = boto3.resource("dynamodb", region_name=region)
    table = resource.Table(str(settings["interactions_table"]))
    print(f"Scan bảng {settings['interactions_table']}...", flush=True)

    written = 0
    skipped_stale = 0
    skipped_unusable = 0
    unsupported: set[str] = set()
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for event in scan_events(
            table,
            str(settings["sort_key"]),
            int(settings["scan_page_size"]),
            arguments.limit,
        ):
            if event["user_id"] is None or event["movie_id"] is None:
                skipped_unusable += 1
                continue
            event_type = str(event.get("event_type") or "").strip().lower()
            if event_type not in SUPPORTED_EVENT_TYPES:
                # Exported anyway: retrain.py counts and reports these, and a
                # filter here would hide a frontend/model mismatch instead of
                # surfacing it.
                unsupported.add(event_type or "<empty>")
            if since and event["timestamp"]:
                try:
                    stamp = datetime.fromisoformat(
                        str(event["timestamp"]).replace("Z", "+00:00")
                    )
                except ValueError:
                    stamp = None
                if stamp is not None:
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    if stamp < since:
                        skipped_stale += 1
                        continue
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            written += 1

    print(f"  {written:,} event -> {destination}", flush=True)
    if skipped_stale:
        print(f"  {skipped_stale:,} event cũ hơn --since, đã bỏ.", flush=True)
    if skipped_unusable:
        print(f"  {skipped_unusable:,} event thiếu user_id/movie_id, đã bỏ.", flush=True)
    if unsupported:
        print(
            "  Event type model chưa hỗ trợ (vẫn xuất ra để đối chiếu): "
            + ", ".join(sorted(unsupported)),
            flush=True,
        )

    if arguments.upload:
        s3 = s3_sync.client(aws_config)
        location = s3_sync.location_for(aws_config, "events")
        uri = s3_sync.upload_file(s3, destination, location)
        print(f"  Đã tải lên {uri}", flush=True)
        print(f"\nChạy tiếp: python retrain.py --version vX.Y.Z --events {uri}")
    else:
        print(f"\nChạy tiếp: python retrain.py --version vX.Y.Z --events {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
