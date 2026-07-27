"""Push the dataset and artifacts to S3, or pull them back down.

Usage:
    python scripts/aws_sync.py push --dry-run
    python scripts/aws_sync.py push
    python scripts/aws_sync.py push --only artifacts
    python scripts/aws_sync.py pull --only splits serving
    python scripts/aws_sync.py list

The first upload of the processed dataset, and the way a fresh EC2 instance or a
Processing Job gets its inputs. Which local directory maps to which prefix is
declared in `configs/aws.yaml` under `sync.pairs`, so the mapping is data rather
than an argument someone can get wrong at 2am.

Raw Kaggle CSVs are not in `sync.pairs` on purpose: they are 700 MB, they never
change, and nothing downstream of cleaning reads them. Upload them once by hand
if the team wants the lineage stored:

    aws s3 sync data/movies_dataset_raw s3://<bucket>/raw/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config

# Parquet checksums and pytest leftovers have no business in the bucket.
DEFAULT_EXCLUDES = ("*.tmp", "**/*.tmp", ".gitkeep", "**/__pycache__/**")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["push", "pull", "list"])
    parser.add_argument("--aws-config", default="configs/aws.yaml")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="PREFIX",
        help="Limit to these configured prefixes, e.g. artifacts splits.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="On push, skip objects that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would move without transferring anything.",
    )
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    aws_config = load_config(REPOSITORY_ROOT / arguments.aws_config)

    from src.aws import s3_sync

    try:
        s3 = s3_sync.client(aws_config)
        bucket = s3_sync.resolve_setting(aws_config, "bucket")
    except (s3_sync.AwsDependencyError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    pairs = [
        pair
        for pair in aws_config["sync"]["pairs"]
        if arguments.only is None or pair["prefix"] in arguments.only
    ]
    if not pairs:
        print(
            f"Không có cặp nào khớp --only {arguments.only}. Đang cấu hình: "
            + ", ".join(pair["prefix"] for pair in aws_config["sync"]["pairs"]),
            file=sys.stderr,
        )
        return 2

    print(f"Bucket: s3://{bucket}", flush=True)
    results = []

    for pair in pairs:
        location = s3_sync.location_for(aws_config, pair["prefix"])
        local = REPOSITORY_ROOT / pair["local"]

        if arguments.action == "list":
            objects = list(s3_sync.list_objects(s3, location))
            total = sum(size for _, size in objects)
            print(
                f"  {location.uri}  {len(objects):,} object, "
                f"{total / 1_048_576:.1f} MB",
                flush=True,
            )
            results.append(
                {"prefix": location.uri, "objects": len(objects), "bytes": total}
            )
            continue

        if arguments.action == "push":
            result = s3_sync.upload_directory(
                s3,
                local,
                location,
                exclude=DEFAULT_EXCLUDES,
                overwrite=not arguments.no_overwrite,
                dry_run=arguments.dry_run,
            )
            print(
                f"  {pair['local']} -> {location.uri}  "
                f"{result.get('uploaded', 0):,} file, "
                f"{result.get('bytes', 0) / 1_048_576:.1f} MB"
                + (f"  ({result['skipped']:,} bỏ qua)" if result.get("skipped") else ""),
                flush=True,
            )
        else:
            result = s3_sync.download_directory(
                s3, location, local, dry_run=arguments.dry_run
            )
            print(
                f"  {location.uri} -> {pair['local']}  "
                f"{result.get('downloaded', 0):,} file, "
                f"{result.get('bytes', 0) / 1_048_576:.1f} MB",
                flush=True,
            )
        results.append(result)

    if arguments.dry_run:
        print("\nDry run: chưa truyền file nào.", file=sys.stderr)
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
