"""CLI for Phase C metadata, mapping, content-source, and ratings cleaning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.cleaning import run_cleaning
from src.data.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    return parser.parse_args()


def main() -> int:
    config = load_config(parse_args().config)
    summary = run_cleaning(config)
    print("Cleaning completed:")
    for table in ("movies", "id_mapping", "ratings"):
        item = summary[table]
        print(
            f"- {table}: {item['rows_before']:,} -> {item['rows_after']:,} "
            f"({item['rows_removed']:,} removed)"
        )
    print(f"- status: {summary['overall_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

