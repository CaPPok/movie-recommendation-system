"""CLI for Phase E chronological splits and local serving exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config, output_dir
from src.data.serving_export import build_serving_exports
from src.data.splitting import build_interaction_splits
from src.utils.reporting import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    return parser.parse_args()


def main() -> int:
    config = load_config(parse_args().config)
    splits = build_interaction_splits(config)
    serving = build_serving_exports(config)
    write_json(
        output_dir(config, "validation_dir") / "serving_export_summary.json",
        serving,
    )
    print(
        f"Splits: train={splits['splits']['train']['rows']:,}, "
        f"validation={splits['splits']['validation']['rows']:,}, "
        f"test={splits['splits']['test']['rows']:,}"
    )
    print(
        f"Serving: {serving['movies_serving_rows']:,} movies, "
        f"{serving['popular_ranking_records']:,} ranking records"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

