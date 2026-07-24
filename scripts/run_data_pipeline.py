"""Run the complete validated local movie data pipeline; no cloud operations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    return parser.parse_args()


def main() -> int:
    config = load_config(parse_args().config)
    result = run_pipeline(config)
    print(
        "Complete local pipeline finished with validation status: "
        f"{result['validation']['overall_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

