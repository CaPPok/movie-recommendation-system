"""CLI for raw profiling and raw validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config
from src.data.profiling import run_raw_profiling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data_pipeline.yaml",
        help="Repository-relative YAML configuration path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    outputs = run_raw_profiling(config)
    print("Raw profiling completed:")
    for output in outputs:
        print(f"- {output.relative_to(config['_project_root'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

