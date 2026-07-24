"""CLI for the complete post-processing validation pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config
from src.data.final_validation import run_final_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    return parser.parse_args()


def main() -> int:
    report = run_final_validation(load_config(parse_args().config))
    counts = {
        status: sum(rule["status"] == status for rule in report["rules"])
        for status in ("PASS", "WARNING", "FAIL")
    }
    print(
        f"Final validation: {report['overall_status']} "
        f"({counts['PASS']} PASS, {counts['WARNING']} WARNING, {counts['FAIL']} FAIL)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

