"""Rerun core transformations and compare SHA-256 hashes of generated artifacts."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config, output_dir
from src.data.final_validation import run_final_validation
from src.pipeline import run_pipeline
from src.utils.reporting import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    return parser.parse_args()


def _artifact_paths(root: Path) -> list[Path]:
    patterns = [
        "data/interim/*.parquet",
        "data/processed/*.parquet",
        "data/features/*.parquet",
        "data/splits/*.parquet",
        "data/serving/*",
        "data/samples/*",
        "artifacts/content_based/*",
    ]
    paths = {
        path
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file() and not path.name.endswith(".tmp")
    }
    return sorted(paths)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _hash_file(path)
        for path in _artifact_paths(root)
    }


def main() -> int:
    config = load_config(parse_args().config)
    root = Path(config["_project_root"])
    before = _manifest(root)
    run_pipeline(config, include_profiling=False, include_validation=False)
    after = _manifest(root)
    all_names = sorted(set(before) | set(after))
    changed = [
        name for name in all_names if before.get(name) != after.get(name)
    ]
    summary: dict[str, Any] = {
        "all_match": not changed,
        "artifact_count_before": len(before),
        "artifact_count_after": len(after),
        "changed_artifacts": changed,
        "before_sha256": before,
        "after_sha256": after,
        "scope": (
            "Core deterministic data, model artifact, split, serving, and sample "
            "outputs. Time-stamped human reports are intentionally excluded."
        ),
    }
    validation_dir = output_dir(config, "validation_dir")
    write_json(validation_dir / "determinism_summary.json", summary)
    lines = [
        "# Determinism check",
        "",
        f"Status: **{'PASS' if not changed else 'FAIL'}**",
        "",
        f"Compared {len(all_names)} core generated artifacts before and after a full Phase C-E rerun using SHA-256.",
        "",
    ]
    if changed:
        lines.extend(["Changed artifacts:", ""])
        lines.extend(f"- `{name}`" for name in changed)
    else:
        lines.append("Every compared artifact matched byte-for-byte.")
    (validation_dir / "determinism_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    run_final_validation(config)
    print(
        f"Determinism: {'PASS' if not changed else 'FAIL'} "
        f"({len(all_names)} artifacts compared)"
    )
    return 0 if not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())

