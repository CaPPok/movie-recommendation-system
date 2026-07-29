"""Convert an existing vectorizer.joblib into the portable artifact files.

Usage:
    python scripts/convert_vectorizer.py
    python scripts/convert_vectorizer.py --check
    python scripts/convert_vectorizer.py --config configs/data_pipeline.yaml

Run once, on the machine whose scikit-learn wrote the pickle. Nothing is
recomputed: the script opens the fitted vectorizer, copies out the vocabulary,
the idf weights and the analyzer settings, and writes them as JSON and `.npy`.
No pipeline stage reruns, no model is retrained, and no AWS service is called.

`--check` verifies the two paths agree instead of writing: it transforms the
same strings through the original pickle and through the rebuilt vectorizer and
reports the largest difference, which must be exactly 0.

After a full `scripts/build_features.py` run this is unnecessary --
`src/features/content.py` writes both forms. It exists for artifacts that were
built before that change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import joblib
import numpy as np

from src.data.config import load_config, output_dir
from src.features.text_vectorizer import (
    PORTABLE_FILES,
    load_portable_vectorizer,
    save_portable_vectorizer,
)

# Chosen to exercise the paths onboarding actually takes: genre text, an empty
# string, and text whose terms are all outside the vocabulary.
SAMPLES = (
    "action adventure thriller",
    "romance drama comedy family",
    "science fiction space war alien",
    "",
    "zzzz khong co tu nao trong tu dien",
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the two implementations without writing anything.",
    )
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def compare(original, rebuilt) -> float:
    """Largest absolute difference between the two transforms, over SAMPLES."""
    reference = original.transform(SAMPLES).tocsr()
    candidate = rebuilt.transform(SAMPLES).tocsr()
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Kích thước khác nhau: {reference.shape} và {candidate.shape}"
        )
    difference = abs(reference - candidate)
    return float(difference.max()) if difference.nnz else 0.0


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    config = load_config(REPOSITORY_ROOT / arguments.config)
    artifacts_dir = output_dir(config, "artifacts_dir", create=False)

    pickle_path = artifacts_dir / "vectorizer.joblib"
    if not pickle_path.is_file():
        print(f"Không tìm thấy {pickle_path}.", file=sys.stderr)
        return 2

    original = joblib.load(pickle_path)
    print(f"Đã đọc {pickle_path}")
    print(f"  từ vựng : {len(original.vocabulary_):,}")
    print(f"  idf     : {np.asarray(original.idf_).shape[0]:,}")
    print(f"  dtype   : {np.dtype(original.dtype).name}")

    if arguments.check:
        rebuilt = load_portable_vectorizer(artifacts_dir)
        worst = compare(original, rebuilt)
        print(f"\nSai lệch lớn nhất: {worst:.3e}")
        if worst != 0.0:
            print("KHÁC NHAU — không dùng được.", file=sys.stderr)
            return 1
        print("Giống hệt từng bit.")
        return 0

    written = save_portable_vectorizer(original, artifacts_dir)
    print(f"\nĐã ghi vào {artifacts_dir}:")
    for name in PORTABLE_FILES:
        size = (artifacts_dir / name).stat().st_size
        print(f"  {name}  ({size / 1e6:.2f} MB)")

    rebuilt = load_portable_vectorizer(artifacts_dir)
    worst = compare(original, rebuilt)
    print(f"\nĐối chiếu với bản gốc: sai lệch lớn nhất {worst:.3e}")
    if worst != 0.0:
        print(
            "KHÁC NHAU. Không dùng bundle này; báo lại trước khi deploy.",
            file=sys.stderr,
        )
        return 1
    print(f"Giống hệt từng bit ({written['terms']:,} từ, dtype {written['dtype']}).")
    print("\nChạy tiếp: python scripts/build_model_bundle.py --upload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
