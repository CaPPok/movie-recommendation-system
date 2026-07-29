"""Package the recommendation engine's artifacts into a SageMaker model bundle.

Usage:
    python scripts/build_model_bundle.py --dry-run
    python scripts/build_model_bundle.py
    python scripts/build_model_bundle.py --upload
    python scripts/build_model_bundle.py --als-version v1.0.0 --upload

SageMaker unpacks `model.tar.gz` into `/opt/ml/model` before the first request.
`deploy/recommendation_handler.py` then loads the configs from
`<model_dir>/configs/`, which makes the project root resolve inside that
directory, so the archive has to mirror the repository layout:

    configs/                        data_pipeline.yaml, model_serving.yaml
    artifacts/LATEST.json
    artifacts/content_based/        TF-IDF matrix, vectorizer, index
    artifacts/collaborative/<ver>/  ALS factor matrices
    data/serving/                   catalogue and guest rankings
    data/features/                  content feature table
    data/processed/                 movie metadata

Only the files the engine opens at start-up are staged. `data/splits`,
`data/raw` and the second ALS version are ~700 MB between them, none of it read
at serving time, and every megabyte here is downloaded again on every endpoint
deployment.

Deliberately not `aws_sync.py`: that moves whole directories between the local
tree and S3 for training, and a serving bundle is a curated subset of several
directories collapsed into one archive.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config

BUNDLE_NAME = "model.tar.gz"

# Opened by RecommendationEngine.__init__ and its four recommenders. A missing
# entry here is a start-up crash inside the container, several minutes after
# deployment, so the script checks all of them before building anything.
REQUIRED_FILES = (
    "configs/data_pipeline.yaml",
    "configs/model_serving.yaml",
    "artifacts/LATEST.json",
    # The portable vectorizer, not vectorizer.joblib. The pickle fails inside
    # the inference image's older scikit-learn -- silently, at the first
    # onboarding request rather than at load. See src/features/text_vectorizer.py.
    "artifacts/content_based/vectorizer_vocabulary.json",
    "artifacts/content_based/vectorizer_idf.npy",
    "artifacts/content_based/vectorizer_params.json",
    "artifacts/content_based/movie_matrix.npz",
    "artifacts/content_based/movie_index.parquet",
    "data/serving/movies_serving.parquet",
    "data/serving/top_rated_all.parquet",
    "data/serving/top_rated_by_genre.parquet",
    "data/features/movie_content_features.parquet",
    "data/processed/movies_clean.parquet",
)

# "Because you watched" degrades to unavailable instead of failing the engine,
# so a bundle without this file still serves; it just answers less.
OPTIONAL_FILES = ("artifacts/content_based/similar_movies_top50.parquet",)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    parser.add_argument("--aws-config", default="configs/aws.yaml")
    parser.add_argument(
        "--als-version",
        default=None,
        help="ALS artifact to package; omit to follow artifacts/LATEST.json.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"Local output path; defaults to dist/{BUNDLE_NAME}.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to the configured S3 models prefix after building.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be packaged and exit without writing.",
    )
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def resolve_als_version(root: Path, requested: str | None) -> str:
    """Pick the ALS version to package, the same way the engine picks one."""
    if requested:
        return requested
    pointer = root / "artifacts" / "LATEST.json"
    if not pointer.exists():
        raise FileNotFoundError(
            "Thiếu artifacts/LATEST.json. Chạy train.py trước, hoặc truyền "
            "--als-version."
        )
    version = json.loads(pointer.read_text(encoding="utf-8")).get("collaborative")
    if not version:
        raise ValueError("artifacts/LATEST.json không có khóa 'collaborative'.")
    return str(version)


def collect_members(root: Path, als_version: str) -> list[tuple[Path, str]]:
    """Pair every source file with the path it takes inside the archive."""
    members: list[tuple[Path, str]] = []
    missing: list[str] = []

    for relative in REQUIRED_FILES:
        source = root / relative
        if source.is_file():
            members.append((source, relative))
        else:
            missing.append(relative)

    for relative in OPTIONAL_FILES:
        source = root / relative
        if source.is_file():
            members.append((source, relative))

    als_directory = root / "artifacts" / "collaborative" / als_version
    if not als_directory.is_dir():
        missing.append(f"artifacts/collaborative/{als_version}/")
    else:
        for source in sorted(als_directory.rglob("*")):
            if source.is_file():
                members.append((source, source.relative_to(root).as_posix()))

    if missing:
        raise FileNotFoundError(
            "Thiếu file bắt buộc cho bundle:\n  " + "\n  ".join(missing)
        )
    return members


def build_archive(members: list[tuple[Path, str]], destination: Path) -> None:
    """Write the archive through a temporary file so a crash leaves no stub."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="movie-rec-bundle-"))
    temporary = staging / BUNDLE_NAME
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for source, arcname in members:
                archive.add(source, arcname=arcname)
        shutil.move(str(temporary), str(destination))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)

    als_version = resolve_als_version(REPOSITORY_ROOT, arguments.als_version)
    members = collect_members(REPOSITORY_ROOT, als_version)
    raw_bytes = sum(source.stat().st_size for source, _ in members)
    destination = Path(
        arguments.out or REPOSITORY_ROOT / "dist" / BUNDLE_NAME
    )

    print(f"ALS version : {als_version}")
    print(f"Số file     : {len(members)}")
    print(f"Chưa nén    : {raw_bytes / 1e6:.1f} MB")
    print(f"Đích         : {destination}")

    if arguments.dry_run:
        for _, arcname in members:
            print(f"  {arcname}")
        print("\nDry run: chưa tạo file. Bỏ --dry-run để đóng gói thật.", file=sys.stderr)
        return 0

    print("\nĐang nén...", flush=True)
    build_archive(members, destination)
    packed = destination.stat().st_size
    print(f"  {destination}  ({packed / 1e6:.1f} MB sau khi nén)")

    if not arguments.upload:
        print(
            "\nChạy tiếp: python scripts/build_model_bundle.py --upload "
            "để đẩy lên S3."
        )
        return 0

    from src.aws import s3_sync

    aws_config = load_config(REPOSITORY_ROOT / arguments.aws_config)
    location = s3_sync.location_for(aws_config, "models").child(als_version)
    s3 = s3_sync.client(aws_config)
    print(f"\nĐang tải lên {location.uri}...", flush=True)
    uri = s3_sync.upload_file(s3, destination, location, name=BUNDLE_NAME)
    print(f"  {uri}")
    print(
        "\nDùng URI này cho bước tạo SageMaker Model "
        "(ModelDataUrl / model_data)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
