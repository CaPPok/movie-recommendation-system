"""The staged source bundle must carry code and nothing else.

Regression cover for a submission that tarred the repository root: `source_dir`
is uploaded on every run and the SDK honours no ignore file, so `data/` and the
local virtualenv went with it — about 2.2 GB per job, none of it needed, since
the job pulls its inputs from S3 itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.sagemaker_retrain_job import (
    CONTAINER_REQUIREMENTS,
    SOURCE_INCLUDES,
    build_source_bundle,
)


def _fake_repository(root: Path) -> None:
    """A repository root holding both what belongs in the bundle and what does not."""
    for name in SOURCE_INCLUDES:
        target = root / name
        if name.endswith(".py"):
            target.write_text("print('code')\n", encoding="utf-8")
        else:
            target.mkdir()
            (target / "module.py").write_text("value = 1\n", encoding="utf-8")

    (root / CONTAINER_REQUIREMENTS).write_text("pandas>=2.0\n", encoding="utf-8")
    (root / "requirements.txt").write_text("pandas==2.3.3\n", encoding="utf-8")
    (root / "requirements-aws.txt").write_text("boto3>=1.34\n", encoding="utf-8")

    # The bulk that must never reach the tarball.
    (root / "data" / "movies_dataset_raw").mkdir(parents=True)
    (root / "data" / "movies_dataset_raw" / "ratings.csv").write_text(
        "userId,movieId,rating\n1,110,1.0\n", encoding="utf-8"
    )
    (root / "artifacts" / "collaborative").mkdir(parents=True)
    (root / "artifacts" / "collaborative" / "user_factors.npy").write_bytes(b"\x00" * 64)
    (root / ".venv" / "Lib").mkdir(parents=True)
    (root / ".venv" / "Lib" / "site.py").write_text("x = 1\n", encoding="utf-8")
    (root / "reports").mkdir()
    (root / "tests").mkdir()

    # Caches sit inside directories that are otherwise included.
    (root / "src" / "__pycache__").mkdir()
    (root / "src" / "__pycache__" / "module.cpython-311.pyc").write_bytes(b"\x00" * 32)


def test_bundle_carries_code_and_omits_data(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _fake_repository(root)
    destination = tmp_path / "bundle"

    summary = build_source_bundle(root, destination)

    for name in SOURCE_INCLUDES:
        assert (destination / name).exists(), f"{name} missing from bundle"

    for excluded in ("data", "artifacts", ".venv", "reports", "tests"):
        assert not (destination / excluded).exists(), f"{excluded} leaked into bundle"

    assert not (destination / "src" / "__pycache__").exists()
    assert summary["files"] > 0
    # Anything near the dataset's size means an exclusion stopped working.
    assert summary["bytes"] < 5_000_000


def test_container_requirements_replace_the_pinned_file(tmp_path: Path) -> None:
    """The container installs `requirements.txt`, so the loose file must land there.

    Shipping the pinned local file instead would make the job depend on the base
    image's interpreter being new enough for those exact versions.
    """
    root = tmp_path / "repository"
    root.mkdir()
    _fake_repository(root)
    destination = tmp_path / "bundle"

    build_source_bundle(root, destination)

    installed = (destination / "requirements.txt").read_text(encoding="utf-8")
    assert installed == "pandas>=2.0\n"
    assert (destination / "requirements-aws.txt").exists()


def test_missing_container_requirements_is_an_error(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    _fake_repository(root)
    (root / CONTAINER_REQUIREMENTS).unlink()

    with pytest.raises(FileNotFoundError, match=CONTAINER_REQUIREMENTS):
        build_source_bundle(root, tmp_path / "bundle")


def test_missing_source_entry_is_an_error(tmp_path: Path) -> None:
    """A rename that silently drops a package from the job is worth failing on."""
    root = tmp_path / "repository"
    root.mkdir()
    _fake_repository(root)
    for path in (root / "src").rglob("*"):
        if path.is_file():
            path.unlink()
    (root / "src" / "__pycache__").rmdir()
    (root / "src").rmdir()

    with pytest.raises(FileNotFoundError, match="src"):
        build_source_bundle(root, tmp_path / "bundle")
