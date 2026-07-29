"""Entrypoint executed inside the SageMaker Processing Job container.

SageMaker copies `source_dir` to `/opt/ml/processing/input/code`, installs
`requirements.txt`, then runs this file with the arguments the launcher passed.
Everything it does is a thin wrapper around `retrain.py`; the retraining logic
itself is not duplicated here, so a local run and a cloud run cannot diverge.

Two things only this layer handles:

* the AWS extras (`boto3`) are not in the project's `requirements.txt`, because
  the repository has to install and run without them; the job needs them, so
  they are installed here;
* the container's working directory is not the repository root, and every
  config path in the project is repository-relative.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Consumed here rather than forwarded: retrain.py knows nothing about serving.
BUILD_BUNDLE_FLAG = "--build-bundle"

# The code directory SageMaker unpacks source_dir into. Falls back to this
# file's parent so the script can also be run by hand on EC2 for debugging.
CODE_DIR = Path(
    os.environ.get("SAGEMAKER_CODE_DIR", str(Path(__file__).resolve().parents[1]))
)
if not (CODE_DIR / "retrain.py").exists():
    CODE_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(CODE_DIR))

# BLAS threading has to be capped before implicit is imported anywhere.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def install_aws_extras() -> None:
    """Install boto3 into the container if the base image does not have it."""
    try:
        import boto3  # noqa: F401

        return
    except ImportError:
        pass

    requirements = CODE_DIR / "requirements-aws.txt"
    command = [sys.executable, "-m", "pip", "install", "--quiet"]
    command.extend(
        ["-r", str(requirements)] if requirements.exists() else ["boto3>=1.34"]
    )
    print(f"Cài AWS extras: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def log_environment() -> None:
    """Record the interpreter the base image actually shipped.

    The container installs `requirements.txt` over whatever Python the framework
    image provides, and a version mismatch there fails the job minutes after
    submission with a pip resolution error rather than an obvious one. Printing
    it first means the log answers the question without a second run.
    """
    import platform

    print(
        f"Python {platform.python_version()} ({platform.platform()})",
        flush=True,
    )


def promoted_version() -> str | None:
    """Which artifact LATEST.json currently points at, or None if unreadable."""
    pointer = CODE_DIR / "artifacts" / "LATEST.json"
    if not pointer.exists():
        return None
    try:
        return json.loads(pointer.read_text(encoding="utf-8")).get("collaborative")
    except (json.JSONDecodeError, OSError):
        return None


def build_model_bundle(version: str) -> int:
    """Package the promoted artifact for serving and upload it.

    Done inside the job because this is the only machine that has the new
    artifacts on disk. Building it later from a laptop would mean pulling ~180 MB
    back down first, and would leave a window where LATEST.json advertises a
    version no bundle exists for.
    """
    command = [
        sys.executable,
        str(CODE_DIR / "scripts" / "build_model_bundle.py"),
        "--als-version",
        version,
        "--upload",
    ]
    print(f"Đóng gói bundle cho {version}: {' '.join(command)}", flush=True)
    return subprocess.run(command, check=False).returncode


def main() -> int:
    log_environment()
    install_aws_extras()
    os.chdir(CODE_DIR)

    import retrain

    arguments = [item for item in sys.argv[1:] if item != BUILD_BUNDLE_FLAG]
    wants_bundle = BUILD_BUNDLE_FLAG in sys.argv[1:]

    before = promoted_version()
    print(f"Chạy retrain.py {' '.join(arguments)}", flush=True)
    print(f"Thư mục làm việc: {CODE_DIR}", flush=True)
    status = retrain.main(arguments)
    if status != 0 or not wants_bundle:
        return status

    after = promoted_version()
    if after is None:
        print("Không đọc được LATEST.json; bỏ qua bước đóng gói.", flush=True)
        return status
    if after == before:
        # The promotion gate blocked this candidate. That is the gate working,
        # not a failure -- the endpoint keeps serving what it already serves, so
        # there is nothing new to package.
        print(
            f"Cổng kiểm duyệt giữ nguyên {before}; không cần bundle mới.",
            flush=True,
        )
        return status

    print(f"Đã thăng cấp {before} -> {after}.", flush=True)
    bundle_status = build_model_bundle(after)
    if bundle_status != 0:
        print(
            "Đóng gói bundle thất bại. Artifact đã lên S3 và LATEST.json đã đổi, "
            "nhưng endpoint chưa có bản mới. Chạy tay: "
            f"python scripts/build_model_bundle.py --als-version {after} --upload",
            file=sys.stderr,
            flush=True,
        )
        return bundle_status

    print(
        "\nBundle đã lên S3. Cập nhật endpoint bằng:\n"
        f"  python scripts/deploy_endpoint.py --model-version {after}",
        flush=True,
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
