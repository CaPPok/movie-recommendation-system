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

import os
import subprocess
import sys
from pathlib import Path

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


def main() -> int:
    log_environment()
    install_aws_extras()
    os.chdir(CODE_DIR)

    import retrain

    arguments = sys.argv[1:]
    print(f"Chạy retrain.py {' '.join(arguments)}", flush=True)
    print(f"Thư mục làm việc: {CODE_DIR}", flush=True)
    return retrain.main(arguments)


if __name__ == "__main__":
    sys.exit(main())
