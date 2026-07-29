"""Submit the retraining run as a SageMaker Processing Job.

Usage:
    python scripts/sagemaker_retrain_job.py --dry-run
    python scripts/sagemaker_retrain_job.py --version v1.1.0 --events s3://.../events/
    python scripts/sagemaker_retrain_job.py --wait

A **Processing Job**, not a Training Job and not an endpoint. Three reasons:

* the run is not only a fit — it ingests events, rebuilds splits, evaluates and
  decides on promotion, and a Processing Job is the shape that fits a script;
* the artifact this project serves is a pair of `.npy` factor matrices read
  straight from S3, not a SageMaker Model, so Training Job's model packaging
  buys nothing;
* an always-on real-time endpoint is the one line item that could exceed the
  budget (`MODEL_DESIGN_SPEC.md` sections 2 and 4.4), and nothing here creates
  one.

The job pulls its own inputs from S3 and pushes its own outputs back, rather
than using ProcessingInput/ProcessingOutput channels. The dataset lives in a
nested tree whose layout the code already knows, and round-tripping it through
channel directories would mean two path conventions for the same files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config

ENTRYPOINT = "deploy/sagemaker_retrain.py"

# Everything the job needs and nothing else. `data/`, `artifacts/`, `reports/`
# and the local virtualenv are all absent on purpose: see build_source_bundle.
SOURCE_INCLUDES = (
    "configs",
    "deploy",
    "scripts",
    "src",
    "evaluate.py",
    "inference.py",
    "retrain.py",
    "train.py",
)
CONTAINER_REQUIREMENTS = "requirements-container.txt"
EXCLUDED_DIRECTORIES = frozenset({"__pycache__", ".pytest_cache", ".git"})


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws-config", default="configs/aws.yaml")
    parser.add_argument("--version", default=None, help="Artifact version to produce.")
    parser.add_argument(
        "--events",
        default=None,
        help="S3 prefix of exported events; omit to retrain on existing data.",
    )
    parser.add_argument("--instance-type", default=None)
    parser.add_argument("--role-arn", default=None)
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Pass --force-promote through to retrain.py.",
    )
    parser.add_argument("--sample-users", type=int, default=None)
    parser.add_argument(
        "--no-build-bundle",
        action="store_true",
        help=(
            "Skip packaging the promoted artifact for the endpoint. Only useful "
            "for a training-only experiment; without the bundle the endpoint "
            "cannot serve what LATEST.json now points at."
        ),
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block until the job finishes and stream its status.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job configuration and exit without calling AWS.",
    )
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def resolve_role(aws_config: dict[str, Any], override: str | None) -> str:
    role = (
        override
        or os.environ.get("MOVIE_REC_SAGEMAKER_ROLE")
        or str(aws_config["sagemaker"].get("role_arn") or "")
    )
    if not role:
        raise ValueError(
            "Chưa có SageMaker execution role. Đặt sagemaker.role_arn trong "
            "configs/aws.yaml, hoặc biến môi trường MOVIE_REC_SAGEMAKER_ROLE, "
            "hoặc truyền --role-arn.\n"
            "Role cần quyền: đọc/ghi bucket dữ liệu, và trust policy cho "
            "sagemaker.amazonaws.com."
        )
    return role


def build_source_bundle(root: Path, destination: Path) -> dict[str, Any]:
    """Stage only the code the job needs, and report what was staged.

    `source_dir` is tarred and uploaded on every submission, and the SDK honours
    no ignore file, so pointing it at the repository root ships the 1.7 GB
    dataset, the local virtualenv and every artifact with each run. The job
    pulls its inputs from S3 itself (`--pull`), so none of that belongs in the
    tarball; the difference is roughly 2.2 GB against a few hundred kilobytes.

    The container's `requirements.txt` comes from `requirements-container.txt`
    rather than the local pinned file. Why: see the header of that file.
    """

    def ignore(directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in EXCLUDED_DIRECTORIES}

    destination.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_INCLUDES:
        source = root / name
        if not source.exists():
            raise FileNotFoundError(
                f"Thiếu {name} trong {root}; không dựng được source bundle."
            )
        if source.is_dir():
            shutil.copytree(source, destination / name, ignore=ignore)
        else:
            shutil.copy2(source, destination / name)

    requirements = root / CONTAINER_REQUIREMENTS
    if not requirements.exists():
        raise FileNotFoundError(f"Thiếu {CONTAINER_REQUIREMENTS} trong {root}.")
    shutil.copy2(requirements, destination / "requirements.txt")
    shutil.copy2(root / "requirements-aws.txt", destination / "requirements-aws.txt")

    staged = [path for path in destination.rglob("*") if path.is_file()]
    return {
        "files": len(staged),
        "bytes": sum(path.stat().st_size for path in staged),
    }


def build_job_arguments(arguments: argparse.Namespace) -> list[str]:
    """Command-line arguments handed to the entrypoint inside the container."""
    job_arguments = ["--pull", "--push"]
    if arguments.version:
        job_arguments.extend(["--version", arguments.version])
    if arguments.events:
        job_arguments.extend(["--events", arguments.events])
    if arguments.sample_users is not None:
        job_arguments.extend(["--sample-users", str(arguments.sample_users)])
    if arguments.force_promote:
        job_arguments.append("--force-promote")
    if not arguments.no_build_bundle:
        # Default on: a promotion that does not reach the endpoint changes
        # nothing a user can see, and the job is the only place with the new
        # artifacts already on local disk.
        job_arguments.append("--build-bundle")
    return job_arguments


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    aws_config = load_config(REPOSITORY_ROOT / arguments.aws_config)
    settings = aws_config["sagemaker"]

    from src.aws import s3_sync

    region = s3_sync.resolve_setting(aws_config, "region")
    bucket = s3_sync.resolve_setting(aws_config, "bucket")
    instance_type = arguments.instance_type or str(settings["instance_type"])
    job_name = (
        f"{settings['base_job_name']}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    )
    job_arguments = build_job_arguments(arguments)

    bundle_root = Path(tempfile.mkdtemp(prefix="movie-rec-source-"))
    try:
        bundle = build_source_bundle(REPOSITORY_ROOT, bundle_root)
        plan = {
            "job_name": job_name,
            "region": region,
            "bucket": bucket,
            "instance_type": instance_type,
            "instance_count": int(settings["instance_count"]),
            "volume_size_gb": int(settings["volume_size_gb"]),
            "max_runtime_seconds": int(settings["max_runtime_seconds"]),
            "framework": f"{settings['framework']} {settings['framework_version']}",
            "entrypoint": ENTRYPOINT,
            "arguments": job_arguments,
            "source_dir": str(bundle_root),
            "source_bundle": {
                "files": bundle["files"],
                "megabytes": round(bundle["bytes"] / 1e6, 2),
                "includes": list(SOURCE_INCLUDES),
            },
        }

        if arguments.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            print(
                "\nDry run: chưa gọi AWS. Bỏ --dry-run để chạy thật.",
                file=sys.stderr,
            )
            return 0

        role = resolve_role(aws_config, arguments.role_arn)

        try:
            import sagemaker
            from sagemaker.processing import FrameworkProcessor
            from sagemaker.sklearn.estimator import SKLearn
        except ImportError as error:
            print(
                "Thiếu SDK sagemaker. Chạy: pip install -r requirements-aws.txt",
                file=sys.stderr,
            )
            raise SystemExit(2) from error

        session = sagemaker.Session(default_bucket=bucket)
        processor = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=str(settings["framework_version"]),
            role=role,
            instance_type=instance_type,
            instance_count=int(settings["instance_count"]),
            volume_size_in_gb=int(settings["volume_size_gb"]),
            max_runtime_in_seconds=int(settings["max_runtime_seconds"]),
            base_job_name=str(settings["base_job_name"]),
            sagemaker_session=session,
            env={
                # The container installs requirements.txt from source_dir, then
                # the entrypoint installs the AWS extras. Region and bucket
                # travel as environment variables so configs/aws.yaml does not
                # have to be edited per environment.
                "AWS_REGION": region,
                "MOVIE_REC_BUCKET": bucket,
            },
        )

        print(
            f"Gửi Processing Job {job_name} ({instance_type}); "
            f"source bundle {plan['source_bundle']['megabytes']} MB, "
            f"{bundle['files']} file...",
            flush=True,
        )
        processor.run(
            code=ENTRYPOINT,
            source_dir=str(bundle_root),
            arguments=job_arguments,
            job_name=job_name,
            wait=arguments.wait,
            logs=arguments.wait,
        )
    finally:
        # The tarball is uploaded during run(), so the staging copy is only
        # needed until then.
        shutil.rmtree(bundle_root, ignore_errors=True)

    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not arguments.wait:
        print(
            "\nJob đang chạy nền. Theo dõi:\n"
            f"  aws sagemaker describe-processing-job --processing-job-name {job_name} "
            f"--region {region}\n"
            f"  aws logs tail /aws/sagemaker/ProcessingJobs --follow --region {region}"
        )
    print(
        "\nKết quả sẽ nằm ở:\n"
        f"  s3://{bucket}/{aws_config['aws']['prefixes']['artifacts']}\n"
        f"  s3://{bucket}/{aws_config['aws']['prefixes']['reports']}validation/retrain_report.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
