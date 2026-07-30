"""Create, inspect or delete the SageMaker real-time recommendation endpoint.

Usage:
    python scripts/deploy_endpoint.py --dry-run
    python scripts/deploy_endpoint.py
    python scripts/deploy_endpoint.py --status
    python scripts/deploy_endpoint.py --delete

This is the one script in the repository that creates a resource billed by the
hour for as long as it exists, independently of traffic. Three consequences are
built into it:

* `--dry-run` prints the full plan and calls no AWS API, so the configuration
  can be checked before anything starts costing money;
* `--delete` removes the endpoint, its config and the model together, because
  deleting only the endpoint leaves the other two behind and a later deploy
  then fails on a name collision;
* `--status` exists so checking whether something is still running never
  requires opening the console.

The bundle itself is built by `scripts/build_model_bundle.py`; this script only
points an endpoint at one that is already on S3.
"""

from __future__ import annotations

import argparse
import json
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

HANDLER_SOURCE = "deploy/recommendation_handler.py"
# Staged at the root of source_dir rather than under deploy/. The inference
# toolkit imports the entry point as a top-level module, and a flat name keeps
# `from src...` inside it resolving against the same directory.
#
# The name matters. `sagemaker_inference.py` -- the obvious choice -- shadows the
# real `sagemaker_inference` package the serving stack imports from, and
# /opt/ml/code precedes site-packages on sys.path. The worker died at load with
# `No module named 'sagemaker_inference.default_handler_service'; is not a
# package`, while /ping kept answering 200 long enough for the endpoint to reach
# InService and only then start failing.
HANDLER_NAME = "recommendation_handler.py"
ENDPOINT_REQUIREMENTS = "requirements-endpoint.txt"
EXCLUDED_DIRECTORIES = frozenset({"__pycache__", ".pytest_cache", ".git"})


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws-config", default="configs/aws.yaml")
    parser.add_argument(
        "--model-version",
        default=None,
        help="Bundle version on S3; omit to follow artifacts/LATEST.json.",
    )
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument("--instance-type", default=None)
    parser.add_argument("--role-arn", default=None)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report whether the endpoint exists and what it is doing.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the endpoint, its endpoint-config and its model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without calling AWS.",
    )
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def resolve_model_version(
    root: Path, aws_config: dict[str, Any], requested: str | None
) -> str:
    """Decide which artifact version the endpoint should serve.

    S3 first, local file second. Retraining runs in the cloud and moves
    LATEST.json there, so the copy on a laptop is stale as soon as a job
    promotes anything -- deploying from it would quietly roll the endpoint back
    to an older model.
    """
    if requested:
        return requested

    from src.aws import s3_sync

    location = s3_sync.location_for(aws_config, "artifacts")
    key = f"{location.prefix.rstrip('/')}/LATEST.json"
    try:
        s3 = s3_sync.client(aws_config)
        body = s3.get_object(Bucket=location.bucket, Key=key)["Body"].read()
        version = json.loads(body.decode("utf-8")).get("collaborative")
        if version:
            print(f"LATEST.json trên S3 trỏ tới {version}.")
            return str(version)
    except Exception as error:  # noqa: BLE001 - any S3 failure falls back to local
        print(f"Không đọc được s3://{location.bucket}/{key} ({error}); dùng bản local.")

    pointer = root / "artifacts" / "LATEST.json"
    if not pointer.exists():
        raise FileNotFoundError(
            "Không có LATEST.json trên S3 lẫn ở local; truyền --model-version."
        )
    version = json.loads(pointer.read_text(encoding="utf-8")).get("collaborative")
    if not version:
        raise ValueError("artifacts/LATEST.json không có khóa 'collaborative'.")
    return str(version)


def resolve_role(aws_config: dict[str, Any], override: str | None) -> str:
    import os

    role = (
        override
        or os.environ.get("MOVIE_REC_SAGEMAKER_ROLE")
        or str(aws_config["sagemaker"].get("role_arn") or "")
    )
    if not role:
        raise ValueError(
            "Chưa có SageMaker execution role. Đặt sagemaker.role_arn trong "
            "configs/aws.yaml, hoặc biến môi trường MOVIE_REC_SAGEMAKER_ROLE, "
            "hoặc truyền --role-arn."
        )
    return role


def build_source_bundle(root: Path, destination: Path) -> dict[str, Any]:
    """Stage the handler plus `src/`, and nothing else.

    source_dir is uploaded on every deployment. Artifacts and data travel in
    model.tar.gz instead, so shipping them here as well would double a 158 MB
    transfer for no benefit.
    """

    def ignore(directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in EXCLUDED_DIRECTORIES}

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root / "src", destination / "src", ignore=ignore)
    shutil.copy2(root / HANDLER_SOURCE, destination / HANDLER_NAME)
    shutil.copy2(root / ENDPOINT_REQUIREMENTS, destination / "requirements.txt")

    staged = [path for path in destination.rglob("*") if path.is_file()]
    return {
        "files": len(staged),
        "bytes": sum(path.stat().st_size for path in staged),
    }


def _sagemaker_client(aws_config: dict[str, Any]) -> Any:
    from src.aws import s3_sync

    return s3_sync.client(aws_config, "sagemaker")


def report_status(aws_config: dict[str, Any], endpoint_name: str) -> int:
    client = _sagemaker_client(aws_config)
    try:
        described = client.describe_endpoint(EndpointName=endpoint_name)
    except client.exceptions.ClientError:
        print(f"Endpoint {endpoint_name!r}: không tồn tại (không tính tiền).")
        return 0
    print(f"Endpoint      : {endpoint_name}")
    print(f"Trạng thái    : {described['EndpointStatus']}")
    print(f"Tạo lúc       : {described['CreationTime']}")
    for variant in described.get("ProductionVariants", []):
        print(
            f"  variant {variant['VariantName']}: "
            f"{variant.get('CurrentInstanceCount')} x "
            f"{variant.get('CurrentInstanceType', '?')}"
        )
    if described["EndpointStatus"] == "InService":
        print("\nĐang tính tiền theo giờ. Xoá bằng: "
              "python scripts/deploy_endpoint.py --delete")
    return 0


def delete_endpoint(aws_config: dict[str, Any], endpoint_name: str) -> int:
    """Remove all three resources; a leftover config or model blocks redeploy."""
    client = _sagemaker_client(aws_config)
    model_name = None
    config_name = None

    try:
        described = client.describe_endpoint(EndpointName=endpoint_name)
        config_name = described.get("EndpointConfigName")
    except client.exceptions.ClientError:
        print(f"Endpoint {endpoint_name!r} không tồn tại; kiểm tra phần còn lại.")

    if config_name:
        try:
            config = client.describe_endpoint_config(EndpointConfigName=config_name)
            variants = config.get("ProductionVariants", [])
            model_name = variants[0]["ModelName"] if variants else None
        except client.exceptions.ClientError:
            pass

    for label, call in (
        ("endpoint", lambda: client.delete_endpoint(EndpointName=endpoint_name)),
        (
            "endpoint-config",
            lambda: client.delete_endpoint_config(EndpointConfigName=config_name),
        ),
        ("model", lambda: client.delete_model(ModelName=model_name)),
    ):
        target = {"endpoint": endpoint_name, "endpoint-config": config_name, "model": model_name}[label]
        if not target:
            continue
        try:
            call()
            print(f"  đã xoá {label}: {target}")
        except client.exceptions.ClientError as error:
            print(f"  bỏ qua {label} {target}: {error}")

    print("\nKiểm tra lại: aws sagemaker list-endpoints --region "
          f"{aws_config['aws']['region']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    aws_config = load_config(REPOSITORY_ROOT / arguments.aws_config)
    settings = aws_config["sagemaker"]
    endpoint_settings = settings["endpoint"]

    from src.aws import s3_sync

    region = s3_sync.resolve_setting(aws_config, "region")
    bucket = s3_sync.resolve_setting(aws_config, "bucket")
    endpoint_name = arguments.endpoint_name or str(endpoint_settings["name"])

    if arguments.status:
        return report_status(aws_config, endpoint_name)
    if arguments.delete:
        return delete_endpoint(aws_config, endpoint_name)

    model_version = resolve_model_version(
        REPOSITORY_ROOT, aws_config, arguments.model_version
    )
    instance_type = arguments.instance_type or str(endpoint_settings["instance_type"])
    version_prefix = f"{aws_config['aws']['prefixes']['models']}{model_version}/"
    model_data = f"s3://{bucket}/{version_prefix}model.tar.gz"
    # Where the SDK puts `sourcedir.tar.gz`. Without this it derives its own
    # location from the model name and writes to the bucket root, leaving a
    # `movie-rec-model-<version>-<timestamp>/` directory behind on every deploy.
    # Pointing it at the version directory keeps the bundle and the code that
    # serves it in one place, which is also what a rollback needs.
    code_location = f"s3://{bucket}/{version_prefix}"
    model_name = f"movie-rec-model-{model_version.replace('.', '-')}-" + (
        f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    )

    bundle_root = Path(tempfile.mkdtemp(prefix="movie-rec-endpoint-src-"))
    try:
        bundle = build_source_bundle(REPOSITORY_ROOT, bundle_root)
        plan = {
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "region": region,
            "model_data": model_data,
            "code_location": code_location,
            "instance_type": instance_type,
            "instance_count": int(endpoint_settings["instance_count"]),
            "framework": (
                f"{endpoint_settings['framework']} "
                f"{endpoint_settings['framework_version']} "
                f"{endpoint_settings['python_version']}"
            ),
            "entry_point": HANDLER_NAME,
            "source_bundle": {
                "files": bundle["files"],
                "megabytes": round(bundle["bytes"] / 1e6, 2),
            },
        }

        if arguments.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            print(
                "\nDry run: chưa gọi AWS, chưa tốn tiền. Bỏ --dry-run để tạo thật.",
                file=sys.stderr,
            )
            return 0

        role = resolve_role(aws_config, arguments.role_arn)

        try:
            import sagemaker
            from sagemaker.pytorch.model import PyTorchModel
        except ImportError as error:
            print(
                "Thiếu SDK sagemaker. Chạy: pip install -r requirements-aws.txt",
                file=sys.stderr,
            )
            raise SystemExit(2) from error

        session = sagemaker.Session(default_bucket=bucket)
        # PyTorchModel for a model with no torch in it: see the `framework`
        # comment in configs/aws.yaml. Both framework classes drive the same
        # model_fn / input_fn / predict_fn / output_fn contract, so the handler
        # is unchanged; only the base image differs.
        model = PyTorchModel(
            model_data=model_data,
            role=role,
            entry_point=HANDLER_NAME,
            source_dir=str(bundle_root),
            code_location=code_location,
            framework_version=str(endpoint_settings["framework_version"]),
            py_version=str(endpoint_settings["python_version"]),
            name=model_name,
            sagemaker_session=session,
        )

        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        print(
            f"\nĐang tạo endpoint {endpoint_name} ({instance_type}). "
            "Mất khoảng 5-10 phút, và bắt đầu tính tiền khi InService.",
            flush=True,
        )
        model.deploy(
            initial_instance_count=int(endpoint_settings["instance_count"]),
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            # The default startup budget assumes a container that is ready as
            # soon as it boots. This one first downloads a 141 MB bundle, then
            # pip-installs pandas, pyarrow, scipy and scikit-learn over the base
            # image, then spends ~4 s loading artifacts into memory. Measured
            # locally the load is fast; the installs are not, and a health check
            # that fires during them fails an endpoint that would have worked.
            model_data_download_timeout=900,
            container_startup_health_check_timeout=900,
        )
    finally:
        shutil.rmtree(bundle_root, ignore_errors=True)

    print(f"\nEndpoint {endpoint_name} đã InService.")
    print("Thử gọi   : python scripts/invoke_endpoint.py --demo")
    print("Xem trạng thái: python scripts/deploy_endpoint.py --status")
    print("XOÁ khi xong : python scripts/deploy_endpoint.py --delete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
