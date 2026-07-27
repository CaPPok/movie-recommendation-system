"""Move data and artifacts between the local tree and S3.

Three jobs share this module: the initial upload of the processed dataset, the
retraining job pulling that dataset back down inside SageMaker or EC2, and the
same job pushing the new artifact up afterwards.

boto3 is imported lazily. The repository has to keep running end to end with no
AWS account and no boto3 installed, so a missing dependency must fail only when
someone actually calls an AWS function, and with a message that says what to
install.

Environment overrides win over the YAML file, because the same file has to serve
a laptop, a Processing Job and an EC2 instance.
"""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

ENVIRONMENT_OVERRIDES = {
    "region": ("AWS_REGION", "AWS_DEFAULT_REGION"),
    "bucket": ("MOVIE_REC_BUCKET",),
}


class AwsDependencyError(RuntimeError):
    """Raised when an AWS call is attempted without boto3 installed."""


def _boto3() -> Any:
    try:
        import boto3
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise AwsDependencyError(
            "boto3 chưa được cài. Chạy: pip install -r requirements-aws.txt"
        ) from error
    return boto3


@dataclass(frozen=True)
class S3Location:
    """A bucket plus a key prefix, the unit every sync function works on."""

    bucket: str
    prefix: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}"

    def child(self, suffix: str) -> S3Location:
        base = self.prefix.rstrip("/")
        tail = suffix.strip("/")
        joined = f"{base}/{tail}" if base else tail
        return S3Location(self.bucket, f"{joined}/" if tail else f"{base}/")


def parse_s3_uri(uri: str) -> S3Location:
    """Split `s3://bucket/prefix` into its parts."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Không phải S3 URI: {uri!r}")
    remainder = uri[len("s3://") :]
    bucket, _, prefix = remainder.partition("/")
    if not bucket:
        raise ValueError(f"S3 URI thiếu tên bucket: {uri!r}")
    return S3Location(bucket, prefix)


def resolve_setting(aws_config: Mapping[str, Any], key: str) -> str:
    """Read one AWS setting, letting the environment override the YAML file."""
    for variable in ENVIRONMENT_OVERRIDES.get(key, ()):
        value = os.environ.get(variable)
        if value:
            return value
    value = str(aws_config["aws"].get(key) or "")
    if not value:
        raise ValueError(
            f"Thiếu cấu hình aws.{key}. Đặt trong configs/aws.yaml hoặc biến môi "
            f"trường {ENVIRONMENT_OVERRIDES.get(key, ('',))[0]}."
        )
    return value


def location_for(aws_config: Mapping[str, Any], prefix_key: str) -> S3Location:
    """S3 location of one configured prefix, e.g. `splits` or `artifacts`."""
    prefixes = aws_config["aws"]["prefixes"]
    if prefix_key not in prefixes:
        raise KeyError(
            f"Prefix {prefix_key!r} không có trong configs/aws.yaml; "
            f"đang có: {sorted(prefixes)}"
        )
    bucket = resolve_setting(aws_config, "bucket")
    return S3Location(bucket, str(prefixes[prefix_key]))


def client(aws_config: Mapping[str, Any], service: str = "s3") -> Any:
    """A boto3 client bound to the configured region."""
    return _boto3().client(service, region_name=resolve_setting(aws_config, "region"))


def _iter_local_files(root: Path, exclude: tuple[str, ...]) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in exclude):
            continue
        yield path


def upload_directory(
    s3: Any,
    local_dir: Path,
    location: S3Location,
    exclude: tuple[str, ...] = (),
    overwrite: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Upload a local tree, preserving relative paths under the prefix."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        return {"local": str(local_dir), "skipped": "directory does not exist"}

    uploaded: list[str] = []
    skipped: list[str] = []
    total_bytes = 0
    for path in _iter_local_files(local_dir, exclude):
        relative = path.relative_to(local_dir).as_posix()
        key = f"{location.prefix.rstrip('/')}/{relative}"
        if not overwrite and object_exists(s3, location.bucket, key):
            skipped.append(relative)
            continue
        if not dry_run:
            s3.upload_file(str(path), location.bucket, key)
        uploaded.append(relative)
        total_bytes += path.stat().st_size
    return {
        "local": str(local_dir),
        "destination": location.uri,
        "uploaded": len(uploaded),
        "skipped": len(skipped),
        "bytes": total_bytes,
        "files": uploaded,
        "dry_run": dry_run,
    }


def download_directory(
    s3: Any,
    location: S3Location,
    local_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Download everything under a prefix into a local tree."""
    local_dir = Path(local_dir)
    downloaded: list[str] = []
    total_bytes = 0
    for key, size in list_objects(s3, location):
        relative = key[len(location.prefix) :].lstrip("/")
        if not relative:
            continue
        target = local_dir / relative
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(location.bucket, key, str(target))
        downloaded.append(relative)
        total_bytes += size
    return {
        "source": location.uri,
        "local": str(local_dir),
        "downloaded": len(downloaded),
        "bytes": total_bytes,
        "dry_run": dry_run,
    }


def list_objects(s3: Any, location: S3Location) -> Iterator[tuple[str, int]]:
    """Yield `(key, size)` for every object under a prefix, handling paging."""
    token: str | None = None
    while True:
        arguments: dict[str, Any] = {
            "Bucket": location.bucket,
            "Prefix": location.prefix,
        }
        if token:
            arguments["ContinuationToken"] = token
        response = s3.list_objects_v2(**arguments)
        for item in response.get("Contents", []):
            if item["Key"].endswith("/"):
                continue
            yield item["Key"], int(item.get("Size", 0))
        if not response.get("IsTruncated"):
            return
        token = response.get("NextContinuationToken")


def object_exists(s3: Any, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def upload_file(s3: Any, path: Path, location: S3Location, name: str | None = None) -> str:
    """Upload one file and return the resulting S3 URI."""
    key = f"{location.prefix.rstrip('/')}/{name or Path(path).name}"
    s3.upload_file(str(path), location.bucket, key)
    return f"s3://{location.bucket}/{key}"


def read_jsonl(s3: Any, location: S3Location) -> Iterator[dict[str, Any]]:
    """Stream every JSON object from every `.jsonl` file under a prefix.

    Malformed lines are skipped rather than raised: an export written by another
    process may be truncated, and one bad line must not throw away a whole
    retraining run's worth of feedback. The caller counts what it received and
    reports the gap.
    """
    for key, _ in list_objects(s3, location):
        if not key.endswith((".jsonl", ".json")):
            continue
        body = s3.get_object(Bucket=location.bucket, Key=key)["Body"].read()
        for record in _parse_json_bytes(body):
            yield record


def read_local_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Same as `read_jsonl`, for a local file or directory."""
    path = Path(path)
    files = (
        sorted(p for p in path.rglob("*") if p.suffix in {".jsonl", ".json"})
        if path.is_dir()
        else [path]
    )
    for item in files:
        for record in _parse_json_bytes(item.read_bytes()):
            yield record


def _parse_json_bytes(body: bytes) -> Iterator[dict[str, Any]]:
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return
    if text.startswith("["):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return
        for record in payload:
            if isinstance(record, dict):
                yield record
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record
