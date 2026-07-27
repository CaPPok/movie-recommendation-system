"""Retrain on production feedback and promote the result only if it earns it.

One command, six steps, the same on a laptop, inside a SageMaker Processing Job
and on an EC2 instance:

    1. pull the dataset and current artifacts from S3      (--pull)
    2. fold new interaction events into the interaction table
    3. rebuild the chronological train/validation/test splits
    4. train a new ALS artifact under a fresh version
    5. evaluate it against the popularity baseline and the model in production
    6. promote it in LATEST.json, or keep the old pointer   (--push)

Usage:
    python retrain.py --version v1.1.0 --events data/events/
    python retrain.py --version v1.1.0 --events s3://bucket/events/2026-07-27/
    python retrain.py --version v1.1.0 --pull --push
    python retrain.py --version v1.1.0 --dry-run

Why a promotion gate exists at all: retraining is scheduled, so nobody reviews
each run. Without a gate, one bad export of feedback silently replaces a working
model and the first sign of trouble is the recommendations getting worse. The
gate makes the default outcome "keep what works".

Steps 2 and 3 are skipped when no events are supplied, which makes this script
usable as a plain scheduled retrain on the existing dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Must precede the implicit import inside train.py; see the note there.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

import evaluate as evaluate_module
import train as train_module
from src.data.config import load_config, output_dir, project_path
from src.data.feedback_ingest import (
    events_to_interactions,
    merge_into_interactions,
    restrict_to_catalogue,
)
from src.data.splitting import build_interaction_splits
from src.utils.reporting import write_json

LATEST_FILE = "LATEST.json"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/model_serving.yaml")
    parser.add_argument("--aws-config", default="configs/aws.yaml")
    parser.add_argument(
        "--version",
        default=None,
        help="Artifact version; defaults to a timestamped v0.0.0-YYYYMMDDHHMM.",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="Local file/directory or s3:// prefix of exported interaction events.",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Download dataset and artifacts from S3 before training.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload the new artifact, updated data and reports to S3 afterwards.",
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Move LATEST.json even if the promotion gate fails. Logged as such.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Train only. Implies the artifact is never promoted.",
    )
    parser.add_argument("--factors", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--regularization", type=float, default=None)
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Train on a reproducible user sample; for smoke tests, not releases.",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=None,
        help=(
            "Override the configured evaluation sample. Smoke tests only: it "
            "makes the metric incomparable with the model in production, and the "
            "promotion gate will say so."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what each step would do and stop before training.",
    )
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def default_version() -> str:
    """A timestamped version, so an unattended run never collides with itself.

    Semantic versions are chosen by a human deciding what changed
    (`MODEL_DESIGN_SPEC.md` section 13.6). A scheduled job has no such opinion,
    so it stamps the clock and leaves the semantic bump to whoever reviews it.
    """
    return f"v0.0.0-{datetime.now(timezone.utc):%Y%m%d%H%M}"


def read_latest(config: dict[str, Any]) -> dict[str, Any]:
    path = project_path(config, "artifacts") / LATEST_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_latest(config: dict[str, Any], latest: dict[str, Any]) -> Path:
    path = project_path(config, "artifacts") / LATEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(latest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def previous_metrics(config: dict[str, Any], version: str | None) -> dict[str, Any]:
    """Metrics recorded in the manifest of the artifact currently in production."""
    if not version:
        return {}
    manifest = (
        project_path(config, "artifacts/collaborative") / version / "manifest.json"
    )
    if not manifest.exists():
        return {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return payload.get("metrics") or {}


# ----------------------------------------------------------------------
# Step 1 and 6: S3
# ----------------------------------------------------------------------


def pull_from_s3(aws_config: dict[str, Any], dry_run: bool) -> list[dict[str, Any]]:
    from src.aws import s3_sync

    s3 = s3_sync.client(aws_config)
    results = []
    for pair in aws_config["sync"]["pairs"]:
        location = s3_sync.location_for(aws_config, pair["prefix"])
        results.append(
            s3_sync.download_directory(
                s3, location, REPOSITORY_ROOT / pair["local"], dry_run=dry_run
            )
        )
    return results


def push_to_s3(
    aws_config: dict[str, Any], only: tuple[str, ...] | None, dry_run: bool
) -> list[dict[str, Any]]:
    from src.aws import s3_sync

    s3 = s3_sync.client(aws_config)
    results = []
    for pair in aws_config["sync"]["pairs"]:
        if only is not None and pair["prefix"] not in only:
            continue
        location = s3_sync.location_for(aws_config, pair["prefix"])
        results.append(
            s3_sync.upload_directory(
                s3, REPOSITORY_ROOT / pair["local"], location, dry_run=dry_run
            )
        )
    return results


def load_events(source: str, aws_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Read exported events from a local path or an S3 prefix."""
    from src.aws import s3_sync

    if source.startswith("s3://"):
        s3 = s3_sync.client(aws_config)
        return list(s3_sync.read_jsonl(s3, s3_sync.parse_s3_uri(source)))
    return list(s3_sync.read_local_jsonl(Path(source)))


# ----------------------------------------------------------------------
# Step 5: the promotion gate
# ----------------------------------------------------------------------


def evaluate_promotion(
    payload: dict[str, Any],
    baseline_metrics: dict[str, Any],
    promotion_config: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether the candidate replaces the model in production.

    Three independent checks, each of which can only block:

    * enough users were scored for the number to mean anything;
    * the candidate beats the popularity baseline measured in the same run,
      because a personalised model that cannot do that is not worth serving;
    * the candidate does not fall more than the tolerated fraction below the
      model it would replace.

    The last check is a tolerance, not equality: retraining on shifted data moves
    this metric by a few percent in both directions, and demanding a strict
    improvement would block every run forever.
    """
    metric = str(promotion_config["metric"])
    models = {item["model"]: item for item in payload.get("models", [])}
    candidate = models.get("collaborative_als")
    popularity = models.get("popularity_train")

    decision: dict[str, Any] = {
        "metric": metric,
        "candidate_value": None,
        "popularity_value": None,
        "previous_value": baseline_metrics.get(metric),
        "users_scored": 0,
        "checks": {},
        "promote": False,
    }
    if candidate is None:
        decision["checks"]["candidate_scored"] = False
        decision["reason"] = "Không có kết quả cho collaborative_als trong báo cáo."
        return decision

    value = float(candidate["metrics"].get(metric, 0.0))
    decision["candidate_value"] = value
    decision["users_scored"] = int(candidate.get("users_scored", 0))

    minimum_users = int(promotion_config["minimum_users_scored"])
    decision["checks"]["enough_users"] = decision["users_scored"] >= minimum_users

    if promotion_config.get("must_beat_popularity") and popularity is not None:
        reference = float(popularity["metrics"].get(metric, 0.0))
        decision["popularity_value"] = reference
        decision["checks"]["beats_popularity"] = value > reference
    else:
        decision["checks"]["beats_popularity"] = True

    previous = decision["previous_value"]
    if previous:
        tolerance = float(promotion_config["max_relative_regression"])
        floor = float(previous) * (1.0 - tolerance)
        decision["regression_floor"] = round(floor, 6)
        decision["checks"]["no_regression"] = value >= floor

        # Two HitRate figures are only comparable when they were measured the
        # same way. The evaluation protocol lives in configs/aws.yaml precisely
        # so it stays fixed across runs, but --sample-users can override it, and
        # a metric from 1,500 users against one from 5,000 differs by more than
        # the 5% tolerance through sampling noise alone. A mismatch does not
        # unblock the gate — it says the verdict is not evidence either way.
        previous_users = baseline_metrics.get("users_scored")
        if previous_users:
            ratio = decision["users_scored"] / float(previous_users)
            if not 0.8 <= ratio <= 1.25:
                decision["protocol_mismatch"] = {
                    "previous_users_scored": int(previous_users),
                    "candidate_users_scored": decision["users_scored"],
                    "note": (
                        "Hai chỉ số đo trên số user khác nhau nên không so sánh "
                        "được. Chạy lại với đúng retraining.evaluation.sample_users "
                        "trong configs/aws.yaml, đừng truyền --sample-users."
                    ),
                }
    else:
        # Nothing to regress against on the first promotion.
        decision["checks"]["no_regression"] = True

    failed = [name for name, passed in decision["checks"].items() if not passed]
    decision["promote"] = not failed
    decision["failed_checks"] = failed
    decision["reason"] = (
        "Đạt toàn bộ điều kiện."
        if not failed
        else "Không đạt: " + ", ".join(failed)
    )
    return decision


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def report_markdown(report: dict[str, Any]) -> str:
    decision = report.get("promotion") or {}
    lines = [
        "# Retrain run",
        "",
        f"Phiên bản ứng viên: **{report['version']}** · "
        f"Bắt đầu: {report['started_at']} · "
        f"Thời lượng: {report['duration_seconds']:.0f} giây",
        "",
        f"Phiên bản đang phục vụ trước khi chạy: `{report['previous_version']}`",
        f"Phiên bản đang phục vụ sau khi chạy: `{report['promoted_version']}`",
        "",
        "## Các bước",
        "",
        "| Bước | Kết quả |",
        "|---|---|",
    ]
    for name, detail in report["steps"].items():
        lines.append(f"| {name} | {detail} |")

    ingest = report.get("ingest")
    if ingest:
        lines.extend(
            [
                "",
                "## Sự kiện đưa vào huấn luyện",
                "",
                "| Chỉ số | Giá trị |",
                "|---|---:|",
                f"| Event nhận được | {ingest['events_received']:,} |",
                f"| Event được tính | {ingest['events_counted']:,} |",
                f"| Event bỏ qua | {ingest['events_ignored']:,} |",
                f"| Event vượt ngưỡng lặp | {ingest['events_capped']:,} |",
                f"| Event thiếu user_id | {ingest['events_without_user_id']:,} |",
                f"| Event thiếu timestamp | {ingest['events_without_timestamp']:,} |",
                f"| Phim ngoài catalog bị loại | {ingest.get('rows_outside_catalogue', 0):,} |",
                f"| Dòng huấn luyện sinh ra | {ingest['rows']:,} |",
                f"| Người dùng | {ingest['users']:,} |",
            ]
        )
        if ingest.get("unsupported_event_types"):
            lines.append("")
            lines.append(
                "Event type chưa hỗ trợ (frontend đang bắn nhưng model chưa biết): "
                + ", ".join(f"`{name}`" for name in ingest["unsupported_event_types"])
            )

    if decision:
        lines.extend(
            [
                "",
                "## Quyết định thăng cấp",
                "",
                f"Chỉ số xét duyệt: `{decision['metric']}`",
                "",
                "| Điều kiện | Đạt |",
                "|---|:---:|",
            ]
        )
        for name, passed in decision.get("checks", {}).items():
            lines.append(f"| {name} | {'có' if passed else '**không**'} |")
        lines.extend(
            [
                "",
                f"- Ứng viên: {decision.get('candidate_value')}",
                f"- Baseline popularity cùng lần chạy: {decision.get('popularity_value')}",
                f"- Model đang phục vụ: {decision.get('previous_value')}",
                f"- Ngưỡng sàn cho phép: {decision.get('regression_floor', '—')}",
                f"- User được chấm: {decision.get('users_scored') or 0:,}",
                "",
                f"**Kết luận:** {decision.get('reason')}",
            ]
        )
        mismatch = decision.get("protocol_mismatch")
        if mismatch:
            lines.extend(
                [
                    "",
                    f"> **Cảnh báo:** model đang phục vụ được đo trên "
                    f"{mismatch['previous_users_scored']:,} user, ứng viên đo trên "
                    f"{mismatch['candidate_users_scored']:,} user. "
                    f"{mismatch['note']}",
                ]
            )
        if report.get("forced"):
            lines.append("")
            lines.append(
                "> Đã thăng cấp bằng `--force-promote` bất chấp kết quả trên."
            )

    lines.extend(
        [
            "",
            "## Quay lui",
            "",
            "Sửa `artifacts/LATEST.json` trỏ về phiên bản cũ rồi khởi động lại "
            "service. Không có cơ chế nạp nóng (spec mục 14.3).",
            "",
        ]
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    config = load_config(REPOSITORY_ROOT / arguments.config)
    model_config = load_config(REPOSITORY_ROOT / arguments.model_config)
    aws_config = load_config(REPOSITORY_ROOT / arguments.aws_config)
    retraining = aws_config["retraining"]

    version = arguments.version or default_version()
    latest_before = read_latest(config)
    previous_version = latest_before.get("collaborative")

    report: dict[str, Any] = {
        "version": version,
        "started_at": started_at,
        "previous_version": previous_version,
        "promoted_version": previous_version,
        "steps": {},
        "dry_run": arguments.dry_run,
        "forced": False,
    }

    print(f"=== Retrain {version} ===", flush=True)
    print(f"Phiên bản đang phục vụ: {previous_version or '(chưa có)'}", flush=True)

    # -- Step 1: pull ---------------------------------------------------
    if arguments.pull:
        print("\n[1/6] Tải dataset và artifact từ S3...", flush=True)
        try:
            results = pull_from_s3(aws_config, arguments.dry_run)
        except Exception as error:
            # Failing here means training would run on whatever happens to be on
            # disk — possibly a stale dataset from a previous run. Stopping is
            # the only safe outcome.
            print(f"Không tải được dữ liệu từ S3: {error}", file=sys.stderr)
            return 1
        total = sum(item.get("downloaded", 0) for item in results)
        report["steps"]["pull"] = f"{total:,} file từ S3"
        report["pull"] = results
        print(f"  {total:,} file.", flush=True)
    else:
        report["steps"]["pull"] = "bỏ qua (--pull không bật)"

    # -- Step 2: ingest events -----------------------------------------
    if arguments.events:
        print(f"\n[2/6] Nạp event từ {arguments.events}...", flush=True)
        events = load_events(arguments.events, aws_config)
        frame, ingest_summary = events_to_interactions(
            events, model_config, retraining
        )
        frame, outside = restrict_to_catalogue(frame, config)
        ingest_summary["rows_outside_catalogue"] = outside
        ingest_summary["rows"] = int(len(frame))
        print(
            f"  {ingest_summary['events_received']:,} event -> "
            f"{ingest_summary['rows']:,} dòng huấn luyện "
            f"({ingest_summary['users']:,} user).",
            flush=True,
        )
        if ingest_summary["unsupported_event_types"]:
            print(
                "  Event type chưa hỗ trợ: "
                + ", ".join(ingest_summary["unsupported_event_types"]),
                flush=True,
            )

        if arguments.dry_run:
            report["steps"]["ingest"] = (
                f"dry-run: {ingest_summary['rows']:,} dòng sẽ được ghép"
            )
        else:
            merge = merge_into_interactions(
                config, frame, str(retraining["event_conflict_policy"])
            )
            ingest_summary["merge"] = merge
            print(
                f"  Ghép vào bảng interaction: {merge['merged_rows']:,} dòng thêm, "
                f"{merge['replaced_rows']:,} dòng bị bỏ do trùng "
                f"({merge['source_rows']:,} -> {merge['output_rows']:,}).",
                flush=True,
            )
            report["steps"]["ingest"] = (
                f"{ingest_summary['rows']:,} dòng, ghép {merge['merged_rows']:,}"
            )
        report["ingest"] = ingest_summary
    else:
        report["steps"]["ingest"] = "bỏ qua (không có --events)"

    # -- Step 3: rebuild splits ----------------------------------------
    # Always rerun when events arrived: the chronological holdout has to be
    # recomputed over the union, otherwise the newest interactions would sit in
    # train while older ones stay in test, and the evaluation would be measuring
    # the model's ability to predict the past from the future.
    if arguments.events and not arguments.dry_run:
        print("\n[3/6] Dựng lại split theo thời gian...", flush=True)
        splits = build_interaction_splits(config)
        report["splits"] = splits["splits"]
        report["steps"]["splits"] = (
            f"train {splits['splits']['train']['rows']:,} · "
            f"val {splits['splits']['validation']['rows']:,} · "
            f"test {splits['splits']['test']['rows']:,}"
        )
        print(f"  {report['steps']['splits']}", flush=True)
    else:
        report["steps"]["splits"] = "bỏ qua (dùng split hiện có)"

    if arguments.dry_run:
        report["steps"]["train"] = "dry-run: dừng trước khi huấn luyện"
        report["duration_seconds"] = time.perf_counter() - started
        _write_report(config, report)
        print("\nDry run: dừng trước bước huấn luyện.", flush=True)
        return 0

    # -- Step 4: train --------------------------------------------------
    print(f"\n[4/6] Huấn luyện ALS phiên bản {version}...", flush=True)
    train_argv = [
        "--config",
        str(REPOSITORY_ROOT / arguments.config),
        "--model-config",
        str(REPOSITORY_ROOT / arguments.model_config),
        "--version",
        version,
    ]
    for flag, value in (
        ("--factors", arguments.factors),
        ("--iterations", arguments.iterations),
        ("--regularization", arguments.regularization),
    ):
        if value is not None:
            train_argv.extend([flag, str(value)])
    if arguments.subset:
        train_argv.append("--subset")

    if train_module.main(train_argv) != 0:
        report["steps"]["train"] = "**thất bại**"
        report["duration_seconds"] = time.perf_counter() - started
        _write_report(config, report)
        print("Huấn luyện thất bại; LATEST.json giữ nguyên.", file=sys.stderr)
        return 1
    report["steps"]["train"] = f"artifact {version}"

    # train.py points LATEST.json at whatever it just produced, which is the
    # right default for a manual run but not for an unattended one. The pointer
    # is put back immediately and only moved again if the gate passes.
    write_latest(config, latest_before)

    # -- Step 5: evaluate ----------------------------------------------
    if arguments.skip_evaluation:
        report["steps"]["evaluate"] = "bỏ qua (--skip-evaluation)"
        report["steps"]["promote"] = "không thăng cấp: chưa đánh giá"
        report["duration_seconds"] = time.perf_counter() - started
        _write_report(config, report)
        print(
            f"\nĐã train {version} nhưng không đánh giá, nên không thăng cấp.",
            flush=True,
        )
        return 0

    evaluation_config = retraining["evaluation"]
    sample_users = arguments.sample_users
    if sample_users is None:
        sample_users = int(evaluation_config["sample_users"])
    print(f"\n[5/6] Đánh giá trên {sample_users:,} user...", flush=True)
    evaluate_argv = [
        "--config",
        str(REPOSITORY_ROOT / arguments.config),
        "--model-config",
        str(REPOSITORY_ROOT / arguments.model_config),
        "--split",
        str(evaluation_config["split"]),
        "--sample-users",
        str(sample_users),
        "--seed",
        str(evaluation_config["seed"]),
        "--als-version",
        version,
    ]
    if evaluate_module.main(evaluate_argv) != 0:
        report["steps"]["evaluate"] = "**thất bại**"
        report["duration_seconds"] = time.perf_counter() - started
        _write_report(config, report)
        print("Đánh giá thất bại; LATEST.json giữ nguyên.", file=sys.stderr)
        return 1

    payload = json.loads(
        (output_dir(config, "validation_dir") / "model_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    report["steps"]["evaluate"] = f"{payload['users_evaluated']:,} user"
    report["evaluation"] = payload

    # -- Step 6: promote or keep ---------------------------------------
    decision = evaluate_promotion(
        payload, previous_metrics(config, previous_version), retraining["promotion"]
    )
    report["promotion"] = decision

    if decision["promote"] or arguments.force_promote:
        report["forced"] = bool(arguments.force_promote and not decision["promote"])
        latest = dict(latest_before)
        latest["collaborative"] = version
        latest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_latest(config, latest)
        report["promoted_version"] = version
        report["steps"]["promote"] = f"thăng cấp {previous_version} -> {version}"
        print(f"\n[6/6] Thăng cấp: {previous_version} -> {version}", flush=True)
    else:
        report["steps"]["promote"] = f"giữ {previous_version}: {decision['reason']}"
        print(
            f"\n[6/6] KHÔNG thăng cấp. {decision['reason']}\n"
            f"       LATEST.json vẫn trỏ về {previous_version}. "
            f"Artifact {version} vẫn được giữ lại để xem xét.",
            flush=True,
        )

    report["duration_seconds"] = time.perf_counter() - started
    report_paths = _write_report(config, report)

    # -- Push -----------------------------------------------------------
    if arguments.push:
        print("\nĐẩy artifact, dữ liệu và báo cáo lên S3...", flush=True)
        results = push_to_s3(aws_config, None, arguments.dry_run)
        total = sum(item.get("uploaded", 0) for item in results)
        report["push"] = results
        report["steps"]["push"] = f"{total:,} file lên S3"
        _write_report(config, report)
        print(f"  {total:,} file.", flush=True)

    print(f"\nBáo cáo: {report_paths['markdown']}")
    # A blocked promotion is the gate working, not the job failing. Exiting
    # non-zero would mark the SageMaker job Failed and send everyone looking for
    # a crash that did not happen; the outcome is in the report instead.
    return 0


def _write_report(config: dict[str, Any], report: dict[str, Any]) -> dict[str, Path]:
    """Write the run report. Every run leaves one, promoted or not."""
    report.setdefault("duration_seconds", 0.0)
    validation_dir = output_dir(config, "validation_dir")
    json_path = validation_dir / "retrain_report.json"
    markdown_path = validation_dir / "retrain_report.md"
    write_json(json_path, report)
    markdown_path.write_text(report_markdown(report), encoding="utf-8", newline="\n")

    # A per-version copy as well, because the two files above are overwritten by
    # the next run and the history is what tells you when quality changed.
    history = validation_dir / "retrain_history"
    history.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(json_path, history / f"{report['version']}.json")
    return {"json": json_path, "markdown": markdown_path}


if __name__ == "__main__":
    sys.exit(main())
