"""Run offline evaluation and write the model comparison report.

Usage:
    python evaluate.py --config configs/data_pipeline.yaml
    python evaluate.py --config configs/data_pipeline.yaml --sample-users 5000
    python evaluate.py --config configs/data_pipeline.yaml --split validation

The popularity baseline is the reference every personalised model must beat. A
model that cannot outrank "show everyone the same popular movies" provides no
value, so the report states that comparison explicitly instead of leaving it to
the reader.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.config import load_config, output_dir
from src.data.config import project_path
from src.models.collaborative import CollaborativeRecommender
from src.models.evaluation import (
    RELEVANT_RATING_THRESHOLD,
    EvaluationSet,
    ModelScores,
    build_popularity_baseline,
    load_evaluation_set,
    rank_with_collaborative,
    rank_with_content,
    rank_with_hybrid,
    rank_with_popularity,
    score_rankings,
)
from src.models.hybrid_ranking import HybridRanker
from src.recommenders.content_based import ContentBasedRecommender

DEFAULT_SAMPLE_USERS = 5_000
CUTOFFS = (10, 20)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument(
        "--sample-users",
        type=int,
        default=DEFAULT_SAMPLE_USERS,
        help="Number of holdout users to score; 0 evaluates every eligible user.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=max(CUTOFFS))
    parser.add_argument(
        "--skip-content",
        action="store_true",
        help="Skip the content-based model.",
    )
    parser.add_argument(
        "--skip-collaborative",
        action="store_true",
        help="Skip the ALS model even if an artifact exists.",
    )
    parser.add_argument(
        "--skip-hybrid",
        action="store_true",
        help="Skip the hybrid layer (it needs both ALS and content candidates).",
    )
    parser.add_argument(
        "--als-version",
        default=None,
        help="Artifact version under artifacts/collaborative; defaults to LATEST.json.",
    )
    parser.add_argument("--model-config", default="configs/model_serving.yaml")
    return parser.parse_args(argv)


def resolve_als_directory(
    config: dict[str, Any], requested: str | None
) -> Path | None:
    """Locate the ALS artifact, following LATEST.json when no version is given."""
    base = project_path(config, "artifacts/collaborative")
    if requested:
        candidate = base / requested
        return candidate if candidate.exists() else None
    pointer = project_path(config, "artifacts") / "LATEST.json"
    if not pointer.exists():
        return None
    latest = json.loads(pointer.read_text(encoding="utf-8"))
    version = latest.get("collaborative")
    if not version:
        return None
    candidate = base / version
    return candidate if candidate.exists() else None


def write_metrics_into_manifest(
    manifest_path: Path, results: list[ModelScores]
) -> None:
    """Record the measured metrics back into the artifact manifest.

    Spec section 14.2 requires an artifact to carry its own evaluation numbers, so
    a saved model can always be traced to the result it produced.
    """
    scores = next(
        (item for item in results if item.model == "collaborative_als"), None
    )
    if scores is None or scores.users_scored == 0 or not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metrics"] = {
        "hit_rate_at_10": round(scores.metrics["hit_rate_at_10"], 6),
        "hit_rate_at_20": round(scores.metrics["hit_rate_at_20"], 6),
        "ndcg_at_10": round(scores.metrics["ndcg_at_10"], 6),
        "distinct_movies_recommended": int(
            scores.metrics["distinct_movies_recommended"]
        ),
        "users_scored": scores.users_scored,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _coverage_ratio(scores: ModelScores, catalog_size: int) -> float:
    distinct = scores.metrics.get("distinct_movies_recommended", 0.0)
    return distinct / catalog_size if catalog_size else 0.0


def build_report(
    arguments: argparse.Namespace,
    evaluation: EvaluationSet,
    results: list[ModelScores],
    catalog_size: int,
    baseline_rows: int,
) -> tuple[str, dict[str, Any]]:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    baseline = next(
        (item for item in results if item.model == "popularity_train"), None
    )

    lines = [
        "# Model evaluation",
        "",
        f"Split: **{arguments.split}** · Ngày chạy: {generated_at}",
        "",
        "## Cấu hình đánh giá",
        "",
        "| Tham số | Giá trị |",
        "|---|---|",
        f"| Ngưỡng relevant | `rating >= {RELEVANT_RATING_THRESHOLD}` |",
        f"| User holdout có item relevant | {evaluation.population:,} |",
        f"| User được chấm | {len(evaluation.user_ids):,} |",
        f"| Lấy mẫu | {'có' if evaluation.sampled else 'không (toàn bộ)'} |",
        f"| Seed | {arguments.seed} |",
        f"| Top-K sinh ra | {arguments.top_k} |",
        f"| Phim trong baseline (đủ ngưỡng vote) | {baseline_rows:,} |",
        f"| Catalog | {catalog_size:,} |",
        "",
        "Baseline popularity được dựng lại **chỉ từ `interactions_train.parquet`**. "
        "File `data/serving/top_rated_all.parquet` tính từ toàn bộ rating nên đã "
        "chứa tín hiệu của validation và test; dùng nó làm baseline sẽ rò rỉ dữ liệu.",
        "",
        "Mọi phim user đã tương tác trong train đều bị loại khỏi danh sách gợi ý "
        "trước khi chấm điểm.",
        "",
        "## Kết quả",
        "",
        "| Model | User chấm | HitRate@10 | HitRate@20 | NDCG@10 | Precision@10 | Recall@20 | Phim khác nhau | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for scores in results:
        if scores.users_scored == 0:
            lines.append(f"| {scores.model} | 0 | — | — | — | — | — | — | — |")
            continue
        metrics = scores.metrics
        lines.append(
            f"| {scores.model} "
            f"| {scores.users_scored:,} "
            f"| {metrics['hit_rate_at_10']:.4f} "
            f"| {metrics['hit_rate_at_20']:.4f} "
            f"| {metrics['ndcg_at_10']:.4f} "
            f"| {metrics['precision_at_10']:.4f} "
            f"| {metrics['recall_at_20']:.4f} "
            f"| {int(metrics['distinct_movies_recommended']):,} "
            f"| {_coverage_ratio(scores, catalog_size):.2%} |"
        )

    lines.extend(
        [
            "",
            "Vì mỗi user chỉ có đúng một item bị giấu, `Recall@K` bằng `HitRate@K` và "
            "`Precision@K` bằng `HitRate@K / K`. Hai cột đó không mang thông tin độc "
            "lập; chỉ số cần đọc là **HitRate** và **NDCG**.",
            "",
            "## So với baseline",
            "",
        ]
    )

    if baseline is None or baseline.users_scored == 0:
        lines.append("Chưa có baseline để so sánh.")
    else:
        reference = baseline.metrics["hit_rate_at_10"]
        comparisons = [item for item in results if item.model != baseline.model]
        if not comparisons:
            lines.append(
                "Mới chỉ chạy baseline. Chưa có model cá nhân hóa nào để so sánh."
            )
        for scores in comparisons:
            if scores.users_scored == 0:
                continue
            value = scores.metrics["hit_rate_at_10"]
            if reference > 0:
                delta = (value - reference) / reference
                verdict = "**vượt**" if value > reference else "**không vượt**"
                lines.append(
                    f"- `{scores.model}` {verdict} baseline trên HitRate@10: "
                    f"{value:.4f} so với {reference:.4f} ({delta:+.1%})."
                )
            else:
                lines.append(f"- `{scores.model}`: HitRate@10 = {value:.4f}.")

    notes = [item for item in results if item.notes]
    if notes:
        lines.extend(["", "## Ghi chú", ""])
        lines.extend(f"- `{item.model}`: {item.notes}" for item in notes)

    lines.extend(
        [
            "",
            "## Tái tạo",
            "",
            "```",
            f"python evaluate.py --config {arguments.config} "
            f"--split {arguments.split} --sample-users {arguments.sample_users} "
            f"--seed {arguments.seed}",
            "```",
            "",
            f"Python {platform.python_version()} · pandas {pd.__version__}",
            "",
        ]
    )

    payload = {
        "generated_at": generated_at,
        "split": arguments.split,
        "relevant_rating_threshold": RELEVANT_RATING_THRESHOLD,
        "eligible_users": evaluation.population,
        "users_evaluated": len(evaluation.user_ids),
        "sampled": evaluation.sampled,
        "seed": arguments.seed,
        "top_k": arguments.top_k,
        "catalog_size": catalog_size,
        "baseline_rows": baseline_rows,
        "models": [
            {
                "model": scores.model,
                "users_scored": scores.users_scored,
                "metrics": scores.metrics,
                "coverage_ratio": _coverage_ratio(scores, catalog_size),
                "notes": scores.notes,
            }
            for scores in results
        ],
    }
    return "\n".join(lines), payload


def _use_utf8_console() -> None:
    """Windows consoles default to a legacy codepage that cannot print Vietnamese."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    config = load_config(arguments.config)

    print("Dựng popularity baseline từ train...", flush=True)
    baseline = build_popularity_baseline(config)
    print(f"  {len(baseline):,} phim đủ ngưỡng vote.", flush=True)

    print(f"Nạp tập đánh giá ({arguments.split})...", flush=True)
    evaluation = load_evaluation_set(
        config,
        split=arguments.split,
        sample_users=arguments.sample_users,
        seed=arguments.seed,
    )
    print(
        f"  {evaluation.population:,} user có item relevant; "
        f"chấm {len(evaluation.user_ids):,} user.",
        flush=True,
    )

    serving_dir = output_dir(config, "serving_dir", create=False)
    catalog_size = len(
        pd.read_parquet(serving_dir / "movies_serving.parquet", columns=["movie_id"])
    )

    model_config = load_config(arguments.model_config)
    candidates_config = model_config["candidates"]

    results: list[ModelScores] = []
    als_manifest_path: Path | None = None
    # Candidate pools are generated once at the configured depth and reused: the
    # per-model tables score their first `top_k`, the hybrid layer fuses the full
    # lists. Regenerating them per model would triple the runtime for no gain.
    collaborative_pool: dict[int, list[int]] = {}
    content_pool: dict[int, list[int]] = {}

    print("Chấm popularity baseline...", flush=True)
    popularity_pool = rank_with_popularity(
        baseline, evaluation, int(candidates_config["popularity_topn"])
    )
    popularity_rankings = {
        user_id: ranked[: arguments.top_k]
        for user_id, ranked in popularity_pool.items()
    }
    results.append(
        score_rankings(
            "popularity_train",
            popularity_rankings,
            evaluation,
            cutoffs=CUTOFFS,
            notes="Cùng một danh sách cho mọi user, chỉ loại phim đã xem.",
        )
    )

    if not arguments.skip_collaborative:
        als_directory = resolve_als_directory(config, arguments.als_version)
        if als_directory is None:
            print(
                "Bỏ qua collaborative: chưa có artifact. Chạy `python train.py` trước.",
                flush=True,
            )
        else:
            print(f"Chấm collaborative ALS ({als_directory.name})...", flush=True)
            als_manifest_path = als_directory / "manifest.json"
            als = CollaborativeRecommender.load(als_directory)
            collaborative_pool, unknown = rank_with_collaborative(
                als, evaluation, int(candidates_config["cf_topn"])
            )
            als_rankings = {
                user_id: ranked[: arguments.top_k]
                for user_id, ranked in collaborative_pool.items()
            }
            note = (
                f"Artifact `{als_directory.name}`, "
                f"{als.item_factors.shape[0]:,} phim trong index. "
                "Đã loại toàn bộ lịch sử train của user."
            )
            if unknown:
                note += (
                    f" {unknown:,} user không có trong ALS index (cold-start), "
                    "không sinh được gợi ý."
                )
            results.append(
                score_rankings(
                    "collaborative_als",
                    als_rankings,
                    evaluation,
                    cutoffs=CUTOFFS,
                    notes=note,
                )
            )

    if not arguments.skip_content:
        print("Chấm content-based (mô phỏng onboarding)...", flush=True)
        recommender = ContentBasedRecommender(config)
        content_pool, skipped = rank_with_content(
            recommender, evaluation, int(candidates_config["content_topn"])
        )
        content_rankings = {
            user_id: ranked[: arguments.top_k]
            for user_id, ranked in content_pool.items()
        }
        note = (
            "Mỗi user dùng 3 phim được đánh giá cao sớm nhất trong train làm "
            "lựa chọn onboarding."
        )
        if skipped:
            note += f" Bỏ qua {skipped:,} user không đủ dữ liệu onboarding."
        results.append(
            score_rankings(
                "content_tfidf",
                content_rankings,
                evaluation,
                cutoffs=CUTOFFS,
                notes=note,
            )
        )

    if not arguments.skip_hybrid and collaborative_pool and content_pool:
        print("Chấm hybrid (weighted RRF)...", flush=True)
        serving = pd.read_parquet(
            serving_dir / "movies_serving.parquet", columns=["movie_id", "genres"]
        )
        genres_by_movie = {
            int(row.movie_id): (
                row.genres.tolist() if hasattr(row.genres, "tolist") else list(row.genres)
            )
            for row in serving.itertuples(index=False)
        }
        ranker = HybridRanker(model_config)
        hybrid_rankings, levels = rank_with_hybrid(
            ranker,
            evaluation,
            collaborative_pool,
            content_pool,
            popularity_pool,
            genres_by_movie,
            arguments.top_k,
        )
        hybrid_settings = model_config["hybrid"]
        level_summary = ", ".join(
            f"{name}={count:,}" for name, count in sorted(levels.items())
        )
        results.append(
            score_rankings(
                "hybrid_rrf",
                hybrid_rankings,
                evaluation,
                cutoffs=CUTOFFS,
                notes=(
                    f"Weighted RRF (k={hybrid_settings['rrf_k']}, "
                    f"full_cf_confidence_at={hybrid_settings['full_cf_confidence_at']}), "
                    f"tối đa {hybrid_settings['max_per_genre_in_top20']} phim/genre. "
                    f"Tầng fallback: {level_summary}."
                ),
            )
        )

    if als_manifest_path is not None:
        write_metrics_into_manifest(als_manifest_path, results)

    validation_dir = output_dir(config, "validation_dir")
    report, payload = build_report(
        arguments, evaluation, results, catalog_size, len(baseline)
    )
    (validation_dir / "model_evaluation.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    (validation_dir / "model_evaluation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print()
    print(report)
    print(f"Đã ghi: {Path(validation_dir) / 'model_evaluation.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
