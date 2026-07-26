"""Sweep hybrid fusion weights on the validation split.

Tuning happens on validation and never on test, so the test numbers in
`model_evaluation.md` stay an honest estimate of unseen performance.

The candidate pools are the expensive part - content-based scoring in particular -
so they are generated once and re-fused for every weight setting. A sweep
therefore costs barely more than a single evaluation run.

Usage:
    python scripts/tune_hybrid_weights.py --sample-users 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import load_config, output_dir  # noqa: E402
from src.models.collaborative import CollaborativeRecommender  # noqa: E402
from src.models.evaluation import (  # noqa: E402
    build_popularity_baseline,
    load_evaluation_set,
    rank_with_collaborative,
    rank_with_content,
    rank_with_hybrid,
    rank_with_popularity,
    score_rankings,
)
from src.models.hybrid_ranking import HybridRanker  # noqa: E402
from src.recommenders.content_based import ContentBasedRecommender  # noqa: E402

CONTENT_FLOORS = (0.0, 0.1, 0.25, 0.5, 0.75)
POPULARITY_WEIGHTS = (0.0, 0.2)
# 0 disables the cap, so the sweep can separate the cost of mixing in a weaker
# source from the cost of enforcing genre variety.
GENRE_CAPS = (0, 4, 8)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/model_serving.yaml")
    parser.add_argument("--sample-users", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    config = load_config(arguments.config)
    model_config = load_config(arguments.model_config)
    candidates_config = model_config["candidates"]

    print("Tập validation (không dùng test cho tuning).", flush=True)
    baseline = build_popularity_baseline(config)
    evaluation = load_evaluation_set(
        config,
        split="validation",
        sample_users=arguments.sample_users,
        seed=arguments.seed,
    )
    print(f"  Chấm {len(evaluation.user_ids):,} user.", flush=True)

    print("Sinh candidate pool (một lần)...", flush=True)
    popularity_pool = rank_with_popularity(
        baseline, evaluation, int(candidates_config["popularity_topn"])
    )
    als = CollaborativeRecommender.load(
        Path(config["_project_root"]) / "artifacts" / "collaborative" / "v1.0.0"
    )
    collaborative_pool, _ = rank_with_collaborative(
        als, evaluation, int(candidates_config["cf_topn"])
    )
    content_pool, _ = rank_with_content(
        ContentBasedRecommender(config), evaluation, int(candidates_config["content_topn"])
    )

    serving_dir = output_dir(config, "serving_dir", create=False)
    serving = pd.read_parquet(
        serving_dir / "movies_serving.parquet", columns=["movie_id", "genres"]
    )
    genres_by_movie = {
        int(row.movie_id): (
            row.genres.tolist() if hasattr(row.genres, "tolist") else list(row.genres)
        )
        for row in serving.itertuples(index=False)
    }
    catalog_size = len(serving)

    reference = score_rankings(
        "collaborative_als",
        {
            user_id: ranked[: arguments.top_k]
            for user_id, ranked in collaborative_pool.items()
        },
        evaluation,
    )

    rows = []
    for cap in GENRE_CAPS:
        for floor in CONTENT_FLOORS:
            for w_pop in POPULARITY_WEIGHTS:
                trial = deepcopy(model_config)
                trial["hybrid"]["w_content_floor"] = floor
                trial["hybrid"]["w_pop"] = w_pop
                trial["hybrid"]["max_per_genre_in_top20"] = cap
                rankings, _ = rank_with_hybrid(
                    HybridRanker(trial),
                    evaluation,
                    collaborative_pool,
                    content_pool,
                    popularity_pool,
                    genres_by_movie,
                    arguments.top_k,
                )
                scores = score_rankings("hybrid", rankings, evaluation)
                coverage = (
                    scores.metrics["distinct_movies_recommended"] / catalog_size
                    if catalog_size
                    else 0.0
                )
                rows.append(
                    {
                        "max_per_genre": cap,
                        "w_content_floor": floor,
                        "w_pop": w_pop,
                        "hit_rate_at_10": scores.metrics["hit_rate_at_10"],
                        "ndcg_at_10": scores.metrics["ndcg_at_10"],
                        "distinct_movies": int(
                            scores.metrics["distinct_movies_recommended"]
                        ),
                        "coverage": coverage,
                        "users_scored": scores.users_scored,
                    }
                )
                print(
                    f"  cap={cap or 'off':<4} floor={floor:<5} w_pop={w_pop:<4} "
                    f"HR@10={scores.metrics['hit_rate_at_10']:.4f} "
                    f"NDCG@10={scores.metrics['ndcg_at_10']:.4f} "
                    f"coverage={coverage:.2%}",
                    flush=True,
                )

    validation_dir = output_dir(config, "validation_dir")
    lines = [
        "# Hybrid weight sweep (validation)",
        "",
        f"Tập validation, {len(evaluation.user_ids):,} user, seed {arguments.seed}.",
        "Tuning chỉ chạy trên validation; tập test không được dùng ở bước này.",
        "",
        f"Tham chiếu — ALS đơn lẻ trên cùng tập: HitRate@10 = "
        f"{reference.metrics['hit_rate_at_10']:.4f}, "
        f"NDCG@10 = {reference.metrics['ndcg_at_10']:.4f}, "
        f"{int(reference.metrics['distinct_movies_recommended']):,} phim "
        f"({reference.metrics['distinct_movies_recommended'] / catalog_size:.2%}).",
        "",
        "`max_per_genre = off` tắt giới hạn đa dạng, dùng để tách phần mất độ "
        "chính xác do trộn nguồn yếu khỏi phần mất do ép đa dạng genre.",
        "",
        "| max_per_genre | w_content_floor | w_pop | HitRate@10 | NDCG@10 | Phim khác nhau | Coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cap = row["max_per_genre"] or "off"
        lines.append(
            f"| {cap} | {row['w_content_floor']} | {row['w_pop']} "
            f"| {row['hit_rate_at_10']:.4f} | {row['ndcg_at_10']:.4f} "
            f"| {row['distinct_movies']:,} | {row['coverage']:.2%} |"
        )
    lines.append("")

    (validation_dir / "hybrid_weight_sweep.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    (validation_dir / "hybrid_weight_sweep.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "users_evaluated": len(evaluation.user_ids),
                "seed": arguments.seed,
                "reference_collaborative": reference.metrics,
                "trials": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print()
    print(f"Đã ghi: {validation_dir / 'hybrid_weight_sweep.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
