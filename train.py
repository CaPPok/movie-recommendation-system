"""Train the collaborative filtering model and write a versioned artifact.

Usage:
    python train.py
    python train.py --version v1.0.1 --factors 128
    python train.py --subset            # reproducible user sample, see --help

The script only orchestrates: the rating-to-confidence rules and the ALS call
live in `src/models/collaborative.py`. Everything needed to reproduce a run is
recorded in `manifest.json` next to the factors, so an artifact can always be
traced back to the data and code that produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# BLAS threading must be capped before implicit is imported, otherwise its own
# thread pool and the BLAS pool oversubscribe the CPU and the fit slows down.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from src.data.config import load_config, output_dir, project_path
from src.models.collaborative import (
    CollaborativeRecommender,
    build_training_matrix,
    train_als,
)

DEFAULT_VERSION = "v1.0.0"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/model_serving.yaml")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--factors", type=int, default=None)
    parser.add_argument("--regularization", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--neutral-policy",
        choices=["exclude", "low_confidence"],
        default=None,
        help="Override how ratings between the negative and positive thresholds are used.",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Train on a reproducible user sample instead of the full split.",
    )
    parser.add_argument(
        "--subset-users",
        type=int,
        default=None,
        help="Number of users when --subset is set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing artifact version.",
    )
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def apply_overrides(
    model_config: dict[str, Any], arguments: argparse.Namespace
) -> dict[str, Any]:
    settings = model_config["collaborative"]
    if arguments.factors is not None:
        settings["factors"] = arguments.factors
    if arguments.regularization is not None:
        settings["regularization"] = arguments.regularization
    if arguments.iterations is not None:
        settings["iterations"] = arguments.iterations
    if arguments.neutral_policy is not None:
        settings["neutral_rating_policy"] = arguments.neutral_policy
    if arguments.subset:
        settings["subset"]["enabled"] = True
    if arguments.subset_users is not None:
        settings["subset"]["user_sample_size"] = arguments.subset_users
    return model_config


def apply_subset(
    train: pd.DataFrame, model_config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Take a reproducible sample of users, keeping each sampled user's full history.

    Sampling by user rather than by row matters: cutting rows at random thins out
    everybody's history and destroys the co-occurrence signal collaborative
    filtering depends on.
    """
    subset = model_config["collaborative"]["subset"]
    if not subset.get("enabled"):
        return train, None

    size = int(subset["user_sample_size"])
    seed = int(subset["seed"])
    all_users = np.sort(train["user_id"].unique())
    if size >= len(all_users):
        return train, None

    generator = np.random.default_rng(seed)
    chosen = np.sort(generator.choice(all_users, size=size, replace=False))
    reduced = train.loc[train["user_id"].isin(pd.Index(chosen))]
    summary = {
        "enabled": True,
        "seed": seed,
        "users_before": int(len(all_users)),
        "users_after": int(reduced["user_id"].nunique()),
        "movies_before": int(train["movie_id"].nunique()),
        "movies_after": int(reduced["movie_id"].nunique()),
        "interactions_before": int(len(train)),
        "interactions_after": int(len(reduced)),
    }
    return reduced, summary


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    config = load_config(arguments.config)
    model_config = apply_overrides(load_config(arguments.model_config), arguments)
    settings = model_config["collaborative"]

    root = Path(config["_project_root"])
    destination = project_path(config, "artifacts/collaborative") / arguments.version
    if destination.exists() and not arguments.overwrite:
        print(
            f"Artifact đã tồn tại: {destination}\n"
            "Dùng --version để tạo phiên bản mới, hoặc --overwrite nếu thực sự "
            "muốn ghi đè.",
            file=sys.stderr,
        )
        return 1

    splits_dir = output_dir(config, "splits_dir", create=False)
    print("Nạp interactions_train.parquet...", flush=True)
    train = pd.read_parquet(
        splits_dir / "interactions_train.parquet",
        columns=["user_id", "movie_id", "interaction_value"],
    )
    print(f"  {len(train):,} interaction.", flush=True)

    train, subset_summary = apply_subset(train, model_config)
    if subset_summary:
        print(
            f"  Subset: {subset_summary['users_after']:,} user, "
            f"{subset_summary['interactions_after']:,} interaction.",
            flush=True,
        )

    print("Dựng ma trận confidence...", flush=True)
    training = build_training_matrix(train, model_config)
    stats = training.stats
    print(
        f"  {stats['matrix_rows']:,} user x {stats['matrix_columns']:,} phim, "
        f"{stats['matrix_nonzero']:,} ô khác 0 "
        f"(sparsity {stats['sparsity']:.6%}).",
        flush=True,
    )
    print(
        f"  Loại {stats['movies_dropped_below_minimum']:,} phim dưới "
        f"{stats['minimum_positive_interactions_per_movie']} lượt; "
        f"{stats['negative_rows_excluded']:,} rating thấp không đưa vào model.",
        flush=True,
    )

    print(
        f"Huấn luyện ALS (factors={settings['factors']}, "
        f"iterations={settings['iterations']}, "
        f"regularization={settings['regularization']}, "
        f"alpha={settings['alpha']})...",
        flush=True,
    )
    started = time.perf_counter()
    model = train_als(training.matrix, model_config)
    elapsed = time.perf_counter() - started
    print(f"  Xong sau {elapsed:.1f} giây.", flush=True)

    user_factors = np.asarray(model.user_factors, dtype="float32")
    item_factors = np.asarray(model.item_factors, dtype="float32")
    recommender = CollaborativeRecommender(
        user_factors=user_factors,
        item_factors=item_factors,
        user_ids=training.user_ids,
        movie_ids=training.movie_ids,
    )

    manifest = {
        "model_name": "collaborative_als",
        "model_version": arguments.version.lstrip("v"),
        "artifact_version": arguments.version,
        "library": "implicit.als.AlternatingLeastSquares",
        "dataset_snapshot": "data/splits/interactions_train.parquet",
        "git_commit": _git_commit(root),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": round(elapsed, 2),
        "hyperparameters": {
            "factors": int(settings["factors"]),
            "regularization": float(settings["regularization"]),
            "iterations": int(settings["iterations"]),
            "alpha": float(settings["alpha"]),
            "random_state": int(settings["random_state"]),
        },
        "feature_config": {
            "positive_rating_threshold": float(settings["positive_rating_threshold"]),
            "negative_rating_threshold": float(settings["negative_rating_threshold"]),
            "neutral_rating_policy": settings["neutral_rating_policy"],
            "min_positive_interactions_per_movie": int(
                settings["min_positive_interactions_per_movie"]
            ),
        },
        "data_stats": stats,
        "subset": subset_summary,
        "index_files": {"user": "user_index.parquet", "movie": "movie_index.parquet"},
        "factor_shapes": {
            "user_factors": list(user_factors.shape),
            "item_factors": list(item_factors.shape),
        },
        "metrics": {"hit_rate_at_10": None, "ndcg_at_10": None},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    recommender.save(destination, manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    latest_path = project_path(config, "artifacts") / "LATEST.json"
    latest: dict[str, Any] = {}
    if latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["collaborative"] = arguments.version
    latest["updated_at"] = manifest["trained_at"]
    latest_path.write_text(
        json.dumps(latest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print()
    print(f"Artifact: {destination}")
    print(f"Con trỏ phiên bản: {latest_path}")
    print(
        "Bước tiếp theo: python evaluate.py --sample-users 5000 "
        "để đo model này so với baseline."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
