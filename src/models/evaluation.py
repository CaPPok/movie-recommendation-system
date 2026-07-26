"""Offline evaluation for recommendation models.

The evaluation protocol follows the chronological leave-one-out split produced by
`src/data/splitting.py`. Every holdout user has exactly one hidden item, so a model
is scored by whether that single item appears in its ranked output.

Two rules exist to keep the numbers honest and are enforced here rather than left
to the caller. First, the popularity baseline is rebuilt from the training split
only; the serving ranking in `data/serving/top_rated_all.parquet` is computed from
every clean rating and therefore already contains validation and test signal.
Second, items a user already interacted with during training are removed from every
candidate list, otherwise a model is rewarded for repeating known history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.config import output_dir
from src.recommenders.guest import weighted_rating

RELEVANT_RATING_THRESHOLD = 4.0


@dataclass(frozen=True)
class EvaluationSet:
    """Users being scored, their hidden item, and their training history."""

    user_ids: np.ndarray
    held_out_movie_ids: np.ndarray
    seen_movie_ids: dict[int, np.ndarray]
    onboarding_movie_ids: dict[int, list[int]]
    sampled: bool
    population: int


@dataclass
class ModelScores:
    """Aggregated metrics for one model over one evaluation set."""

    model: str
    users_scored: int
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""


def build_popularity_baseline(config: dict[str, Any]) -> pd.DataFrame:
    """Rebuild the weighted-rating ranking using training interactions only.

    Mirrors `src/recommenders/guest.build_top_rated_rankings` but recomputes the
    global mean `C` and the vote threshold `m` from the training split, so the
    baseline never observes a validation or test interaction.
    """
    splits_dir = output_dir(config, "splits_dir", create=False)
    eval_dir = output_dir(config, "eval_dir")
    train = pd.read_parquet(
        splits_dir / "interactions_train.parquet",
        columns=["movie_id", "interaction_value"],
    )

    stats = train.groupby("movie_id", sort=True)["interaction_value"].agg(
        ["count", "sum"]
    )
    stats.columns = ["rating_count", "rating_sum"]
    stats = stats.reset_index()
    stats["average_rating"] = stats["rating_sum"] / stats["rating_count"]

    global_average = float(stats["rating_sum"].sum() / stats["rating_count"].sum())
    selected = float(config["ranking"]["selected_percentile"])
    minimum_count = int(
        stats["rating_count"].quantile(selected, interpolation="higher")
    )

    eligible = stats.loc[stats["rating_count"] >= minimum_count].copy()
    eligible["score"] = weighted_rating(
        eligible["average_rating"],
        eligible["rating_count"],
        global_average,
        minimum_count,
    )
    eligible = eligible.sort_values(
        ["score", "rating_count", "average_rating", "movie_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    eligible.insert(0, "rank", range(1, len(eligible) + 1))
    eligible = eligible[
        ["rank", "movie_id", "score", "average_rating", "rating_count"]
    ]

    baseline_path = eval_dir / "popularity_baseline_train.parquet"
    temporary = baseline_path.with_name(f"{baseline_path.name}.tmp")
    eligible.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(baseline_path)
    return eligible


def load_evaluation_set(
    config: dict[str, Any],
    split: str = "test",
    sample_users: int = 0,
    seed: int = 42,
    onboarding_size: int = 3,
) -> EvaluationSet:
    """Collect holdout users with a relevant hidden item and their train history.

    A holdout item counts as relevant only when its rating reaches
    `RELEVANT_RATING_THRESHOLD`. Users whose hidden item is a low rating are
    excluded: predicting a movie someone disliked is not a success.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if sample_users < 0:
        raise ValueError("sample_users must be zero or positive")

    splits_dir = output_dir(config, "splits_dir", create=False)
    holdout = pd.read_parquet(
        splits_dir / f"interactions_{split}.parquet",
        columns=["user_id", "movie_id", "interaction_value"],
    )
    relevant = holdout.loc[
        holdout["interaction_value"] >= RELEVANT_RATING_THRESHOLD
    ].sort_values("user_id", kind="mergesort")
    population = len(relevant)

    sampled = 0 < sample_users < population
    if sampled:
        generator = np.random.default_rng(seed)
        positions = np.sort(
            generator.choice(population, size=sample_users, replace=False)
        )
        relevant = relevant.iloc[positions]

    user_ids = relevant["user_id"].to_numpy(dtype="int64")
    held_out = relevant["movie_id"].to_numpy(dtype="int64")

    train = pd.read_parquet(
        splits_dir / "interactions_train.parquet",
        columns=["user_id", "movie_id", "interaction_value", "timestamp"],
    )
    train = train.loc[train["user_id"].isin(pd.Index(user_ids))]

    seen: dict[int, np.ndarray] = {
        int(user): group.to_numpy(dtype="int64")
        for user, group in train.groupby("user_id", sort=False)["movie_id"]
    }

    positive = train.loc[
        train["interaction_value"] >= RELEVANT_RATING_THRESHOLD
    ].sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort")
    onboarding: dict[int, list[int]] = {
        int(user): group.head(onboarding_size).to_numpy(dtype="int64").tolist()
        for user, group in positive.groupby("user_id", sort=False)["movie_id"]
    }

    return EvaluationSet(
        user_ids=user_ids,
        held_out_movie_ids=held_out,
        seen_movie_ids=seen,
        onboarding_movie_ids=onboarding,
        sampled=sampled,
        population=population,
    )


def _rank_of_hit(ranked: Iterable[int], target: int, cutoff: int) -> int | None:
    """Return the 1-based position of `target` within the first `cutoff` items."""
    for position, movie_id in enumerate(ranked, start=1):
        if position > cutoff:
            return None
        if movie_id == target:
            return position
    return None


def score_rankings(
    model: str,
    rankings: dict[int, list[int]],
    evaluation: EvaluationSet,
    cutoffs: tuple[int, ...] = (10, 20),
    notes: str = "",
) -> ModelScores:
    """Turn per-user ranked lists into HitRate, NDCG, Precision, Recall, Coverage.

    Because the split hides exactly one item per user, Recall@K equals HitRate@K
    and Precision@K equals HitRate@K / K. Both are reported for completeness but
    they carry no information beyond HitRate.
    """
    largest = max(cutoffs)
    hits = {cutoff: 0 for cutoff in cutoffs}
    discounted = 0.0
    covered: set[int] = set()
    scored = 0

    for user_id, target in zip(
        evaluation.user_ids, evaluation.held_out_movie_ids, strict=True
    ):
        ranked = rankings.get(int(user_id))
        if not ranked:
            continue
        scored += 1
        covered.update(ranked[:largest])
        position = _rank_of_hit(ranked, int(target), largest)
        if position is None:
            continue
        for cutoff in cutoffs:
            if position <= cutoff:
                hits[cutoff] += 1
        if position <= 10:
            discounted += 1.0 / np.log2(position + 1.0)

    if scored == 0:
        return ModelScores(model=model, users_scored=0, notes="Không có user nào được chấm.")

    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        metrics[f"hit_rate_at_{cutoff}"] = hits[cutoff] / scored
        metrics[f"recall_at_{cutoff}"] = hits[cutoff] / scored
        metrics[f"precision_at_{cutoff}"] = hits[cutoff] / (scored * cutoff)
    metrics["ndcg_at_10"] = discounted / scored
    metrics["distinct_movies_recommended"] = float(len(covered))
    return ModelScores(model=model, users_scored=scored, metrics=metrics, notes=notes)


def rank_with_popularity(
    baseline: pd.DataFrame, evaluation: EvaluationSet, top_k: int
) -> dict[int, list[int]]:
    """Recommend the same train-only popularity ranking to everyone, minus history."""
    ordered = baseline["movie_id"].to_numpy(dtype="int64")
    rankings: dict[int, list[int]] = {}
    for user_id in evaluation.user_ids:
        user = int(user_id)
        seen = evaluation.seen_movie_ids.get(user)
        if seen is None or len(seen) == 0:
            rankings[user] = ordered[:top_k].tolist()
            continue
        mask = ~np.isin(ordered, seen, assume_unique=False)
        rankings[user] = ordered[mask][:top_k].tolist()
    return rankings


def rank_with_hybrid(
    ranker: Any,
    evaluation: EvaluationSet,
    collaborative: dict[int, list[int]],
    content: dict[int, list[int]],
    popularity: dict[int, list[int]],
    genres_by_movie: dict[int, list[str]],
    top_k: int,
) -> tuple[dict[int, list[int]], dict[str, int]]:
    """Fuse the three candidate sets per user and return the final ordering.

    Interaction count comes from the training history, which is what decides how
    far the collaborative weight has ramped up for that user.
    """
    rankings: dict[int, list[int]] = {}
    levels: dict[str, int] = {}
    for user_id in evaluation.user_ids:
        user = int(user_id)
        seen = evaluation.seen_movie_ids.get(user, np.empty(0, dtype="int64"))
        items, level, _ = ranker.rank(
            collaborative=collaborative.get(user, []),
            content=content.get(user, []),
            popularity=popularity.get(user, []),
            valid_interaction_count=int(len(seen)),
            limit=top_k,
            exclude_movie_ids=seen.tolist(),
            genres_by_movie=genres_by_movie,
        )
        if items:
            rankings[user] = [item.movie_id for item in items]
        levels[level] = levels.get(level, 0) + 1
    return rankings, levels


def rank_with_collaborative(
    recommender: Any, evaluation: EvaluationSet, top_k: int
) -> tuple[dict[int, list[int]], int]:
    """Rank with ALS, excluding each user's full training history.

    Users missing from the ALS index produce no ranking. That is the honest
    outcome for a collaborative model and is what the scenario router downgrades
    on; filling the gap here would hide the cold-start rate from the report.
    """
    rankings = recommender.recommend_batch(
        evaluation.user_ids, top_k, evaluation.seen_movie_ids
    )
    unknown = sum(
        1 for user_id in evaluation.user_ids if not recommender.knows_user(int(user_id))
    )
    return rankings, unknown


def rank_with_content(
    recommender: Any, evaluation: EvaluationSet, top_k: int
) -> tuple[dict[int, list[int]], int]:
    """Simulate onboarding: use each user's earliest highly rated training movies.

    This reproduces scenario 2 rather than scenario 3. The user is treated as if
    they had only just chosen a few favourites, which is the situation the
    content model is actually designed for.
    """
    rankings: dict[int, list[int]] = {}
    skipped = 0
    for user_id in evaluation.user_ids:
        user = int(user_id)
        selections = evaluation.onboarding_movie_ids.get(user)
        if not selections:
            skipped += 1
            continue
        seen = evaluation.seen_movie_ids.get(user, np.empty(0, dtype="int64"))
        # Ask for enough headroom that removing the full training history still
        # leaves top_k genuinely unseen candidates.
        requested = min(top_k + len(seen), 45_000)
        result = recommender.recommend(
            selected_movie_ids=[int(value) for value in selections],
            selected_genres=[],
            top_k=requested,
        )
        if result.fallback_used:
            skipped += 1
            continue
        seen_set = set(seen.tolist())
        ranked = [
            int(item["movie_id"])
            for item in result.recommendations
            if int(item["movie_id"]) not in seen_set
        ]
        rankings[user] = ranked[:top_k]
    return rankings, skipped
