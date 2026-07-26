"""Collaborative filtering with implicit-feedback ALS.

The dataset holds explicit 0.5-5.0 ratings, but the production system collects
behavioural signals, so the model is trained as an implicit-feedback recommender.
Ratings are translated into a preference/confidence pair rather than used as
regression targets:

    rating >= 4.0   positive, confidence grows with the rating
    rating 3.0-3.5  neutral, excluded by default (configurable)
    rating <= 2.5   negative, never entered into the matrix
    no rating       unknown, and explicitly not treated as a dislike

Negative feedback is kept as a per-user exclusion list instead of a negative
confidence, because `implicit` has no representation for "confidently disliked".
Filtering happens at candidate selection, which is where the spec puts it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

USER_INDEX_FILE = "user_index.parquet"
MOVIE_INDEX_FILE = "movie_index.parquet"
USER_FACTORS_FILE = "user_factors.npy"
ITEM_FACTORS_FILE = "item_factors.npy"
CONFIG_FILE = "config.json"

# Scoring 45k items for many users at once would allocate gigabytes, so the
# score matrix is materialised in slices.
SCORING_CHUNK_USERS = 512


@dataclass
class TrainingMatrix:
    """Confidence matrix plus the index files that make its rows interpretable."""

    matrix: sparse.csr_matrix
    user_ids: np.ndarray
    movie_ids: np.ndarray
    stats: dict[str, Any]


def _settings(model_config: dict[str, Any]) -> dict[str, Any]:
    return model_config["collaborative"]


def confidence_from_rating(
    ratings: np.ndarray, model_config: dict[str, Any]
) -> np.ndarray:
    """Map ratings to ALS confidence values `c = 1 + alpha * w`.

    `implicit` stores the full confidence in the matrix and treats every stored
    entry as preference 1, so the `1 +` term belongs here rather than inside the
    library.
    """
    settings = _settings(model_config)
    positive_threshold = float(settings["positive_rating_threshold"])
    alpha = float(settings["alpha"])
    maximum_rating = 5.0

    span = maximum_rating - positive_threshold + 0.5
    weights = np.clip((ratings - (positive_threshold - 0.5)) / span, 0.0, 1.0)
    return (1.0 + alpha * weights).astype("float32")


def build_training_matrix(
    train: pd.DataFrame, model_config: dict[str, Any]
) -> TrainingMatrix:
    """Turn cleaned training interactions into a user x item confidence matrix."""
    settings = _settings(model_config)
    positive_threshold = float(settings["positive_rating_threshold"])
    neutral_policy = str(settings["neutral_rating_policy"])
    negative_threshold = float(settings["negative_rating_threshold"])
    minimum_per_movie = int(settings["min_positive_interactions_per_movie"])

    if neutral_policy not in {"exclude", "low_confidence"}:
        raise ValueError(
            "neutral_rating_policy must be 'exclude' or 'low_confidence', "
            f"got {neutral_policy!r}"
        )

    total_rows = len(train)
    positive = train.loc[train["interaction_value"] >= positive_threshold]
    neutral = train.loc[
        (train["interaction_value"] > negative_threshold)
        & (train["interaction_value"] < positive_threshold)
    ]
    negative_rows = int((train["interaction_value"] <= negative_threshold).sum())

    if neutral_policy == "low_confidence":
        neutral = neutral.copy()
        alpha = float(settings["alpha"])
        weight = float(settings["neutral_confidence_weight"])
        neutral["_confidence"] = np.float32(1.0 + alpha * weight)
        positive = positive.copy()
        positive["_confidence"] = confidence_from_rating(
            positive["interaction_value"].to_numpy(dtype="float32"), model_config
        )
        used = pd.concat([positive, neutral], ignore_index=True)
    else:
        used = positive.copy()
        used["_confidence"] = confidence_from_rating(
            used["interaction_value"].to_numpy(dtype="float32"), model_config
        )

    movie_counts = used["movie_id"].value_counts()
    kept_movies = movie_counts[movie_counts >= minimum_per_movie].index
    dropped_movies = int(len(movie_counts) - len(kept_movies))
    used = used.loc[used["movie_id"].isin(pd.Index(kept_movies))]

    if used.empty:
        raise ValueError("No interactions remain after filtering; check thresholds.")

    # Sorting first makes factorize deterministic, so the saved index files are
    # identical across runs with the same input.
    used = used.sort_values(["user_id", "movie_id"], kind="mergesort")
    user_codes, user_ids = pd.factorize(used["user_id"], sort=True)
    movie_codes, movie_ids = pd.factorize(used["movie_id"], sort=True)

    matrix = sparse.csr_matrix(
        (
            used["_confidence"].to_numpy(dtype="float32"),
            (user_codes, movie_codes),
        ),
        shape=(len(user_ids), len(movie_ids)),
        dtype="float32",
    )
    matrix.sum_duplicates()

    stats = {
        "train_rows": total_rows,
        "positive_rows": int(len(positive)),
        "neutral_rows_available": int(len(neutral)),
        "neutral_rows_used": int(len(neutral)) if neutral_policy == "low_confidence" else 0,
        "negative_rows_excluded": negative_rows,
        "matrix_rows": int(matrix.shape[0]),
        "matrix_columns": int(matrix.shape[1]),
        "matrix_nonzero": int(matrix.nnz),
        "movies_dropped_below_minimum": dropped_movies,
        "minimum_positive_interactions_per_movie": minimum_per_movie,
        "neutral_rating_policy": neutral_policy,
        "sparsity": 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
    }
    return TrainingMatrix(
        matrix=matrix,
        user_ids=np.asarray(user_ids, dtype="int64"),
        movie_ids=np.asarray(movie_ids, dtype="int64"),
        stats=stats,
    )


def negative_movie_ids(
    train: pd.DataFrame, model_config: dict[str, Any]
) -> dict[int, np.ndarray]:
    """Per-user movies rated at or below the negative threshold."""
    threshold = float(_settings(model_config)["negative_rating_threshold"])
    disliked = train.loc[train["interaction_value"] <= threshold]
    return {
        int(user): group.to_numpy(dtype="int64")
        for user, group in disliked.groupby("user_id", sort=False)["movie_id"]
    }


def train_als(matrix: sparse.csr_matrix, model_config: dict[str, Any]) -> Any:
    """Fit implicit ALS on the confidence matrix."""
    from implicit.als import AlternatingLeastSquares

    settings = _settings(model_config)
    model = AlternatingLeastSquares(
        factors=int(settings["factors"]),
        regularization=float(settings["regularization"]),
        iterations=int(settings["iterations"]),
        random_state=int(settings["random_state"]),
        use_gpu=bool(settings["use_gpu"]),
    )
    model.fit(matrix)
    return model


class CollaborativeRecommender:
    """Serve recommendations from saved ALS factors.

    Only the factor matrices and the two index files are needed at serve time;
    the training data is not loaded. Scores are computed directly rather than
    through `implicit.recommend` so that an arbitrary per-user exclusion set can
    be applied exactly, instead of over-requesting and hoping the list is long
    enough.
    """

    def __init__(
        self,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        user_ids: np.ndarray,
        movie_ids: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.user_factors = np.ascontiguousarray(user_factors, dtype="float32")
        self.item_factors = np.ascontiguousarray(item_factors, dtype="float32")
        self.user_ids = np.asarray(user_ids, dtype="int64")
        self.movie_ids = np.asarray(movie_ids, dtype="int64")
        self.metadata = metadata or {}
        self.row_by_user = {
            int(user_id): row for row, user_id in enumerate(self.user_ids)
        }

    @classmethod
    def load(cls, directory: str | Path) -> CollaborativeRecommender:
        path = Path(directory)
        user_index = pd.read_parquet(path / USER_INDEX_FILE)
        movie_index = pd.read_parquet(path / MOVIE_INDEX_FILE)
        metadata_path = path / CONFIG_FILE
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        return cls(
            user_factors=np.load(path / USER_FACTORS_FILE),
            item_factors=np.load(path / ITEM_FACTORS_FILE),
            user_ids=user_index["user_id"].to_numpy(dtype="int64"),
            movie_ids=movie_index["movie_id"].to_numpy(dtype="int64"),
            metadata=metadata,
        )

    def save(self, directory: str | Path, metadata: dict[str, Any]) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / USER_FACTORS_FILE, self.user_factors)
        np.save(path / ITEM_FACTORS_FILE, self.item_factors)
        pd.DataFrame(
            {
                "row_index": range(len(self.user_ids)),
                "user_id": self.user_ids,
            }
        ).to_parquet(path / USER_INDEX_FILE, index=False, compression="snappy")
        pd.DataFrame(
            {
                "row_index": range(len(self.movie_ids)),
                "movie_id": self.movie_ids,
            }
        ).to_parquet(path / MOVIE_INDEX_FILE, index=False, compression="snappy")
        (path / CONFIG_FILE).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def knows_user(self, user_id: int) -> bool:
        return int(user_id) in self.row_by_user

    def recommend_batch(
        self,
        user_ids: np.ndarray | list[int],
        top_n: int,
        exclude: dict[int, np.ndarray] | None = None,
    ) -> dict[int, list[int]]:
        """Rank items for many users, dropping each user's excluded movies.

        Users absent from the ALS index are omitted from the result rather than
        given a fabricated list; deciding what to show them belongs to the
        scenario router, not to this model.
        """
        if top_n <= 0:
            raise ValueError("top_n must be positive")

        exclude = exclude or {}
        requested = [int(value) for value in user_ids]
        known = [value for value in requested if value in self.row_by_user]
        rankings: dict[int, list[int]] = {}
        # Position lookup so an excluded movie_id becomes a column index.
        column_by_movie = {
            int(movie_id): column for column, movie_id in enumerate(self.movie_ids)
        }

        for start in range(0, len(known), SCORING_CHUNK_USERS):
            batch = known[start : start + SCORING_CHUNK_USERS]
            rows = np.fromiter(
                (self.row_by_user[value] for value in batch),
                dtype="int64",
                count=len(batch),
            )
            scores = self.user_factors[rows] @ self.item_factors.T

            for offset, user_id in enumerate(batch):
                blocked = exclude.get(user_id)
                if blocked is not None and len(blocked):
                    columns = [
                        column_by_movie[int(movie_id)]
                        for movie_id in blocked
                        if int(movie_id) in column_by_movie
                    ]
                    if columns:
                        scores[offset, columns] = -np.inf
                row = scores[offset]
                take = min(top_n, row.shape[0])
                candidates = np.argpartition(-row, take - 1)[:take]
                ordered = candidates[np.argsort(-row[candidates], kind="stable")]
                ordered = ordered[np.isfinite(row[ordered])]
                rankings[user_id] = self.movie_ids[ordered].tolist()

        return rankings

    def recommend(
        self,
        user_id: int,
        top_n: int = 100,
        exclude: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Single-user ranking in the shape the serving layer consumes."""
        rankings = self.recommend_batch(
            [int(user_id)],
            top_n,
            {int(user_id): exclude} if exclude is not None else None,
        )
        ranked = rankings.get(int(user_id), [])
        row = self.row_by_user.get(int(user_id))
        if row is None:
            return []
        scores = self.user_factors[row] @ self.item_factors.T
        column_by_movie = {
            int(movie_id): column for column, movie_id in enumerate(self.movie_ids)
        }
        return [
            {
                "movie_id": int(movie_id),
                "score": float(scores[column_by_movie[int(movie_id)]]),
                "reason_code": "similar_users",
                "reason_context": {},
            }
            for movie_id in ranked
        ]
