"""Scenario 2 onboarding recommendations from selected movies and genres only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from src.data.config import load_config, output_dir
from src.recommenders.guest import GuestRecommender


@dataclass(frozen=True)
class ContentRecommendationResult:
    """Deterministic recommendations plus validation/fallback context."""

    recommendations: list[dict[str, Any]]
    warnings: list[str]
    fallback_used: bool


class ContentBasedRecommender:
    """Local TF-IDF recommender for first-login onboarding preferences."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        artifacts_dir = output_dir(config, "artifacts_dir", create=False)
        features_dir = output_dir(config, "features_dir", create=False)
        processed_dir = output_dir(config, "processed_dir", create=False)
        self.vectorizer = joblib.load(artifacts_dir / "vectorizer.joblib")
        self.matrix = sparse.load_npz(artifacts_dir / "movie_matrix.npz").tocsr()
        self.index = pd.read_parquet(artifacts_dir / "movie_index.parquet")
        self.features = pd.read_parquet(
            features_dir / "movie_content_features.parquet"
        )
        metadata = pd.read_parquet(
            processed_dir / "movies_clean.parquet",
            columns=["movie_id", "title", "genres"],
        )
        self.catalog = self.index.merge(metadata, on="movie_id", how="left")
        self.row_by_movie = {
            int(movie_id): int(row_index)
            for row_index, movie_id in zip(
                self.index["row_index"], self.index["movie_id"], strict=True
            )
        }
        genre_values = {
            str(genre)
            for values in metadata["genres"]
            for genre in (values.tolist() if hasattr(values, "tolist") else values)
        }
        self.genre_by_casefold = {value.casefold(): value for value in genre_values}
        self.guest = GuestRecommender(config)

    @staticmethod
    def _genre_token(genre: str) -> str:
        import re

        slug = re.sub(r"[^\w\s]", " ", genre.lower())
        slug = re.sub(r"\s+", "_", slug.strip())
        return f"genre_{slug}"

    def _fallback(
        self, top_k: int, warning_messages: list[str]
    ) -> ContentRecommendationResult:
        ranking = self.guest.get_guest_recommendations(top_k=top_k)
        titles = self.catalog[["movie_id", "title", "genres"]]
        ranking = ranking.merge(titles, on="movie_id", how="left")
        records = [
            {
                "rank": int(row.rank),
                "movie_id": int(row.movie_id),
                "title": str(row.title),
                "genres": (
                    row.genres.tolist()
                    if hasattr(row.genres, "tolist")
                    else list(row.genres)
                ),
                "score": float(row.score),
                "method": "fallback_top_rated",
            }
            for row in ranking.itertuples(index=False)
        ]
        return ContentRecommendationResult(records, warning_messages, True)

    def recommend(
        self,
        selected_movie_ids: list[Any] | None,
        selected_genres: list[Any] | None,
        top_k: int = 20,
    ) -> ContentRecommendationResult:
        """Validate onboarding input, build a preference profile, and rank movies."""
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        warnings: list[str] = []
        if selected_movie_ids is None:
            selected_movie_ids = []
        if selected_genres is None:
            selected_genres = []
        if not isinstance(selected_movie_ids, list):
            raise TypeError("selected_movie_ids must be a list")
        if not isinstance(selected_genres, list):
            raise TypeError("selected_genres must be a list")

        valid_movies: list[int] = []
        for raw_id in selected_movie_ids:
            try:
                movie_id = int(raw_id)
            except (TypeError, ValueError):
                warnings.append(f"Ignored invalid movie ID {raw_id!r}.")
                continue
            if movie_id not in self.row_by_movie:
                warnings.append(f"Ignored unknown movie ID {raw_id!r}.")
                continue
            if movie_id not in valid_movies:
                valid_movies.append(movie_id)

        valid_genres: list[str] = []
        for raw_genre in selected_genres:
            genre = str(raw_genre).strip()
            canonical = self.genre_by_casefold.get(genre.casefold())
            if not canonical:
                warnings.append(f"Ignored unknown genre {raw_genre!r}.")
                continue
            if canonical not in valid_genres:
                valid_genres.append(canonical)

        if not valid_movies and not valid_genres:
            warnings.append(
                "No usable onboarding preferences; returned the guest top-rated fallback."
            )
            return self._fallback(top_k, warnings)

        movie_profile = None
        if valid_movies:
            rows = [self.row_by_movie[movie_id] for movie_id in valid_movies]
            movie_profile = sparse.csr_matrix(self.matrix[rows].mean(axis=0))
        genre_profile = None
        if valid_genres:
            genre_text = " ".join(self._genre_token(genre) for genre in valid_genres)
            genre_profile = self.vectorizer.transform([genre_text]).tocsr()

        if movie_profile is not None and genre_profile is not None:
            settings = self.config["content_features"]
            movie_weight = float(settings["onboarding_movie_weight"])
            genre_weight = float(settings["onboarding_genre_weight"])
            total = movie_weight + genre_weight
            profile = (
                movie_profile * (movie_weight / total)
                + genre_profile * (genre_weight / total)
            )
        else:
            profile = movie_profile if movie_profile is not None else genre_profile
        if profile is None or profile.nnz == 0:
            warnings.append(
                "Usable IDs/genres produced no content terms; returned the guest top-rated fallback."
            )
            return self._fallback(top_k, warnings)

        profile = normalize(profile, norm="l2")
        scores = (self.matrix @ profile.T).toarray().ravel()
        candidates = self.catalog.copy()
        candidates["score"] = scores[candidates["row_index"].to_numpy()]
        candidates = candidates.loc[
            ~candidates["movie_id"].isin(valid_movies) & candidates["score"].gt(0)
        ]
        candidates = candidates.sort_values(
            ["score", "movie_id"],
            ascending=[False, True],
            kind="mergesort",
        ).drop_duplicates("movie_id").head(top_k)
        if candidates.empty:
            warnings.append(
                "No positive-similarity unseen movies; returned the guest top-rated fallback."
            )
            return self._fallback(top_k, warnings)
        records = []
        for rank, row in enumerate(candidates.itertuples(index=False), start=1):
            genres = (
                row.genres.tolist()
                if hasattr(row.genres, "tolist")
                else list(row.genres)
            )
            records.append(
                {
                    "rank": rank,
                    "movie_id": int(row.movie_id),
                    "title": str(row.title),
                    "genres": genres,
                    "score": float(row.score),
                    "method": "content_tfidf",
                }
            )
        return ContentRecommendationResult(records, warnings, False)


def get_onboarding_recommendations(
    payload: dict[str, Any],
    config_path: str | Path = "configs/data_pipeline.yaml",
) -> ContentRecommendationResult:
    """Execute the documented onboarding input contract."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")
    return ContentBasedRecommender(load_config(config_path)).recommend(
        selected_movie_ids=payload.get("selected_movie_ids", []),
        selected_genres=payload.get("selected_genres", []),
        top_k=payload.get("top_k", 20),
    )

