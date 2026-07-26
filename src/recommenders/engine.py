"""Recommendation engine: scenario routing, fusion, backfill, and the API contract.

This is the only component that decides which model answers a request. Backend
supplies a `scenario_hint` because only it knows whether onboarding finished and
how many valid interactions a user has; the engine may downgrade that hint
because only it knows whether the artifacts can actually serve it. The two facts
travel in separate fields (`scenario_hint` in, `scenario_applied` out) so neither
side has to guess what the other meant.

Three rules are enforced here rather than left to callers:

* The response never contains a title, overview, or poster. Backend joins those
  from its own catalogue.
* Every returned `movie_id` is checked against the serving catalogue before the
  response is built, because an index-mapping mistake produces plausible-looking
  wrong movies rather than an error.
* A request never comes back empty while the catalogue still has candidates.
  Personalised results are topped up from weaker sources, not replaced by them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from src.data.config import output_dir, project_path
from src.features.similarity import SIMILAR_MOVIES_FILE, SimilarMovieLookup
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid_ranking import HybridRanker
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.guest import GuestRecommender

SERVICE_NAME = "hybrid_recommender"
SERVICE_VERSION = "1.0.0"

SCENARIO_GUEST = "guest"
SCENARIO_ONBOARDING = "onboarding_user"
SCENARIO_RETURNING = "returning_user"
VALID_SCENARIOS = (SCENARIO_GUEST, SCENARIO_ONBOARDING, SCENARIO_RETURNING)


class RecommendationRequestError(ValueError):
    """Raised for malformed input, which is the only case answered with a 4xx."""


@dataclass
class RecommendationResponse:
    """The engine's output, shaped exactly as the API contract specifies."""

    model_name: str
    model_version: str
    scenario_applied: str
    recommendation_type: str
    fallback_used: bool
    fallback_level: str
    generated_at: str
    recommendations: list[dict[str, Any]]
    artifact_versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "scenario_applied": self.scenario_applied,
            "recommendation_type": self.recommendation_type,
            "fallback_used": self.fallback_used,
            "fallback_level": self.fallback_level,
            "generated_at": self.generated_at,
            "artifact_versions": self.artifact_versions,
            "recommendations": self.recommendations,
        }


def _as_int_list(values: Iterable[Any] | None) -> list[int]:
    if not values:
        return []
    result: list[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


class RecommendationEngine:
    """Load every artifact once at start-up and answer requests from memory.

    Artifacts are read when the process starts and kept resident. Re-reading S3
    or disk per request would add a network round trip to every page load, which
    is the cost the batch-first architecture exists to avoid.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        model_config: Mapping[str, Any],
        als_version: str | None = None,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.candidates = model_config["candidates"]

        self.guest = GuestRecommender(config)
        self.content = ContentBasedRecommender(config)
        self.hybrid = HybridRanker(model_config)

        # Optional: "Because you watched" degrades to unavailable rather than
        # failing the whole engine if the table has not been built yet.
        self.similar_movies: SimilarMovieLookup | None = None
        artifacts_dir = output_dir(config, "artifacts_dir", create=False)
        if (artifacts_dir / SIMILAR_MOVIES_FILE).exists():
            self.similar_movies = SimilarMovieLookup(config)

        self.collaborative: CollaborativeRecommender | None = None
        self.artifact_versions: dict[str, str] = {"content_based": "local"}
        resolved = self._resolve_als_directory(als_version)
        if resolved is not None:
            self.collaborative = CollaborativeRecommender.load(resolved)
            self.artifact_versions["collaborative"] = resolved.name

        serving_dir = output_dir(config, "serving_dir", create=False)
        serving = pd.read_parquet(
            serving_dir / "movies_serving.parquet", columns=["movie_id", "genres"]
        )
        self.catalogue: set[int] = {
            int(value) for value in serving["movie_id"].to_numpy()
        }
        self.genres_by_movie: dict[int, list[str]] = {
            int(row.movie_id): (
                row.genres.tolist()
                if hasattr(row.genres, "tolist")
                else list(row.genres)
            )
            for row in serving.itertuples(index=False)
        }

    def _resolve_als_directory(self, requested: str | None) -> Path | None:
        base = project_path(self.config, "artifacts/collaborative")
        if requested:
            candidate = base / requested
            return candidate if candidate.exists() else None
        pointer = project_path(self.config, "artifacts") / "LATEST.json"
        if not pointer.exists():
            return None
        version = json.loads(pointer.read_text(encoding="utf-8")).get("collaborative")
        if not version:
            return None
        candidate = base / version
        return candidate if candidate.exists() else None

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _popularity_candidates(self, top_n: int) -> list[int]:
        ranking = self.guest.get_guest_recommendations(top_k=top_n)
        return [int(value) for value in ranking["movie_id"].to_numpy()]

    def _genre_candidates(self, genre: str | None, top_n: int) -> list[int]:
        if not genre:
            return []
        ranking = self.guest.get_guest_recommendations(genre=genre, top_k=top_n)
        return [int(value) for value in ranking["movie_id"].to_numpy()]

    def _content_candidates(
        self,
        selected_movie_ids: Sequence[int],
        selected_genres: Sequence[str],
        top_n: int,
    ) -> list[int]:
        """Content candidates, or nothing if the recommender had to fall back.

        A fallback result is the guest ranking wearing a content label. Passing it
        into the fusion would count popularity twice and inflate its influence.
        """
        if not selected_movie_ids and not selected_genres:
            return []
        result = self.content.recommend(
            selected_movie_ids=list(selected_movie_ids),
            selected_genres=list(selected_genres),
            top_k=top_n,
        )
        if result.fallback_used:
            return []
        return [int(item["movie_id"]) for item in result.recommendations]

    def _collaborative_candidates(
        self, user_id: int | None, top_n: int, exclude: Sequence[int]
    ) -> list[int]:
        if self.collaborative is None or user_id is None:
            return []
        if not self.collaborative.knows_user(user_id):
            return []
        rankings = self.collaborative.recommend_batch(
            [user_id],
            top_n,
            {user_id: np.asarray(exclude, dtype="int64")} if exclude else None,
        )
        return rankings.get(user_id, [])

    def _recent_similarity(
        self, recent_movie_ids: Sequence[int], candidates: Sequence[int]
    ) -> dict[int, float]:
        """Cosine similarity between the candidates and recently engaged movies.

        Computed from the request rather than a stored profile table, so the
        engine has no dependency on a persistence layer the project has not
        built yet.
        """
        rows = [
            self.content.row_by_movie[movie_id]
            for movie_id in recent_movie_ids
            if movie_id in self.content.row_by_movie
        ]
        if not rows or not candidates:
            return {}
        profile = normalize(
            sparse.csr_matrix(self.content.matrix[rows].mean(axis=0)), norm="l2"
        )
        candidate_rows = [
            (movie_id, self.content.row_by_movie[movie_id])
            for movie_id in candidates
            if movie_id in self.content.row_by_movie
        ]
        if not candidate_rows:
            return {}
        matrix = self.content.matrix[[row for _, row in candidate_rows]]
        scores = (matrix @ profile.T).toarray().ravel()
        return {
            movie_id: float(score)
            for (movie_id, _), score in zip(candidate_rows, scores, strict=True)
        }

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def recommend(self, request: Mapping[str, Any]) -> RecommendationResponse:
        """Route one request to the right model and guarantee a usable answer."""
        hint = str(request.get("scenario_hint") or SCENARIO_GUEST)
        if hint not in VALID_SCENARIOS:
            raise RecommendationRequestError(
                f"scenario_hint must be one of {VALID_SCENARIOS}, got {hint!r}"
            )

        limit = int(request.get("limit") or self.candidates["api_default_limit"])
        maximum = int(self.candidates["api_max_limit"])
        if limit <= 0:
            raise RecommendationRequestError("limit must be positive")
        limit = min(limit, maximum)

        raw_user = request.get("user_id")
        user_id: int | None
        if raw_user in (None, ""):
            user_id = None
        else:
            try:
                user_id = int(raw_user)
            except (TypeError, ValueError):
                raise RecommendationRequestError(
                    "user_id must be an integer identifier or null"
                ) from None

        selected_movie_ids = _as_int_list(request.get("selected_movie_ids"))
        selected_genres = [
            str(value) for value in (request.get("selected_genres") or [])
        ]
        recent = request.get("recent_interactions") or []
        recent_movie_ids = _as_int_list(
            item.get("movie_id") for item in recent if isinstance(item, Mapping)
        )
        max_recent = int(self.model_config["hybrid"]["max_recent_interactions"])
        recent_movie_ids = recent_movie_ids[:max_recent]

        excluded = set(_as_int_list(request.get("exclude_movie_ids")))
        excluded.update(recent_movie_ids)

        interaction_count = int(request.get("valid_interaction_count_90d") or 0)
        onboarding_completed = bool(request.get("onboarding_completed"))

        # Backend's hint is a claim about the account; the engine still verifies
        # it can be served before accepting it.
        if hint == SCENARIO_RETURNING and (
            user_id is None
            or self.collaborative is None
            or not self.collaborative.knows_user(user_id)
        ):
            hint = SCENARIO_ONBOARDING
        if hint == SCENARIO_ONBOARDING and not (
            selected_movie_ids or selected_genres or recent_movie_ids
        ):
            hint = SCENARIO_GUEST
        if hint != SCENARIO_GUEST and user_id is None:
            hint = SCENARIO_GUEST
        if hint == SCENARIO_ONBOARDING and not onboarding_completed and not (
            selected_movie_ids or selected_genres
        ):
            hint = SCENARIO_GUEST

        preferred_genre = selected_genres[0] if selected_genres else None

        if hint == SCENARIO_GUEST:
            return self._respond_guest(limit, excluded)
        if hint == SCENARIO_ONBOARDING:
            return self._respond_onboarding(
                selected_movie_ids, selected_genres, preferred_genre, limit, excluded
            )
        return self._respond_returning(
            user_id,
            selected_genres,
            recent_movie_ids,
            preferred_genre,
            interaction_count,
            limit,
            excluded,
        )

    def _respond_guest(
        self, limit: int, excluded: set[int]
    ) -> RecommendationResponse:
        ranked = [
            movie_id
            for movie_id in self._popularity_candidates(
                limit + len(excluded) + int(self.candidates["popularity_topn"])
            )
            if movie_id not in excluded
        ][:limit]
        items = [
            {
                "movie_id": movie_id,
                "score": round(1.0 / (position + 1), 6),
                "reason_code": "top_rated",
                "reason_context": {},
            }
            for position, movie_id in enumerate(ranked, start=1)
        ]
        return self._build_response(
            SCENARIO_GUEST, "top_rated", items, fallback_level="none"
        )

    def _respond_onboarding(
        self,
        selected_movie_ids: Sequence[int],
        selected_genres: Sequence[str],
        preferred_genre: str | None,
        limit: int,
        excluded: set[int],
    ) -> RecommendationResponse:
        content = self._content_candidates(
            selected_movie_ids,
            selected_genres,
            int(self.candidates["content_topn"]),
        )
        blocked = set(excluded) | set(selected_movie_ids)
        ranked = [movie_id for movie_id in content if movie_id not in blocked]

        fallback_level = "none"
        if len(ranked) < limit:
            seen = set(ranked)
            for level, source in (
                (
                    "genre",
                    self._genre_candidates(
                        preferred_genre, int(self.candidates["popularity_topn"])
                    ),
                ),
                (
                    "global",
                    self._popularity_candidates(
                        int(self.candidates["popularity_topn"])
                    ),
                ),
            ):
                for movie_id in source:
                    if len(ranked) >= limit:
                        break
                    if movie_id in seen or movie_id in blocked:
                        continue
                    ranked.append(movie_id)
                    seen.add(movie_id)
                    fallback_level = level
                if len(ranked) >= limit:
                    break

        content_set = set(content)
        items = []
        for position, movie_id in enumerate(ranked[:limit], start=1):
            if movie_id in content_set:
                reason_code = "similar_to_onboarding"
                context: dict[str, Any] = (
                    {"source_movie_id": str(selected_movie_ids[0])}
                    if selected_movie_ids
                    else {"genre": preferred_genre} if preferred_genre else {}
                )
                if not selected_movie_ids and preferred_genre:
                    reason_code = "genre_match"
            elif fallback_level == "genre" and preferred_genre:
                reason_code = "top_rated_genre"
                context = {"genre": preferred_genre}
            else:
                reason_code = "top_rated"
                context = {}
            items.append(
                {
                    "movie_id": movie_id,
                    "score": round(1.0 / (position + 1), 6),
                    "reason_code": reason_code,
                    "reason_context": context,
                }
            )
        return self._build_response(
            SCENARIO_ONBOARDING, "content_based", items, fallback_level
        )

    def _respond_returning(
        self,
        user_id: int | None,
        selected_genres: Sequence[str],
        recent_movie_ids: Sequence[int],
        preferred_genre: str | None,
        interaction_count: int,
        limit: int,
        excluded: set[int],
    ) -> RecommendationResponse:
        keep = int(self.candidates["hybrid_keep"])
        collaborative = self._collaborative_candidates(
            user_id, int(self.candidates["cf_topn"]), sorted(excluded)
        )
        content = self._content_candidates(
            recent_movie_ids, selected_genres, int(self.candidates["content_topn"])
        )
        popularity = self._popularity_candidates(
            int(self.candidates["popularity_topn"])
        )
        genre_fallback = self._genre_candidates(
            preferred_genre, int(self.candidates["popularity_topn"])
        )

        pool = list(dict.fromkeys([*collaborative, *content, *popularity]))
        similarity = self._recent_similarity(recent_movie_ids, pool)

        ranked_items, level, _ = self.hybrid.rank(
            collaborative=collaborative,
            content=content,
            popularity=popularity,
            valid_interaction_count=interaction_count,
            limit=min(limit, keep),
            exclude_movie_ids=excluded,
            genres_by_movie=self.genres_by_movie,
            recent_similarity=similarity,
            genre_fallback=genre_fallback,
            preferred_genre=preferred_genre,
        )
        items = [
            {
                "movie_id": item.movie_id,
                "score": item.score,
                "reason_code": item.reason_code,
                "reason_context": item.reason_context,
            }
            for item in ranked_items
        ]
        return self._build_response(SCENARIO_RETURNING, "hybrid", items, level)

    def because_you_watched(
        self,
        movie_ids: Sequence[int],
        limit: int | None = None,
        exclude_movie_ids: Iterable[int] = (),
    ) -> RecommendationResponse:
        """Serve the "Because you watched X" row from the precomputed table.

        This is a separate homepage row rather than an input to the hybrid fusion,
        so it stays a plain lookup: no model call, no matrix multiply, and each
        result names the specific film that produced it.
        """
        limit = int(limit or self.candidates["api_default_limit"])
        limit = min(max(limit, 1), int(self.candidates["api_max_limit"]))
        seeds = _as_int_list(movie_ids)
        if not seeds:
            raise RecommendationRequestError(
                "because_you_watched requires at least one movie_id"
            )
        if self.similar_movies is None:
            raise RecommendationRequestError(
                "similar_movies_top50.parquet chưa được dựng; chạy "
                "scripts/build_similar_movies.py trước."
            )

        blocked = set(_as_int_list(exclude_movie_ids))
        pairs = self.similar_movies.similar_to_any(seeds, top_n=limit + len(blocked))
        items = []
        for position, (movie_id, seed) in enumerate(pairs, start=1):
            if movie_id in blocked:
                continue
            items.append(
                {
                    "movie_id": movie_id,
                    "score": round(1.0 / (position + 1), 6),
                    "reason_code": "similar_to_watched_movies",
                    "reason_context": {"source_movie_id": str(seed)},
                }
            )
            if len(items) >= limit:
                break

        level = "none" if items else "global"
        if not items:
            for position, movie_id in enumerate(
                self._popularity_candidates(limit + len(blocked)), start=1
            ):
                if movie_id in blocked:
                    continue
                items.append(
                    {
                        "movie_id": movie_id,
                        "score": round(1.0 / (position + 1), 6),
                        "reason_code": "top_rated",
                        "reason_context": {},
                    }
                )
                if len(items) >= limit:
                    break

        return self._build_response(
            SCENARIO_RETURNING, "because_you_watched", items, level
        )

    # ------------------------------------------------------------------
    # Response assembly
    # ------------------------------------------------------------------

    def _build_response(
        self,
        scenario_applied: str,
        recommendation_type: str,
        items: list[dict[str, Any]],
        fallback_level: str,
    ) -> RecommendationResponse:
        # Scores from the three paths live on different scales; only their order
        # is meaningful, so they are rescaled to a common (0, 1] here. Doing it
        # in one place keeps every scenario consistent for the frontend.
        if items:
            highest = max(float(item["score"]) for item in items)
            if highest > 0:
                for item in items:
                    item["score"] = round(float(item["score"]) / highest, 6)

        unknown = [
            item["movie_id"]
            for item in items
            if int(item["movie_id"]) not in self.catalogue
        ]
        if unknown:
            # An index-mapping error returns real-looking IDs for the wrong
            # films and never raises on its own, so it is caught here instead.
            raise RuntimeError(
                "Model returned movie_id values missing from the serving "
                f"catalogue: {unknown[:5]}"
            )
        return RecommendationResponse(
            model_name=SERVICE_NAME,
            model_version=SERVICE_VERSION,
            scenario_applied=scenario_applied,
            recommendation_type=recommendation_type,
            fallback_used=fallback_level != "none",
            fallback_level=fallback_level,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            recommendations=items,
            artifact_versions=dict(self.artifact_versions),
        )
