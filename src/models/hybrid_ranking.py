"""Hybrid ranking layer: fuse candidates from popularity, content, and ALS.

The three sources produce scores on incompatible scales - weighted rating on
0-5, cosine similarity on 0-1, and an unbounded ALS dot product - so adding them
directly lets whichever source happens to have the largest range dominate.
Weighted Reciprocal Rank Fusion sidesteps the problem entirely by discarding the
scores and using only positions:

    rrf_score(m) = sum_s  w_s * 1 / (k + rank_s(m))

Source weights are not fixed. A user five interactions past the cold-start
threshold has a weak collaborative signal, while a user with two hundred has a
strong one, so `w_cf` ramps up with history and `w_cb` gives way as it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

SOURCE_COLLABORATIVE = "cf"
SOURCE_CONTENT = "cb"
SOURCE_POPULARITY = "pop"

REASON_BY_SOURCE = {
    SOURCE_COLLABORATIVE: "similar_users",
    SOURCE_CONTENT: "similar_to_watched_movies",
    SOURCE_POPULARITY: "top_rated",
}


@dataclass(frozen=True)
class SourceWeights:
    """Per-source fusion weights derived from how much history a user has."""

    collaborative: float
    content: float
    popularity: float
    cf_confidence: float

    def as_mapping(self) -> dict[str, float]:
        return {
            SOURCE_COLLABORATIVE: self.collaborative,
            SOURCE_CONTENT: self.content,
            SOURCE_POPULARITY: self.popularity,
        }


@dataclass
class RankedItem:
    """One fused candidate plus the provenance the API contract requires."""

    movie_id: int
    score: float
    reason_code: str
    reason_context: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


def compute_source_weights(
    valid_interaction_count: int, hybrid_config: Mapping[str, Any]
) -> SourceWeights:
    """Ramp the collaborative weight up with history, content weight down.

    `full_cf_confidence_at` is the number of interactions at which collaborative
    filtering is trusted completely. Below it the ratio scales linearly, so the
    handover from content-based to collaborative is gradual rather than a jump at
    the scenario threshold.
    """
    full_confidence_at = int(hybrid_config["full_cf_confidence_at"])
    if full_confidence_at <= 0:
        raise ValueError("full_cf_confidence_at must be positive")

    floor = float(hybrid_config["w_content_floor"])
    if not 0.0 <= floor <= 1.0:
        raise ValueError("w_content_floor must be between 0 and 1")

    confidence = min(max(valid_interaction_count, 0) / full_confidence_at, 1.0)
    return SourceWeights(
        collaborative=confidence,
        # Starts at 1.0 for a user with no history and decays to the floor once
        # the collaborative signal is fully trusted.
        content=1.0 - (1.0 - floor) * confidence,
        popularity=float(hybrid_config["w_pop"]),
        cf_confidence=confidence,
    )


def reciprocal_rank_fusion(
    candidates: Mapping[str, Sequence[int]],
    weights: Mapping[str, float],
    rrf_k: int = 60,
) -> list[tuple[int, float, list[str]]]:
    """Merge ranked lists into one ordering, keeping track of contributing sources.

    A movie absent from a source contributes nothing from that source rather than
    a zero, so appearing in two lists genuinely beats appearing in one.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    totals: dict[int, float] = {}
    provenance: dict[int, list[str]] = {}
    for source, ranked in candidates.items():
        weight = float(weights.get(source, 0.0))
        if weight == 0.0:
            continue
        for position, movie_id in enumerate(ranked, start=1):
            key = int(movie_id)
            totals[key] = totals.get(key, 0.0) + weight / (rrf_k + position)
            provenance.setdefault(key, []).append(source)

    # Ties break on movie_id so the output is reproducible across runs.
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return [(movie_id, score, provenance[movie_id]) for movie_id, score in ordered]


def normalise_scores(scored: Sequence[tuple[int, float, list[str]]]) -> dict[int, float]:
    """Min-max the fusion scores onto 0-1.

    RRF scores sit around 1/60, while any similarity term added later lives on
    0-1. Without this rescaling the similarity term would swamp the fusion result
    instead of nudging it.
    """
    if not scored:
        return {}
    values = [score for _, score, _ in scored]
    lowest = min(values)
    highest = max(values)
    if highest == lowest:
        return {movie_id: 1.0 for movie_id, _, _ in scored}
    span = highest - lowest
    return {
        movie_id: (score - lowest) / span for movie_id, score, _ in scored
    }


def apply_recent_preference(
    scored: Sequence[tuple[int, float, list[str]]],
    similarity: Mapping[int, float] | None,
    weight: float,
) -> list[tuple[int, float, list[str]]]:
    """Nudge the ranking toward what the user engaged with most recently."""
    normalised = normalise_scores(scored)
    if not similarity or weight == 0.0:
        adjusted = [
            (movie_id, normalised[movie_id], sources)
            for movie_id, _, sources in scored
        ]
    else:
        adjusted = [
            (
                movie_id,
                normalised[movie_id] + weight * float(similarity.get(movie_id, 0.0)),
                sources,
            )
            for movie_id, _, sources in scored
        ]
    return sorted(adjusted, key=lambda item: (-item[1], item[0]))


def enforce_genre_diversity(
    ranked: Sequence[tuple[int, float, list[str]]],
    genres_by_movie: Mapping[int, Sequence[str]],
    maximum_per_genre: int,
    limit: int,
) -> list[tuple[int, float, list[str]]]:
    """Cap how many results share a primary genre.

    Without this a strong collaborative signal easily returns twenty variations of
    the same genre. Movies whose genre is unknown are never blocked, because
    penalising missing metadata would quietly hide part of the catalogue.
    """
    if maximum_per_genre <= 0:
        return list(ranked[:limit])

    counts: dict[str, int] = {}
    kept: list[tuple[int, float, list[str]]] = []
    deferred: list[tuple[int, float, list[str]]] = []

    for item in ranked:
        movie_id = item[0]
        genres = genres_by_movie.get(movie_id) or []
        primary = str(genres[0]) if len(genres) else None
        if primary is None:
            kept.append(item)
        elif counts.get(primary, 0) < maximum_per_genre:
            counts[primary] = counts.get(primary, 0) + 1
            kept.append(item)
        else:
            deferred.append(item)
        if len(kept) >= limit:
            return kept[:limit]

    # Diversity is a preference, not a reason to return a short list.
    for item in deferred:
        if len(kept) >= limit:
            break
        kept.append(item)
    return kept[:limit]


def backfill(
    ranked: Sequence[int],
    fallback_sources: Sequence[tuple[str, Sequence[int]]],
    limit: int,
    excluded: Iterable[int] = (),
) -> tuple[list[int], dict[int, str], str]:
    """Top the list up from weaker sources without discarding what is already there.

    The point of backfilling rather than replacing: four personalised results plus
    sixteen popular ones is a better answer than twenty popular ones. Returns the
    filled list, the source each backfilled movie came from, and the deepest
    fallback level reached.
    """
    blocked = {int(value) for value in excluded}
    filled = [int(value) for value in ranked if int(value) not in blocked]
    seen = set(filled)
    origin: dict[int, str] = {}
    level = "none"

    for name, source in fallback_sources:
        if len(filled) >= limit:
            break
        for movie_id in source:
            key = int(movie_id)
            if key in seen or key in blocked:
                continue
            filled.append(key)
            seen.add(key)
            origin[key] = name
            level = name
            if len(filled) >= limit:
                break
    return filled[:limit], origin, level


class HybridRanker:
    """Fuse candidate lists, apply business rules, and guarantee a full response."""

    def __init__(self, model_config: Mapping[str, Any]) -> None:
        self.hybrid_config = model_config["hybrid"]
        self.candidates_config = model_config["candidates"]

    def rank(
        self,
        collaborative: Sequence[int],
        content: Sequence[int],
        popularity: Sequence[int],
        valid_interaction_count: int,
        limit: int,
        exclude_movie_ids: Iterable[int] = (),
        genres_by_movie: Mapping[int, Sequence[str]] | None = None,
        recent_similarity: Mapping[int, float] | None = None,
        genre_fallback: Sequence[int] = (),
        preferred_genre: str | None = None,
    ) -> tuple[list[RankedItem], str, SourceWeights]:
        """Produce the final ordered list plus the fallback level that was needed."""
        if limit <= 0:
            raise ValueError("limit must be positive")

        blocked = {int(value) for value in exclude_movie_ids}
        weights = compute_source_weights(valid_interaction_count, self.hybrid_config)

        def allowed(values: Sequence[int]) -> list[int]:
            return [int(value) for value in values if int(value) not in blocked]

        candidates = {
            SOURCE_COLLABORATIVE: allowed(collaborative),
            SOURCE_CONTENT: allowed(content),
            SOURCE_POPULARITY: allowed(popularity),
        }

        fused = reciprocal_rank_fusion(
            candidates,
            weights.as_mapping(),
            rrf_k=int(self.hybrid_config["rrf_k"]),
        )
        adjusted = apply_recent_preference(
            fused,
            recent_similarity,
            float(self.hybrid_config["w_recent"]),
        )

        diversified = enforce_genre_diversity(
            adjusted,
            genres_by_movie or {},
            int(self.hybrid_config["max_per_genre_in_top20"]),
            limit,
        )

        score_by_movie = {movie_id: score for movie_id, score, _ in diversified}
        sources_by_movie = {movie_id: sources for movie_id, _, sources in diversified}

        filled, origin, level = backfill(
            [movie_id for movie_id, _, _ in diversified],
            [
                ("content", allowed(content)),
                ("genre", allowed(genre_fallback)),
                ("global", allowed(popularity)),
            ],
            limit,
            excluded=blocked,
        )

        items: list[RankedItem] = []
        for position, movie_id in enumerate(filled, start=1):
            sources = sources_by_movie.get(movie_id, [])
            if movie_id in origin:
                source_name = origin[movie_id]
                reason = (
                    "top_rated_genre" if source_name == "genre" else
                    "similar_to_watched_movies" if source_name == "content" else
                    "top_rated"
                )
                context: dict[str, Any] = {}
                if reason == "top_rated_genre" and preferred_genre:
                    context = {"genre": preferred_genre}
                # Backfilled items were never fused, so they carry no RRF score.
                # A decreasing positional value keeps the response monotonic.
                score = 1.0 / (position + 1)
            else:
                primary = sources[0] if sources else SOURCE_POPULARITY
                reason = REASON_BY_SOURCE[primary]
                context = {}
                score = score_by_movie.get(movie_id, 0.0)
            items.append(
                RankedItem(
                    movie_id=int(movie_id),
                    score=round(float(score), 6),
                    reason_code=reason,
                    reason_context=context,
                    sources=sources,
                )
            )
        return items, level, weights
