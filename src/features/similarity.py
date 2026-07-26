"""Precompute the nearest neighbours of every movie for "Because you watched".

The answer to "what is similar to film X" never changes between requests, so
computing it on demand would repeat the same 45,430-way comparison for every page
view. It is built once here and served as a lookup.

The full similarity matrix is 45,430 x 45,430 - roughly 8 GB in float32 - so it is
never materialised. Rows are processed in blocks, each block keeps only its top
neighbours, and the rest is discarded before the next block is read.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.data.config import output_dir

SIMILAR_MOVIES_FILE = "similar_movies_top50.parquet"

# Each block holds a dense (chunk x 45,430) float32 score matrix. 256 rows is
# about 47 MB, which stays comfortable while keeping the BLAS call large enough
# to be efficient.
DEFAULT_CHUNK_SIZE = 256


def build_quality_scores(
    config: dict[str, Any], movie_ids: np.ndarray, percentile: float
) -> np.ndarray:
    """IMDb-style shrunk rating per movie, rescaled to 0-1 in catalogue order.

    Content similarity has no notion of whether a film is any good, so an obscure
    title sharing a few keywords ranks alongside a classic. This supplies the
    missing signal. Shrinkage matters as much as the rating itself: a 10/10 from
    three voters carries far less evidence than an 8.5 from thousands, and the
    formula pulls the former back toward the catalogue mean.
    """
    serving_dir = output_dir(config, "serving_dir", create=False)
    serving = pd.read_parquet(
        serving_dir / "movies_serving.parquet",
        columns=["movie_id", "vote_average", "vote_count"],
    )
    ordered = (
        pd.DataFrame({"movie_id": movie_ids})
        .merge(serving, on="movie_id", how="left")
    )
    averages = ordered["vote_average"].astype("float64").fillna(0.0).to_numpy()
    counts = ordered["vote_count"].astype("float64").fillna(0.0).to_numpy()

    rated = counts > 0
    total_votes = counts[rated].sum()
    global_average = (
        float((averages[rated] * counts[rated]).sum() / total_votes)
        if total_votes > 0
        else 0.0
    )
    minimum_count = float(np.quantile(counts[rated], percentile)) if rated.any() else 0.0

    weighted = (counts / (counts + minimum_count)) * averages + (
        minimum_count / (counts + minimum_count)
    ) * global_average
    weighted = np.nan_to_num(weighted, nan=global_average)

    lowest, highest = float(weighted.min()), float(weighted.max())
    if highest == lowest:
        return np.ones_like(weighted, dtype="float32")
    return ((weighted - lowest) / (highest - lowest)).astype("float32")


def _ineligible_columns(
    config: dict[str, Any], movie_ids: np.ndarray, min_vote_count: float
) -> np.ndarray:
    """Boolean mask of movies that may not be returned as a neighbour."""
    if min_vote_count <= 0:
        return np.zeros(len(movie_ids), dtype=bool)
    serving_dir = output_dir(config, "serving_dir", create=False)
    serving = pd.read_parquet(
        serving_dir / "movies_serving.parquet", columns=["movie_id", "vote_count"]
    )
    ordered = pd.DataFrame({"movie_id": movie_ids}).merge(
        serving, on="movie_id", how="left"
    )
    counts = ordered["vote_count"].astype("float64").fillna(0.0).to_numpy()
    return counts < min_vote_count


def build_similar_movies(
    config: dict[str, Any],
    model_config: dict[str, Any] | None = None,
    top_n: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: bool = True,
) -> dict[str, Any]:
    """Write the nearest neighbours of every movie, ranked by relevance and quality.

    Two stages on purpose. A pool is selected by pure cosine similarity, so
    nothing off-topic can enter no matter how well reviewed it is. Quality then
    reorders that pool, which is where it belongs: relevance decides who is
    eligible, quality decides who goes first.
    """
    settings = (model_config or {}).get("similar_movies", {})
    top_n = int(top_n or settings.get("top_n", 50))
    pool_size = int(settings.get("pool_size", max(top_n * 4, top_n)))
    w_quality = float(settings.get("w_quality", 0.0))
    percentile = float(settings.get("vote_count_percentile", 0.90))
    min_vote_count = float(settings.get("min_vote_count", 0))

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0.0 <= w_quality <= 1.0:
        raise ValueError("w_quality must be between 0 and 1")
    pool_size = max(pool_size, top_n)

    artifacts_dir = output_dir(config, "artifacts_dir", create=False)
    matrix = sparse.load_npz(artifacts_dir / "movie_matrix.npz").tocsr()
    index = pd.read_parquet(artifacts_dir / "movie_index.parquet")
    movie_ids = index.sort_values("row_index")["movie_id"].to_numpy(dtype="int64")

    if matrix.shape[0] != len(movie_ids):
        raise ValueError(
            "Matrix rows and index length disagree: "
            f"{matrix.shape[0]} vs {len(movie_ids)}"
        )

    total = matrix.shape[0]
    quality = (
        build_quality_scores(config, movie_ids, percentile)
        if w_quality > 0
        else np.zeros(total, dtype="float32")
    )

    # Eligibility is decided before ranking, so a movie whose closest content
    # matches are all unrated still ends up with a full set of usable neighbours
    # instead of a short list padded with noise.
    ineligible = _ineligible_columns(config, movie_ids, min_vote_count)
    eligible_count = total - int(ineligible.sum())
    if eligible_count <= 1:
        raise ValueError(
            f"min_vote_count={min_vote_count} leaves {eligible_count} eligible "
            "movies; lower the threshold."
        )

    take = min(top_n, eligible_count - 1)
    pool = min(pool_size, eligible_count - 1)
    transposed = matrix.T.tocsc()

    source_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    empty_rows = 0

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        scores = (matrix[start:stop] @ transposed).toarray()

        # A movie is always its own closest match; remove it before ranking so
        # every stored neighbour is a genuine recommendation.
        rows = np.arange(stop - start)
        scores[rows, np.arange(start, stop)] = -np.inf
        scores[:, ineligible] = -np.inf

        # Stage one: shortlist on content similarity alone.
        partitioned = np.argpartition(-scores, pool - 1, axis=1)[:, :pool]
        pool_scores = np.take_along_axis(scores, partitioned, axis=1)

        if w_quality > 0:
            # Stage two: re-rank the shortlist. Cosine is min-maxed within each
            # row's own pool because its absolute range varies a lot between
            # movies, and blending two differently scaled terms would let one
            # silently dominate.
            lowest = pool_scores.min(axis=1, keepdims=True)
            highest = pool_scores.max(axis=1, keepdims=True)
            span = np.where(highest > lowest, highest - lowest, 1.0)
            relevance = (pool_scores - lowest) / span
            combined = (1.0 - w_quality) * relevance + w_quality * quality[partitioned]
        else:
            combined = pool_scores

        best = np.argpartition(-combined, take - 1, axis=1)[:, :take]
        best_combined = np.take_along_axis(combined, best, axis=1)
        order = np.argsort(-best_combined, axis=1, kind="stable")
        selected = np.take_along_axis(best, order, axis=1)
        neighbours = np.take_along_axis(partitioned, selected, axis=1)
        # The stored score stays the raw cosine, so the column keeps a single
        # meaning regardless of how the ordering was decided.
        neighbour_scores = np.take_along_axis(pool_scores, selected, axis=1)

        # Movies whose text shares no term with anything else score zero all the
        # way across; storing those pairs would be noise presented as similarity.
        keep = neighbour_scores > 0
        empty_rows += int((~keep.any(axis=1)).sum())

        row_offsets = np.repeat(np.arange(start, stop), keep.sum(axis=1))
        source_parts.append(movie_ids[row_offsets])
        rank_parts.append(
            np.concatenate(
                [np.arange(1, count + 1, dtype="int16") for count in keep.sum(axis=1)]
            )
            if keep.any()
            else np.empty(0, dtype="int16")
        )
        target_parts.append(movie_ids[neighbours[keep]])
        score_parts.append(neighbour_scores[keep].astype("float32"))

        if progress:
            print(
                f"  {stop:,}/{total:,} phim", end="\r", flush=True
            )

    if progress:
        print()

    table = pd.DataFrame(
        {
            "movie_id": np.concatenate(source_parts),
            "rank": np.concatenate(rank_parts),
            "similar_movie_id": np.concatenate(target_parts),
            "score": np.concatenate(score_parts),
        }
    )

    features_dir = output_dir(config, "artifacts_dir")
    destination = features_dir / SIMILAR_MOVIES_FILE
    temporary = destination.with_name(f"{destination.name}.tmp")
    table.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(destination)

    return {
        "movies": int(total),
        "eligible_as_neighbour": int(eligible_count),
        "min_vote_count": min_vote_count,
        "top_n": int(take),
        "pool_size": int(pool),
        "w_quality": w_quality,
        "rows": int(len(table)),
        "movies_without_neighbours": empty_rows,
        "mean_neighbours_per_movie": round(len(table) / total, 2),
        "score_min": float(table["score"].min()) if len(table) else 0.0,
        "score_max": float(table["score"].max()) if len(table) else 0.0,
    }


class SimilarMovieLookup:
    """Read-side wrapper over the precomputed neighbour table."""

    def __init__(self, config: dict[str, Any]) -> None:
        artifacts_dir = output_dir(config, "artifacts_dir", create=False)
        table = pd.read_parquet(artifacts_dir / SIMILAR_MOVIES_FILE)
        table = table.sort_values(["movie_id", "rank"], kind="mergesort")
        self._by_movie: dict[int, list[int]] = {
            int(movie_id): group.to_numpy(dtype="int64").tolist()
            for movie_id, group in table.groupby("movie_id", sort=False)[
                "similar_movie_id"
            ]
        }

    def similar_to(self, movie_id: int, top_n: int = 20) -> list[int]:
        return self._by_movie.get(int(movie_id), [])[:top_n]

    def similar_to_any(
        self, movie_ids: list[int], top_n: int = 20
    ) -> list[tuple[int, int]]:
        """Interleave neighbours of several movies, keeping which seed produced each.

        Round-robin rather than concatenation, so a single seed cannot fill the
        whole row when the user has watched several different things.
        """
        pools = [
            (int(movie_id), self._by_movie.get(int(movie_id), []))
            for movie_id in movie_ids
        ]
        results: list[tuple[int, int]] = []
        seen = {int(movie_id) for movie_id in movie_ids}
        depth = 0
        while len(results) < top_n and any(depth < len(pool) for _, pool in pools):
            for seed, pool in pools:
                if depth >= len(pool):
                    continue
                candidate = pool[depth]
                if candidate in seen:
                    continue
                seen.add(candidate)
                results.append((candidate, seed))
                if len(results) >= top_n:
                    break
            depth += 1
        return results
