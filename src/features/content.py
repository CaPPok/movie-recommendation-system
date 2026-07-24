"""Canonical movie content features and local TF-IDF artifacts."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.config import output_dir


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if hasattr(value, "tolist"):
        return [str(item) for item in value.tolist() if str(item).strip()]
    return []


def _slug(value: str) -> str:
    normalized = re.sub(r"[^\w\s]", " ", str(value).lower(), flags=re.UNICODE)
    return re.sub(r"\s+", "_", normalized.strip())


def _clean_free_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"[^\w\s]", " ", str(value).lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _prefixed_tokens(prefix: str, values: list[str]) -> list[str]:
    return [f"{prefix}_{token}" for value in values if (token := _slug(value))]


def build_movie_content_features(config: dict[str, Any]) -> dict[str, Any]:
    """Build one-row-per-movie feature sources and fitted sparse TF-IDF artifacts."""
    processed_dir = output_dir(config, "processed_dir")
    features_dir = output_dir(config, "features_dir")
    artifacts_dir = output_dir(config, "artifacts_dir")
    settings = config["content_features"]

    movies = pd.read_parquet(processed_dir / "movies_clean.parquet")
    keywords = pd.read_parquet(processed_dir / "movie_keywords_clean.parquet")
    credits = pd.read_parquet(processed_dir / "movie_credits_clean.parquet")
    movies = movies.merge(keywords, on="movie_id", how="left").merge(
        credits, on="movie_id", how="left"
    )
    list_columns = [
        "genres",
        "production_companies",
        "production_countries",
        "keywords",
        "cast_names",
        "director_names",
    ]
    for column in list_columns:
        movies[column] = movies[column].map(_as_list)

    max_cast = int(settings["max_cast_members"])
    text_rows: list[str] = []
    for row in movies.itertuples(index=False):
        tokens: list[str] = []
        title = _clean_free_text(row.title)
        overview = _clean_free_text(row.overview)
        if title:
            tokens.append(title)
        if settings["include_overview"] and overview:
            tokens.append(overview)
        if settings["include_genres"]:
            tokens.extend(_prefixed_tokens("genre", row.genres))
        if settings["include_keywords"]:
            tokens.extend(_prefixed_tokens("keyword", row.keywords))
        if settings["include_cast"]:
            tokens.extend(_prefixed_tokens("cast", row.cast_names[:max_cast]))
        if settings["include_director"]:
            tokens.extend(_prefixed_tokens("director", row.director_names))
        if settings["include_companies"]:
            tokens.extend(_prefixed_tokens("company", row.production_companies))
        if settings["include_countries"]:
            tokens.extend(_prefixed_tokens("country", row.production_countries))
        if row.original_language is not pd.NA and pd.notna(row.original_language):
            language = _slug(str(row.original_language))
            if language:
                tokens.append(f"language_{language}")
        if row.release_year is not pd.NA and pd.notna(row.release_year):
            tokens.append(f"year_{int(row.release_year)}")
        text_rows.append(" ".join(tokens))

    feature_table = pd.DataFrame(
        {
            "movie_id": movies["movie_id"].astype("int64"),
            "cleaned_text": text_rows,
            "genres": movies["genres"],
            "keywords": movies["keywords"],
            "cast_names": movies["cast_names"].map(lambda values: values[:max_cast]),
            "director_names": movies["director_names"],
            "original_language": movies["original_language"].astype("string"),
            "release_year": movies["release_year"].astype("Int16"),
            "runtime": movies["runtime"].astype("Float32"),
            "vote_average": movies["vote_average"].astype("Float32"),
            "vote_count": movies["vote_count"].astype("Int64"),
            "popularity": movies["popularity"].astype("Float32"),
            "has_overview": movies["overview"].notna(),
            "has_genres": movies["genres"].map(bool),
            "has_keywords": movies["keywords"].map(bool),
            "has_cast": movies["cast_names"].map(bool),
            "has_director": movies["director_names"].map(bool),
            "has_runtime": movies["runtime"].notna(),
        }
    ).sort_values("movie_id", kind="mergesort").reset_index(drop=True)
    feature_path = features_dir / "movie_content_features.parquet"
    temporary_feature = feature_path.with_name(f"{feature_path.name}.tmp")
    feature_table.to_parquet(temporary_feature, index=False, compression="snappy")
    os.replace(temporary_feature, feature_path)

    vectorizer_config = settings["vectorizer"]
    vectorizer = TfidfVectorizer(
        max_features=int(vectorizer_config["max_features"]),
        ngram_range=tuple(int(value) for value in vectorizer_config["ngram_range"]),
        min_df=int(vectorizer_config["min_df"]),
        max_df=float(vectorizer_config["max_df"]),
        sublinear_tf=bool(vectorizer_config["sublinear_tf"]),
        dtype=np.float32,
        norm="l2",
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(feature_table["cleaned_text"])
    # Canonicalize pickle-only state. Vocabulary lookup does not depend on
    # insertion order, and `_stop_words_id` is a process-specific object ID
    # cache that sklearn safely recreates on first transform.
    vectorizer.vocabulary_ = dict(sorted(vectorizer.vocabulary_.items()))
    if hasattr(vectorizer, "_stop_words_id"):
        delattr(vectorizer, "_stop_words_id")

    vectorizer_path = artifacts_dir / "vectorizer.joblib"
    temporary_vectorizer = vectorizer_path.with_name(f"{vectorizer_path.name}.tmp")
    joblib.dump(vectorizer, temporary_vectorizer)
    os.replace(temporary_vectorizer, vectorizer_path)

    matrix_path = artifacts_dir / "movie_matrix.npz"
    temporary_matrix = artifacts_dir / "movie_matrix.tmp.npz"
    sparse.save_npz(temporary_matrix, matrix, compressed=True)
    os.replace(temporary_matrix, matrix_path)

    index = pd.DataFrame(
        {
            "row_index": range(len(feature_table)),
            "movie_id": feature_table["movie_id"],
        }
    )
    index_path = artifacts_dir / "movie_index.parquet"
    temporary_index = index_path.with_name(f"{index_path.name}.tmp")
    index.to_parquet(temporary_index, index=False, compression="snappy")
    os.replace(temporary_index, index_path)
    return {
        "movies": len(feature_table),
        "text_nonempty": int(feature_table["cleaned_text"].str.len().gt(0).sum()),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "matrix_nonzero": int(matrix.nnz),
        "feature_availability": {
            column: int(feature_table[column].sum())
            for column in [
                "has_overview",
                "has_genres",
                "has_keywords",
                "has_cast",
                "has_director",
                "has_runtime",
            ]
        },
    }
