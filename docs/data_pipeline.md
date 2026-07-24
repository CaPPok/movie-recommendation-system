# Data pipeline

## Scope and source

The pipeline reads the seven UTF-8 comma-delimited files in `movies_dataset/` in place. Raw files are treated as read-only. The full ratings/link pair drives production outputs; small files are profiled only.

No cloud operation is present. Every result remains under `data/`, `artifacts/`, `reports/`, or `docs/`.

## Phase A — inspection

The repository and every raw schema were inspected before implementation. `movies_metadata.id`, `credits.id`, and `keywords.id` are TMDB IDs; rating `movieId` values are MovieLens IDs requiring `links.csv`.

## Phase B — profiling

`scripts/profile_data.py` reports shapes, types, memory, nulls, uniqueness, duplicate keys/rows, ranges, representative values, list parsing, timestamp/rating ranges, outliers, and cross-table references. Critical validation failures stop execution.

## Phase C — cleaning

- Canonical `movie_id`: positive integer TMDB metadata ID.
- Exact duplicate metadata rows are removed first.
- Remaining duplicate movie IDs retain the most complete row, then highest vote count, then earliest source row.
- Empty-title and malformed-ID movies are rejected.
- Genres, companies, countries, keywords, cast, and directors are safely parsed and normalized.
- Ratings map MovieLens IDs through the clean ID mapping.
- Missing mappings/metadata and invalid values are rejected with reason codes.
- Canonical alias duplicates keep the latest timestamp, then larger MovieLens ID, then later source row.
- Sparse users are retained.

All full tables are Parquet; small human-readable samples are CSV.

## Phase D — features and scenarios

Movie content features combine clean text, prefixed categorical tokens, supported numeric metadata, and availability indicators. A local TF-IDF vectorizer and sparse matrix are stored separately.

Guest rankings use:

`(v/(v+m))*R + (m/(v+m))*C`

The pipeline evaluates configured rating-count percentiles and selects the configured 90th-percentile strategy from current data. Onboarding builds a weighted profile from selected movies and genres only. Returning-user interactions use ratings only.

## Phase E — splits and serving

For users with at least three interactions, chronological order assigns latest to test, penultimate to validation, and earlier interactions to train. Users with fewer than three interactions remain train-only. Ties use movie ID and interaction value for deterministic ordering.

Serving exports allowlist backend metadata and exclude sparse vectors, embeddings, model artifacts, raw duplicate text, images, social fields, and unnecessary external IDs. JSONL uses ordinary logical JSON types.

## Phase F — validation

Final validation checks required outputs, key uniqueness, foreign keys, required values, types/ranges/timestamps, feature coverage, guest/onboarding behavior, split leakage, JSON parsing/type preservation, and deterministic artifact hashes. Every rule is labeled PASS, WARNING, or FAIL.

See `README.md` for exact commands and `reports/validation/` for evidence.

