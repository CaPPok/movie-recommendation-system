# Local serving schema

All files described here are local artifacts only. The entities below are proposed future logical schemas; no resource has been created or deployed, no credential is required, and no network request is made.

## Current local exports

- `data/serving/movies_serving.parquet` and `.jsonl`: backend-relevant movie metadata only.
- `data/serving/top_rated_all.parquet`: guest global ranking.
- `data/serving/top_rated_by_genre.parquet`: guest genre rankings.
- `data/serving/popular_movies.jsonl`: one compact logical ranking record per global/genre list.
- `data/samples/interaction_event_examples.json`: three schema examples derived from observed ratings. Missing `session_id` is explicitly null.

Large embeddings, sparse matrices, raw training-text duplicates, model artifacts, image files, social fields, and unnecessary external identifiers are excluded from serving exports.

## Proposed future entities

These are design candidates for a later cloud phase, not deployed tables.

### Movies

- Partition key candidate: `movie_id`.
- Attributes: title, release year, genres, overview, poster path, vote metadata, popularity, runtime, original language, companies, and countries.

### Users

- Partition key candidate: `user_id`.
- Attributes: onboarding genres, onboarding movie IDs, and profile status.
- The raw dataset has no onboarding records; these fields are future application data and are not fabricated locally.

### Interactions

- Partition key candidate: `user_id`.
- Sort key candidate: `interaction_timestamp#movie_id`.
- Attributes: movie ID, interaction type, interaction value, timestamp, and session ID.
- The current data supports only `rating`; session ID is unavailable and remains null in schema examples.

### RecommendationCache

- Partition key candidate: `user_id`.
- Sort key candidate: `scenario`.
- Attributes: recommended movie IDs, scores, model version, generated time, and expiration time.

### PopularMovies

- Partition key candidate: `ranking_type`.
- Sort key candidate: `genre`.
- Attributes: ranked movie IDs, scores, and generated time.
- The local export uses the latest clean interaction timestamp as a deterministic data-as-of marker.

## Serialization

`src/data/serving_export.py` converts pandas/numpy values to ordinary logical JSON types while preserving strings, numbers, booleans, lists, maps, and nulls. It does not emit vendor-specific attribute wrappers and does not import a cloud SDK.

