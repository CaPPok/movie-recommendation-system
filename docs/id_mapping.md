# ID mapping

## Canonical identifiers

- `movie_id` is the positive integer TMDB ID from `movies_metadata.id`.
- `user_id` is the positive integer MovieLens ID from `ratings.userId`.
- `movielens_movie_id` is retained in `id_mapping_clean.parquet` only for source-to-canonical translation.
- `imdb_id` is retained only for traceability and is not a model key.

## Mapping direction

`ratings.movieId` -> `links.movieId` -> `links.tmdbId` -> `movies_clean.movie_id`

The clean mapping has 45,460 MovieLens-to-canonical rows. A MovieLens ID maps to at most one canonical ID. A canonical ID may have multiple source aliases: 29 canonical movies have more than one MovieLens ID (59 participating mapping rows).

## Unmatched and duplicate records

- Missing TMDB mappings removed from the mapping: 219.
- Mappings whose TMDB ID has no retained metadata: 164.
- Rating rows rejected for missing TMDB mapping: 13,503.
- Rating rows rejected because metadata is missing: 29,219.
- Canonical alias rating duplicates removed: 125.
- Invalid/duplicate/empty-title movie rows rejected: 36.

## Resolution rules

Metadata exact duplicates are removed first. For remaining duplicate TMDB IDs, the record with the most populated model/serving fields wins; ties use higher vote count and then earlier source row. Multiple MovieLens aliases remain in the mapping because they are real source identifiers. If aliases produce more than one rating for the same canonical user/movie pair, the latest timestamp wins; ties use the larger MovieLens ID and then later source row. Every losing row is retained in `data/interim/rejected_ratings.parquet`.
