# Data dictionary

This dictionary covers canonical processed tables. Feature, split, and serving schemas are added by their respective pipeline phases.

| Final table | Column | Type | Nullable | Description | Source | Transformation | Backend | Model |
|---|---|---|---|---|---|---|---|---|
| movies_clean | movie_id | int64 | no | Canonical TMDB movie ID | movies_metadata.id | Numeric, positive, deduplicated | yes | yes |
| movies_clean | imdb_id | string | yes | IMDb traceability ID | movies_metadata.imdb_id | Trim; missing tokens to null | no | no |
| movies_clean | title | string | no | Display title | movies_metadata.title | Trim/collapse whitespace; reject empty | yes | yes |
| movies_clean | original_title | string | yes | Original-language title | movies_metadata.original_title | Trim/collapse whitespace | yes | yes |
| movies_clean | overview | string | yes | Movie synopsis | movies_metadata.overview | Trim/collapse whitespace | yes | yes |
| movies_clean | genres | list<string> | no | Normalized genre names | movies_metadata.genres | Safe literal parse; unique order | yes | yes |
| movies_clean | original_language | string | yes | Lowercase source language code | movies_metadata.original_language | Trim and lowercase | yes | yes |
| movies_clean | release_date | timestamp | yes | Parsed release date | movies_metadata.release_date | Coerce invalid/out-of-range values to null | yes | yes |
| movies_clean | release_year | int16 | yes | Release year | movies_metadata.release_date | Derive from valid release date | yes | yes |
| movies_clean | runtime | float32 | yes | Runtime in minutes | movies_metadata.runtime | Numeric; non-positive to null | yes | yes |
| movies_clean | production_companies | list<string> | no | Production company names | movies_metadata.production_companies | Safe literal parse and normalize | yes | yes |
| movies_clean | production_countries | list<string> | no | Production country names | movies_metadata.production_countries | Safe literal parse and normalize | yes | yes |
| movies_clean | poster_path | string | yes | Relative poster path | movies_metadata.poster_path | Trim; missing tokens to null | yes | no |
| movies_clean | vote_average | float32 | yes | TMDB vote mean for descriptive metadata | movies_metadata.vote_average | Numeric in [0,10] | yes | yes |
| movies_clean | vote_count | int64 | yes | TMDB vote count | movies_metadata.vote_count | Non-negative integer | yes | yes |
| movies_clean | popularity | float32 | yes | TMDB popularity measure | movies_metadata.popularity | Non-negative numeric | yes | yes |
| ratings_clean | user_id | int64 | no | MovieLens user ID | ratings.userId | Positive integer | no | yes |
| ratings_clean | movie_id | int64 | no | Canonical TMDB movie ID | ratings.movieId + links.tmdbId | Map through cleaned ID mapping | no | yes |
| ratings_clean | rating | float32 | no | Explicit rating value | ratings.rating | Numeric in configured range | no | yes |
| ratings_clean | timestamp | timestamp[UTC] | no | Interaction time | ratings.timestamp | Unix seconds to UTC | no | yes |
| users_clean | user_id | int64 | no | Observed rating user | ratings.userId | Distinct valid user | no | yes |
| users_clean | interaction_count | int64 | no | Clean rating count | ratings rows | Count after mapping/deduplication | no | yes |
| users_clean | first_interaction_timestamp | timestamp[UTC] | no | Earliest rating time | ratings.timestamp | Per-user minimum | no | yes |
| users_clean | last_interaction_timestamp | timestamp[UTC] | no | Latest rating time | ratings.timestamp | Per-user maximum | no | yes |
| id_mapping_clean | movielens_movie_id | int64 | no | Source MovieLens movie ID | links.movieId | Positive and unique | no | yes |
| id_mapping_clean | movie_id | int64 | no | Canonical TMDB movie ID | links.tmdbId | Must exist in movies_clean | no | yes |
| id_mapping_clean | imdb_id | string | yes | IMDb traceability ID | links.imdbId | Prefix tt; pad to at least 7 digits | no | no |
| movie_genres_clean | movie_id | int64 | no | Canonical movie foreign key | movies_metadata.id | Reference movies_clean | no | yes |
| movie_genres_clean | genre | string | no | Normalized genre | movies_metadata.genres.name | Trim and deduplicate | yes | yes |
| movie_companies_clean | movie_id | int64 | no | Canonical movie foreign key | movies_metadata.id | Reference movies_clean | no | yes |
| movie_companies_clean | company_id | int64 | yes | Source company identifier | production_companies.id | Numeric when present | no | no |
| movie_companies_clean | company | string | no | Company name | production_companies.name | Trim and deduplicate | yes | yes |
| movie_countries_clean | movie_id | int64 | no | Canonical movie foreign key | movies_metadata.id | Reference movies_clean | no | yes |
| movie_countries_clean | country_code | string | yes | ISO country code | production_countries.iso_3166_1 | Trim and uppercase | yes | yes |
| movie_countries_clean | country | string | no | Country name | production_countries.name | Trim and deduplicate | yes | yes |
| movie_keywords_clean | movie_id | int64 | no | Canonical movie foreign key | keywords.id | Reference movies_clean; merge duplicates | no | yes |
| movie_keywords_clean | keywords | list<string> | no | Normalized keywords | keywords.keywords.name | Safe parse; unique order | no | yes |
| movie_credits_clean | movie_id | int64 | no | Canonical movie foreign key | credits.id | Reference movies_clean; merge duplicates | no | yes |
| movie_credits_clean | cast_names | list<string> | no | Ordered cast names | credits.cast.name | Safe parse; unique order | no | yes |
| movie_credits_clean | director_names | list<string> | no | Director names | credits.crew | Filter job=Director; unique order | no | yes |
| movie_content_features | movie_id | int64 | no | Canonical movie key | movies_clean.movie_id | One row per clean movie | no | yes |
| movie_content_features | cleaned_text | string | no | TF-IDF source document | clean metadata, keywords, credits | Normalize text and add prefixed categorical tokens | no | yes |
| movie_content_features | genres | list<string> | no | Normalized genres | movies_clean.genres | Pass through canonical list | no | yes |
| movie_content_features | keywords | list<string> | no | Normalized keywords | movie_keywords_clean.keywords | Left join; missing to empty list | no | yes |
| movie_content_features | cast_names | list<string> | no | Top ordered cast names | movie_credits_clean.cast_names | Limit to configured count | no | yes |
| movie_content_features | director_names | list<string> | no | Director names | movie_credits_clean.director_names | Left join; missing to empty list | no | yes |
| movie_content_features | numeric metadata | mixed | yes | Language, year, runtime, vote and popularity features | movies_clean | Typed canonical values | no | yes |
| movie_content_features | availability indicators | bool | no | Feature presence flags | movies_clean/content joins | Boolean derivation | no | yes |
| user_item_interactions | user_id | int64 | no | Returning-user key | ratings_clean.user_id | Pass through | no | yes |
| user_item_interactions | movie_id | int64 | no | Canonical item key | ratings_clean.movie_id | Pass through | no | yes |
| user_item_interactions | interaction_value | float32 | no | Observed rating | ratings_clean.rating | Rename only | no | yes |
| user_item_interactions | interaction_type | string | no | Observed signal type | dataset capability | Constant rating; no fabricated signals | no | yes |
| user_item_interactions | timestamp | timestamp[UTC] | no | Observed rating time | ratings_clean.timestamp | Pass through | no | yes |
| interaction splits | all interaction columns | canonical | no | Chronological train/validation/test partitions | user_item_interactions | Latest=test, penultimate=validation for eligible users | no | yes |
| top_rated rankings | ranking_type | string | no | ALL or GENRE ranking | derived | Configured scenario label | yes | no |
| top_rated rankings | genre | string | no | ALL or normalized genre | movies_clean.genres | Exploded normalized genre | yes | no |
| top_rated rankings | rank | int64 | no | One-based deterministic rank | weighted score | Score/count/mean/movie ID ordering | yes | no |
| top_rated rankings | movie_id | int64 | no | Canonical ranked movie | ratings_clean + movies_clean | Foreign-key join | yes | no |
| top_rated rankings | score | float64 | no | Weighted rating score | rating aggregates | IMDb-style shrinkage | yes | no |
| top_rated rankings | average_rating | float64 | no | Clean mean rating | ratings_clean.rating | Per-movie mean | yes | no |
| top_rated rankings | rating_count | int64 | no | Clean rating count | ratings_clean rows | Per-movie count | yes | no |
| movies_serving | movie_id | int64 | no | Backend movie key | movies_clean.movie_id | Pass through | yes | no |
| movies_serving | display metadata | mixed | partly | Title, year, genres, overview, poster, vote, popularity, runtime, language, companies, countries | movies_clean | Backend allowlist and field renames | yes | no |
