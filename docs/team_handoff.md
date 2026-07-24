# Team handoff

| Output | Purpose | Intended consumer |
|---|---|---|
| `data/processed/movies_clean.parquet` | Canonical clean movie metadata | model training, backend preparation, manual verification |
| `data/processed/ratings_clean.parquet` | Canonical explicit ratings | model training, manual verification |
| `data/processed/users_clean.parquet` | Observed users and history bounds | model training, analysis |
| `data/processed/id_mapping_clean.parquet` | MovieLens-to-TMDB translation | model preprocessing, traceability |
| `data/processed/movie_*_clean.parquet` | Normalized content child tables | model feature generation, manual verification |
| `data/interim/rejected_*.parquet` | Reason-coded removals | manual verification and audit |
| `data/features/movie_content_features.parquet` | One-row-per-movie model feature source | content-model training |
| `artifacts/content_based/*` | Local TF-IDF baseline artifacts | local onboarding inference |
| `data/features/user_item_interactions.parquet` | Ratings-only canonical interactions | collaborative/hybrid model training |
| `data/splits/interactions_*.parquet` | Leakage-safe chronological partitions | model training and evaluation |
| `data/serving/top_rated_*.parquet` | Guest global/genre rankings | backend serving; future upload candidate |
| `data/serving/movies_serving.parquet` | Compact typed movie records | backend serving; future upload candidate |
| `data/serving/movies_serving.jsonl` | Portable movie records | future backend/cloud ingestion candidate |
| `data/serving/popular_movies.jsonl` | Compact ranked-list records | future DynamoDB ingestion candidate |
| `data/samples/*` | Small readable examples | manual verification |
| `reports/profiling/*` | Raw schema/profile evidence | data review |
| `reports/validation/*` | Cleaning, split, scenario, and final checks | engineering/model review |
| `docs/data_dictionary.md` | Column-level lineage and consumers | backend/model teams |
| `docs/id_mapping.md` | Canonical ID rules and unmatched records | backend/model teams |
| `docs/local_serving_schema.md` | Proposed future logical entities | later backend/cloud design |

“Future” labels describe readiness only. Nothing has been uploaded, inserted, provisioned, or deployed.

## Known limitations

- Ratings are the only interaction signal.
- Some links have no usable TMDB mapping or metadata and are rejected.
- Some movies lack overview, genre, keywords, credits, runtime, or poster metadata.
- Item cold-start exists in chronological holdouts and is reported.
- Content examples are sanity checks, not evidence of production recommendation quality.

