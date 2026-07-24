You are working inside an existing repository for a Netflix-style Movie Recommendation System.

The dataset has already been downloaded to the local machine. Do not download another dataset unless an expected file is genuinely missing, and do not replace or modify the original raw files.

The project has a 7-day deadline and a 3-person AI team. Your task in this phase is to build and validate the complete local data-processing pipeline only.

## Critical stopping rule

This task must stop before any cloud operation.

Do not:

* create AWS resources;
* connect to AWS;
* use boto3;
* use AWS CLI;
* upload files to S3;
* create or modify DynamoDB tables;
* invoke SageMaker;
* require AWS credentials;
* add Terraform, CloudFormation, CDK, or deployment scripts;
* simulate successful cloud uploads.

You may prepare local files that will later be uploaded to S3 or inserted into DynamoDB, but all outputs must remain on the local machine.

---

# 1. Project context

The system has three recommendation scenarios.

## Scenario 1 — Guest user

A guest user is not tracked.

The system only returns top-rated movies.

Do not create or use:

* guest user profiles;
* guest interaction histories;
* guest_recent_movie_ids;
* guest session recommendation models.

The output for this scenario should be a precomputed top-rated movie ranking, optionally separated by genre.

## Scenario 2 — First-login onboarding user

A newly registered user selects some preferred movies and genres during onboarding.

Recommendations must use only:

* selected_movie_ids;
* selected_genres;
* cleaned movie metadata and content features.

Do not use guest_recent_movie_ids.

This scenario should be handled by a content-based recommendation dataset and local baseline recommender.

## Scenario 3 — Returning user

A returning user has historical interactions.

Recommendations may use:

* ratings;
* clicks;
* watches;
* likes;
* completed views;
* other available interaction signals.

Build a dataset suitable for collaborative filtering or a hybrid recommender.

Do not invent interaction types that are not present in the current dataset. If the downloaded dataset only contains ratings, use ratings as the initial interaction signal and document this limitation.

---

# 2. Initial repository inspection

Before changing any file:

1. Inspect the complete repository structure.
2. Locate all downloaded dataset files.
3. Identify file formats, encodings, delimiters, and approximate sizes.
4. Inspect the first rows and schemas of every relevant file.
5. Identify relationships between files.
6. Inspect existing notebooks, cleaning scripts, reports, and processed outputs.
7. Do not assume the dataset is standard MovieLens or TMDB until the files are inspected.
8. Do not assume column names from this prompt.
9. Reuse correct existing code where practical, but do not silently trust it.
10. Detect whether cleaning or EDA has already been completed.

Before implementation, print a concise inspection report containing:

* repository tree;
* dataset filenames;
* row and column counts;
* key columns;
* probable primary keys;
* probable foreign keys;
* duplicate ID issues;
* encoding or parsing issues;
* existing cleaning work;
* missing files;
* schema conflicts;
* assumptions that still need validation.

Then propose an implementation plan based on the actual repository.

Do not start by rewriting the repository from scratch.

---

# 3. Required local directory structure

Adapt the existing repository rather than duplicating equivalent folders.

Use this structure where suitable:

```text
data/
├── raw/
├── interim/
├── processed/
├── features/
├── splits/
├── serving/
└── samples/

reports/
├── profiling/
├── validation/
└── figures/

configs/

scripts/

src/
├── data/
├── features/
├── recommenders/
└── utils/

tests/

docs/
```

Rules:

* Raw files must be treated as read-only.
* Do not overwrite raw files.
* Intermediate outputs go to `data/interim/`.
* Final cleaned tables go to `data/processed/`.
* Model-ready feature tables go to `data/features/`.
* Train, validation, and test files go to `data/splits/`.
* Local files prepared for future backend or DynamoDB ingestion go to `data/serving/`.
* Small human-readable samples go to `data/samples/`.

Add generated large data folders to `.gitignore` where appropriate, but do not remove existing tracked source code.

---

# 4. Data profiling

Create a reproducible profiling module and CLI command.

The profiler must report for every table:

* filename;
* file type;
* row count;
* column count;
* column names;
* inferred data types;
* memory usage;
* null count and null percentage;
* unique value count;
* duplicate row count;
* duplicate primary-key count;
* minimum and maximum for numeric columns;
* representative values for categorical columns;
* invalid or malformed values;
* list-like columns stored as strings;
* timestamp ranges;
* rating ranges;
* suspicious outliers;
* broken references between related tables.

Check referential integrity where relevant, such as:

* rating movie IDs that do not exist in the movie table;
* link IDs that do not map to a movie;
* genre, company, or country records that cannot map to a movie;
* duplicated movie mappings;
* multiple IDs referring to the same movie;
* invalid user IDs;
* invalid movie IDs.

Generate:

```text
reports/profiling/raw_profile.json
reports/profiling/raw_profile.md
reports/validation/raw_validation.json
reports/validation/raw_validation.md
```

The Markdown report must be understandable without opening the code.

---

# 5. Canonical schema decisions

After inspecting the files, define a canonical local schema.

Use `movie_id` as the canonical internal movie identifier unless the existing repository clearly establishes another correct identifier.

External identifiers such as TMDB, IMDb, or MovieLens IDs may be retained only when they are useful for traceability or integration.

Do not keep unnecessary attributes merely because they exist in the raw dataset.

Based on the previous project decisions, unnecessary fields may include:

* social-media IDs;
* wiki IDs;
* duplicated timestamps;
* duplicated external mapping IDs;
* fields explicitly removed in existing repository work.

However, inspect the current code and documentation before removing anything. Do not reintroduce columns that the project has already intentionally removed.

Produce:

```text
docs/data_dictionary.md
docs/id_mapping.md
```

`data_dictionary.md` must contain:

* final table name;
* column name;
* data type;
* nullable status;
* description;
* source column;
* transformation rule;
* whether the backend needs the column;
* whether a model needs the column.

`id_mapping.md` must explain:

* canonical movie ID;
* user ID;
* external IDs;
* mapping direction;
* unmatched records;
* duplicate mappings;
* chosen resolution rules.

---

# 6. Data-cleaning pipeline

Implement cleaning as modular Python code rather than one large notebook.

The pipeline should be executable from the command line.

Example target command:

```bash
python scripts/run_data_pipeline.py --config configs/data_pipeline.yaml
```

Do not hard-code machine-specific absolute paths.

## Required cleaning operations

Apply only operations supported by the actual data.

Possible operations include:

* normalize column names;
* trim whitespace;
* standardize missing-value representations;
* normalize text encoding;
* convert numeric columns safely;
* parse timestamps;
* normalize boolean fields;
* normalize list-like fields;
* standardize genres;
* standardize companies;
* standardize countries;
* remove exact duplicate rows;
* resolve duplicate IDs using explicit rules;
* remove invalid ratings;
* remove invalid movie references;
* handle missing movie metadata;
* handle empty titles;
* handle invalid release years;
* normalize language codes;
* normalize poster paths or URLs;
* aggregate repeated ratings only if justified;
* retain interaction timestamps where available.

Every transformation must be documented.

Do not silently drop rows.

For each major table, report:

* rows before cleaning;
* rows after cleaning;
* rows removed;
* rows modified;
* duplicates removed;
* null values before and after;
* reasons for removed records.

Generate a machine-readable rejection log where practical:

```text
data/interim/rejected_movies.parquet
data/interim/rejected_ratings.parquet
reports/validation/cleaning_summary.json
reports/validation/cleaning_summary.md
```

---

# 7. Final cleaned tables

Create only tables supported by the actual dataset.

Expected logical outputs may include:

```text
data/processed/movies_clean.parquet
data/processed/ratings_clean.parquet
data/processed/users_clean.parquet
data/processed/movie_genres_clean.parquet
data/processed/movie_companies_clean.parquet
data/processed/movie_countries_clean.parquet
data/processed/id_mapping_clean.parquet
```

Also create CSV samples for manual inspection:

```text
data/samples/movies_clean_sample.csv
data/samples/ratings_clean_sample.csv
```

Prefer Parquet for complete processed datasets.

CSV is optional for full datasets, but create readable samples.

Do not create empty placeholder tables. If a logical table cannot be built, document why.

---

# 8. Movie feature dataset

Build a model-ready movie feature table for content-based recommendation.

Use only features supported by the actual data.

Possible features:

* genres;
* overview or description;
* keywords;
* original language;
* release year;
* runtime;
* production companies;
* production countries;
* vote average;
* vote count;
* popularity.

Separate descriptive metadata from model-transformed features.

Create a canonical feature source table:

```text
data/features/movie_content_features.parquet
```

It should contain one row per movie and at minimum:

* movie_id;
* cleaned text representation;
* normalized categorical features;
* usable numeric features;
* feature availability indicators.

Do not store a huge sparse matrix directly in this Parquet file unless there is a strong reason.

If TF-IDF or another vectorizer is implemented, save local artifacts separately, for example:

```text
artifacts/content_based/vectorizer.joblib
artifacts/content_based/movie_matrix.npz
artifacts/content_based/movie_index.parquet
```

Do not upload these files anywhere.

---

# 9. Scenario 1 dataset — Guest top-rated ranking

Create a robust top-rated ranking.

Do not rank movies using raw average rating alone without considering rating count.

Implement a documented weighted-rating method, such as:

```text
weighted_rating =
(v / (v + m)) * R
+
(m / (v + m)) * C
```

Where:

* `R` is the movie’s average rating;
* `v` is the number of ratings for the movie;
* `C` is the global average rating;
* `m` is a minimum-rating-count threshold.

Do not arbitrarily hard-code `m`.

Evaluate sensible candidates such as rating-count percentiles and document the chosen value.

Generate:

```text
data/serving/top_rated_all.parquet
data/serving/top_rated_by_genre.parquet
reports/validation/top_rated_summary.md
```

Expected fields:

* ranking_type;
* genre;
* rank;
* movie_id;
* score;
* average_rating;
* rating_count.

Requirements:

* `ALL` ranking for general guest access;
* genre-specific rankings where sufficient data exists;
* no user ID;
* no guest tracking;
* deterministic ordering for ties;
* configurable top-K.

Also create a local function or baseline class:

```python
get_guest_recommendations(genre: str | None, top_k: int)
```

---

# 10. Scenario 2 dataset — Onboarding content-based recommendation

Build a local content-based baseline using selected movies and selected genres.

Input contract:

```json
{
  "selected_movie_ids": ["movie_1", "movie_2"],
  "selected_genres": ["Drama", "Thriller"],
  "top_k": 20
}
```

Requirements:

* validate selected movie IDs;
* ignore invalid IDs with warnings;
* combine selected-movie profiles and selected-genre preferences;
* exclude selected movies from recommendations;
* return unique movies;
* return ranked results;
* include recommendation scores;
* provide deterministic behavior;
* handle genre-only onboarding;
* handle movie-only onboarding;
* handle both movies and genres;
* provide a documented fallback to top-rated movies when inputs are empty or unusable.

Do not use:

* guest_recent_movie_ids;
* guest interactions;
* fabricated user history.

Create a baseline implementation under:

```text
src/recommenders/content_based.py
```

Create local evaluation or sanity-check cases under:

```text
reports/validation/onboarding_recommendation_examples.md
```

Include several understandable examples showing:

* input selected movies;
* input genres;
* returned recommendations;
* why the recommendations are plausible.

Do not claim model quality solely from these examples.

---

# 11. Scenario 3 dataset — Returning-user interactions

Create a model-ready interaction table.

Expected fields, where available:

* user_id;
* movie_id;
* interaction_value;
* interaction_type;
* timestamp.

If only ratings exist:

* set `interaction_type` to `rating`;
* preserve the original rating as `interaction_value`;
* do not invent clicks, watches, or likes.

Generate:

```text
data/features/user_item_interactions.parquet
```

Report:

* number of users;
* number of movies;
* number of interactions;
* sparsity;
* interactions per user;
* interactions per movie;
* cold-start users;
* cold-start movies;
* users with too few interactions;
* rating distribution;
* timestamp coverage.

Document minimum-interaction filtering rules.

Do not remove sparse users without reporting how many are removed and why.

---

# 12. Train, validation, and test split

Prevent data leakage.

If valid timestamps exist, use a time-aware per-user split.

Recommended baseline:

* earlier interactions for training;
* later interaction for validation;
* latest interaction for testing;

or another justified chronological split.

Requirements:

* no future interaction may appear in training for the same user;
* avoid random row-level splitting when timestamps are available;
* every test user should have appropriate training history;
* document treatment of users with too few interactions;
* ensure there are no duplicate user-movie interactions across splits unless explicitly justified.

If timestamps do not exist, use a reproducible user-aware split and clearly document the limitation.

Generate:

```text
data/splits/interactions_train.parquet
data/splits/interactions_validation.parquet
data/splits/interactions_test.parquet
reports/validation/split_summary.json
reports/validation/split_summary.md
```

Validate:

* row counts;
* user overlap;
* movie overlap;
* duplicate interactions;
* chronological ordering;
* cold-start conditions;
* leakage checks.

---

# 13. Local serving datasets

Prepare local backend-serving files, but do not upload them.

## Movies serving dataset

Create:

```text
data/serving/movies_serving.parquet
data/serving/movies_serving.jsonl
```

Include only backend-relevant metadata, such as:

* movie_id;
* title;
* release_year;
* genres;
* overview;
* poster_path or poster URL;
* vote_average;
* vote_count;
* popularity;
* runtime;
* original_language;
* companies;
* countries.

Use the actual approved columns in the repository.

Do not include:

* large embeddings;
* sparse vectors;
* raw training text duplicates;
* model artifacts;
* image files;
* social-media fields that have already been removed;
* unnecessary external IDs.

## Popular ranking serving dataset

Create:

```text
data/serving/popular_movies.jsonl
```

Represent future logical records using:

* ranking_type;
* genre;
* movie_ids;
* scores;
* generated_at.

## Optional local interaction examples

Create a small schema example only:

```text
data/samples/interaction_event_examples.json
```

Do not fabricate a large synthetic interaction dataset.

---

# 14. Future DynamoDB preparation without AWS access

Prepare local serialization utilities only.

Create a module that can convert cleaned Python values into DynamoDB-compatible logical values later, but do not import or call boto3.

The local export must correctly preserve:

* strings;
* numbers;
* booleans;
* lists;
* maps;
* null handling.

Create:

```text
src/data/serving_export.py
```

Generate local JSONL files that are easy to load later.

Do not create tables.

Do not send network requests.

Do not require AWS credentials.

Create documentation:

```text
docs/local_serving_schema.md
```

Describe the future logical entities:

### Movies

* partition key candidate: movie_id;
* movie metadata required by the backend.

### Users

* partition key candidate: user_id;
* onboarding genres;
* onboarding movie IDs;
* profile status.

### Interactions

* partition key candidate: user_id;
* sort key candidate: interaction_timestamp#movie_id;
* movie ID;
* interaction type;
* interaction value;
* timestamp;
* session ID.

### RecommendationCache

* partition key candidate: user_id;
* sort key candidate: scenario;
* recommended movie IDs;
* scores;
* model version;
* generated time;
* expiration time.

### PopularMovies

* partition key candidate: ranking_type;
* sort key candidate: genre;
* ranked movie IDs;
* scores;
* generated time.

Clearly label these as proposed future schemas, not deployed resources.

---

# 15. Validation after processing

Run a second complete validation pass on all final outputs.

Check:

* canonical key uniqueness;
* foreign-key integrity;
* null constraints;
* data types;
* rating ranges;
* timestamp validity;
* duplicate records;
* movie feature coverage;
* serving-field completeness;
* top-rated ranking uniqueness;
* onboarding recommender behavior;
* interaction split leakage;
* JSON serialization;
* deterministic pipeline execution.

Create:

```text
reports/validation/final_validation.json
reports/validation/final_validation.md
```

The final report must include a PASS, WARNING, or FAIL status for every validation rule.

The pipeline must exit with a non-zero status when a critical validation rule fails.

Warnings must not be hidden.

---

# 16. Testing requirements

Use the testing framework already present in the repository. If none exists, use `pytest`.

Create tests for:

* schema detection;
* ID mapping;
* duplicate handling;
* missing-value handling;
* genre normalization;
* rating validation;
* movie-reference validation;
* weighted-rating calculation;
* top-rated ranking;
* content-based input validation;
* exclusion of selected movies;
* onboarding fallback behavior;
* interaction table construction;
* chronological splitting;
* leakage prevention;
* serving JSON serialization;
* deterministic output.

Use small fixtures rather than loading the complete dataset in every unit test.

Add at least one integration test that runs the local pipeline on a small sample.

Do not modify tests merely to make incorrect code pass.

Run all tests and report:

* command used;
* passed tests;
* failed tests;
* warnings;
* unresolved issues.

---

# 17. Documentation

Create or update:

```text
README.md
docs/data_pipeline.md
docs/data_dictionary.md
docs/id_mapping.md
docs/recommendation_scenarios.md
docs/local_serving_schema.md
docs/team_handoff.md
```

## README requirements

Include exact commands for:

* environment setup;
* dependency installation;
* data profiling;
* cleaning;
* feature generation;
* split generation;
* serving export;
* validation;
* testing;
* complete local pipeline.

## Recommendation scenario documentation

Explain:

1. Guest:

   * no tracking;
   * top-rated output only.

2. Onboarding:

   * uses selected movie IDs and genres;
   * does not use guest_recent_movie_ids.

3. Returning user:

   * uses available historical interactions.

## Team handoff documentation

Explain each output file and identify whether it is intended for:

* model training;
* backend serving;
* manual verification;
* future cloud upload;
* future DynamoDB ingestion.

---

# 18. Configuration

Create a readable configuration file such as:

```text
configs/data_pipeline.yaml
```

Include configurable values such as:

* input paths;
* output paths;
* random seed;
* minimum rating;
* maximum rating;
* minimum user interactions;
* minimum movie interactions;
* weighted-rating threshold strategy;
* top-K values;
* split strategy;
* text-feature configuration;
* enabled or disabled optional features.

Do not hide business rules inside source code when they can be configured.

Use a fixed random seed where randomness is unavoidable.

---

# 19. Execution phases

Implement and execute in this order:

## Phase A — Inspection

* inspect repository;
* inspect datasets;
* report schemas and conflicts;
* propose exact implementation plan.

## Phase B — Profiling

* build raw profiler;
* run raw validation;
* produce reports.

## Phase C — Cleaning

* implement modular cleaning;
* create processed tables;
* create rejection logs;
* create data dictionary.

## Phase D — Features and scenarios

* build movie content features;
* build guest top-rated rankings;
* implement onboarding content-based baseline;
* build returning-user interaction table.

## Phase E — Splits and serving exports

* create leakage-safe splits;
* create local serving files;
* create future logical schema documentation.

## Phase F — Validation and tests

* run final validation;
* run unit and integration tests;
* document results.

After each phase:

1. show files created or changed;
2. show commands executed;
3. summarize results;
4. report warnings or failures;
5. do not proceed past a critical validation failure without fixing or clearly documenting it.

---

# 20. Final stopping point

The task is complete only when the following local outputs are available and validated:

* cleaned movie data;
* cleaned interaction or rating data;
* movie content features;
* guest top-rated rankings;
* onboarding content-based baseline;
* returning-user interaction dataset;
* train, validation, and test splits;
* local serving datasets;
* local JSONL exports;
* data dictionary;
* ID mapping documentation;
* validation reports;
* passing tests;
* exact local execution commands.

At the end, provide a concise final summary containing:

* dataset files discovered;
* final tables created;
* row counts before and after cleaning;
* records removed and reasons;
* canonical ID decisions;
* feature datasets generated;
* split strategy;
* leakage-check results;
* guest ranking method;
* onboarding baseline method;
* known dataset limitations;
* tests passed and failed;
* files ready for future cloud upload;
* unresolved decisions that require human review.

Stop at this point.

Do not begin AWS, S3, DynamoDB, SageMaker, deployment, or cloud-cost work.
