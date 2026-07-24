# Netflix-style movie recommendation data pipeline

This repository builds and validates a complete local data-processing pipeline from the seven CSV files in `movies_dataset/`. It stops before all cloud work: there are no cloud SDKs, credentials, uploads, resource definitions, or network calls.

The production interaction source is the full `ratings.csv` + `links.csv` pair. The `*_small.csv` files are profiled auxiliary subsets and are never mixed into training data.

## Environment setup

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Exact commands

Run raw profiling and validation:

```powershell
python scripts/profile_data.py --config configs/data_pipeline.yaml
```

Run cleaning and canonical table generation:

```powershell
python scripts/clean_data.py --config configs/data_pipeline.yaml
```

Run content features, returning-user features, guest rankings, and onboarding examples:

```powershell
python scripts/build_features.py --config configs/data_pipeline.yaml
```

Run chronological split generation and local serving exports:

```powershell
python scripts/build_splits_and_serving.py --config configs/data_pipeline.yaml
```

Run final validation:

```powershell
python scripts/validate_data.py --config configs/data_pipeline.yaml
```

Run all tests:

```powershell
python -m pytest -q
```

Run the complete local pipeline:

```powershell
python scripts/run_data_pipeline.py --config configs/data_pipeline.yaml
```

Rerun core transformations and compare deterministic artifact hashes:

```powershell
python scripts/check_determinism.py --config configs/data_pipeline.yaml
```

Any critical raw or final validation failure raises an error and exits non-zero. Warnings remain visible in Markdown and JSON reports.

## Recommendation scenarios

- Guest: precomputed weighted top-rated lists only; no user or session tracking.
- First login: TF-IDF content similarity from selected canonical movie IDs and genres, with invalid values ignored and an explicit top-rated fallback.
- Returning user: explicit historical ratings only, represented as `interaction_type = rating`.

See [data pipeline](docs/data_pipeline.md), [recommendation scenarios](docs/recommendation_scenarios.md), [data dictionary](docs/data_dictionary.md), [ID mapping](docs/id_mapping.md), [local serving schema](docs/local_serving_schema.md), and [team handoff](docs/team_handoff.md).

