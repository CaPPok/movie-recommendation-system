from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.config import load_config
from src.pipeline import run_pipeline


def _metadata_row(
    movie_id: object,
    title: object,
    genres: list[str],
    *,
    overview: object = "A useful overview",
) -> dict[str, object]:
    return {
        "adult": "False",
        "belongs_to_collection": None,
        "budget": "0",
        "genres": repr(
            [{"id": index + 1, "name": genre} for index, genre in enumerate(genres)]
        ),
        "homepage": None,
        "id": movie_id,
        "imdb_id": f"tt{int(movie_id):07d}" if isinstance(movie_id, int) else None,
        "original_language": " en ",
        "original_title": title,
        "overview": overview,
        "popularity": "2.5",
        "poster_path": f"/{movie_id}.jpg",
        "production_companies": repr([{"id": 1, "name": " Example Studio "}]),
        "production_countries": repr(
            [{"iso_3166_1": "us", "name": "United States"}]
        ),
        "release_date": "2001-01-01",
        "revenue": 0.0,
        "runtime": 100.0,
        "spoken_languages": repr([{"iso_639_1": "en", "name": "English"}]),
        "status": "Released",
        "tagline": None,
        "title": title,
        "video": False,
        "vote_average": 7.0,
        "vote_count": 10.0,
    }


@pytest.fixture(scope="session")
def sample_project(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    root = tmp_path_factory.mktemp("movie_pipeline")
    raw = root / "movies_dataset"
    raw.mkdir()
    rows = [
        _metadata_row(10, "Alpha", ["Drama", "Thriller"]),
        _metadata_row(11, "Beta", ["Animation", "Family"]),
        _metadata_row(12, "Gamma", ["Drama", "Crime"]),
        _metadata_row(13, "Delta", ["Comedy"]),
        _metadata_row(14, "Epsilon", ["Animation", "Comedy"]),
        _metadata_row(15, "Zeta", ["Crime"]),
    ]
    rows.append(rows[-1].copy())
    rows.append(_metadata_row("not-an-id", "Broken", ["Drama"]))
    rows.append(_metadata_row(16, None, ["Drama"]))
    pd.DataFrame(rows).to_csv(raw / "movies_metadata.csv", index=False)

    ids = [10, 11, 12, 13, 14, 15]
    pd.DataFrame(
        {
            "id": ids,
            "keywords": [
                repr([{"id": value, "name": name}])
                for value, name in zip(
                    ids,
                    ["mystery", "toy", "detective", "funny", "cartoon", "crime"],
                    strict=True,
                )
            ],
        }
    ).to_csv(raw / "keywords.csv", index=False)
    pd.DataFrame(
        {
            "cast": [
                repr([{"name": f"Actor {value}", "order": 0}]) for value in ids
            ],
            "crew": [
                repr([{"name": f"Director {value}", "job": "Director"}])
                for value in ids
            ],
            "id": ids,
        }
    ).to_csv(raw / "credits.csv", index=False)
    links = pd.DataFrame(
        {
            "movieId": range(1, 9),
            "imdbId": range(1001, 1009),
            "tmdbId": [10, 11, 12, 13, 14, 15, None, 999],
        }
    )
    links.to_csv(raw / "links.csv", index=False)
    links.head(6).to_csv(raw / "links_small.csv", index=False)
    ratings = pd.DataFrame(
        [
            (1, 1, 4.0, 100),
            (1, 2, 5.0, 200),
            (1, 3, 3.0, 300),
            (1, 7, 2.0, 400),
            (2, 1, 3.0, 100),
            (2, 3, 4.0, 200),
            (2, 4, 4.5, 300),
            (3, 2, 2.0, 100),
            (3, 4, 3.0, 200),
            (4, 2, 4.0, 100),
            (4, 3, 4.5, 200),
            (4, 5, 5.0, 300),
            (4, 6, 4.0, 400),
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    ratings.to_csv(raw / "ratings.csv", index=False)
    ratings.head(8).to_csv(raw / "ratings_small.csv", index=False)

    main = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "data_pipeline.yaml").read_text(
            encoding="utf-8"
        )
    )
    main["processing"]["chunk_size"] = 5
    main["processing"]["sample_rows"] = 5
    main["ranking"].update(
        {
            "top_k_all": 5,
            "top_k_per_genre": 3,
            "minimum_genre_candidates": 1,
        }
    )
    main["content_features"]["vectorizer"].update(
        {"max_features": 100, "min_df": 1, "max_df": 1.0}
    )
    config_dir = root / "configs"
    config_dir.mkdir()
    config_path = config_dir / "data_pipeline.yaml"
    config_path.write_text(yaml.safe_dump(main, sort_keys=False), encoding="utf-8")
    docs = root / "docs"
    docs.mkdir()
    (docs / "local_serving_schema.md").write_text(
        "# Proposed local serving schema\n", encoding="utf-8"
    )
    config = load_config(config_path)
    run_pipeline(config)
    return root, config

