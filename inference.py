"""Serve recommendations for one request, or demonstrate all three scenarios.

Usage:
    python inference.py --demo
    python inference.py --request request.json
    echo '{"scenario_hint": "guest", "limit": 5}' | python inference.py

The engine loads its artifacts once and answers from memory. Backend is expected
to hold a single instance for the lifetime of the process rather than construct
one per request; see `src/recommenders/engine.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.data.config import load_config
from src.recommenders.engine import (
    RecommendationEngine,
    RecommendationRequestError,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/model_serving.yaml")
    parser.add_argument(
        "--request",
        default=None,
        help="Path to a JSON request; omit to read stdin.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run one request per scenario and print each response.",
    )
    parser.add_argument("--als-version", default=None)
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def demo_requests(engine: RecommendationEngine) -> list[dict[str, Any]]:
    """Build one request per scenario using identifiers that exist in the data."""
    popular = engine.guest.get_guest_recommendations(top_k=5)
    seeds = [int(value) for value in popular["movie_id"].to_numpy()[:3]]

    known_user = None
    if engine.collaborative is not None and len(engine.collaborative.user_ids):
        known_user = int(engine.collaborative.user_ids[0])

    return [
        {
            "user_id": None,
            "scenario_hint": "guest",
            "onboarding_completed": False,
            "valid_interaction_count_90d": 0,
            "selected_movie_ids": [],
            "selected_genres": [],
            "recent_interactions": [],
            "exclude_movie_ids": [],
            "limit": 5,
        },
        {
            "user_id": 999999999,
            "scenario_hint": "onboarding_user",
            "onboarding_completed": True,
            "valid_interaction_count_90d": 0,
            "selected_movie_ids": seeds,
            "selected_genres": ["Action"],
            "recent_interactions": [],
            "exclude_movie_ids": [],
            "limit": 5,
        },
        {
            "user_id": known_user,
            "scenario_hint": "returning_user",
            "onboarding_completed": True,
            "valid_interaction_count_90d": 25,
            "selected_movie_ids": [],
            "selected_genres": ["Drama"],
            "recent_interactions": [
                {"movie_id": seed, "event_type": "rating", "value": 4.5}
                for seed in seeds
            ],
            "exclude_movie_ids": [],
            "limit": 5,
        },
    ]


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    config = load_config(arguments.config)
    model_config = load_config(arguments.model_config)

    engine = RecommendationEngine(config, model_config, arguments.als_version)
    print(
        f"Artifact đang dùng: {engine.artifact_versions}",
        file=sys.stderr,
        flush=True,
    )

    if arguments.demo:
        for request in demo_requests(engine):
            print(f"--- scenario_hint = {request['scenario_hint']} ---")
            try:
                response = engine.recommend(request)
            except RecommendationRequestError as error:
                print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
                continue
            print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
            print()
        return 0

    if arguments.request:
        payload = json.loads(Path(arguments.request).read_text(encoding="utf-8"))
    else:
        raw = sys.stdin.read().strip()
        if not raw:
            print(
                "Không có request. Dùng --demo, --request <file>, hoặc đưa JSON "
                "qua stdin.",
                file=sys.stderr,
            )
            return 2
        payload = json.loads(raw)

    try:
        response = engine.recommend(payload)
    except RecommendationRequestError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
