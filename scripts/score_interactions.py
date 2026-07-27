"""Score an interaction event stream from JSON; the backend contract, runnable.

Usage:
    python scripts/score_interactions.py --demo
    python scripts/score_interactions.py --payload events.json
    echo '{"user_id": 1, "events": [...]}' | python scripts/score_interactions.py

Exists so the backend team can check what the model does with a given payload
without importing the package or loading any artifact. Same code path as
`src/recommenders/feedback.score_interaction_events`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config
from src.recommenders.feedback import (
    InteractionPayloadError,
    build_recommend_request,
    score_interaction_events,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model_serving.yaml")
    parser.add_argument("--payload", default=None, help="JSON file; omit to read stdin.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Score a built-in payload covering every supported event type.",
    )
    parser.add_argument(
        "--recommend-request",
        action="store_true",
        help="Also print the /model/recommend body built from this stream.",
    )
    return parser.parse_args(argv)


def demo_payload() -> dict[str, Any]:
    """One user, one of each event type, including the two new ones."""
    return {
        "user_id": 1,
        "onboarding_completed": True,
        # Fixed so the demo prints the same numbers on every machine.
        "as_of": "2026-07-27T12:00:00Z",
        "events": [
            {"movie_id": 862, "event_type": "share", "timestamp": "2026-07-27T11:00:00Z"},
            {
                "movie_id": 862,
                "event_type": "comment",
                "value": "positive",
                "timestamp": "2026-07-27T10:55:00Z",
            },
            {
                "movie_id": 862,
                "event_type": "watch",
                "value": 0.91,
                "timestamp": "2026-07-26T20:00:00Z",
            },
            {
                "movie_id": 550,
                "event_type": "comment",
                "value": -0.9,
                "timestamp": "2026-07-25T09:00:00Z",
            },
            {"movie_id": 550, "event_type": "click", "timestamp": "2026-07-25T08:59:00Z"},
            {
                "movie_id": 13,
                "event_type": "rating",
                "value": 4.5,
                "timestamp": "2026-06-27T12:00:00Z",
            },
            {"movie_id": 13, "event_type": "share", "timestamp": "2026-06-27T12:01:00Z"},
            {"movie_id": 13, "event_type": "share", "timestamp": "2026-06-27T12:02:00Z"},
            {"movie_id": 13, "event_type": "share", "timestamp": "2026-06-27T12:03:00Z"},
            {"movie_id": 13, "event_type": "share", "timestamp": "2026-06-27T12:04:00Z"},
            # Comment with no sentiment: dropped, not scored as mildly positive.
            # Reported under events_ignored_by_reason.missing_sentiment.
            {
                "movie_id": 424,
                "event_type": "comment",
                "timestamp": "2026-07-27T08:00:00Z",
            },
            {
                "movie_id": 680,
                "event_type": "add_to_watchlist",
                "timestamp": "2026-07-27T09:00:00Z",
            },
        ],
    }


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    model_config = load_config(REPOSITORY_ROOT / arguments.model_config)

    if arguments.demo:
        payload = demo_payload()
    elif arguments.payload:
        payload = json.loads(Path(arguments.payload).read_text(encoding="utf-8"))
    else:
        raw = sys.stdin.read().strip()
        if not raw:
            print(
                "Không có payload. Dùng --demo, --payload <file>, hoặc đưa JSON "
                "qua stdin.",
                file=sys.stderr,
            )
            return 2
        payload = json.loads(raw)

    try:
        result = score_interaction_events(payload, model_config)
        if arguments.recommend_request:
            result = {
                "profile": result,
                "recommend_request": build_recommend_request(payload, model_config),
            }
    except InteractionPayloadError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
