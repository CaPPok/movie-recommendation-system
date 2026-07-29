"""Call the deployed endpoint the way backend will call it.

Usage:
    python scripts/invoke_endpoint.py --demo
    python scripts/invoke_endpoint.py --request request.json
    echo '{"scenario_hint": "guest", "limit": 5}' | python scripts/invoke_endpoint.py

This is the counterpart to `inference.py`, which runs the same engine in-process.
Same request shape, same response shape; the only difference is that the work
happens on the endpoint. Running both on one request is how a suspected
serving-side difference gets confirmed or ruled out.

The three demo requests cover the scenarios the engine routes between: a guest
with no history, a first-login user with onboarding selections, and a returning
user with recent interactions. `--demo` needs no artifacts locally -- the movie
ids it seeds with come back from the endpoint's own guest response.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws-config", default="configs/aws.yaml")
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument("--request", default=None, help="Path to a JSON request.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Send one request per scenario and print each response.",
    )
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def invoke(runtime: Any, endpoint_name: str, payload: dict) -> tuple[dict, float]:
    started = time.time()
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    body = json.loads(response["Body"].read().decode("utf-8"))
    return body, (time.time() - started) * 1000.0


def base_request(limit: int) -> dict:
    return {
        "user_id": None,
        "scenario_hint": "guest",
        "onboarding_completed": False,
        "valid_interaction_count_90d": 0,
        "selected_movie_ids": [],
        "selected_genres": [],
        "recent_interactions": [],
        "exclude_movie_ids": [],
        "limit": limit,
    }


def demo_requests(seeds: list[int], limit: int) -> list[tuple[str, dict]]:
    onboarding = base_request(limit) | {
        "user_id": 999999999,
        "scenario_hint": "onboarding_user",
        "onboarding_completed": True,
        "selected_movie_ids": seeds,
        "selected_genres": ["Action"],
    }
    returning = base_request(limit) | {
        # A user id the ALS matrix knows. 1 exists in the MovieLens history the
        # model was trained on; a stranger would be answered by the fallback
        # path instead, which is not what this request is meant to exercise.
        "user_id": 1,
        "scenario_hint": "returning_user",
        "onboarding_completed": True,
        "valid_interaction_count_90d": 25,
        "selected_genres": ["Drama"],
        "recent_interactions": [
            {"movie_id": seed, "event_type": "rating", "value": 4.5} for seed in seeds
        ],
    }
    return [("onboarding_user", onboarding), ("returning_user", returning)]


def report(label: str, body: dict, milliseconds: float) -> None:
    print(f"=== {label} ===")
    if "error" in body:
        print(f"  error: {body['error']}  ({milliseconds:.0f} ms)")
        print()
        return
    print(f"  scenario_applied : {body['scenario_applied']}")
    print(f"  fallback_level   : {body['fallback_level']}")
    print(f"  artifact_versions: {body['artifact_versions']}")
    print(f"  kết quả          : {len(body['recommendations'])}  ({milliseconds:.0f} ms)")
    for item in body["recommendations"][:3]:
        print(f"    {item}")
    print()


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    aws_config = load_config(REPOSITORY_ROOT / arguments.aws_config)

    from src.aws import s3_sync

    endpoint_name = arguments.endpoint_name or str(
        aws_config["sagemaker"]["endpoint"]["name"]
    )
    runtime = s3_sync.client(aws_config, "sagemaker-runtime")
    print(f"Endpoint: {endpoint_name}\n")

    if arguments.demo:
        body, milliseconds = invoke(runtime, endpoint_name, base_request(arguments.limit))
        report("guest", body, milliseconds)
        seeds = [
            int(item["movie_id"]) for item in body.get("recommendations", [])[:3]
        ]
        for label, payload in demo_requests(seeds, arguments.limit):
            body, milliseconds = invoke(runtime, endpoint_name, payload)
            report(label, body, milliseconds)
        return 0

    if arguments.request:
        payload = json.loads(Path(arguments.request).read_text(encoding="utf-8"))
    else:
        raw = sys.stdin.read().strip()
        if not raw:
            print("Không có request. Dùng --demo, --request <file>, hoặc stdin.", file=sys.stderr)
            return 2
        payload = json.loads(raw)

    body, milliseconds = invoke(runtime, endpoint_name, payload)
    print(json.dumps(body, ensure_ascii=False, indent=2))
    print(f"\n{milliseconds:.0f} ms", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
