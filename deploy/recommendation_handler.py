"""SageMaker real-time inference handler for the recommendation engine.

The container calls four functions in this file, in this order, and nothing
else: `model_fn` once when the worker starts, then `input_fn` / `predict_fn` /
`output_fn` for every request.

    invoke_endpoint(Body=...)
      -> input_fn      bytes -> dict
      -> predict_fn    dict  -> dict          (RecommendationEngine.recommend)
      -> output_fn     dict  -> bytes

`model_fn` is where the cost sits: it loads the TF-IDF matrix, both factor
matrices and the serving catalogue into memory. That happens once per worker,
not once per request, which is the only reason an always-on endpoint is
affordable to answer with at all.

Layout this file depends on, produced by `scripts/build_model_bundle.py`:

    /opt/ml/model/          <- model.tar.gz unpacked here by SageMaker
      configs/
      artifacts/
      data/
    /opt/ml/code/           <- source_dir, already on sys.path
      src/
      recommendation_handler.py

`src.data.config.load_config` derives the project root as
`<config file>.parent.parent`. Putting the configs at `<model_dir>/configs/`
therefore makes every relative path inside them resolve within the unpacked
tarball, with no path rewriting and no change to the engine. That is why the
bundle mirrors the repository layout instead of flattening it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONTENT_TYPE_JSON = "application/json"


def model_fn(model_dir: str) -> Any:
    """Build the engine once per worker and keep it resident."""
    from src.data.config import load_config
    from src.recommenders.engine import RecommendationEngine

    root = Path(model_dir)
    config = load_config(root / "configs" / "data_pipeline.yaml")
    model_config = load_config(root / "configs" / "model_serving.yaml")

    # Pinning a version is for rollback: set ALS_VERSION on the endpoint to
    # serve an older artifact without rebuilding the bundle. Empty means follow
    # artifacts/LATEST.json, which is the normal path.
    als_version = os.environ.get("ALS_VERSION") or None
    engine = RecommendationEngine(config, model_config, als_version)

    # Printed to CloudWatch so the log says which artifacts a worker actually
    # loaded. Without it, a bundle built from stale files is invisible.
    print(f"Engine ready, artifact_versions={engine.artifact_versions}", flush=True)
    return engine


def input_fn(request_body: Any, request_content_type: str = CONTENT_TYPE_JSON) -> dict:
    """Decode the request body into the mapping the engine expects."""
    if request_content_type and not request_content_type.startswith(CONTENT_TYPE_JSON):
        raise ValueError(
            f"Chỉ hỗ trợ {CONTENT_TYPE_JSON}, nhận được {request_content_type!r}."
        )
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    payload = json.loads(request_body)
    if not isinstance(payload, dict):
        raise ValueError("Request body phải là một JSON object.")
    return payload


def predict_fn(request: dict, engine: Any) -> dict:
    """Answer one request, converting malformed input into a payload.

    A raised exception becomes a 500 `ModelError` at the caller, which backend
    cannot tell apart from the endpoint genuinely being broken. A bad request is
    the caller's mistake and has to stay distinguishable from an outage, so it
    comes back as a normal 200 carrying an `error` field. Backend checks for that
    field; see `docs/interaction_events_api.md` for the request contract.

    `ValueError` is caught rather than only `RecommendationRequestError`, which
    subclasses it. The engine coerces several request fields directly -- e.g.
    `int(request.get("limit") ...)` in `recommend()` -- so a non-numeric `limit`
    arrives here as a bare `ValueError` and would otherwise 500 on what is
    plainly malformed input. `TypeError` covers the same coercions receiving a
    list or a dict.
    """
    try:
        return engine.recommend(request).to_dict()
    except (ValueError, TypeError) as error:
        return {"error": str(error), "error_type": "invalid_request"}


def output_fn(prediction: dict, accept: str = CONTENT_TYPE_JSON) -> tuple[str, str]:
    """Serialize the response. Always JSON, whatever the caller asked for."""
    return json.dumps(prediction, ensure_ascii=False), CONTENT_TYPE_JSON
