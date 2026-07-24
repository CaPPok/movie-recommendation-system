"""Deterministic JSON and Markdown report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_value(value: Any) -> Any:
    """Convert pandas/numpy/path values into JSON-safe native values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write stable, UTF-8, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_value(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def truncate(value: Any, limit: int = 120) -> str:
    """Return a compact single-line value for Markdown reports."""
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= limit else f"{text[: limit - 3]}..."

