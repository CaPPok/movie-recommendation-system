"""Configuration loading and repository-relative path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML and attach the repository root used for all relative paths."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    config["_config_path"] = path
    config["_project_root"] = path.parent.parent
    return config


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a path relative to the repository, never to a machine path."""
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def input_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a configured raw input file."""
    raw_dir = project_path(config, config["inputs"]["raw_dir"])
    return raw_dir / config["inputs"][key]


def output_dir(config: dict[str, Any], key: str, create: bool = True) -> Path:
    """Resolve and optionally create a configured output directory."""
    path = project_path(config, config["outputs"][key])
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

