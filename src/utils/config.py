"""
Config loader for QUEST.

Merges base.yaml with a dataset-specific override yaml using a simple
recursive dict merge. No external dependency (no Hydra/OmegaConf needed).

Usage:
    cfg = load_config("configs/nextqa.yaml")  # base is auto-loaded
    cfg.data.fps       # dot-access on any depth
    cfg["data"]["fps"] # or dict-style
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# DotDict: recursive dot-access dict
# ---------------------------------------------------------------------------

class DotDict(dict):
    """A dict subclass that allows attribute-style access at every depth."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'") from None
        return DotDict(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins on conflicts)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(dataset_config: str | Path, base_config: str | Path | None = None) -> DotDict:
    """
    Load and merge config files.

    Args:
        dataset_config: Path to the dataset-specific yaml (e.g. configs/nextqa.yaml).
        base_config:    Path to base.yaml. Defaults to configs/base.yaml relative
                        to the project root (two parents above this file).

    Returns:
        DotDict with merged configuration.
    """
    dataset_config = Path(dataset_config)
    if not dataset_config.exists():
        raise FileNotFoundError(f"Config not found: {dataset_config}")

    if base_config is None:
        # Resolve relative to this file's location: src/utils/ → ../../configs/
        base_config = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"

    base_config = Path(base_config)
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    base = _load_yaml(base_config)
    override = _load_yaml(dataset_config)
    merged = _deep_merge(base, override)
    return DotDict(merged)


def ensure_dirs(cfg: DotDict) -> None:
    """Create all output directories declared in cfg.paths if they don't exist."""
    paths = cfg.get("paths", {})
    for key, p in paths.items():
        if key != "data_root":
            Path(p).mkdir(parents=True, exist_ok=True)