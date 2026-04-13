"""
Config loading utilities.
Loads yaml configs with dataset-specific overrides merged on top of base.

Usage:
    cfg = load_config("configs/nextqa.yaml")
    print(cfg.training.learning_rate)
"""
from __future__ import annotations

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str, overrides: list[str] | None = None) -> DictConfig:
    """
    Load a yaml config file, merging with base.yaml if a 'defaults' key is present.

    Args:
        config_path: Path to the dataset-specific yaml config.
        overrides:   Optional list of CLI-style overrides, e.g. ["training.lr=1e-3"].

    Returns:
        Merged OmegaConf DictConfig object.

    Example:
        >>> cfg = load_config("configs/nextqa.yaml", overrides=["training.epochs=5"])
        >>> cfg.training.epochs
        5
    """
    config_path = Path(config_path)
    cfg = OmegaConf.load(config_path)

    # Merge with base if 'defaults' key points to it
    if "defaults" in cfg:
        base_path = config_path.parent / "base.yaml"
        base_cfg = OmegaConf.load(base_path)
        cfg = OmegaConf.merge(base_cfg, cfg)
        OmegaConf.set_struct(cfg, False)

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg
