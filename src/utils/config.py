"""Configuration loader — reads config.yaml and provides utilities."""

import os
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration file.
    
    Args:
        config_path: Optional path to config.yaml. Defaults to PROJECT_ROOT/config.yaml.
    
    Returns:
        Dictionary with all configuration parameters.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_path(config: dict, key: str) -> Path:
    """Get an absolute path from a config-relative path.
    
    Args:
        config: Config dictionary.
        key: Dot-separated key like 'data.raw_dir'.
    
    Returns:
        Absolute Path object.
    """
    keys = key.split(".")
    value = config
    for k in keys:
        value = value[k]
    
    path = PROJECT_ROOT / value
    path.mkdir(parents=True, exist_ok=True)
    return path
