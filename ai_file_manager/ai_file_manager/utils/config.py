"""
utils/config.py — YAML configuration loader with dot-access
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any


class Config:
    """Recursive dot-access wrapper over a YAML-loaded dict."""

    def __init__(self, data: dict):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list):
                setattr(self, key, value)
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({self._data})"


_config_instance: Config | None = None


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML file. Defaults to config/settings.yaml."""
    global _config_instance
    if _config_instance is not None:
        return _config_instance

    if path is None:
        base = Path(__file__).resolve().parent.parent
        path = base / "config" / "settings.yaml"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _config_instance = Config(raw)
    return _config_instance


def reload_config(path: str | Path | None = None) -> Config:
    """Force reload config from disk."""
    global _config_instance
    _config_instance = None
    return load_config(path)
