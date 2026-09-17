"""Public configuration API.

The project previously contained both ``config.py`` and ``config/``. Python
resolves the package directory first, which made ``from bottle_vision.config
import load_config`` fail because the package initializer was empty. Keep the
public API here so all callers have one stable import path.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its mapping."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return data


__all__ = ["load_config"]
