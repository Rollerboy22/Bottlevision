"""Persistent JSON storage primitives for Bottle Vision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(path: str | Path, payload: Any) -> None:
    """Atomically write JSON data, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def load_json(path: str | Path) -> Any:
    """Load JSON data from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
