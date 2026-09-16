"""Minimal Google Colab runner for the Bottle Vision SAM 3 pipeline.

This module intentionally keeps the notebook layer thin: configuration and
model loading live in the package, while Colab only installs dependencies,
loads an image, runs segmentation, and prints review-safe results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bottle_vision.config import load_config
from bottle_vision.segmentation import SegmentationResult, build_segmenter


ROOT = Path.cwd()
CONFIG_PATH = ROOT / "configs" / "default.yaml"


def load_bottle_vision_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the repository configuration."""
    return load_config(path)


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image as HxWx3 uint8 RGB data."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def run_segmentation(image: np.ndarray, config: dict[str, Any] | None = None) -> SegmentationResult:
    """Run the configured segmenter and return pixel masks plus quality signals."""
    cfg = config or load_bottle_vision_config()
    segmenter = build_segmenter(cfg)
    return segmenter.segment(image)


def summarize_result(result: SegmentationResult) -> dict[str, Any]:
    """Return a JSON-friendly summary suitable for Colab output."""
    return {
        "model": result.model_name,
        "model_version": result.model_version,
        "prompt": result.prompt,
        "image_shape": list(result.image_shape),
        "instance_count": len(result.instances),
        "error": result.error,
        "instances": [
            {
                "instance_id": item.instance_id,
                "confidence": item.confidence,
                "quality_score": item.quality.score if item.quality else None,
                "quality_valid": item.quality.valid if item.quality else False,
                "quality_reasons": list(item.quality.reasons) if item.quality else ["not_scored"],
            }
            for item in result.instances
        ],
    }


if __name__ == "__main__":
    print("Bottle Vision Colab runner loaded. Call run_segmentation(image) from the notebook.")
