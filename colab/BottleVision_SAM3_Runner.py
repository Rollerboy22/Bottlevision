"""Thin Google Colab runner for the Bottle Vision SAM 3 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bottle_vision.config import load_config
from bottle_vision.pipeline import PipelineResult, run_pipeline
from bottle_vision.segmentation import SegmentationResult, make_review_views

ROOT = Path.cwd()
CONFIG_PATH = ROOT / "configs" / "default.yaml"


def load_bottle_vision_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return load_config(path)


def load_rgb_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def run_segmentation(image: np.ndarray, config: dict[str, Any] | None = None) -> SegmentationResult:
    """Run the real configured pipeline and return its canonical segmentation result."""
    cfg = config or load_bottle_vision_config()
    return run_pipeline(image, cfg).segmentation


def build_review_views(
    image: np.ndarray,
    result: SegmentationResult,
    *,
    alpha: float = 0.45,
) -> dict[str, np.ndarray]:
    return make_review_views(image, result, alpha=alpha)


def summarize_result(result: SegmentationResult) -> dict[str, Any]:
    """Return a JSON-friendly result summary."""
    decision = "reject" if result.error else (
        "review" if not result.instances else
        "accept" if all(bool(x.metadata.get("accepted_by_gate", False)) for x in result.instances)
        else "review"
    )
    return {
        "decision": decision,
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
                "accepted_by_gate": bool(item.metadata.get("accepted_by_gate", False)),
            }
            for item in result.instances
        ],
    }


if __name__ == "__main__":
    print("Bottle Vision Colab runner loaded.")
