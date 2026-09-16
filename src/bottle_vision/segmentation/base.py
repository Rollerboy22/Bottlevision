"""Segmentation backend interface and safe backend selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .types import SegmentationResult


class Segmenter(ABC):
    """Backend contract for SAM 3, SAM 2, or another segmentation model."""

    model_name = "unknown"
    model_version = "unknown"
    prompt = "bottle"

    @abstractmethod
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Return pixel-level instance masks for an image."""
        raise NotImplementedError


class UnconfiguredSegmenter(Segmenter):
    """Safe backend used when no model runtime is configured."""

    model_name = "unconfigured"
    model_version = "0"

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Return no instances instead of fabricating a segmentation."""
        array = np.asarray(image)
        if array.ndim < 2:
            raise ValueError("Image must have at least two dimensions")
        return SegmentationResult(
            instances=(),
            image_shape=(int(array.shape[0]), int(array.shape[1])),
            prompt=self.prompt,
            model_name=self.model_name,
            model_version=self.model_version,
            error="segmentation_backend_not_configured",
        )


def build_segmenter(config: dict[str, Any]) -> Segmenter:
    """Build the configured segmentation backend without importing heavy ML packages."""
    segmentation = config.get("segmentation", {})
    quality = config.get("quality", {})
    preferred = str(segmentation.get("preferred_model", "sam3")).lower()

    if preferred == "sam3":
        from .sam3 import Sam3Segmenter

        return Sam3Segmenter(
            checkpoint_path=segmentation.get("checkpoint_path"),
            device=str(segmentation.get("device", "cuda")),
            prompt=str(segmentation.get("prompt", "bottle")),
            min_confidence=float(segmentation.get("min_confidence", 0.50)),
            min_quality_score=float(segmentation.get("min_quality_score", 0.50)),
            min_area_ratio=float(quality.get("min_area_ratio", 0.001)),
            max_area_ratio=float(quality.get("max_area_ratio", 0.95)),
            load_from_hf=bool(segmentation.get("load_from_hf", True)),
        )

    # SAM 2 and other backends remain explicit future adapters. Returning the
    # safe stub is preferable to silently substituting a different model.
    return UnconfiguredSegmenter()
