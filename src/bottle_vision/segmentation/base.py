"""Segmentation backend interface and safe placeholder implementation."""

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
    """Safe backend used until a real foundation model is configured."""

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
    """Build the configured backend; currently only the safe stub is available."""
    segmentation = config.get("segmentation", {})
    preferred = str(segmentation.get("preferred_model", "sam3")).lower()
    if preferred in {"sam3", "sam2"}:
        return UnconfiguredSegmenter()
    return UnconfiguredSegmenter()
