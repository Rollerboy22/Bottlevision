"""In-memory contracts for segmentation results and masks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaskQuality:
    """Independent quality signals for one binary instance mask."""

    score: float
    area_ratio: float
    border_touch_ratio: float
    valid: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SegmentationInstance:
    """One bottle instance represented by a pixel mask, never by a bbox."""

    mask: np.ndarray
    confidence: float
    instance_id: int
    prompt: str = "bottle"
    model_name: str = "unknown"
    model_version: str = "unknown"
    quality: MaskQuality | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.ndim != 2:
            raise ValueError("Segmentation mask must be a 2D array")
        if mask.dtype != np.bool_:
            object.__setattr__(self, "mask", mask.astype(bool, copy=False))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Segmentation confidence must be between 0 and 1")
        if self.instance_id < 0:
            raise ValueError("instance_id must be non-negative")


@dataclass(frozen=True)
class SegmentationResult:
    """Result of a segmentation backend for one input image."""

    instances: tuple[SegmentationInstance, ...]
    image_shape: tuple[int, int]
    prompt: str = "bottle"
    model_name: str = "unknown"
    model_version: str = "unknown"
    error: str | None = None

    @property
    def has_instances(self) -> bool:
        return bool(self.instances)

    @property
    def all_quality_valid(self) -> bool:
        return all(instance.quality is not None and instance.quality.valid for instance in self.instances)
