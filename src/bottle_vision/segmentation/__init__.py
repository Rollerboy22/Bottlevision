"""Segmentation foundation for Bottle Vision."""

from .base import Segmenter, UnconfiguredSegmenter, build_segmenter
from .quality import score_mask
from .types import MaskQuality, SegmentationInstance, SegmentationResult

__all__ = [
    "MaskQuality",
    "Segmenter",
    "SegmentationInstance",
    "SegmentationResult",
    "UnconfiguredSegmenter",
    "build_segmenter",
    "score_mask",
]
