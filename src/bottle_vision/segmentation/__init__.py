"""Segmentation foundation for Bottle Vision."""

from .base import Segmenter, UnconfiguredSegmenter, build_segmenter
from .quality import score_mask
from .sam3 import Sam3Segmenter
from .types import MaskQuality, SegmentationInstance, SegmentationResult

__all__ = [
    "MaskQuality",
    "Sam3Segmenter",
    "Segmenter",
    "SegmentationInstance",
    "SegmentationResult",
    "UnconfiguredSegmenter",
    "build_segmenter",
    "score_mask",
]
