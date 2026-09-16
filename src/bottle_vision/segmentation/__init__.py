"""Segmentation foundation for Bottle Vision."""

from .base import Segmenter, UnconfiguredSegmenter, build_segmenter
from .quality import score_mask
from .sam3 import Sam3Segmenter
from .types import MaskQuality, SegmentationInstance, SegmentationResult
from .visualize import iter_instance_masks, make_review_views, render_mask, render_overlay

__all__ = [
    "MaskQuality",
    "Sam3Segmenter",
    "Segmenter",
    "SegmentationInstance",
    "SegmentationResult",
    "UnconfiguredSegmenter",
    "build_segmenter",
    "iter_instance_masks",
    "make_review_views",
    "render_mask",
    "render_overlay",
    "score_mask",
]
