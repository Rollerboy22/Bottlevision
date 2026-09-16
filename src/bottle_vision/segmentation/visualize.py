"""Visualization helpers for human review of segmentation masks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import SegmentationInstance, SegmentationResult

# Fixed, high-contrast colors make neighboring instances easy to distinguish.
INSTANCE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 64, 64),
    (64, 160, 255),
    (64, 220, 120),
    (255, 190, 50),
    (190, 100, 255),
    (40, 220, 210),
    (255, 110, 180),
    (160, 220, 60),
)


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an image to an RGB uint8 array without mutating the input."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("Image must have shape (height, width, channels)")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array.copy()


def _color_for_instance(instance_id: int) -> tuple[int, int, int]:
    """Return a deterministic display color for an instance id."""
    return INSTANCE_PALETTE[instance_id % len(INSTANCE_PALETTE)]


def render_mask(
    result: SegmentationResult,
    *,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Render all instance masks as a color-coded RGB image."""
    height, width = result.image_shape
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[...] = background

    for instance in result.instances:
        mask = np.asarray(instance.mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(f"Mask shape {mask.shape} does not match result image shape")
        canvas[mask] = _color_for_instance(instance.instance_id)

    return canvas


def render_overlay(
    image: np.ndarray,
    result: SegmentationResult,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend color-coded instance masks over the original image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    base = _as_rgb_uint8(image)
    if base.shape[:2] != result.image_shape:
        raise ValueError("Image shape does not match segmentation result")

    overlay = base.astype(np.float32)
    for instance in result.instances:
        mask = np.asarray(instance.mask, dtype=bool)
        if mask.shape != result.image_shape:
            raise ValueError(f"Mask shape {mask.shape} does not match result image shape")
        color = np.asarray(_color_for_instance(instance.instance_id), dtype=np.float32)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color

    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_review_views(
    image: np.ndarray,
    result: SegmentationResult,
    *,
    alpha: float = 0.45,
) -> dict[str, np.ndarray]:
    """Return the three standard human-review views: Original, Mask, Overlay."""
    original = _as_rgb_uint8(image)
    if original.shape[:2] != result.image_shape:
        raise ValueError("Image shape does not match segmentation result")
    return {
        "Original": original,
        "Mask": render_mask(result),
        "Overlay": render_overlay(original, result, alpha=alpha),
    }


def iter_instance_masks(result: SegmentationResult) -> Sequence[np.ndarray]:
    """Expose masks in instance order for downstream review UIs."""
    return tuple(np.asarray(item.mask, dtype=bool).copy() for item in result.instances)
