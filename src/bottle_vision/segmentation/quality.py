"""Model-independent validation heuristics for binary segmentation masks."""

from __future__ import annotations

import numpy as np

from .types import MaskQuality


def score_mask(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    *,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.95,
) -> MaskQuality:
    """Validate and score a mask using conservative geometry-only signals.

    This is an independent gate, not a substitute for the segmentation model.
    It deliberately avoids image color so background color cannot influence the
    validation decision.
    """
    binary = np.asarray(mask, dtype=bool)
    height, width = image_shape
    reasons: list[str] = []

    if binary.ndim != 2:
        return MaskQuality(0.0, 0.0, 0.0, False, ("mask_not_2d",))
    if binary.shape != (height, width):
        return MaskQuality(0.0, 0.0, 0.0, False, ("mask_shape_mismatch",))
    if height <= 0 or width <= 0:
        return MaskQuality(0.0, 0.0, 0.0, False, ("invalid_image_shape",))

    area = int(binary.sum())
    total = height * width
    area_ratio = area / total
    if area == 0:
        reasons.append("empty_mask")
    if area_ratio < min_area_ratio:
        reasons.append("mask_too_small")
    if area_ratio > max_area_ratio:
        reasons.append("mask_too_large")

    border = np.zeros_like(binary)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_pixels = int((binary & border).sum())
    border_touch_ratio = border_pixels / area if area else 1.0

    # Touching the frame is not automatically wrong, but extensive contact is
    # a useful review signal for cropped/partial objects.
    if border_touch_ratio > 0.75:
        reasons.append("excessive_border_contact")

    area_score = 1.0 if not reasons[:2] else 0.0
    border_score = max(0.0, 1.0 - max(0.0, border_touch_ratio - 0.25) / 0.75)
    score = max(0.0, min(1.0, 0.75 * area_score + 0.25 * border_score))
    valid = not reasons
    return MaskQuality(score, area_ratio, border_touch_ratio, valid, tuple(reasons))
