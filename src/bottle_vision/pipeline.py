"""Model-independent orchestration for the Bottle Vision inference path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from bottle_vision.segmentation import SegmentationResult, build_segmenter


class Decision(StrEnum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True)
class PipelineResult:
    """Segmentation result plus a conservative automatic decision."""

    decision: Decision
    segmentation: SegmentationResult
    pipeline_version: str = "0.2.0"

    @property
    def instances(self):
        return self.segmentation.instances


def run_pipeline(image: np.ndarray, config: dict[str, Any]) -> PipelineResult:
    """Run configured segmentation and apply the project decision policy.

    Automatic acceptance is intentionally conservative: every returned
    instance must pass both confidence and independent mask-quality gates.
    Anything uncertain is REVIEW; an actual backend error is REJECT.
    """
    segmenter = build_segmenter(config)
    result = segmenter.segment(image)
    if result.error:
        return PipelineResult(Decision.REJECT, result)
    if not result.instances:
        return PipelineResult(Decision.REVIEW, result)

    all_accepted = all(bool(item.metadata.get("accepted_by_gate", False)) for item in result.instances)
    return PipelineResult(Decision.ACCEPT if all_accepted else Decision.REVIEW, result)


def empty_result(reason: str = "segmentation_not_implemented") -> PipelineResult:
    """Backward-compatible safe result for callers that need an empty state."""
    from bottle_vision.segmentation.types import SegmentationResult

    result = SegmentationResult(
        instances=(),
        image_shape=(0, 0),
        error=reason,
    )
    return PipelineResult(Decision.REVIEW, result)
