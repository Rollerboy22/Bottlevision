"""Core pipeline contracts.

Model-specific implementations will be added later. The foundation keeps
segmentation and classification stages explicit so they cannot be silently
replaced by bounding-box-only logic.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Decision(StrEnum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True)
class PipelineResult:
    """Model-independent result passed between UI, storage, and training."""

    decision: Decision
    instances: list[dict[str, Any]]
    pipeline_version: str = "0.1.0"


def empty_result(reason: str = "segmentation_not_implemented") -> PipelineResult:
    """Return a safe REVIEW result until a real segmenter is configured."""
    return PipelineResult(
        decision=Decision.REVIEW,
        instances=[{"status": reason}],
    )
