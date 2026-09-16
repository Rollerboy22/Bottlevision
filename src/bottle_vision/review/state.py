"""Review state and human decisions for Bottle Vision."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ReviewAction(StrEnum):
    ACCEPT = "accept"
    CHANGE_CLASS = "change_class"
    REJECT = "reject"
    RESEGMENT = "resegment"


@dataclass
class ReviewState:
    """Mutable review state for one processed image."""

    image_id: str
    action: ReviewAction | None = None
    class_id: str | None = None
    notes: str = ""
    verified: bool = False
    instances: list[dict[str, Any]] = field(default_factory=list)

    def apply(
        self,
        action: ReviewAction,
        *,
        class_id: str | None = None,
        notes: str = "",
    ) -> "ReviewState":
        """Apply a review action while enforcing class requirements."""
        if action is ReviewAction.CHANGE_CLASS and not class_id:
            raise ValueError("class_id is required for CHANGE_CLASS")
        self.action = action
        self.class_id = class_id
        self.notes = notes
        self.verified = action in {ReviewAction.ACCEPT, ReviewAction.CHANGE_CLASS}
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable review metadata."""
        return {
            "image_id": self.image_id,
            "action": self.action.value if self.action else None,
            "class_id": self.class_id,
            "notes": self.notes,
            "verified": self.verified,
            "instances": self.instances,
        }
