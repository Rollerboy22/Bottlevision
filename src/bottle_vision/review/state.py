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

    def set_instance_class(self, instance_id: int, class_id: str) -> "ReviewState":
        """Assign a human-verified class to one segmentation instance."""
        if not class_id:
            raise ValueError("class_id is required")
        for instance in self.instances:
            if int(instance.get("instance_id", -1)) == int(instance_id):
                instance["class_id"] = class_id
                return self
        raise ValueError(f"Unknown instance_id: {instance_id}")

    def all_instances_classified(self) -> bool:
        """Return whether every segmented instance has a class assignment."""
        return bool(self.instances) and all(bool(item.get("class_id")) for item in self.instances)

    def apply(
        self,
        action: ReviewAction,
        *,
        class_id: str | None = None,
        notes: str = "",
    ) -> "ReviewState":
        """Apply a review action while enforcing verification requirements."""
        if action in {ReviewAction.ACCEPT, ReviewAction.CHANGE_CLASS} and not self.all_instances_classified():
            if class_id:
                for instance in self.instances:
                    instance["class_id"] = class_id
            else:
                raise ValueError("Every instance must have a class before verification")
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
