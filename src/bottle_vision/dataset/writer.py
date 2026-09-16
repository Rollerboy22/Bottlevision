"""Verified dataset writer for Bottle Vision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bottle_vision.review.state import ReviewAction, ReviewState
from bottle_vision.segmentation.types import SegmentationResult


class VerifiedDatasetWriter:
    """Persist only human-verified image/mask pairs into the dataset tree."""

    def __init__(self, root: str | Path = "dataset") -> None:
        self.root = Path(root)
        self._ensure_tree()

    def _ensure_tree(self) -> None:
        for split in ("train", "val", "test"):
            (self.root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.root / "masks" / split).mkdir(parents=True, exist_ok=True)
            (self.root / "metadata" / split).mkdir(parents=True, exist_ok=True)
        (self.root / "queue").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def image_id(image: np.ndarray) -> str:
        """Generate a stable content id for an image."""
        array = np.asarray(image, dtype=np.uint8)
        return hashlib.sha256(array.tobytes()).hexdigest()[:16]

    def save_verified(self, image: np.ndarray, result: SegmentationResult, review: ReviewState, *, split: str = "train") -> list[Path]:
        """Save reviewed masks and metadata; reject unverified actions."""
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if not review.verified or review.action not in {ReviewAction.ACCEPT, ReviewAction.CHANGE_CLASS}:
            raise ValueError("Only verified ACCEPT or CHANGE_CLASS reviews may be saved")

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("image must have shape (height, width, channels)")
        if result.image_shape != array.shape[:2]:
            raise ValueError("image and segmentation result shapes do not match")
        if not result.instances:
            raise ValueError("Cannot save a verified sample without segmentation instances")

        image_id = review.image_id if review.image_id != "current" else self.image_id(array)
        image_path = self.root / "images" / split / f"{image_id}.png"
        Image.fromarray(array[:, :, :3].astype(np.uint8), mode="RGB").save(image_path)

        written: list[Path] = []
        instance_meta: list[dict[str, Any]] = []
        for item in result.instances:
            mask = np.asarray(item.mask, dtype=bool)
            if mask.shape != result.image_shape:
                raise ValueError("segmentation mask shape does not match image")
            mask_path = self.root / "masks" / split / f"{image_id}_{item.instance_id}.png"
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
            written.append(mask_path)
            instance_meta.append({
                "instance_id": item.instance_id,
                "mask_file": mask_path.name,
                "confidence": item.confidence,
                "quality_score": item.quality.score if item.quality else None,
                "model_name": item.model_name,
                "model_version": item.model_version,
            })

        metadata = {
            "schema_version": 1,
            "image_id": image_id,
            "image_file": image_path.name,
            "split": split,
            "verified": True,
            "review": review.to_dict(),
            "instances": instance_meta,
        }
        metadata_path = self.root / "metadata" / split / f"{image_id}.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.extend([image_path, metadata_path])
        return written

    def save_queue(self, review: ReviewState) -> Path:
        """Persist pending/rejected/resegment state outside training data."""
        status = review.action.value if review.action else "pending"
        path = self.root / "queue" / f"{review.image_id}_{status}.json"
        path.write_text(json.dumps(review.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path
