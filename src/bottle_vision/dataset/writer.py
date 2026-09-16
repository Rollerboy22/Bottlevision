"""Persistent verified dataset and review-queue storage for Bottle Vision."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bottle_vision.review.state import ReviewAction, ReviewState
from bottle_vision.segmentation.types import SegmentationInstance, SegmentationResult


class VerifiedDatasetWriter:
    """Persist reviewed samples and keep non-verified work outside training data."""

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

    def save_pending(self, image: np.ndarray, result: SegmentationResult) -> ReviewState:
        """Persist an analyzed sample so a browser refresh/restart cannot lose it."""
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("image must have shape (height, width, channels)")
        if result.image_shape != array.shape[:2]:
            raise ValueError("image and segmentation result shapes do not match")

        image_id = self.image_id(array)
        session_dir = self.root / "queue" / image_id
        masks_dir = session_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array[:, :, :3].astype(np.uint8), mode="RGB").save(session_dir / "image.png")

        instances: list[dict[str, Any]] = []
        for item in result.instances:
            mask = np.asarray(item.mask, dtype=bool)
            if mask.shape != result.image_shape:
                raise ValueError("segmentation mask shape does not match image")
            mask_name = f"{item.instance_id}.png"
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(masks_dir / mask_name)
            instances.append({
                "instance_id": item.instance_id,
                "mask_file": mask_name,
                "confidence": item.confidence,
                "quality_score": item.quality.score if item.quality else None,
                "quality_valid": item.quality.valid if item.quality else False,
                "quality_reasons": list(item.quality.reasons) if item.quality else [],
                "model_name": item.model_name,
                "model_version": item.model_version,
            })

        payload = {
            "schema_version": 1,
            "image_id": image_id,
            "image_shape": list(result.image_shape),
            "prompt": result.prompt,
            "model_name": result.model_name,
            "model_version": result.model_version,
            "error": result.error,
            "instances": instances,
            "review": ReviewState(
                image_id=image_id,
                instances=[
                    {"instance_id": item["instance_id"], "confidence": item["confidence"], "quality_score": item["quality_score"]}
                    for item in instances
                ],
            ).to_dict(),
        }
        (session_dir / "session.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return ReviewState(image_id=image_id, instances=payload["review"]["instances"])

    def load_pending(self, image_id: str) -> tuple[np.ndarray, SegmentationResult, ReviewState]:
        """Load a durable review session from the queue."""
        session_dir = self.root / "queue" / image_id
        payload = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        image = np.asarray(Image.open(session_dir / "image.png").convert("RGB"), dtype=np.uint8)
        instances: list[SegmentationInstance] = []
        review_instances: list[dict[str, Any]] = []
        for item in payload["instances"]:
            mask = np.asarray(Image.open(session_dir / "masks" / item["mask_file"]).convert("L")) > 127
            from bottle_vision.segmentation.types import MaskQuality
            quality = None
            if item.get("quality_score") is not None:
                quality = MaskQuality(
                    score=float(item["quality_score"]),
                    area_ratio=0.0,
                    border_touch_ratio=0.0,
                    valid=bool(item.get("quality_valid", False)),
                    reasons=tuple(item.get("quality_reasons", [])),
                )
            instances.append(SegmentationInstance(
                mask=mask,
                confidence=float(item["confidence"]),
                instance_id=int(item["instance_id"]),
                prompt=payload.get("prompt", "bottle"),
                model_name=item.get("model_name", payload.get("model_name", "unknown")),
                model_version=item.get("model_version", payload.get("model_version", "unknown")),
                quality=quality,
            ))
            review_instances.append({
                "instance_id": int(item["instance_id"]),
                "confidence": float(item["confidence"]),
                "quality_score": item.get("quality_score"),
            })
        result = SegmentationResult(
            instances=tuple(instances),
            image_shape=tuple(payload["image_shape"]),
            prompt=payload.get("prompt", "bottle"),
            model_name=payload.get("model_name", "unknown"),
            model_version=payload.get("model_version", "unknown"),
            error=payload.get("error"),
        )
        review_payload = payload.get("review", {})
        review = ReviewState(
            image_id=image_id,
            action=ReviewAction(review_payload["action"]) if review_payload.get("action") else None,
            class_id=review_payload.get("class_id"),
            notes=review_payload.get("notes", ""),
            verified=bool(review_payload.get("verified", False)),
            instances=review_payload.get("instances", review_instances),
        )
        return image, result, review

    def save_verified(
        self,
        image: np.ndarray,
        result: SegmentationResult,
        review: ReviewState,
        *,
        split: str = "train",
    ) -> list[Path]:
        """Save only human-verified image/mask pairs into the selected split."""
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if not review.verified or review.action not in {ReviewAction.ACCEPT, ReviewAction.CHANGE_CLASS}:
            raise ValueError("Only verified ACCEPT or CHANGE_CLASS reviews may be saved")
        if not review.all_instances_classified():
            raise ValueError("Every instance must have a class before saving")

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
            review_item = next(x for x in review.instances if int(x["instance_id"]) == item.instance_id)
            instance_meta.append({
                "instance_id": item.instance_id,
                "class_id": review_item["class_id"],
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
        self.mark_completed(image_id)
        return written

    def update_queue_review(self, review: ReviewState) -> Path:
        """Persist the current review decision inside the durable session."""
        session_dir = self.root / "queue" / review.image_id
        session_path = session_dir / "session.json"
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        payload["review"] = review.to_dict()
        session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return session_path

    def mark_completed(self, image_id: str) -> None:
        """Remove a successfully verified session from the pending queue."""
        shutil.rmtree(self.root / "queue" / image_id, ignore_errors=True)

    def save_queue(self, review: ReviewState) -> Path:
        """Persist pending/rejected/resegment state outside training data."""
        return self.update_queue_review(review)
