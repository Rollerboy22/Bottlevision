"""Gradio review interface for Bottle Vision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from bottle_vision.config import load_config
from bottle_vision.dataset.writer import VerifiedDatasetWriter
from bottle_vision.segmentation import SegmentationResult, build_segmenter
from bottle_vision.segmentation.visualize import make_review_views
from .state import ReviewAction, ReviewState

ROOT = Path.cwd()
CONFIG_PATH = ROOT / "configs" / "default.yaml"
CLASSES_PATH = ROOT / "classes.json"


def load_classes(path: str | Path = CLASSES_PATH) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("classes", []))


def _summary(result: SegmentationResult) -> list[list[Any]]:
    rows = []
    for item in result.instances:
        quality = item.quality
        rows.append([
            item.instance_id,
            round(item.confidence, 3),
            round(quality.score, 3) if quality else None,
            quality.valid if quality else False,
            ", ".join(quality.reasons) if quality else "not_scored",
        ])
    return rows


def analyze_image(image: np.ndarray | None):
    """Segment an uploaded image and persist a durable review session."""
    if image is None:
        return None, None, None, [], None, gr.update(choices=[]), "No image selected."

    array = np.asarray(image)
    config = load_config(CONFIG_PATH)
    result = build_segmenter(config).segment(array)
    views = make_review_views(array, result)
    writer = VerifiedDatasetWriter(config.get("paths", {}).get("dataset", "dataset"))
    state = writer.save_pending(array, result)
    instance_ids = [str(item["instance_id"]) for item in state.instances]
    return (
        views["Original"],
        views["Mask"],
        views["Overlay"],
        _summary(result),
        state.to_dict(),
        gr.update(choices=instance_ids, value=instance_ids[0] if instance_ids else None),
        f"Pending review saved: {state.image_id}",
    )


def assign_class(instance_id: str | None, class_id: str | None, state_payload: dict[str, Any] | None):
    """Assign a class to one instance and persist it in the review session."""
    if not state_payload:
        return state_payload, "Analyze an image first."
    if instance_id is None or class_id is None:
        return state_payload, "Select an instance and a class."
    state = ReviewState(
        image_id=str(state_payload.get("image_id", "current")),
        instances=list(state_payload.get("instances", [])),
    )
    try:
        state.set_instance_class(int(instance_id), class_id)
        writer = VerifiedDatasetWriter(load_config(CONFIG_PATH).get("paths", {}).get("dataset", "dataset"))
        writer.update_queue_review(state)
    except (ValueError, FileNotFoundError) as exc:
        return state_payload, f"Class assignment failed: {exc}"
    return state.to_dict(), f"Instance {instance_id} → {class_id}"


def apply_review(
    action: str,
    class_id: str | None,
    notes: str,
    state_payload: dict[str, Any] | None,
    split: str,
):
    """Finalize a human review or keep it in the durable queue."""
    if not state_payload:
        return "No image has been analyzed yet.", state_payload, ""

    config = load_config(CONFIG_PATH)
    writer = VerifiedDatasetWriter(config.get("paths", {}).get("dataset", "dataset"))
    image_id = str(state_payload.get("image_id", "current"))
    try:
        image, result, stored_review = writer.load_pending(image_id)
        state = ReviewState(
            image_id=image_id,
            instances=stored_review.instances,
        )
        state.apply(ReviewAction(action), class_id=class_id or None, notes=notes or "")
        if state.verified:
            paths = writer.save_verified(image, result, state, split=split)
            message = f"VERIFIED and saved to {split}: {len(paths)} files. Queue cleared for {image_id}."
        else:
            writer.save_queue(state)
            message = f"Saved to review queue as {action}: {image_id}"
        return message, state.to_dict(), json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    except (ValueError, FileNotFoundError) as exc:
        return f"Review failed: {exc}", state_payload, ""


def build_app() -> gr.Blocks:
    """Build the standalone Gradio review application."""
    class_ids = [item["id"] for item in load_classes()]
    with gr.Blocks(title="Bottle Vision Review") as app:
        gr.Markdown("# Bottle Vision — Human Review")
        gr.Markdown("Only verified ACCEPT / CHANGE CLASS decisions enter train/val/test data.")

        with gr.Row():
            input_image = gr.Image(type="numpy", label="Input image")
            with gr.Column():
                original = gr.Image(label="Original", interactive=False)
                mask = gr.Image(label="Mask", interactive=False)
                overlay = gr.Image(label="Overlay", interactive=False)

        table = gr.Dataframe(
            headers=["Instance", "Confidence", "Quality", "Valid", "Reasons"],
            datatype=["number", "number", "number", "bool", "str"],
            label="Segmentation review",
            interactive=False,
        )
        analyze = gr.Button("Analyze / Rerun Segmentation", variant="primary")

        with gr.Row():
            instance_id = gr.Dropdown(choices=[], label="Instance to classify")
            class_id = gr.Dropdown(choices=class_ids, label="Class", allow_custom_value=False)
            assign = gr.Button("SET CLASS")
        notes = gr.Textbox(label="Notes", lines=2)
        split = gr.Dropdown(choices=["train", "val", "test"], value="train", label="Dataset split")

        with gr.Row():
            accept = gr.Button("ACCEPT")
            change = gr.Button("CHANGE CLASS")
            reject = gr.Button("REJECT")
            resegment = gr.Button("RESEGMENT")

        state = gr.State()
        status = gr.Markdown("Ready.")
        result = gr.Code(label="Review result", language="json")

        analyze.click(
            analyze_image,
            inputs=input_image,
            outputs=[original, mask, overlay, table, state, instance_id, status],
        )
        assign.click(
            assign_class,
            inputs=[instance_id, class_id, state],
            outputs=[state, status],
        )
        for button, action in [
            (accept, ReviewAction.ACCEPT.value),
            (change, ReviewAction.CHANGE_CLASS.value),
            (reject, ReviewAction.REJECT.value),
            (resegment, ReviewAction.RESEGMENT.value),
        ]:
            button.click(
                lambda class_id, notes, state, split, action=action: apply_review(
                    action, class_id, notes, state, split
                ),
                inputs=[class_id, notes, state, split],
                outputs=[status, state, result],
            )

    return app


def launch() -> None:
    """Launch the review UI."""
    build_app().launch()


if __name__ == "__main__":
    launch()
