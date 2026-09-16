"""Gradio review interface for Bottle Vision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from bottle_vision.config import load_config
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
    """Segment an uploaded image and prepare review views."""
    if image is None:
        return None, None, None, [], None

    config = load_config(CONFIG_PATH)
    result = build_segmenter(config).segment(np.asarray(image))
    views = make_review_views(np.asarray(image), result)
    state = ReviewState(image_id="current", instances=[
        {
            "instance_id": item.instance_id,
            "confidence": item.confidence,
            "quality_score": item.quality.score if item.quality else None,
        }
        for item in result.instances
    ])
    return views["Original"], views["Mask"], views["Overlay"], _summary(result), state.to_dict()


def apply_review(action: str, class_id: str | None, notes: str, state_payload: dict[str, Any] | None):
    """Apply a human review decision to the current state."""
    if not state_payload:
        return "No image has been analyzed yet."
    state = ReviewState(
        image_id=str(state_payload.get("image_id", "current")),
        instances=list(state_payload.get("instances", [])),
    )
    state.apply(ReviewAction(action), class_id=class_id or None, notes=notes or "")
    return json.dumps(state.to_dict(), ensure_ascii=False, indent=2)


def build_app() -> gr.Blocks:
    """Build the standalone Gradio review application."""
    class_ids = [item["id"] for item in load_classes()]
    with gr.Blocks(title="Bottle Vision Review") as app:
        gr.Markdown("# Bottle Vision — Human Review")
        gr.Markdown("Segmentation-first review. Only verified decisions may enter training data.")

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
            accept = gr.Button("ACCEPT")
            change = gr.Button("CHANGE CLASS")
            reject = gr.Button("REJECT")
            resegment = gr.Button("RESEGMENT")
        class_id = gr.Dropdown(choices=class_ids, label="Class", allow_custom_value=False)
        notes = gr.Textbox(label="Notes", lines=2)
        state = gr.State()
        result = gr.Code(label="Review result", language="json")

        analyze.click(
            analyze_image,
            inputs=input_image,
            outputs=[original, mask, overlay, table, state],
        )
        for button, action in [
            (accept, ReviewAction.ACCEPT.value),
            (change, ReviewAction.CHANGE_CLASS.value),
            (reject, ReviewAction.REJECT.value),
            (resegment, ReviewAction.RESEGMENT.value),
        ]:
            button.click(
                lambda class_id, notes, state, action=action: apply_review(action, class_id, notes, state),
                inputs=[class_id, notes, state],
                outputs=result,
            )

    return app


def launch() -> None:
    """Launch the review UI."""
    build_app().launch()


if __name__ == "__main__":
    launch()
