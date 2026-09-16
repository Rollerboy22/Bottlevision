# Bottle Vision — Master Specification

## Purpose
Bottle Vision is a segmentation-first computer-vision system for detecting bottles, extracting pixel-level masks, classifying bottle material/color, collecting verified training data, and later exporting lightweight recognition models.

## Non-negotiable rules
- Segmentation is the primary annotation. Bounding boxes may be used internally but are never the final annotation.
- Prefer modern foundation segmentation models (SAM 3 first where practical, then SAM 2 or another suitable model).
- Use a bottle-oriented prompt such as `bottle` and support multiple bottle instances.
- Every automatic mask must pass quality/vision validation before acceptance.
- Bad masks or low confidence must enter REVIEW rather than being silently accepted.
- Analyze bottle color/material from the mask while minimizing contamination from labels, text, reflections, and background.
- Transparent bottles require explicit background-aware handling; background color must not become the bottle color.
- Classes are configuration-driven and extensible; do not hard-code the class list in processing logic.
- Store metadata, model versions, pipeline versions, confidence, quality scores, and verification state.
- Only human-verified data may be used for training.
- Prevent dataset leakage when splitting images belonging to the same physical object/session.
- The system must support persistence, resume, retry, and failure handling.
- Review UI must expose Original / Mask / Overlay and actions ACCEPT / CHANGE CLASS / REJECT / RESEGMENT.
- The PC/Colab pipeline comes before TinyML deployment.
- ESP32-S3 is a target for a lightweight INT8 recognition model, not for running SAM.
- Unknown/review is preferable to a wrong automatic label.

## Dataset layout
```text
dataset/
├── images/train val test
├── masks/train val test
├── metadata/train.json val.json test.json
├── classes.json
└── dataset_manifest.json
```

## Target split
Default split is 80/10/10, with leakage prevention and grouping of samples from the same physical object where applicable.

## Architecture modules
```text
src/bottle_vision/
├── config/
├── segmentation/
├── classification/
├── color_analysis/
├── quality/
├── dataset/
├── review/
├── recognition/
├── training/
├── export/
└── ui/
```

## Quality gates
1. Input validation.
2. Segmentation generation.
3. Mask postprocessing.
4. Independent mask quality/vision validation.
5. Material-aware color analysis.
6. Decision engine: ACCEPT / REVIEW / REJECT.
7. Persistent metadata and artifacts.
8. Human verification before dataset promotion.
9. Training only from verified samples.
10. Metrics and reproducible experiment metadata.

This document is the project source of truth. Changes to architecture or these rules should be deliberate, documented, and reviewed.