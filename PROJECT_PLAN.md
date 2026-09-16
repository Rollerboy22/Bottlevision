# Bottle Vision — Project Plan

## Phase 0 — Foundation
- Repository rules and architecture.
- Configuration and class schema.
- Metadata model.
- Persistence abstraction.
- Test foundation.

## Phase 1 — Segmentation
- Model adapter interface.
- SAM 3 integration where practical.
- SAM 2/fallback adapter.
- Multiple-instance masks.
- Mask postprocessing.
- Segmentation quality scoring.

## Phase 2 — Classification and color analysis
- Material/color classes.
- Mask-aware color extraction.
- Transparent-object handling.
- Confidence and decision engine.

## Phase 3 — Review UI
- Original / Mask / Overlay.
- ACCEPT / CHANGE CLASS / REJECT / RESEGMENT.
- Human verification persistence.

## Phase 4 — Batch and dataset
- Queue/state machine.
- Resume/retry/failure handling.
- Dataset promotion.
- 80/10/10 leakage-safe split.

## Phase 5 — Training
- Verified-data-only training pipeline.
- Classification and segmentation metrics.
- Experiment tracking and reproducibility.

## Phase 6 — Recognition
- Camera workflow.
- Real-time inference interface.

## Phase 7 — Export
- PyTorch / ONNX / TFLite as appropriate.
- INT8 quantization.
- Size/latency/accuracy validation.

## Phase 8 — ESP32-S3
- Select lightweight architecture.
- Export and validate INT8 model.
- Embedded inference integration.
- Device-side performance tests.

## Current milestone
Establish the repository foundation before implementing model-specific code.