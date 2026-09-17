# Bottle Vision — Project Plan

## Delivery strategy

Build the project in working vertical slices. Every slice must be runnable and tested before the next layer is added. Google Colab uses one normal GPU runtime; no nested Python/Conda/uv environment is part of the primary path.

## Phase 0 — Foundation — complete
- Repository rules and architecture.
- Configuration and class schema.
- Metadata model.
- Persistence abstraction.
- Test foundation.
- Fixed `config.py` / `config/` import collision.

## Phase 1 — SAM 3 inference — current
- Stable Colab GPU bootstrap.
- Explicit CUDA-compatible PyTorch/TorchVision installation.
- NumPy compatibility guard.
- Explicit `einops` dependency.
- SAM 3 import/API smoke test.
- Hugging Face checkpoint authentication path.
- Text-prompted multi-instance masks.
- Independent mask quality gate.
- Original / Mask / Overlay review views.

**Exit condition:** a fresh Colab GPU runtime can upload a bottle image and produce reviewable masks without a second Python environment.

## Phase 2 — Review and verified dataset — next
- Durable queue/state machine.
- ACCEPT / CHANGE CLASS / REJECT / RESEGMENT.
- Configuration-driven classes.
- Verified-only dataset promotion.
- Metadata and artifact manifests.
- Resume/retry after interruption.

## Phase 3 — Material and color analysis
- Mask-aware color extraction.
- Label/background/reflection suppression.
- Transparent-object handling.
- Material-aware classification.
- Confidence and REVIEW policy.

## Phase 4 — Batch dataset pipeline
- Image queue.
- Resume/retry/failure states.
- Group-aware 80/10/10 split.
- Leakage prevention by physical object/session.
- Dataset validation and manifest generation.

## Phase 5 — Training
- Verified-data-only training.
- Reproducible experiment configuration.
- Classification metrics.
- Segmentation metrics where applicable.
- Checkpoint/version metadata.

## Phase 6 — Recognition
- Camera workflow.
- Real-time lightweight inference.
- Unknown/review output instead of forced labels.

## Phase 7 — Export
- PyTorch / ONNX / TFLite as appropriate.
- INT8 quantization.
- Size, latency and accuracy validation.

## Phase 8 — ESP32-S3
- Select lightweight recognition architecture.
- Export and validate INT8 model.
- Embedded inference integration.
- Device-side performance tests.

## Non-negotiable acceptance criteria

A feature is not considered complete merely because it imports. It must have a runnable path, deterministic configuration, safe failure behavior, persistence where required, and automated tests for its CPU-testable logic.
