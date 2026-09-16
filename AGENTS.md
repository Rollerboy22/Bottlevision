# Agent Instructions — Bottle Vision

## Source of truth
`MASTER_SPEC.md` is authoritative. When implementation details conflict with it, stop and resolve the conflict rather than silently weakening the specification.

## Hard rules
1. Never replace segmentation masks with bounding-box annotations.
2. Never bypass mask validation/quality gates.
3. Never train on data that is not explicitly human-verified.
4. Never hard-code bottle classes in processing code; use configuration/schema files.
5. Preserve metadata, model versions, pipeline versions, confidence, and verification state.
6. Preserve persistence, resume, retry, and failure states in batch workflows.
7. Do not silently accept uncertain predictions; route them to REVIEW.
8. Keep material/color analysis mask-aware and robust to labels, reflections, and background.
9. Do not design the ESP32-S3 path around a foundation segmentation model; use a separate lightweight exported model.
10. Add or update tests when behavior changes.
11. Inspect the existing repository before modifying files.
12. Do not invent external APIs, model capabilities, dataset assumptions, or hardware constraints. Record assumptions explicitly.
13. Prefer small, modular changes with clear commits.
14. Keep secrets and credentials out of source control.

## Development order
Foundation → segmentation → quality → classification/color → UI → persistence/batch → training → recognition → export → ESP32.

## Definition of done
A feature is not done until its implementation, configuration, persistence/metadata implications, tests, and documentation are addressed as applicable.