# Bottle Vision — Architecture

```text
Input image
   ↓
Segmentation adapter
   ↓
Mask postprocessing
   ↓
Independent mask quality validation
   ↓
Material-aware color analysis
   ↓
Decision engine
   ├── ACCEPT
   ├── REVIEW
   └── REJECT
   ↓
Persistence + metadata
   ↓
Human verification
   ↓
Verified dataset
   ↓
Training / evaluation
   ↓
Recognition model
   ↓
Export / INT8
   ↓
ESP32-S3 (lightweight model)
```

## Design principles
- Adapters isolate model-specific APIs from business logic.
- Pipeline stages should be independently testable.
- Configuration controls classes, thresholds, paths, and model choices.
- Artifacts and metadata are first-class outputs.
- Review is a normal state, not an error.
- Training consumes only verified dataset records.