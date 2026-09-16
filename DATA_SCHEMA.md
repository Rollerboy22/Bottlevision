# Bottle Vision — Data Schema

The initial schema is intentionally minimal and versioned.

## Sample metadata
Recommended fields:
- `sample_id`
- `image_path`
- `mask_path`
- `class_id`
- `class_name`
- `confidence`
- `mask_quality`
- `decision` (`ACCEPT`, `REVIEW`, `REJECT`)
- `verification_status` (`UNVERIFIED`, `VERIFIED`, `REJECTED`)
- `source`
- `model_name`
- `model_version`
- `pipeline_version`
- `created_at`
- `verified_at`
- `group_id` for leakage-safe splitting

## Class schema
`classes.json` is the source for configurable classes. Each class should have a stable identifier and human-readable name.

## Manifest
`dataset_manifest.json` should record dataset version, counts, split policy, class mapping, and relevant preprocessing/model versions.

Schema changes must be versioned and documented.