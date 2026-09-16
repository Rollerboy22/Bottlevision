# Bottle Vision

Segmentation-first bottle vision pipeline for Google Colab/Gradio, dataset collection, verified training, camera recognition, and later lightweight INT8 deployment.

See `MASTER_SPEC.md` and `AGENTS.md` for the project rules.

## Quick start — Gradio review

Run these commands from the repository root:

```bash
git clone https://github.com/Rollerboy22/Bottlevision.git
cd Bottlevision
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ui]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### Run the review UI

```bash
python -m bottle_vision.review.gradio_app
```

Then open the local Gradio URL shown in the terminal.

### If you want the SAM 3 backend

Install the optional SAM 3 dependencies in the same environment:

```bash
python -m pip install -e ".[ui,sam3]"
```

The SAM 3 checkpoint/authentication requirements depend on the installed SAM 3 package and model source. Keep the model configuration in `configs/default.yaml`; do not hardcode model paths in the UI.

## Review workflow

1. Upload a bottle image.
2. Click **Analyze / Rerun Segmentation**.
3. Inspect **Original / Mask / Overlay** and the quality table.
4. For each detected instance, select its instance and class, then click **SET CLASS**.
5. Choose `train`, `val`, or `test`.
6. Click **ACCEPT** only after the masks and classes are verified.
7. Verified samples are written to `dataset/images/<split>`, `dataset/masks/<split>`, and `dataset/metadata/<split>`.
8. **REJECT** and **RESEGMENT** remain in the persistent `dataset/queue/<image_id>/` review session instead of entering training data.

Only human-verified samples are allowed into the training dataset. The queue stores the original image, masks, segmentation metadata, and current review state so an interrupted review can be recovered.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
```

The current implementation is the foundation of the full pipeline. Group-aware 80/10/10 split assignment, model training, camera recognition, and embedded INT8 export remain subsequent stages.
