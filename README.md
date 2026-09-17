# Bottle Vision

Segmentation-first bottle vision pipeline for Google Colab/Gradio, verified dataset collection, training, camera recognition, and later lightweight INT8 deployment.

See `MASTER_SPEC.md` and `AGENTS.md` for the project rules.

## Supported Colab path

The primary development path is **one normal Colab GPU runtime**. Bottle Vision does not create a second Python/Conda/uv environment inside Colab. This avoids CUDA runtime mismatches between Colab's installed driver stack and a separately managed interpreter.

Start a fresh Colab GPU runtime, then run `colab/BottleVision_SAM3_Runner.ipynb` from top to bottom.

The bootstrap deliberately installs the ML stack in a controlled order:

1. keep Colab's Python runtime;
2. install the matching PyTorch/TorchVision CUDA wheels;
3. pin `numpy<2` because the current SAM 3 package declares that constraint;
4. install SAM 3 and Bottle Vision without letting an editable install unexpectedly replace the CUDA stack;
5. install `einops` explicitly;
6. verify Python, NumPy, PyTorch, TorchVision, SAM 3, CUDA and GPU before inference.

Current upstream SAM 3 also documents Python 3.12+, PyTorch 2.7+, and CUDA 12.6+ as prerequisites. SAM 3 checkpoints require Hugging Face access/authentication.

## Local Gradio review

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ui]"
python -m bottle_vision.review.gradio_app
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Review workflow

1. Upload a bottle image.
2. Click **Analyze / Rerun Segmentation**.
3. Inspect **Original / Mask / Overlay** and the quality table.
4. Select each detected instance and assign a configuration-driven class.
5. Choose `train`, `val`, or `test`.
6. Click **ACCEPT** only after the masks and classes are verified.
7. Verified samples are written to `dataset/images/<split>`, `dataset/masks/<split>`, and `dataset/metadata/<split>`.
8. **REJECT** and **RESEGMENT** stay in the persistent review queue.

Only human-verified samples may enter the training dataset.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
```

## Architecture status

The repository is being completed in vertical slices: first a reliable SAM 3 Colab inference path, then review persistence/dataset promotion, then color/material analysis, batch processing, training, recognition and export. ESP32-S3 is a deployment target for the lightweight recognition model, never for SAM 3 itself.
