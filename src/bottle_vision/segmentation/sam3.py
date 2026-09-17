"""Lazy SAM 3 image-segmentation adapter.

SAM 3 is intentionally an optional dependency. Importing Bottle Vision must
remain possible in a lightweight test/Colab environment until the SAM 3
runtime is installed and its checkpoint access is configured.

The adapter also contains a small compatibility shim for legacy CUDA GPUs
such as NVIDIA Tesla T4. Current SAM 3 ViT MLP inference uses a fused
addmm_act path that casts its first-layer activation to bfloat16. T4 is
pre-Ampere, so we use the equivalent unfused PyTorch MLP path there and keep
the model/input tensors in float32.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np

from .base import Segmenter
from .quality import score_mask
from .types import SegmentationInstance, SegmentationResult


_SAM3_LEGACY_MLP_PATCHED = False


def _patch_sam3_vit_mlp_for_legacy_cuda() -> None:
    """Disable SAM 3's bfloat16 fused ViT MLP path on pre-Ampere CUDA GPUs."""
    global _SAM3_LEGACY_MLP_PATCHED
    if _SAM3_LEGACY_MLP_PATCHED:
        return

    from sam3.model.vitdet import Mlp

    if getattr(Mlp, "_bottle_vision_float32_fallback", False):
        _SAM3_LEGACY_MLP_PATCHED = True
        return

    def forward_float32(self: Any, x: Any) -> Any:
        # SAM3's fused addmm_act may return BF16 on pre-Ampere GPUs even when
        # the ViT parameters are float32. Keep the first MLP branch in its
        # native output dtype, then explicitly restore float32 before fc2.
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = x.float()
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    Mlp.forward = forward_float32
    Mlp._bottle_vision_float32_fallback = True
    _SAM3_LEGACY_MLP_PATCHED = True


def _is_legacy_cuda(device: str) -> bool:
    """Return whether device points at a pre-Ampere CUDA GPU."""
    if not str(device).startswith("cuda"):
        return False

    import torch

    if not torch.cuda.is_available():
        return False

    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(index)
    return major < 8


class Sam3Segmenter(Segmenter):
    """Text-prompted SAM 3 image segmenter with a mask quality gate."""

    model_name = "sam3"

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        prompt: str = "bottle",
        min_confidence: float = 0.50,
        min_quality_score: float = 0.50,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.95,
        load_from_hf: bool = True,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 0.0 <= min_quality_score <= 1.0:
            raise ValueError("min_quality_score must be between 0 and 1")

        self.prompt = prompt
        self.min_confidence = min_confidence
        self.min_quality_score = min_quality_score
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._load_from_hf = load_from_hf
        self._model = model
        self._processor = processor
        self.model_version = "unknown"

    def _ensure_loaded(self) -> None:
        if self._processor is not None:
            return

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 3 is not installed. Install the optional SAM 3 runtime "
                "before using Sam3Segmenter."
            ) from exc

        legacy_cuda = _is_legacy_cuda(self._device)

        if legacy_cuda:
            _patch_sam3_vit_mlp_for_legacy_cuda()

        if self._model is None:
            self._model = build_sam3_image_model(
                checkpoint_path=self._checkpoint_path,
                load_from_HF=self._load_from_hf,
                device=self._device,
                eval_mode=True,
                enable_segmentation=True,
            )

        if legacy_cuda:
            self._model = self._model.float().eval()

        self._processor = Sam3Processor(self._model)
        self.model_version = getattr(self._model, "__version__", "sam3")

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Run the text prompt and return only masks that pass the gate."""
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Image must have shape (height, width, channels)")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        self._ensure_loaded()

        import torch
        from PIL import Image

        pil_image = Image.fromarray(array, mode="RGB")
        if _is_legacy_cuda(self._device):
            inference_context = torch.autocast(device_type="cuda", enabled=False)
        else:
            inference_context = nullcontext()
        with inference_context:
            state = self._processor.set_image(pil_image)
            output = self._processor.set_text_prompt(state=state, prompt=self.prompt)

        masks = output.get("masks")
        scores = output.get("scores")
        if masks is None or scores is None:
            return SegmentationResult(
                instances=(),
                image_shape=(array.shape[0], array.shape[1]),
                prompt=self.prompt,
                model_name=self.model_name,
                model_version=self.model_version,
                error="sam3_output_missing_masks_or_scores",
            )

        masks_np = _to_numpy_masks(masks)
        scores_np = _to_numpy_scores(scores)
        instances: list[SegmentationInstance] = []

        for index, mask in enumerate(masks_np):
            confidence = float(scores_np[index]) if index < len(scores_np) else 0.0
            binary = np.asarray(mask, dtype=bool)
            if binary.shape != array.shape[:2]:
                binary = _resize_mask_nearest(binary, array.shape[:2])

            quality = score_mask(
                binary,
                array.shape[:2],
                min_area_ratio=self.min_area_ratio,
                max_area_ratio=self.max_area_ratio,
            )
            metadata = {
                "accepted_by_gate": bool(
                    confidence >= self.min_confidence
                    and quality.valid
                    and quality.score >= self.min_quality_score
                ),
            }
            instances.append(
                SegmentationInstance(
                    mask=binary,
                    confidence=confidence,
                    instance_id=index,
                    prompt=self.prompt,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    quality=quality,
                    metadata=metadata,
                )
            )

        return SegmentationResult(
            instances=tuple(instances),
            image_shape=(array.shape[0], array.shape[1]),
            prompt=self.prompt,
            model_name=self.model_name,
            model_version=self.model_version,
        )


def _to_numpy_masks(value: Any) -> np.ndarray:
    """Convert SAM 3 tensor/list output to (N,H,W) boolean masks."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"Unexpected SAM 3 mask shape: {array.shape}")
    return array > 0.5


def _to_numpy_scores(value: Any) -> np.ndarray:
    """Convert SAM 3 scores to a flat NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)


def _resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask without adding an image-processing dependency."""
    if mask.shape == shape:
        return mask
    row_idx = np.minimum(
        (np.arange(shape[0]) * mask.shape[0] / shape[0]).astype(int), mask.shape[0] - 1
    )
    col_idx = np.minimum(
        (np.arange(shape[1]) * mask.shape[1] / shape[1]).astype(int), mask.shape[1] - 1
    )
    return mask[np.ix_(row_idx, col_idx)]
