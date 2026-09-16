import numpy as np
import pytest

from bottle_vision.segmentation.sam3 import Sam3Segmenter


class FakeProcessor:
    def set_image(self, image):
        return {"image": image}

    def set_text_prompt(self, *, state, prompt):
        assert prompt == "bottle"
        return {
            "masks": np.array([[[0, 0, 0], [0, 1, 0], [0, 1, 0]]], dtype=np.float32),
            "scores": np.array([0.91], dtype=np.float32),
        }


def test_adapter_uses_masks_and_quality_gate():
    segmenter = Sam3Segmenter(
        processor=FakeProcessor(),
        min_area_ratio=0.1,
        min_quality_score=0.5,
    )

    result = segmenter.segment(np.zeros((3, 3, 3), dtype=np.uint8))

    assert result.has_instances
    instance = result.instances[0]
    assert instance.mask.dtype == np.bool_
    assert instance.mask.shape == (3, 3)
    assert instance.confidence == pytest.approx(0.91, abs=1e-3)
    assert instance.metadata["accepted_by_gate"] is True


def test_adapter_is_lazy():
    segmenter = Sam3Segmenter()
    assert segmenter.model_name == "sam3"
    assert segmenter.prompt == "bottle"
    assert segmenter._processor is None
