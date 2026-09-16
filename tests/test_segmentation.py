import numpy as np
import pytest

from bottle_vision.segmentation import SegmentationInstance, UnconfiguredSegmenter, score_mask


def test_quality_accepts_reasonable_mask():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 40:60] = True

    quality = score_mask(mask, mask.shape)

    assert quality.valid
    assert 0.0 <= quality.score <= 1.0
    assert quality.area_ratio == pytest.approx(0.12)


def test_quality_rejects_empty_mask():
    mask = np.zeros((20, 30), dtype=bool)

    quality = score_mask(mask, mask.shape)

    assert not quality.valid
    assert "empty_mask" in quality.reasons


def test_instance_mask_is_binary_and_confidence_is_bounded():
    instance = SegmentationInstance(
        mask=np.array([[0, 1], [2, 0]], dtype=np.uint8),
        confidence=0.9,
        instance_id=0,
    )

    assert instance.mask.dtype == np.bool_
    assert instance.mask.tolist() == [[False, True], [True, False]]

    with pytest.raises(ValueError):
        SegmentationInstance(mask=np.zeros((2, 2)), confidence=1.1, instance_id=0)


def test_unconfigured_segmenter_never_fabricates_instances():
    result = UnconfiguredSegmenter().segment(np.zeros((32, 48, 3), dtype=np.uint8))

    assert result.instances == ()
    assert result.error == "segmentation_backend_not_configured"
    assert result.image_shape == (32, 48)
