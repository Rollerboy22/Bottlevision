import numpy as np
import pytest

from bottle_vision.segmentation import SegmentationInstance, SegmentationResult, make_review_views


def _result() -> SegmentationResult:
    mask_a = np.zeros((4, 5), dtype=bool)
    mask_a[1:3, 1:3] = True
    mask_b = np.zeros((4, 5), dtype=bool)
    mask_b[0:2, 3:5] = True
    return SegmentationResult(
        instances=(
            SegmentationInstance(mask=mask_a, confidence=0.9, instance_id=0),
            SegmentationInstance(mask=mask_b, confidence=0.8, instance_id=1),
        ),
        image_shape=(4, 5),
    )


def test_review_views_have_expected_shapes_and_do_not_mutate_input():
    image = np.full((4, 5, 3), 100, dtype=np.uint8)
    original = image.copy()

    views = make_review_views(image, _result())

    assert set(views) == {"Original", "Mask", "Overlay"}
    assert all(view.shape == image.shape for view in views.values())
    assert all(view.dtype == np.uint8 for view in views.values())
    np.testing.assert_array_equal(image, original)
    assert np.any(views["Mask"] != 0)
    assert np.any(views["Overlay"] != image)


def test_review_views_reject_mismatched_image_shape():
    image = np.zeros((3, 5, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Image shape"):
        make_review_views(image, _result())


def test_overlay_alpha_is_validated():
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="alpha"):
        make_review_views(image, _result(), alpha=1.1)
