import numpy as np
import pytest

from bottle_vision.dataset.writer import VerifiedDatasetWriter
from bottle_vision.review.state import ReviewAction, ReviewState
from bottle_vision.segmentation.types import SegmentationInstance, SegmentationResult


def _result():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 3:5] = True
    instance = SegmentationInstance(mask=mask, confidence=0.9, instance_id=0)
    return SegmentationResult(instances=(instance,), image_shape=(8, 8))


def test_unverified_review_cannot_enter_dataset(tmp_path):
    writer = VerifiedDatasetWriter(tmp_path / "dataset")
    review = ReviewState(image_id="sample")
    review.apply(ReviewAction.REJECT)

    with pytest.raises(ValueError, match="verified"):
        writer.save_verified(np.zeros((8, 8, 3), dtype=np.uint8), _result(), review)


def test_verified_review_writes_image_masks_and_metadata(tmp_path):
    writer = VerifiedDatasetWriter(tmp_path / "dataset")
    review = ReviewState(image_id="sample")
    review.apply(ReviewAction.ACCEPT)
    paths = writer.save_verified(np.zeros((8, 8, 3), dtype=np.uint8), _result(), review)

    assert (tmp_path / "dataset/images/train/sample.png").exists()
    assert (tmp_path / "dataset/masks/train/sample_0.png").exists()
    assert (tmp_path / "dataset/metadata/train/sample.json").exists()
    assert len(paths) == 3
