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
    review = ReviewState(image_id="sample", instances=[{"instance_id": 0}])
    review.apply(ReviewAction.REJECT)

    with pytest.raises(ValueError, match="verified"):
        writer.save_verified(np.zeros((8, 8, 3), dtype=np.uint8), _result(), review)


def test_verified_review_requires_instance_class(tmp_path):
    writer = VerifiedDatasetWriter(tmp_path / "dataset")
    review = ReviewState(image_id="sample", instances=[{"instance_id": 0}])

    with pytest.raises(ValueError, match="Every instance"):
        review.apply(ReviewAction.ACCEPT)


def test_verified_review_writes_class_and_files(tmp_path):
    writer = VerifiedDatasetWriter(tmp_path / "dataset")
    review = ReviewState(image_id="sample", instances=[{"instance_id": 0}])
    review.set_instance_class(0, "green")
    review.apply(ReviewAction.ACCEPT)
    paths = writer.save_verified(np.zeros((8, 8, 3), dtype=np.uint8), _result(), review)

    assert (tmp_path / "dataset/images/train/sample.png").exists()
    assert (tmp_path / "dataset/masks/train/sample_0.png").exists()
    metadata = (tmp_path / "dataset/metadata/train/sample.json").read_text(encoding="utf-8")
    assert '"class_id": "green"' in metadata
    assert len(paths) == 3


def test_pending_session_round_trip(tmp_path):
    writer = VerifiedDatasetWriter(tmp_path / "dataset")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    review = writer.save_pending(image, _result())

    loaded_image, loaded_result, loaded_review = writer.load_pending(review.image_id)

    assert np.array_equal(loaded_image, image)
    assert loaded_result.image_shape == (8, 8)
    assert len(loaded_result.instances) == 1
    assert loaded_review.image_id == review.image_id


def test_verified_save_clears_pending_session(tmp_path):
    writer = VerifiedDatasetWriter(tmp_path / "dataset")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    review = writer.save_pending(image, _result())
    review.set_instance_class(0, "brown")
    review.apply(ReviewAction.ACCEPT)
    writer.save_verified(image, _result(), review)

    assert not (tmp_path / "dataset/queue" / review.image_id).exists()
