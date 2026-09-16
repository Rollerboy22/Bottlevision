import pytest

from bottle_vision.review import ReviewAction, ReviewState


def test_accept_marks_state_verified():
    state = ReviewState(image_id="sample-1")
    state.apply(ReviewAction.ACCEPT)

    assert state.verified is True
    assert state.action is ReviewAction.ACCEPT


def test_change_class_requires_class_id():
    state = ReviewState(image_id="sample-1")

    with pytest.raises(ValueError, match="class_id"):
        state.apply(ReviewAction.CHANGE_CLASS)


def test_resegment_is_not_verified():
    state = ReviewState(image_id="sample-1")
    state.apply(ReviewAction.RESEGMENT, notes="Mask needs another pass")

    assert state.verified is False
    assert state.to_dict()["notes"] == "Mask needs another pass"
