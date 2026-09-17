from bottle_vision.pipeline import Decision, empty_result


def test_unimplemented_pipeline_requires_review():
    result = empty_result()

    assert result.decision is Decision.REVIEW
    assert not result.instances
    assert result.segmentation.error == "segmentation_not_implemented"
