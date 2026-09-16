from bottle_vision.storage import load_json, save_json


def test_json_round_trip(tmp_path):
    path = tmp_path / "nested" / "sample.json"
    payload = {"class": "green", "verified": True}

    save_json(path, payload)

    assert load_json(path) == payload
