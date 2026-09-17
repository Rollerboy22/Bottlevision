from pathlib import Path

from bottle_vision.config import load_config


def test_config_package_exports_loader(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("project:\n  name: test\n", encoding="utf-8")
    assert load_config(path)["project"]["name"] == "test"
