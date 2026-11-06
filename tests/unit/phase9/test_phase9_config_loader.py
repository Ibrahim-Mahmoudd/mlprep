from __future__ import annotations

import json

import pytest

from mlprep.phases.phase9.config_loader import Phase9ConfigLoader


def test_phase9_config_loads_yaml_json_and_dict(tmp_path) -> None:
    assert Phase9ConfigLoader.from_dict({}).require_disjoint_rows is True
    assert Phase9ConfigLoader.from_dict(
        {"copy_partitions": False}
    ).copy_partitions is False

    yaml_path = tmp_path / "phase9.yaml"
    yaml_path.write_text(
        "require_disjoint_rows: false\ncopy_partitions: true\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "phase9.json"
    json_path.write_text(
        json.dumps({"require_disjoint_rows": False, "copy_partitions": True}),
        encoding="utf-8",
    )

    assert Phase9ConfigLoader.load(yaml_path).require_disjoint_rows is False
    assert Phase9ConfigLoader.load(json_path).copy_partitions is True


def test_phase9_config_loader_rejects_unknown_or_invalid_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown Phase 9"):
        Phase9ConfigLoader.from_dict({"unknown": True})
    with pytest.raises(FileNotFoundError):
        Phase9ConfigLoader.load(tmp_path / "missing.yaml")
    with pytest.raises(IsADirectoryError):
        Phase9ConfigLoader.load(tmp_path)

    unsupported = tmp_path / "phase9.ini"
    unsupported.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported configuration extension"):
        Phase9ConfigLoader.load(unsupported)

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON configuration"):
        Phase9ConfigLoader.load(invalid_json)

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("- one\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        Phase9ConfigLoader.load(scalar)
