from __future__ import annotations

import json

import pytest
import yaml

from mlprep.phases.phase3.config import Phase3Config
from mlprep.phases.phase3.config_loader import Phase3ConfigLoader


def test_phase3_config_round_trips_yaml_and_json(tmp_path) -> None:
    config = Phase3Config()
    expected = config.to_dict()

    for suffix in (".yaml", ".json"):
        path = tmp_path / f"phase3{suffix}"
        Phase3ConfigLoader.save(config, path)
        assert Phase3ConfigLoader.load(path).to_dict() == expected

    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(expected), encoding="utf-8")
    assert Phase3ConfigLoader.load(raw_path).to_dict() == expected
    assert yaml.safe_load(
        (tmp_path / "phase3.yaml").read_text(encoding="utf-8")
    ) == expected


def test_phase3_config_loader_merges_dotted_overrides_and_validates(
    tmp_path,
) -> None:
    path = tmp_path / "phase3.yaml"
    path.write_text(
        yaml.safe_dump({"reporting": {"indent": 2}}),
        encoding="utf-8",
    )
    config = Phase3ConfigLoader.load(
        path,
        overrides={"reporting": {"include_state": True}},
        dotted_overrides={"reporting.indent": 4},
    )

    assert config.reporting.indent == 4
    assert config.reporting.include_state is True
    assert Phase3ConfigLoader.validate_file(path) == []

    with pytest.raises(ValueError, match="Unknown Phase 3"):
        Phase3ConfigLoader.from_dict({"unknown": True})


def test_phase3_config_loader_reports_invalid_file_and_override_errors(
    tmp_path,
) -> None:
    with pytest.raises(FileNotFoundError):
        Phase3ConfigLoader.load(tmp_path / "missing.yaml")
    with pytest.raises(IsADirectoryError):
        Phase3ConfigLoader.load(tmp_path)

    unsupported = tmp_path / "config.ini"
    unsupported.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported configuration extension"):
        Phase3ConfigLoader.load(unsupported)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("invalid: [", encoding="utf-8")
    errors = Phase3ConfigLoader.validate_file(invalid)
    assert errors
    assert "Invalid YAML configuration" in errors[0]

    with pytest.raises(ValueError, match="dotted"):
        Phase3ConfigLoader.apply_dot_overrides(
            values={"reporting": {"indent": 2}},
            overrides={"reporting..indent": 3},
        )
