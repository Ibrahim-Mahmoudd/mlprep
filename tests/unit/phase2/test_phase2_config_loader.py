from __future__ import annotations

import json

import pytest
import yaml

from mlprep.phases.phase2.config import Phase2Config
from mlprep.phases.phase2.config_loader import Phase2ConfigLoader
from mlprep.phases.phase2.output_governance.metadata import SplitStrategyName


def test_phase2_config_round_trips_yaml_and_json(tmp_path) -> None:
    config = Phase2Config.create(strategy=SplitStrategyName.RANDOM)
    expected = config.to_dict()

    for suffix in (".yaml", ".json"):
        path = tmp_path / f"phase2{suffix}"
        Phase2ConfigLoader.save(config, path)
        assert Phase2ConfigLoader.load(path).to_dict() == expected

    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(expected), encoding="utf-8")
    assert Phase2ConfigLoader.load(raw_path).to_dict() == expected
    assert yaml.safe_load(
        (tmp_path / "phase2.yaml").read_text(encoding="utf-8")
    ) == expected


def test_phase2_config_loader_merges_overrides_and_validates_sections(
    tmp_path,
) -> None:
    path = tmp_path / "phase2.yaml"
    path.write_text(
        yaml.safe_dump({"strategy": "random", "random_seed": 1}),
        encoding="utf-8",
    )
    config = Phase2ConfigLoader.load(
        path,
        overrides={"execution": {"enabled": False}},
        dotted_overrides={"random_seed": 17},
    )

    assert config.random_seed == 17
    assert config.execution.enabled is False

    with pytest.raises(ValueError, match="Unknown Phase 2"):
        Phase2ConfigLoader.from_dict({"unknown": True})


def test_phase2_loader_rejects_bad_file_inputs_and_invalid_overrides(
    tmp_path,
) -> None:
    with pytest.raises(FileNotFoundError):
        Phase2ConfigLoader.load(tmp_path / "missing.yaml")
    with pytest.raises(IsADirectoryError):
        Phase2ConfigLoader.load(tmp_path)

    unsupported = tmp_path / "phase2.ini"
    unsupported.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported configuration extension"):
        Phase2ConfigLoader.load(unsupported)

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        Phase2ConfigLoader.load(empty)

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        Phase2ConfigLoader.load(invalid_json)

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("invalid: [", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        Phase2ConfigLoader.load(invalid_yaml)


def test_phase2_config_loader_validates_strategy_options() -> None:
    with pytest.raises(ValueError, match="strategy"):
        Phase2ConfigLoader.from_dict({"strategy": "not-a-strategy"})
