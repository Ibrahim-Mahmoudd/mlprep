from __future__ import annotations

import json

import pytest
import yaml

from mlprep.phases.phase5.config import Phase5Config
from mlprep.phases.phase5.config_loader import Phase5ConfigLoader


def test_default_config_round_trips_through_yaml_and_json(tmp_path) -> None:
    config = Phase5Config()
    expected = config.to_dict()

    for suffix in (".yaml", ".json"):
        path = tmp_path / f"phase5{suffix}"
        Phase5ConfigLoader.save(config, path)
        restored = Phase5ConfigLoader.load(path)
        assert restored.to_dict() == expected

    raw_json = tmp_path / "raw.json"
    raw_json.write_text(json.dumps(expected), encoding="utf-8")
    assert Phase5ConfigLoader.load(raw_json).to_dict() == expected
    assert yaml.safe_load(
        (tmp_path / "phase5.yaml").read_text(encoding="utf-8")
    ) == expected


def test_config_loader_applies_merge_and_dotted_overrides(tmp_path) -> None:
    path = tmp_path / "phase5.yaml"
    path.write_text(
        yaml.safe_dump({"execution": {"enabled": True}}),
        encoding="utf-8",
    )
    config = Phase5ConfigLoader.load(
        path,
        overrides={"execution": {"stop_on_plan_failure": False}},
        dotted_overrides={"execution.enabled": False},
    )

    assert config.execution.enabled is False
    assert config.execution.stop_on_plan_failure is False


def test_config_loader_rejects_unknown_sections_and_invalid_paths(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="Unknown Phase 5"):
        Phase5ConfigLoader.from_dict({"unknown": True})
    with pytest.raises(FileNotFoundError):
        Phase5ConfigLoader.load(tmp_path / "missing.yaml")
    with pytest.raises(IsADirectoryError):
        Phase5ConfigLoader.load(tmp_path)

    unsupported = tmp_path / "config.toml"
    unsupported.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported configuration extension"):
        Phase5ConfigLoader.load(unsupported)

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Configuration is empty"):
        Phase5ConfigLoader.load(empty)

    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps(1), encoding="utf-8")
    with pytest.raises(TypeError, match="root must be a mapping"):
        Phase5ConfigLoader.load(scalar)


def test_dot_overrides_reject_invalid_paths_and_immutable_merge_input() -> None:
    source = {"execution": {"enabled": True}}
    with pytest.raises(ValueError, match="dotted"):
        Phase5ConfigLoader.apply_dot_overrides(
            values=source,
            overrides={"": False},
        )

    merged = Phase5ConfigLoader.merge(
        base=source,
        override={"execution": {"enabled": False}},
    )
    assert source["execution"]["enabled"] is True
    assert merged["execution"]["enabled"] is False
