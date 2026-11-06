from __future__ import annotations

import json

import pytest
import yaml

from mlprep.phases.phase6.config import Phase6Config
from mlprep.phases.phase6.config_loader import Phase6ConfigLoader


def test_default_config_round_trips_through_yaml_and_json(tmp_path) -> None:
    config = Phase6Config()
    expected = config.to_dict()

    for suffix in (".yaml", ".json"):
        path = tmp_path / f"phase6{suffix}"
        Phase6ConfigLoader.save(config, path)
        loaded = Phase6ConfigLoader.load(path)
        assert loaded.to_dict() == expected

    json_path = tmp_path / "raw.json"
    json_path.write_text(json.dumps(expected), encoding="utf-8")
    assert Phase6ConfigLoader.load(json_path).to_dict() == expected
    assert yaml.safe_load(
        (tmp_path / "phase6.yaml").read_text(encoding="utf-8")
    ) == expected


def test_config_loader_merges_overrides_and_rejects_unknown_keys(
    tmp_path,
) -> None:
    path = tmp_path / "phase6.yaml"
    path.write_text(
        yaml.safe_dump({"execution": {"enabled": True}}),
        encoding="utf-8",
    )
    config = Phase6ConfigLoader.load(
        path,
        overrides={"execution": {"stop_on_plan_failure": False}},
        dotted_overrides={"execution.enabled": False},
    )

    assert config.execution.enabled is False
    assert config.execution.stop_on_plan_failure is False

    with pytest.raises(ValueError, match="Unknown Phase 6 sections"):
        Phase6ConfigLoader.from_dict({"unknown": True})

    with pytest.raises(ValueError, match="Conflicting Phase 6 numeric"):
        Phase6ConfigLoader.from_dict(
            {
                "numeric": {},
                "pipeline": {"numeric": {"numeric_strategy": "invalid"}},
            }
        )


def test_phase6_loader_rejects_invalid_paths_and_roots(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        Phase6ConfigLoader.load(tmp_path / "missing.yaml")
    with pytest.raises(IsADirectoryError):
        Phase6ConfigLoader.load(tmp_path)

    unsupported = tmp_path / "settings.toml"
    unsupported.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported configuration extension"):
        Phase6ConfigLoader.load(unsupported)

    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps([1]), encoding="utf-8")
    with pytest.raises(TypeError, match="root must be a mapping"):
        Phase6ConfigLoader.load(scalar)
