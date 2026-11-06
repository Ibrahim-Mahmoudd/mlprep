from __future__ import annotations

import json

import pytest
import yaml

from mlprep.phases.phase4.config import Phase4Config
from mlprep.phases.phase4.config_loader import Phase4ConfigLoader


def test_default_config_round_trips_as_yaml_and_json(tmp_path) -> None:
    config = Phase4Config()
    payload = config.to_dict()
    assert payload["pipeline"]["missing_tokens"]["default_tokens"] == sorted(
        config.pipeline.missing_tokens.default_tokens,
        key=repr,
    )

    for suffix in (".yaml", ".json"):
        path = tmp_path / f"phase4{suffix}"
        Phase4ConfigLoader.save(config, path)
        loaded = Phase4ConfigLoader.load(path)
        assert loaded.to_dict() == payload

    json_path = tmp_path / "source.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    assert Phase4ConfigLoader.load(json_path).to_dict() == payload
    assert yaml.safe_load(
        (tmp_path / "phase4.yaml").read_text(encoding="utf-8")
    ) == payload


def test_config_loader_merges_overrides_and_rejects_unknown_keys(
    tmp_path,
) -> None:
    path = tmp_path / "phase4.yaml"
    path.write_text(
        yaml.safe_dump({"execution": {"enabled": True}}),
        encoding="utf-8",
    )
    config = Phase4ConfigLoader.load(
        path,
        overrides={"execution": {"stop_on_plan_failure": False}},
        dotted_overrides={"execution.enabled": False},
    )

    assert config.execution.enabled is False
    assert config.execution.stop_on_plan_failure is False

    with pytest.raises(ValueError, match="Unknown Phase 4 sections"):
        Phase4ConfigLoader.from_dict({"extra": True})
    with pytest.raises(ValueError, match="Unknown pipeline options"):
        Phase4ConfigLoader.from_dict(
            {"pipeline": {"unexpected": True}}
        )


def test_config_loader_rejects_invalid_paths_and_roots(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        Phase4ConfigLoader.load(tmp_path / "missing.yaml")
    with pytest.raises(IsADirectoryError):
        Phase4ConfigLoader.load(tmp_path)

    unsupported = tmp_path / "settings.toml"
    unsupported.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported configuration extension"):
        Phase4ConfigLoader.load(unsupported)

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Configuration is empty"):
        Phase4ConfigLoader.load(empty)

    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps([1]), encoding="utf-8")
    with pytest.raises(TypeError, match="root must be a mapping"):
        Phase4ConfigLoader.load(scalar)
