from __future__ import annotations

import json

import pytest
import yaml

from mlprep.phases.phase7.config import Phase7Config
from mlprep.phases.phase7.config_loader import Phase7ConfigLoader


def test_default_config_round_trips_through_yaml_and_json(tmp_path) -> None:
    config = Phase7Config()

    for suffix in (".yaml", ".json"):
        path = tmp_path / f"phase7{suffix}"
        Phase7ConfigLoader.save(config, path)
        loaded = Phase7ConfigLoader.load(path)

        assert Phase7ConfigLoader.to_dict(loaded) == (
            Phase7ConfigLoader.to_dict(config)
        )


def test_load_merges_overrides_and_dotted_paths(tmp_path) -> None:
    path = tmp_path / "phase7.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "execution": {"enabled": True},
                "contract": {"require_phase6_state": True},
            }
        ),
        encoding="utf-8",
    )

    loaded = Phase7ConfigLoader.load(
        path,
        overrides={"execution": {"stop_on_plan_failure": False}},
        dotted_overrides={
            "execution.enabled": False,
            "contract.require_phase6_state": False,
        },
    )

    assert loaded.execution.enabled is False
    assert loaded.execution.stop_on_plan_failure is False
    assert loaded.contract.require_phase6_state is False


def test_dotted_overrides_reject_invalid_and_unknown_paths() -> None:
    with pytest.raises(ValueError, match="Invalid dotted override"):
        Phase7ConfigLoader.apply_dot_overrides(
            {"execution": {"enabled": True}},
            {"execution..enabled": False},
        )
    with pytest.raises(KeyError, match="Unknown dotted override path"):
        Phase7ConfigLoader.apply_dot_overrides(
            {"execution": {"enabled": True}},
            {"execution.unknown": False},
        )
    with pytest.raises(KeyError, match="Unknown dotted override path"):
        Phase7ConfigLoader.apply_dot_overrides(
            {"execution": {"enabled": True}},
            {"contract.enabled": False},
        )


def test_loader_rejects_unknown_fields_and_invalid_file_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown Phase 7 sections"):
        Phase7ConfigLoader.from_dict({"extra": True})

    with pytest.raises(ValueError, match="Unknown Phase7ExecutionOptions"):
        Phase7ConfigLoader.from_dict(
            {"execution": {"unknown": True}}
        )

    with pytest.raises(FileNotFoundError):
        Phase7ConfigLoader.load(tmp_path / "missing.yaml")
    with pytest.raises(IsADirectoryError):
        Phase7ConfigLoader.load(tmp_path)
    unsupported_path = tmp_path / "unknown.ini"
    unsupported_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported configuration extension"):
        Phase7ConfigLoader.load(unsupported_path)


def test_invalid_yaml_json_and_non_mapping_roots_are_rejected(tmp_path) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("invalid: [", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML configuration"):
        Phase7ConfigLoader.load(yaml_path)

    json_path = tmp_path / "bad.json"
    json_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON configuration"):
        Phase7ConfigLoader.load(json_path)

    scalar_path = tmp_path / "scalar.json"
    scalar_path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(TypeError, match="root must be a mapping"):
        Phase7ConfigLoader.load(scalar_path)


def test_save_rejects_unsupported_extension(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported configuration extension"):
        Phase7ConfigLoader.save(Phase7Config(), tmp_path / "phase7.toml")
