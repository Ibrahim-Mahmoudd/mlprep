"""Tests for the flat, versioned unified configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mlprep.pipeline import (
    MLPrepConfig,
    MLPrepConfigLoader,
    PipelineExecutionManifest,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIRECTORY = _REPOSITORY_ROOT / "configs"


@pytest.mark.parametrize(
    ("filename", "expected_task", "expected_cv"),
    [
        ("default.yaml", "binary_classification", "stratified"),
        ("binary_classification.yaml", "binary_classification", "stratified"),
        (
            "multiclass_classification.yaml",
            "multiclass_classification",
            "stratified",
        ),
        ("regression.yaml", "regression", "k_fold"),
        ("grouped_data.yaml", "binary_classification", "group"),
        ("time_ordered_data.yaml", "binary_classification", "time_series"),
        ("severe_imbalance.yaml", "binary_classification", "stratified"),
    ],
)
def test_scenario_configurations_resolve(
    filename: str,
    expected_task: str,
    expected_cv: str,
) -> None:
    config = MLPrepConfigLoader.load(_CONFIG_DIRECTORY / filename)

    assert isinstance(config, MLPrepConfig)
    assert config.schema_version == "1"
    assert config.config_version == "1"
    assert config.phase1.problem_contract.task_type == expected_task
    assert config.cross_validation.strategy.value == expected_cv
    assert set(config.resolved_configuration) == {
        *(f"phase{number}" for number in range(1, 10)),
        "config_version",
        "cross_validation",
        "pipeline",
    }


def test_json_configuration_round_trip_preserves_fingerprint(tmp_path) -> None:
    yaml_path = _CONFIG_DIRECTORY / "default.yaml"
    original = MLPrepConfigLoader.load(yaml_path)
    raw_config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    json_path = tmp_path / "pipeline.json"
    json_path.write_text(
        json.dumps(raw_config),
        encoding="utf-8",
    )

    restored = MLPrepConfigLoader.load(json_path)

    assert restored.resolved_configuration == original.resolved_configuration
    assert (
        restored.configuration_fingerprint
        == original.configuration_fingerprint
    )


def test_unknown_root_and_nested_configuration_keys_are_rejected() -> None:
    config_path = _CONFIG_DIRECTORY / "default.yaml"
    with pytest.raises(ValueError, match="Unknown pipeline configuration keys"):
        MLPrepConfigLoader.load(
            config_path,
            dotted_overrides={"unknown.setting": True},
        )
    with pytest.raises(ValueError, match="Unknown cross_validation"):
        MLPrepConfigLoader.load(
            config_path,
            dotted_overrides={"cross_validation.unknown": True},
        )


def test_dotted_and_environment_overrides_are_typed_and_ordered() -> None:
    source = _CONFIG_DIRECTORY / "default.yaml"
    config = MLPrepConfigLoader.load(
        source,
        dotted_overrides={
            "phase2.random_seed": 13,
            "cross_validation.n_splits": 4,
        },
        environ={
            "MLPREP_PHASE2__RANDOM_SEED": "21",
            "MLPREP_PIPELINE__SAVE_INTERMEDIATE_STATES": "false",
        },
    )

    assert config.phase2.random_seed == 21
    assert config.cross_validation.n_splits == 4
    assert config.pipeline.save_intermediate_states is False
    assert config.save_intermediate_states is False


def test_unified_config_fingerprint_and_report_do_not_expose_values(
    tmp_path,
) -> None:
    config = MLPrepConfigLoader.load(
        _CONFIG_DIRECTORY / "default.yaml",
        dotted_overrides={
            "pipeline.output_directory": str(
                tmp_path / "secret-path-marker"
            ),
            "phase1.problem_contract.unit_of_observation": (
                "secret-config-marker"
            ),
        },
    )
    assert config.configuration_fingerprint == (
        PipelineExecutionManifest.fingerprint_configuration(
            config.resolved_configuration
        )
    )

    manifest = PipelineExecutionManifest(
        execution_id="execution",
        source_fingerprint="source",
        configuration_fingerprint=config.configuration_fingerprint,
    )
    report = json.dumps(manifest.model_dump(mode="json"))

    assert "secret-path-marker" not in report
    assert "secret-config-marker" not in report


def test_config_version_alias_and_conflicts_are_validated() -> None:
    config = MLPrepConfigLoader.from_dict(
        {
            "phase1": yaml.safe_load(
                (_CONFIG_DIRECTORY / "default.yaml").read_text(
                    encoding="utf-8"
                )
            )["phase1"],
            "phase2": yaml.safe_load(
                (_CONFIG_DIRECTORY / "default.yaml").read_text(
                    encoding="utf-8"
                )
            )["phase2"],
            "schema_version": "1",
        }
    )
    assert config.config_version == "1"
    assert "config_version" in config.to_dict()
    assert "schema_version" not in config.to_dict()

    with pytest.raises(ValueError, match="conflicts with the legacy"):
        MLPrepConfigLoader.from_dict(
            {
                "phase1": {},
                "phase2": {},
                "config_version": "1",
                "schema_version": "2",
            }
        )

    with pytest.raises(ValueError, match="Unsupported MLPrep config version"):
        MLPrepConfigLoader.load(
            _CONFIG_DIRECTORY / "default.yaml",
            dotted_overrides={"config_version": "2"},
        )

    with pytest.raises(ValueError, match="Invalid YAML value"):
        MLPrepConfigLoader.load(
            _CONFIG_DIRECTORY / "default.yaml",
            environ={
                "MLPREP_PIPELINE__SAVE_INTERMEDIATE_STATES": (
                    "!!python/object/apply:os.system ['whoami']"
                )
            },
        )


def test_legacy_schema_version_is_accepted_and_emits_config_version() -> None:
    source = yaml.safe_load(
        (_CONFIG_DIRECTORY / "default.yaml").read_text(encoding="utf-8")
    )
    source["schema_version"] = "1"
    source.pop("config_version")

    config = MLPrepConfigLoader.from_dict(source)

    assert config.schema_version == "1"
    assert config.config_version == "1"
    assert config.resolved_configuration["config_version"] == "1"
    assert "schema_version" not in config.resolved_configuration


def test_data_source_configuration_is_loaded_and_fingerprinted(tmp_path) -> None:
    source = yaml.safe_load(
        (_CONFIG_DIRECTORY / "default.yaml").read_text(encoding="utf-8")
    )
    source["data_source"] = {
        "type": "csv",
        "config": {
            "path": str(tmp_path / "data.csv"),
            "maximum_file_size_mb": 10,
        },
    }

    config = MLPrepConfigLoader.from_dict(source)

    assert config.data_source is not None
    assert config.data_source.source_type == "csv"
    assert config.resolved_configuration["data_source"]["type"] == "csv"
    assert "data_source" in config.to_dict()
