"""Tests for the unified Phase 1 configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlprep.phases.phase1.config import (
    Phase1Config,
    Phase1ExecutionOptions,
)
from mlprep.phases.phase1.config_loader import (
    Phase1ConfigLoader,
)
from mlprep.types import TaskType


def test_create_binary_classification_config() -> None:
    """Create a valid binary-classification configuration."""

    config = Phase1Config.create(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        positive_class=1,
        unit_of_observation=(
            "One customer at one monthly snapshot"
        ),
        observation_key=[
            "customer_id",
            "snapshot_date",
        ],
        entity_columns=[
            "customer_id",
        ],
        group_columns=[
            "customer_id",
        ],
        prediction_timestamp_column="snapshot_date",
        identifier_columns=[
            "customer_id",
        ],
    )

    assert (
        config.problem_contract.task_type
        == TaskType.BINARY_CLASSIFICATION
    )

    assert (
        config.problem_contract.target_column
        == "churn"
    )

    assert (
        config.target_validation.target_column
        == "churn"
    )

    assert (
        config.leakage_risks.target_column
        == "churn"
    )

    assert (
        config.column_roles.target_column
        == "churn"
    )

    assert (
        config.problem_contract.positive_class
        == 1
    )

    assert (
        config.target_validation.positive_class
        == 1
    )

    assert (
        config.duplicate_rows.observation_key
        == [
            "customer_id",
            "snapshot_date",
        ]
    )


def test_create_regression_config() -> None:
    """Create a valid regression configuration."""

    config = Phase1Config.create(
        task_type=TaskType.REGRESSION,
        target_column="price",
        unit_of_observation="One property listing",
        observation_key=[
            "property_id",
        ],
        entity_columns=[
            "property_id",
        ],
        identifier_columns=[
            "property_id",
        ],
    )

    assert (
        config.problem_contract.task_type
        == TaskType.REGRESSION
    )

    assert (
        config.target_validation.task_type
        == TaskType.REGRESSION
    )

    assert config.problem_contract.positive_class is None

    assert config.target_validation.positive_class is None


def test_binary_classification_requires_positive_class() -> None:
    """Reject binary classification without a positive class."""

    with pytest.raises(
        ValueError,
        match="positive_class is required",
    ):
        Phase1Config.create(
            task_type=TaskType.BINARY_CLASSIFICATION,
            target_column="churn",
            unit_of_observation="One customer",
        )


def test_regression_rejects_positive_class() -> None:
    """Reject a positive class for regression."""

    with pytest.raises(
        ValueError,
        match=(
            "positive_class is supported only "
            "for binary classification"
        ),
    ):
        Phase1Config.create(
            task_type=TaskType.REGRESSION,
            target_column="price",
            positive_class=1,
            unit_of_observation="One property",
        )


def test_target_settings_are_synchronized() -> None:
    """Synchronize the authoritative target across components."""

    config = Phase1Config.create(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=True,
        unit_of_observation="One record",
    )

    assert (
        config.problem_contract.target_column
        == "target"
    )

    assert (
        config.column_roles.target_column
        == "target"
    )

    assert (
        config.target_validation.target_column
        == "target"
    )

    assert (
        config.leakage_risks.target_column
        == "target"
    )

    assert (
        config.duplicate_rows.target_column
        == "target"
    )


def test_timestamp_settings_are_synchronized() -> None:
    """Synchronize timestamp declarations across components."""

    config = Phase1Config.create(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=1,
        unit_of_observation="One account snapshot",
        prediction_timestamp_column="snapshot_date",
        outcome_timestamp_column="outcome_date",
    )

    assert (
        config.column_roles.prediction_timestamp_column
        == "snapshot_date"
    )

    assert (
        config.column_roles.outcome_timestamp_column
        == "outcome_date"
    )

    assert (
        "snapshot_date"
        in config.column_roles.timestamp_columns
    )

    assert (
        "outcome_date"
        in config.column_roles.timestamp_columns
    )


def test_row_identity_settings_are_synchronized() -> None:
    """Use one internal row-ID name in related components."""

    config = Phase1Config.create(
        task_type=TaskType.REGRESSION,
        target_column="target",
        unit_of_observation="One observation",
    )

    assert (
        config.row_identity.column_name
        == config.duplicate_rows.internal_row_id_column
    )

    assert (
        config.row_identity.column_name
        in config.column_names.reserved_names
    )


def test_inference_unavailable_columns_are_synchronized() -> None:
    """Propagate inference-unavailable columns to leakage checks."""

    config = Phase1Config.create(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=1,
        unit_of_observation="One observation",
        inference_unavailable_columns=[
            "future_status",
            "resolution_date",
        ],
    )

    assert (
        config.column_roles.inference_unavailable_columns
        == [
            "future_status",
            "resolution_date",
        ]
    )

    assert set(
        config.leakage_risks.inference_unavailable_columns
    ) == {
        "future_status",
        "resolution_date",
    }


def test_execution_options_defaults() -> None:
    """Validate default execution behavior."""

    options = Phase1ExecutionOptions()

    assert options.enabled is True
    assert options.copy_input_dataframe is True
    assert options.stop_on_critical_finding is False
    assert options.stop_on_component_exception is True
    assert options.store_phase_state_in_result is True
    assert options.validate_after_each_mutation is True


def test_config_to_dict_contains_all_sections() -> None:
    """Serialize the complete config to a dictionary."""

    config = Phase1Config.create(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=1,
        unit_of_observation="One record",
    )

    values = config.to_dict()

    expected_sections = {
        "execution",
        "problem_contract",
        "column_roles",
        "row_identity",
        "column_names",
        "text_normalization",
        "missing_tokens",
        "physical_types",
        "mixed_types",
        "temporal_detection",
        "semantic_types",
        "cardinality",
        "duplicate_columns",
        "duplicate_rows",
        "target_validation",
        "constraints",
        "sensitive_data",
        "leakage_risks",
        "memory",
        "schema",
        "fingerprint",
        "readiness",
        "reporting",
    }

    assert set(values) == expected_sections

    assert (
        values["problem_contract"]["target_column"]
        == "target"
    )

    assert (
        values["problem_contract"]["task_type"]
        == "binary_classification"
    )


def test_config_from_dict_round_trip() -> None:
    """Rebuild a configuration from serialized values."""

    original_config = Phase1Config.create(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        positive_class=1,
        unit_of_observation="One customer",
        observation_key=[
            "customer_id",
        ],
        identifier_columns=[
            "customer_id",
        ],
    )

    values = original_config.to_dict()

    rebuilt_config = Phase1Config.from_dict(
        values
    )

    assert (
        rebuilt_config.problem_contract.target_column
        == "churn"
    )

    assert (
        rebuilt_config.problem_contract.task_type
        == TaskType.BINARY_CLASSIFICATION
    )

    assert (
        rebuilt_config.problem_contract.observation_key
        == ["customer_id"]
    )

    assert (
        rebuilt_config.column_roles.identifier_columns
        == ["customer_id"]
    )


def test_config_from_dict_rejects_unknown_section() -> None:
    """Reject unsupported root configuration sections."""

    config = Phase1Config.create(
        task_type=TaskType.REGRESSION,
        target_column="target",
        unit_of_observation="One record",
    )

    values = config.to_dict()

    values["unknown_section"] = {
        "enabled": True,
    }

    with pytest.raises(
        ValueError,
        match="Unknown Phase 1 configuration sections",
    ):
        Phase1Config.from_dict(values)


def test_config_from_dict_rejects_missing_section() -> None:
    """Reject missing required configuration sections."""

    config = Phase1Config.create(
        task_type=TaskType.REGRESSION,
        target_column="target",
        unit_of_observation="One record",
    )

    values = config.to_dict()

    values.pop("problem_contract")

    with pytest.raises(
        ValueError,
        match=(
            "Missing required Phase 1 "
            "configuration sections"
        ),
    ):
        Phase1Config.from_dict(values)


def test_save_and_load_yaml(
    tmp_path: Path,
) -> None:
    """Save and reload a YAML configuration."""

    config = Phase1Config.create(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        positive_class=1,
        unit_of_observation="One customer",
        observation_key=[
            "customer_id",
        ],
        identifier_columns=[
            "customer_id",
        ],
    )

    output_path = tmp_path / "phase1.yaml"

    Phase1ConfigLoader.save(
        config=config,
        path=output_path,
    )

    assert output_path.exists()

    loaded_config = Phase1ConfigLoader.load(
        output_path
    )

    assert (
        loaded_config.problem_contract.target_column
        == "churn"
    )

    assert (
        loaded_config.problem_contract.observation_key
        == ["customer_id"]
    )


def test_save_and_load_json(
    tmp_path: Path,
) -> None:
    """Save and reload a JSON configuration."""

    config = Phase1Config.create(
        task_type=TaskType.REGRESSION,
        target_column="price",
        unit_of_observation="One property",
        observation_key=[
            "property_id",
        ],
    )

    output_path = tmp_path / "phase1.json"

    Phase1ConfigLoader.save(
        config=config,
        path=output_path,
    )

    assert output_path.exists()

    loaded_config = Phase1ConfigLoader.load(
        output_path
    )

    assert (
        loaded_config.problem_contract.target_column
        == "price"
    )

    assert (
        loaded_config.problem_contract.task_type
        == TaskType.REGRESSION
    )


def test_save_rejects_existing_file(
    tmp_path: Path,
) -> None:
    """Prevent accidental configuration overwrites."""

    config = Phase1Config.create(
        task_type=TaskType.REGRESSION,
        target_column="target",
        unit_of_observation="One record",
    )

    output_path = tmp_path / "phase1.yaml"

    Phase1ConfigLoader.save(
        config=config,
        path=output_path,
    )

    with pytest.raises(
        FileExistsError,
    ):
        Phase1ConfigLoader.save(
            config=config,
            path=output_path,
            overwrite=False,
        )


def test_save_allows_explicit_overwrite(
    tmp_path: Path,
) -> None:
    """Allow a configuration file to be overwritten explicitly."""

    config = Phase1Config.create(
        task_type=TaskType.REGRESSION,
        target_column="target",
        unit_of_observation="One record",
    )

    output_path = tmp_path / "phase1.yaml"

    Phase1ConfigLoader.save(
        config=config,
        path=output_path,
    )

    Phase1ConfigLoader.save(
        config=config,
        path=output_path,
        overwrite=True,
    )

    assert output_path.exists()


def test_recursive_config_merge() -> None:
    """Merge nested configuration sections recursively."""

    base = {
        "column_names": {
            "enabled": True,
            "case_style": "snake_case",
            "maximum_length": 128,
        },
        "memory": {
            "level": "safe",
        },
    }

    override = {
        "column_names": {
            "case_style": "lower",
        },
    }

    merged = Phase1ConfigLoader.merge(
        base=base,
        override=override,
    )

    assert (
        merged["column_names"]["enabled"]
        is True
    )

    assert (
        merged["column_names"]["case_style"]
        == "lower"
    )

    assert (
        merged["column_names"]["maximum_length"]
        == 128
    )

    assert (
        merged["memory"]["level"]
        == "safe"
    )


def test_dot_override() -> None:
    """Apply a nested setting through a dotted path."""

    values = {
        "column_names": {
            "case_style": "snake_case",
        },
        "memory": {
            "level": "safe",
        },
    }

    updated = (
        Phase1ConfigLoader.apply_dot_overrides(
            values=values,
            overrides={
                "column_names.case_style": "lower",
                "memory.level": "disabled",
            },
        )
    )

    assert (
        updated["column_names"]["case_style"]
        == "lower"
    )

    assert (
        updated["memory"]["level"]
        == "disabled"
    )


def test_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Reject nonexistent configuration paths."""

    missing_path = (
        tmp_path
        / "missing_configuration.yaml"
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        Phase1ConfigLoader.load(
            missing_path
        )


def test_loader_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    """Reject unsupported configuration file extensions."""

    input_path = tmp_path / "phase1.txt"

    input_path.write_text(
        "invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported configuration extension",
    ):
        Phase1ConfigLoader.load(
            input_path
        )


def test_validate_file_returns_no_errors(
    tmp_path: Path,
) -> None:
    """Return no errors for a valid configuration file."""

    config = Phase1Config.create(
        task_type=TaskType.REGRESSION,
        target_column="target",
        unit_of_observation="One record",
    )

    output_path = tmp_path / "phase1.yaml"

    Phase1ConfigLoader.save(
        config=config,
        path=output_path,
    )

    errors = Phase1ConfigLoader.validate_file(
        output_path
    )

    assert errors == []


def test_validate_file_returns_errors(
    tmp_path: Path,
) -> None:
    """Return error messages for an invalid configuration."""

    output_path = tmp_path / "invalid.yaml"

    output_path.write_text(
        "problem_contract: [invalid",
        encoding="utf-8",
    )

    errors = Phase1ConfigLoader.validate_file(
        output_path
    )

    assert len(errors) == 1
