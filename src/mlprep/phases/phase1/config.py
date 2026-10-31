"""Unified configuration for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mlprep.phases.phase1.contract.column_roles import (
    ColumnRoleOptions,
)
from mlprep.phases.phase1.contract.problem_contract import (
    ProblemContractOptions,
)
from mlprep.phases.phase1.deterministic_normalization.column_names import (
    ColumnNameOptions,
)
from mlprep.phases.phase1.deterministic_normalization.missing_tokens import (
    MissingTokenOptions,
)
from mlprep.phases.phase1.deterministic_normalization.text_normalization import (
    TextNormalizationOptions,
)
from mlprep.phases.phase1.inspection.cardinality import (
    CardinalityInspectionOptions,
)
from mlprep.phases.phase1.inspection.mixed_types import (
    MixedTypeDetectionOptions,
)
from mlprep.phases.phase1.inspection.physical_types import (
    PhysicalTypeInspectionOptions,
)
from mlprep.phases.phase1.inspection.semantic_types import (
    SemanticTypeInferenceOptions,
)
from mlprep.phases.phase1.inspection.temporal_detection import (
    TemporalDetectionOptions,
)
from mlprep.phases.phase1.output_governance.fingerprint import (
    FingerprintOptions,
)
from mlprep.phases.phase1.output_governance.memory import (
    MemoryOptimizerOptions,
)
from mlprep.phases.phase1.output_governance.readiness import (
    ReadinessOptions,
)
from mlprep.phases.phase1.output_governance.report import (
    ReportOptions,
)
from mlprep.phases.phase1.validation_and_risk.row_identity import (
    RowIdentityOptions,
)
from mlprep.phases.phase1.output_governance.schema import (
    SchemaBuilderOptions,
)
from mlprep.phases.phase1.validation_and_risk.constraints import (
    ConstraintOptions,
)
from mlprep.phases.phase1.validation_and_risk.duplicate_columns import (
    DuplicateColumnOptions,
)
from mlprep.phases.phase1.validation_and_risk.duplicate_rows import (
    DuplicateRowOptions,
)
from mlprep.phases.phase1.validation_and_risk.leakage_risks import (
    LeakageRiskOptions,
)
from mlprep.phases.phase1.validation_and_risk.sensitive_data import (
    SensitiveDataOptions,
)
from mlprep.phases.phase1.validation_and_risk.target_validation import (
    TargetValidationOptions,
)
from mlprep.types import TaskType


@dataclass(slots=True)
class Phase1ExecutionOptions:
    """General execution options for Phase 1."""

    enabled: bool = True

    copy_input_dataframe: bool = True

    stop_on_critical_finding: bool = False
    stop_on_component_exception: bool = True

    store_phase_state_in_result: bool = True

    validate_after_each_mutation: bool = True

    def __post_init__(self) -> None:
        """Validate execution options."""

        if not isinstance(self.enabled, bool):
            raise TypeError(
                "enabled must be a boolean."
            )

        if not isinstance(
            self.copy_input_dataframe,
            bool,
        ):
            raise TypeError(
                "copy_input_dataframe must be a boolean."
            )


@dataclass(slots=True)
class Phase1Config:
    """Complete configuration for Phase 1."""

    problem_contract: ProblemContractOptions
    column_roles: ColumnRoleOptions
    target_validation: TargetValidationOptions
    leakage_risks: LeakageRiskOptions

    execution: Phase1ExecutionOptions = field(
        default_factory=Phase1ExecutionOptions
    )

    row_identity: RowIdentityOptions = field(
        default_factory=RowIdentityOptions
    )

    column_names: ColumnNameOptions = field(
        default_factory=ColumnNameOptions
    )

    text_normalization: TextNormalizationOptions = field(
        default_factory=TextNormalizationOptions
    )

    missing_tokens: MissingTokenOptions = field(
        default_factory=MissingTokenOptions
    )

    physical_types: PhysicalTypeInspectionOptions = field(
        default_factory=PhysicalTypeInspectionOptions
    )

    mixed_types: MixedTypeDetectionOptions = field(
        default_factory=MixedTypeDetectionOptions
    )

    temporal_detection: TemporalDetectionOptions = field(
        default_factory=TemporalDetectionOptions
    )

    semantic_types: SemanticTypeInferenceOptions = field(
        default_factory=SemanticTypeInferenceOptions
    )

    cardinality: CardinalityInspectionOptions = field(
        default_factory=CardinalityInspectionOptions
    )

    duplicate_columns: DuplicateColumnOptions = field(
        default_factory=DuplicateColumnOptions
    )

    duplicate_rows: DuplicateRowOptions = field(
        default_factory=DuplicateRowOptions
    )

    constraints: ConstraintOptions = field(
        default_factory=ConstraintOptions
    )

    sensitive_data: SensitiveDataOptions = field(
        default_factory=SensitiveDataOptions
    )

    memory: MemoryOptimizerOptions = field(
        default_factory=MemoryOptimizerOptions
    )

    schema: SchemaBuilderOptions = field(
        default_factory=SchemaBuilderOptions
    )

    fingerprint: FingerprintOptions = field(
        default_factory=FingerprintOptions
    )

    readiness: ReadinessOptions = field(
        default_factory=ReadinessOptions
    )

    reporting: ReportOptions = field(
        default_factory=ReportOptions
    )

    def __post_init__(self) -> None:
        """Validate relationships between configuration sections."""

        self._validate_target_consistency()
        self._validate_task_consistency()
        self._validate_positive_class_consistency()
        self._validate_timestamp_consistency()
        self._validate_observation_key_consistency()
        self._validate_row_identity_consistency()
        self._synchronize_component_options()

    def _validate_target_consistency(
        self,
    ) -> None:
        """Ensure all components use the same target column."""

        declared_targets = {
            self.problem_contract.target_column,
            self.column_roles.target_column,
            self.target_validation.target_column,
            self.leakage_risks.target_column,
        }

        if len(declared_targets) == 1:
            return

        raise ValueError(
            "All Phase 1 components must use the same target "
            f"column. Received: {sorted(declared_targets)}"
        )

    def _validate_task_consistency(
        self,
    ) -> None:
        """Ensure problem and target validation use the same task."""

        if (
            self.problem_contract.task_type
            == self.target_validation.task_type
        ):
            return

        raise ValueError(
            "problem_contract.task_type and "
            "target_validation.task_type must match."
        )

    def _validate_positive_class_consistency(
        self,
    ) -> None:
        """Ensure binary-classification positive classes match."""

        if (
            self.problem_contract.task_type
            != TaskType.BINARY_CLASSIFICATION
        ):
            return

        contract_positive_class = (
            self.problem_contract.positive_class
        )

        validation_positive_class = (
            self.target_validation.positive_class
        )

        if (
            contract_positive_class
            == validation_positive_class
        ):
            return

        raise ValueError(
            "problem_contract.positive_class and "
            "target_validation.positive_class must match."
        )

    def _validate_timestamp_consistency(
        self,
    ) -> None:
        """Ensure timestamp declarations are consistent."""

        contract_prediction_timestamp = (
            self.problem_contract
            .prediction_timestamp_column
        )

        role_prediction_timestamp = (
            self.column_roles
            .prediction_timestamp_column
        )

        if (
            contract_prediction_timestamp
            != role_prediction_timestamp
        ):
            raise ValueError(
                "Prediction timestamp declarations must match "
                "between problem_contract and column_roles."
            )

        contract_outcome_timestamp = (
            self.problem_contract
            .outcome_timestamp_column
        )

        role_outcome_timestamp = (
            self.column_roles
            .outcome_timestamp_column
        )

        if (
            contract_outcome_timestamp
            != role_outcome_timestamp
        ):
            raise ValueError(
                "Outcome timestamp declarations must match "
                "between problem_contract and column_roles."
            )

    def _validate_observation_key_consistency(
        self,
    ) -> None:
        """Ensure duplicate detection uses the contract key."""

        configured_duplicate_key = (
            self.duplicate_rows.observation_key
        )

        contract_key = (
            self.problem_contract.observation_key
        )

        if (
            configured_duplicate_key
            and configured_duplicate_key
            != contract_key
        ):
            raise ValueError(
                "duplicate_rows.observation_key must match "
                "problem_contract.observation_key."
            )

    def _validate_row_identity_consistency(
        self,
    ) -> None:
        """Ensure internal row-ID declarations match."""

        if (
            self.row_identity.column_name
            == self.duplicate_rows
            .internal_row_id_column
        ):
            return

        raise ValueError(
            "row_identity.column_name and "
            "duplicate_rows.internal_row_id_column must match."
        )

    def _synchronize_component_options(
        self,
    ) -> None:
        """Propagate authoritative settings to related components."""

        target_column = (
            self.problem_contract.target_column
        )

        self.column_roles.target_column = target_column

        self.target_validation.target_column = (
            target_column
        )

        self.leakage_risks.target_column = target_column

        self.duplicate_rows.target_column = (
            target_column
        )

        self.duplicate_rows.observation_key = list(
            self.problem_contract.observation_key
        )

        self.duplicate_rows.internal_row_id_column = (
            self.row_identity.column_name
        )

        self.column_roles.group_columns = list(
            self.problem_contract.group_columns
        )

        self.column_roles.prediction_timestamp_column = (
            self.problem_contract
            .prediction_timestamp_column
        )

        self.column_roles.outcome_timestamp_column = (
            self.problem_contract
            .outcome_timestamp_column
        )

        timestamp_columns = list(
            self.column_roles.timestamp_columns
        )

        prediction_timestamp = (
            self.problem_contract
            .prediction_timestamp_column
        )

        outcome_timestamp = (
            self.problem_contract
            .outcome_timestamp_column
        )

        if (
            prediction_timestamp is not None
            and prediction_timestamp
            not in timestamp_columns
        ):
            timestamp_columns.append(
                prediction_timestamp
            )

        if (
            outcome_timestamp is not None
            and outcome_timestamp
            not in timestamp_columns
        ):
            timestamp_columns.append(
                outcome_timestamp
            )

        self.column_roles.timestamp_columns = (
            timestamp_columns
        )

        self.leakage_risks.inference_unavailable_columns = list(
            dict.fromkeys(
                [
                    *self.leakage_risks
                    .inference_unavailable_columns,
                    *self.column_roles
                    .inference_unavailable_columns,
                ]
            )
        )

        reserved_names = set(
            self.column_names.reserved_names
        )

        reserved_names.add(
            self.row_identity.column_name
        )

        self.column_names.reserved_names = (
            reserved_names
        )

    @classmethod
    def create(
        cls,
        *,
        task_type: TaskType,
        target_column: str,
        unit_of_observation: str,
        positive_class: str | int | bool | None = None,
        observation_key: list[str] | None = None,
        entity_columns: list[str] | None = None,
        group_columns: list[str] | None = None,
        prediction_timestamp_column: str | None = None,
        outcome_timestamp_column: str | None = None,
        target_horizon: str | None = None,
        feature_columns: list[str] | None = None,
        identifier_columns: list[str] | None = None,
        sensitive_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
        ignored_columns: list[str] | None = None,
        inference_unavailable_columns: list[str] | None = None,
    ) -> Phase1Config:
        """Create a Phase 1 configuration using common settings."""

        resolved_observation_key = list(
            observation_key or []
        )

        resolved_entity_columns = list(
            entity_columns or []
        )

        resolved_group_columns = list(
            group_columns or []
        )

        resolved_identifier_columns = list(
            identifier_columns or []
        )

        resolved_sensitive_columns = list(
            sensitive_columns or []
        )

        resolved_metadata_columns = list(
            metadata_columns or []
        )

        resolved_ignored_columns = list(
            ignored_columns or []
        )

        resolved_inference_unavailable = list(
            inference_unavailable_columns or []
        )

        problem_contract = ProblemContractOptions(
            task_type=task_type,
            target_column=target_column,
            positive_class=positive_class,
            unit_of_observation=unit_of_observation,
            observation_key=resolved_observation_key,
            entity_columns=resolved_entity_columns,
            group_columns=resolved_group_columns,
            prediction_timestamp_column=(
                prediction_timestamp_column
            ),
            outcome_timestamp_column=(
                outcome_timestamp_column
            ),
            target_horizon=target_horizon,
        )

        column_roles = ColumnRoleOptions(
            target_column=target_column,
            feature_columns=feature_columns,
            identifier_columns=(
                resolved_identifier_columns
            ),
            group_columns=resolved_group_columns,
            sensitive_columns=(
                resolved_sensitive_columns
            ),
            metadata_columns=(
                resolved_metadata_columns
            ),
            ignored_columns=(
                resolved_ignored_columns
            ),
            inference_unavailable_columns=(
                resolved_inference_unavailable
            ),
            prediction_timestamp_column=(
                prediction_timestamp_column
            ),
            outcome_timestamp_column=(
                outcome_timestamp_column
            ),
        )

        target_validation = TargetValidationOptions(
            target_column=target_column,
            task_type=task_type,
            positive_class=positive_class,
        )

        leakage_risks = LeakageRiskOptions(
            target_column=target_column,
            inference_unavailable_columns=(
                resolved_inference_unavailable
            ),
        )

        duplicate_rows = DuplicateRowOptions(
            observation_key=resolved_observation_key,
            target_column=target_column,
        )

        return cls(
            problem_contract=problem_contract,
            column_roles=column_roles,
            target_validation=target_validation,
            leakage_risks=leakage_risks,
            duplicate_rows=duplicate_rows,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the complete configuration into a dictionary."""

        return {
            "execution": self._dataclass_to_dict(
                self.execution
            ),
            "problem_contract": self._dataclass_to_dict(
                self.problem_contract
            ),
            "column_roles": self._dataclass_to_dict(
                self.column_roles
            ),
            "row_identity": self._dataclass_to_dict(
                self.row_identity
            ),
            "column_names": self._dataclass_to_dict(
                self.column_names
            ),
            "text_normalization": (
                self._dataclass_to_dict(
                    self.text_normalization
                )
            ),
            "missing_tokens": self._dataclass_to_dict(
                self.missing_tokens
            ),
            "physical_types": self._dataclass_to_dict(
                self.physical_types
            ),
            "mixed_types": self._dataclass_to_dict(
                self.mixed_types
            ),
            "temporal_detection": (
                self._dataclass_to_dict(
                    self.temporal_detection
                )
            ),
            "semantic_types": self._dataclass_to_dict(
                self.semantic_types
            ),
            "cardinality": self._dataclass_to_dict(
                self.cardinality
            ),
            "duplicate_columns": (
                self._dataclass_to_dict(
                    self.duplicate_columns
                )
            ),
            "duplicate_rows": self._dataclass_to_dict(
                self.duplicate_rows
            ),
            "target_validation": (
                self._dataclass_to_dict(
                    self.target_validation
                )
            ),
            "constraints": self._dataclass_to_dict(
                self.constraints
            ),
            "sensitive_data": self._dataclass_to_dict(
                self.sensitive_data
            ),
            "leakage_risks": self._dataclass_to_dict(
                self.leakage_risks
            ),
            "memory": self._dataclass_to_dict(
                self.memory
            ),
            "schema": self._dataclass_to_dict(
                self.schema
            ),
            "fingerprint": self._dataclass_to_dict(
                self.fingerprint
            ),
            "readiness": self._dataclass_to_dict(
                self.readiness
            ),
            "reporting": self._dataclass_to_dict(
                self.reporting
            ),
        }

    def save_yaml(
        self,
        path: str | Path,
    ) -> Path:
        """Save the resolved Phase 1 configuration as YAML."""

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            yaml.safe_dump(
                self.to_dict(),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        return output_path

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any],
    ) -> Phase1Config:
        """Create Phase1Config from a configuration dictionary."""

        allowed_sections = {
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

        unknown_sections = set(values).difference(
            allowed_sections
        )

        if unknown_sections:
            formatted_sections = ", ".join(
                sorted(unknown_sections)
            )

            raise ValueError(
                "Unknown Phase 1 configuration sections: "
                f"{formatted_sections}"
            )

        required_sections = {
            "problem_contract",
            "column_roles",
            "target_validation",
            "leakage_risks",
        }

        missing_sections = required_sections.difference(
            values
        )

        if missing_sections:
            formatted_sections = ", ".join(
                sorted(missing_sections)
            )

            raise ValueError(
                "Missing required Phase 1 configuration sections: "
                f"{formatted_sections}"
            )

        return cls(
            execution=Phase1ExecutionOptions(
                **values.get(
                    "execution",
                    {},
                )
            ),
            problem_contract=ProblemContractOptions(
                **values["problem_contract"]
            ),
            column_roles=ColumnRoleOptions(
                **values["column_roles"]
            ),
            row_identity=RowIdentityOptions(
                **values.get(
                    "row_identity",
                    {},
                )
            ),
            column_names=ColumnNameOptions(
                **values.get(
                    "column_names",
                    {},
                )
            ),
            text_normalization=TextNormalizationOptions(
                **values.get(
                    "text_normalization",
                    {},
                )
            ),
            missing_tokens=MissingTokenOptions(
                **values.get(
                    "missing_tokens",
                    {},
                )
            ),
            physical_types=PhysicalTypeInspectionOptions(
                **values.get(
                    "physical_types",
                    {},
                )
            ),
            mixed_types=MixedTypeDetectionOptions(
                **values.get(
                    "mixed_types",
                    {},
                )
            ),
            temporal_detection=TemporalDetectionOptions(
                **values.get(
                    "temporal_detection",
                    {},
                )
            ),
            semantic_types=SemanticTypeInferenceOptions(
                **values.get(
                    "semantic_types",
                    {},
                )
            ),
            cardinality=CardinalityInspectionOptions(
                **values.get(
                    "cardinality",
                    {},
                )
            ),
            duplicate_columns=DuplicateColumnOptions(
                **values.get(
                    "duplicate_columns",
                    {},
                )
            ),
            duplicate_rows=DuplicateRowOptions(
                **values.get(
                    "duplicate_rows",
                    {},
                )
            ),
            target_validation=TargetValidationOptions(
                **values["target_validation"]
            ),
            constraints=ConstraintOptions(
                **values.get(
                    "constraints",
                    {},
                )
            ),
            sensitive_data=SensitiveDataOptions(
                **values.get(
                    "sensitive_data",
                    {},
                )
            ),
            leakage_risks=LeakageRiskOptions(
                **values["leakage_risks"]
            ),
            memory=MemoryOptimizerOptions(
                **values.get(
                    "memory",
                    {},
                )
            ),
            schema=SchemaBuilderOptions(
                **values.get(
                    "schema",
                    {},
                )
            ),
            fingerprint=FingerprintOptions(
                **values.get(
                    "fingerprint",
                    {},
                )
            ),
            readiness=ReadinessOptions(
                **values.get(
                    "readiness",
                    {},
                )
            ),
            reporting=ReportOptions(
                **values.get(
                    "reporting",
                    {},
                )
            ),
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> Phase1Config:
        """Load Phase1Config from a YAML file."""

        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Configuration file does not exist: "
                f"{input_path}"
            )

        loaded_content = yaml.safe_load(
            input_path.read_text(
                encoding="utf-8"
            )
        )

        if loaded_content is None:
            raise ValueError(
                "The configuration file is empty."
            )

        if not isinstance(
            loaded_content,
            dict,
        ):
            raise TypeError(
                "The Phase 1 YAML root must be a mapping."
            )

        return cls.from_dict(
            loaded_content
        )

    @classmethod
    def _dataclass_to_dict(
        cls,
        value: Any,
    ) -> Any:
        """Recursively convert dataclass values to YAML-safe values."""

        if hasattr(value, "__dataclass_fields__"):
            return {
                field_name: cls._dataclass_to_dict(
                    getattr(
                        value,
                        field_name,
                    )
                )
                for field_name in value.__dataclass_fields__
            }

        if isinstance(value, dict):
            return {
                cls._dataclass_to_dict(key): (
                    cls._dataclass_to_dict(item)
                )
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                cls._dataclass_to_dict(item)
                for item in value
            ]

        if hasattr(value, "value"):
            return value.value

        if callable(value):
            callable_name = getattr(
                value,
                "__qualname__",
                getattr(
                    value,
                    "__name__",
                    repr(value),
                ),
            )

            callable_module = getattr(
                value,
                "__module__",
                None,
            )

            if callable_module is None:
                return callable_name

            return (
                f"{callable_module}."
                f"{callable_name}"
            )

        return value