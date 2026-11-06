"""Unified configuration for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from mlprep.phases.phase4.contract.imputation_contract import ImputationContractOptions
from mlprep.phases.phase4.contract.imputation_plan import ImputationPlanOptions
from mlprep.phases.phase4.detection.missingness_scanner import MissingnessScannerOptions
from mlprep.phases.phase4.orchestration.imputation_pipeline import (
    ImputationPipelineOptions,
)
from mlprep.phases.phase4.output_governance.readiness import Phase4ReadinessOptions
from mlprep.phases.phase4.output_governance.report import Phase4ReportOptions
from mlprep.phases.phase4.validation.distribution_impact import (
    DistributionImpactOptions,
)
from mlprep.phases.phase4.validation.remaining_missingness import (
    RemainingMissingnessOptions,
)
from mlprep.phases.phase4.validation.row_integrity import RowIntegrityOptions
from mlprep.phases.phase4.validation.schema_consistency import SchemaConsistencyOptions
from mlprep.phases.phase4.validation.target_integrity import TargetIntegrityOptions


@dataclass(slots=True)
class Phase4ExecutionOptions:
    """General execution options for Phase 4."""

    enabled: bool = True
    stop_on_contract_failure: bool = True
    stop_on_plan_failure: bool = True
    stop_on_component_exception: bool = True
    store_phase_state_in_result: bool = True
    store_execution_metadata: bool = True
    require_authorized_test_partition: bool = True
    relock_test_partition: bool = True
    generate_validation_findings: bool = True

    def __post_init__(self) -> None:
        """Validate execution options."""
        for option in fields(self):
            name = option.name
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean.")


@dataclass(slots=True)
class Phase4ValidationOptions:
    """Complete Phase 4 validation configuration."""

    remaining_missingness: RemainingMissingnessOptions = field(
        default_factory=RemainingMissingnessOptions
    )
    schema_consistency: SchemaConsistencyOptions = field(
        default_factory=SchemaConsistencyOptions
    )
    target_integrity: TargetIntegrityOptions = field(
        default_factory=TargetIntegrityOptions
    )
    row_integrity: RowIntegrityOptions = field(default_factory=RowIntegrityOptions)
    distribution_impact: DistributionImpactOptions = field(
        default_factory=DistributionImpactOptions
    )


@dataclass(slots=True)
class Phase4Config:
    """Complete configuration for Phase 4."""

    execution: Phase4ExecutionOptions = field(default_factory=Phase4ExecutionOptions)
    contract: ImputationContractOptions = field(
        default_factory=ImputationContractOptions
    )
    plan: ImputationPlanOptions = field(default_factory=ImputationPlanOptions)
    missingness_scanner: MissingnessScannerOptions = field(
        default_factory=MissingnessScannerOptions
    )
    pipeline: ImputationPipelineOptions = field(
        default_factory=ImputationPipelineOptions
    )
    validation: Phase4ValidationOptions = field(default_factory=Phase4ValidationOptions)
    readiness: Phase4ReadinessOptions = field(default_factory=Phase4ReadinessOptions)
    reporting: Phase4ReportOptions = field(default_factory=Phase4ReportOptions)

    def __post_init__(self) -> None:
        """Validate and synchronize configuration."""
        self.contract.require_locked_test_partition = (
            self.execution.require_authorized_test_partition
        )
        self.readiness.require_authorized_test_transform = (
            self.execution.require_authorized_test_partition
        )
        self.readiness.require_locked_test_partition = (
            self.execution.relock_test_partition
        )

        allow_remaining = (
            self.pipeline.partition_transformer.allow_unresolved_missing_values
        )
        if allow_remaining:
            self.validation.remaining_missingness.allow_any_remaining_missingness = True

        if self.plan.allow_drop_feature != (
            self.pipeline.partition_transformer.allow_drop_feature_entries
        ):
            raise ValueError(
                "Feature-dropping policy must match between the plan and transformer."
            )

    @classmethod
    def create(
        cls,
        *,
        allow_remaining_missingness: bool = False,
        generate_missing_indicators: bool = True,
        allow_feature_dropping: bool = False,
        enable_model_imputation: bool = False,
        stop_on_component_exception: bool = True,
        expose_learned_values_in_report: bool = False,
    ) -> Phase4Config:
        """Create a common Phase 4 configuration."""
        execution = Phase4ExecutionOptions(
            stop_on_component_exception=stop_on_component_exception
        )
        plan = ImputationPlanOptions(allow_drop_feature=allow_feature_dropping)
        pipeline = ImputationPipelineOptions()
        pipeline.indicators.enabled = generate_missing_indicators
        pipeline.partition_transformer.allow_drop_feature_entries = (
            allow_feature_dropping
        )
        pipeline.partition_transformer.allow_unresolved_missing_values = (
            allow_remaining_missingness
        )
        pipeline.model.enabled = enable_model_imputation

        validation = Phase4ValidationOptions()
        validation.remaining_missingness.allow_any_remaining_missingness = (
            allow_remaining_missingness
        )
        validation.remaining_missingness.maximum_total_missing_count = (
            None if allow_remaining_missingness else 0
        )
        reporting = Phase4ReportOptions(
            expose_learned_values=expose_learned_values_in_report
        )
        return cls(
            execution=execution,
            plan=plan,
            pipeline=pipeline,
            validation=validation,
            reporting=reporting,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration into a serializable dictionary."""
        serialized = self._serialize(self)
        if not isinstance(serialized, dict):
            raise TypeError("Phase 4 configuration did not serialize to a mapping.")
        return serialized

    @classmethod
    def _serialize(cls, value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            return {
                name: cls._serialize(getattr(value, name))
                for name in value.__dataclass_fields__
            }
        if hasattr(value, "model_dump"):
            return cls._serialize(value.model_dump(mode="json"))
        if isinstance(value, dict):
            return {str(key): cls._serialize(item) for key, item in value.items()}
        if isinstance(value, set):
            return [
                cls._serialize(item)
                for item in sorted(value, key=repr)
            ]
        if isinstance(value, (list, tuple)):
            return [cls._serialize(item) for item in value]
        if hasattr(value, "value"):
            return value.value
        return value
