"""Unified configuration for Phase 7 feature selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from mlprep.phases.phase7.contract import (
    SelectionContractOptions,
    SelectionPlanOptions,
)
from mlprep.phases.phase7.orchestration import (
    SelectionPipelineOptions,
)
from mlprep.phases.phase7.output_governance import (
    Phase7ReadinessOptions,
    Phase7ReportOptions,
)
from mlprep.phases.phase7.validation import (
    FeatureCountOptions,
    ProtectedFeaturesOptions,
    RowIntegrityOptions,
    SchemaConsistencyOptions,
    TargetIntegrityOptions,
)


@dataclass(slots=True)
class Phase7ExecutionOptions:
    """General execution and test-governance options."""

    enabled: bool = True
    stop_on_contract_failure: bool = True
    stop_on_plan_failure: bool = True
    stop_on_component_exception: bool = True
    keep_test_partition_protected: bool = True
    store_phase_state_in_result: bool = True
    store_execution_metadata: bool = True


@dataclass(slots=True)
class Phase7ValidationOptions:
    """Validation settings for Phase 7 output."""

    feature_count: FeatureCountOptions = field(
        default_factory=FeatureCountOptions
    )
    protected_features: ProtectedFeaturesOptions = field(
        default_factory=ProtectedFeaturesOptions
    )
    row_integrity: RowIntegrityOptions = field(
        default_factory=RowIntegrityOptions
    )
    schema_consistency: SchemaConsistencyOptions = field(
        default_factory=SchemaConsistencyOptions
    )
    target_integrity: TargetIntegrityOptions = field(
        default_factory=TargetIntegrityOptions
    )


@dataclass(slots=True)
class Phase7Config:
    """Complete configuration for Phase 7."""

    execution: Phase7ExecutionOptions = field(
        default_factory=Phase7ExecutionOptions
    )
    contract: SelectionContractOptions = field(
        default_factory=SelectionContractOptions
    )
    plan: SelectionPlanOptions = field(default_factory=SelectionPlanOptions)
    pipeline: SelectionPipelineOptions = field(
        default_factory=SelectionPipelineOptions
    )
    validation: Phase7ValidationOptions = field(
        default_factory=Phase7ValidationOptions
    )
    readiness: Phase7ReadinessOptions = field(
        default_factory=Phase7ReadinessOptions
    )
    reporting: Phase7ReportOptions = field(
        default_factory=Phase7ReportOptions
    )