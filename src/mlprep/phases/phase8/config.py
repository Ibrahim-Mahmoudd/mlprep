"""Unified configuration for Phase 8 training-set balancing."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

from mlprep.phases.phase8.analysis import (
    ClassDistributionOptions,
    ImbalanceMetricsOptions,
    StrategyRecommenderOptions,
)
from mlprep.phases.phase8.contract import (
    BalancingContractOptions,
    BalancingPlanOptions,
)
from mlprep.phases.phase8.orchestration import (
    BalancingPipelineOptions,
    PartitionGuardOptions,
)
from mlprep.phases.phase8.output_governance import (
    Phase8ReadinessOptions,
    Phase8ReportOptions,
)
from mlprep.phases.phase8.validation import (
    DistributionValidationOptions,
    PartitionIntegrityOptions,
    SchemaConsistencyOptions,
    SyntheticIntegrityOptions,
    TargetIntegrityOptions,
)
from mlprep.phases.phase8.weighting import (
    ClassWeightOptions,
    SampleWeightOptions,
)


@dataclass(slots=True)
class Phase8ExecutionOptions:
    """Execution and governance options for Phase 8."""

    enabled: bool = True
    stop_on_contract_failure: bool = True
    stop_on_plan_failure: bool = True
    stop_on_component_exception: bool = True
    store_phase_state_in_result: bool = True
    store_execution_metadata: bool = True
    generate_validation_findings: bool = True


@dataclass(slots=True)
class Phase8ValidationOptions:
    """Validation settings for Phase 8 outputs."""

    distribution: DistributionValidationOptions = field(
        default_factory=DistributionValidationOptions
    )
    schema_consistency: SchemaConsistencyOptions = field(
        default_factory=SchemaConsistencyOptions
    )
    synthetic_integrity: SyntheticIntegrityOptions = field(
        default_factory=SyntheticIntegrityOptions
    )
    target_integrity: TargetIntegrityOptions = field(
        default_factory=TargetIntegrityOptions
    )
    partition_integrity: PartitionIntegrityOptions = field(
        default_factory=PartitionIntegrityOptions
    )


@dataclass(slots=True)
class Phase8Config:
    """Complete configuration for Phase 8."""

    execution: Phase8ExecutionOptions = field(
        default_factory=Phase8ExecutionOptions
    )
    contract: BalancingContractOptions = field(
        default_factory=BalancingContractOptions
    )
    plan: BalancingPlanOptions = field(
        default_factory=BalancingPlanOptions
    )
    distribution_analysis: ClassDistributionOptions = field(
        default_factory=ClassDistributionOptions
    )
    imbalance_metrics: ImbalanceMetricsOptions = field(
        default_factory=ImbalanceMetricsOptions
    )
    strategy_recommender: StrategyRecommenderOptions = field(
        default_factory=StrategyRecommenderOptions
    )
    class_weights: ClassWeightOptions = field(
        default_factory=ClassWeightOptions
    )
    sample_weights: SampleWeightOptions = field(
        default_factory=SampleWeightOptions
    )
    partition_guard: PartitionGuardOptions = field(
        default_factory=PartitionGuardOptions
    )
    pipeline: BalancingPipelineOptions = field(
        default_factory=BalancingPipelineOptions
    )
    validation: Phase8ValidationOptions = field(
        default_factory=Phase8ValidationOptions
    )
    readiness: Phase8ReadinessOptions = field(
        default_factory=Phase8ReadinessOptions
    )
    reporting: Phase8ReportOptions = field(
        default_factory=Phase8ReportOptions
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe representation of this config."""

        return self._serialize(self)

    @classmethod
    def _serialize(cls, value: Any) -> Any:
        if is_dataclass(value):
            return {
                item.name: cls._serialize(getattr(value, item.name))
                for item in fields(value)
            }
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {
                str(key): cls._serialize(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [cls._serialize(item) for item in value]
        if hasattr(value, "model_dump"):
            return cls._serialize(value.model_dump(mode="json"))
        return value
