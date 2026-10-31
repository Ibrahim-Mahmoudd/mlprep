"""Unified configuration for Phase 5."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mlprep.phases.phase5.contract.outlier_contract import (
    OutlierContractOptions,
)
from mlprep.phases.phase5.contract.treatment_plan import (
    TreatmentKind,
    TreatmentPlanOptions,
)
from mlprep.phases.phase5.detection.detector_engine import (
    DetectorEngineOptions,
)
from mlprep.phases.phase5.orchestration.outlier_pipeline import (
    OutlierPipelineOptions,
)
from mlprep.phases.phase5.orchestration.partition_transformer import (
    PartitionTransformerOptions,
)
from mlprep.phases.phase5.output_governance.readiness import (
    Phase5ReadinessOptions,
)
from mlprep.phases.phase5.output_governance.report import (
    Phase5ReportOptions,
)
from mlprep.phases.phase5.validation.distribution_impact import (
    DistributionImpactOptions,
)
from mlprep.phases.phase5.validation.remaining_outliers import (
    RemainingOutliersOptions,
)
from mlprep.phases.phase5.validation.row_integrity import (
    RowIntegrityOptions,
)
from mlprep.phases.phase5.validation.schema_consistency import (
    SchemaConsistencyOptions,
)
from mlprep.phases.phase5.validation.target_integrity import (
    TargetIntegrityOptions,
)


@dataclass(slots=True)
class Phase5ExecutionOptions:
    """General execution options for Phase 5."""

    enabled: bool = True

    stop_on_contract_failure: bool = True
    stop_on_plan_failure: bool = True
    stop_on_component_exception: bool = True

    require_authorized_test_partition: bool = True
    keep_test_partition_protected: bool = True

    store_phase_state_in_result: bool = True
    store_execution_metadata: bool = True

    generate_validation_findings: bool = True

    def __post_init__(self) -> None:
        """Validate execution options."""

        boolean_options = {
            "enabled": self.enabled,
            "stop_on_contract_failure": (
                self.stop_on_contract_failure
            ),
            "stop_on_plan_failure": (
                self.stop_on_plan_failure
            ),
            "stop_on_component_exception": (
                self.stop_on_component_exception
            ),
            "require_authorized_test_partition": (
                self.require_authorized_test_partition
            ),
            "keep_test_partition_protected": (
                self.keep_test_partition_protected
            ),
            "store_phase_state_in_result": (
                self.store_phase_state_in_result
            ),
            "store_execution_metadata": (
                self.store_execution_metadata
            ),
            "generate_validation_findings": (
                self.generate_validation_findings
            ),
        }

        for name, value in boolean_options.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a boolean."
                )


@dataclass(slots=True)
class Phase5ValidationOptions:
    """Complete Phase 5 validation configuration."""

    remaining_outliers: RemainingOutliersOptions = field(
        default_factory=RemainingOutliersOptions
    )

    schema_consistency: SchemaConsistencyOptions = field(
        default_factory=SchemaConsistencyOptions
    )

    target_integrity: TargetIntegrityOptions = field(
        default_factory=TargetIntegrityOptions
    )

    row_integrity: RowIntegrityOptions = field(
        default_factory=RowIntegrityOptions
    )

    distribution_impact: DistributionImpactOptions = field(
        default_factory=DistributionImpactOptions
    )


@dataclass(slots=True)
class Phase5Config:
    """Complete configuration for Phase 5."""

    execution: Phase5ExecutionOptions = field(
        default_factory=Phase5ExecutionOptions
    )

    contract: OutlierContractOptions = field(
        default_factory=OutlierContractOptions
    )

    plan: TreatmentPlanOptions = field(
        default_factory=TreatmentPlanOptions
    )

    detector_engine: DetectorEngineOptions = field(
        default_factory=DetectorEngineOptions
    )

    partition_transformer: PartitionTransformerOptions = field(
        default_factory=PartitionTransformerOptions
    )

    pipeline: OutlierPipelineOptions = field(
        default_factory=OutlierPipelineOptions
    )

    validation: Phase5ValidationOptions = field(
        default_factory=Phase5ValidationOptions
    )

    readiness: Phase5ReadinessOptions = field(
        default_factory=Phase5ReadinessOptions
    )

    reporting: Phase5ReportOptions = field(
        default_factory=Phase5ReportOptions
    )

    def __post_init__(self) -> None:
        """Validate and synchronize Phase 5 configuration."""

        self._synchronize_pipeline_options()
        self._synchronize_test_governance()
        self._synchronize_remaining_outlier_policy()
        self._validate_indicator_policy()
        self._validate_set_missing_policy()

    @classmethod
    def create(
        cls,
        *,
        add_outlier_indicators: bool = True,
        allow_report_only: bool = True,
        allow_set_missing: bool = False,
        allow_remaining_outliers: bool = False,
        maximum_changed_percentage: float = 20.0,
        stop_on_component_exception: bool = True,
        expose_fitted_bounds: bool = True,
    ) -> Phase5Config:
        """Create a common Phase 5 configuration."""

        if not (
            0.0
            <= maximum_changed_percentage
            <= 100.0
        ):
            raise ValueError(
                "maximum_changed_percentage must be "
                "between zero and one hundred."
            )

        execution = Phase5ExecutionOptions(
            stop_on_component_exception=(
                stop_on_component_exception
            )
        )

        plan = TreatmentPlanOptions(
            add_indicators_by_default=(
                add_outlier_indicators
            ),
            allow_report_only=allow_report_only,
            allow_set_missing=allow_set_missing,
        )

        detector_engine = DetectorEngineOptions()

        partition_transformer = (
            PartitionTransformerOptions(
                add_outlier_indicators=(
                    add_outlier_indicators
                ),
                allow_new_missing_values=False,
                allow_nonfinite_values=False,
            )
        )

        pipeline = OutlierPipelineOptions(
            detector_engine=detector_engine,
            partition_transformer=(
                partition_transformer
            ),
            test_transform_authorized=True,
        )

        validation = Phase5ValidationOptions()

        validation.remaining_outliers.allow_report_only_outliers = (
            allow_report_only
        )

        validation.remaining_outliers.maximum_total_remaining_count = (
            None
            if allow_remaining_outliers
            else 0
        )

        validation.distribution_impact.maximum_changed_percentage = (
            maximum_changed_percentage
        )

        reporting = Phase5ReportOptions(
            expose_fitted_bounds=(
                expose_fitted_bounds
            )
        )

        return cls(
            execution=execution,
            plan=plan,
            detector_engine=detector_engine,
            partition_transformer=(
                partition_transformer
            ),
            pipeline=pipeline,
            validation=validation,
            reporting=reporting,
        )

    def _synchronize_pipeline_options(
        self,
    ) -> None:
        """Synchronize pipeline child options."""

        self.pipeline.detector_engine = (
            self.detector_engine
        )

        self.pipeline.partition_transformer = (
            self.partition_transformer
        )

    def _synchronize_test_governance(
        self,
    ) -> None:
        """Synchronize test-partition governance."""

        self.pipeline.test_transform_authorized = (
            self.execution
            .require_authorized_test_partition
        )

        self.readiness.require_authorized_test_transform = (
            self.execution
            .require_authorized_test_partition
        )

        self.readiness.require_protected_test_partition = (
            self.execution
            .keep_test_partition_protected
        )

    def _synchronize_remaining_outlier_policy(
        self,
    ) -> None:
        """Synchronize report-only outlier policy."""

        self.validation.remaining_outliers.allow_report_only_outliers = (
            self.plan.allow_report_only
        )

    def _validate_indicator_policy(
        self,
    ) -> None:
        """Validate outlier-indicator policy consistency."""

        if (
            self.plan.add_indicators_by_default
            and not self.partition_transformer
            .add_outlier_indicators
        ):
            raise ValueError(
                "The treatment plan requests outlier "
                "indicators, but indicator creation is "
                "disabled in the partition transformer."
            )

        if (
            not self.partition_transformer
            .add_outlier_indicators
            and any(
                self.plan
                .indicator_overrides
                .values()
            )
        ):
            raise ValueError(
                "Indicator overrides cannot enable indicators "
                "when partition-level indicator generation "
                "is disabled."
            )

    def _validate_set_missing_policy(
        self,
    ) -> None:
        """Validate set-missing treatment policy."""

        set_missing_columns = [
            column
            for column, treatment
            in self.plan
            .treatment_overrides
            .items()
            if treatment == TreatmentKind.SET_MISSING
        ]

        if (
            set_missing_columns
            and not self.plan.allow_set_missing
        ):
            raise ValueError(
                "The configuration contains set_missing "
                "treatment overrides, but set_missing "
                "treatment is disabled."
            )

        missing_fallback_columns = [
            column
            for column
            in set_missing_columns
            if column
            not in self.plan
            .set_missing_fallbacks
        ]

        if missing_fallback_columns:
            raise ValueError(
                "Every set_missing treatment requires an "
                "explicit finite fallback. Missing fallbacks "
                f"for: {missing_fallback_columns}"
            )

        if (
            self.plan.allow_set_missing
            and self.partition_transformer
            .allow_new_missing_values
        ):
            raise ValueError(
                "Phase 5 cannot permit unresolved missing "
                "values when set_missing treatment is enabled. "
                "Use explicit fallback values instead."
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert configuration to a serializable dictionary."""

        return self._serialize(
            asdict(self)
        )

    @classmethod
    def _serialize(
        cls,
        value: Any,
    ) -> Any:
        """Recursively serialize configuration values."""

        if isinstance(value, dict):
            return {
                str(key): cls._serialize(
                    item
                )
                for key, item
                in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                cls._serialize(item)
                for item in value
            ]

        if hasattr(value, "model_dump"):
            return cls._serialize(
                value.model_dump(
                    mode="json"
                )
            )

        if hasattr(value, "value"):
            return value.value

        return value