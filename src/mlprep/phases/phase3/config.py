"""Unified configuration for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlprep.phases.phase3.contract.column_selection import (
    ColumnSelectionOptions,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContractOptions,
)
from mlprep.phases.phase3.diagnostics.data_quality import (
    DataQualityDiagnosticOptions,
)
from mlprep.phases.phase3.diagnostics.distribution_shift import (
    DistributionShiftOptions,
)
from mlprep.phases.phase3.diagnostics.high_cardinality import (
    HighCardinalityDiagnosticOptions,
)
from mlprep.phases.phase3.diagnostics.rare_categories import (
    RareCategoryDiagnosticOptions,
)
from mlprep.phases.phase3.diagnostics.skewness import (
    SkewnessDiagnosticOptions,
)
from mlprep.phases.phase3.diagnostics.temporal_patterns import (
    TemporalPatternOptions,
)
from mlprep.phases.phase3.output_governance.readiness import (
    Phase3ReadinessOptions,
)
from mlprep.phases.phase3.output_governance.report import (
    Phase3ReportOptions,
)
from mlprep.phases.phase3.profiling.categorical_profile import (
    CategoricalProfilerOptions,
)
from mlprep.phases.phase3.profiling.dataset_profile import (
    DatasetProfilerOptions,
)
from mlprep.phases.phase3.profiling.datetime_profile import (
    DatetimeProfilerOptions,
)
from mlprep.phases.phase3.profiling.missingness_profile import (
    MissingnessProfilerOptions,
)
from mlprep.phases.phase3.profiling.numeric_profile import (
    NumericProfilerOptions,
)
from mlprep.phases.phase3.profiling.target_profile import (
    TargetProfilerOptions,
)
from mlprep.phases.phase3.profiling.text_profile import (
    TextProfilerOptions,
)
from mlprep.phases.phase3.recommendations.recommendation_engine import (
    RecommendationEngineOptions,
)
from mlprep.phases.phase3.relationships.categorical_associations import (
    CategoricalAssociationOptions,
)
from mlprep.phases.phase3.relationships.interaction_candidates import (
    InteractionCandidateOptions,
)
from mlprep.phases.phase3.relationships.missingness_associations import (
    MissingnessAssociationOptions,
)
from mlprep.phases.phase3.relationships.mixed_associations import (
    MixedAssociationOptions,
)
from mlprep.phases.phase3.relationships.multicollinearity import (
    MulticollinearityOptions,
)
from mlprep.phases.phase3.relationships.numeric_associations import (
    NumericAssociationOptions,
)
from mlprep.phases.phase3.relationships.target_associations import (
    TargetAssociationOptions,
)
from mlprep.phases.phase3.visualization.plot_manager import (
    PlotManagerOptions,
)


@dataclass(slots=True)
class Phase3ExecutionOptions:
    """General execution options for Phase 3."""

    enabled: bool = True

    stop_on_contract_failure: bool = True
    stop_on_component_exception: bool = False

    generate_findings_from_diagnostics: bool = True

    store_phase_state_in_result: bool = True

    copy_training_data: bool = True
    copy_validation_data: bool = True

    require_test_lock: bool = True

    def __post_init__(self) -> None:
        """Validate Phase 3 execution options."""

        boolean_options = {
            "enabled": self.enabled,
            "stop_on_contract_failure": (
                self.stop_on_contract_failure
            ),
            "stop_on_component_exception": (
                self.stop_on_component_exception
            ),
            "generate_findings_from_diagnostics": (
                self.generate_findings_from_diagnostics
            ),
            "store_phase_state_in_result": (
                self.store_phase_state_in_result
            ),
            "copy_training_data": (
                self.copy_training_data
            ),
            "copy_validation_data": (
                self.copy_validation_data
            ),
            "require_test_lock": (
                self.require_test_lock
            ),
        }

        for name, value in boolean_options.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a boolean."
                )


@dataclass(slots=True)
class Phase3ProfilingOptions:
    """Complete Phase 3 profiling configuration."""

    dataset: DatasetProfilerOptions = field(
        default_factory=DatasetProfilerOptions
    )

    numeric: NumericProfilerOptions = field(
        default_factory=NumericProfilerOptions
    )

    categorical: CategoricalProfilerOptions = field(
        default_factory=CategoricalProfilerOptions
    )

    datetime: DatetimeProfilerOptions = field(
        default_factory=DatetimeProfilerOptions
    )

    text: TextProfilerOptions = field(
        default_factory=TextProfilerOptions
    )

    missingness: MissingnessProfilerOptions = field(
        default_factory=MissingnessProfilerOptions
    )

    target: TargetProfilerOptions = field(
        default_factory=TargetProfilerOptions
    )


@dataclass(slots=True)
class Phase3RelationshipOptions:
    """Complete relationship-analysis configuration."""

    numeric: NumericAssociationOptions = field(
        default_factory=NumericAssociationOptions
    )

    categorical: CategoricalAssociationOptions = field(
        default_factory=CategoricalAssociationOptions
    )

    mixed: MixedAssociationOptions = field(
        default_factory=MixedAssociationOptions
    )

    target: TargetAssociationOptions = field(
        default_factory=TargetAssociationOptions
    )

    missingness: MissingnessAssociationOptions = field(
        default_factory=MissingnessAssociationOptions
    )

    multicollinearity: MulticollinearityOptions = field(
        default_factory=MulticollinearityOptions
    )

    interactions: InteractionCandidateOptions = field(
        default_factory=InteractionCandidateOptions
    )


@dataclass(slots=True)
class Phase3DiagnosticOptions:
    """Complete analytical-diagnostic configuration."""

    skewness: SkewnessDiagnosticOptions = field(
        default_factory=SkewnessDiagnosticOptions
    )

    rare_categories: RareCategoryDiagnosticOptions = field(
        default_factory=RareCategoryDiagnosticOptions
    )

    high_cardinality: HighCardinalityDiagnosticOptions = field(
        default_factory=HighCardinalityDiagnosticOptions
    )

    distribution_shift: DistributionShiftOptions = field(
        default_factory=DistributionShiftOptions
    )

    temporal_patterns: TemporalPatternOptions = field(
        default_factory=TemporalPatternOptions
    )

    data_quality: DataQualityDiagnosticOptions = field(
        default_factory=DataQualityDiagnosticOptions
    )


@dataclass(slots=True)
class Phase3Config:
    """Complete configuration for Phase 3."""

    execution: Phase3ExecutionOptions = field(
        default_factory=Phase3ExecutionOptions
    )

    contract: EDAContractOptions = field(
        default_factory=EDAContractOptions
    )

    column_selection: ColumnSelectionOptions = field(
        default_factory=ColumnSelectionOptions
    )

    profiling: Phase3ProfilingOptions = field(
        default_factory=Phase3ProfilingOptions
    )

    relationships: Phase3RelationshipOptions = field(
        default_factory=Phase3RelationshipOptions
    )

    diagnostics: Phase3DiagnosticOptions = field(
        default_factory=Phase3DiagnosticOptions
    )

    visualizations: PlotManagerOptions = field(
        default_factory=PlotManagerOptions
    )

    recommendations: RecommendationEngineOptions = field(
        default_factory=RecommendationEngineOptions
    )

    readiness: Phase3ReadinessOptions = field(
        default_factory=Phase3ReadinessOptions
    )

    reporting: Phase3ReportOptions = field(
        default_factory=Phase3ReportOptions
    )

    def __post_init__(self) -> None:
        """Validate and synchronize Phase 3 configuration."""

        self._synchronize_contract_options()
        self._synchronize_validation_comparison()
        self._validate_test_protection()
        self._validate_analysis_limits()

    @classmethod
    def create(
        cls,
        *,
        include_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        include_sensitive_columns: bool = False,
        include_identifier_columns: bool = False,
        enable_validation_comparison: bool = True,
        generate_plots: bool = True,
        plot_output_directory: str = (
            "artifacts/phase3/plots"
        ),
        stop_on_component_exception: bool = False,
    ) -> Phase3Config:
        """Create a common Phase 3 configuration."""

        execution = Phase3ExecutionOptions(
            stop_on_component_exception=(
                stop_on_component_exception
            )
        )

        contract = EDAContractOptions(
            allow_validation_analysis=(
                enable_validation_comparison
            ),
            allow_test_analysis=False,
            require_locked_test_partition=True,
        )

        column_selection = ColumnSelectionOptions(
            include_columns=(
                list(include_columns)
                if include_columns is not None
                else None
            ),
            exclude_columns=list(
                exclude_columns or []
            ),
            include_sensitive_columns=(
                include_sensitive_columns
            ),
            include_identifier_columns=(
                include_identifier_columns
            ),
        )

        profiling = Phase3ProfilingOptions()

        profiling.missingness.compare_validation = (
            enable_validation_comparison
        )

        profiling.target.include_validation_comparison = (
            enable_validation_comparison
        )

        diagnostics = Phase3DiagnosticOptions()

        diagnostics.rare_categories.compare_validation = (
            enable_validation_comparison
        )

        diagnostics.distribution_shift.enabled = (
            enable_validation_comparison
        )

        visualizations = PlotManagerOptions(
            enabled=generate_plots,
            output_directory=(
                plot_output_directory
            ),
        )

        return cls(
            execution=execution,
            contract=contract,
            column_selection=column_selection,
            profiling=profiling,
            diagnostics=diagnostics,
            visualizations=visualizations,
        )

    def _synchronize_contract_options(
        self,
    ) -> None:
        """Synchronize execution and contract settings."""

        self.contract.copy_training_data = (
            self.execution.copy_training_data
        )

        self.contract.copy_validation_data = (
            self.execution.copy_validation_data
        )

        self.contract.require_locked_test_partition = (
            self.execution.require_test_lock
        )

        self.contract.allow_test_analysis = False

    def _synchronize_validation_comparison(
        self,
    ) -> None:
        """Synchronize validation-analysis settings."""

        validation_analysis_enabled = (
            self.contract.allow_validation_analysis
        )

        self.profiling.missingness.compare_validation = (
            validation_analysis_enabled
            and self.profiling
            .missingness
            .compare_validation
        )

        self.profiling.target.include_validation_comparison = (
            validation_analysis_enabled
            and self.profiling
            .target
            .include_validation_comparison
        )

        self.diagnostics.rare_categories.compare_validation = (
            validation_analysis_enabled
            and self.diagnostics
            .rare_categories
            .compare_validation
        )

        self.diagnostics.distribution_shift.enabled = (
            validation_analysis_enabled
            and self.diagnostics
            .distribution_shift
            .enabled
        )

    def _validate_test_protection(
        self,
    ) -> None:
        """Ensure Phase 3 cannot analyze the test partition."""

        if self.contract.allow_test_analysis:
            raise ValueError(
                "Phase 3 configuration cannot enable "
                "test-set analysis."
            )

        if (
            self.readiness.require_locked_test_partition
            and not self.execution.require_test_lock
        ):
            raise ValueError(
                "Readiness requires a locked test partition, "
                "but execution does not require the test lock."
            )

    def _validate_analysis_limits(
        self,
    ) -> None:
        """Validate cross-component analysis limits."""

        if (
            self.visualizations.maximum_total_plots < 1
            and self.visualizations.enabled
        ):
            raise ValueError(
                "Visualizations require a positive plot limit."
            )

        if (
            self.relationships
            .multicollinearity
            .maximum_columns
            > self.relationships
            .numeric
            .maximum_columns
        ):
            self.relationships.multicollinearity.maximum_columns = (
                self.relationships.numeric.maximum_columns
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the complete configuration to a dictionary."""

        return self._serialize_value(
            self
        )

    @classmethod
    def _serialize_value(
        cls,
        value: Any,
    ) -> Any:
        """Recursively serialize configuration values."""

        if hasattr(
            value,
            "__dataclass_fields__",
        ):
            return {
                name: cls._serialize_value(
                    getattr(value, name)
                )
                for name in value.__dataclass_fields__
            }

        if hasattr(value, "model_dump"):
            return cls._serialize_value(
                value.model_dump(
                    mode="json"
                )
            )

        if isinstance(value, dict):
            return {
                str(key): cls._serialize_value(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                cls._serialize_value(item)
                for item in value
            ]

        if hasattr(value, "value"):
            return value.value

        return value