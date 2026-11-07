"""Main orchestration pipeline for Phase 3."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.phases.phase2.output_governance.result import (
    LockedTestPartition,
    Phase2Result,
)
from mlprep.phases.phase2.output_governance.state import (
    Phase2State,
)
from mlprep.phases.phase3.config import Phase3Config
from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
    ColumnSelector,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
    EDAContractIssue,
    EDAContractResolver,
)
from mlprep.phases.phase3.diagnostics.data_quality import (
    DataQualityDiagnosticAnalyzer,
    DataQualityDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.distribution_shift import (
    DistributionShiftAnalyzer,
    DistributionShiftResult,
)
from mlprep.phases.phase3.diagnostics.high_cardinality import (
    HighCardinalityDiagnosticAnalyzer,
    HighCardinalityDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.rare_categories import (
    RareCategoryDiagnosticAnalyzer,
    RareCategoryDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.skewness import (
    SkewnessDiagnosticAnalyzer,
    SkewnessDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.temporal_patterns import (
    TemporalPatternAnalyzer,
    TemporalPatternResult,
)
from mlprep.phases.phase3.output_governance.fingerprint import (
    Phase3FingerprintManager,
)
from mlprep.phases.phase3.output_governance.readiness import (
    Phase3ReadinessEvaluator,
)
from mlprep.phases.phase3.output_governance.report import (
    Phase3ReportBuilder,
)
from mlprep.phases.phase3.output_governance.result import (
    Phase3Finding,
    Phase3FindingCategory,
    Phase3FindingSeverity,
    Phase3ReadinessDecision,
    Phase3Result,
    Phase3Status,
)
from mlprep.phases.phase3.output_governance.state import (
    Phase3State,
)
from mlprep.phases.phase3.profiling.categorical_profile import (
    CategoricalProfiler,
)
from mlprep.phases.phase3.profiling.dataset_profile import (
    DatasetProfiler,
)
from mlprep.phases.phase3.profiling.datetime_profile import (
    DatetimeProfiler,
)
from mlprep.phases.phase3.profiling.missingness_profile import (
    MissingnessProfiler,
)
from mlprep.phases.phase3.profiling.numeric_profile import (
    NumericProfiler,
)
from mlprep.phases.phase3.profiling.target_profile import (
    TargetProfiler,
)
from mlprep.phases.phase3.profiling.text_profile import (
    TextProfiler,
)
from mlprep.phases.phase3.recommendations.recommendation_engine import (
    RecommendationEngine,
)
from mlprep.phases.phase3.relationships.categorical_associations import (
    CategoricalAssociationAnalyzer,
    CategoricalAssociationResult,
)
from mlprep.phases.phase3.relationships.interaction_candidates import (
    InteractionCandidateAnalyzer,
    InteractionCandidateResult,
)
from mlprep.phases.phase3.relationships.missingness_associations import (
    MissingnessAssociationAnalyzer,
    MissingnessAssociationResult,
)
from mlprep.phases.phase3.relationships.mixed_associations import (
    MixedAssociationAnalyzer,
    MixedAssociationResult,
)
from mlprep.phases.phase3.relationships.multicollinearity import (
    MulticollinearityAnalyzer,
    MulticollinearityResult,
)
from mlprep.phases.phase3.relationships.numeric_associations import (
    NumericAssociationAnalyzer,
    NumericAssociationResult,
)
from mlprep.phases.phase3.relationships.target_associations import (
    TargetAssociationAnalyzer,
    TargetAssociationResult,
)
from mlprep.phases.phase3.visualization.plot_manager import (
    PlotManager,
)
from mlprep.phases.phase3.visualization.plot_spec import (
    PlotCollection,
)

ComponentResult = TypeVar("ComponentResult")


class Phase3:
    """Execute training-only exploratory data analysis."""

    phase_name = "phase_3"

    def __init__(
        self,
        config: Phase3Config | None = None,
    ) -> None:
        """Initialize Phase 3."""

        self.config = config or Phase3Config()

        if not isinstance(
            self.config,
            Phase3Config,
        ):
            raise TypeError(
                "config must be an instance of Phase3Config."
            )

        self.result: Phase3Result | None = None
        self.state: Phase3State | None = None
        self.contract: EDAContract | None = None

        self.fingerprint_manager = (
            Phase3FingerprintManager()
        )

        self.report_builder = Phase3ReportBuilder(
            options=self.config.reporting
        )

    def run(
        self,
        *,
        phase2_result: Phase2Result,
        phase1_state: Phase1State | None = None,
        phase2_state: Phase2State | None = None,
    ) -> Phase3Result:
        """Execute the complete Phase 3 workflow."""

        self._validate_phase2_result(
            phase2_result
        )

        result = Phase3Result()

        self.result = result

        if not self.config.execution.enabled:
            return self._build_disabled_result(
                phase2_result=phase2_result
            )

        resolved_phase2_state = (
            phase2_state
            or phase2_result.state
        )

        contract = self._resolve_contract(
            phase2_result=phase2_result,
            phase1_state=phase1_state,
            phase2_state=resolved_phase2_state,
        )

        self.contract = contract

        columns = self._resolve_columns(
            contract=contract
        )

        state = Phase3State(
            analyzed_training_only=True,
            validation_comparison_enabled=(
                self.config
                .contract
                .allow_validation_analysis
            ),
            test_partition_access_count_before=(
                resolved_phase2_state
                .test_lock
                .access_count
            ),
            test_partition_access_count_after=(
                resolved_phase2_state
                .test_lock
                .access_count
            ),
            test_partition_remained_locked=(
                resolved_phase2_state
                .test_lock
                .locked
            ),
            columns=columns,
            phase2_split_fingerprint=(
                resolved_phase2_state
                .split_fingerprint
            ),
        )

        self.state = state

        result.add_findings(
            self._contract_findings(
                contract.issues
            )
        )

        if (
            self.config.execution
            .stop_on_contract_failure
            and contract.has_blocking_issues
        ):
            return self._finalize_failed_contract(
                result=result,
                state=state,
            )

        contract.assert_ready()

        self.fingerprint_manager.capture_before(
            contract=contract,
            state=state,
        )

        state.configuration_fingerprint = (
            self.fingerprint_manager
            .configuration_fingerprint(
                self.config
            )
        )

        self._run_profiling(
            contract=contract,
            columns=columns,
            result=result,
            state=state,
        )

        self._run_relationships(
            contract=contract,
            columns=columns,
            result=result,
            state=state,
        )

        self._run_diagnostics(
            contract=contract,
            columns=columns,
            result=result,
            state=state,
        )

        self._run_recommendations(
            columns=columns,
            result=result,
            state=state,
        )

        self._run_visualizations(
            contract=contract,
            columns=columns,
            result=result,
            state=state,
        )

        result.add_findings(
            self.fingerprint_manager.verify_after(
                contract=contract,
                state=state,
            )
        )

        state.test_partition_access_count_after = (
            resolved_phase2_state
            .test_lock
            .access_count
        )

        state.test_partition_remained_locked = bool(
            resolved_phase2_state
            .test_lock
            .locked
            and isinstance(
                phase2_result.test,
                LockedTestPartition,
            )
        )

        state.analysis_fingerprint = (
            self.fingerprint_manager
            .analysis_fingerprint(
                state=state
            )
        )

        self._generate_diagnostic_findings(
            result=result
        )

        evaluator = Phase3ReadinessEvaluator(
            options=self.config.readiness
        )

        evaluator.evaluate(
            result=result,
            state=state,
        )

        if (
            self.config.execution
            .store_phase_state_in_result
        ):
            result.phase_state = (
                state.model_dump(
                    mode="json"
                )
            )

        return result

    def _run_profiling(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
        result: Phase3Result,
        state: Phase3State,
    ) -> None:
        """Run all Phase 3 profiling components."""

        result.dataset_profile = self._execute(
            component="dataset_profile",
            operation=lambda: DatasetProfiler(
                self.config.profiling.dataset
            ).profile(
                contract=contract,
                columns=columns,
            ),
            fallback=None,
            state=state,
        )

        result.numeric_profiles = self._execute(
            component="numeric_profile",
            operation=lambda: NumericProfiler(
                self.config.profiling.numeric
            ).profile(
                contract=contract,
                columns=columns,
            ),
            fallback={},
            state=state,
        )

        result.categorical_profiles = self._execute(
            component="categorical_profile",
            operation=lambda: CategoricalProfiler(
                self.config.profiling.categorical
            ).profile(
                contract=contract,
                columns=columns,
            ),
            fallback={},
            state=state,
        )

        result.datetime_profiles = self._execute(
            component="datetime_profile",
            operation=lambda: DatetimeProfiler(
                self.config.profiling.datetime
            ).profile(
                contract=contract,
                columns=columns,
            ),
            fallback={},
            state=state,
        )

        result.text_profiles = self._execute(
            component="text_profile",
            operation=lambda: TextProfiler(
                self.config.profiling.text
            ).profile(
                contract=contract,
                columns=columns,
            ),
            fallback={},
            state=state,
        )

        result.missingness_profile = self._execute(
            component="missingness_profile",
            operation=lambda: MissingnessProfiler(
                self.config.profiling.missingness
            ).profile(
                contract=contract
            ),
            fallback=None,
            state=state,
        )

        result.target_profile = self._execute(
            component="target_profile",
            operation=lambda: TargetProfiler(
                self.config.profiling.target
            ).profile(
                contract=contract
            ),
            fallback=None,
            state=state,
        )

    def _run_relationships(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
        result: Phase3Result,
        state: Phase3State,
    ) -> None:
        """Run all feature-relationship analyses."""

        result.numeric_associations = self._execute(
            component="numeric_associations",
            operation=lambda: NumericAssociationAnalyzer(
                self.config.relationships.numeric
            ).analyze(
                contract=contract,
                columns=columns,
            ),
            fallback=NumericAssociationResult(
                primary_method=(
                    self.config
                    .relationships
                    .numeric
                    .primary_method
                )
            ),
            state=state,
        )

        result.categorical_associations = self._execute(
            component="categorical_associations",
            operation=lambda: (
                CategoricalAssociationAnalyzer(
                    self.config
                    .relationships
                    .categorical
                ).analyze(
                    contract=contract,
                    columns=columns,
                )
            ),
            fallback=CategoricalAssociationResult(),
            state=state,
        )

        result.mixed_associations = self._execute(
            component="mixed_associations",
            operation=lambda: MixedAssociationAnalyzer(
                self.config.relationships.mixed
            ).analyze(
                contract=contract,
                columns=columns,
            ),
            fallback=MixedAssociationResult(),
            state=state,
        )

        result.target_associations = self._execute(
            component="target_associations",
            operation=lambda: TargetAssociationAnalyzer(
                self.config.relationships.target
            ).analyze(
                contract=contract,
                columns=columns,
            ),
            fallback=TargetAssociationResult(
                target_column=contract.target_column,
                task_type="unknown",
            ),
            state=state,
        )

        result.missingness_associations = (
            self._execute(
                component=(
                    "missingness_associations"
                ),
                operation=lambda: (
                    MissingnessAssociationAnalyzer(
                        self.config
                        .relationships
                        .missingness
                    ).analyze(
                        contract=contract
                    )
                ),
                fallback=(
                    MissingnessAssociationResult()
                ),
                state=state,
            )
        )

        result.multicollinearity = self._execute(
            component="multicollinearity",
            operation=lambda: MulticollinearityAnalyzer(
                self.config
                .relationships
                .multicollinearity
            ).analyze(
                contract=contract,
                columns=columns,
            ),
            fallback=MulticollinearityResult(),
            state=state,
        )

        numeric_associations = (
            result.numeric_associations
            or NumericAssociationResult(
                primary_method=(
                    self.config
                    .relationships
                    .numeric
                    .primary_method
                )
            )
        )

        mixed_associations = (
            result.mixed_associations
            or MixedAssociationResult()
        )

        target_associations = (
            result.target_associations
            or TargetAssociationResult(
                target_column=contract.target_column,
                task_type="unknown",
            )
        )

        result.interaction_candidates = self._execute(
            component="interaction_candidates",
            operation=lambda: InteractionCandidateAnalyzer(
                self.config
                .relationships
                .interactions
            ).analyze(
                columns=columns,
                numeric_associations=(
                    numeric_associations
                ),
                mixed_associations=(
                    mixed_associations
                ),
                target_associations=(
                    target_associations
                ),
            ),
            fallback=InteractionCandidateResult(),
            state=state,
        )

    def _run_diagnostics(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
        result: Phase3Result,
        state: Phase3State,
    ) -> None:
        """Run all Phase 3 diagnostics."""

        result.skewness_diagnostics = self._execute(
            component="skewness",
            operation=lambda: SkewnessDiagnosticAnalyzer(
                self.config.diagnostics.skewness
            ).analyze(
                contract=contract,
                columns=columns,
            ),
            fallback=SkewnessDiagnosticResult(),
            state=state,
        )

        result.rare_category_diagnostics = (
            self._execute(
                component="rare_categories",
                operation=lambda: (
                    RareCategoryDiagnosticAnalyzer(
                        self.config
                        .diagnostics
                        .rare_categories
                    ).analyze(
                        contract=contract,
                        columns=columns,
                    )
                ),
                fallback=(
                    RareCategoryDiagnosticResult()
                ),
                state=state,
            )
        )

        result.cardinality_diagnostics = self._execute(
            component="high_cardinality",
            operation=lambda: (
                HighCardinalityDiagnosticAnalyzer(
                    self.config
                    .diagnostics
                    .high_cardinality
                ).analyze(
                    contract=contract,
                    columns=columns,
                )
            ),
            fallback=(
                HighCardinalityDiagnosticResult()
            ),
            state=state,
        )

        result.distribution_shift = self._execute(
            component="distribution_shift",
            operation=lambda: DistributionShiftAnalyzer(
                self.config
                .diagnostics
                .distribution_shift
            ).analyze(
                contract=contract,
                columns=columns,
            ),
            fallback=DistributionShiftResult(),
            state=state,
        )

        result.temporal_patterns = self._execute(
            component="temporal_patterns",
            operation=lambda: TemporalPatternAnalyzer(
                self.config
                .diagnostics
                .temporal_patterns
            ).analyze(
                contract=contract,
                columns=columns,
            ),
            fallback=TemporalPatternResult(),
            state=state,
        )

        result.data_quality = self._execute(
            component="data_quality",
            operation=lambda: DataQualityDiagnosticAnalyzer(
                self.config
                .diagnostics
                .data_quality
            ).analyze(
                contract=contract,
                columns=columns,
            ),
            fallback=DataQualityDiagnosticResult(),
            state=state,
        )

    def _run_recommendations(
        self,
        *,
        columns: AnalysisColumnGroups,
        result: Phase3Result,
        state: Phase3State,
    ) -> None:
        """Build downstream recommendations."""

        recommendations = self._execute(
            component="recommendation_engine",
            operation=lambda: RecommendationEngine(
                self.config.recommendations
            ).build(
                columns=columns,
                numeric_profiles=(
                    result.numeric_profiles
                ),
                missingness=self._require_value(
                    result.missingness_profile,
                    "missingness_profile",
                ),
                skewness=self._require_value(
                    result.skewness_diagnostics,
                    "skewness_diagnostics",
                ),
                rare_categories=self._require_value(
                    result
                    .rare_category_diagnostics,
                    "rare_category_diagnostics",
                ),
                cardinality=self._require_value(
                    result
                    .cardinality_diagnostics,
                    "cardinality_diagnostics",
                ),
                missingness_associations=(
                    self._require_value(
                        result
                        .missingness_associations,
                        "missingness_associations",
                    )
                ),
                interaction_candidates=(
                    self._require_value(
                        result
                        .interaction_candidates,
                        "interaction_candidates",
                    )
                ),
            ),
            fallback=None,
            state=state,
        )

        result.recommendations = recommendations

        if recommendations is not None:
            state.recommendation_count = sum(
                [
                    len(
                        recommendations
                        .transformations
                        .recommendations
                    ),
                    len(
                        recommendations
                        .imputation
                        .recommendations
                    ),
                    len(
                        recommendations
                        .outliers
                        .recommendations
                    ),
                    len(
                        recommendations
                        .encoding
                        .recommendations
                    ),
                    len(
                        recommendations
                        .interactions
                        .recommendations
                    ),
                ]
            )

    def _run_visualizations(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
        result: Phase3Result,
        state: Phase3State,
    ) -> None:
        """Generate Phase 3 visualizations."""

        if not self.config.visualizations.enabled:
            state.mark_skipped(
                "plot_manager"
            )

            result.plots = PlotCollection(
                metadata={
                    "enabled": False,
                }
            )

            return

        numeric_associations = self._require_value(
            result.numeric_associations,
            "numeric_associations",
        )

        target_associations = self._require_value(
            result.target_associations,
            "target_associations",
        )

        result.plots = self._execute(
            component="plot_manager",
            operation=lambda: PlotManager(
                self.config.visualizations
            ).generate(
                contract=contract,
                columns=columns,
                numeric_associations=(
                    numeric_associations
                ),
                target_associations=(
                    target_associations
                ),
            ),
            fallback=PlotCollection(),
            state=state,
        )

        state.generated_plot_count = (
            result.plots.generated_count
        )

    def _resolve_contract(
        self,
        *,
        phase2_result: Phase2Result,
        phase1_state: Phase1State | None,
        phase2_state: Phase2State,
    ) -> EDAContract:
        """Resolve the Phase 3 input contract."""

        resolver = EDAContractResolver(
            options=self.config.contract
        )

        return resolver.resolve(
            phase2_result=phase2_result,
            phase1_state=phase1_state,
            phase2_state=phase2_state,
        )

    def _resolve_columns(
        self,
        *,
        contract: EDAContract,
    ) -> AnalysisColumnGroups:
        """Resolve Phase 3 analysis columns."""

        selector = ColumnSelector(
            options=self.config.column_selection
        )

        return selector.resolve(
            contract=contract
        )

    def _execute(
        self,
        *,
        component: str,
        operation: Callable[
            [],
            ComponentResult,
        ],
        fallback: ComponentResult,
        state: Phase3State,
    ) -> ComponentResult:
        """Execute one Phase 3 component safely."""

        try:
            value = operation()

            state.mark_completed(
                component
            )

            return value

        except Exception as error:
            state.mark_failed(
                component
            )

            if self.result is not None:
                self.result.add_finding(
                    Phase3Finding(
                        code="P3-EXECUTION-001",
                        category=(
                            Phase3FindingCategory
                            .EXECUTION
                        ),
                        severity=(
                            Phase3FindingSeverity
                            .CRITICAL
                        ),
                        component=component,
                        message=(
                            f"Component {component!r} "
                            f"failed with "
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                        blocks_readiness=True,
                        details={
                            "exception_type": (
                                type(error).__name__
                            ),
                            "exception_message": (
                                str(error)
                            ),
                        },
                    )
                )

            if (
                self.config.execution
                .stop_on_component_exception
            ):
                raise

            return fallback

    def _generate_diagnostic_findings(
        self,
        *,
        result: Phase3Result,
    ) -> None:
        """Convert major diagnostics into Phase 3 findings."""

        if not (
            self.config.execution
            .generate_findings_from_diagnostics
        ):
            return

        if result.data_quality is not None:
            for issue in result.data_quality.issues:
                severity = {
                    "info": (
                        Phase3FindingSeverity.INFO
                    ),
                    "warning": (
                        Phase3FindingSeverity.WARNING
                    ),
                    "error": (
                        Phase3FindingSeverity.ERROR
                    ),
                }[issue.severity]

                result.add_finding(
                    Phase3Finding(
                        code=issue.code,
                        category=(
                            Phase3FindingCategory
                            .DIAGNOSTIC
                        ),
                        severity=severity,
                        component="data_quality",
                        message=issue.message,
                        columns=issue.columns,
                        blocks_readiness=False,
                        details=issue.details,
                    )
                )

        if result.distribution_shift is not None:
            for column in (
                result.distribution_shift
                .high_shift_columns
            ):
                result.add_finding(
                    Phase3Finding(
                        code="P3-SHIFT-001",
                        category=(
                            Phase3FindingCategory
                            .DIAGNOSTIC
                        ),
                        severity=(
                            Phase3FindingSeverity.WARNING
                        ),
                        component="distribution_shift",
                        message=(
                            f"Feature {column!r} shows high "
                            "training-validation distribution shift."
                        ),
                        columns=[column],
                        blocks_readiness=False,
                    )
                )

        if result.recommendations is not None:
            for conflict in (
                result.recommendations.conflicts
            ):
                result.add_finding(
                    Phase3Finding(
                        code="P3-RECOMMENDATION-001",
                        category=(
                            Phase3FindingCategory
                            .RECOMMENDATION
                        ),
                        severity=(
                            Phase3FindingSeverity.WARNING
                        ),
                        component="recommendation_engine",
                        message=conflict,
                        blocks_readiness=False,
                    )
                )

        for plot in result.plots.plots:
            if plot.status.value != "failed":
                continue

            result.add_finding(
                Phase3Finding(
                    code="P3-PLOT-001",
                    category=(
                        Phase3FindingCategory
                        .VISUALIZATION
                    ),
                    severity=(
                        Phase3FindingSeverity.WARNING
                    ),
                    component="plot_manager",
                    message=(
                        f"Plot {plot.plot_id!r} failed: "
                        f"{plot.error_message}"
                    ),
                    columns=plot.columns,
                    blocks_readiness=False,
                )
            )

    @staticmethod
    def _contract_findings(
        issues: list[EDAContractIssue],
    ) -> list[object]:
        """Convert contract issues into Phase 3 findings."""

        severities = {
            "info": Phase3FindingSeverity.INFO,
            "warning": Phase3FindingSeverity.WARNING,
            "error": Phase3FindingSeverity.ERROR,
            "critical": (
                Phase3FindingSeverity.CRITICAL
            ),
        }

        return [
            Phase3Finding(
                code=issue.code,
                category=(
                    Phase3FindingCategory.CONTRACT
                ),
                severity=severities[
                    issue.severity
                ],
                component="eda_contract",
                message=issue.message,
                columns=issue.columns,
                blocks_readiness=(
                    issue.blocks_execution
                ),
                details=issue.details,
            )
            for issue in issues
        ]

    def _finalize_failed_contract(
        self,
        *,
        result: Phase3Result,
        state: Phase3State,
    ) -> Phase3Result:
        """Finalize a result after contract failure."""

        blocking = (
            result.get_blocking_findings()
        )

        decision = Phase3ReadinessDecision(
            ready=False,
            status=Phase3Status.FAILED,
            blocking_finding_ids=[
                finding.finding_id
                for finding in blocking
            ],
            blocking_finding_codes=[
                finding.code
                for finding in blocking
            ],
            reasons=[
                finding.message
                for finding in blocking
            ],
        )

        state.ready_for_phase_4 = False

        result.finalize(
            decision
        )

        if (
            self.config.execution
            .store_phase_state_in_result
        ):
            result.phase_state = (
                state.model_dump(
                    mode="json"
                )
            )

        return result

    def _build_disabled_result(
        self,
        *,
        phase2_result: Phase2Result,
    ) -> Phase3Result:
        """Build a result when Phase 3 is disabled."""

        result = self.result or Phase3Result()

        state = Phase3State(
            analyzed_training_only=True,
            validation_comparison_enabled=False,
            test_partition_access_count_before=(
                phase2_result
                .state
                .test_lock
                .access_count
            ),
            test_partition_access_count_after=(
                phase2_result
                .state
                .test_lock
                .access_count
            ),
            test_partition_remained_locked=(
                phase2_result
                .state
                .test_lock
                .locked
            ),
        )

        self.state = state

        result.add_finding(
            Phase3Finding(
                code="P3-EXECUTION-002",
                category=(
                    Phase3FindingCategory.EXECUTION
                ),
                severity=(
                    Phase3FindingSeverity.INFO
                ),
                component=self.phase_name,
                message=(
                    "Phase 3 execution is disabled."
                ),
                blocks_readiness=False,
            )
        )

        result.finalize(
            Phase3ReadinessDecision(
                ready=False,
                status=Phase3Status.FAILED,
                reasons=[
                    "Phase 3 execution is disabled."
                ],
            )
        )

        if (
            self.config.execution
            .store_phase_state_in_result
        ):
            result.phase_state = (
                state.model_dump(
                    mode="json"
                )
            )

        return result

    def build_report_dictionary(
        self,
    ) -> dict[str, Any]:
        """Build the latest Phase 3 report dictionary."""

        result, state = self._require_run()

        return self.report_builder.build_dictionary(
            result=result,
            state=state,
        )

    def build_json_report(
        self,
    ) -> str:
        """Build the latest Phase 3 JSON report."""

        result, state = self._require_run()

        return self.report_builder.build_json(
            result=result,
            state=state,
        )

    def build_text_report(
        self,
    ) -> str:
        """Build the latest Phase 3 text report."""

        result, state = self._require_run()

        return self.report_builder.build_text(
            result=result,
            state=state,
        )

    def _require_run(
        self,
    ) -> tuple[
        Phase3Result,
        Phase3State,
    ]:
        """Return the latest Phase 3 result and state."""

        if (
            self.result is None
            or self.state is None
        ):
            raise RuntimeError(
                "Phase 3 has not been executed."
            )

        return (
            self.result,
            self.state,
        )

    @staticmethod
    def _validate_phase2_result(
        phase2_result: Phase2Result,
    ) -> None:
        """Validate the Phase 2 result object."""

        if not isinstance(
            phase2_result,
            Phase2Result,
        ):
            raise TypeError(
                "phase2_result must be an instance "
                "of Phase2Result."
            )

    @staticmethod
    def _require_value(
        value: ComponentResult | None,
        name: str,
    ) -> ComponentResult:
        """Return a required component result."""

        if value is None:
            raise ValueError(
                f"Required Phase 3 result "
                f"{name!r} is missing."
            )

        return value