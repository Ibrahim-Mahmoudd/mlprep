"""Main orchestration pipeline for Phase 5."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase4.output_governance.result import (
    Phase4Result,
)
from mlprep.phases.phase4.output_governance.state import (
    Phase4State,
)
from mlprep.phases.phase5.config import (
    Phase5Config,
)
from mlprep.phases.phase5.contract.outlier_contract import (
    OutlierContract,
    OutlierContractIssue,
    OutlierContractResolver,
)
from mlprep.phases.phase5.contract.treatment_plan import (
    TreatmentPlan,
    TreatmentPlanBuilder,
    TreatmentPlanStatus,
)
from mlprep.phases.phase5.orchestration.outlier_pipeline import (
    OutlierPipeline,
    OutlierPipelineResult,
)
from mlprep.phases.phase5.output_governance.fingerprint import (
    Phase5FingerprintManager,
)
from mlprep.phases.phase5.output_governance.metadata import (
    DetectorExecutionStatus,
    FeatureTreatmentMetadata,
    PartitionTreatmentSummary,
    Phase5ExecutionMetadata,
    Phase5Finding,
    Phase5FindingAction,
    Phase5FindingCategory,
    Phase5FindingSeverity,
    Phase5Status,
    TreatmentExecutionStatus,
)
from mlprep.phases.phase5.output_governance.readiness import (
    Phase5ReadinessEvaluator,
)
from mlprep.phases.phase5.output_governance.report import (
    Phase5ReportBuilder,
)
from mlprep.phases.phase5.output_governance.result import (
    Phase5ReadinessDecision,
    Phase5Result,
)
from mlprep.phases.phase5.output_governance.state import (
    Phase5State,
)
from mlprep.phases.phase5.validation.distribution_impact import (
    DistributionImpactValidator,
)
from mlprep.phases.phase5.validation.remaining_outliers import (
    RemainingOutliersValidator,
)
from mlprep.phases.phase5.validation.row_integrity import (
    RowIntegrityValidator,
)
from mlprep.phases.phase5.validation.schema_consistency import (
    SchemaConsistencyValidator,
)
from mlprep.phases.phase5.validation.target_integrity import (
    TargetIntegrityValidator,
)

ComponentResult = TypeVar(
    "ComponentResult"
)


class Phase5:
    """Execute outlier detection, treatment, and validation."""

    phase_name = "phase_5"

    def __init__(
        self,
        config: Phase5Config | None = None,
    ) -> None:
        """Initialize Phase 5."""

        self.config = (
            config or Phase5Config()
        )

        if not isinstance(
            self.config,
            Phase5Config,
        ):
            raise TypeError(
                "config must be a Phase5Config."
            )

        self.contract: OutlierContract | None = None
        self.plan: TreatmentPlan | None = None
        self.pipeline: OutlierPipeline | None = None

        self.state: Phase5State | None = None
        self.result: Phase5Result | None = None

        self.fingerprint_manager = (
            Phase5FingerprintManager()
        )

        self.report_builder = (
            Phase5ReportBuilder(
                options=self.config.reporting
            )
        )

    def run(
        self,
        *,
        phase4_result: Phase4Result,
        phase4_state: Phase4State | None = None,
        authorized_test_partition: (
            DatasetPartition | None
        ) = None,
    ) -> Phase5Result:
        """Execute the complete Phase 5 workflow."""

        if not isinstance(
            phase4_result,
            Phase4Result,
        ):
            raise TypeError(
                "phase4_result must be a Phase4Result."
            )

        resolved_state = (
            phase4_state
            or self._resolve_phase4_state(
                phase4_result
            )
        )

        test_partition = (
            authorized_test_partition
            or phase4_result.test
        )

        if not isinstance(
            test_partition,
            DatasetPartition,
        ):
            raise PermissionError(
                "Phase 5 requires an authorized "
                "Phase 4 test partition."
            )

        if not self.config.execution.enabled:
            return self._disabled_result(
                phase4_result=phase4_result,
                test_partition=test_partition,
            )

        contract = OutlierContractResolver(
            options=self.config.contract
        ).resolve(
            phase4_result=phase4_result,
            phase4_state=resolved_state,
            authorized_test_partition=(
                test_partition
            ),
        )

        self.contract = contract

        state = Phase5State(
            fitted_on_training_only=True,
            test_transform_authorized=(
                authorized_test_partition
                is not None
                or isinstance(
                    phase4_result.test,
                    DatasetPartition,
                )
            ),
            test_partition_remained_protected=(
                self.config.execution
                .keep_test_partition_protected
            ),
            source_feature_columns=(
                contract.feature_columns
            ),
            phase4_output_fingerprint=(
                resolved_state
                .output_fingerprint
            ),
        )

        self.state = state

        initial_result = Phase5Result(
            train=contract.train,
            validation=contract.validation,
            test=contract.test,
        )

        self.result = initial_result

        initial_result.add_findings(
            self._contract_findings(
                contract.issues
            )
        )

        if (
            self.config.execution
            .stop_on_contract_failure
            and contract.has_blocking_issues
        ):
            return self._failed_result(
                result=initial_result,
                state=state,
            )

        contract.assert_ready()

        plan = self._execute(
            component="treatment_plan",
            operation=lambda: TreatmentPlanBuilder(
                options=self.config.plan
            ).build(
                contract=contract
            ),
            fallback=None,
            state=state,
        )

        if plan is None:
            return self._failed_result(
                result=initial_result,
                state=state,
            )

        self.plan = plan

        blocked_columns = [
            entry.column
            for entry in plan.entries.values()
            if (
                entry.status
                == TreatmentPlanStatus.BLOCKED
            )
        ]

        if blocked_columns:
            initial_result.add_finding(
                Phase5Finding(
                    code="P5-PLAN-001",
                    category=(
                        Phase5FindingCategory.PLAN
                    ),
                    severity=(
                        Phase5FindingSeverity.ERROR
                    ),
                    component="treatment_plan",
                    message=(
                        "The treatment plan contains "
                        "blocked entries."
                    ),
                    columns=blocked_columns,
                    recommended_action=(
                        Phase5FindingAction
                        .RECONFIGURE
                    ),
                    applied_action=(
                        Phase5FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

            if (
                self.config.execution
                .stop_on_plan_failure
            ):
                return self._failed_result(
                    result=initial_result,
                    state=state,
                )

        state.configuration_fingerprint = (
            self.fingerprint_manager
            .configuration_fingerprint(
                self.config
            )
        )

        state.treatment_plan_fingerprint = (
            self.fingerprint_manager
            .plan_fingerprint(
                plan
            )
        )

        self.fingerprint_manager.capture_inputs(
            train=contract.train,
            validation=contract.validation,
            test=contract.test,
            state=state,
        )

        pipeline = OutlierPipeline(
            options=self.config.pipeline
        )

        self.pipeline = pipeline

        pipeline_output = self._execute(
            component="outlier_pipeline",
            operation=lambda: pipeline.fit_transform(
                contract=contract,
                plan=plan,
            ),
            fallback=None,
            state=state,
        )

        if pipeline_output is None:
            return self._failed_result(
                result=initial_result,
                state=state,
            )

        (
            transformed_train,
            transformed_validation,
            transformed_test,
            pipeline_result,
        ) = pipeline_output

        result = Phase5Result(
            train=transformed_train,
            validation=transformed_validation,
            test=transformed_test,
        )

        result.add_findings(
            initial_result.findings
        )

        self.result = result

        self._register_pipeline_state(
            state=state,
            pipeline_result=pipeline_result,
        )

        self._run_validations(
            contract=contract,
            plan=plan,
            pipeline=pipeline,
            result=result,
            state=state,
        )

        self.fingerprint_manager.capture_outputs(
            train=result.train,
            validation=result.validation,
            test=result.test,
            state=state,
        )

        result.add_findings(
            self.fingerprint_manager
            .verify_protected_data(
                state=state
            )
        )

        state.fitted_state_fingerprint = (
            self.fingerprint_manager
            .fitted_state_fingerprint(
                state=state
            )
        )

        state.output_fingerprint = (
            self.fingerprint_manager
            .output_fingerprint(
                state=state
            )
        )

        self._register_execution_metadata(
            result=result,
            state=state,
            plan=plan,
            pipeline_result=pipeline_result,
        )

        evaluator = Phase5ReadinessEvaluator(
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

    def _run_validations(
        self,
        *,
        contract: OutlierContract,
        plan: TreatmentPlan,
        pipeline: OutlierPipeline,
        result: Phase5Result,
        state: Phase5State,
    ) -> None:
        """Run every Phase 5 validator."""

        remaining = self._execute(
            component="remaining_outliers",
            operation=lambda: RemainingOutliersValidator(
                self.config
                .validation
                .remaining_outliers
            ).validate(
                train=result.train,
                validation=result.validation,
                test=result.test,
                plan=plan,
                detector_engine=(
                    pipeline.detector_engine
                ),
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.remaining_outliers = (
            remaining
        )

        if remaining is not None:
            state.total_remaining_outlier_count = (
                remaining
                .total_remaining_outlier_count
            )

        schema = self._execute(
            component="schema_consistency",
            operation=lambda: SchemaConsistencyValidator(
                self.config
                .validation
                .schema_consistency
            ).validate(
                train=result.train,
                validation=result.validation,
                test=result.test,
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.schema_consistency = (
            schema
        )

        if schema is not None:
            state.schemas_consistent = (
                schema.valid
            )

        target_integrity = self._execute(
            component="target_integrity",
            operation=lambda: TargetIntegrityValidator(
                self.config
                .validation
                .target_integrity
            ).validate(
                source_train=contract.train,
                transformed_train=result.train,
                source_validation=(
                    contract.validation
                ),
                transformed_validation=(
                    result.validation
                ),
                source_test=contract.test,
                transformed_test=result.test,
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.target_integrity = (
            target_integrity
        )

        row_integrity = self._execute(
            component="row_integrity",
            operation=lambda: RowIntegrityValidator(
                self.config
                .validation
                .row_integrity
            ).validate(
                source_train=contract.train,
                transformed_train=result.train,
                source_validation=(
                    contract.validation
                ),
                transformed_validation=(
                    result.validation
                ),
                source_test=contract.test,
                transformed_test=result.test,
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.row_integrity = (
            row_integrity
        )

        distribution_impact = self._execute(
            component="distribution_impact",
            operation=lambda: DistributionImpactValidator(
                self.config
                .validation
                .distribution_impact
            ).validate(
                source=contract.train,
                transformed=result.train,
                fitted_bounds=(
                    state.detector_states
                ),
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.distribution_impact = (
            distribution_impact
        )

    def _register_pipeline_state(
        self,
        *,
        state: Phase5State,
        pipeline_result: OutlierPipelineResult,
    ) -> None:
        """Register fitted detector and treatment state."""

        state.detector_states = {
            column: dict(detector_state)
            for column, detector_state
            in pipeline_result
            .detector_states
            .items()
        }

        for column, treatment_state in (
            pipeline_result
            .treatment_states
            .items()
        ):
            state.register_treatment_state(
                column=column,
                treatment_state=(
                    treatment_state
                ),
            )

        state.indicator_mapping = dict(
            pipeline_result.indicator_mapping
        )

        state.output_feature_columns = (
            pipeline_result
            .partitions[
                PartitionName.TRAIN
            ]
            .treatment
            .columns
            .keys()
        )

        state.output_feature_columns = list(
            self.result.train.feature_names()
            if self.result is not None
            else state.source_feature_columns
        )

        state.total_detected_outlier_count = sum(
            partition_result
            .treatment
            .detected_outlier_count
            for partition_result
            in pipeline_result
            .partitions
            .values()
        )

        state.total_changed_value_count = sum(
            partition_result
            .treatment
            .changed_value_count
            for partition_result
            in pipeline_result
            .partitions
            .values()
        )

        state.generated_indicator_count = len(
            state.indicator_mapping
        )

        state.fitted_on_training_only = (
            pipeline_result
            .fitted_on_training_only
        )

    def _register_execution_metadata(
        self,
        *,
        result: Phase5Result,
        state: Phase5State,
        plan: TreatmentPlan,
        pipeline_result: OutlierPipelineResult,
    ) -> None:
        """Register auditable execution metadata."""

        if not (
            self.config.execution
            .store_execution_metadata
        ):
            return

        features: dict[
            str,
            FeatureTreatmentMetadata,
        ] = {}

        for column, entry in (
            plan.entries.items()
        ):
            training = (
                pipeline_result
                .partitions[
                    PartitionName.TRAIN
                ]
                .treatment
                .columns
                .get(column)
            )

            validation = (
                pipeline_result
                .partitions[
                    PartitionName.VALIDATION
                ]
                .treatment
                .columns
                .get(column)
            )

            test = (
                pipeline_result
                .partitions[
                    PartitionName.TEST
                ]
                .treatment
                .columns
                .get(column)
            )

            detector_state = (
                state.detector_states.get(
                    column,
                    {},
                )
            )

            features[column] = (
                FeatureTreatmentMetadata(
                    column=column,
                    detector=entry.detector.value,
                    treatment=(
                        entry.treatment.value
                    ),
                    detector_status=(
                        DetectorExecutionStatus
                        .APPLIED
                        if training is not None
                        else DetectorExecutionStatus
                        .SKIPPED
                    ),
                    treatment_status=(
                        TreatmentExecutionStatus
                        .APPLIED
                        if (
                            training is not None
                            and training.transformed
                        )
                        else TreatmentExecutionStatus
                        .SKIPPED
                    ),
                    fitted_on_training_only=(
                        state
                        .fitted_on_training_only
                    ),
                    lower_bound=(
                        self._optional_float(
                            detector_state.get(
                                "lower_bound"
                            )
                        )
                    ),
                    upper_bound=(
                        self._optional_float(
                            detector_state.get(
                                "upper_bound"
                            )
                        )
                    ),
                    add_indicator=(
                        entry.add_indicator
                    ),
                    indicator_column=(
                        state
                        .indicator_mapping
                        .get(column)
                    ),
                    training_outlier_count=(
                        training.outlier_count
                        if training is not None
                        else 0
                    ),
                    validation_outlier_count=(
                        validation.outlier_count
                        if validation is not None
                        else 0
                    ),
                    test_outlier_count=(
                        test.outlier_count
                        if test is not None
                        else 0
                    ),
                    training_changed_count=(
                        training.changed_value_count
                        if training is not None
                        else 0
                    ),
                    validation_changed_count=(
                        validation.changed_value_count
                        if validation is not None
                        else 0
                    ),
                    test_changed_count=(
                        test.changed_value_count
                        if test is not None
                        else 0
                    ),
                    warnings=list(
                        entry.warnings
                    ),
                )
            )

        partitions: dict[
            PartitionName,
            PartitionTreatmentSummary,
        ] = {}

        for partition_name, partition_result in (
            pipeline_result
            .partitions
            .items()
        ):
            treatment = (
                partition_result.treatment
            )

            partitions[
                partition_name
            ] = PartitionTreatmentSummary(
                partition=partition_name,
                row_count=(
                    treatment.row_count_after
                ),
                feature_count_before=(
                    treatment
                    .feature_count_before
                ),
                feature_count_after=(
                    treatment
                    .feature_count_after
                ),
                detected_outlier_count=(
                    treatment
                    .detected_outlier_count
                ),
                changed_value_count=(
                    treatment.changed_value_count
                ),
                remaining_outlier_count=(
                    treatment
                    .remaining_outlier_count
                ),
                indicator_count=len(
                    treatment
                    .generated_indicator_columns
                ),
                indicator_columns=list(
                    treatment
                    .generated_indicator_columns
                ),
                input_fingerprint=(
                    state
                    .input_partition_fingerprints
                    .get(partition_name)
                ),
                output_fingerprint=(
                    state
                    .output_partition_fingerprints
                    .get(partition_name)
                ),
            )

        result.execution_metadata = (
            Phase5ExecutionMetadata(
                fitted_on_training_only=(
                    state
                    .fitted_on_training_only
                ),
                test_transform_authorized=(
                    state
                    .test_transform_authorized
                ),
                test_partition_remained_protected=(
                    state
                    .test_partition_remained_protected
                ),
                features=features,
                partitions=partitions,
                configuration_fingerprint=(
                    state
                    .configuration_fingerprint
                ),
                treatment_plan_fingerprint=(
                    state
                    .treatment_plan_fingerprint
                ),
                fitted_state_fingerprint=(
                    state
                    .fitted_state_fingerprint
                ),
                output_fingerprint=(
                    state.output_fingerprint
                ),
            )
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
        state: Phase5State,
    ) -> ComponentResult:
        """Execute one component safely."""

        try:
            result = operation()

            state.mark_completed(
                component
            )

            return result

        except Exception as error:
            state.mark_failed(
                component
            )

            if self.result is not None:
                self.result.add_finding(
                    Phase5Finding(
                        code="P5-EXECUTION-001",
                        category=(
                            Phase5FindingCategory
                            .EXECUTION
                        ),
                        severity=(
                            Phase5FindingSeverity
                            .CRITICAL
                        ),
                        component=component,
                        message=(
                            f"Component {component!r} failed "
                            f"with {type(error).__name__}: "
                            f"{error}"
                        ),
                        recommended_action=(
                            Phase5FindingAction.RAISE
                        ),
                        applied_action=(
                            Phase5FindingAction.REPORT
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

    @staticmethod
    def _contract_findings(
        issues: list[OutlierContractIssue],
    ) -> list[object]:
        """Convert contract issues into findings."""

        severity_mapping = {
            "info": Phase5FindingSeverity.INFO,
            "warning": (
                Phase5FindingSeverity.WARNING
            ),
            "error": Phase5FindingSeverity.ERROR,
            "critical": (
                Phase5FindingSeverity.CRITICAL
            ),
        }

        return [
            Phase5Finding(
                code=issue.code,
                category=(
                    Phase5FindingCategory.CONTRACT
                ),
                severity=severity_mapping[
                    issue.severity
                ],
                component="outlier_contract",
                message=issue.message,
                columns=list(
                    issue.columns
                ),
                partitions=[
                    partition.value
                    for partition
                    in issue.partitions
                ],
                recommended_action=(
                    Phase5FindingAction
                    .RECONFIGURE
                ),
                applied_action=(
                    Phase5FindingAction.REPORT
                ),
                blocks_readiness=(
                    issue.blocks_execution
                ),
                details=dict(
                    issue.details
                ),
            )
            for issue in issues
        ]

    def _failed_result(
        self,
        *,
        result: Phase5Result,
        state: Phase5State,
    ) -> Phase5Result:
        """Finalize a failed Phase 5 result."""

        blocking = (
            result.blocking_findings()
        )

        decision = Phase5ReadinessDecision(
            ready=False,
            status=Phase5Status.FAILED,
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

        state.ready_for_phase_6 = False

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

    def _disabled_result(
        self,
        *,
        phase4_result: Phase4Result,
        test_partition: DatasetPartition,
    ) -> Phase5Result:
        """Build a disabled Phase 5 result."""

        result = Phase5Result(
            train=phase4_result.train,
            validation=phase4_result.validation,
            test=test_partition,
        )

        state = Phase5State(
            fitted_on_training_only=False,
            test_transform_authorized=False,
            test_partition_remained_protected=True,
        )

        self.result = result
        self.state = state

        result.add_finding(
            Phase5Finding(
                code="P5-EXECUTION-002",
                category=(
                    Phase5FindingCategory.EXECUTION
                ),
                severity=(
                    Phase5FindingSeverity.INFO
                ),
                component=self.phase_name,
                message=(
                    "Phase 5 execution is disabled."
                ),
                recommended_action=(
                    Phase5FindingAction.NONE
                ),
                applied_action=(
                    Phase5FindingAction.NONE
                ),
                blocks_readiness=False,
            )
        )

        result.finalize(
            Phase5ReadinessDecision(
                ready=False,
                status=Phase5Status.DISABLED,
                reasons=[
                    "Phase 5 execution is disabled."
                ],
            )
        )

        return result

    @staticmethod
    def _resolve_phase4_state(
        result: Phase4Result,
    ) -> Phase4State:
        """Resolve Phase 4 state from its result."""

        direct_state = getattr(
            result,
            "state",
            None,
        )

        if isinstance(
            direct_state,
            Phase4State,
        ):
            return direct_state

        if result.phase_state:
            return Phase4State.model_validate(
                result.phase_state
            )

        raise ValueError(
            "Phase 4 state is unavailable."
        )

    @staticmethod
    def _optional_float(
        value: object,
    ) -> float | None:
        """Convert an optional value to float."""

        if value is None:
            return None

        return float(value)

    def build_report_dictionary(
        self,
    ) -> dict[str, Any]:
        """Build the latest report dictionary."""

        result, state = self._require_run()

        return self.report_builder.build_dictionary(
            result=result,
            state=state,
        )

    def build_json_report(
        self,
    ) -> str:
        """Build the latest JSON report."""

        result, state = self._require_run()

        return self.report_builder.build_json(
            result=result,
            state=state,
        )

    def build_text_report(
        self,
    ) -> str:
        """Build the latest text report."""

        result, state = self._require_run()

        return self.report_builder.build_text(
            result=result,
            state=state,
        )

    def _require_run(
        self,
    ) -> tuple[
        Phase5Result,
        Phase5State,
    ]:
        """Return the latest result and state."""

        if (
            self.result is None
            or self.state is None
        ):
            raise RuntimeError(
                "Phase 5 has not been executed."
            )

        return (
            self.result,
            self.state,
        )