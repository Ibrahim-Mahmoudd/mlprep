"""Main orchestration pipeline for Phase 4."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    Phase2Result,
)
from mlprep.phases.phase2.output_governance.state import (
    Phase2State,
)
from mlprep.phases.phase3.output_governance.result import (
    Phase3Result,
)
from mlprep.phases.phase3.output_governance.state import (
    Phase3State,
)
from mlprep.phases.phase4.config import (
    Phase4Config,
)
from mlprep.phases.phase4.contract.imputation_contract import (
    ImputationContract,
    ImputationContractIssue,
    ImputationContractResolver,
)
from mlprep.phases.phase4.contract.imputation_plan import (
    ImputationPlan,
    ImputationPlanBuilder,
)
from mlprep.phases.phase4.detection.missingness_scanner import (
    MissingnessScanner,
)
from mlprep.phases.phase4.orchestration.imputation_pipeline import (
    ImputationPipeline,
    ImputationPipelineResult,
)
from mlprep.phases.phase4.output_governance.fingerprint import (
    Phase4FingerprintManager,
)
from mlprep.phases.phase4.output_governance.metadata import (
    ImputationExecutionStatus,
    ImputationMethodMetadata,
    PartitionImputationMetadata,
    Phase4ExecutionMetadata,
    Phase4FindingAction,
    Phase4FindingCategory,
    Phase4FindingSeverity,
    Phase4Status,
)
from mlprep.phases.phase4.output_governance.readiness import (
    Phase4ReadinessEvaluator,
)
from mlprep.phases.phase4.output_governance.report import (
    Phase4ReportBuilder,
)
from mlprep.phases.phase4.output_governance.result import (
    Phase4Finding,
    Phase4ReadinessDecision,
    Phase4Result,
)
from mlprep.phases.phase4.output_governance.state import (
    Phase4State,
)
from mlprep.phases.phase4.validation.distribution_impact import (
    DistributionImpactValidator,
)
from mlprep.phases.phase4.validation.remaining_missingness import (
    RemainingMissingnessValidator,
)
from mlprep.phases.phase4.validation.row_integrity import (
    RowIntegrityValidator,
)
from mlprep.phases.phase4.validation.schema_consistency import (
    SchemaConsistencyValidator,
)
from mlprep.phases.phase4.validation.target_integrity import (
    TargetIntegrityValidator,
)

ComponentResult = TypeVar(
    "ComponentResult"
)


class Phase4:
    """Execute missing-value treatment and validation."""

    phase_name = "phase_4"

    def __init__(
        self,
        config: Phase4Config | None = None,
    ) -> None:
        """Initialize Phase 4."""

        self.config = (
            config or Phase4Config()
        )

        if not isinstance(
            self.config,
            Phase4Config,
        ):
            raise TypeError(
                "config must be a Phase4Config."
            )

        self.contract: (
            ImputationContract | None
        ) = None

        self.plan: ImputationPlan | None = None
        self.pipeline: (
            ImputationPipeline | None
        ) = None

        self.state: Phase4State | None = None
        self.result: Phase4Result | None = None

        self.fingerprint_manager = (
            Phase4FingerprintManager()
        )

        self.report_builder = (
            Phase4ReportBuilder(
                options=self.config.reporting
            )
        )

    def run(
        self,
        *,
        phase2_result: Phase2Result,
        phase3_result: Phase3Result,
        phase1_state: Phase1State,
        authorized_test_partition: (
            DatasetPartition | None
        ) = None,
        phase2_state: Phase2State | None = None,
        phase3_state: Phase3State | None = None,
    ) -> Phase4Result:
        """Execute the complete Phase 4 workflow."""

        resolved_phase2_state = (
            phase2_state
            or phase2_result.state
        )

        resolved_phase3_state = (
            phase3_state
            or self._resolve_phase3_state(
                phase3_result
            )
        )

        test_partition = (
            authorized_test_partition
        )

        if test_partition is None:
            test_partition = self._resolve_unlocked_test(
                phase2_result
            )

        if not self.config.execution.enabled:
            return self._disabled_result(
                phase2_result=phase2_result,
                test_partition=test_partition,
            )

        contract = ImputationContractResolver(
            options=self.config.contract
        ).resolve(
            phase2_result=phase2_result,
            phase3_result=phase3_result,
            phase1_state=phase1_state,
            phase2_state=resolved_phase2_state,
            phase3_state=resolved_phase3_state,
            test_partition=test_partition,
        )

        self.contract = contract

        state = Phase4State(
            fitted_on_training_only=True,
            test_transform_authorized=(
                test_partition is not None
            ),
            test_partition_remained_locked=(
                self.config.execution
                .relock_test_partition
            ),
            test_access_count_before=(
                resolved_phase2_state
                .test_lock
                .access_count
            ),
            test_access_count_after=(
                resolved_phase2_state
                .test_lock
                .access_count
            ),
            source_feature_columns=(
                contract.feature_columns
            ),
        )

        self.state = state

        empty_result = Phase4Result(
            train=contract.train,
            validation=contract.validation,
            test=contract.test,
        )

        self.result = empty_result

        empty_result.add_findings(
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
                result=empty_result,
                state=state,
            )

        contract.assert_ready()

        plan = self._execute(
            component="imputation_plan",
            operation=lambda: ImputationPlanBuilder(
                options=self.config.plan
            ).build(
                contract=contract
            ),
            fallback=None,
            state=state,
        )

        if plan is None:
            return self._failed_result(
                result=empty_result,
                state=state,
            )

        self.plan = plan

        blocked_entries = [
            entry.column
            for entry in plan.entries.values()
            if entry.status.value == "blocked"
        ]

        if blocked_entries:
            empty_result.add_finding(
                Phase4Finding(
                    code="P4-PLAN-001",
                    category=(
                        Phase4FindingCategory.PLAN
                    ),
                    severity=(
                        Phase4FindingSeverity.ERROR
                    ),
                    component="imputation_plan",
                    message=(
                        "The imputation plan contains "
                        "blocked feature entries."
                    ),
                    columns=blocked_entries,
                    recommended_action=(
                        Phase4FindingAction
                        .RECONFIGURE
                    ),
                    applied_action=(
                        Phase4FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

            if (
                self.config.execution
                .stop_on_plan_failure
            ):
                return self._failed_result(
                    result=empty_result,
                    state=state,
                )

        state.configuration_fingerprint = (
            self.fingerprint_manager
            .configuration_fingerprint(
                self.config
            )
        )

        state.imputation_plan_fingerprint = (
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

        missingness_scan = self._execute(
            component="missingness_scanner",
            operation=lambda: MissingnessScanner(
                options=(
                    self.config
                    .missingness_scanner
                )
            ).scan(
                contract=contract,
                plan=plan,
            ),
            fallback=None,
            state=state,
        )

        if (
            missingness_scan is not None
            and missingness_scan
            .features_missing_without_plan
        ):
            empty_result.add_finding(
                Phase4Finding(
                    code="P4-DETECTION-001",
                    category=(
                        Phase4FindingCategory
                        .DETECTION
                    ),
                    severity=(
                        Phase4FindingSeverity.ERROR
                    ),
                    component=(
                        "missingness_scanner"
                    ),
                    message=(
                        "Features with missing values lack "
                        "an imputation plan."
                    ),
                    columns=(
                        missingness_scan
                        .features_missing_without_plan
                    ),
                    recommended_action=(
                        Phase4FindingAction
                        .RECONFIGURE
                    ),
                    applied_action=(
                        Phase4FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        pipeline = ImputationPipeline(
            options=self.config.pipeline
        )

        self.pipeline = pipeline

        pipeline_output = self._execute(
            component="imputation_pipeline",
            operation=lambda: pipeline.fit_transform(
                contract=contract,
                plan=plan,
                test_authorized=(
                    state.test_transform_authorized
                ),
            ),
            fallback=None,
            state=state,
        )

        if pipeline_output is None:
            return self._failed_result(
                result=empty_result,
                state=state,
            )

        (
            transformed_train,
            transformed_validation,
            transformed_test,
            pipeline_result,
        ) = pipeline_output

        result = Phase4Result(
            train=transformed_train,
            validation=transformed_validation,
            test=transformed_test,
        )

        result.add_findings(
            empty_result.findings
        )

        self.result = result

        self._register_pipeline_state(
            state=state,
            pipeline_result=pipeline_result,
            plan=plan,
        )

        self._run_validations(
            contract=contract,
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
            pipeline_result=pipeline_result,
            plan=plan,
        )

        evaluator = Phase4ReadinessEvaluator(
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
            result.phase_state = state.model_dump(
                mode="json"
            )

        return result

    def _run_validations(
        self,
        *,
        contract: ImputationContract,
        result: Phase4Result,
        state: Phase4State,
    ) -> None:
        """Run every Phase 4 validator."""

        remaining = self._execute(
            component=(
                "remaining_missingness"
            ),
            operation=lambda: (
                RemainingMissingnessValidator(
                    self.config
                    .validation
                    .remaining_missingness
                ).validate(
                    train=result.train,
                    validation=result.validation,
                    test=result.test,
                )
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.remaining_missingness = (
            remaining
        )

        if remaining is not None:
            state.remaining_missing_count = (
                remaining
                .total_remaining_missing_count
            )

        schema = self._execute(
            component="schema_consistency",
            operation=lambda: (
                SchemaConsistencyValidator(
                    self.config
                    .validation
                    .schema_consistency
                ).validate(
                    train=result.train,
                    validation=result.validation,
                    test=result.test,
                )
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
            operation=lambda: (
                TargetIntegrityValidator(
                    self.config
                    .validation
                    .target_integrity
                ).validate(
                    source_train=contract.train,
                    transformed_train=(
                        result.train
                    ),
                    source_validation=(
                        contract.validation
                    ),
                    transformed_validation=(
                        result.validation
                    ),
                    source_test=contract.test,
                    transformed_test=result.test,
                )
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
            operation=lambda: (
                DistributionImpactValidator(
                    self.config
                    .validation
                    .distribution_impact
                ).validate(
                    source=contract.train,
                    transformed=result.train,
                )
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
        state: Phase4State,
        pipeline_result: ImputationPipelineResult,
        plan: ImputationPlan,
    ) -> None:
        """Register fitted pipeline state."""

        state.indicator_mapping = dict(
            pipeline_result.indicator_mapping
        )

        for column, imputer_state in (
            pipeline_result
            .imputer_states
            .items()
        ):
            state.register_imputer_state(
                column=column,
                state=imputer_state,
            )

        state.output_feature_columns = (
            self._output_feature_columns(
                source_columns=(
                    state.source_feature_columns
                ),
                plan=plan,
                indicator_mapping=(
                    state.indicator_mapping
                ),
            )
        )

        state.fitted_on_training_only = (
            pipeline_result
            .fitted_on_training_only
        )

        state.imputed_value_count = sum(
            partition_result
            .transformation
            .total_imputed_count
            for partition_result
            in pipeline_result
            .partitions
            .values()
        )

        state.generated_indicator_count = len(
            state.indicator_mapping
        )

    def _register_execution_metadata(
        self,
        *,
        result: Phase4Result,
        state: Phase4State,
        pipeline_result: ImputationPipelineResult,
        plan: ImputationPlan,
    ) -> None:
        """Register execution metadata."""

        if not (
            self.config.execution
            .store_execution_metadata
        ):
            return

        feature_methods: dict[
            str,
            ImputationMethodMetadata,
        ] = {}

        for column, entry in (
            plan.entries.items()
        ):
            training_metadata = (
                pipeline_result
                .partitions[
                    PartitionName.TRAIN
                ]
                .transformation
                .columns
                .get(column)
            )

            validation_metadata = (
                pipeline_result
                .partitions[
                    PartitionName.VALIDATION
                ]
                .transformation
                .columns
                .get(column)
            )

            test_metadata = (
                pipeline_result
                .partitions[
                    PartitionName.TEST
                ]
                .transformation
                .columns
                .get(column)
            )

            feature_methods[column] = (
                ImputationMethodMetadata(
                    column=column,
                    feature_family=(
                        entry
                        .feature_family
                        .value
                    ),
                    method=entry.method.value,
                    execution_status=(
                        ImputationExecutionStatus
                        .TRANSFORMED
                        if (
                            training_metadata
                            is not None
                            and training_metadata
                            .transformed
                        )
                        else (
                            ImputationExecutionStatus
                            .SKIPPED
                        )
                    ),
                    requires_training_fit=(
                        entry
                        .requires_training_fit
                    ),
                    fitted_on_training_only=(
                        state
                        .fitted_on_training_only
                    ),
                    add_missing_indicator=(
                        entry
                        .add_missing_indicator
                    ),
                    indicator_column=(
                        state
                        .indicator_mapping
                        .get(column)
                    ),
                    training_missing_count=(
                        training_metadata
                        .missing_before
                        if training_metadata
                        is not None
                        else 0
                    ),
                    training_imputed_count=(
                        training_metadata
                        .imputed_count
                        if training_metadata
                        is not None
                        else 0
                    ),
                    validation_imputed_count=(
                        validation_metadata
                        .imputed_count
                        if validation_metadata
                        is not None
                        else 0
                    ),
                    test_imputed_count=(
                        test_metadata
                        .imputed_count
                        if test_metadata
                        is not None
                        else 0
                    ),
                    input_dtype=(
                        training_metadata
                        .input_dtype
                        if training_metadata
                        is not None
                        else None
                    ),
                    output_dtype=(
                        training_metadata
                        .output_dtype
                        if training_metadata
                        is not None
                        else None
                    ),
                    warnings=list(
                        entry.warnings
                    ),
                )
            )

        partition_metadata: dict[
            PartitionName,
            PartitionImputationMetadata,
        ] = {}

        for partition, partition_result in (
            pipeline_result
            .partitions
            .items()
        ):
            transformation = (
                partition_result
                .transformation
            )

            partition_metadata[
                partition
            ] = PartitionImputationMetadata(
                partition=partition,
                row_count=(
                    transformation.row_count_after
                ),
                feature_count_before=(
                    transformation
                    .feature_count_before
                ),
                feature_count_after=(
                    transformation
                    .feature_count_after
                ),
                missing_count_before=(
                    transformation
                    .missing_before
                ),
                missing_count_after=(
                    transformation
                    .missing_after
                ),
                imputed_value_count=(
                    transformation
                    .total_imputed_count
                ),
                invalid_value_count=(
                    partition_result
                    .invalid_value_scan
                    .total_invalid_count
                ),
                missing_token_count=(
                    partition_result
                    .missing_token_scan
                    .total_matched_count
                ),
                indicator_count=(
                    len(
                        transformation
                        .generated_indicator_columns
                    )
                ),
                indicator_columns=list(
                    transformation
                    .generated_indicator_columns
                ),
                dropped_columns=list(
                    transformation
                    .dropped_columns
                ),
                input_fingerprint=(
                    state
                    .input_partition_fingerprints
                    .get(partition)
                ),
                output_fingerprint=(
                    state
                    .output_partition_fingerprints
                    .get(partition)
                ),
            )

        result.execution_metadata = (
            Phase4ExecutionMetadata(
                fitted_on_training_only=(
                    state
                    .fitted_on_training_only
                ),
                test_transform_authorized=(
                    state
                    .test_transform_authorized
                ),
                test_partition_remained_locked=(
                    state
                    .test_partition_remained_locked
                ),
                feature_methods=feature_methods,
                partitions=partition_metadata,
                configuration_fingerprint=(
                    state
                    .configuration_fingerprint
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
        state: Phase4State,
    ) -> ComponentResult:
        """Execute one component using configured policy."""

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
                    Phase4Finding(
                        code="P4-EXECUTION-001",
                        category=(
                            Phase4FindingCategory
                            .EXECUTION
                        ),
                        severity=(
                            Phase4FindingSeverity
                            .CRITICAL
                        ),
                        component=component,
                        message=(
                            f"Component {component!r} failed "
                            f"with {type(error).__name__}: "
                            f"{error}"
                        ),
                        recommended_action=(
                            Phase4FindingAction.RAISE
                        ),
                        applied_action=(
                            Phase4FindingAction
                            .REPORT
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
        issues: list[
            ImputationContractIssue
        ],
    ) -> list[object]:
        """Convert contract issues to findings."""

        severity_mapping = {
            "info": (
                Phase4FindingSeverity.INFO
            ),
            "warning": (
                Phase4FindingSeverity.WARNING
            ),
            "error": (
                Phase4FindingSeverity.ERROR
            ),
            "critical": (
                Phase4FindingSeverity.CRITICAL
            ),
        }

        return [
            Phase4Finding(
                code=issue.code,
                category=(
                    Phase4FindingCategory.CONTRACT
                ),
                severity=severity_mapping[
                    issue.severity
                ],
                component=(
                    "imputation_contract"
                ),
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
                    Phase4FindingAction
                    .RECONFIGURE
                ),
                applied_action=(
                    Phase4FindingAction.REPORT
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
        result: Phase4Result,
        state: Phase4State,
    ) -> Phase4Result:
        """Finalize a failed Phase 4 result."""

        blocking = (
            result.blocking_findings()
        )

        decision = Phase4ReadinessDecision(
            ready=False,
            status=Phase4Status.FAILED,
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

        state.ready_for_phase_5 = False

        result.finalize(
            decision
        )

        if (
            self.config.execution
            .store_phase_state_in_result
        ):
            result.phase_state = state.model_dump(
                mode="json"
            )

        return result

    def _disabled_result(
        self,
        *,
        phase2_result: Phase2Result,
        test_partition: DatasetPartition,
    ) -> Phase4Result:
        """Build a disabled Phase 4 result."""

        result = Phase4Result(
            train=phase2_result.train,
            validation=phase2_result.validation,
            test=test_partition,
        )

        state = Phase4State(
            fitted_on_training_only=False,
            test_transform_authorized=False,
            test_partition_remained_locked=True,
        )

        self.state = state
        self.result = result

        result.add_finding(
            Phase4Finding(
                code="P4-EXECUTION-002",
                category=(
                    Phase4FindingCategory.EXECUTION
                ),
                severity=(
                    Phase4FindingSeverity.INFO
                ),
                component=self.phase_name,
                message=(
                    "Phase 4 execution is disabled."
                ),
                recommended_action=(
                    Phase4FindingAction.NONE
                ),
                applied_action=(
                    Phase4FindingAction.NONE
                ),
                blocks_readiness=False,
            )
        )

        result.finalize(
            Phase4ReadinessDecision(
                ready=False,
                status=Phase4Status.DISABLED,
                reasons=[
                    "Phase 4 execution is disabled."
                ],
            )
        )

        return result

    @staticmethod
    def _resolve_phase3_state(
        phase3_result: Phase3Result,
    ) -> Phase3State:
        """Resolve stored Phase 3 state."""

        if not phase3_result.phase_state:
            raise ValueError(
                "Phase 3 state is unavailable."
            )

        return Phase3State.model_validate(
            phase3_result.phase_state
        )

    @staticmethod
    def _resolve_unlocked_test(
        phase2_result: Phase2Result,
    ) -> DatasetPartition:
        """Resolve a directly available test partition."""

        if isinstance(
            phase2_result.test,
            DatasetPartition,
        ):
            return phase2_result.test

        raise PermissionError(
            "Pass authorized_test_partition after "
            "requesting authorized test access from Phase 2."
        )

    @staticmethod
    def _output_feature_columns(
        *,
        source_columns: list[str],
        plan: ImputationPlan,
        indicator_mapping: dict[
            str,
            str,
        ],
    ) -> list[object]:
        """Resolve the final output feature order."""

        dropped = set(
            plan.drop_feature_columns
        )

        output_columns = [
            column
            for column in source_columns
            if column not in dropped
        ]

        for indicator in (
            indicator_mapping.values()
        ):
            if indicator not in output_columns:
                output_columns.append(
                    indicator
                )

        return output_columns

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
        Phase4Result,
        Phase4State,
    ]:
        """Return the latest result and state."""

        if (
            self.result is None
            or self.state is None
        ):
            raise RuntimeError(
                "Phase 4 has not been executed."
            )

        return (
            self.result,
            self.state,
        )
