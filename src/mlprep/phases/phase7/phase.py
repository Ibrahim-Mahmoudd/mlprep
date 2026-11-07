"""Main orchestration pipeline for Phase 7."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase6.output_governance import (
    Phase6Result,
    Phase6State,
)
from mlprep.phases.phase7.config import (
    Phase7Config,
)
from mlprep.phases.phase7.contract import (
    SelectionContractIssue,
    SelectionContractResolver,
    SelectionPlan,
    SelectionPlanBuilder,
)
from mlprep.phases.phase7.orchestration import (
    SelectionPipeline,
    SelectionPipelineResult,
)
from mlprep.phases.phase7.output_governance import (
    FeatureSelectionMetadata,
    FeatureSelectionStatus,
    PartitionSelectionSummary,
    Phase7ExecutionMetadata,
    Phase7Finding,
    Phase7FindingAction,
    Phase7FindingCategory,
    Phase7FindingSeverity,
    Phase7FingerprintManager,
    Phase7ReadinessDecision,
    Phase7ReadinessEvaluator,
    Phase7ReportBuilder,
    Phase7Result,
    Phase7State,
    Phase7Status,
)
from mlprep.phases.phase7.validation import (
    FeatureCountValidator,
    ProtectedFeaturesValidator,
    RowIntegrityValidator,
    SchemaConsistencyValidator,
    TargetIntegrityValidator,
)

ComponentResult = TypeVar(
    "ComponentResult"
)


class Phase7:
    """Execute feature selection and dimensionality governance."""

    phase_name = "phase_7"

    def __init__(
        self,
        config: Phase7Config | None = None,
    ) -> None:
        """Initialize Phase 7."""

        self.config = (
            config
            or Phase7Config()
        )

        if not isinstance(
            self.config,
            Phase7Config,
        ):
            raise TypeError(
                "config must be a Phase7Config."
            )

        self.state: Phase7State | None = None
        self.result: Phase7Result | None = None

        self.plan: SelectionPlan | None = None
        self.pipeline: SelectionPipeline | None = None

        self.fingerprint_manager = (
            Phase7FingerprintManager()
        )

        self.report_builder = Phase7ReportBuilder(
            options=self.config.reporting
        )

    def run(
        self,
        *,
        phase6_result: Phase6Result,
        phase6_state: Phase6State | None = None,
        authorized_test_partition: (
            DatasetPartition | None
        ) = None,
    ) -> Phase7Result:
        """Execute the complete Phase 7 workflow."""

        if not isinstance(
            phase6_result,
            Phase6Result,
        ):
            raise TypeError(
                "phase6_result must be a Phase6Result."
            )

        resolved_state = (
            phase6_state
            or self._resolve_phase6_state(
                phase6_result
            )
        )

        test_partition = (
            authorized_test_partition
            or phase6_result.test
        )

        if not isinstance(
            test_partition,
            DatasetPartition,
        ):
            raise PermissionError(
                "Phase 7 requires an authorized "
                "Phase 6 test partition."
            )

        if not self.config.execution.enabled:
            return self._disabled_result(
                phase6_result=phase6_result,
                test_partition=test_partition,
            )

        contract = SelectionContractResolver(
            options=self.config.contract
        ).resolve(
            phase6_result=phase6_result,
            phase6_state=resolved_state,
            authorized_test_partition=(
                test_partition
            ),
        )

        state = Phase7State(
            fitted_on_training_only=True,
            test_transform_authorized=(
                authorized_test_partition is not None
                or isinstance(
                    phase6_result.test,
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
            source_lineage=(
                resolved_state.lineage.model_copy(
                    deep=True
                )
                if resolved_state.lineage is not None
                else None
            ),
            phase6_output_fingerprint=(
                resolved_state.output_fingerprint
            ),
        )

        self.state = state

        initial_result = Phase7Result(
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
            component="selection_plan",
            operation=lambda: SelectionPlanBuilder(
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

        if plan.has_blocked_components:
            initial_result.add_finding(
                Phase7Finding(
                    code="P7-PLAN-001",
                    category=(
                        Phase7FindingCategory.PLAN
                    ),
                    severity=(
                        Phase7FindingSeverity.ERROR
                    ),
                    component="selection_plan",
                    message=(
                        "The selection plan contains "
                        "blocked components."
                    ),
                    recommended_action=(
                        Phase7FindingAction.RECONFIGURE
                    ),
                    applied_action=(
                        Phase7FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                    details={
                        "blocked_components": list(
                            plan.blocked_components
                        ),
                    },
                )
            )

            if self.config.execution.stop_on_plan_failure:
                return self._failed_result(
                    result=initial_result,
                    state=state,
                )

        state.required_features = list(
            plan.required_features
        )

        state.protected_features = list(
            plan.protected_features
        )

        state.prohibited_features = list(
            plan.prohibited_features
        )

        state.configuration_fingerprint = (
            self.fingerprint_manager
            .configuration_fingerprint(
                self.config
            )
        )

        state.selection_plan_fingerprint = (
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

        pipeline = SelectionPipeline(
            options=self.config.pipeline
        )

        self.pipeline = pipeline

        pipeline_output = self._execute(
            component="selection_pipeline",
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

        result = Phase7Result(
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
            source_train=contract.train,
            source_validation=contract.validation,
            source_test=contract.test,
            result=result,
            state=state,
            plan=plan,
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
                source_partitions={
                    PartitionName.TRAIN: (
                        contract.train
                    ),
                    PartitionName.VALIDATION: (
                        contract.validation
                    ),
                    PartitionName.TEST: (
                        contract.test
                    ),
                },
                output_partitions={
                    PartitionName.TRAIN: (
                        result.train
                    ),
                    PartitionName.VALIDATION: (
                        result.validation
                    ),
                    PartitionName.TEST: (
                        result.test
                    ),
                },
                state=state,
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
        )

        Phase7ReadinessEvaluator(
            options=self.config.readiness
        ).evaluate(
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
        source_train: DatasetPartition,
        source_validation: DatasetPartition,
        source_test: DatasetPartition,
        result: Phase7Result,
        state: Phase7State,
        plan: SelectionPlan,
    ) -> None:
        """Run every Phase 7 validator."""

        schema = self._execute(
            component="schema_consistency",
            operation=lambda: SchemaConsistencyValidator(
                options=(
                    self.config
                    .validation
                    .schema_consistency
                )
            ).validate(
                train=result.train,
                validation=result.validation,
                test=result.test,
                selected_features=(
                    state.selected_feature_columns
                ),
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.schema_consistency = (
            schema
        )

        if schema is not None:
            state.schemas_consistent = schema.valid

        feature_count = self._execute(
            component="feature_count",
            operation=lambda: FeatureCountValidator(
                options=(
                    self.config
                    .validation
                    .feature_count
                )
            ).validate(
                source_features=(
                    state.source_feature_columns
                ),
                selected_features=(
                    state.selected_feature_columns
                ),
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.feature_count = (
            feature_count
        )

        if feature_count is not None:
            state.feature_count_valid = (
                feature_count.valid
            )

        protected_features = self._execute(
            component="protected_features",
            operation=lambda: ProtectedFeaturesValidator(
                options=(
                    self.config
                    .validation
                    .protected_features
                )
            ).validate(
                selected_features=(
                    state.selected_feature_columns
                ),
                required_features=(
                    plan.required_features
                ),
                protected_features=(
                    plan.protected_features
                ),
                prohibited_features=(
                    plan.prohibited_features
                ),
                decisions=state.decisions,
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.protected_features = (
            protected_features
        )

        if protected_features is not None:
            state.protected_features_valid = (
                protected_features.valid
            )

        target_integrity = self._execute(
            component="target_integrity",
            operation=lambda: TargetIntegrityValidator(
                options=(
                    self.config
                    .validation
                    .target_integrity
                )
            ).validate(
                source_train=source_train,
                transformed_train=result.train,
                source_validation=(
                    source_validation
                ),
                transformed_validation=(
                    result.validation
                ),
                source_test=source_test,
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
                options=(
                    self.config
                    .validation
                    .row_integrity
                )
            ).validate(
                source_train=source_train,
                transformed_train=result.train,
                source_validation=(
                    source_validation
                ),
                transformed_validation=(
                    result.validation
                ),
                source_test=source_test,
                transformed_test=result.test,
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.row_integrity = (
            row_integrity
        )

    @staticmethod
    def _register_pipeline_state(
        *,
        state: Phase7State,
        pipeline_result: SelectionPipelineResult,
    ) -> None:
        """Register fitted selection state."""

        state.source_feature_columns = list(
            pipeline_result.source_features
        )

        state.selected_feature_columns = list(
            pipeline_result.selected_features
        )

        state.removed_feature_columns = list(
            pipeline_result.removed_features
        )

        state.decisions = {
            column: decision.model_copy(
                deep=True
            )
            for column, decision
            in pipeline_result.decisions.items()
        }

        state.filter_result = (
            pipeline_result
            .filter_result
            .model_copy(
                deep=True
            )
        )

        state.selector_result = (
            pipeline_result
            .selector_result
            .model_copy(
                deep=True
            )
        )

        state.stability_result = (
            pipeline_result
            .stability_result
            .model_copy(
                deep=True
            )
            if pipeline_result
            .stability_result
            is not None
            else None
        )

        state.selected_lineage = (
            pipeline_result
            .selected_lineage
            .model_copy(
                deep=True
            )
            if pipeline_result
            .selected_lineage
            is not None
            else None
        )

        state.fitted_on_training_only = (
            pipeline_result
            .fitted_on_training_only
        )

        state.supervised_target_access_used = (
            pipeline_result
            .supervised_target_access_used
        )

        state.supervised_target_access_authorized = (
            pipeline_result
            .supervised_target_access_authorized
        )

        state.validation_used_for_evaluation_only = (
            pipeline_result
            .validation_used_for_evaluation_only
        )

        state.stability_valid = bool(
            pipeline_result.stability_result is None
            or pipeline_result
            .stability_result
            .valid
        )

    def _register_execution_metadata(
        self,
        *,
        result: Phase7Result,
        state: Phase7State,
        pipeline_result: SelectionPipelineResult,
    ) -> None:
        """Register auditable execution metadata."""

        if not (
            self.config.execution
            .store_execution_metadata
        ):
            return

        selected_position = {
            column: position
            for position, column
            in enumerate(
                state.selected_feature_columns
            )
        }

        feature_metadata: dict[
            str,
            FeatureSelectionMetadata,
        ] = {}

        for source_position, column in enumerate(
            state.source_feature_columns
        ):
            decision = state.decisions[
                column
            ]

            if decision.prohibited:
                status = (
                    FeatureSelectionStatus.PROHIBITED
                )

            elif decision.required:
                status = (
                    FeatureSelectionStatus.REQUIRED
                )

            elif decision.protected:
                status = (
                    FeatureSelectionStatus.PROTECTED
                )

            elif decision.selected:
                status = (
                    FeatureSelectionStatus.SELECTED
                )

            else:
                status = (
                    FeatureSelectionStatus.REMOVED
                )

            selected_by: list[str] = []

            if decision.selected_by_filter_engine:
                selected_by.append(
                    "filter_engine"
                )

            if decision.selected_by_selector_engine:
                selected_by.append(
                    "selector_engine"
                )

            if decision.selected_by_stability_engine:
                selected_by.append(
                    "stability_engine"
                )

            feature_metadata[
                column
            ] = FeatureSelectionMetadata(
                column=column,
                status=status,
                selected=decision.selected,
                required=decision.required,
                protected=decision.protected,
                prohibited=decision.prohibited,
                source_position=source_position,
                selected_position=(
                    selected_position.get(
                        column
                    )
                ),
                selection_score=(
                    decision.selection_score
                ),
                selection_reasons=list(
                    decision.selection_reasons
                ),
                removal_reasons=list(
                    decision.removal_reasons
                ),
                selected_by=selected_by,
            )

        partition_metadata: dict[
            PartitionName,
            PartitionSelectionSummary,
        ] = {}

        for partition_name, partition_result in (
            pipeline_result.partitions.items()
        ):
            selection = partition_result.selection

            partition_metadata[
                partition_name
            ] = PartitionSelectionSummary(
                partition=partition_name,
                row_count=(
                    selection.row_count_after
                ),
                feature_count_before=(
                    selection.feature_count_before
                ),
                feature_count_after=(
                    selection.feature_count_after
                ),
                selected_feature_count=len(
                    selection.selected_columns
                ),
                removed_feature_count=len(
                    selection.removed_columns
                ),
                feature_values_unchanged=(
                    selection
                    .feature_values_unchanged
                ),
                target_unchanged=(
                    selection.target_unchanged
                ),
                row_ids_unchanged=(
                    selection.row_ids_unchanged
                ),
                row_order_unchanged=(
                    selection.row_order_unchanged
                ),
                input_fingerprint=(
                    state
                    .input_partition_fingerprints
                    .get(
                        partition_name
                    )
                ),
                output_fingerprint=(
                    state
                    .output_partition_fingerprints
                    .get(
                        partition_name
                    )
                ),
            )

        result.execution_metadata = (
            Phase7ExecutionMetadata(
                fitted_on_training_only=(
                    state.fitted_on_training_only
                ),
                supervised_target_access_used=(
                    state
                    .supervised_target_access_used
                ),
                supervised_target_access_authorized=(
                    state
                    .supervised_target_access_authorized
                ),
                validation_used_for_evaluation_only=(
                    state
                    .validation_used_for_evaluation_only
                ),
                test_transform_authorized=(
                    state.test_transform_authorized
                ),
                test_partition_remained_protected=(
                    state
                    .test_partition_remained_protected
                ),
                source_feature_count=len(
                    state.source_feature_columns
                ),
                selected_feature_count=len(
                    state.selected_feature_columns
                ),
                removed_feature_count=len(
                    state.removed_feature_columns
                ),
                features=feature_metadata,
                partitions=partition_metadata,
                configuration_fingerprint=(
                    state.configuration_fingerprint
                ),
                selection_plan_fingerprint=(
                    state.selection_plan_fingerprint
                ),
                fitted_state_fingerprint=(
                    state.fitted_state_fingerprint
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
        state: Phase7State,
    ) -> ComponentResult:
        """Execute one component safely."""

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
                    Phase7Finding(
                        code="P7-EXECUTION-001",
                        category=(
                            Phase7FindingCategory
                            .EXECUTION
                        ),
                        severity=(
                            Phase7FindingSeverity
                            .CRITICAL
                        ),
                        component=component,
                        message=(
                            f"Component {component!r} failed "
                            f"with {type(error).__name__}: "
                            f"{error}"
                        ),
                        recommended_action=(
                            Phase7FindingAction.RAISE
                        ),
                        applied_action=(
                            Phase7FindingAction.REPORT
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
        issues: list[SelectionContractIssue],
    ) -> list[object]:
        """Convert contract issues into findings."""

        severity_mapping = {
            "info": Phase7FindingSeverity.INFO,
            "warning": (
                Phase7FindingSeverity.WARNING
            ),
            "error": Phase7FindingSeverity.ERROR,
            "critical": (
                Phase7FindingSeverity.CRITICAL
            ),
        }

        return [
            Phase7Finding(
                code=issue.code,
                category=(
                    Phase7FindingCategory.CONTRACT
                ),
                severity=severity_mapping[
                    issue.severity
                ],
                component="selection_contract",
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
                    Phase7FindingAction.RECONFIGURE
                ),
                applied_action=(
                    Phase7FindingAction.REPORT
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
        result: Phase7Result,
        state: Phase7State,
    ) -> Phase7Result:
        """Finalize a failed Phase 7 result."""

        blocking = result.blocking_findings()

        decision = Phase7ReadinessDecision(
            ready=False,
            status=Phase7Status.FAILED,
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

        state.ready_for_phase_8 = False

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
        phase6_result: Phase6Result,
        test_partition: DatasetPartition,
    ) -> Phase7Result:
        """Build a disabled Phase 7 result."""

        result = Phase7Result(
            train=phase6_result.train,
            validation=phase6_result.validation,
            test=test_partition,
        )

        state = Phase7State(
            fitted_on_training_only=False,
            test_transform_authorized=False,
            test_partition_remained_protected=True,
            source_feature_columns=(
                phase6_result.train.feature_names()
            ),
            selected_feature_columns=(
                phase6_result.train.feature_names()
            ),
        )

        self.result = result
        self.state = state

        result.add_finding(
            Phase7Finding(
                code="P7-EXECUTION-002",
                category=(
                    Phase7FindingCategory.EXECUTION
                ),
                severity=(
                    Phase7FindingSeverity.INFO
                ),
                component=self.phase_name,
                message=(
                    "Phase 7 execution is disabled."
                ),
                recommended_action=(
                    Phase7FindingAction.NONE
                ),
                applied_action=(
                    Phase7FindingAction.NONE
                ),
                blocks_readiness=False,
            )
        )

        result.finalize(
            Phase7ReadinessDecision(
                ready=False,
                status=Phase7Status.DISABLED,
                reasons=[
                    "Phase 7 execution is disabled."
                ],
            )
        )

        return result

    @staticmethod
    def _resolve_phase6_state(
        result: Phase6Result,
    ) -> Phase6State:
        """Resolve Phase 6 state from its result."""

        direct_state = getattr(
            result,
            "state",
            None,
        )

        if isinstance(
            direct_state,
            Phase6State,
        ):
            return direct_state

        if result.phase_state:
            return Phase6State.model_validate(
                result.phase_state
            )

        raise ValueError(
            "Phase 6 state is unavailable."
        )

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
        Phase7Result,
        Phase7State,
    ]:
        """Return the latest result and state."""

        if (
            self.result is None
            or self.state is None
        ):
            raise RuntimeError(
                "Phase 7 has not been executed."
            )

        return (
            self.result,
            self.state,
        )
