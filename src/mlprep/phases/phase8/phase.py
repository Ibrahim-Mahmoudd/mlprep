"""Main orchestration pipeline for Phase 8."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase7.output_governance import (
    Phase7Result,
    Phase7State,
)
from mlprep.phases.phase8.config import (
    Phase8Config,
)
from mlprep.phases.phase8.contract import (
    BalancingContractIssue,
    BalancingContractResolver,
    BalancingPlan,
    BalancingPlanBuilder,
)
from mlprep.phases.phase8.lineage import (
    SyntheticIdGeneratorState,
)
from mlprep.phases.phase8.orchestration import (
    BalancingPipeline,
    BalancingPipelineResult,
)
from mlprep.phases.phase8.output_governance import (
    PartitionBalancingSummary,
    Phase8ExecutionMetadata,
    Phase8Finding,
    Phase8FindingAction,
    Phase8FindingCategory,
    Phase8FindingSeverity,
    Phase8FingerprintManager,
    Phase8ReadinessDecision,
    Phase8ReadinessEvaluator,
    Phase8ReportBuilder,
    Phase8Result,
    Phase8State,
    Phase8Status,
    TrainingRowMetadata,
    TrainingRowStatus,
)
from mlprep.phases.phase8.samplers import (
    ResampledRowType,
)
from mlprep.phases.phase8.validation import (
    DistributionValidator,
    PartitionIntegrityValidator,
    SchemaConsistencyValidator,
    SyntheticIntegrityValidator,
    TargetIntegrityValidator,
)

ComponentResult = TypeVar(
    "ComponentResult"
)


class Phase8:
    """Execute training-only balancing and governance."""

    phase_name = "phase_8"

    def __init__(
        self,
        config: Phase8Config | None = None,
    ) -> None:
        """Initialize Phase 8."""

        self.config = (
            config
            or Phase8Config()
        )

        if not isinstance(
            self.config,
            Phase8Config,
        ):
            raise TypeError(
                "config must be a Phase8Config."
            )

        self.state: Phase8State | None = None
        self.result: Phase8Result | None = None

        self.plan: BalancingPlan | None = None
        self.pipeline: BalancingPipeline | None = None

        self.fingerprint_manager = (
            Phase8FingerprintManager()
        )

        self.report_builder = Phase8ReportBuilder(
            options=self.config.reporting
        )

    def run(
        self,
        *,
        phase7_result: Phase7Result,
        phase7_state: Phase7State | None = None,
        authorized_test_partition: DatasetPartition | None = None,
    ) -> Phase8Result:
        """Execute the complete Phase 8 workflow."""

        if not isinstance(
            phase7_result,
            Phase7Result,
        ):
            raise TypeError(
                "phase7_result must be a Phase7Result."
            )

        resolved_phase7_state = (
            phase7_state
            or self._resolve_phase7_state(
                phase7_result
            )
        )

        test_partition = (
            authorized_test_partition
            if authorized_test_partition is not None
            else phase7_result.test
        )

        if not isinstance(
            test_partition,
            DatasetPartition,
        ):
            raise PermissionError(
                "Phase 8 requires an authorized Phase 7 "
                "test partition."
            )

        if not self.config.execution.enabled:
            return self._disabled_result(
                phase7_result=phase7_result,
                test_partition=test_partition,
                phase7_state=resolved_phase7_state,
            )

        contract = BalancingContractResolver(
            options=self.config.contract
        ).resolve(
            phase7_result=phase7_result,
            phase7_state=resolved_phase7_state,
            authorized_test_partition=test_partition,
        )

        state = Phase8State(
            fitted_on_training_only=True,
            source_training_row_count=(
                contract.train.row_count()
            ),
            source_feature_columns=(
                contract.feature_columns
            ),
            source_feature_dtypes={
                str(column): str(
                    contract.train.X[
                        column
                    ].dtype
                )
                for column
                in contract.train.X.columns
            },
            class_identity_mapping=(
                contract.class_identity_mapping.model_copy(
                    deep=True
                )
            ),
            selected_feature_lineage=(
                resolved_phase7_state
                .selected_lineage
                .model_copy(
                    deep=True
                )
                if resolved_phase7_state
                .selected_lineage
                is not None
                else None
            ),
            phase7_output_fingerprint=(
                resolved_phase7_state
                .output_fingerprint
            ),
        )

        self.state = state

        initial_result = Phase8Result(
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
            component="balancing_plan",
            operation=lambda: BalancingPlanBuilder(
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
                Phase8Finding(
                    code="P8-PLAN-001",
                    category=(
                        Phase8FindingCategory.PLAN
                    ),
                    severity=(
                        Phase8FindingSeverity.ERROR
                    ),
                    component="balancing_plan",
                    message=(
                        "The balancing plan contains "
                        "blocked components."
                    ),
                    recommended_action=(
                        Phase8FindingAction.RECONFIGURE
                    ),
                    applied_action=(
                        Phase8FindingAction.REPORT
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

        for warning in plan.warnings:
            initial_result.add_finding(
                Phase8Finding(
                    code="P8-PLAN-002",
                    category=(
                        Phase8FindingCategory.PLAN
                    ),
                    severity=(
                        Phase8FindingSeverity.WARNING
                    ),
                    component="balancing_plan",
                    message=warning,
                    recommended_action=(
                        Phase8FindingAction.REVIEW
                    ),
                    applied_action=(
                        Phase8FindingAction.REPORT
                    ),
                    blocks_readiness=False,
                )
            )

        state.configured_strategy = (
            plan.sampling.strategy.value
        )

        state.target_access_authorized = (
            plan.target_access_authorized
        )

        state.resampling_authorized = (
            plan.resampling_authorized
        )

        state.synthetic_generation_authorized = (
            plan.synthetic_generation_authorized
        )

        state.configuration_fingerprint = (
            self.fingerprint_manager
            .configuration_fingerprint(
                self.config
            )
        )

        state.balancing_plan_fingerprint = (
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

        pipeline = BalancingPipeline(
            options=self.config.pipeline
        )

        self.pipeline = pipeline

        pipeline_output = self._execute(
            component="balancing_pipeline",
            operation=lambda: pipeline.fit_resample(
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
            output_train,
            output_validation,
            output_test,
            pipeline_result,
        ) = pipeline_output

        result = Phase8Result(
            train=output_train,
            validation=output_validation,
            test=output_test,
        )

        result.add_findings(
            initial_result.findings
        )

        self.result = result

        self._register_pipeline_state(
            state=state,
            pipeline_result=pipeline_result,
        )

        if pipeline_result.class_weights is not None:
            result.class_weights = dict(
                pipeline_result
                .class_weights
                .class_weights
            )

        if pipeline_result.sample_weights is not None:
            result.sample_weights = (
                pipeline_result
                .sample_weights
                .weights
                .copy(
                    deep=True
                )
            )

        self._run_validations(
            source_train=contract.train,
            source_validation=contract.validation,
            source_test=contract.test,
            result=result,
            state=state,
            pipeline_result=pipeline_result,
        )

        self.fingerprint_manager.capture_outputs(
            train=result.train,
            validation=result.validation,
            test=result.test,
            state=state,
        )

        result.add_findings(
            self.fingerprint_manager
            .verify_protected_partitions(
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
        )

        Phase8ReadinessEvaluator(
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
                self.fingerprint_manager.serialize_model(
                    state
                )
            )

        return result

    def _run_validations(
        self,
        *,
        source_train: DatasetPartition,
        source_validation: DatasetPartition,
        source_test: DatasetPartition,
        result: Phase8Result,
        state: Phase8State,
        pipeline_result: BalancingPipelineResult,
    ) -> None:
        """Run all configured Phase 8 validators."""

        distribution = self._execute(
            component="distribution_validation",
            operation=lambda: DistributionValidator(
                options=(
                    self.config
                    .validation
                    .distribution
                )
            ).validate(
                source_distribution=(
                    pipeline_result
                    .distribution_before
                ),
                output_distribution=(
                    pipeline_result
                    .distribution_after
                ),
                output_metrics=(
                    pipeline_result.metrics_after
                ),
                configured_strategy=(
                    pipeline_result
                    .configured_strategy
                ),
                executed_strategy=(
                    pipeline_result
                    .executed_strategy
                ),
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.distribution = (
            distribution
        )

        if distribution is not None:
            state.distribution_valid = (
                distribution.valid
            )

            self._validation_findings(
                result=result,
                component="distribution_validation",
                category=(
                    Phase8FindingCategory.DISTRIBUTION
                ),
                code="P8-VALIDATION-001",
                valid=distribution.valid,
                reasons=(
                    distribution.blocking_reasons
                ),
                warnings=distribution.warnings,
            )

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
                expected_columns=(
                    state.source_feature_columns
                ),
                source_dtypes=(
                    state.source_feature_dtypes
                ),
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.schema_consistency = (
            schema
        )

        if schema is not None:
            state.schema_consistent = (
                schema.valid
            )

            self._validation_findings(
                result=result,
                component="schema_consistency",
                category=(
                    Phase8FindingCategory.SCHEMA
                ),
                code="P8-VALIDATION-002",
                valid=schema.valid,
                reasons=(
                    schema.blocking_reasons
                ),
            )

        source_class_values = [
            entry.class_value
            for entry
            in pipeline_result
            .distribution_before
            .classes
            .values()
        ]

        synthetic = self._execute(
            component="synthetic_integrity",
            operation=lambda: SyntheticIntegrityValidator(
                options=(
                    self.config
                    .validation
                    .synthetic_integrity
                )
            ).validate(
                source_train=source_train,
                output_train=result.train,
                lineage=(
                    pipeline_result.row_lineage
                ),
                valid_class_values=(
                    source_class_values
                ),
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.synthetic_integrity = (
            synthetic
        )

        if synthetic is not None:
            state.synthetic_integrity_valid = (
                synthetic.valid
            )

            self._validation_findings(
                result=result,
                component="synthetic_integrity",
                category=(
                    Phase8FindingCategory
                    .SYNTHETIC_INTEGRITY
                ),
                code="P8-VALIDATION-003",
                valid=synthetic.valid,
                reasons=(
                    synthetic.blocking_reasons
                ),
            )

        target = self._execute(
            component="target_integrity",
            operation=lambda: TargetIntegrityValidator(
                options=(
                    self.config
                    .validation
                    .target_integrity
                )
            ).validate(
                source_train=source_train,
                output_train=result.train,
                source_validation=(
                    source_validation
                ),
                output_validation=(
                    result.validation
                ),
                source_test=source_test,
                output_test=result.test,
                lineage=(
                    pipeline_result.row_lineage
                ),
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.target_integrity = (
            target
        )

        if target is not None:
            state.target_integrity_valid = (
                target.valid
            )

            self._validation_findings(
                result=result,
                component="target_integrity",
                category=(
                    Phase8FindingCategory
                    .TARGET_INTEGRITY
                ),
                code="P8-VALIDATION-004",
                valid=target.valid,
                reasons=(
                    target.blocking_reasons
                ),
            )

        partition = self._execute(
            component="partition_integrity",
            operation=lambda: PartitionIntegrityValidator(
                options=(
                    self.config
                    .validation
                    .partition_integrity
                )
            ).validate(
                train=result.train,
                validation=result.validation,
                test=result.test,
                lineage=(
                    pipeline_result.row_lineage
                ),
                validation_guard=(
                    pipeline_result
                    .validation_guard
                ),
                test_guard=(
                    pipeline_result.test_guard
                ),
            ),
            fallback=None,
            state=state,
        )

        result.validation_results.partition_integrity = (
            partition
        )

        if partition is not None:
            state.partition_integrity_valid = (
                partition.valid
            )

            self._validation_findings(
                result=result,
                component="partition_integrity",
                category=(
                    Phase8FindingCategory
                    .PARTITION_INTEGRITY
                ),
                code="P8-VALIDATION-005",
                valid=partition.valid,
                reasons=(
                    partition.blocking_reasons
                ),
            )

    @staticmethod
    def _register_pipeline_state(
        *,
        state: Phase8State,
        pipeline_result: BalancingPipelineResult,
    ) -> None:
        """Register fitted balancing state."""

        state.fitted_on_training_only = (
            pipeline_result
            .fitted_on_training_only
        )

        state.executed_strategy = (
            pipeline_result
            .executed_strategy
            .value
        )

        state.output_training_row_count = (
            pipeline_result
            .sampler_state
            .output_row_count
        )

        state.source_class_distribution = (
            pipeline_result
            .distribution_before
            .model_copy(
                deep=True
            )
        )

        state.output_class_distribution = (
            pipeline_result
            .distribution_after
            .model_copy(
                deep=True
            )
        )

        state.imbalance_metrics_before = (
            pipeline_result
            .metrics_before
            .model_copy(
                deep=True
            )
        )

        state.imbalance_metrics_after = (
            pipeline_result
            .metrics_after
            .model_copy(
                deep=True
            )
        )

        state.strategy_recommendation = (
            pipeline_result
            .recommendation
            .model_copy(
                deep=True
            )
        )

        state.sampler_state = (
            pipeline_result
            .sampler_state
            .model_copy(
                deep=True
            )
        )
        synthetic_id_state = (
            pipeline_result
            .sampler_state
            .metadata
            .get("synthetic_id_state")
        )
        if synthetic_id_state is not None:
            state.synthetic_id_state = (
                SyntheticIdGeneratorState.model_validate(
                    synthetic_id_state
                )
            )

        state.class_weights = (
            pipeline_result
            .class_weights
            .model_copy(
                deep=True
            )
            if pipeline_result
            .class_weights
            is not None
            else None
        )

        state.sample_weights = (
            pipeline_result
            .sample_weights
            .model_copy(
                deep=True
            )
            if pipeline_result
            .sample_weights
            is not None
            else None
        )

        state.row_lineage = (
            pipeline_result
            .row_lineage
            .model_copy(
                deep=True
            )
        )

        state.retained_original_row_ids = list(
            pipeline_result
            .row_lineage
            .retained_original_row_ids
        )

        state.duplicate_row_ids = list(
            pipeline_result
            .row_lineage
            .duplicated_row_ids
        )

        state.synthetic_row_ids = list(
            pipeline_result
            .row_lineage
            .synthetic_row_ids
        )

        state.removed_original_row_ids = list(
            pipeline_result
            .row_lineage
            .removed_original_row_ids
        )

        state.validation_resampled = (
            pipeline_result
            .validation_resampled
        )

        state.test_resampled = (
            pipeline_result
            .test_resampled
        )

        state.validation_remained_protected = (
            pipeline_result
            .validation_remained_protected
        )

        state.test_remained_protected = (
            pipeline_result
            .test_remained_protected
        )

        state.target_access_authorized = (
            pipeline_result
            .target_access_authorized
        )

        state.resampling_authorized = (
            pipeline_result
            .resampling_authorized
        )

        state.synthetic_generation_authorized = (
            pipeline_result
            .synthetic_generation_authorized
        )

        state.lineage_complete = bool(
            pipeline_result
            .row_lineage
            .complete_output_coverage
            and pipeline_result
            .row_lineage
            .complete_removed_row_coverage
        )

        state.class_weights_valid = (
            Phase8._class_weights_valid(
                pipeline_result
            )
        )

        state.sample_weights_valid = (
            Phase8._sample_weights_valid(
                pipeline_result
            )
        )

    @staticmethod
    def _class_weights_valid(
        pipeline_result: BalancingPipelineResult,
    ) -> bool:
        """Return whether fitted class weights are valid."""

        class_weights = (
            pipeline_result.class_weights
        )

        if class_weights is None:
            return True

        return bool(
            class_weights.enabled
            and class_weights.class_weights
            and all(
                np.isfinite(
                    weight
                )
                and weight > 0.0
                for weight
                in class_weights
                .class_weights
                .values()
            )
        )

    @staticmethod
    def _sample_weights_valid(
        pipeline_result: BalancingPipelineResult,
    ) -> bool:
        """Return whether fitted sample weights are valid."""

        sample_weights = (
            pipeline_result.sample_weights
        )

        if sample_weights is None:
            return True

        weights = (
            sample_weights.weights
        )

        return bool(
            sample_weights.enabled
            and len(
                weights
            )
            == pipeline_result
            .sampler_state
            .output_row_count
            and not weights.isna().any()
            and np.isfinite(
                weights.to_numpy(
                    dtype=float
                )
            ).all()
            and (
                weights
                > 0.0
            ).all()
        )

    def _register_execution_metadata(
        self,
        *,
        result: Phase8Result,
        state: Phase8State,
        pipeline_result: BalancingPipelineResult,
    ) -> None:
        """Register auditable execution metadata."""

        if not (
            self.config.execution
            .store_execution_metadata
        ):
            return

        sample_weight_mapping: dict[
            int,
            float,
        ] = {}

        if (
            pipeline_result.sample_weights
            is not None
        ):
            sample_weight_mapping = {
                entry.output_position: (
                    entry.weight
                )
                for entry
                in pipeline_result
                .sample_weights
                .entries
            }

        row_metadata: list[
            TrainingRowMetadata
        ] = []

        for entry in (
            pipeline_result
            .row_lineage
            .entries
        ):
            if (
                entry.row_type
                == ResampledRowType.SYNTHETIC
            ):
                status = (
                    TrainingRowStatus.SYNTHETIC
                )

            elif (
                entry.row_type
                == ResampledRowType.DUPLICATE
            ):
                status = (
                    TrainingRowStatus.DUPLICATED
                )

            elif (
                entry.row_type
                == ResampledRowType
                .RETAINED_AFTER_CLEANING
            ):
                status = (
                    TrainingRowStatus.RETAINED
                )

            else:
                status = (
                    TrainingRowStatus.ORIGINAL
                )

            row_metadata.append(
                TrainingRowMetadata(
                    output_position=(
                        entry.output_position
                    ),
                    output_row_id=(
                        entry.output_row_id
                    ),
                    status=status,
                    class_value=(
                        entry.class_value
                    ),
                    source_row_ids=list(
                        entry.source_row_ids
                    ),
                    source_positions=list(
                        entry.source_positions
                    ),
                    synthetic=entry.synthetic,
                    duplicated=entry.duplicated,
                    original=entry.original,
                    sample_weight=(
                        sample_weight_mapping.get(
                            entry.output_position
                        )
                    ),
                )
            )

        partitions = {
            PartitionName.TRAIN: (
                PartitionBalancingSummary(
                    partition=PartitionName.TRAIN,
                    row_count_before=(
                        state
                        .source_training_row_count
                    ),
                    row_count_after=(
                        state
                        .output_training_row_count
                    ),
                    feature_count_before=len(
                        state.source_feature_columns
                    ),
                    feature_count_after=(
                        result.train.feature_count()
                    ),
                    resampled=bool(
                        state.configured_strategy
                        not in {
                            "none",
                            "class_weights_only",
                        }
                    ),
                    protected=False,
                    feature_values_unchanged=bool(
                        not state.synthetic_row_ids
                        and not state.duplicate_row_ids
                        and not state
                        .removed_original_row_ids
                    ),
                    target_unchanged=bool(
                        state
                        .source_training_row_count
                        == state
                        .output_training_row_count
                        and state.configured_strategy
                        in {
                            "none",
                            "class_weights_only",
                        }
                    ),
                    row_ids_unchanged=bool(
                        not state.synthetic_row_ids
                        and not state.duplicate_row_ids
                        and not state
                        .removed_original_row_ids
                    ),
                    row_order_unchanged=bool(
                        state.configured_strategy
                        in {
                            "none",
                            "class_weights_only",
                        }
                    ),
                    input_fingerprint=(
                        state
                        .input_partition_fingerprints
                        .get(
                            PartitionName.TRAIN
                        )
                    ),
                    output_fingerprint=(
                        state
                        .output_partition_fingerprints
                        .get(
                            PartitionName.TRAIN
                        )
                    ),
                )
            ),
            PartitionName.VALIDATION: (
                PartitionBalancingSummary(
                    partition=(
                        PartitionName.VALIDATION
                    ),
                    row_count_before=(
                        result.validation.row_count()
                    ),
                    row_count_after=(
                        result.validation.row_count()
                    ),
                    feature_count_before=(
                        result.validation
                        .feature_count()
                    ),
                    feature_count_after=(
                        result.validation
                        .feature_count()
                    ),
                    resampled=False,
                    protected=True,
                    feature_values_unchanged=(
                        state
                        .validation_remained_protected
                    ),
                    target_unchanged=(
                        state
                        .validation_remained_protected
                    ),
                    row_ids_unchanged=(
                        state
                        .validation_remained_protected
                    ),
                    row_order_unchanged=(
                        state
                        .validation_remained_protected
                    ),
                    input_fingerprint=(
                        state
                        .input_partition_fingerprints
                        .get(
                            PartitionName.VALIDATION
                        )
                    ),
                    output_fingerprint=(
                        state
                        .output_partition_fingerprints
                        .get(
                            PartitionName.VALIDATION
                        )
                    ),
                )
            ),
            PartitionName.TEST: (
                PartitionBalancingSummary(
                    partition=PartitionName.TEST,
                    row_count_before=(
                        result.test.row_count()
                    ),
                    row_count_after=(
                        result.test.row_count()
                    ),
                    feature_count_before=(
                        result.test.feature_count()
                    ),
                    feature_count_after=(
                        result.test.feature_count()
                    ),
                    resampled=False,
                    protected=True,
                    feature_values_unchanged=(
                        state
                        .test_remained_protected
                    ),
                    target_unchanged=(
                        state
                        .test_remained_protected
                    ),
                    row_ids_unchanged=(
                        state
                        .test_remained_protected
                    ),
                    row_order_unchanged=(
                        state
                        .test_remained_protected
                    ),
                    input_fingerprint=(
                        state
                        .input_partition_fingerprints
                        .get(
                            PartitionName.TEST
                        )
                    ),
                    output_fingerprint=(
                        state
                        .output_partition_fingerprints
                        .get(
                            PartitionName.TEST
                        )
                    ),
                )
            ),
        }

        result.execution_metadata = (
            Phase8ExecutionMetadata(
                execution_id=(
                    pipeline_result
                    .metadata
                    .get(
                        "execution_id",
                        "",
                    )
                    or Phase8ExecutionMetadata()
                    .execution_id
                ),
                fitted_on_training_only=(
                    state.fitted_on_training_only
                ),
                target_access_authorized=(
                    state
                    .target_access_authorized
                ),
                resampling_authorized=(
                    state
                    .resampling_authorized
                ),
                synthetic_generation_authorized=(
                    state
                    .synthetic_generation_authorized
                ),
                configured_strategy=(
                    state.configured_strategy
                ),
                executed_strategy=(
                    state.executed_strategy
                ),
                source_training_row_count=(
                    state
                    .source_training_row_count
                ),
                output_training_row_count=(
                    state
                    .output_training_row_count
                ),
                original_output_count=(
                    pipeline_result
                    .sampler_state
                    .original_output_count
                ),
                duplicate_output_count=(
                    pipeline_result
                    .sampler_state
                    .duplicate_output_count
                ),
                synthetic_output_count=(
                    pipeline_result
                    .sampler_state
                    .synthetic_output_count
                ),
                removed_original_count=(
                    pipeline_result
                    .sampler_state
                    .removed_original_count
                ),
                validation_resampled=(
                    state.validation_resampled
                ),
                test_resampled=(
                    state.test_resampled
                ),
                validation_remained_protected=(
                    state
                    .validation_remained_protected
                ),
                test_remained_protected=(
                    state
                    .test_remained_protected
                ),
                rows=row_metadata,
                partitions=partitions,
                configuration_fingerprint=(
                    state.configuration_fingerprint
                ),
                balancing_plan_fingerprint=(
                    state.balancing_plan_fingerprint
                ),
                fitted_state_fingerprint=(
                    state.fitted_state_fingerprint
                ),
                output_fingerprint=(
                    state.output_fingerprint
                ),
            )
        )

    def _validation_findings(
        self,
        *,
        result: Phase8Result,
        component: str,
        category: Phase8FindingCategory,
        code: str,
        valid: bool,
        reasons: list[str],
        warnings: list[str] | None = None,
    ) -> None:
        """Convert validation outcomes into findings."""

        if not (
            self.config.execution
            .generate_validation_findings
        ):
            return

        if not valid:
            result.add_finding(
                Phase8Finding(
                    code=code,
                    category=category,
                    severity=(
                        Phase8FindingSeverity.ERROR
                    ),
                    component=component,
                    message=(
                        reasons[
                            0
                        ]
                        if reasons
                        else (
                            f"Component {component!r} "
                            "failed validation."
                        )
                    ),
                    recommended_action=(
                        Phase8FindingAction.REVIEW
                    ),
                    applied_action=(
                        Phase8FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                    details={
                        "reasons": list(
                            reasons
                        ),
                    },
                )
            )

        for warning in warnings or []:
            result.add_finding(
                Phase8Finding(
                    code=f"{code}-WARNING",
                    category=category,
                    severity=(
                        Phase8FindingSeverity.WARNING
                    ),
                    component=component,
                    message=warning,
                    recommended_action=(
                        Phase8FindingAction.REVIEW
                    ),
                    applied_action=(
                        Phase8FindingAction.REPORT
                    ),
                    blocks_readiness=False,
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
        state: Phase8State,
    ) -> ComponentResult:
        """Execute one Phase 8 component safely."""

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
                    Phase8Finding(
                        code="P8-EXECUTION-001",
                        category=(
                            Phase8FindingCategory
                            .EXECUTION
                        ),
                        severity=(
                            Phase8FindingSeverity
                            .CRITICAL
                        ),
                        component=component,
                        message=(
                            f"Component {component!r} failed "
                            f"with {type(error).__name__}: "
                            f"{error}"
                        ),
                        recommended_action=(
                            Phase8FindingAction.RAISE
                        ),
                        applied_action=(
                            Phase8FindingAction.REPORT
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
        issues: list[BalancingContractIssue],
    ) -> list[Phase8Finding]:
        """Convert contract issues into findings."""

        severity_mapping = {
            "info": Phase8FindingSeverity.INFO,
            "warning": (
                Phase8FindingSeverity.WARNING
            ),
            "error": Phase8FindingSeverity.ERROR,
            "critical": (
                Phase8FindingSeverity.CRITICAL
            ),
        }

        return [
            Phase8Finding(
                code=issue.code,
                category=(
                    Phase8FindingCategory.CONTRACT
                ),
                severity=severity_mapping[
                    issue.severity
                ],
                component="balancing_contract",
                message=issue.message,
                columns=list(
                    issue.columns
                ),
                partitions=[
                    partition.value
                    for partition
                    in issue.partitions
                ],
                classes=list(
                    issue.classes
                ),
                recommended_action=(
                    Phase8FindingAction.RECONFIGURE
                ),
                applied_action=(
                    Phase8FindingAction.REPORT
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
        result: Phase8Result,
        state: Phase8State,
    ) -> Phase8Result:
        """Finalize a failed Phase 8 result."""

        blocking = result.blocking_findings()

        decision = Phase8ReadinessDecision(
            ready=False,
            status=Phase8Status.FAILED,
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

        state.ready_for_phase_9 = False

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
        phase7_result: Phase7Result,
        test_partition: DatasetPartition,
        phase7_state: Phase7State,
    ) -> Phase8Result:
        """Build a disabled Phase 8 result."""

        result = Phase8Result(
            train=phase7_result.train,
            validation=phase7_result.validation,
            test=test_partition,
        )

        state = Phase8State(
            fitted_on_training_only=False,
            configured_strategy="none",
            executed_strategy="none",
            source_training_row_count=(
                phase7_result.train.row_count()
            ),
            output_training_row_count=(
                phase7_result.train.row_count()
            ),
            source_feature_columns=(
                phase7_result.train.feature_names()
            ),
            selected_feature_lineage=(
                phase7_state
                .selected_lineage
                .model_copy(
                    deep=True
                )
                if phase7_state
                .selected_lineage
                is not None
                else None
            ),
            phase7_output_fingerprint=(
                phase7_state.output_fingerprint
            ),
            validation_remained_protected=True,
            test_remained_protected=True,
        )

        self.result = result
        self.state = state

        result.add_finding(
            Phase8Finding(
                code="P8-EXECUTION-002",
                category=(
                    Phase8FindingCategory.EXECUTION
                ),
                severity=(
                    Phase8FindingSeverity.INFO
                ),
                component=self.phase_name,
                message=(
                    "Phase 8 execution is disabled."
                ),
                recommended_action=(
                    Phase8FindingAction.NONE
                ),
                applied_action=(
                    Phase8FindingAction.NONE
                ),
                blocks_readiness=False,
            )
        )

        result.finalize(
            Phase8ReadinessDecision(
                ready=False,
                status=Phase8Status.DISABLED,
                reasons=[
                    "Phase 8 execution is disabled."
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

    @staticmethod
    def _resolve_phase7_state(
        result: Phase7Result,
    ) -> Phase7State:
        """Resolve Phase 7 state from its result."""

        direct_state = getattr(
            result,
            "state",
            None,
        )

        if isinstance(
            direct_state,
            Phase7State,
        ):
            return direct_state

        if result.phase_state:
            return Phase7State.model_validate(
                result.phase_state
            )

        raise ValueError(
            "Phase 7 state is unavailable."
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
        Phase8Result,
        Phase8State,
    ]:
        """Return the latest Phase 8 result and state."""

        if (
            self.result is None
            or self.state is None
        ):
            raise RuntimeError(
                "Phase 8 has not been executed."
            )

        return (
            self.result,
            self.state,
        )
