"""Main orchestration pipeline for Phase 6."""

from __future__ import annotations

from typing import TypeVar

from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase5.output_governance import (
    Phase5Result,
    Phase5State,
)
from mlprep.phases.phase6.config import (
    Phase6Config,
)
from mlprep.phases.phase6.contract import (
    TransformationContractResolver,
    TransformationPlanBuilder,
)
from mlprep.phases.phase6.orchestration import (
    TransformationPipeline,
)
from mlprep.phases.phase6.output_governance import (
    Phase6Finding,
    Phase6FindingAction,
    Phase6FindingCategory,
    Phase6FindingSeverity,
    Phase6FingerprintManager,
    Phase6ReadinessEvaluator,
    Phase6Result,
    Phase6State,
)
from mlprep.phases.phase6.validation import (
    NumericIntegrityValidator,
    RowIntegrityValidator,
    SchemaConsistencyValidator,
    TargetIntegrityValidator,
    UnknownCategoryValidator,
)


ComponentResult = TypeVar(
    "ComponentResult"
)


class Phase6:
    """Execute feature transformation and encoding."""

    phase_name = "phase_6"

    def __init__(
        self,
        config: Phase6Config | None = None,
    ) -> None:
        """Initialize Phase 6."""

        self.config = (
            config or Phase6Config()
        )

        self.state: Phase6State | None = None
        self.result: Phase6Result | None = None

        self.fingerprint_manager = (
            Phase6FingerprintManager()
        )

    def run(
        self,
        *,
        phase5_result: Phase5Result,
        phase5_state: Phase5State | None = None,
        authorized_test_partition: DatasetPartition | None = None,
    ) -> Phase6Result:
        """Execute the complete Phase 6 workflow."""

        resolved_state = (
            phase5_state
            or self._resolve_phase5_state(
                phase5_result
            )
        )

        contract = TransformationContractResolver(
            self.config.contract
        ).resolve(
            phase5_result=phase5_result,
            phase5_state=resolved_state,
            authorized_test_partition=(
                authorized_test_partition
            ),
        )

        contract.assert_ready()

        state = Phase6State(
            fitted_on_training_only=True,
            test_transform_authorized=(
                authorized_test_partition is not None
                or isinstance(
                    phase5_result.test,
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
            phase5_output_fingerprint=(
                resolved_state.output_fingerprint
            ),
        )

        self.state = state

        initial_result = Phase6Result(
            train=contract.train,
            validation=contract.validation,
            test=contract.test,
        )

        self.result = initial_result

        plan = TransformationPlanBuilder(
            self.config.plan
        ).build(
            contract=contract
        )

        if plan.blocked_columns:
            initial_result.add_finding(
                Phase6Finding(
                    code="P6-PLAN-001",
                    category=(
                        Phase6FindingCategory.PLAN
                    ),
                    severity=(
                        Phase6FindingSeverity.ERROR
                    ),
                    component="transformation_plan",
                    message=(
                        "The transformation plan contains "
                        "blocked entries."
                    ),
                    columns=list(
                        plan.blocked_columns
                    ),
                    recommended_action=(
                        Phase6FindingAction.RECONFIGURE
                    ),
                    applied_action=(
                        Phase6FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

            if (
                self.config.execution
                .stop_on_plan_failure
            ):
                raise ValueError(
                    "The transformation plan contains "
                    "blocked entries."
                )

        state.configuration_fingerprint = (
            self.fingerprint_manager
            .configuration_fingerprint(
                self.config
            )
        )

        state.transformation_plan_fingerprint = (
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

        pipeline = TransformationPipeline(
            self.config.pipeline
        )

        (
            transformed_train,
            transformed_validation,
            transformed_test,
            pipeline_result,
        ) = pipeline.fit_transform(
            contract=contract,
            plan=plan,
        )

        result = Phase6Result(
            train=transformed_train,
            validation=transformed_validation,
            test=transformed_test,
        )

        result.add_findings(
            initial_result.findings
        )

        self.result = result

        state.output_feature_columns = list(
            pipeline_result.output_columns
        )

        state.fitted_on_training_only = (
            pipeline_result
            .fitted_on_training_only
        )

        state.lineage = (
            pipeline_result.lineage.model_copy(
                deep=True
            )
        )

        state.timedelta_output_columns = dict(
            pipeline_result
            .timedelta_output_columns
        )

        for column, value in (
            pipeline_result.numeric_states.items()
        ):
            state.register_numeric_state(
                column=column,
                state=value,
            )

        for column, value in (
            pipeline_result
            .categorical_states
            .items()
        ):
            state.register_categorical_state(
                column=column,
                state=value,
            )

        state.boolean_states = {
            column: value.model_copy(
                deep=True
            )
            for column, value
            in pipeline_result
            .boolean_states
            .items()
        }

        state.datetime_states = {
            column: value.model_copy(
                deep=True
            )
            for column, value
            in pipeline_result
            .datetime_states
            .items()
        }

        state.text_states = {
            column: value.model_copy(
                deep=True
            )
            for column, value
            in pipeline_result.text_states.items()
        }

        schema_result = SchemaConsistencyValidator(
            self.config
            .validation
            .schema_consistency
        ).validate(
            train=result.train,
            validation=result.validation,
            test=result.test,
            expected_output_columns=(
                state.output_feature_columns
            ),
        )

        result.validation_results.schema_consistency = (
            schema_result
        )

        state.schemas_consistent = (
            schema_result.valid
        )

        numeric_result = NumericIntegrityValidator(
            self.config
            .validation
            .numeric_integrity
        ).validate(
            train=result.train,
            validation=result.validation,
            test=result.test,
        )

        result.validation_results.numeric_integrity = (
            numeric_result
        )

        state.numeric_integrity_valid = (
            numeric_result.valid
        )

        state.total_missing_output_count = (
            numeric_result.total_missing_count
        )

        state.total_nonfinite_output_count = (
            numeric_result.total_nonfinite_count
        )

        unknown_result = UnknownCategoryValidator(
            self.config
            .validation
            .unknown_categories
        ).validate(
            pipeline_result=pipeline_result
        )

        result.validation_results.unknown_categories = (
            unknown_result
        )

        state.unknown_categories_valid = (
            unknown_result.valid
        )

        state.total_unknown_category_count = (
            unknown_result.total_unknown_count
        )

        target_result = TargetIntegrityValidator(
            self.config
            .validation
            .target_integrity
        ).validate(
            source_train=contract.train,
            transformed_train=result.train,
            source_validation=contract.validation,
            transformed_validation=result.validation,
            source_test=contract.test,
            transformed_test=result.test,
        )

        result.validation_results.target_integrity = (
            target_result
        )

        row_result = RowIntegrityValidator(
            self.config
            .validation
            .row_integrity
        ).validate(
            source_train=contract.train,
            transformed_train=result.train,
            source_validation=contract.validation,
            transformed_validation=result.validation,
            source_test=contract.test,
            transformed_test=result.test,
        )

        result.validation_results.row_integrity = (
            row_result
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

        Phase6ReadinessEvaluator(
            self.config.readiness
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

    @staticmethod
    def _resolve_phase5_state(
        result: Phase5Result,
    ) -> Phase5State:
        """Resolve Phase 5 state."""

        direct_state = getattr(
            result,
            "state",
            None,
        )

        if isinstance(
            direct_state,
            Phase5State,
        ):
            return direct_state

        if result.phase_state:
            return Phase5State.model_validate(
                result.phase_state
            )

        raise ValueError(
            "Phase 5 state is unavailable."
        )