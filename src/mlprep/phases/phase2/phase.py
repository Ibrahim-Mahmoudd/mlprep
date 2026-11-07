"""Phase 2 orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.phases.phase2.config import Phase2Config
from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
    SplitContractResolver,
)
from mlprep.phases.phase2.contract.split_roles import (
    SplitRoleResolver,
)
from mlprep.phases.phase2.leakage_firewall.firewall import (
    LeakageFirewall,
)
from mlprep.phases.phase2.output_governance.fingerprint import (
    Phase2FingerprintGenerator,
)
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    Phase2Status,
    SplitExecutionMetadata,
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.readiness import (
    Phase2ReadinessEvaluator,
)
from mlprep.phases.phase2.output_governance.report import (
    Phase2ReportBuilder,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    LockedTestPartition,
    Phase2ReadinessDecision,
    Phase2Result,
    SplitFinding,
)
from mlprep.phases.phase2.output_governance.state import (
    Phase2State,
    SplitColumnState,
)
from mlprep.phases.phase2.output_governance.test_lock import (
    TestPartitionLocker,
)
from mlprep.phases.phase2.strategies.base import (
    BaseSplitStrategy,
    StrategyOutput,
)
from mlprep.phases.phase2.strategies.group_split import (
    GroupSplitStrategy,
)
from mlprep.phases.phase2.strategies.group_stratified_split import (
    GroupStratifiedSplitStrategy,
)
from mlprep.phases.phase2.strategies.predefined_split import (
    PredefinedSplitStrategy,
)
from mlprep.phases.phase2.strategies.purged_time_split import (
    PurgedTimeSplitStrategy,
)
from mlprep.phases.phase2.strategies.random_split import (
    RandomSplitStrategy,
)
from mlprep.phases.phase2.strategies.regression_stratified_split import (
    RegressionStratifiedSplitStrategy,
)
from mlprep.phases.phase2.strategies.stratified_split import (
    StratifiedSplitStrategy,
)
from mlprep.phases.phase2.strategies.time_split import (
    TimeSplitStrategy,
)
from mlprep.phases.phase2.validation.class_distribution import (
    ClassDistributionValidator,
)
from mlprep.phases.phase2.validation.group_distribution import (
    GroupDistributionValidator,
)
from mlprep.phases.phase2.validation.partition_integrity import (
    PartitionIntegrityValidator,
)
from mlprep.phases.phase2.validation.split_sizes import (
    SplitSizeValidator,
)
from mlprep.phases.phase2.validation.target_distribution import (
    TargetDistributionValidator,
)
from mlprep.phases.phase2.validation.temporal_boundaries import (
    TemporalBoundaryValidator,
)
from mlprep.types import TaskType

ComponentResult = TypeVar("ComponentResult")


class Phase2:
    """Execute Phase 2 splitting and leakage protection."""

    phase_name = "phase_2"

    def __init__(
        self,
        config: Phase2Config,
    ) -> None:
        """Initialize Phase 2."""

        if not isinstance(config, Phase2Config):
            raise TypeError(
                "config must be an instance of Phase2Config."
            )

        self.config = config

        self.state: Phase2State | None = None
        self.result: Phase2Result | None = None

        self.report_builder = Phase2ReportBuilder(
            options=config.reporting
        )

        self.test_lock_token: str | None = None

    def run(
        self,
        *,
        phase1_result: Phase1Result,
        phase1_state: Phase1State | None = None,
    ) -> Phase2Result:
        """Execute the complete Phase 2 workflow."""

        self._validate_phase1_input(
            phase1_result=phase1_result,
        )

        resolved_phase1_state = (
            phase1_state
            or self._resolve_phase1_state(
                phase1_result
            )
        )

        if not self.config.execution.enabled:
            return self._build_disabled_result(
                phase1_result=phase1_result,
                phase1_state=resolved_phase1_state,
            )

        columns, role_findings = (
            self._resolve_split_roles(
                phase1_result=phase1_result,
                phase1_state=resolved_phase1_state,
            )
        )

        self._synchronize_contract_columns(
            columns=columns
        )

        contract = self._resolve_contract(
            phase1_result=phase1_result,
            phase1_state=resolved_phase1_state,
        )

        state = Phase2State(
            strategy=self.config.strategy,
            columns=columns,
            source_dataset_fingerprint=(
                resolved_phase1_state
                .dataset_fingerprint
            ),
        )

        self.state = state

        contract_findings = [
            *role_findings,
            *contract.findings,
        ]

        if (
            self.config.execution
            .stop_on_contract_failure
            and self._has_blocking_findings(
                contract_findings
            )
        ):
            return self._build_failed_contract_result(
                contract=contract,
                state=state,
                findings=contract_findings,
            )

        firewall = LeakageFirewall(
            options=self.config.leakage_firewall
        )

        pre_split_findings = self._execute_component(
            component_name="leakage_firewall_pre_split",
            operation=lambda: (
                firewall.inspect_before_split(
                    contract=contract,
                    state=state,
                )
            ),
            fallback=[],
        )

        all_pre_split_findings = [
            *contract_findings,
            *pre_split_findings,
        ]

        if self._has_blocking_findings(
            all_pre_split_findings
        ):
            return self._build_failed_contract_result(
                contract=contract,
                state=state,
                findings=all_pre_split_findings,
            )

        strategy = self._build_strategy()

        strategy_output = self._execute_component(
            component_name="split_strategy",
            operation=lambda: strategy.split(
                contract
            ),
            fallback=None,
        )

        if strategy_output is None:
            raise RuntimeError(
                "The split strategy failed to produce output."
            )

        result = Phase2Result(
            train=strategy_output.train,
            validation=strategy_output.validation,
            test=strategy_output.test,
            state=state,
            lock_test_set=False,
        )

        self.result = result

        result.add_findings(
            all_pre_split_findings
        )

        result.add_findings(
            strategy_output.findings
        )

        self._register_execution_metadata(
            contract=contract,
            strategy_output=strategy_output,
            result=result,
            state=state,
        )

        post_split_findings = self._execute_component(
            component_name="leakage_firewall_post_split",
            operation=lambda: (
                firewall.inspect_after_split(
                    contract=contract,
                    strategy_output=strategy_output,
                    state=state,
                )
            ),
            fallback=[],
        )

        result.add_findings(
            post_split_findings
        )

        self._run_validators(
            contract=contract,
            strategy_output=strategy_output,
            result=result,
            state=state,
        )

        fingerprint_generator = (
            Phase2FingerprintGenerator(
                options=self.config.fingerprint
            )
        )

        fingerprint_findings = (
            self._execute_component(
                component_name="fingerprint",
                operation=lambda: (
                    fingerprint_generator.generate(
                        train=result.train,
                        validation=result.validation,
                        test=strategy_output.test,
                        state=state,
                    )
                ),
                fallback=[],
            )
        )

        result.add_findings(
            fingerprint_findings
        )

        self._lock_test_partition(
            result=result,
            test=strategy_output.test,
            state=state,
        )

        evaluator = Phase2ReadinessEvaluator(
            options=self.config.readiness
        )

        evaluator.evaluate(
            result=result,
            state=state,
        )

        return result

    def build_report_dictionary(
        self,
    ) -> dict[str, object]:
        """Build the latest Phase 2 report dictionary."""

        result, state = self._require_completed_run()

        return self.report_builder.build_dictionary(
            result=result,
            state=state,
        )

    def build_json_report(
        self,
    ) -> str:
        """Build the latest Phase 2 JSON report."""

        result, state = self._require_completed_run()

        return self.report_builder.build_json(
            result=result,
            state=state,
        )

    def build_text_report(
        self,
    ) -> str:
        """Build the latest Phase 2 text report."""

        result, state = self._require_completed_run()

        return self.report_builder.build_text(
            result=result,
            state=state,
        )

    def access_test_partition(
        self,
        *,
        reason: str,
        token: str | None = None,
    ) -> DatasetPartition:
        """Access the locked test partition with authorization."""

        result, state = self._require_completed_run()

        if not isinstance(
            result.test,
            LockedTestPartition,
        ):
            return result.test

        locker = TestPartitionLocker(
            options=self.config.test_lock
        )

        return locker.authorize_access(
            locked_test=result.test,
            state=state,
            reason=reason,
            token=token,
        )

    def _resolve_split_roles(
        self,
        *,
        phase1_result: Phase1Result,
        phase1_state: Phase1State,
    ) -> tuple[
        SplitColumnState,
        list[SplitFinding],
    ]:
        """Resolve Phase 2 column roles."""

        resolver = SplitRoleResolver(
            options=self.config.split_roles
        )

        return self._execute_component(
            component_name="split_roles",
            operation=lambda: resolver.resolve(
                data=phase1_result.data,
                phase1_state=phase1_state,
                strategy=self.config.strategy,
            ),
            fallback=(
                self._empty_column_state(),
                [],
            ),
        )

    def _resolve_contract(
        self,
        *,
        phase1_result: Phase1Result,
        phase1_state: Phase1State,
    ) -> SplitContract:
        """Resolve the complete Phase 2 contract."""

        if self.config.contract is None:
            raise RuntimeError(
                "Phase 2 contract options are missing."
            )

        resolver = SplitContractResolver(
            options=self.config.contract
        )

        contract = self._execute_component(
            component_name="split_contract",
            operation=lambda: resolver.resolve(
                phase1_result=phase1_result,
                phase1_state=phase1_state,
            ),
            fallback=None,
        )

        if contract is None:
            raise RuntimeError(
                "The split contract could not be resolved."
            )

        return contract

    def _build_strategy(
        self,
    ) -> BaseSplitStrategy:
        """Build the configured split strategy."""

        options = (
            self.config.selected_strategy_options()
        )

        if (
            self.config.strategy
            == SplitStrategyName.RANDOM
        ):
            return RandomSplitStrategy(
                options=options
            )

        if (
            self.config.strategy
            == SplitStrategyName.STRATIFIED
        ):
            return StratifiedSplitStrategy(
                options=options
            )

        if (
            self.config.strategy
            == SplitStrategyName.GROUP
        ):
            return GroupSplitStrategy(
                options=options
            )

        if (
            self.config.strategy
            == SplitStrategyName.GROUP_STRATIFIED
        ):
            return GroupStratifiedSplitStrategy(
                options=options
            )

        if (
            self.config.strategy
            == SplitStrategyName.TIME
        ):
            return TimeSplitStrategy(
                options=options
            )

        if (
            self.config.strategy
            == SplitStrategyName.PURGED_TIME
        ):
            return PurgedTimeSplitStrategy(
                options=options
            )

        if (
            self.config.strategy
            == SplitStrategyName.REGRESSION_STRATIFIED
        ):
            return RegressionStratifiedSplitStrategy(
                options=options
            )

        if (
            self.config.strategy
            == SplitStrategyName.PREDEFINED
        ):
            return PredefinedSplitStrategy(
                options=options
            )

        raise ValueError(
            "Unsupported Phase 2 split strategy: "
            f"{self.config.strategy!r}"
        )

    def _run_validators(
        self,
        *,
        contract: SplitContract,
        strategy_output: StrategyOutput,
        result: Phase2Result,
        state: Phase2State,
    ) -> None:
        """Run every configured split validator."""

        size_validator = SplitSizeValidator(
            options=(
                self.config.validation
                .split_sizes
            )
        )

        size_findings = self._execute_component(
            component_name="split_sizes",
            operation=lambda: size_validator.validate(
                train=strategy_output.train,
                validation=strategy_output.validation,
                test=strategy_output.test,
            ),
            fallback=[],
        )

        result.add_findings(
            size_findings
        )

        integrity_validator = (
            PartitionIntegrityValidator(
                options=(
                    self.config.validation
                    .partition_integrity
                )
            )
        )

        integrity_findings = self._execute_component(
            component_name="partition_integrity",
            operation=lambda: (
                integrity_validator.validate(
                    train=strategy_output.train,
                    validation=(
                        strategy_output.validation
                    ),
                    test=strategy_output.test,
                )
            ),
            fallback=[],
        )

        result.add_findings(
            integrity_findings
        )

        self._run_task_validator(
            phase1_state=contract.phase1_state,
            strategy_output=strategy_output,
            result=result,
        )

        group_validator = (
            GroupDistributionValidator(
                options=(
                    self.config.validation
                    .group_distribution
                )
            )
        )

        group_findings = self._execute_component(
            component_name="group_distribution",
            operation=lambda: (
                group_validator.validate(
                    contract=contract,
                    indices=strategy_output.indices,
                )
            ),
            fallback=[],
        )

        result.add_findings(
            group_findings
        )

        temporal_validator = (
            TemporalBoundaryValidator(
                options=(
                    self.config.validation
                    .temporal_boundaries
                )
            )
        )

        (
            temporal_findings,
            boundaries,
        ) = self._execute_component(
            component_name="temporal_boundaries",
            operation=lambda: (
                temporal_validator.validate(
                    contract=contract,
                    indices=strategy_output.indices,
                )
            ),
            fallback=(
                [],
                state.boundaries,
            ),
        )

        result.add_findings(
            temporal_findings
        )

        state.boundaries = boundaries

    def _run_task_validator(
        self,
        *,
        phase1_state: Phase1State,
        strategy_output: StrategyOutput,
        result: Phase2Result,
    ) -> None:
        """Run the task-specific target validator."""

        problem_contract = (
            phase1_state.problem_contract
        )

        if problem_contract is None:
            return

        classification_tasks = {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }

        if (
            problem_contract.task_type
            in classification_tasks
        ):
            validator = (
                ClassDistributionValidator(
                    options=(
                        self.config.validation
                        .class_distribution
                    )
                )
            )

            findings = self._execute_component(
                component_name="class_distribution",
                operation=lambda: validator.validate(
                    train=strategy_output.train,
                    validation=(
                        strategy_output.validation
                    ),
                    test=strategy_output.test,
                ),
                fallback=[],
            )

            result.add_findings(findings)

            return

        if (
            problem_contract.task_type
            == TaskType.REGRESSION
        ):
            validator = (
                TargetDistributionValidator(
                    options=(
                        self.config.validation
                        .target_distribution
                    )
                )
            )

            findings = self._execute_component(
                component_name="target_distribution",
                operation=lambda: validator.validate(
                    train=strategy_output.train,
                    validation=(
                        strategy_output.validation
                    ),
                    test=strategy_output.test,
                ),
                fallback=[],
            )

            result.add_findings(findings)

    def _register_execution_metadata(
        self,
        *,
        contract: SplitContract,
        strategy_output: StrategyOutput,
        result: Phase2Result,
        state: Phase2State,
    ) -> None:
        """Register split execution metadata."""

        if not (
            self.config.execution
            .store_execution_metadata
        ):
            return

        total_count = len(contract.data)

        assigned_count = (
            len(
                strategy_output
                .indices
                .train_indices
            )
            + len(
                strategy_output
                .indices
                .validation_indices
            )
            + len(
                strategy_output
                .indices
                .test_indices
            )
        )

        if assigned_count <= 0:
            raise ValueError(
                "The split strategy did not assign "
                "any source rows."
            )

        ratio_denominator = assigned_count

        actual_ratios = SplitRatios(
            train=(
                len(
                    strategy_output
                    .indices
                    .train_indices
                )
                / ratio_denominator
            ),
            validation=(
                len(
                    strategy_output
                    .indices
                    .validation_indices
                )
                / ratio_denominator
            ),
            test=(
                len(
                    strategy_output
                    .indices
                    .test_indices
                )
                / ratio_denominator
            ),
            tolerance=1e-6,
        )

        unassigned_row_count = max(
            0,
            total_count - assigned_count,
        )

        strategy_output.metadata[
            "source_row_count"
        ] = total_count

        strategy_output.metadata[
            "assigned_row_count"
        ] = assigned_count

        strategy_output.metadata[
            "unassigned_row_count"
        ] = unassigned_row_count

        strategy_output.metadata[
            "assigned_row_fraction"
        ] = (
            assigned_count / total_count
            if total_count > 0
            else 0.0
        )

        metadata = SplitExecutionMetadata(
            strategy=self.config.strategy,
            random_seed=self.config.random_seed,
            shuffle=bool(
                strategy_output.metadata.get(
                    "shuffle",
                    False,
                )
            ),
            stratified=bool(
                strategy_output.metadata.get(
                    "stratified",
                    False,
                )
            ),
            grouped=bool(
                strategy_output.metadata.get(
                    "grouped",
                    False,
                )
            ),
            temporal=bool(
                strategy_output.metadata.get(
                    "temporal",
                    False,
                )
            ),
            requested_ratios=(
                self.config.ratios.model_copy(
                    deep=True
                )
            ),
            actual_ratios=actual_ratios,
            boundaries=state.boundaries,
            source_dataset_fingerprint=(
                state.source_dataset_fingerprint
            ),
            partition_metadata={
                PartitionName.TRAIN: (
                    strategy_output
                    .train
                    .metadata
                ),
                PartitionName.VALIDATION: (
                    strategy_output
                    .validation
                    .metadata
                ),
                PartitionName.TEST: (
                    strategy_output
                    .test
                    .metadata
                ),
            },
            metadata=dict(
                strategy_output.metadata
            ),
        )

        state.execution_metadata = metadata
        result.execution_metadata = metadata

    def _lock_test_partition(
        self,
        *,
        result: Phase2Result,
        test: DatasetPartition,
        state: Phase2State,
    ) -> None:
        """Lock the test partition."""

        locker = TestPartitionLocker(
            options=self.config.test_lock
        )

        (
            protected_test,
            lock_token,
            findings,
        ) = locker.lock(
            test=test,
            state=state,
        )

        result.test = protected_test
        result.add_findings(findings)

        self.test_lock_token = lock_token

    def _synchronize_contract_columns(
        self,
        *,
        columns: SplitColumnState,
    ) -> None:
        """Synchronize resolved roles with contract options."""

        contract_options = self.config.contract

        if contract_options is None:
            raise RuntimeError(
                "Phase 2 contract options are missing."
            )

        contract_options.target_column = (
            columns.target_column
        )

        contract_options.feature_columns = list(
            columns.feature_columns
        )

        contract_options.identifier_columns = list(
            columns.identifier_columns
        )

        contract_options.group_columns = list(
            columns.group_columns
        )

        contract_options.timestamp_column = (
            columns.timestamp_column
        )

        contract_options.observation_key = list(
            columns.observation_key
        )

        contract_options.excluded_feature_columns = list(
            columns.excluded_feature_columns
        )

        contract_options.internal_row_id_column = (
            columns.internal_row_id_column
        )

    def _build_failed_contract_result(
        self,
        *,
        contract: SplitContract,
        state: Phase2State,
        findings: list[SplitFinding],
    ) -> Phase2Result:
        """Build a failed result before normal splitting."""

        empty_train = self._empty_partition(
            name=PartitionName.TRAIN,
            contract=contract,
        )

        empty_validation = self._empty_partition(
            name=PartitionName.VALIDATION,
            contract=contract,
        )

        empty_test = self._empty_partition(
            name=PartitionName.TEST,
            contract=contract,
        )

        result = Phase2Result(
            train=empty_train,
            validation=empty_validation,
            test=empty_test,
            state=state,
            lock_test_set=False,
        )

        result.add_findings(findings)

        blocking_findings = [
            finding
            for finding in findings
            if (
                finding.blocks_readiness
                or finding.severity
                == SplitFindingSeverity.CRITICAL
            )
        ]

        decision = Phase2ReadinessDecision(
            ready=False,
            status=Phase2Status.FAILED,
            blocking_finding_ids=[
                finding.finding_id
                for finding in blocking_findings
            ],
            blocking_finding_codes=[
                finding.code
                for finding in blocking_findings
            ],
            reasons=[
                finding.message
                for finding in blocking_findings
            ],
        )

        result.finalize(decision)

        self.result = result

        return result

    def _build_disabled_result(
        self,
        *,
        phase1_result: Phase1Result,
        phase1_state: Phase1State,
    ) -> Phase2Result:
        """Build a failed result when Phase 2 is disabled."""

        columns = self._empty_column_state()

        state = Phase2State(
            strategy=self.config.strategy,
            columns=columns,
            source_dataset_fingerprint=(
                phase1_state.dataset_fingerprint
            ),
        )

        empty_index = phase1_result.data.index[
            0:0
        ]

        empty_features = pd.DataFrame(
            index=empty_index
        )

        empty_target = pd.Series(
            index=empty_index,
            dtype="object",
            name=columns.target_column,
        )

        empty_row_ids = pd.Series(
            index=empty_index,
            dtype="object",
            name=columns.internal_row_id_column,
        )

        train = DatasetPartition(
            name=PartitionName.TRAIN,
            X=empty_features.copy(),
            y=empty_target.copy(),
            row_ids=empty_row_ids.copy(),
        )

        validation = DatasetPartition(
            name=PartitionName.VALIDATION,
            X=empty_features.copy(),
            y=empty_target.copy(),
            row_ids=empty_row_ids.copy(),
        )

        test = DatasetPartition(
            name=PartitionName.TEST,
            X=empty_features.copy(),
            y=empty_target.copy(),
            row_ids=empty_row_ids.copy(),
        )

        result = Phase2Result(
            train=train,
            validation=validation,
            test=test,
            state=state,
            lock_test_set=False,
        )

        finding = SplitFinding(
            code="P2-EXECUTION-001",
            category=SplitFindingCategory.EXECUTION,
            severity=SplitFindingSeverity.INFO,
            component=self.phase_name,
            message="Phase 2 execution is disabled.",
            recommended_action=(
                SplitFindingAction.NONE
            ),
            applied_action=(
                SplitFindingAction.NONE
            ),
            blocks_readiness=False,
        )

        result.add_finding(finding)

        result.finalize(
            Phase2ReadinessDecision(
                ready=False,
                status=Phase2Status.FAILED,
                reasons=[
                    "Phase 2 execution is disabled."
                ],
            )
        )

        self.state = state
        self.result = result

        return result

    @staticmethod
    def _empty_partition(
        *,
        name: PartitionName,
        contract: SplitContract,
    ) -> DatasetPartition:
        """Build an empty partition using the contract schema."""

        empty_data = contract.data.iloc[
            0:0
        ]

        return DatasetPartition(
            name=name,
            X=empty_data.loc[
                :,
                contract.feature_columns,
            ].copy(),
            y=empty_data[
                contract.target_column
            ].copy(),
            row_ids=empty_data[
                contract.internal_row_id_column
            ].copy(),
        )

    def _execute_component(
        self,
        *,
        component_name: str,
        operation: Callable[
            [],
            ComponentResult,
        ],
        fallback: ComponentResult,
    ) -> ComponentResult:
        """Execute one component using the configured policy."""

        try:
            return operation()

        except Exception as error:
            if (
                self.config.execution
                .stop_on_component_exception
            ):
                raise

            if self.result is not None:
                self.result.add_finding(
                    SplitFinding(
                        code="P2-EXECUTION-002",
                        category=(
                            SplitFindingCategory.EXECUTION
                        ),
                        severity=(
                            SplitFindingSeverity.CRITICAL
                        ),
                        component=component_name,
                        message=(
                            f"Component {component_name!r} "
                            f"failed with "
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                        recommended_action=(
                            SplitFindingAction.RAISE
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
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

            return fallback

    @staticmethod
    def _validate_phase1_input(
        *,
        phase1_result: Phase1Result,
    ) -> None:
        """Validate the Phase 1 result object."""

        if not isinstance(
            phase1_result,
            Phase1Result,
        ):
            raise TypeError(
                "phase1_result must be an instance "
                "of Phase1Result."
            )

    @staticmethod
    def _resolve_phase1_state(
        phase1_result: Phase1Result,
    ) -> Phase1State:
        """Rebuild Phase 1 state from the result."""

        if not phase1_result.phase_state:
            raise ValueError(
                "Phase 1 state is missing. Pass "
                "phase1_state explicitly or enable "
                "state storage in Phase 1."
            )

        return Phase1State.model_validate(
            phase1_result.phase_state
        )

    @staticmethod
    def _has_blocking_findings(
        findings: list[SplitFinding],
    ) -> bool:
        """Return whether blocking findings exist."""

        return any(
            finding.blocks_readiness
            or finding.severity
            == SplitFindingSeverity.CRITICAL
            for finding in findings
        )

    def _require_completed_run(
        self,
    ) -> tuple[Phase2Result, Phase2State]:
        """Return the latest result and state."""

        if (
            self.result is None
            or self.state is None
        ):
            raise RuntimeError(
                "Phase 2 has not been executed yet."
            )

        return self.result, self.state

    def _empty_column_state(
        self,
    ) -> SplitColumnState:
        """Build a placeholder column state."""

        return SplitColumnState(
            target_column=(
                self.config.split_roles
                .target_column
                or "__unresolved_target__"
            ),
            feature_columns=[],
            identifier_columns=[],
            group_columns=[],
            observation_key=[],
            timestamp_column=None,
            excluded_feature_columns=[],
            internal_row_id_column=(
                self.config.split_roles
                .internal_row_id_column
            ),
        )