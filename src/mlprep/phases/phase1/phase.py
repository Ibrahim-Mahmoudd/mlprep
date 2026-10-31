"""Phase 1 orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd

from mlprep.phases.phase1.config import Phase1Config
from mlprep.phases.phase1.contract.column_roles import (
    ColumnRoleAssigner,
)
from mlprep.phases.phase1.contract.problem_contract import (
    ProblemContractResolver,
)
from mlprep.phases.phase1.deterministic_normalization.column_names import (
    ColumnNameNormalizer,
)
from mlprep.phases.phase1.deterministic_normalization.missing_tokens import (
    MissingTokenNormalizer,
)
from mlprep.phases.phase1.deterministic_normalization.text_normalization import (
    TextNormalizer,
)
from mlprep.phases.phase1.inspection.cardinality import (
    CardinalityInspector,
)
from mlprep.phases.phase1.inspection.mixed_types import (
    MixedTypeDetector,
)
from mlprep.phases.phase1.inspection.physical_types import (
    PhysicalTypeInspector,
)
from mlprep.phases.phase1.inspection.semantic_types import (
    SemanticTypeInferer,
)
from mlprep.phases.phase1.inspection.temporal_detection import (
    TemporalDetector,
)
from mlprep.phases.phase1.output_governance.fingerprint import (
    FingerprintGenerator,
)
from mlprep.phases.phase1.output_governance.memory import (
    MemoryOptimizer,
)
from mlprep.phases.phase1.output_governance.readiness import (
    ReadinessEvaluator,
)
from mlprep.phases.phase1.output_governance.report import (
    Phase1ReportBuilder,
)
from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.validation_and_risk.row_identity import (
    RowIdentityManager,
)
from mlprep.phases.phase1.output_governance.schema import (
    SchemaBuilder,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
    ReadinessDecision,
)
from mlprep.phases.phase1.validation_and_risk.constraints import (
    ConstraintValidator,
)
from mlprep.phases.phase1.validation_and_risk.duplicate_columns import (
    DuplicateColumnDetector,
)
from mlprep.phases.phase1.validation_and_risk.duplicate_rows import (
    DuplicateRowDetector,
)
from mlprep.phases.phase1.validation_and_risk.leakage_risks import (
    LeakageRiskDetector,
)
from mlprep.phases.phase1.validation_and_risk.sensitive_data import (
    SensitiveDataDetector,
)
from mlprep.phases.phase1.validation_and_risk.target_validation import (
    TargetValidator,
)
from mlprep.types import (
    FindingAction,
    FindingCategory,
    PhaseStatus,
    Severity,
)

ComponentResult = TypeVar("ComponentResult")


class Phase1:
    """Execute Phase 1 components in dependency-safe order."""

    phase_name = "phase_1"

    def __init__(
        self,
        config: Phase1Config,
    ) -> None:
        """Initialize the Phase 1 pipeline."""

        if not isinstance(config, Phase1Config):
            raise TypeError(
                "config must be an instance of Phase1Config."
            )

        self.config = config

        self.state: Phase1State | None = None
        self.result: Phase1Result | None = None

        self.report_builder = Phase1ReportBuilder(
            options=config.reporting
        )

    def run(
        self,
        data: pd.DataFrame,
    ) -> Phase1Result:
        """Execute the complete Phase 1 workflow."""

        self._validate_input(data)

        working_data = self._prepare_input(data)

        state = Phase1State(
            internal_row_id_column=(
                self.config.row_identity.column_name
            )
        )

        result = Phase1Result(
            data=working_data,
            internal_row_id_column=(
                self.config.row_identity.column_name
            ),
        )

        self.state = state
        self.result = result

        if not self.config.execution.enabled:
            self._finalize_disabled_phase(
                result=result,
            )

            return result

        working_data = self._run_column_names(
            data=working_data,
            result=result,
            state=state,
        )

        self._normalize_configured_column_references(
            state=state
        )

        if self._should_stop(result):
            return self._finalize_early(
                data=working_data,
                result=result,
                state=state,
            )

        self._run_problem_contract(
            data=working_data,
            result=result,
            state=state,
        )

        if self._should_stop(result):
            return self._finalize_early(
                data=working_data,
                result=result,
                state=state,
            )

        self._run_column_roles(
            data=working_data,
            result=result,
            state=state,
        )

        if self._should_stop(result):
            return self._finalize_early(
                data=working_data,
                result=result,
                state=state,
            )

        working_data = self._run_row_identity(
            data=working_data,
            result=result,
            state=state,
        )

        working_data = self._run_text_normalization(
            data=working_data,
            result=result,
        )

        working_data = self._run_missing_tokens(
            data=working_data,
            result=result,
            state=state,
        )

        self._run_physical_types(
            data=working_data,
            result=result,
            state=state,
        )

        self._run_mixed_types(
            data=working_data,
            result=result,
            state=state,
        )

        working_data = self._run_temporal_detection(
            data=working_data,
            result=result,
            state=state,
        )

        self._run_semantic_types(
            data=working_data,
            result=result,
            state=state,
        )

        self._run_cardinality(
            data=working_data,
            result=result,
            state=state,
        )

        self._run_duplicate_columns(
            data=working_data,
            result=result,
            state=state,
        )

        working_data = self._run_duplicate_rows(
            data=working_data,
            result=result,
            state=state,
        )

        self._run_target_validation(
            data=working_data,
            result=result,
            state=state,
        )

        self._run_constraints(
            data=working_data,
            result=result,
            state=state,
        )

        self._run_sensitive_data(
            data=working_data,
            result=result,
            state=state,
        )

        self._run_leakage_risks(
            data=working_data,
            result=result,
            state=state,
        )

        working_data = self._run_memory_optimization(
            data=working_data,
            result=result,
            state=state,
        )

        self._run_schema_builder(
            data=working_data,
            result=result,
            state=state,
        )

        self._run_fingerprints(
            data=working_data,
            result=result,
            state=state,
        )

        result.data = working_data

        if (
            self.config.execution
            .store_phase_state_in_result
        ):
            result.phase_state = state.model_dump(
                mode="json"
            )

        self._run_readiness(
            result=result,
            state=state,
        )

        return result

    def build_report_dictionary(
        self,
    ) -> dict[str, Any]:
        """Build a dictionary report for the latest run."""

        result, state = self._require_completed_run()

        return self.report_builder.build_dictionary(
            result=result,
            state=state,
        )

    def build_json_report(
        self,
    ) -> str:
        """Build a JSON report for the latest run."""

        result, state = self._require_completed_run()

        return self.report_builder.build_json(
            result=result,
            state=state,
        )

    def build_text_report(
        self,
    ) -> str:
        """Build a text report for the latest run."""

        result, state = self._require_completed_run()

        return self.report_builder.build_text(
            result=result,
            state=state,
        )

    def _run_column_names(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Run deterministic column-name normalization."""

        normalizer = ColumnNameNormalizer(
            options=self.config.column_names
        )

        return self._execute_mutating_component(
            component_name="column_names",
            operation=lambda: normalizer.normalize(
                data=data,
                result=result,
                state=state,
            ),
            fallback=data,
            result=result,
        )

    def _run_problem_contract(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Resolve the problem contract."""

        resolver = ProblemContractResolver(
            options=self.config.problem_contract
        )

        self._execute_component(
            component_name="problem_contract",
            operation=lambda: resolver.resolve(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_column_roles(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Assign functional column roles."""

        assigner = ColumnRoleAssigner(
            options=self.config.column_roles
        )

        self._execute_component(
            component_name="column_roles",
            operation=lambda: assigner.assign(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_row_identity(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Create the internal row identity."""

        manager = RowIdentityManager(
            options=self.config.row_identity
        )

        updated_data = self._execute_mutating_component(
            component_name="row_identity",
            operation=lambda: manager.create(
                data=data,
                result=result,
                state=state,
            ),
            fallback=data,
            result=result,
        )

        if (
            self.config.execution
            .validate_after_each_mutation
        ):
            self._execute_component(
                component_name=(
                    "row_identity_validation"
                ),
                operation=lambda: manager.validate(
                    data=updated_data,
                    result=result,
                ),
                result=result,
            )

        return updated_data

    def _run_text_normalization(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> pd.DataFrame:
        """Normalize deterministic text values."""

        normalizer = TextNormalizer(
            options=self.config.text_normalization
        )

        return self._execute_mutating_component(
            component_name="text_normalization",
            operation=lambda: normalizer.normalize(
                data=data,
                result=result,
            ),
            fallback=data,
            result=result,
        )

    def _run_missing_tokens(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Normalize configured missing tokens."""

        normalizer = MissingTokenNormalizer(
            options=self.config.missing_tokens
        )

        return self._execute_mutating_component(
            component_name="missing_tokens",
            operation=lambda: normalizer.normalize(
                data=data,
                result=result,
                state=state,
            ),
            fallback=data,
            result=result,
        )

    def _run_physical_types(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Inspect physical column types."""

        inspector = PhysicalTypeInspector(
            options=self.config.physical_types
        )

        self._execute_component(
            component_name="physical_types",
            operation=lambda: inspector.inspect(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_mixed_types(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Detect mixed runtime value types."""

        detector = MixedTypeDetector(
            options=self.config.mixed_types
        )

        self._execute_component(
            component_name="mixed_types",
            operation=lambda: detector.inspect(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_temporal_detection(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Detect and optionally parse temporal columns."""

        detector = TemporalDetector(
            options=self.config.temporal_detection
        )

        return self._execute_mutating_component(
            component_name="temporal_detection",
            operation=lambda: detector.inspect(
                data=data,
                result=result,
                state=state,
            ),
            fallback=data,
            result=result,
        )

    def _run_semantic_types(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Infer semantic column types."""

        inferer = SemanticTypeInferer(
            options=self.config.semantic_types
        )

        self._execute_component(
            component_name="semantic_types",
            operation=lambda: inferer.inspect(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_cardinality(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Inspect cardinality and constancy."""

        inspector = CardinalityInspector(
            options=self.config.cardinality
        )

        self._execute_component(
            component_name="cardinality",
            operation=lambda: inspector.inspect(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_duplicate_columns(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Detect duplicate columns."""

        detector = DuplicateColumnDetector(
            options=self.config.duplicate_columns
        )

        self._execute_component(
            component_name="duplicate_columns",
            operation=lambda: detector.inspect(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_duplicate_rows(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Detect and optionally resolve duplicate rows."""

        detector = DuplicateRowDetector(
            options=self.config.duplicate_rows
        )

        return self._execute_mutating_component(
            component_name="duplicate_rows",
            operation=lambda: detector.inspect(
                data=data,
                result=result,
                state=state,
            ),
            fallback=data,
            result=result,
        )

    def _run_target_validation(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Validate the target column."""

        validator = TargetValidator(
            options=self.config.target_validation
        )

        self._execute_component(
            component_name="target_validation",
            operation=lambda: validator.validate(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_constraints(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Validate configured data constraints."""

        validator = ConstraintValidator(
            options=self.config.constraints
        )

        self._execute_component(
            component_name="constraints",
            operation=lambda: validator.validate(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_sensitive_data(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Detect potentially sensitive columns."""

        detector = SensitiveDataDetector(
            options=self.config.sensitive_data
        )

        self._execute_component(
            component_name="sensitive_data",
            operation=lambda: detector.inspect(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_leakage_risks(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Detect preliminary leakage risks."""

        detector = LeakageRiskDetector(
            options=self.config.leakage_risks
        )

        self._execute_component(
            component_name="leakage_risks",
            operation=lambda: detector.inspect(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_memory_optimization(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Apply safe physical memory optimization."""

        optimizer = MemoryOptimizer(
            options=self.config.memory
        )

        return self._execute_mutating_component(
            component_name="memory",
            operation=lambda: optimizer.optimize(
                data=data,
                result=result,
                state=state,
            ),
            fallback=data,
            result=result,
        )

    def _run_schema_builder(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Build the canonical dataset schema."""

        builder = SchemaBuilder(
            options=self.config.schema
        )

        self._execute_component(
            component_name="schema",
            operation=lambda: builder.build(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_fingerprints(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Generate dataset and schema fingerprints."""

        generator = FingerprintGenerator(
            options=self.config.fingerprint
        )

        self._execute_component(
            component_name="fingerprint",
            operation=lambda: generator.generate(
                data=data,
                result=result,
                state=state,
            ),
            result=result,
        )

    def _run_readiness(
        self,
        *,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Evaluate readiness for Phase 2."""

        evaluator = ReadinessEvaluator(
            options=self.config.readiness
        )

        evaluator.evaluate(
            result=result,
            state=state,
        )

    def _normalize_configured_column_references(
        self,
        *,
        state: Phase1State,
    ) -> None:
        """Update configured column references after name normalization."""

        mapping = state.original_to_normalized_names

        if not mapping:
            return

        self._normalize_problem_contract_references(
            mapping=mapping
        )

        self._normalize_column_role_references(
            mapping=mapping
        )

        self._normalize_component_references(
            mapping=mapping
        )

        self.config._synchronize_component_options()

    def _normalize_problem_contract_references(
        self,
        *,
        mapping: dict[str, str],
    ) -> None:
        """Normalize problem-contract column references."""

        options = self.config.problem_contract

        options.target_column = self._map_name(
            options.target_column,
            mapping,
        )

        options.observation_key = self._map_names(
            options.observation_key,
            mapping,
        )

        options.entity_columns = self._map_names(
            options.entity_columns,
            mapping,
        )

        options.group_columns = self._map_names(
            options.group_columns,
            mapping,
        )

        if options.prediction_timestamp_column is not None:
            options.prediction_timestamp_column = (
                self._map_name(
                    options.prediction_timestamp_column,
                    mapping,
                )
            )

        if options.outcome_timestamp_column is not None:
            options.outcome_timestamp_column = (
                self._map_name(
                    options.outcome_timestamp_column,
                    mapping,
                )
            )

    def _normalize_column_role_references(
        self,
        *,
        mapping: dict[str, str],
    ) -> None:
        """Normalize column-role references."""

        options = self.config.column_roles

        options.target_column = self._map_name(
            options.target_column,
            mapping,
        )

        if options.feature_columns is not None:
            options.feature_columns = self._map_names(
                options.feature_columns,
                mapping,
            )

        options.identifier_columns = self._map_names(
            options.identifier_columns,
            mapping,
        )

        options.group_columns = self._map_names(
            options.group_columns,
            mapping,
        )

        options.timestamp_columns = self._map_names(
            options.timestamp_columns,
            mapping,
        )

        options.sensitive_columns = self._map_names(
            options.sensitive_columns,
            mapping,
        )

        options.metadata_columns = self._map_names(
            options.metadata_columns,
            mapping,
        )

        options.ignored_columns = self._map_names(
            options.ignored_columns,
            mapping,
        )

        options.inference_unavailable_columns = (
            self._map_names(
                options.inference_unavailable_columns,
                mapping,
            )
        )

        if options.prediction_timestamp_column is not None:
            options.prediction_timestamp_column = (
                self._map_name(
                    options.prediction_timestamp_column,
                    mapping,
                )
            )

        if options.outcome_timestamp_column is not None:
            options.outcome_timestamp_column = (
                self._map_name(
                    options.outcome_timestamp_column,
                    mapping,
                )
            )

        if options.sample_weight_column is not None:
            options.sample_weight_column = self._map_name(
                options.sample_weight_column,
                mapping,
            )

        if options.fold_column is not None:
            options.fold_column = self._map_name(
                options.fold_column,
                mapping,
            )

    def _normalize_component_references(
        self,
        *,
        mapping: dict[str, str],
    ) -> None:
        """Normalize column references in remaining components."""

        self.config.target_validation.target_column = (
            self._map_name(
                self.config
                .target_validation
                .target_column,
                mapping,
            )
        )

        self.config.leakage_risks.target_column = (
            self._map_name(
                self.config.leakage_risks.target_column,
                mapping,
            )
        )

        self.config.leakage_risks.declared_post_outcome_columns = (
            self._map_names(
                self.config
                .leakage_risks
                .declared_post_outcome_columns,
                mapping,
            )
        )

        self.config.leakage_risks.inference_unavailable_columns = (
            self._map_names(
                self.config
                .leakage_risks
                .inference_unavailable_columns,
                mapping,
            )
        )

        self.config.duplicate_rows.observation_key = (
            self._map_names(
                self.config
                .duplicate_rows
                .observation_key,
                mapping,
            )
        )

        if self.config.duplicate_rows.target_column is not None:
            self.config.duplicate_rows.target_column = (
                self._map_name(
                    self.config
                    .duplicate_rows
                    .target_column,
                    mapping,
                )
            )

        self.config.sensitive_data.declared_sensitive_columns = (
            self._map_names(
                self.config
                .sensitive_data
                .declared_sensitive_columns,
                mapping,
            )
        )

        self.config.schema.required_columns = (
            self._map_names(
                self.config.schema.required_columns,
                mapping,
            )
        )

        self.config.schema.optional_columns = (
            self._map_names(
                self.config.schema.optional_columns,
                mapping,
            )
        )

        self.config.schema.descriptions = {
            self._map_name(column, mapping): description
            for column, description
            in self.config.schema.descriptions.items()
        }

        for constraint in self.config.constraints.ranges:
            constraint.column = self._map_name(
                constraint.column,
                mapping,
            )

        for constraint in (
            self.config.constraints.allowed_values
        ):
            constraint.column = self._map_name(
                constraint.column,
                mapping,
            )

        for constraint in (
            self.config.constraints.regex_patterns
        ):
            constraint.column = self._map_name(
                constraint.column,
                mapping,
            )

        for constraint in (
            self.config.constraints.cross_column_rules
        ):
            constraint.columns = self._map_names(
                constraint.columns,
                mapping,
            )

    def _execute_component(
        self,
        *,
        component_name: str,
        operation: Callable[[], ComponentResult],
        result: Phase1Result,
    ) -> ComponentResult | None:
        """Execute a nonmutating component safely."""

        try:
            return operation()
        except Exception as error:
            self._handle_component_exception(
                component_name=component_name,
                error=error,
                result=result,
            )

            return None

    def _execute_mutating_component(
        self,
        *,
        component_name: str,
        operation: Callable[[], pd.DataFrame],
        fallback: pd.DataFrame,
        result: Phase1Result,
    ) -> pd.DataFrame:
        """Execute a DataFrame-mutating component safely."""

        try:
            updated_data = operation()

            if not isinstance(
                updated_data,
                pd.DataFrame,
            ):
                raise TypeError(
                    f"Component {component_name!r} did not "
                    "return a pandas DataFrame."
                )

            result.data = updated_data

            return updated_data

        except Exception as error:
            self._handle_component_exception(
                component_name=component_name,
                error=error,
                result=result,
            )

            return fallback

    def _handle_component_exception(
        self,
        *,
        component_name: str,
        error: Exception,
        result: Phase1Result,
    ) -> None:
        """Handle an unexpected component exception."""

        finding = Finding(
            code="P1-EXECUTION-001",
            category=FindingCategory.INPUT,
            severity=Severity.CRITICAL,
            component=component_name,
            message=(
                f"Component {component_name!r} failed with "
                f"{type(error).__name__}: {error}"
            ),
            recommended_action=FindingAction.RAISE,
            applied_action=FindingAction.REPORT,
            blocks_readiness=True,
            details={
                "exception_type": (
                    type(error).__name__
                ),
                "exception_message": str(error),
            },
        )

        result.add_finding(finding)

        if (
            self.config.execution
            .stop_on_component_exception
        ):
            raise

    def _should_stop(
        self,
        result: Phase1Result,
    ) -> bool:
        """Return whether Phase 1 should stop early."""

        if not (
            self.config.execution
            .stop_on_critical_finding
        ):
            return False

        return bool(
            result.get_findings_by_severity(
                Severity.CRITICAL
            )
        )

    def _finalize_early(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> Phase1Result:
        """Finalize Phase 1 after an early stop."""

        result.data = data

        if (
            self.config.execution
            .store_phase_state_in_result
        ):
            result.phase_state = state.model_dump(
                mode="json"
            )

        self._run_readiness(
            result=result,
            state=state,
        )

        return result

    @staticmethod
    def _finalize_disabled_phase(
        *,
        result: Phase1Result,
    ) -> None:
        """Finalize a disabled Phase 1 execution."""

        result.add_finding(
            Finding(
                code="P1-EXECUTION-002",
                category=FindingCategory.INPUT,
                severity=Severity.INFO,
                component="phase_1",
                message="Phase 1 execution is disabled.",
                recommended_action=FindingAction.NONE,
                applied_action=FindingAction.NONE,
                blocks_readiness=False,
            )
        )

        result.finalize(
            readiness=ReadinessDecision(
                ready=False,
                status=PhaseStatus.FAILED,
                reasons=[
                    "Phase 1 execution is disabled."
                ],
            )
        )

    @staticmethod
    def _validate_input(
        data: object,
    ) -> None:
        """Validate the Phase 1 input object."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Phase1.run requires a pandas DataFrame."
            )

    def _prepare_input(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare the working DataFrame."""

        if (
            self.config.execution
            .copy_input_dataframe
        ):
            return data.copy(deep=True)

        return data

    def _require_completed_run(
        self,
    ) -> tuple[Phase1Result, Phase1State]:
        """Return the latest result and state."""

        if (
            self.result is None
            or self.state is None
        ):
            raise RuntimeError(
                "Phase 1 has not been executed yet."
            )

        return self.result, self.state

    @staticmethod
    def _map_name(
        column: str,
        mapping: dict[str, str],
    ) -> str:
        """Map one column name."""

        return mapping.get(column, column)

    @classmethod
    def _map_names(
        cls,
        columns: list[str],
        mapping: dict[str, str],
    ) -> list[object]:
        """Map multiple column names."""

        return [
            cls._map_name(
                column,
                mapping,
            )
            for column in columns
        ]