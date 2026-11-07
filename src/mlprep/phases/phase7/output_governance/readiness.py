"""Phase 8 readiness evaluation for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass

from mlprep.phases.phase7.output_governance.metadata import (
    Phase7Finding,
    Phase7FindingAction,
    Phase7FindingCategory,
    Phase7FindingSeverity,
    Phase7Status,
)
from mlprep.phases.phase7.output_governance.result import (
    Phase7ReadinessDecision,
    Phase7Result,
)
from mlprep.phases.phase7.output_governance.state import (
    Phase7State,
)


@dataclass(slots=True)
class Phase7ReadinessOptions:
    """Options controlling readiness for Phase 8."""

    block_on_critical_findings: bool = True
    block_on_error_findings: bool = True
    block_on_explicit_blockers: bool = True

    require_schema_validation: bool = True
    require_feature_count_validation: bool = True
    require_protected_feature_validation: bool = True
    require_target_integrity_validation: bool = True
    require_row_integrity_validation: bool = True

    require_training_only_fit: bool = True
    require_authorized_supervised_target_access: bool = True
    require_validation_evaluation_only: bool = True

    require_consistent_schemas: bool = True
    require_valid_feature_count: bool = True
    require_valid_protected_features: bool = True

    require_unchanged_selected_values: bool = True
    require_unchanged_targets: bool = True
    require_unchanged_row_ids: bool = True

    require_authorized_test_transform: bool = True
    require_protected_test_partition: bool = True

    require_selected_lineage: bool = True
    require_fitted_state_fingerprint: bool = True
    require_output_fingerprint: bool = True

    maximum_warning_count: int | None = None

    def __post_init__(self) -> None:
        """Validate readiness options."""

        if (
            self.maximum_warning_count is not None
            and self.maximum_warning_count < 0
        ):
            raise ValueError(
                "maximum_warning_count cannot be negative."
            )


class Phase7ReadinessEvaluator:
    """Determine whether Phase 8 may begin."""

    component_name = "readiness"

    def __init__(
        self,
        options: Phase7ReadinessOptions | None = None,
    ) -> None:
        """Initialize the readiness evaluator."""

        self.options = (
            options
            or Phase7ReadinessOptions()
        )

    def evaluate(
        self,
        *,
        result: Phase7Result,
        state: Phase7State,
    ) -> Phase7ReadinessDecision:
        """Evaluate Phase 7 readiness."""

        result.add_findings(
            self._validation_findings(
                result=result,
                state=state,
            )
        )

        blocking = [
            finding
            for finding in result.findings
            if self._finding_blocks(
                finding
            )
        ]

        warning_count = len(
            result.findings_by_severity(
                Phase7FindingSeverity.WARNING
            )
        )

        if (
            self.options.maximum_warning_count
            is not None
            and warning_count
            > self.options.maximum_warning_count
        ):
            finding = self._blocking_finding(
                code="P7-READINESS-001",
                message=(
                    "The warning count exceeds the "
                    "configured maximum."
                ),
            )

            result.add_finding(
                finding
            )

            blocking.append(
                finding
            )

        blocking = self._deduplicate(
            blocking
        )

        ready = not blocking

        if ready and warning_count > 0:
            status = (
                Phase7Status
                .PASSED_WITH_WARNINGS
            )

        elif ready:
            status = Phase7Status.PASSED

        else:
            status = Phase7Status.FAILED

        decision = Phase7ReadinessDecision(
            ready=ready,
            status=status,
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

        state.ready_for_phase_8 = ready

        result.finalize(
            decision
        )

        return decision

    def _validation_findings(
        self,
        *,
        result: Phase7Result,
        state: Phase7State,
    ) -> list[object]:
        """Convert readiness requirements into findings."""

        findings: list[
            Phase7Finding
        ] = []

        validation = result.validation_results

        required_results = [
            (
                self.options.require_schema_validation,
                validation.schema_consistency,
                "P7-READINESS-002",
                "Schema validation was not completed.",
            ),
            (
                self.options
                .require_feature_count_validation,
                validation.feature_count,
                "P7-READINESS-003",
                "Feature-count validation was not completed.",
            ),
            (
                self.options
                .require_protected_feature_validation,
                validation.protected_features,
                "P7-READINESS-004",
                (
                    "Protected-feature validation was "
                    "not completed."
                ),
            ),
            (
                self.options
                .require_target_integrity_validation,
                validation.target_integrity,
                "P7-READINESS-005",
                "Target-integrity validation was not completed.",
            ),
            (
                self.options
                .require_row_integrity_validation,
                validation.row_integrity,
                "P7-READINESS-006",
                "Row-integrity validation was not completed.",
            ),
        ]

        for required, value, code, message in required_results:
            if required and value is None:
                findings.append(
                    self._blocking_finding(
                        code=code,
                        message=message,
                    )
                )

        invalid_results = [
            (
                self.options.require_consistent_schemas,
                validation.schema_consistency,
                "P7-READINESS-007",
                "Selected partition schemas are inconsistent.",
            ),
            (
                self.options.require_valid_feature_count,
                validation.feature_count,
                "P7-READINESS-008",
                (
                    "The selected feature count violates "
                    "the configured policy."
                ),
            ),
            (
                self.options.require_valid_protected_features,
                validation.protected_features,
                "P7-READINESS-009",
                (
                    "Required, protected, or prohibited "
                    "feature policies were violated."
                ),
            ),
            (
                self.options.require_unchanged_targets,
                validation.target_integrity,
                "P7-READINESS-010",
                "One or more target partitions changed.",
            ),
            (
                self.options.require_unchanged_row_ids,
                validation.row_integrity,
                "P7-READINESS-011",
                (
                    "Row identities or partition memberships "
                    "changed."
                ),
            ),
        ]

        for required, value, code, message in invalid_results:
            if (
                required
                and value is not None
                and not value.valid
            ):
                findings.append(
                    self._blocking_finding(
                        code=code,
                        message=message,
                    )
                )

        state_requirements = [
            (
                self.options.require_training_only_fit
                and not state.fitted_on_training_only,
                "P7-READINESS-012",
                (
                    "Selection state was not fitted "
                    "exclusively on training data."
                ),
            ),
            (
                self.options
                .require_authorized_supervised_target_access
                and state.supervised_target_access_used
                and not state
                .supervised_target_access_authorized,
                "P7-READINESS-013",
                (
                    "Supervised selection used unauthorized "
                    "target access."
                ),
            ),
            (
                self.options
                .require_validation_evaluation_only
                and not state
                .validation_used_for_evaluation_only,
                "P7-READINESS-014",
                (
                    "Validation data affected fitted "
                    "selection state."
                ),
            ),
            (
                self.options.require_consistent_schemas
                and not state.schemas_consistent,
                "P7-READINESS-015",
                (
                    "Selected schemas were not verified "
                    "as consistent."
                ),
            ),
            (
                self.options.require_valid_feature_count
                and not state.feature_count_valid,
                "P7-READINESS-016",
                "Feature-count policy was not verified.",
            ),
            (
                self.options.require_valid_protected_features
                and not state.protected_features_valid,
                "P7-READINESS-017",
                (
                    "Protected-feature policy was not "
                    "verified."
                ),
            ),
            (
                self.options.require_unchanged_selected_values
                and not state.selected_values_unchanged,
                "P7-READINESS-018",
                (
                    "Selected feature values were not "
                    "verified as unchanged."
                ),
            ),
            (
                self.options.require_unchanged_targets
                and not state.targets_unchanged,
                "P7-READINESS-019",
                "Targets were not verified as unchanged.",
            ),
            (
                self.options.require_unchanged_row_ids
                and not state.row_ids_unchanged,
                "P7-READINESS-020",
                "Row IDs were not verified as unchanged.",
            ),
            (
                self.options.require_authorized_test_transform
                and not state.test_transform_authorized,
                "P7-READINESS-021",
                "Test schema application was not authorized.",
            ),
            (
                self.options.require_protected_test_partition
                and not state.test_partition_remained_protected,
                "P7-READINESS-022",
                (
                    "The test partition did not remain "
                    "protected."
                ),
            ),
            (
                self.options.require_selected_lineage
                and state.selected_lineage is None,
                "P7-READINESS-023",
                "Selected feature lineage is missing.",
            ),
            (
                self.options.require_fitted_state_fingerprint
                and state.fitted_state_fingerprint is None,
                "P7-READINESS-024",
                (
                    "The fitted selection-state fingerprint "
                    "is missing."
                ),
            ),
            (
                self.options.require_output_fingerprint
                and state.output_fingerprint is None,
                "P7-READINESS-025",
                "The selected-output fingerprint is missing.",
            ),
        ]

        for condition, code, message in state_requirements:
            if condition:
                findings.append(
                    self._blocking_finding(
                        code=code,
                        message=message,
                    )
                )

        return findings

    def _blocking_finding(
        self,
        *,
        code: str,
        message: str,
    ) -> Phase7Finding:
        """Build one readiness-blocking finding."""

        return Phase7Finding(
            code=code,
            category=(
                Phase7FindingCategory.READINESS
            ),
            severity=(
                Phase7FindingSeverity.ERROR
            ),
            component=self.component_name,
            message=message,
            recommended_action=(
                Phase7FindingAction.REVIEW
            ),
            applied_action=(
                Phase7FindingAction.REPORT
            ),
            blocks_readiness=True,
        )

    def _finding_blocks(
        self,
        finding: Phase7Finding,
    ) -> bool:
        """Return whether one finding blocks readiness."""

        if (
            self.options.block_on_explicit_blockers
            and finding.blocks_readiness
        ):
            return True

        if (
            self.options.block_on_critical_findings
            and finding.severity
            == Phase7FindingSeverity.CRITICAL
        ):
            return True

        return bool(
            self.options.block_on_error_findings
            and finding.severity
            == Phase7FindingSeverity.ERROR
        )

    @staticmethod
    def _deduplicate(
        findings: list[Phase7Finding],
    ) -> list[object]:
        """Deduplicate findings using finding IDs."""

        seen: set[str] = set()
        unique: list[
            Phase7Finding
        ] = []

        for finding in findings:
            if finding.finding_id in seen:
                continue

            seen.add(
                finding.finding_id
            )

            unique.append(
                finding
            )

        return unique
