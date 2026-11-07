"""Phase 7 readiness evaluation for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass

from mlprep.phases.phase6.output_governance.metadata import (
    Phase6Finding,
    Phase6FindingAction,
    Phase6FindingCategory,
    Phase6FindingSeverity,
    Phase6Status,
)
from mlprep.phases.phase6.output_governance.result import (
    Phase6ReadinessDecision,
    Phase6Result,
)
from mlprep.phases.phase6.output_governance.state import (
    Phase6State,
)


@dataclass(slots=True)
class Phase6ReadinessOptions:
    """Options controlling readiness for Phase 7."""

    block_on_critical_findings: bool = True
    block_on_error_findings: bool = True
    block_on_explicit_blockers: bool = True

    require_schema_validation: bool = True
    require_numeric_integrity_validation: bool = True
    require_unknown_category_validation: bool = True
    require_target_integrity_validation: bool = True
    require_row_integrity_validation: bool = True

    require_training_only_fit: bool = True

    require_consistent_schemas: bool = True
    require_numeric_integrity: bool = True
    require_valid_unknown_category_policy: bool = True

    require_unchanged_targets: bool = True
    require_unchanged_row_ids: bool = True

    require_authorized_test_transform: bool = True
    require_protected_test_partition: bool = True

    require_feature_lineage: bool = True
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


class Phase6ReadinessEvaluator:
    """Determine whether Phase 7 may begin."""

    component_name = "readiness"

    def __init__(
        self,
        options: Phase6ReadinessOptions | None = None,
    ) -> None:
        """Initialize the readiness evaluator."""

        self.options = (
            options
            or Phase6ReadinessOptions()
        )

    def evaluate(
        self,
        *,
        result: Phase6Result,
        state: Phase6State,
    ) -> Phase6ReadinessDecision:
        """Evaluate Phase 6 readiness."""

        generated_findings = (
            self._validation_findings(
                result=result,
                state=state,
            )
        )

        result.add_findings(
            generated_findings
        )

        blocking_findings = [
            finding
            for finding in result.findings
            if self._finding_blocks(
                finding
            )
        ]

        warning_count = len(
            result.findings_by_severity(
                Phase6FindingSeverity.WARNING
            )
        )

        if (
            self.options.maximum_warning_count
            is not None
            and warning_count
            > self.options.maximum_warning_count
        ):
            warning_finding = Phase6Finding(
                code="P6-READINESS-001",
                category=(
                    Phase6FindingCategory.READINESS
                ),
                severity=(
                    Phase6FindingSeverity.ERROR
                ),
                component=self.component_name,
                message=(
                    "The warning count exceeds the "
                    "configured maximum."
                ),
                recommended_action=(
                    Phase6FindingAction.REVIEW
                ),
                applied_action=(
                    Phase6FindingAction.REPORT
                ),
                blocks_readiness=True,
                details={
                    "warning_count": (
                        warning_count
                    ),
                    "maximum_warning_count": (
                        self.options
                        .maximum_warning_count
                    ),
                },
            )

            result.add_finding(
                warning_finding
            )

            blocking_findings.append(
                warning_finding
            )

        blocking_findings = (
            self._deduplicate_findings(
                blocking_findings
            )
        )

        ready = not blocking_findings

        if ready and warning_count > 0:
            status = (
                Phase6Status
                .PASSED_WITH_WARNINGS
            )

        elif ready:
            status = Phase6Status.PASSED

        else:
            status = Phase6Status.FAILED

        decision = Phase6ReadinessDecision(
            ready=ready,
            status=status,
            blocking_finding_ids=[
                finding.finding_id
                for finding
                in blocking_findings
            ],
            blocking_finding_codes=[
                finding.code
                for finding
                in blocking_findings
            ],
            reasons=[
                finding.message
                for finding
                in blocking_findings
            ],
        )

        state.ready_for_phase_7 = ready

        result.finalize(
            decision
        )

        return decision

    def _validation_findings(
        self,
        *,
        result: Phase6Result,
        state: Phase6State,
    ) -> list[object]:
        """Convert readiness requirements into findings."""

        findings: list[
            Phase6Finding
        ] = []

        validation = (
            result.validation_results
        )

        required_validations = [
            (
                self.options
                .require_schema_validation,
                validation.schema_consistency,
                "P6-READINESS-002",
                "Schema validation was not completed.",
            ),
            (
                self.options
                .require_numeric_integrity_validation,
                validation.numeric_integrity,
                "P6-READINESS-003",
                (
                    "Numeric-integrity validation was "
                    "not completed."
                ),
            ),
            (
                self.options
                .require_unknown_category_validation,
                validation.unknown_categories,
                "P6-READINESS-004",
                (
                    "Unknown-category validation was "
                    "not completed."
                ),
            ),
            (
                self.options
                .require_target_integrity_validation,
                validation.target_integrity,
                "P6-READINESS-005",
                (
                    "Target-integrity validation was "
                    "not completed."
                ),
            ),
            (
                self.options
                .require_row_integrity_validation,
                validation.row_integrity,
                "P6-READINESS-006",
                (
                    "Row-integrity validation was "
                    "not completed."
                ),
            ),
        ]

        for (
            required,
            validation_result,
            code,
            message,
        ) in required_validations:
            if required and validation_result is None:
                findings.append(
                    self._blocking_finding(
                        code=code,
                        message=message,
                    )
                )

        invalid_validations = [
            (
                self.options
                .require_consistent_schemas,
                validation.schema_consistency,
                "P6-READINESS-007",
                (
                    "Transformed partition schemas "
                    "are inconsistent."
                ),
            ),
            (
                self.options
                .require_numeric_integrity,
                validation.numeric_integrity,
                "P6-READINESS-008",
                (
                    "Transformed numeric features violate "
                    "the output-integrity policy."
                ),
            ),
            (
                self.options
                .require_valid_unknown_category_policy,
                validation.unknown_categories,
                "P6-READINESS-009",
                (
                    "Unknown categorical values violate "
                    "the configured policy."
                ),
            ),
            (
                self.options
                .require_unchanged_targets,
                validation.target_integrity,
                "P6-READINESS-010",
                (
                    "One or more target partitions changed."
                ),
            ),
            (
                self.options
                .require_unchanged_row_ids,
                validation.row_integrity,
                "P6-READINESS-011",
                (
                    "Row identities or partition memberships "
                    "changed."
                ),
            ),
        ]

        for (
            required,
            validation_result,
            code,
            message,
        ) in invalid_validations:
            if (
                required
                and validation_result is not None
                and not validation_result.valid
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
                and not state
                .fitted_on_training_only,
                "P6-READINESS-012",
                (
                    "Transformation state was not fitted "
                    "exclusively on training data."
                ),
            ),
            (
                self.options.require_consistent_schemas
                and not state.schemas_consistent,
                "P6-READINESS-013",
                (
                    "Output schemas were not verified "
                    "as consistent."
                ),
            ),
            (
                self.options.require_numeric_integrity
                and not state
                .numeric_integrity_valid,
                "P6-READINESS-014",
                (
                    "Numeric output integrity was not verified."
                ),
            ),
            (
                self.options
                .require_valid_unknown_category_policy
                and not state
                .unknown_categories_valid,
                "P6-READINESS-015",
                (
                    "Unknown-category handling was not "
                    "verified as valid."
                ),
            ),
            (
                self.options.require_unchanged_targets
                and not state.targets_unchanged,
                "P6-READINESS-016",
                "Targets were not verified as unchanged.",
            ),
            (
                self.options.require_unchanged_row_ids
                and not state.row_ids_unchanged,
                "P6-READINESS-017",
                "Row IDs were not verified as unchanged.",
            ),
            (
                self.options
                .require_authorized_test_transform
                and not state
                .test_transform_authorized,
                "P6-READINESS-018",
                "Test transformation was not authorized.",
            ),
            (
                self.options
                .require_protected_test_partition
                and not state
                .test_partition_remained_protected,
                "P6-READINESS-019",
                (
                    "The test partition did not remain "
                    "protected."
                ),
            ),
            (
                self.options.require_feature_lineage
                and state.lineage is None,
                "P6-READINESS-020",
                "Feature lineage is missing.",
            ),
            (
                self.options
                .require_fitted_state_fingerprint
                and state
                .fitted_state_fingerprint
                is None,
                "P6-READINESS-021",
                (
                    "The fitted transformation-state "
                    "fingerprint is missing."
                ),
            ),
            (
                self.options.require_output_fingerprint
                and state.output_fingerprint is None,
                "P6-READINESS-022",
                "The transformed-output fingerprint is missing.",
            ),
        ]

        for condition, code, message in (
            state_requirements
        ):
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
    ) -> Phase6Finding:
        """Build one readiness-blocking finding."""

        return Phase6Finding(
            code=code,
            category=(
                Phase6FindingCategory.READINESS
            ),
            severity=(
                Phase6FindingSeverity.ERROR
            ),
            component=self.component_name,
            message=message,
            recommended_action=(
                Phase6FindingAction.REVIEW
            ),
            applied_action=(
                Phase6FindingAction.REPORT
            ),
            blocks_readiness=True,
        )

    def _finding_blocks(
        self,
        finding: Phase6Finding,
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
            == Phase6FindingSeverity.CRITICAL
        ):
            return True

        return bool(
            self.options.block_on_error_findings
            and finding.severity
            == Phase6FindingSeverity.ERROR
        )

    @staticmethod
    def _deduplicate_findings(
        findings: list[Phase6Finding],
    ) -> list[object]:
        """Deduplicate findings using finding IDs."""

        seen: set[str] = set()

        unique: list[
            Phase6Finding
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
