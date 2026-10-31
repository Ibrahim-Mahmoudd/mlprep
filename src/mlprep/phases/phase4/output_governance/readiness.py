"""Phase 5 readiness evaluation for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass

from mlprep.phases.phase4.output_governance.metadata import (
    Phase4FindingAction,
    Phase4FindingCategory,
    Phase4FindingSeverity,
    Phase4Status,
)
from mlprep.phases.phase4.output_governance.result import (
    Phase4Finding,
    Phase4ReadinessDecision,
    Phase4Result,
)
from mlprep.phases.phase4.output_governance.state import (
    Phase4State,
)


@dataclass(slots=True)
class Phase4ReadinessOptions:
    """Options controlling readiness for Phase 5."""

    block_on_critical_findings: bool = True
    block_on_error_findings: bool = True
    block_on_explicit_blockers: bool = True

    require_remaining_missingness_validation: bool = True
    require_schema_consistency_validation: bool = True
    require_target_integrity_validation: bool = True
    require_row_integrity_validation: bool = True
    require_distribution_impact_validation: bool = True

    require_no_blocking_missingness: bool = True
    require_consistent_schemas: bool = True
    require_unchanged_targets: bool = True
    require_unchanged_row_ids: bool = True

    require_training_only_fit: bool = True
    require_locked_test_partition: bool = True
    require_authorized_test_transform: bool = True

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


class Phase4ReadinessEvaluator:
    """Determine whether Phase 5 may begin."""

    component_name = "readiness"

    def __init__(
        self,
        options: Phase4ReadinessOptions | None = None,
    ) -> None:
        """Initialize the readiness evaluator."""

        self.options = (
            options or Phase4ReadinessOptions()
        )

    def evaluate(
        self,
        *,
        result: Phase4Result,
        state: Phase4State,
    ) -> Phase4ReadinessDecision:
        """Evaluate Phase 4 readiness."""

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
                Phase4FindingSeverity.WARNING
            )
        )

        if (
            self.options.maximum_warning_count
            is not None
            and warning_count
            > self.options.maximum_warning_count
        ):
            warning_limit_finding = Phase4Finding(
                code="P4-READINESS-001",
                category=(
                    Phase4FindingCategory.READINESS
                ),
                severity=(
                    Phase4FindingSeverity.ERROR
                ),
                component=self.component_name,
                message=(
                    "The Phase 4 warning count exceeds "
                    "the configured maximum."
                ),
                recommended_action=(
                    Phase4FindingAction.REVIEW
                ),
                applied_action=(
                    Phase4FindingAction.REPORT
                ),
                blocks_readiness=True,
                details={
                    "warning_count": warning_count,
                    "maximum_warning_count": (
                        self.options
                        .maximum_warning_count
                    ),
                },
            )

            result.add_finding(
                warning_limit_finding
            )

            blocking_findings.append(
                warning_limit_finding
            )

        blocking_findings = (
            self._deduplicate_findings(
                blocking_findings
            )
        )

        ready = not blocking_findings

        if ready and warning_count > 0:
            status = (
                Phase4Status
                .PASSED_WITH_WARNINGS
            )

        elif ready:
            status = Phase4Status.PASSED

        else:
            status = Phase4Status.FAILED

        decision = Phase4ReadinessDecision(
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

        state.ready_for_phase_5 = ready

        result.finalize(
            decision
        )

        return decision

    def _validation_findings(
        self,
        *,
        result: Phase4Result,
        state: Phase4State,
    ) -> list[object]:
        """Convert missing readiness requirements into findings."""

        findings: list[Phase4Finding] = []

        validation = (
            result.validation_results
        )

        if (
            self.options
            .require_remaining_missingness_validation
            and validation
            .remaining_missingness
            is None
        ):
            findings.append(
                self._missing_validation_finding(
                    code="P4-READINESS-002",
                    message=(
                        "Remaining-missingness validation "
                        "was not completed."
                    ),
                )
            )

        elif (
            self.options
            .require_no_blocking_missingness
            and validation
            .remaining_missingness
            is not None
            and not validation
            .remaining_missingness
            .valid
        ):
            findings.append(
                Phase4Finding(
                    code="P4-READINESS-003",
                    category=(
                        Phase4FindingCategory
                        .VALIDATION
                    ),
                    severity=(
                        Phase4FindingSeverity.ERROR
                    ),
                    component=self.component_name,
                    message=(
                        "Unresolved missing values violate "
                        "the Phase 4 policy."
                    ),
                    columns=sorted(
                        {
                            column
                            for columns
                            in validation
                            .remaining_missingness
                            .blocking_columns
                            .values()
                            for column in columns
                        }
                    ),
                    recommended_action=(
                        Phase4FindingAction
                        .REIMPUTE
                    ),
                    applied_action=(
                        Phase4FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        if (
            self.options
            .require_schema_consistency_validation
            and validation
            .schema_consistency
            is None
        ):
            findings.append(
                self._missing_validation_finding(
                    code="P4-READINESS-004",
                    message=(
                        "Schema-consistency validation "
                        "was not completed."
                    ),
                )
            )

        elif (
            self.options.require_consistent_schemas
            and validation.schema_consistency
            is not None
            and not validation
            .schema_consistency
            .valid
        ):
            findings.append(
                Phase4Finding(
                    code="P4-READINESS-005",
                    category=(
                        Phase4FindingCategory.SCHEMA
                    ),
                    severity=(
                        Phase4FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "Transformed partition schemas "
                        "are inconsistent."
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

        if (
            self.options
            .require_target_integrity_validation
            and validation.target_integrity
            is None
        ):
            findings.append(
                self._missing_validation_finding(
                    code="P4-READINESS-006",
                    message=(
                        "Target-integrity validation "
                        "was not completed."
                    ),
                )
            )

        elif (
            self.options.require_unchanged_targets
            and validation.target_integrity
            is not None
            and not validation
            .target_integrity
            .valid
        ):
            findings.append(
                Phase4Finding(
                    code="P4-READINESS-007",
                    category=(
                        Phase4FindingCategory
                        .TARGET_INTEGRITY
                    ),
                    severity=(
                        Phase4FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "One or more target partitions changed "
                        "during Phase 4."
                    ),
                    recommended_action=(
                        Phase4FindingAction
                        .RESTORE_INPUT
                    ),
                    applied_action=(
                        Phase4FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        if (
            self.options
            .require_row_integrity_validation
            and validation.row_integrity
            is None
        ):
            findings.append(
                self._missing_validation_finding(
                    code="P4-READINESS-008",
                    message=(
                        "Row-integrity validation "
                        "was not completed."
                    ),
                )
            )

        elif (
            self.options.require_unchanged_row_ids
            and validation.row_integrity
            is not None
            and not validation
            .row_integrity
            .valid
        ):
            findings.append(
                Phase4Finding(
                    code="P4-READINESS-009",
                    category=(
                        Phase4FindingCategory
                        .ROW_INTEGRITY
                    ),
                    severity=(
                        Phase4FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "Row identities or partition memberships "
                        "changed during Phase 4."
                    ),
                    recommended_action=(
                        Phase4FindingAction
                        .RESTORE_INPUT
                    ),
                    applied_action=(
                        Phase4FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        if (
            self.options
            .require_distribution_impact_validation
            and validation
            .distribution_impact
            is None
        ):
            findings.append(
                self._missing_validation_finding(
                    code="P4-READINESS-010",
                    message=(
                        "Distribution-impact validation "
                        "was not completed."
                    ),
                )
            )

        elif (
            validation.distribution_impact
            is not None
            and not validation
            .distribution_impact
            .valid
        ):
            findings.append(
                Phase4Finding(
                    code="P4-READINESS-011",
                    category=(
                        Phase4FindingCategory
                        .DISTRIBUTION_IMPACT
                    ),
                    severity=(
                        Phase4FindingSeverity.ERROR
                    ),
                    component=self.component_name,
                    message=(
                        "Imputation produced a blocking "
                        "distribution impact."
                    ),
                    columns=list(
                        validation
                        .distribution_impact
                        .high_impact_columns
                    ),
                    recommended_action=(
                        Phase4FindingAction
                        .REIMPUTE
                    ),
                    applied_action=(
                        Phase4FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        if (
            self.options.require_training_only_fit
            and not state.fitted_on_training_only
        ):
            findings.append(
                Phase4Finding(
                    code="P4-READINESS-012",
                    category=(
                        Phase4FindingCategory
                        .IMPUTATION
                    ),
                    severity=(
                        Phase4FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "The imputation state was not fitted "
                        "exclusively on training data."
                    ),
                    recommended_action=(
                        Phase4FindingAction
                        .REIMPUTE
                    ),
                    applied_action=(
                        Phase4FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        if (
            self.options
            .require_locked_test_partition
            and not state
            .test_partition_remained_locked
        ):
            findings.append(
                Phase4Finding(
                    code="P4-READINESS-013",
                    category=(
                        Phase4FindingCategory
                        .TEST_GOVERNANCE
                    ),
                    severity=(
                        Phase4FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "The test partition did not remain "
                        "protected during Phase 4."
                    ),
                    partitions=["test"],
                    recommended_action=(
                        Phase4FindingAction
                        .RESTORE_INPUT
                    ),
                    applied_action=(
                        Phase4FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        if (
            self.options
            .require_authorized_test_transform
            and not state
            .test_transform_authorized
        ):
            findings.append(
                Phase4Finding(
                    code="P4-READINESS-014",
                    category=(
                        Phase4FindingCategory
                        .TEST_GOVERNANCE
                    ),
                    severity=(
                        Phase4FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "The test transformation was not "
                        "authorized through the test-lock policy."
                    ),
                    partitions=["test"],
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
            self.options
            .require_fitted_state_fingerprint
            and state.fitted_state_fingerprint
            is None
        ):
            findings.append(
                self._missing_fingerprint_finding(
                    code="P4-READINESS-015",
                    message=(
                        "The fitted imputation-state "
                        "fingerprint is missing."
                    ),
                )
            )

        if (
            self.options.require_output_fingerprint
            and state.output_fingerprint is None
        ):
            findings.append(
                self._missing_fingerprint_finding(
                    code="P4-READINESS-016",
                    message=(
                        "The transformed-output fingerprint "
                        "is missing."
                    ),
                )
            )

        return findings

    def _finding_blocks(
        self,
        finding: Phase4Finding,
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
            == Phase4FindingSeverity.CRITICAL
        ):
            return True

        return bool(
            self.options.block_on_error_findings
            and finding.severity
            == Phase4FindingSeverity.ERROR
        )

    def _missing_validation_finding(
        self,
        *,
        code: str,
        message: str,
    ) -> Phase4Finding:
        """Build a missing-validation finding."""

        return Phase4Finding(
            code=code,
            category=(
                Phase4FindingCategory.READINESS
            ),
            severity=(
                Phase4FindingSeverity.ERROR
            ),
            component=self.component_name,
            message=message,
            recommended_action=(
                Phase4FindingAction.REVIEW
            ),
            applied_action=(
                Phase4FindingAction.REPORT
            ),
            blocks_readiness=True,
        )

    def _missing_fingerprint_finding(
        self,
        *,
        code: str,
        message: str,
    ) -> Phase4Finding:
        """Build a missing-fingerprint finding."""

        return Phase4Finding(
            code=code,
            category=(
                Phase4FindingCategory.FINGERPRINT
            ),
            severity=(
                Phase4FindingSeverity.ERROR
            ),
            component=self.component_name,
            message=message,
            recommended_action=(
                Phase4FindingAction.RECONFIGURE
            ),
            applied_action=(
                Phase4FindingAction.REPORT
            ),
            blocks_readiness=True,
        )

    @staticmethod
    def _deduplicate_findings(
        findings: list[Phase4Finding],
    ) -> list[object]:
        """Deduplicate findings using their IDs."""

        seen: set[str] = set()
        unique: list[Phase4Finding] = []

        for finding in findings:
            if finding.finding_id in seen:
                continue

            seen.add(
                finding.finding_id
            )

            unique.append(finding)

        return unique