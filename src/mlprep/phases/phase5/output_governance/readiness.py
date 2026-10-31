"""Phase 6 readiness evaluation for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass

from mlprep.phases.phase5.output_governance.metadata import (
    Phase5Finding,
    Phase5FindingAction,
    Phase5FindingCategory,
    Phase5FindingSeverity,
    Phase5Status,
)
from mlprep.phases.phase5.output_governance.result import (
    Phase5ReadinessDecision,
    Phase5Result,
)
from mlprep.phases.phase5.output_governance.state import (
    Phase5State,
)


@dataclass(slots=True)
class Phase5ReadinessOptions:
    """Options controlling readiness for Phase 6."""

    block_on_critical_findings: bool = True
    block_on_error_findings: bool = True
    block_on_explicit_blockers: bool = True

    require_remaining_outlier_validation: bool = True
    require_schema_validation: bool = True
    require_target_integrity_validation: bool = True
    require_row_integrity_validation: bool = True
    require_distribution_impact_validation: bool = True

    require_training_only_fit: bool = True

    require_unchanged_targets: bool = True
    require_unchanged_row_ids: bool = True
    require_consistent_schemas: bool = True

    require_authorized_test_transform: bool = True
    require_protected_test_partition: bool = True

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


class Phase5ReadinessEvaluator:
    """Determine whether Phase 6 may begin."""

    component_name = "readiness"

    def __init__(
        self,
        options: Phase5ReadinessOptions | None = None,
    ) -> None:
        """Initialize the evaluator."""

        self.options = (
            options
            or Phase5ReadinessOptions()
        )

    def evaluate(
        self,
        *,
        result: Phase5Result,
        state: Phase5State,
    ) -> Phase5ReadinessDecision:
        """Evaluate Phase 5 readiness."""

        findings = self._validation_findings(
            result=result,
            state=state,
        )

        result.add_findings(
            findings
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
                Phase5FindingSeverity.WARNING
            )
        )

        if (
            self.options.maximum_warning_count
            is not None
            and warning_count
            > self.options.maximum_warning_count
        ):
            warning_finding = Phase5Finding(
                code="P5-READINESS-001",
                category=(
                    Phase5FindingCategory.READINESS
                ),
                severity=(
                    Phase5FindingSeverity.ERROR
                ),
                component=self.component_name,
                message=(
                    "The warning count exceeds the "
                    "configured maximum."
                ),
                recommended_action=(
                    Phase5FindingAction.REVIEW
                ),
                applied_action=(
                    Phase5FindingAction.REPORT
                ),
                blocks_readiness=True,
            )

            result.add_finding(
                warning_finding
            )

            blocking.append(
                warning_finding
            )

        blocking = self._deduplicate(
            blocking
        )

        ready = not blocking

        if ready and warning_count > 0:
            status = (
                Phase5Status
                .PASSED_WITH_WARNINGS
            )

        elif ready:
            status = Phase5Status.PASSED

        else:
            status = Phase5Status.FAILED

        decision = Phase5ReadinessDecision(
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

        state.ready_for_phase_6 = ready

        result.finalize(
            decision
        )

        return decision

    def _validation_findings(
        self,
        *,
        result: Phase5Result,
        state: Phase5State,
    ) -> list[object]:
        """Convert readiness requirements into findings."""

        findings: list[
            Phase5Finding
        ] = []

        validation = (
            result.validation_results
        )

        required_results = [
            (
                self.options
                .require_remaining_outlier_validation,
                validation.remaining_outliers,
                "P5-READINESS-002",
                "Remaining-outlier validation was not completed.",
            ),
            (
                self.options
                .require_schema_validation,
                validation.schema_consistency,
                "P5-READINESS-003",
                "Schema validation was not completed.",
            ),
            (
                self.options
                .require_target_integrity_validation,
                validation.target_integrity,
                "P5-READINESS-004",
                "Target-integrity validation was not completed.",
            ),
            (
                self.options
                .require_row_integrity_validation,
                validation.row_integrity,
                "P5-READINESS-005",
                "Row-integrity validation was not completed.",
            ),
            (
                self.options
                .require_distribution_impact_validation,
                validation.distribution_impact,
                "P5-READINESS-006",
                "Distribution-impact validation was not completed.",
            ),
        ]

        for (
            required,
            validation_result,
            code,
            message,
        ) in required_results:
            if required and validation_result is None:
                findings.append(
                    self._blocking_finding(
                        code=code,
                        message=message,
                    )
                )

        invalid_results = [
            (
                validation.remaining_outliers,
                "P5-READINESS-007",
                (
                    "Remaining outliers or nonfinite values "
                    "violate the Phase 5 policy."
                ),
            ),
            (
                validation.schema_consistency,
                "P5-READINESS-008",
                (
                    "Transformed partition schemas "
                    "are inconsistent."
                ),
            ),
            (
                validation.target_integrity,
                "P5-READINESS-009",
                (
                    "One or more target partitions changed."
                ),
            ),
            (
                validation.row_integrity,
                "P5-READINESS-010",
                (
                    "Row identities or partition memberships "
                    "changed."
                ),
            ),
            (
                validation.distribution_impact,
                "P5-READINESS-011",
                (
                    "Treatment produced a blocking "
                    "distribution impact."
                ),
            ),
        ]

        for (
            validation_result,
            code,
            message,
        ) in invalid_results:
            if (
                validation_result is not None
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
                "P5-READINESS-012",
                (
                    "Detectors or strategies were not fitted "
                    "exclusively on training data."
                ),
            ),
            (
                self.options.require_unchanged_targets
                and not state.targets_unchanged,
                "P5-READINESS-013",
                "Targets were not verified as unchanged.",
            ),
            (
                self.options.require_unchanged_row_ids
                and not state.row_ids_unchanged,
                "P5-READINESS-014",
                "Row IDs were not verified as unchanged.",
            ),
            (
                self.options.require_consistent_schemas
                and not state.schemas_consistent,
                "P5-READINESS-015",
                "Output schemas were not verified as consistent.",
            ),
            (
                self.options
                .require_authorized_test_transform
                and not state
                .test_transform_authorized,
                "P5-READINESS-016",
                "Test transformation was not authorized.",
            ),
            (
                self.options
                .require_protected_test_partition
                and not state
                .test_partition_remained_protected,
                "P5-READINESS-017",
                "The test partition did not remain protected.",
            ),
            (
                self.options
                .require_fitted_state_fingerprint
                and state
                .fitted_state_fingerprint
                is None,
                "P5-READINESS-018",
                "The fitted-state fingerprint is missing.",
            ),
            (
                self.options
                .require_output_fingerprint
                and state.output_fingerprint
                is None,
                "P5-READINESS-019",
                "The output fingerprint is missing.",
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
    ) -> Phase5Finding:
        """Create one readiness-blocking finding."""

        return Phase5Finding(
            code=code,
            category=(
                Phase5FindingCategory.READINESS
            ),
            severity=(
                Phase5FindingSeverity.ERROR
            ),
            component=self.component_name,
            message=message,
            recommended_action=(
                Phase5FindingAction.REVIEW
            ),
            applied_action=(
                Phase5FindingAction.REPORT
            ),
            blocks_readiness=True,
        )

    def _finding_blocks(
        self,
        finding: Phase5Finding,
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
            == Phase5FindingSeverity.CRITICAL
        ):
            return True

        return bool(
            self.options.block_on_error_findings
            and finding.severity
            == Phase5FindingSeverity.ERROR
        )

    @staticmethod
    def _deduplicate(
        findings: list[Phase5Finding],
    ) -> list[object]:
        """Deduplicate findings using finding IDs."""

        seen: set[str] = set()
        unique: list[
            Phase5Finding
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