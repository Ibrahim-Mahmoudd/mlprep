"""Phase 4 readiness evaluation for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

from mlprep.phases.phase3.output_governance.result import (
    Phase3Finding,
    Phase3FindingCategory,
    Phase3FindingSeverity,
    Phase3ReadinessDecision,
    Phase3Result,
    Phase3Status,
)
from mlprep.phases.phase3.output_governance.state import (
    Phase3State,
)


@dataclass(slots=True)
class Phase3ReadinessOptions:
    """Options controlling readiness for Phase 4."""

    block_on_critical_findings: bool = True
    block_on_error_findings: bool = True
    block_on_explicit_blockers: bool = True

    require_dataset_profile: bool = True
    require_target_profile: bool = True
    require_missingness_profile: bool = True
    require_recommendations: bool = True

    require_training_unchanged: bool = True
    require_validation_unchanged: bool = True

    require_locked_test_partition: bool = True
    prohibit_test_access_during_phase3: bool = True

    require_training_only_analysis: bool = True

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


class Phase3ReadinessEvaluator:
    """Determine whether Phase 4 may begin."""

    component_name = "readiness"

    def __init__(
        self,
        options: Phase3ReadinessOptions | None = None,
    ) -> None:
        """Initialize the readiness evaluator."""

        self.options = (
            options or Phase3ReadinessOptions()
        )

    def evaluate(
        self,
        *,
        result: Phase3Result,
        state: Phase3State,
    ) -> Phase3ReadinessDecision:
        """Evaluate Phase 3 readiness."""

        blocking_findings = [
            finding
            for finding in result.findings
            if self._finding_blocks(
                finding
            )
        ]

        reasons: list[str] = []

        if (
            self.options.require_dataset_profile
            and result.dataset_profile is None
        ):
            reasons.append(
                "The dataset profile is missing."
            )

        if (
            self.options.require_target_profile
            and result.target_profile is None
        ):
            reasons.append(
                "The target profile is missing."
            )

        if (
            self.options.require_missingness_profile
            and result.missingness_profile is None
        ):
            reasons.append(
                "The missingness profile is missing."
            )

        if (
            self.options.require_recommendations
            and result.recommendations is None
        ):
            reasons.append(
                "Downstream recommendations are missing."
            )

        if (
            self.options.require_training_unchanged
            and not state.training_data_unchanged
        ):
            reasons.append(
                "The training partition changed during Phase 3."
            )

        if (
            self.options.require_validation_unchanged
            and not state.validation_data_unchanged
        ):
            reasons.append(
                "The validation partition changed during Phase 3."
            )

        if (
            self.options.require_locked_test_partition
            and not state.test_partition_remained_locked
        ):
            reasons.append(
                "The test partition did not remain locked."
            )

        if (
            self.options
            .prohibit_test_access_during_phase3
            and (
                state.test_partition_access_count_after
                != state.test_partition_access_count_before
            )
        ):
            reasons.append(
                "The test partition was accessed during Phase 3."
            )

        if (
            self.options.require_training_only_analysis
            and not state.analyzed_training_only
        ):
            reasons.append(
                "Phase 3 was not restricted to training-only "
                "analytical decisions."
            )

        warning_count = len(
            result.get_findings_by_severity(
                Phase3FindingSeverity.WARNING
            )
        )

        if (
            self.options.maximum_warning_count is not None
            and warning_count
            > self.options.maximum_warning_count
        ):
            reasons.append(
                "The warning count exceeds the configured limit."
            )

        for reason in reasons:
            finding = Phase3Finding(
                code="P3-READINESS-001",
                category=(
                    Phase3FindingCategory.READINESS
                ),
                severity=Phase3FindingSeverity.ERROR,
                component=self.component_name,
                message=reason,
                blocks_readiness=True,
            )

            result.add_finding(finding)
            blocking_findings.append(finding)

        blocking_findings = self._deduplicate(
            blocking_findings
        )

        ready = not blocking_findings

        if ready and result.has_warnings():
            status = Phase3Status.PASSED_WITH_WARNINGS

        elif ready:
            status = Phase3Status.PASSED

        else:
            status = Phase3Status.FAILED

        decision = Phase3ReadinessDecision(
            ready=ready,
            status=status,
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

        state.ready_for_phase_4 = ready

        result.finalize(
            decision
        )

        return decision

    def _finding_blocks(
        self,
        finding: Phase3Finding,
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
            == Phase3FindingSeverity.CRITICAL
        ):
            return True

        return bool(
            self.options.block_on_error_findings
            and finding.severity
            == Phase3FindingSeverity.ERROR
        )

    @staticmethod
    def _deduplicate(
        findings: list[Phase3Finding],
    ) -> list[object]:
        """Deduplicate findings by finding ID."""

        seen: set[str] = set()
        results: list[Phase3Finding] = []

        for finding in findings:
            if finding.finding_id in seen:
                continue

            seen.add(
                finding.finding_id
            )

            results.append(finding)

        return results
