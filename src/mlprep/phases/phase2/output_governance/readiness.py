"""Phase 3 readiness evaluation for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    Phase2Status,
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
)
from mlprep.phases.phase2.output_governance.result import (
    Phase2ReadinessDecision,
    Phase2Result,
    SplitFinding,
)
from mlprep.phases.phase2.output_governance.state import (
    Phase2State,
)


@dataclass(slots=True)
class Phase2ReadinessOptions:
    """Options controlling Phase 3 readiness."""

    block_on_critical_findings: bool = True
    block_on_error_findings: bool = True
    block_on_explicit_blockers: bool = True

    require_train_partition: bool = True
    require_validation_partition: bool = True
    require_test_partition: bool = True

    require_disjoint_memberships: bool = True
    require_complete_fingerprints: bool = True
    require_leakage_firewall_pass: bool = True
    require_test_lock: bool = True

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


class Phase2ReadinessEvaluator:
    """Decide whether Phase 3 may begin."""

    component_name = "readiness"

    def __init__(
        self,
        options: Phase2ReadinessOptions | None = None,
    ) -> None:
        """Initialize the readiness evaluator."""

        self.options = (
            options or Phase2ReadinessOptions()
        )

    def evaluate(
        self,
        *,
        result: Phase2Result,
        state: Phase2State,
    ) -> Phase2ReadinessDecision:
        """Evaluate all Phase 2 readiness conditions."""

        blocking_findings = [
            finding
            for finding in result.findings
            if self._finding_blocks(finding)
        ]

        structural_reasons: list[str] = []

        required_partitions = []

        if self.options.require_train_partition:
            required_partitions.append(
                PartitionName.TRAIN
            )

        if self.options.require_validation_partition:
            required_partitions.append(
                PartitionName.VALIDATION
            )

        if self.options.require_test_partition:
            required_partitions.append(
                PartitionName.TEST
            )

        for partition in required_partitions:
            membership = state.memberships.get(
                partition
            )

            if (
                membership is None
                or membership.row_count == 0
            ):
                structural_reasons.append(
                    f"Partition {partition.value!r} "
                    "is missing or empty."
                )

        if self.options.require_disjoint_memberships:
            try:
                state.validate_disjoint_memberships()
            except ValueError as error:
                structural_reasons.append(
                    str(error)
                )

        if (
            self.options.require_complete_fingerprints
            and (
                state.train_fingerprint is None
                or state.validation_fingerprint is None
                or state.test_fingerprint is None
                or state.split_fingerprint is None
            )
        ):
            structural_reasons.append(
                "One or more required Phase 2 "
                "fingerprints are missing."
            )

        if (
            self.options.require_leakage_firewall_pass
            and not state.leakage_firewall.passed
        ):
            structural_reasons.append(
                "The leakage firewall did not pass."
            )

        if (
            self.options.require_test_lock
            and not state.test_lock.locked
        ):
            structural_reasons.append(
                "The test partition is not locked."
            )

        warning_count = len(
            result.get_findings_by_severity(
                SplitFindingSeverity.WARNING
            )
        )

        if (
            self.options.maximum_warning_count is not None
            and warning_count
            > self.options.maximum_warning_count
        ):
            structural_reasons.append(
                "The warning count exceeds the "
                "configured readiness limit."
            )

        for reason in structural_reasons:
            finding = SplitFinding(
                code="P2-READINESS-001",
                category=(
                    SplitFindingCategory.READINESS
                ),
                severity=SplitFindingSeverity.ERROR,
                component=self.component_name,
                message=reason,
                recommended_action=(
                    SplitFindingAction.RAISE
                ),
                applied_action=(
                    SplitFindingAction.REPORT
                ),
                blocks_readiness=True,
            )

            result.add_finding(finding)
            blocking_findings.append(finding)

        blocking_findings = (
            self._deduplicate_findings(
                blocking_findings
            )
        )

        ready = not blocking_findings

        if ready and result.has_warnings():
            status = (
                Phase2Status.PASSED_WITH_WARNINGS
            )
        elif ready:
            status = Phase2Status.PASSED
        else:
            status = Phase2Status.FAILED

        decision = Phase2ReadinessDecision(
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

        result.finalize(decision)

        return decision

    def _finding_blocks(
        self,
        finding: SplitFinding,
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
            == SplitFindingSeverity.CRITICAL
        ):
            return True

        return bool(
            self.options.block_on_error_findings
            and finding.severity
            == SplitFindingSeverity.ERROR
        )

    @staticmethod
    def _deduplicate_findings(
        findings: list[SplitFinding],
    ) -> list[object]:
        """Deduplicate findings using their IDs."""

        seen_ids: set[str] = set()
        unique_findings: list[
            SplitFinding
        ] = []

        for finding in findings:
            if finding.finding_id in seen_ids:
                continue

            seen_ids.add(finding.finding_id)
            unique_findings.append(finding)

        return unique_findings
