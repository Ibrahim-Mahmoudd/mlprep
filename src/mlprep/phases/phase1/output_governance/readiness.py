"""Phase 2 readiness evaluation for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
    ReadinessDecision,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.types import (
    FindingAction,
    FindingCategory,
    PhaseStatus,
    Severity,
)


@dataclass(slots=True)
class ReadinessOptions:
    """Options controlling the Phase 2 readiness decision."""

    block_on_critical_findings: bool = True
    block_on_error_findings: bool = True
    block_on_explicit_blockers: bool = True

    require_problem_contract: bool = True
    require_resolved_problem_contract: bool = True
    require_canonical_schema: bool = True
    require_dataset_fingerprint: bool = True

    require_model_features: bool = True
    require_inference_columns: bool = True

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


class ReadinessEvaluator:
    """Decide whether the dataset may proceed to Phase 2."""

    component_name = "readiness"

    def __init__(
        self,
        options: ReadinessOptions | None = None,
    ) -> None:
        """Initialize the readiness evaluator."""

        self.options = options or ReadinessOptions()

    def evaluate(
        self,
        result: Phase1Result,
        state: Phase1State,
    ) -> ReadinessDecision:
        """Evaluate all readiness conditions."""

        blocking_findings: list[Finding] = []
        structural_reasons: list[str] = []

        for finding in result.findings:
            if self._finding_blocks(finding):
                blocking_findings.append(finding)

        if (
            self.options.require_problem_contract
            and state.problem_contract is None
        ):
            structural_reasons.append(
                "The resolved problem contract is missing."
            )

        if (
            self.options.require_resolved_problem_contract
            and (
                state.problem_contract is None
                or not state.problem_contract.resolved
            )
        ):
            structural_reasons.append(
                "The problem contract is not fully resolved."
            )

        if (
            self.options.require_canonical_schema
            and not result.canonical_schema
        ):
            structural_reasons.append(
                "The canonical schema is missing."
            )

        if (
            self.options.require_dataset_fingerprint
            and state.dataset_fingerprint is None
        ):
            structural_reasons.append(
                "The dataset fingerprint is missing."
            )

        if (
            self.options.require_model_features
            and not state.get_model_feature_columns()
        ):
            structural_reasons.append(
                "No model-eligible feature columns exist."
            )

        if (
            self.options.require_inference_columns
            and not state.get_inference_columns()
        ):
            structural_reasons.append(
                "No inference-time columns exist."
            )

        warning_count = len(
            result.get_findings_by_severity(
                Severity.WARNING
            )
        )

        if (
            self.options.maximum_warning_count is not None
            and warning_count
            > self.options.maximum_warning_count
        ):
            structural_reasons.append(
                "The number of warnings exceeds the "
                "configured readiness limit."
            )

        for reason in structural_reasons:
            finding = Finding(
                code="P1-READINESS-001",
                category=FindingCategory.READINESS,
                severity=Severity.ERROR,
                component=self.component_name,
                message=reason,
                recommended_action=FindingAction.RAISE,
                applied_action=FindingAction.REPORT,
                blocks_readiness=True,
            )

            result.add_finding(finding)
            blocking_findings.append(finding)

        blocking_findings = self._deduplicate_findings(
            blocking_findings
        )

        ready = not blocking_findings

        if ready and result.has_warnings():
            status = PhaseStatus.PASSED_WITH_WARNINGS
        elif ready:
            status = PhaseStatus.PASSED
        else:
            status = PhaseStatus.FAILED

        decision = ReadinessDecision(
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
        finding: Finding,
    ) -> bool:
        """Return whether one finding blocks readiness."""

        if (
            self.options.block_on_explicit_blockers
            and finding.blocks_readiness
        ):
            return True

        if (
            self.options.block_on_critical_findings
            and finding.severity == Severity.CRITICAL
        ):
            return True

        return bool(
            self.options.block_on_error_findings
            and finding.severity == Severity.ERROR
        )

    @staticmethod
    def _deduplicate_findings(
        findings: list[Finding],
    ) -> list[object]:
        """Deduplicate findings by their unique ID."""

        seen_ids: set[str] = set()
        unique_findings: list[Finding] = []

        for finding in findings:
            if finding.finding_id in seen_ids:
                continue

            seen_ids.add(finding.finding_id)
            unique_findings.append(finding)

        return unique_findings
