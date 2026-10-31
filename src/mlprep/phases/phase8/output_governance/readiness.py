"""Phase 9 readiness evaluation for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass

from mlprep.phases.phase8.output_governance.metadata import (
    Phase8Finding,
    Phase8FindingAction,
    Phase8FindingCategory,
    Phase8FindingSeverity,
    Phase8Status,
)
from mlprep.phases.phase8.output_governance.result import (
    Phase8ReadinessDecision,
    Phase8Result,
)
from mlprep.phases.phase8.output_governance.state import (
    Phase8State,
)


@dataclass(slots=True)
class Phase8ReadinessOptions:
    """Options controlling readiness for Phase 9."""

    block_on_critical_findings: bool = True
    block_on_error_findings: bool = True
    block_on_explicit_blockers: bool = True

    require_distribution_validation: bool = True
    require_schema_validation: bool = True
    require_synthetic_integrity_validation: bool = True
    require_target_integrity_validation: bool = True
    require_partition_integrity_validation: bool = True

    require_training_only_fit: bool = True
    require_authorized_target_access: bool = True
    require_authorized_resampling: bool = True
    require_authorized_synthetic_generation: bool = True

    require_unchanged_validation: bool = True
    require_unchanged_test: bool = True

    require_complete_lineage: bool = True

    require_selected_feature_lineage: bool = True

    require_valid_class_weights: bool = True
    require_valid_sample_weights: bool = True

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


class Phase8ReadinessEvaluator:
    """Determine whether Phase 9 may begin."""

    component_name = "readiness"

    def __init__(
        self,
        options: Phase8ReadinessOptions | None = None,
    ) -> None:
        """Initialize the readiness evaluator."""

        self.options = (
            options
            or Phase8ReadinessOptions()
        )

    def evaluate(
        self,
        *,
        result: Phase8Result,
        state: Phase8State,
    ) -> Phase8ReadinessDecision:
        """Evaluate Phase 8 readiness."""

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
                Phase8FindingSeverity.WARNING
            )
        )

        if (
            self.options.maximum_warning_count
            is not None
            and warning_count
            > self.options.maximum_warning_count
        ):
            finding = self._blocking_finding(
                code="P8-READINESS-001",
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

        blocking = self._deduplicate_findings(
            blocking
        )

        ready = not blocking

        if ready and warning_count > 0:
            status = (
                Phase8Status.PASSED_WITH_WARNINGS
            )

        elif ready:
            status = Phase8Status.PASSED

        else:
            status = Phase8Status.FAILED

        decision = Phase8ReadinessDecision(
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

        state.ready_for_phase_9 = ready

        result.finalize(
            decision
        )

        return decision

    def _validation_findings(
        self,
        *,
        result: Phase8Result,
        state: Phase8State,
    ) -> list[object]:
        """Convert readiness requirements into findings."""

        findings: list[
            Phase8Finding
        ] = []

        validation = result.validation_results

        requirements = [
            (
                self.options
                .require_distribution_validation,
                validation.distribution,
                state.distribution_valid,
                "P8-READINESS-002",
                (
                    "The output class distribution is "
                    "invalid or was not validated."
                ),
            ),
            (
                self.options.require_schema_validation,
                validation.schema_consistency,
                state.schema_consistent,
                "P8-READINESS-003",
                (
                    "The Phase 8 output schema is invalid "
                    "or was not validated."
                ),
            ),
            (
                self.options
                .require_synthetic_integrity_validation,
                validation.synthetic_integrity,
                state.synthetic_integrity_valid,
                "P8-READINESS-004",
                (
                    "Synthetic-row integrity is invalid "
                    "or was not validated."
                ),
            ),
            (
                self.options
                .require_target_integrity_validation,
                validation.target_integrity,
                state.target_integrity_valid,
                "P8-READINESS-005",
                (
                    "Target integrity is invalid or "
                    "was not validated."
                ),
            ),
            (
                self.options
                .require_partition_integrity_validation,
                validation.partition_integrity,
                state.partition_integrity_valid,
                "P8-READINESS-006",
                (
                    "Partition integrity is invalid or "
                    "was not validated."
                ),
            ),
        ]

        for (
            required,
            result_value,
            state_value,
            code,
            message,
        ) in requirements:
            if (
                required
                and (
                    result_value is None
                    or not state_value
                )
            ):
                findings.append(
                    self._blocking_finding(
                        code=code,
                        message=message,
                    )
                )

        state_checks = [
            (
                self.options.require_training_only_fit
                and not state.fitted_on_training_only,
                "P8-READINESS-007",
                (
                    "Balancing state was not fitted "
                    "exclusively on training data."
                ),
            ),
            (
                self.options
                .require_authorized_target_access
                and not state
                .target_access_authorized,
                "P8-READINESS-008",
                (
                    "Training-target access was not "
                    "authorized."
                ),
            ),
            (
                self.options
                .require_authorized_resampling
                and state.executed_strategy
                not in {
                    "none",
                    "class_weights_only",
                }
                and not state.resampling_authorized,
                "P8-READINESS-009",
                (
                    "Training resampling was not authorized."
                ),
            ),
            (
                self.options
                .require_authorized_synthetic_generation
                and bool(
                    state.synthetic_row_ids
                )
                and not state
                .synthetic_generation_authorized,
                "P8-READINESS-010",
                (
                    "Synthetic generation was not authorized."
                ),
            ),
            (
                self.options.require_unchanged_validation
                and not state
                .validation_remained_protected,
                "P8-READINESS-011",
                (
                    "The validation partition did not "
                    "remain protected."
                ),
            ),
            (
                self.options.require_unchanged_test
                and not state
                .test_remained_protected,
                "P8-READINESS-012",
                (
                    "The test partition did not remain "
                    "protected."
                ),
            ),
            (
                self.options.require_complete_lineage
                and not state.lineage_complete,
                "P8-READINESS-013",
                "Training row lineage is incomplete.",
            ),
            (
                self.options
                .require_selected_feature_lineage
                and state
                .selected_feature_lineage
                is None,
                "P8-READINESS-014",
                (
                    "Selected feature lineage is missing."
                ),
            ),
            (
                self.options.require_valid_class_weights
                and not state.class_weights_valid,
                "P8-READINESS-015",
                "Class weights are invalid.",
            ),
            (
                self.options.require_valid_sample_weights
                and not state.sample_weights_valid,
                "P8-READINESS-016",
                "Sample weights are invalid.",
            ),
            (
                self.options
                .require_fitted_state_fingerprint
                and state
                .fitted_state_fingerprint
                is None,
                "P8-READINESS-017",
                (
                    "The fitted balancing-state "
                    "fingerprint is missing."
                ),
            ),
            (
                self.options.require_output_fingerprint
                and state.output_fingerprint is None,
                "P8-READINESS-018",
                (
                    "The Phase 8 output fingerprint "
                    "is missing."
                ),
            ),
        ]

        for condition, code, message in state_checks:
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
    ) -> Phase8Finding:
        """Build one readiness-blocking finding."""

        return Phase8Finding(
            code=code,
            category=(
                Phase8FindingCategory.READINESS
            ),
            severity=(
                Phase8FindingSeverity.ERROR
            ),
            component=self.component_name,
            message=message,
            recommended_action=(
                Phase8FindingAction.REVIEW
            ),
            applied_action=(
                Phase8FindingAction.REPORT
            ),
            blocks_readiness=True,
        )

    def _finding_blocks(
        self,
        finding: Phase8Finding,
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
            == Phase8FindingSeverity.CRITICAL
        ):
            return True

        return bool(
            self.options.block_on_error_findings
            and finding.severity
            == Phase8FindingSeverity.ERROR
        )

    @staticmethod
    def _deduplicate_findings(
        findings: list[Phase8Finding],
    ) -> list[object]:
        """Deduplicate findings by finding ID."""

        seen: set[str] = set()
        unique: list[
            Phase8Finding
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