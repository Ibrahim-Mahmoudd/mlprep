"""Unified leakage firewall for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field

from mlprep.phases.phase2.contract.split_contract import SplitContract
from mlprep.phases.phase2.leakage_firewall.duplicate_overlap import (
    DuplicateOverlapChecker,
    DuplicateOverlapOptions,
)
from mlprep.phases.phase2.leakage_firewall.group_overlap import (
    GroupOverlapChecker,
    GroupOverlapOptions,
)
from mlprep.phases.phase2.leakage_firewall.post_outcome import (
    PostOutcomeLeakageChecker,
    PostOutcomeLeakageOptions,
)
from mlprep.phases.phase2.leakage_firewall.preprocessing_leakage import (
    PreprocessingLeakageChecker,
    PreprocessingLeakageOptions,
)
from mlprep.phases.phase2.leakage_firewall.row_overlap import (
    RowOverlapChecker,
    RowOverlapOptions,
)
from mlprep.phases.phase2.leakage_firewall.target_leakage import (
    TargetLeakageChecker,
    TargetLeakageOptions,
)
from mlprep.phases.phase2.leakage_firewall.temporal_leakage import (
    TemporalLeakageChecker,
    TemporalLeakageOptions,
)
from mlprep.phases.phase2.output_governance.result import SplitFinding
from mlprep.phases.phase2.output_governance.state import Phase2State
from mlprep.phases.phase2.strategies.base import StrategyOutput


@dataclass(slots=True)
class LeakageFirewallOptions:
    """Complete configuration for the Phase 2 leakage firewall."""

    enabled: bool = True

    target_leakage: TargetLeakageOptions = field(
        default_factory=TargetLeakageOptions
    )

    row_overlap: RowOverlapOptions = field(
        default_factory=RowOverlapOptions
    )

    group_overlap: GroupOverlapOptions = field(
        default_factory=GroupOverlapOptions
    )

    duplicate_overlap: DuplicateOverlapOptions = field(
        default_factory=DuplicateOverlapOptions
    )

    temporal_leakage: TemporalLeakageOptions = field(
        default_factory=TemporalLeakageOptions
    )

    post_outcome: PostOutcomeLeakageOptions = field(
        default_factory=PostOutcomeLeakageOptions
    )

    preprocessing_leakage: PreprocessingLeakageOptions = field(
        default_factory=PreprocessingLeakageOptions
    )


class LeakageFirewall:
    """Run all Phase 2 leakage checks."""

    def __init__(
        self,
        options: LeakageFirewallOptions | None = None,
    ) -> None:
        """Initialize the leakage firewall."""

        self.options = (
            options or LeakageFirewallOptions()
        )

    def inspect_before_split(
        self,
        contract: SplitContract,
        state: Phase2State,
    ) -> list[object]:
        """Run leakage checks that must precede splitting."""

        if not self.options.enabled:
            return []

        findings: list[SplitFinding] = []

        target_checker = TargetLeakageChecker(
            options=self.options.target_leakage
        )

        target_findings = target_checker.inspect(
            contract
        )

        findings.extend(target_findings)

        state.leakage_firewall.target_removed_from_features = (
            contract.target_column
            not in contract.feature_columns
        )

        post_outcome_checker = (
            PostOutcomeLeakageChecker(
                options=self.options.post_outcome
            )
        )

        (
            post_outcome_findings,
            suspicious_features,
        ) = post_outcome_checker.inspect(
            contract
        )

        findings.extend(
            post_outcome_findings
        )

        state.leakage_firewall.post_outcome_checked = True

        state.leakage_firewall.suspicious_features = list(
            dict.fromkeys(
                [
                    *state.leakage_firewall
                    .suspicious_features,
                    *suspicious_features,
                ]
            )
        )

        preprocessing_checker = (
            PreprocessingLeakageChecker(
                options=(
                    self.options
                    .preprocessing_leakage
                )
            )
        )

        preprocessing_findings = (
            preprocessing_checker.inspect(
                contract
            )
        )

        findings.extend(
            preprocessing_findings
        )

        state.leakage_firewall.preprocessing_leakage_checked = (
            True
        )

        return findings

    def inspect_after_split(
        self,
        *,
        contract: SplitContract,
        strategy_output: StrategyOutput,
        state: Phase2State,
    ) -> list[object]:
        """Run cross-partition leakage checks."""

        if not self.options.enabled:
            state.leakage_firewall.passed = True
            return []

        findings: list[SplitFinding] = []

        row_checker = RowOverlapChecker(
            options=self.options.row_overlap
        )

        (
            row_findings,
            row_overlap_state,
        ) = row_checker.inspect(
            train=strategy_output.train,
            validation=strategy_output.validation,
            test=strategy_output.test,
        )

        findings.extend(row_findings)

        state.leakage_firewall.row_overlap_checked = True

        state.leakage_firewall.overlapping_row_ids = (
            row_overlap_state
        )

        group_checker = GroupOverlapChecker(
            options=self.options.group_overlap
        )

        (
            group_findings,
            group_overlap_state,
        ) = group_checker.inspect(
            contract=contract,
            indices=strategy_output.indices,
        )

        findings.extend(group_findings)

        state.leakage_firewall.group_overlap_checked = True

        state.leakage_firewall.overlapping_groups = (
            group_overlap_state
        )

        duplicate_checker = DuplicateOverlapChecker(
            options=self.options.duplicate_overlap
        )

        duplicate_findings = duplicate_checker.inspect(
            train=strategy_output.train,
            validation=strategy_output.validation,
            test=strategy_output.test,
        )

        findings.extend(duplicate_findings)

        state.leakage_firewall.duplicate_overlap_checked = True

        temporal_checker = TemporalLeakageChecker(
            options=self.options.temporal_leakage
        )

        (
            temporal_findings,
            temporal_violations,
        ) = temporal_checker.inspect(
            contract=contract,
            indices=strategy_output.indices,
        )

        findings.extend(temporal_findings)

        state.leakage_firewall.temporal_order_checked = True

        state.leakage_firewall.temporal_violations = (
            temporal_violations
        )

        blocking_findings = [
            finding
            for finding in findings
            if finding.blocks_readiness
        ]

        state.leakage_firewall.passed = (
            not blocking_findings
        )

        return findings
