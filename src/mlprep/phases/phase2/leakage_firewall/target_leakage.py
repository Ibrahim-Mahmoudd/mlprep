"""Target-leakage checks for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pandas as pd

from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
)
from mlprep.phases.phase2.output_governance.metadata import (
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
)
from mlprep.phases.phase2.output_governance.result import (
    SplitFinding,
)


@dataclass(slots=True)
class TargetLeakageOptions:
    """Options controlling target-leakage checks."""

    enabled: bool = True

    detect_target_in_features: bool = True
    detect_exact_target_copies: bool = True
    detect_name_similarity: bool = True

    suspicious_name_terms: list[str] = field(
        default_factory=lambda: [
            "target",
            "label",
            "outcome",
            "final",
            "result",
            "churned",
            "defaulted",
            "fraud_confirmed",
        ]
    )

    name_similarity_threshold: float = 0.90

    block_exact_target_copy: bool = True
    block_target_in_features: bool = True

    def __post_init__(self) -> None:
        """Validate target-leakage options."""

        if not 0.0 <= self.name_similarity_threshold <= 1.0:
            raise ValueError(
                "name_similarity_threshold must be between "
                "zero and one."
            )


class TargetLeakageChecker:
    """Detect obvious target leakage in feature columns."""

    component_name = "target_leakage"

    def __init__(
        self,
        options: TargetLeakageOptions | None = None,
    ) -> None:
        """Initialize the target-leakage checker."""

        self.options = options or TargetLeakageOptions()

    def inspect(
        self,
        contract: SplitContract,
    ) -> list[SplitFinding]:
        """Inspect the complete pre-split feature set."""

        if not self.options.enabled:
            return []

        findings: list[SplitFinding] = []

        target_column = contract.target_column
        feature_columns = contract.feature_columns

        if (
            self.options.detect_target_in_features
            and target_column in feature_columns
        ):
            findings.append(
                SplitFinding(
                    code="P2-TARGET-LEAKAGE-001",
                    category=(
                        SplitFindingCategory.TARGET_LEAKAGE
                    ),
                    severity=SplitFindingSeverity.CRITICAL,
                    component=self.component_name,
                    message=(
                        "The target column is included in the "
                        "resolved feature set."
                    ),
                    columns=[target_column],
                    recommended_action=(
                        SplitFindingAction.DROP_FEATURE
                    ),
                    applied_action=SplitFindingAction.REPORT,
                    blocks_readiness=(
                        self.options.block_target_in_features
                    ),
                )
            )

        target = contract.target_series()

        for feature in feature_columns:
            if feature == target_column:
                continue

            if (
                self.options.detect_exact_target_copies
                and self._series_equal(
                    contract.data[feature],
                    target,
                )
            ):
                findings.append(
                    SplitFinding(
                        code="P2-TARGET-LEAKAGE-002",
                        category=(
                            SplitFindingCategory.TARGET_LEAKAGE
                        ),
                        severity=SplitFindingSeverity.CRITICAL,
                        component=self.component_name,
                        message=(
                            f"Feature {feature!r} duplicates the "
                            "target values exactly."
                        ),
                        columns=[
                            feature,
                            target_column,
                        ],
                        recommended_action=(
                            SplitFindingAction.DROP_FEATURE
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=(
                            self.options.block_exact_target_copy
                        ),
                    )
                )

            if self.options.detect_name_similarity:
                similarity = SequenceMatcher(
                    None,
                    feature.casefold(),
                    target_column.casefold(),
                ).ratio()

                if (
                    similarity
                    >= self.options.name_similarity_threshold
                ):
                    findings.append(
                        SplitFinding(
                            code="P2-TARGET-LEAKAGE-003",
                            category=(
                                SplitFindingCategory.TARGET_LEAKAGE
                            ),
                            severity=SplitFindingSeverity.WARNING,
                            component=self.component_name,
                            message=(
                                f"Feature {feature!r} has a name "
                                "that is highly similar to the "
                                "target column."
                            ),
                            columns=[
                                feature,
                                target_column,
                            ],
                            recommended_action=(
                                SplitFindingAction.WARN
                            ),
                            applied_action=(
                                SplitFindingAction.REPORT
                            ),
                            blocks_readiness=False,
                            details={
                                "similarity": similarity,
                                "threshold": (
                                    self.options
                                    .name_similarity_threshold
                                ),
                            },
                        )
                    )

            suspicious_terms = [
                term
                for term
                in self.options.suspicious_name_terms
                if term.casefold() in feature.casefold()
            ]

            if suspicious_terms:
                findings.append(
                    SplitFinding(
                        code="P2-TARGET-LEAKAGE-004",
                        category=(
                            SplitFindingCategory.TARGET_LEAKAGE
                        ),
                        severity=SplitFindingSeverity.WARNING,
                        component=self.component_name,
                        message=(
                            f"Feature {feature!r} contains one or "
                            "more suspicious leakage-related terms."
                        ),
                        columns=[feature],
                        recommended_action=(
                            SplitFindingAction.WARN
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=False,
                        details={
                            "matched_terms": suspicious_terms,
                        },
                    )
                )

        return findings

    @staticmethod
    def _series_equal(
        left: pd.Series,
        right: pd.Series,
    ) -> bool:
        """Return whether two aligned Series are identical."""

        if len(left) != len(right):
            return False

        if not left.index.equals(right.index):
            right = right.reindex(left.index)

        return bool(left.equals(right))
