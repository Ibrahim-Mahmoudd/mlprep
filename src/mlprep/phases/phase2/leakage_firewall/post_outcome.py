"""Post-outcome feature leakage detection for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from mlprep.phases.phase2.contract.split_contract import SplitContract
from mlprep.phases.phase2.output_governance.metadata import (
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
)
from mlprep.phases.phase2.output_governance.result import SplitFinding


@dataclass(slots=True)
class PostOutcomeLeakageOptions:
    """Options controlling post-outcome leakage checks."""

    enabled: bool = True

    declared_post_outcome_columns: list[str] = field(
        default_factory=list
    )

    declared_inference_unavailable_columns: list[str] = field(
        default_factory=list
    )

    suspicious_name_terms: list[str] = field(
        default_factory=lambda: [
            "final",
            "outcome",
            "resolved",
            "resolution",
            "completed",
            "completion",
            "approved",
            "rejected",
            "closed",
            "churned",
            "defaulted",
            "recovered",
            "recovery",
            "fraud_confirmed",
            "claim_paid",
            "refund_completed",
            "collection_status",
        ]
    )

    outcome_timestamp_name_similarity_threshold: float = 0.90

    block_declared_post_outcome_features: bool = True
    block_inference_unavailable_features: bool = True

    def __post_init__(self) -> None:
        """Validate post-outcome options."""

        if not (
            0.0
            <= self.outcome_timestamp_name_similarity_threshold
            <= 1.0
        ):
            raise ValueError(
                "outcome_timestamp_name_similarity_threshold "
                "must be between zero and one."
            )


class PostOutcomeLeakageChecker:
    """Detect features that may become available after the outcome."""

    component_name = "post_outcome"

    def __init__(
        self,
        options: PostOutcomeLeakageOptions | None = None,
    ) -> None:
        """Initialize the post-outcome leakage checker."""

        self.options = (
            options or PostOutcomeLeakageOptions()
        )

    def inspect(
        self,
        contract: SplitContract,
    ) -> tuple[
        list[SplitFinding],
        list[str],
    ]:
        """Inspect feature availability relative to prediction time."""

        if not self.options.enabled:
            return [], []

        findings: list[SplitFinding] = []
        suspicious_features: set[str] = set()

        feature_columns = contract.feature_columns

        for column in feature_columns:
            if (
                column
                in self.options
                .declared_post_outcome_columns
            ):
                suspicious_features.add(column)

                findings.append(
                    SplitFinding(
                        code="P2-POST-OUTCOME-001",
                        category=(
                            SplitFindingCategory.POST_OUTCOME
                        ),
                        severity=SplitFindingSeverity.CRITICAL,
                        component=self.component_name,
                        message=(
                            f"Feature {column!r} is explicitly "
                            "declared as post-outcome information."
                        ),
                        columns=[column],
                        recommended_action=(
                            SplitFindingAction.DROP_FEATURE
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=(
                            self.options
                            .block_declared_post_outcome_features
                        ),
                    )
                )

            if (
                column
                in self.options
                .declared_inference_unavailable_columns
            ):
                suspicious_features.add(column)

                findings.append(
                    SplitFinding(
                        code="P2-POST-OUTCOME-002",
                        category=(
                            SplitFindingCategory.POST_OUTCOME
                        ),
                        severity=SplitFindingSeverity.ERROR,
                        component=self.component_name,
                        message=(
                            f"Feature {column!r} is unavailable "
                            "at inference time."
                        ),
                        columns=[column],
                        recommended_action=(
                            SplitFindingAction.DROP_FEATURE
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=(
                            self.options
                            .block_inference_unavailable_features
                        ),
                    )
                )

            matched_terms = [
                term
                for term
                in self.options.suspicious_name_terms
                if term.casefold()
                in column.casefold()
            ]

            if matched_terms:
                suspicious_features.add(column)

                findings.append(
                    SplitFinding(
                        code="P2-POST-OUTCOME-003",
                        category=(
                            SplitFindingCategory.POST_OUTCOME
                        ),
                        severity=SplitFindingSeverity.WARNING,
                        component=self.component_name,
                        message=(
                            f"Feature {column!r} contains a name "
                            "associated with post-outcome information."
                        ),
                        columns=[column],
                        recommended_action=(
                            SplitFindingAction.WARN
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=False,
                        details={
                            "matched_terms": matched_terms,
                        },
                    )
                )

        outcome_timestamp_column = None

        if contract.phase1_state.problem_contract is not None:
            outcome_timestamp_column = (
                contract.phase1_state
                .problem_contract
                .outcome_timestamp_column
            )

        if (
            outcome_timestamp_column is not None
            and outcome_timestamp_column
            in feature_columns
        ):
            suspicious_features.add(
                outcome_timestamp_column
            )

            findings.append(
                SplitFinding(
                    code="P2-POST-OUTCOME-004",
                    category=(
                        SplitFindingCategory.POST_OUTCOME
                    ),
                    severity=SplitFindingSeverity.CRITICAL,
                    component=self.component_name,
                    message=(
                        "The outcome timestamp is included in "
                        "the feature set."
                    ),
                    columns=[outcome_timestamp_column],
                    recommended_action=(
                        SplitFindingAction.DROP_FEATURE
                    ),
                    applied_action=SplitFindingAction.REPORT,
                    blocks_readiness=True,
                )
            )

        if outcome_timestamp_column is not None:
            for feature in feature_columns:
                if feature == outcome_timestamp_column:
                    continue

                similarity = SequenceMatcher(
                    None,
                    feature.casefold(),
                    outcome_timestamp_column.casefold(),
                ).ratio()

                if (
                    similarity
                    < self.options
                    .outcome_timestamp_name_similarity_threshold
                ):
                    continue

                suspicious_features.add(feature)

                findings.append(
                    SplitFinding(
                        code="P2-POST-OUTCOME-005",
                        category=(
                            SplitFindingCategory.POST_OUTCOME
                        ),
                        severity=SplitFindingSeverity.WARNING,
                        component=self.component_name,
                        message=(
                            f"Feature {feature!r} has a name "
                            "similar to the outcome timestamp."
                        ),
                        columns=[
                            feature,
                            outcome_timestamp_column,
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
                        },
                    )
                )

        return findings, sorted(
            suspicious_features
        )