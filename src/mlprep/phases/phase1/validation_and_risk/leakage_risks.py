"""Preliminary target-leakage risk detection for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import Phase1State
from mlprep.types import (
    FindingAction,
    FindingCategory,
    Severity,
)


@dataclass(slots=True)
class LeakageRiskOptions:
    """Options controlling preliminary leakage-risk detection."""

    enabled: bool = True

    target_column: str = ""

    declared_post_outcome_columns: list[str] = field(
        default_factory=list
    )

    inference_unavailable_columns: list[str] = field(
        default_factory=list
    )

    suspicious_name_terms: list[str] = field(
        default_factory=lambda: [
            "target",
            "label",
            "outcome",
            "result",
            "final",
            "approved",
            "rejected",
            "completed",
            "resolved",
            "churned",
            "defaulted",
            "fraud_confirmed",
        ]
    )

    detect_target_name_similarity: bool = True
    target_name_similarity_threshold: float = 0.85

    detect_target_value_duplicates: bool = True
    detect_identifier_like_features: bool = True

    block_declared_post_outcome_features: bool = True

    def __post_init__(self) -> None:
        """Validate leakage-risk options."""

        if not self.target_column.strip():
            raise ValueError(
                "target_column cannot be empty."
            )

        if not 0.0 <= self.target_name_similarity_threshold <= 1.0:
            raise ValueError(
                "target_name_similarity_threshold must be "
                "between zero and one."
            )


class LeakageRiskDetector:
    """Detect obvious pre-split leakage risks."""

    component_name = "leakage_risks"

    def __init__(
        self,
        options: LeakageRiskOptions,
    ) -> None:
        """Initialize the leakage-risk detector."""

        self.options = options

    def inspect(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> list[object]:
        """Inspect feature columns for obvious leakage risks."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Leakage-risk detection requires a pandas DataFrame."
            )

        if not self.options.enabled:
            return []

        target = self.options.target_column

        if target not in data.columns:
            return []

        feature_columns = self._feature_columns(
            data=data,
            state=state,
        )

        risky_columns: set[str] = set()

        for column in feature_columns:
            if column in self.options.declared_post_outcome_columns:
                risky_columns.add(column)

                result.add_finding(
                    Finding(
                        code="P1-LEAKAGE-001",
                        category=FindingCategory.LEAKAGE_RISK,
                        severity=Severity.CRITICAL,
                        component=self.component_name,
                        message=(
                            f"Column {column!r} is declared as "
                            "post-outcome information."
                        ),
                        columns=[column],
                        recommended_action=FindingAction.DROP_COLUMN,
                        applied_action=FindingAction.REPORT,
                        blocks_readiness=(
                            self.options
                            .block_declared_post_outcome_features
                        ),
                    )
                )

            if column in self.options.inference_unavailable_columns:
                risky_columns.add(column)

                result.add_finding(
                    Finding(
                        code="P1-LEAKAGE-002",
                        category=FindingCategory.LEAKAGE_RISK,
                        severity=Severity.ERROR,
                        component=self.component_name,
                        message=(
                            f"Feature {column!r} is unavailable "
                            "at inference time."
                        ),
                        columns=[column],
                        recommended_action=FindingAction.DROP_COLUMN,
                        applied_action=FindingAction.REPORT,
                        blocks_readiness=True,
                    )
                )

            if self._has_suspicious_name(column):
                risky_columns.add(column)

                result.add_finding(
                    Finding(
                        code="P1-LEAKAGE-003",
                        category=FindingCategory.LEAKAGE_RISK,
                        severity=Severity.WARNING,
                        component=self.component_name,
                        message=(
                            f"Feature {column!r} has a name "
                            "associated with possible target leakage."
                        ),
                        columns=[column],
                        recommended_action=FindingAction.WARN,
                        applied_action=FindingAction.REPORT,
                        blocks_readiness=False,
                    )
                )

            if (
                self.options.detect_target_name_similarity
                and self._name_similarity(
                    column,
                    target,
                )
                >= self.options
                .target_name_similarity_threshold
            ):
                risky_columns.add(column)

                result.add_finding(
                    Finding(
                        code="P1-LEAKAGE-004",
                        category=FindingCategory.LEAKAGE_RISK,
                        severity=Severity.WARNING,
                        component=self.component_name,
                        message=(
                            f"Feature {column!r} is highly similar "
                            "to the target column name."
                        ),
                        columns=[column],
                        recommended_action=FindingAction.WARN,
                        applied_action=FindingAction.REPORT,
                        blocks_readiness=False,
                    )
                )

            if (
                self.options.detect_target_value_duplicates
                and data[column].equals(data[target])
            ):
                risky_columns.add(column)

                result.add_finding(
                    Finding(
                        code="P1-LEAKAGE-005",
                        category=FindingCategory.LEAKAGE_RISK,
                        severity=Severity.CRITICAL,
                        component=self.component_name,
                        message=(
                            f"Feature {column!r} duplicates the "
                            "target values exactly."
                        ),
                        columns=[column, target],
                        recommended_action=FindingAction.DROP_COLUMN,
                        applied_action=FindingAction.REPORT,
                        blocks_readiness=True,
                    )
                )

            column_state = state.column_states.get(column)

            if (
                self.options.detect_identifier_like_features
                and column_state is not None
                and column_state.is_identifier_like
                and column_state.use_as_model_feature
            ):
                risky_columns.add(column)

                result.add_finding(
                    Finding(
                        code="P1-LEAKAGE-006",
                        category=FindingCategory.LEAKAGE_RISK,
                        severity=Severity.WARNING,
                        component=self.component_name,
                        message=(
                            f"Identifier-like column {column!r} "
                            "is currently model-eligible."
                        ),
                        columns=[column],
                        recommended_action=FindingAction.WARN,
                        applied_action=FindingAction.REPORT,
                        blocks_readiness=False,
                    )
                )

        return sorted(risky_columns)

    def _feature_columns(
        self,
        *,
        data: pd.DataFrame,
        state: Phase1State,
    ) -> list[object]:
        """Return model-eligible feature columns."""

        state_features = state.get_model_feature_columns()

        if state_features:
            return [
                column
                for column in state_features
                if column in data.columns
                and column != self.options.target_column
            ]

        return [
            str(column)
            for column in data.columns
            if str(column) != self.options.target_column
        ]

    def _has_suspicious_name(
        self,
        column: str,
    ) -> bool:
        """Return whether a feature name contains suspicious terms."""

        normalized_name = column.casefold()

        return any(
            term.casefold() in normalized_name
            for term in self.options.suspicious_name_terms
        )

    @staticmethod
    def _name_similarity(
        left: str,
        right: str,
    ) -> float:
        """Calculate normalized column-name similarity."""

        return SequenceMatcher(
            None,
            left.casefold(),
            right.casefold(),
        ).ratio()
