"""Cardinality, uniqueness, and constancy inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    CardinalityState,
    Phase1State,
)
from mlprep.types import (
    ColumnRole,
    FindingAction,
    FindingCategory,
    Severity,
)


CardinalityBand = Literal[
    "empty",
    "constant",
    "binary",
    "low",
    "medium",
    "high",
    "identifier_like",
]


@dataclass(slots=True)
class CardinalityInspectionOptions:
    """Options controlling cardinality inspection."""

    enabled: bool = True

    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(
        default_factory=list
    )

    low_cardinality_maximum: int = 20
    medium_cardinality_maximum: int = 100

    identifier_uniqueness_threshold: float = 0.95
    near_constant_threshold: float = 0.995

    detect_constant_columns: bool = True
    detect_near_constant_columns: bool = True
    detect_identifier_like_columns: bool = True

    ignore_missing_for_constancy: bool = True

    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate cardinality inspection options."""

        if self.low_cardinality_maximum < 2:
            raise ValueError(
                "low_cardinality_maximum must be at least two."
            )

        if (
            self.medium_cardinality_maximum
            <= self.low_cardinality_maximum
        ):
            raise ValueError(
                "medium_cardinality_maximum must be greater "
                "than low_cardinality_maximum."
            )

        if not 0.0 <= self.identifier_uniqueness_threshold <= 1.0:
            raise ValueError(
                "identifier_uniqueness_threshold must be "
                "between zero and one."
            )

        if not 0.0 <= self.near_constant_threshold <= 1.0:
            raise ValueError(
                "near_constant_threshold must be between "
                "zero and one."
            )

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )


class CardinalityInspector:
    """Inspect column cardinality, uniqueness, and constancy."""

    component_name = "cardinality"

    def __init__(
        self,
        options: CardinalityInspectionOptions | None = None,
    ) -> None:
        """Initialize the cardinality inspector."""

        self.options = (
            options or CardinalityInspectionOptions()
        )

    def inspect(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> dict[str, CardinalityState]:
        """Inspect selected columns and register cardinality state."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Cardinality inspection requires a "
                "pandas DataFrame."
            )

        if not self.options.enabled:
            return {}

        states: dict[str, CardinalityState] = {}

        for column in self._select_columns(data):
            cardinality_state = self._inspect_column(
                column=column,
                series=data[column],
                row_count=len(data),
            )

            states[column] = cardinality_state

            state.register_cardinality_state(
                cardinality_state
            )

            self._update_column_state(
                cardinality_state=cardinality_state,
                state=state,
            )

            self._add_findings(
                cardinality_state=cardinality_state,
                result=result,
                state=state,
            )

        return states

    def _inspect_column(
        self,
        *,
        column: str,
        series: pd.Series,
        row_count: int,
    ) -> CardinalityState:
        """Build cardinality state for one column."""

        non_missing = series.dropna()

        non_missing_count = len(non_missing)

        unique_count = int(
            non_missing.nunique(dropna=True)
        )

        uniqueness_ratio = (
            unique_count / non_missing_count
            if non_missing_count > 0
            else 0.0
        )

        if self.options.ignore_missing_for_constancy:
            constancy_series = non_missing
        else:
            constancy_series = series

        value_counts = constancy_series.value_counts(
            dropna=(
                self.options.ignore_missing_for_constancy
            )
        )

        if value_counts.empty:
            most_frequent_count = 0
            most_frequent_ratio = 0.0
        else:
            most_frequent_count = int(
                value_counts.iloc[0]
            )

            most_frequent_ratio = (
                most_frequent_count
                / len(constancy_series)
                if len(constancy_series) > 0
                else 0.0
            )

        is_constant = (
            unique_count <= 1
            and non_missing_count > 0
        )

        is_near_constant = (
            not is_constant
            and most_frequent_ratio
            >= self.options.near_constant_threshold
        )

        is_identifier_like = (
            non_missing_count > 0
            and uniqueness_ratio
            >= self.options.identifier_uniqueness_threshold
        )

        band = self._resolve_band(
            non_missing_count=non_missing_count,
            unique_count=unique_count,
            is_constant=is_constant,
            is_identifier_like=is_identifier_like,
        )

        return CardinalityState(
            column=column,
            non_missing_count=non_missing_count,
            unique_count=unique_count,
            uniqueness_ratio=uniqueness_ratio,
            most_frequent_value_count=(
                most_frequent_count
            ),
            most_frequent_value_ratio=(
                most_frequent_ratio
            ),
            cardinality_band=band,
            is_constant=is_constant,
            is_near_constant=is_near_constant,
            is_identifier_like=is_identifier_like,
            count_is_approximate=False,
        )

    def _resolve_band(
        self,
        *,
        non_missing_count: int,
        unique_count: int,
        is_constant: bool,
        is_identifier_like: bool,
    ) -> CardinalityBand:
        """Resolve the cardinality band for one column."""

        if non_missing_count == 0:
            return "empty"

        if is_constant:
            return "constant"

        if unique_count == 2:
            return "binary"

        if is_identifier_like:
            return "identifier_like"

        if unique_count <= self.options.low_cardinality_maximum:
            return "low"

        if unique_count <= self.options.medium_cardinality_maximum:
            return "medium"

        return "high"

    @staticmethod
    def _update_column_state(
        *,
        cardinality_state: CardinalityState,
        state: Phase1State,
    ) -> None:
        """Update the central column state."""

        column_state = state.column_states.get(
            cardinality_state.column
        )

        if column_state is None:
            return

        column_state.unique_count = (
            cardinality_state.unique_count
        )

        column_state.uniqueness_ratio = (
            cardinality_state.uniqueness_ratio
        )

        column_state.is_constant = (
            cardinality_state.is_constant
        )

        column_state.is_near_constant = (
            cardinality_state.is_near_constant
        )

        if (
            ColumnRole.IDENTIFIER
            not in column_state.roles
        ):
            column_state.is_identifier_like = (
                cardinality_state.is_identifier_like
            )

        state.register_column_state(
            column_state
        )

    def _add_findings(
        self,
        *,
        cardinality_state: CardinalityState,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Add relevant cardinality findings."""

        column = cardinality_state.column

        if (
            self.options.detect_constant_columns
            and cardinality_state.is_constant
        ):
            result.add_finding(
                Finding(
                    code="P1-CARDINALITY-001",
                    category=FindingCategory.CONSTANCY,
                    severity=Severity.WARNING,
                    component=self.component_name,
                    message=(
                        f"Column {column!r} is constant."
                    ),
                    columns=[column],
                    recommended_action=FindingAction.REPORT,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=False,
                    details={
                        "unique_count": (
                            cardinality_state.unique_count
                        ),
                    },
                )
            )

        if (
            self.options.detect_near_constant_columns
            and cardinality_state.is_near_constant
        ):
            result.add_finding(
                Finding(
                    code="P1-CARDINALITY-002",
                    category=FindingCategory.CONSTANCY,
                    severity=Severity.WARNING,
                    component=self.component_name,
                    message=(
                        f"Column {column!r} is near constant."
                    ),
                    columns=[column],
                    recommended_action=FindingAction.REPORT,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=False,
                    details={
                        "most_frequent_value_ratio": (
                            cardinality_state
                            .most_frequent_value_ratio
                        ),
                    },
                )
            )

        column_state = state.column_states.get(column)

        is_declared_identifier = (
            column_state is not None
            and ColumnRole.IDENTIFIER
            in column_state.roles
        )

        if (
            self.options.detect_identifier_like_columns
            and cardinality_state.is_identifier_like
            and not is_declared_identifier
        ):
            result.add_finding(
                Finding(
                    code="P1-CARDINALITY-003",
                    category=FindingCategory.CARDINALITY,
                    severity=Severity.WARNING,
                    component=self.component_name,
                    message=(
                        f"Column {column!r} appears to be "
                        "identifier-like."
                    ),
                    columns=[column],
                    recommended_action=FindingAction.WARN,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=False,
                    details={
                        "unique_count": (
                            cardinality_state.unique_count
                        ),
                        "uniqueness_ratio": (
                            cardinality_state.uniqueness_ratio
                        ),
                    },
                )
            )

    def _select_columns(
        self,
        data: pd.DataFrame,
    ) -> list[object]:
        """Select columns eligible for cardinality inspection."""

        excluded = set(
            self.options.exclude_columns
        )

        if self.options.include_columns is not None:
            return [
                column
                for column in self.options.include_columns
                if column in data.columns
                and column not in excluded
            ]

        return [
            str(column)
            for column in data.columns
            if str(column) not in excluded
        ]