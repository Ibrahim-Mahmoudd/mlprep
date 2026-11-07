"""Duplicate-column detection for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field

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
class DuplicateColumnOptions:
    """Options controlling duplicate-column detection."""

    enabled: bool = True
    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(default_factory=list)

    detect_duplicate_names: bool = True
    detect_identical_values: bool = True
    compare_missing_values_as_equal: bool = True

    maximum_columns_for_value_comparison: int = 1_000
    block_on_duplicate_names: bool = True

    def __post_init__(self) -> None:
        """Validate duplicate-column options."""

        if self.maximum_columns_for_value_comparison < 2:
            raise ValueError(
                "maximum_columns_for_value_comparison must be at least 2."
            )


class DuplicateColumnDetector:
    """Detect duplicate names and value-identical columns."""

    component_name = "duplicate_columns"

    def __init__(
        self,
        options: DuplicateColumnOptions | None = None,
    ) -> None:
        """Initialize the detector."""

        self.options = options or DuplicateColumnOptions()

    def inspect(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> list[list[str]]:
        """Detect duplicate-column groups."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Duplicate-column detection requires a pandas DataFrame."
            )

        if not self.options.enabled:
            return []

        duplicate_groups: list[list[str]] = []

        if self.options.detect_duplicate_names:
            duplicate_groups.extend(
                self._detect_duplicate_names(
                    data=data,
                    result=result,
                )
            )

        if self.options.detect_identical_values:
            duplicate_groups.extend(
                self._detect_identical_values(
                    data=data,
                    result=result,
                )
            )

        state.duplicate_column_groups = duplicate_groups

        return duplicate_groups

    def _detect_duplicate_names(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> list[list[str]]:
        """Detect repeated DataFrame column labels."""

        duplicated_mask = data.columns.duplicated(
            keep=False
        )

        if not bool(duplicated_mask.any()):
            return []

        duplicated_names = [
            str(column)
            for column, duplicated
            in zip(
                data.columns,
                duplicated_mask,
                strict=True,
            )
            if duplicated
        ]

        unique_names = list(
            dict.fromkeys(duplicated_names)
        )

        groups = [
            [name for name in duplicated_names if name == unique_name]
            for unique_name in unique_names
        ]

        result.add_finding(
            Finding(
                code="P1-DUPCOL-001",
                category=FindingCategory.DUPLICATE_COLUMN,
                severity=Severity.CRITICAL,
                component=self.component_name,
                message="The dataset contains duplicate column names.",
                columns=unique_names,
                recommended_action=FindingAction.RENAME,
                applied_action=FindingAction.REPORT,
                blocks_readiness=(
                    self.options.block_on_duplicate_names
                ),
                details={
                    "duplicate_groups": groups,
                },
            )
        )

        return groups

    def _detect_identical_values(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> list[list[str]]:
        """Detect differently named columns with identical values."""

        columns = self._select_columns(data)

        if (
            len(columns)
            > self.options.maximum_columns_for_value_comparison
        ):
            result.add_finding(
                Finding(
                    code="P1-DUPCOL-002",
                    category=FindingCategory.DUPLICATE_COLUMN,
                    severity=Severity.WARNING,
                    component=self.component_name,
                    message=(
                        "Value-based duplicate-column detection was skipped "
                        "because the selected column count exceeds the limit."
                    ),
                    recommended_action=FindingAction.WARN,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=False,
                    details={
                        "selected_column_count": len(columns),
                        "maximum_columns": (
                            self.options
                            .maximum_columns_for_value_comparison
                        ),
                    },
                )
            )

            return []

        groups: list[list[str]] = []
        assigned_columns: set[str] = set()

        for position, column in enumerate(columns):
            if column in assigned_columns:
                continue

            group = [column]

            for candidate in columns[position + 1:]:
                if candidate in assigned_columns:
                    continue

                if self._series_equal(
                    data[column],
                    data[candidate],
                ):
                    group.append(candidate)

            if len(group) <= 1:
                continue

            groups.append(group)
            assigned_columns.update(group)

            result.add_finding(
                Finding(
                    code="P1-DUPCOL-003",
                    category=FindingCategory.DUPLICATE_COLUMN,
                    severity=Severity.WARNING,
                    component=self.component_name,
                    message=(
                        "Multiple columns contain identical values."
                    ),
                    columns=group,
                    recommended_action=FindingAction.REPORT,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=False,
                )
            )

        return groups

    def _series_equal(
        self,
        left: pd.Series,
        right: pd.Series,
    ) -> bool:
        """Compare two Series using the configured missing-value policy."""

        if len(left) != len(right):
            return False

        if self.options.compare_missing_values_as_equal:
            return bool(left.equals(right))

        valid_mask = left.notna() & right.notna()

        if not bool(valid_mask.any()):
            return False

        return bool(
            left.loc[valid_mask].equals(
                right.loc[valid_mask]
            )
        )

    def _select_columns(
        self,
        data: pd.DataFrame,
    ) -> list[str]:
        """Select columns eligible for value comparison."""

        excluded = set(self.options.exclude_columns)

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
