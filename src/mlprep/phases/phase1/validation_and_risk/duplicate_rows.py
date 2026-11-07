"""Duplicate-row detection and resolution for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
    RejectedRecord,
)
from mlprep.phases.phase1.output_governance.state import Phase1State
from mlprep.types import (
    FindingAction,
    FindingCategory,
    Severity,
)

DuplicateRowPolicy = Literal[
    "report",
    "raise",
    "keep_first",
    "keep_last",
    "remove_exact",
    "quarantine",
]


@dataclass(slots=True)
class DuplicateRowOptions:
    """Options controlling duplicate-row detection."""

    enabled: bool = True
    policy: DuplicateRowPolicy = "report"

    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(default_factory=list)

    observation_key: list[str] = field(default_factory=list)
    target_column: str | None = None

    detect_exact_duplicates: bool = True
    detect_observation_key_duplicates: bool = True
    detect_conflicting_targets: bool = True

    internal_row_id_column: str = "__mlprep_row_id__"
    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate duplicate-row options."""

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )


class DuplicateRowDetector:
    """Detect and optionally resolve duplicate rows."""

    component_name = "duplicate_rows"

    def __init__(
        self,
        options: DuplicateRowOptions | None = None,
    ) -> None:
        """Initialize the detector."""

        self.options = options or DuplicateRowOptions()

    def inspect(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Detect and optionally resolve duplicate rows."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Duplicate-row detection requires a pandas DataFrame."
            )

        working_data = data.copy()

        if not self.options.enabled:
            return working_data

        if self.options.detect_exact_duplicates:
            working_data = self._detect_exact_duplicates(
                data=working_data,
                result=result,
            )

        if (
            self.options.detect_observation_key_duplicates
            and self.options.observation_key
        ):
            self._detect_key_duplicates(
                data=working_data,
                result=result,
            )

        return working_data

    def _detect_exact_duplicates(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> pd.DataFrame:
        """Detect exact duplicate rows."""

        subset = self._comparison_columns(data)

        duplicate_mask = data.duplicated(
            subset=subset,
            keep=False,
        )

        duplicate_count = int(duplicate_mask.sum())

        result.dataset_summary.duplicate_row_count = (
            duplicate_count
        )

        if duplicate_count == 0:
            return data

        duplicate_indices = data.index[
            duplicate_mask
        ].tolist()

        blocks_readiness = (
            self.options.policy == "raise"
        )

        result.add_finding(
            Finding(
                code="P1-DUPROW-001",
                category=FindingCategory.DUPLICATE_ROW,
                severity=(
                    Severity.ERROR
                    if blocks_readiness
                    else Severity.WARNING
                ),
                component=self.component_name,
                message="The dataset contains exact duplicate rows.",
                columns=subset,
                affected_row_count=duplicate_count,
                affected_row_percentage=self._percentage(
                    duplicate_count,
                    len(data),
                ),
                example_row_ids=[
                    self._safe_row_id(index)
                    for index in duplicate_indices[
                        : self.options.maximum_examples
                    ]
                ],
                recommended_action=FindingAction.REPORT,
                applied_action=FindingAction.REPORT,
                blocks_readiness=blocks_readiness,
            )
        )

        if self.options.policy in {"report", "raise"}:
            return data

        keep = (
            "last"
            if self.options.policy == "keep_last"
            else "first"
        )

        removal_mask = data.duplicated(
            subset=subset,
            keep=keep,
        )

        removed_data = data.loc[removal_mask]

        for index, row in removed_data.iterrows():
            result.reject_record(
                RejectedRecord(
                    row_id=self._extract_row_id(
                        row=row,
                        index=index,
                    ),
                    reason_code="P1-DUPROW-001",
                    reason="Exact duplicate row.",
                    component=self.component_name,
                    severity=Severity.WARNING,
                    columns=subset,
                    original_index=self._safe_row_id(index),
                )
            )

        return data.loc[~removal_mask].copy()

    def _detect_key_duplicates(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> None:
        """Detect duplicated observation keys and conflicting targets."""

        missing_key_columns = [
            column
            for column in self.options.observation_key
            if column not in data.columns
        ]

        if missing_key_columns:
            result.add_finding(
                Finding(
                    code="P1-DUPROW-002",
                    category=FindingCategory.DUPLICATE_ROW,
                    severity=Severity.ERROR,
                    component=self.component_name,
                    message=(
                        "Duplicate-key detection references missing columns."
                    ),
                    columns=missing_key_columns,
                    recommended_action=FindingAction.RAISE,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=True,
                )
            )

            return

        duplicate_mask = data.duplicated(
            subset=self.options.observation_key,
            keep=False,
        )

        duplicate_count = int(duplicate_mask.sum())

        if duplicate_count == 0:
            return

        result.add_finding(
            Finding(
                code="P1-DUPROW-003",
                category=FindingCategory.DUPLICATE_ROW,
                severity=Severity.ERROR,
                component=self.component_name,
                message="The observation key is duplicated.",
                columns=self.options.observation_key,
                affected_row_count=duplicate_count,
                affected_row_percentage=self._percentage(
                    duplicate_count,
                    len(data),
                ),
                example_row_ids=[
                    self._safe_row_id(index)
                    for index in data.index[
                        duplicate_mask
                    ].tolist()[
                        : self.options.maximum_examples
                    ]
                ],
                recommended_action=FindingAction.RAISE,
                applied_action=FindingAction.REPORT,
                blocks_readiness=True,
            )
        )

        if not self.options.detect_conflicting_targets:
            return

        target_column = self.options.target_column

        if (
            target_column is None
            or target_column not in data.columns
        ):
            return

        duplicated_data = data.loc[
            duplicate_mask,
            [
                *self.options.observation_key,
                target_column,
            ],
        ]

        target_counts = duplicated_data.groupby(
            self.options.observation_key,
            dropna=False,
        )[target_column].nunique(
            dropna=False
        )

        conflicting_keys = target_counts[
            target_counts > 1
        ]

        if conflicting_keys.empty:
            return

        result.add_finding(
            Finding(
                code="P1-DUPROW-004",
                category=FindingCategory.DUPLICATE_ROW,
                severity=Severity.ERROR,
                component=self.component_name,
                message=(
                    "Duplicated observation keys contain conflicting "
                    "target values."
                ),
                columns=[
                    *self.options.observation_key,
                    target_column,
                ],
                affected_row_count=len(conflicting_keys),
                recommended_action=FindingAction.RAISE,
                applied_action=FindingAction.REPORT,
                blocks_readiness=True,
            )
        )

    def _comparison_columns(
        self,
        data: pd.DataFrame,
    ) -> list[object]:
        """Resolve columns used for exact duplicate comparison."""

        excluded = set(self.options.exclude_columns)

        excluded.add(
            self.options.internal_row_id_column
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

    def _extract_row_id(
        self,
        *,
        row: pd.Series,
        index: object,
    ) -> str | int:
        """Extract a stable record identifier."""

        if (
            self.options.internal_row_id_column
            in row.index
        ):
            return self._safe_row_id(
                row[
                    self.options.internal_row_id_column
                ]
            )

        return self._safe_row_id(index)

    @staticmethod
    def _safe_row_id(
        value: object,
    ) -> str | int:
        """Convert an index value into a supported identifier."""

        if isinstance(value, (str, int)):
            return value

        return str(value)

    @staticmethod
    def _percentage(
        affected_count: int,
        total_count: int,
    ) -> float:
        """Calculate a percentage safely."""

        if total_count <= 0:
            return 0.0

        return round(
            affected_count / total_count * 100.0,
            6,
        )