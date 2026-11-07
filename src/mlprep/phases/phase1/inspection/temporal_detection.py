"""Temporal-value detection and validation for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_object_dtype,
    is_string_dtype,
)

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.types import (
    FindingAction,
    FindingCategory,
    PhysicalType,
    SemanticType,
    Severity,
)

DateOrder = Literal[
    "DMY",
    "MDY",
    "YMD",
]


@dataclass(slots=True)
class TemporalDetectionOptions:
    """Options controlling temporal detection."""

    enabled: bool = True

    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(
        default_factory=list
    )

    parse_threshold: float = 0.90
    sample_size: int | None = 20_000

    date_order: DateOrder = "DMY"

    convert_detected_columns: bool = False

    reject_ambiguous_numeric_dates: bool = True
    report_parse_failures: bool = True

    minimum_year: int | None = None
    maximum_year: int | None = None

    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate temporal detection options."""

        if not 0.0 <= self.parse_threshold <= 1.0:
            raise ValueError(
                "parse_threshold must be between zero and one."
            )

        if (
            self.sample_size is not None
            and self.sample_size <= 0
        ):
            raise ValueError(
                "sample_size must be positive or None."
            )

        if (
            self.minimum_year is not None
            and self.maximum_year is not None
            and self.minimum_year > self.maximum_year
        ):
            raise ValueError(
                "minimum_year cannot exceed maximum_year."
            )

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )


class TemporalDetector:
    """Detect date-like and timestamp-like text columns."""

    component_name = "temporal_detection"

    def __init__(
        self,
        options: TemporalDetectionOptions | None = None,
    ) -> None:
        """Initialize the temporal detector."""

        self.options = (
            options or TemporalDetectionOptions()
        )

    def inspect(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Inspect and optionally convert temporal columns."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Temporal detection requires a pandas DataFrame."
            )

        working_data = data.copy()

        if not self.options.enabled:
            return working_data

        for column in self._select_columns(working_data):
            self._inspect_column(
                data=working_data,
                column=column,
                result=result,
                state=state,
            )

        return working_data

    def _inspect_column(
        self,
        *,
        data: pd.DataFrame,
        column: str,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Inspect one possible temporal column."""

        series = data[column]

        if is_datetime64_any_dtype(series.dtype):
            self._register_temporal_resolution(
                column=column,
                parse_ratio=1.0,
                failed_count=0,
                state=state,
                converted=False,
            )

            self._validate_year_range(
                parsed_values=series,
                column=column,
                result=result,
                data=data,
            )

            return

        non_missing_series = series.dropna()

        if non_missing_series.empty:
            return

        sample = self._sample_series(
            non_missing_series
        )

        ambiguous_mask = sample.astype(
            "string"
        ).str.match(
            r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*$",
            na=False,
        )

        ambiguous_count = int(
            ambiguous_mask.sum()
        )

        if (
            ambiguous_count > 0
            and self.options.reject_ambiguous_numeric_dates
        ):
            result.add_finding(
                Finding(
                    code="P1-TEMPORAL-001",
                    category=FindingCategory.TEMPORAL,
                    severity=Severity.WARNING,
                    component=self.component_name,
                    message=(
                        f"Column {column!r} contains potentially "
                        "ambiguous numeric date strings."
                    ),
                    columns=[column],
                    affected_row_count=ambiguous_count,
                    affected_row_percentage=self._percentage(
                        affected_count=ambiguous_count,
                        total_count=len(sample),
                    ),
                    example_values=sample.loc[
                        ambiguous_mask
                    ]
                    .head(self.options.maximum_examples)
                    .astype(str)
                    .tolist(),
                    recommended_action=FindingAction.WARN,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=False,
                    details={
                        "configured_date_order": (
                            self.options.date_order
                        ),
                    },
                )
            )

        parsed_sample = pd.to_datetime(
            sample,
            errors="coerce",
            format="mixed",
            dayfirst=(
                self.options.date_order == "DMY"
            ),
            yearfirst=(
                self.options.date_order == "YMD"
            ),
        )

        parsed_count = int(
            parsed_sample.notna().sum()
        )

        parse_ratio = (
            parsed_count / len(sample)
            if len(sample) > 0
            else 0.0
        )

        if parse_ratio < self.options.parse_threshold:
            return

        parsed_full = pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=(
                self.options.date_order == "DMY"
            ),
            yearfirst=(
                self.options.date_order == "YMD"
            ),
        )

        failed_mask = (
            series.notna()
            & parsed_full.isna()
        )

        failed_count = int(
            failed_mask.sum()
        )

        self._register_temporal_resolution(
            column=column,
            parse_ratio=parse_ratio,
            failed_count=failed_count,
            state=state,
            converted=self.options.convert_detected_columns,
        )

        result.add_finding(
            Finding(
                code="P1-TEMPORAL-002",
                category=FindingCategory.TEMPORAL,
                severity=Severity.INFO,
                component=self.component_name,
                message=(
                    f"Column {column!r} was detected as "
                    "date-like or timestamp-like."
                ),
                columns=[column],
                affected_row_count=parsed_count,
                affected_row_percentage=round(
                    parse_ratio * 100.0,
                    6,
                ),
                recommended_action=FindingAction.COERCE,
                applied_action=(
                    FindingAction.COERCE
                    if self.options.convert_detected_columns
                    else FindingAction.REPORT
                ),
                blocks_readiness=False,
                details={
                    "parse_success_ratio": parse_ratio,
                    "failed_parse_count": failed_count,
                },
            )
        )

        if (
            failed_count > 0
            and self.options.report_parse_failures
        ):
            result.add_finding(
                Finding(
                    code="P1-TEMPORAL-003",
                    category=FindingCategory.TEMPORAL,
                    severity=Severity.WARNING,
                    component=self.component_name,
                    message=(
                        f"Column {column!r} contains temporal "
                        "values that could not be parsed."
                    ),
                    columns=[column],
                    affected_row_count=failed_count,
                    affected_row_percentage=self._percentage(
                        affected_count=failed_count,
                        total_count=len(data),
                    ),
                    example_values=series.loc[
                        failed_mask
                    ]
                    .head(self.options.maximum_examples)
                    .astype(str)
                    .tolist(),
                    recommended_action=FindingAction.COERCE,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=False,
                )
            )

        self._validate_year_range(
            parsed_values=parsed_full,
            column=column,
            result=result,
            data=data,
        )

        if self.options.convert_detected_columns:
            data[column] = parsed_full

    def _register_temporal_resolution(
        self,
        *,
        column: str,
        parse_ratio: float,
        failed_count: int,
        state: Phase1State,
        converted: bool,
    ) -> None:
        """Update type resolution for one temporal column."""

        resolution = (
            state.type_resolution_states.get(column)
        )

        if resolution is None:
            return

        resolution.semantic_type = (
            SemanticType.DATETIME
        )

        resolution.semantic_type_confidence = (
            parse_ratio
        )

        resolution.type_was_inferred = True
        resolution.parse_success_ratio = parse_ratio
        resolution.failed_parse_count = failed_count
        resolution.coercion_was_applied = converted

        if converted:
            resolution.resolved_physical_type = (
                PhysicalType.DATETIME
            )

        state.register_type_resolution(
            resolution
        )

        column_state = state.column_states.get(column)

        if column_state is None:
            return

        column_state.semantic_type = (
            SemanticType.DATETIME
        )

        column_state.semantic_type_confidence = (
            parse_ratio
        )

        if converted:
            column_state.physical_type = (
                PhysicalType.DATETIME
            )

        state.register_column_state(
            column_state
        )

    def _validate_year_range(
        self,
        *,
        parsed_values: pd.Series,
        column: str,
        result: Phase1Result,
        data: pd.DataFrame,
    ) -> None:
        """Validate optional minimum and maximum years."""

        if (
            self.options.minimum_year is None
            and self.options.maximum_year is None
        ):
            return

        valid_values = parsed_values.dropna()

        if valid_values.empty:
            return

        invalid_mask = pd.Series(
            False,
            index=parsed_values.index,
            dtype=bool,
        )

        if self.options.minimum_year is not None:
            invalid_mask |= (
                parsed_values.dt.year
                < self.options.minimum_year
            ).fillna(False)

        if self.options.maximum_year is not None:
            invalid_mask |= (
                parsed_values.dt.year
                > self.options.maximum_year
            ).fillna(False)

        invalid_count = int(
            invalid_mask.sum()
        )

        if invalid_count == 0:
            return

        result.add_finding(
            Finding(
                code="P1-TEMPORAL-004",
                category=FindingCategory.TEMPORAL,
                severity=Severity.WARNING,
                component=self.component_name,
                message=(
                    f"Column {column!r} contains dates outside "
                    "the configured year range."
                ),
                columns=[column],
                affected_row_count=invalid_count,
                affected_row_percentage=self._percentage(
                    affected_count=invalid_count,
                    total_count=len(data),
                ),
                example_values=parsed_values.loc[
                    invalid_mask
                ]
                .head(self.options.maximum_examples)
                .astype(str)
                .tolist(),
                recommended_action=FindingAction.WARN,
                applied_action=FindingAction.REPORT,
                blocks_readiness=False,
                details={
                    "minimum_year": (
                        self.options.minimum_year
                    ),
                    "maximum_year": (
                        self.options.maximum_year
                    ),
                },
            )
        )

    def _select_columns(
        self,
        data: pd.DataFrame,
    ) -> list[object]:
        """Select columns eligible for temporal detection."""

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
            and (
                is_datetime64_any_dtype(
                    data[column].dtype
                )
                or is_string_dtype(
                    data[column].dtype
                )
                or is_object_dtype(
                    data[column].dtype
                )
            )
        ]

    def _sample_series(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Select a deterministic inspection sample."""

        if self.options.sample_size is None:
            return series

        if len(series) <= self.options.sample_size:
            return series

        return series.iloc[
            : self.options.sample_size
        ]

    @staticmethod
    def _percentage(
        *,
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
