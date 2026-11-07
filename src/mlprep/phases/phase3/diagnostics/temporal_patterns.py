"""Temporal-pattern diagnostics for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import EDAContract


class TemporalPatternDiagnostic(BaseModel):
    """Temporal diagnostic for one datetime feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    valid_count: int = Field(
        default=0,
        ge=0,
    )

    invalid_count: int = Field(
        default=0,
        ge=0,
    )

    minimum_timestamp: str | None = None
    maximum_timestamp: str | None = None

    duration_days: float | None = None

    unique_timestamp_count: int = Field(
        default=0,
        ge=0,
    )

    median_interval_seconds: float | None = None
    maximum_interval_seconds: float | None = None

    duplicate_timestamp_count: int = Field(
        default=0,
        ge=0,
    )

    monthly_counts: dict[str, int] = Field(
        default_factory=dict
    )

    weekday_counts: dict[str, int] = Field(
        default_factory=dict
    )

    target_rate_by_period: dict[
        str,
        float | None,
    ] = Field(default_factory=dict)

    irregular_intervals: bool = False
    target_drift_possible: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class TemporalPatternResult(BaseModel):
    """Complete temporal-pattern output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    diagnostics: dict[
        str,
        TemporalPatternDiagnostic,
    ] = Field(default_factory=dict)

    irregular_columns: list[str] = Field(
        default_factory=list
    )

    target_drift_columns: list[str] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class TemporalPatternOptions:
    """Options controlling temporal diagnostics."""

    enabled: bool = True

    day_first: bool = False
    year_first: bool = False

    irregular_interval_ratio_threshold: float = 5.0

    target_rate_range_threshold: float = 0.20

    maximum_periods: int = 120

    def __post_init__(self) -> None:
        """Validate temporal options."""

        if self.irregular_interval_ratio_threshold < 1.0:
            raise ValueError(
                "irregular_interval_ratio_threshold must "
                "be at least one."
            )

        if self.target_rate_range_threshold < 0.0:
            raise ValueError(
                "target_rate_range_threshold cannot be negative."
            )

        if self.maximum_periods < 1:
            raise ValueError(
                "maximum_periods must be positive."
            )


class TemporalPatternAnalyzer:
    """Analyze temporal coverage, intervals, and target drift."""

    component_name = "temporal_patterns"

    def __init__(
        self,
        options: TemporalPatternOptions | None = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options or TemporalPatternOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> TemporalPatternResult:
        """Analyze selected datetime features."""

        if not self.options.enabled:
            return TemporalPatternResult()

        frame = contract.training_features()
        target = contract.training_target()

        diagnostics: dict[
            str,
            TemporalPatternDiagnostic,
        ] = {}

        for column in columns.datetime:
            if column not in frame.columns:
                continue

            diagnostics[column] = (
                self._analyze_column(
                    column=column,
                    series=frame[column],
                    target=target,
                )
            )

        return TemporalPatternResult(
            diagnostics=diagnostics,
            irregular_columns=[
                column
                for column, diagnostic
                in diagnostics.items()
                if diagnostic.irregular_intervals
            ],
            target_drift_columns=[
                column
                for column, diagnostic
                in diagnostics.items()
                if diagnostic.target_drift_possible
            ],
        )

    def _analyze_column(
        self,
        *,
        column: str,
        series: pd.Series,
        target: pd.Series,
    ) -> TemporalPatternDiagnostic:
        """Analyze one datetime feature."""

        parsed = pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=self.options.day_first,
            yearfirst=self.options.year_first,
        )

        invalid_count = int(
            (
                series.notna()
                & parsed.isna()
            ).sum()
        )

        valid = parsed.dropna().sort_values()

        if valid.empty:
            return TemporalPatternDiagnostic(
                column=column,
                invalid_count=invalid_count,
            )

        intervals = (
            valid.drop_duplicates()
            .sort_values()
            .diff()
            .dropna()
            .dt.total_seconds()
        )

        median_interval = (
            float(intervals.median())
            if not intervals.empty
            else None
        )

        maximum_interval = (
            float(intervals.max())
            if not intervals.empty
            else None
        )

        irregular = bool(
            median_interval is not None
            and median_interval > 0
            and maximum_interval is not None
            and (
                maximum_interval
                / median_interval
                >= self.options
                .irregular_interval_ratio_threshold
            )
        )

        period_labels = parsed.dt.to_period(
            "M"
        ).astype("string")

        monthly_counts = (
            period_labels
            .value_counts()
            .sort_index()
            .head(
                self.options.maximum_periods
            )
        )

        weekday_counts = (
            parsed.dt.day_name().value_counts()
        )

        target_rates: dict[
            str,
            float | None,
        ] = {}

        numeric_target = pd.to_numeric(
            target,
            errors="coerce",
        )

        analysis_frame = pd.DataFrame(
            {
                "period": period_labels,
                "target": numeric_target,
            }
        ).dropna()

        if not analysis_frame.empty:
            grouped_target = (
                analysis_frame.groupby(
                    "period",
                    dropna=False,
                )["target"].mean()
            )

            target_rates = {
                str(period): float(value)
                for period, value
                in grouped_target.head(
                    self.options.maximum_periods
                ).items()
            }

        finite_target_rates = list(
            target_rates.values()
        )

        target_rate_range = (
            max(finite_target_rates)
            - min(finite_target_rates)
            if len(finite_target_rates) >= 2
            else 0.0
        )

        return TemporalPatternDiagnostic(
            column=column,
            valid_count=len(valid),
            invalid_count=invalid_count,
            minimum_timestamp=(
                valid.min().isoformat()
            ),
            maximum_timestamp=(
                valid.max().isoformat()
            ),
            duration_days=float(
                (
                    valid.max()
                    - valid.min()
                ).total_seconds()
                / 86_400.0
            ),
            unique_timestamp_count=int(
                valid.nunique()
            ),
            median_interval_seconds=(
                median_interval
            ),
            maximum_interval_seconds=(
                maximum_interval
            ),
            duplicate_timestamp_count=int(
                valid.duplicated(
                    keep=False
                ).sum()
            ),
            monthly_counts={
                str(period): int(count)
                for period, count
                in monthly_counts.items()
            },
            weekday_counts={
                str(day): int(count)
                for day, count
                in weekday_counts.items()
            },
            target_rate_by_period=target_rates,
            irregular_intervals=irregular,
            target_drift_possible=bool(
                target_rate_range
                >= self.options
                .target_rate_range_threshold
            ),
            metadata={
                "target_rate_range": (
                    target_rate_range
                ),
            },
        )
