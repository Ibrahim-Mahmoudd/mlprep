"""Datetime feature profiling for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)


class DatetimeColumnProfile(BaseModel):
    """Analytical profile for one datetime feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    dtype: str

    row_count: int = Field(ge=0)
    nonmissing_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)

    missing_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    unique_count: int = Field(ge=0)

    minimum_timestamp: str | None = None
    maximum_timestamp: str | None = None

    duration_seconds: float | None = None

    most_frequent_years: list[
        dict[str, int]
    ] = Field(default_factory=list)

    most_frequent_months: list[
        dict[str, int]
    ] = Field(default_factory=list)

    most_frequent_weekdays: list[
        dict[str, int]
    ] = Field(default_factory=list)

    most_frequent_hours: list[
        dict[str, int]
    ] = Field(default_factory=list)

    median_interval_seconds: float | None = None

    timezone: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class DatetimeProfilerOptions:
    """Options controlling datetime profiling."""

    enabled: bool = True

    maximum_frequency_values: int = 10

    infer_from_text: bool = True
    day_first: bool = False
    year_first: bool = False

    calculate_intervals: bool = True

    def __post_init__(self) -> None:
        """Validate datetime-profiler options."""

        if self.maximum_frequency_values < 1:
            raise ValueError(
                "maximum_frequency_values must be "
                "at least one."
            )


class DatetimeProfiler:
    """Profile datetime-like training features."""

    component_name = "datetime_profile"

    def __init__(
        self,
        options: DatetimeProfilerOptions | None = None,
    ) -> None:
        """Initialize the datetime profiler."""

        self.options = (
            options or DatetimeProfilerOptions()
        )

    def profile(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> dict[str, DatetimeColumnProfile]:
        """Profile selected datetime features."""

        if not self.options.enabled:
            return {}

        frame = contract.training_features()

        profiles: dict[
            str,
            DatetimeColumnProfile,
        ] = {}

        for column in columns.datetime:
            if column not in frame.columns:
                continue

            profiles[column] = (
                self._profile_column(
                    column=column,
                    series=frame[column],
                )
            )

        return profiles

    def _profile_column(
        self,
        *,
        column: str,
        series: pd.Series,
    ) -> DatetimeColumnProfile:
        """Profile one datetime feature."""

        parsed = pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=self.options.day_first,
            yearfirst=self.options.year_first,
        )

        row_count = len(series)

        missing_count = int(
            series.isna().sum()
        )

        invalid_mask = (
            series.notna()
            & parsed.isna()
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        valid = parsed.dropna()

        nonmissing_count = len(valid)

        missing_percentage = (
            missing_count / row_count * 100.0
            if row_count > 0
            else 0.0
        )

        unique_count = int(
            valid.nunique()
        )

        minimum_timestamp = None
        maximum_timestamp = None
        duration_seconds = None
        median_interval_seconds = None

        if not valid.empty:
            minimum = valid.min()
            maximum = valid.max()

            minimum_timestamp = (
                minimum.isoformat()
            )

            maximum_timestamp = (
                maximum.isoformat()
            )

            duration_seconds = float(
                (
                    maximum - minimum
                ).total_seconds()
            )

            if (
                self.options.calculate_intervals
                and len(valid) > 1
            ):
                ordered = valid.sort_values()

                intervals = (
                    ordered.diff()
                    .dropna()
                    .dt.total_seconds()
                )

                if not intervals.empty:
                    median_interval_seconds = (
                        float(
                            intervals.median()
                        )
                    )

        timezone = None

        if hasattr(parsed.dt, "tz"):
            timezone_value = parsed.dt.tz

            if timezone_value is not None:
                timezone = str(
                    timezone_value
                )

        return DatetimeColumnProfile(
            column=column,
            dtype=str(series.dtype),
            row_count=row_count,
            nonmissing_count=nonmissing_count,
            missing_count=missing_count,
            invalid_count=invalid_count,
            missing_percentage=round(
                missing_percentage,
                6,
            ),
            unique_count=unique_count,
            minimum_timestamp=minimum_timestamp,
            maximum_timestamp=maximum_timestamp,
            duration_seconds=duration_seconds,
            most_frequent_years=(
                self._frequency_records(
                    valid.dt.year
                )
            ),
            most_frequent_months=(
                self._frequency_records(
                    valid.dt.month
                )
            ),
            most_frequent_weekdays=(
                self._frequency_records(
                    valid.dt.day_name()
                )
            ),
            most_frequent_hours=(
                self._frequency_records(
                    valid.dt.hour
                )
            ),
            median_interval_seconds=(
                median_interval_seconds
            ),
            timezone=timezone,
        )

    def _frequency_records(
        self,
        values: pd.Series,
    ) -> list[dict[str, int]]:
        """Build frequency records."""

        counts = values.value_counts(
            dropna=False
        ).head(
            self.options.maximum_frequency_values
        )

        return [
            {
                "value": str(value),
                "count": int(count),
            }
            for value, count
            in counts.items()
        ]