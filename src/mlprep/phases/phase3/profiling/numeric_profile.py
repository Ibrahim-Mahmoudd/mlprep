"""Numeric feature profiling for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)


class NumericColumnProfile(BaseModel):
    """Analytical profile for one numeric feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    dtype: str

    row_count: int = Field(ge=0)
    nonmissing_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)

    missing_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    unique_count: int = Field(ge=0)
    uniqueness_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )

    minimum: float | None = None
    maximum: float | None = None

    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    variance: float | None = None

    first_quartile: float | None = None
    third_quartile: float | None = None

    interquartile_range: float | None = None
    median_absolute_deviation: float | None = None

    skewness: float | None = None
    kurtosis: float | None = None

    zero_count: int = Field(ge=0)
    zero_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    negative_count: int = Field(ge=0)
    negative_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    positive_count: int = Field(ge=0)
    infinite_count: int = Field(ge=0)

    quantiles: dict[str, float | None] = Field(
        default_factory=dict
    )

    most_frequent_values: list[
        dict[str, Any]
    ] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class NumericProfilerOptions:
    """Options controlling numeric profiling."""

    enabled: bool = True

    quantiles: tuple[float, ...] = (
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    )

    maximum_frequent_values: int = 5

    calculate_skewness: bool = True
    calculate_kurtosis: bool = True
    calculate_mad: bool = True

    def __post_init__(self) -> None:
        """Validate numeric-profiler options."""

        if any(
            not 0.0 <= quantile <= 1.0
            for quantile in self.quantiles
        ):
            raise ValueError(
                "Every quantile must be between zero and one."
            )

        if self.maximum_frequent_values < 0:
            raise ValueError(
                "maximum_frequent_values cannot be negative."
            )


class NumericProfiler:
    """Profile all numeric training features."""

    component_name = "numeric_profile"

    def __init__(
        self,
        options: NumericProfilerOptions | None = None,
    ) -> None:
        """Initialize the numeric profiler."""

        self.options = (
            options or NumericProfilerOptions()
        )

    def profile(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> dict[str, NumericColumnProfile]:
        """Profile every selected numeric column."""

        if not self.options.enabled:
            return {}

        frame = contract.training_features()

        profiles: dict[
            str,
            NumericColumnProfile,
        ] = {}

        for column in columns.numeric:
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
    ) -> NumericColumnProfile:
        """Profile one numeric feature."""

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        row_count = len(numeric)

        missing_count = int(
            numeric.isna().sum()
        )

        nonmissing = numeric.dropna()

        nonmissing_count = len(nonmissing)

        unique_count = int(
            nonmissing.nunique(
                dropna=True
            )
        )

        uniqueness_ratio = (
            unique_count / nonmissing_count
            if nonmissing_count > 0
            else 0.0
        )

        missing_percentage = (
            missing_count / row_count * 100.0
            if row_count > 0
            else 0.0
        )

        finite = nonmissing[
            np.isfinite(
                nonmissing.astype(float)
            )
        ]

        infinite_count = (
            nonmissing_count - len(finite)
        )

        zero_count = int(
            (finite == 0).sum()
        )

        negative_count = int(
            (finite < 0).sum()
        )

        positive_count = int(
            (finite > 0).sum()
        )

        zero_percentage = (
            zero_count / nonmissing_count * 100.0
            if nonmissing_count > 0
            else 0.0
        )

        negative_percentage = (
            negative_count
            / nonmissing_count
            * 100.0
            if nonmissing_count > 0
            else 0.0
        )

        if finite.empty:
            return NumericColumnProfile(
                column=column,
                dtype=str(series.dtype),
                row_count=row_count,
                nonmissing_count=nonmissing_count,
                missing_count=missing_count,
                missing_percentage=round(
                    missing_percentage,
                    6,
                ),
                unique_count=unique_count,
                uniqueness_ratio=(
                    uniqueness_ratio
                ),
                zero_count=zero_count,
                zero_percentage=(
                    zero_percentage
                ),
                negative_count=negative_count,
                negative_percentage=(
                    negative_percentage
                ),
                positive_count=positive_count,
                infinite_count=infinite_count,
            )

        first_quartile = float(
            finite.quantile(0.25)
        )

        third_quartile = float(
            finite.quantile(0.75)
        )

        median = float(
            finite.median()
        )

        absolute_deviations = (
            finite - median
        ).abs()

        quantile_values = {
            str(quantile): self._safe_float(
                finite.quantile(quantile)
            )
            for quantile in self.options.quantiles
        }

        frequencies = finite.value_counts(
            dropna=False
        ).head(
            self.options.maximum_frequent_values
        )

        frequent_values = [
            {
                "value": self._safe_float(value),
                "count": int(count),
                "percentage": (
                    float(count)
                    / nonmissing_count
                    * 100.0
                    if nonmissing_count > 0
                    else 0.0
                ),
            }
            for value, count
            in frequencies.items()
        ]

        return NumericColumnProfile(
            column=column,
            dtype=str(series.dtype),
            row_count=row_count,
            nonmissing_count=nonmissing_count,
            missing_count=missing_count,
            missing_percentage=round(
                missing_percentage,
                6,
            ),
            unique_count=unique_count,
            uniqueness_ratio=(
                uniqueness_ratio
            ),
            minimum=self._safe_float(
                finite.min()
            ),
            maximum=self._safe_float(
                finite.max()
            ),
            mean=self._safe_float(
                finite.mean()
            ),
            median=median,
            standard_deviation=self._safe_float(
                finite.std(ddof=0)
            ),
            variance=self._safe_float(
                finite.var(ddof=0)
            ),
            first_quartile=first_quartile,
            third_quartile=third_quartile,
            interquartile_range=(
                third_quartile
                - first_quartile
            ),
            median_absolute_deviation=(
                self._safe_float(
                    absolute_deviations.median()
                )
                if self.options.calculate_mad
                else None
            ),
            skewness=(
                self._safe_float(
                    finite.skew()
                )
                if (
                    self.options.calculate_skewness
                    and len(finite) >= 3
                )
                else None
            ),
            kurtosis=(
                self._safe_float(
                    finite.kurtosis()
                )
                if (
                    self.options.calculate_kurtosis
                    and len(finite) >= 4
                )
                else None
            ),
            zero_count=zero_count,
            zero_percentage=round(
                zero_percentage,
                6,
            ),
            negative_count=negative_count,
            negative_percentage=round(
                negative_percentage,
                6,
            ),
            positive_count=positive_count,
            infinite_count=infinite_count,
            quantiles=quantile_values,
            most_frequent_values=(
                frequent_values
            ),
        )

    @staticmethod
    def _safe_float(
        value: object,
    ) -> float | None:
        """Convert a numeric value safely."""

        if pd.isna(value):
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result