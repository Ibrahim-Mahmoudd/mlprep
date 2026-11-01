"""Categorical feature profiling for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)


class CategoricalColumnProfile(BaseModel):
    """Analytical profile for one categorical feature."""

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

    dominant_value: str | None = None

    dominant_value_count: int = Field(
        default=0,
        ge=0,
    )

    dominant_value_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    entropy: float = Field(
        default=0.0,
        ge=0.0,
    )

    normalized_entropy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    rare_category_count: int = Field(
        default=0,
        ge=0,
    )

    rare_categories: list[str] = Field(
        default_factory=list
    )

    high_cardinality: bool = False
    identifier_like: bool = False

    category_frequencies: list[
        dict[str, Any]
    ] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class CategoricalProfilerOptions:
    """Options controlling categorical profiling."""

    enabled: bool = True

    maximum_categories_in_profile: int = 50

    rare_category_minimum_count: int = 5
    rare_category_maximum_ratio: float = 0.01

    high_cardinality_threshold: int = 100
    identifier_uniqueness_threshold: float = 0.95

    include_missing_as_category: bool = False

    def __post_init__(self) -> None:
        """Validate categorical-profiler options."""

        if self.maximum_categories_in_profile < 1:
            raise ValueError(
                "maximum_categories_in_profile must be "
                "at least one."
            )

        if self.rare_category_minimum_count < 1:
            raise ValueError(
                "rare_category_minimum_count must be "
                "at least one."
            )

        bounded_values = {
            "rare_category_maximum_ratio": (
                self.rare_category_maximum_ratio
            ),
            "identifier_uniqueness_threshold": (
                self.identifier_uniqueness_threshold
            ),
        }

        for name, value in bounded_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between zero and one."
                )


class CategoricalProfiler:
    """Profile categorical and Boolean training features."""

    component_name = "categorical_profile"

    def __init__(
        self,
        options: (
            CategoricalProfilerOptions | None
        ) = None,
    ) -> None:
        """Initialize the categorical profiler."""

        self.options = (
            options
            or CategoricalProfilerOptions()
        )

    def profile(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> dict[str, CategoricalColumnProfile]:
        """Profile selected categorical columns."""

        if not self.options.enabled:
            return {}

        frame = contract.training_features()

        selected_columns = list(
            dict.fromkeys(
                [
                    *columns.categorical,
                    *columns.boolean,
                ]
            )
        )

        profiles: dict[
            str,
            CategoricalColumnProfile,
        ] = {}

        for column in selected_columns:
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
    ) -> CategoricalColumnProfile:
        """Profile one categorical feature."""

        row_count = len(series)

        missing_count = int(
            series.isna().sum()
        )

        nonmissing = series.dropna()

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

        counted_series = series

        if not self.options.include_missing_as_category:
            counted_series = nonmissing

        counts = counted_series.value_counts(
            dropna=(
                not self.options
                .include_missing_as_category
            )
        )

        denominator = int(
            counts.sum()
        )

        dominant_value = None
        dominant_count = 0
        dominant_percentage = 0.0

        if not counts.empty:
            dominant_value = self._safe_text(
                counts.index[0]
            )

            dominant_count = int(
                counts.iloc[0]
            )

            dominant_percentage = (
                dominant_count
                / denominator
                * 100.0
                if denominator > 0
                else 0.0
            )

        probabilities = [
            float(count) / denominator
            for count in counts.tolist()
            if denominator > 0
        ]

        entropy = -sum(
            probability * log2(probability)
            for probability in probabilities
            if probability > 0
        )

        maximum_entropy = (
            log2(len(probabilities))
            if len(probabilities) > 1
            else 0.0
        )

        normalized_entropy = (
            min(1.0, entropy / maximum_entropy)
            if maximum_entropy > 0
            else 0.0
        )

        rare_categories: list[str] = []

        for value, count in counts.items():
            count_value = int(count)

            ratio = (
                count_value / denominator
                if denominator > 0
                else 0.0
            )

            if (
                count_value
                < self.options
                .rare_category_minimum_count
                or ratio
                <= self.options
                .rare_category_maximum_ratio
            ):
                rare_categories.append(
                    self._safe_text(value)
                )

        frequencies = [
            {
                "value": self._safe_text(value),
                "count": int(count),
                "percentage": (
                    int(count)
                    / denominator
                    * 100.0
                    if denominator > 0
                    else 0.0
                ),
            }
            for value, count
            in counts.head(
                self.options
                .maximum_categories_in_profile
            ).items()
        ]

        return CategoricalColumnProfile(
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
            dominant_value=dominant_value,
            dominant_value_count=dominant_count,
            dominant_value_percentage=round(
                dominant_percentage,
                6,
            ),
            entropy=float(entropy),
            normalized_entropy=float(
                normalized_entropy
            ),
            rare_category_count=len(
                rare_categories
            ),
            rare_categories=rare_categories,
            high_cardinality=(
                unique_count
                >= self.options
                .high_cardinality_threshold
            ),
            identifier_like=(
                uniqueness_ratio
                >= self.options
                .identifier_uniqueness_threshold
            ),
            category_frequencies=frequencies,
        )

    @staticmethod
    def _safe_text(
        value: object,
    ) -> str:
        """Convert a category value safely."""

        if pd.isna(value):
            return "__MISSING__"

        return str(value)
