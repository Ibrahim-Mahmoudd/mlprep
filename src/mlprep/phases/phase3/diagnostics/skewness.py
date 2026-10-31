"""Numeric skewness diagnostics for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import EDAContract


SkewnessLevel = Literal[
    "symmetric",
    "moderate",
    "high",
    "undefined",
]

TransformationSuggestion = Literal[
    "none",
    "log",
    "log1p",
    "box_cox",
    "yeo_johnson",
    "quantile",
    "review",
]


class SkewnessDiagnostic(BaseModel):
    """Skewness diagnostic for one numeric feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    valid_count: int = Field(
        default=0,
        ge=0,
    )

    unique_count: int = Field(
        default=0,
        ge=0,
    )

    minimum: float | None = None
    maximum: float | None = None

    skewness: float | None = None
    absolute_skewness: float | None = None

    level: SkewnessLevel

    contains_zero: bool = False
    contains_negative: bool = False

    suggested_transformation: TransformationSuggestion
    reason: str

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class SkewnessDiagnosticResult(BaseModel):
    """Complete numeric skewness diagnostic output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    diagnostics: dict[
        str,
        SkewnessDiagnostic,
    ] = Field(default_factory=dict)

    moderately_skewed_columns: list[str] = Field(
        default_factory=list
    )

    highly_skewed_columns: list[str] = Field(
        default_factory=list
    )

    transformation_candidates: dict[
        str,
        TransformationSuggestion,
    ] = Field(default_factory=dict)


@dataclass(slots=True)
class SkewnessDiagnosticOptions:
    """Options controlling skewness diagnostics."""

    enabled: bool = True

    minimum_valid_count: int = 3
    minimum_unique_count: int = 3

    moderate_threshold: float = 0.50
    high_threshold: float = 1.00

    recommend_quantile_above: float = 3.00

    def __post_init__(self) -> None:
        """Validate skewness options."""

        if self.minimum_valid_count < 3:
            raise ValueError(
                "minimum_valid_count must be at least three."
            )

        if self.minimum_unique_count < 2:
            raise ValueError(
                "minimum_unique_count must be at least two."
            )

        if self.moderate_threshold < 0.0:
            raise ValueError(
                "moderate_threshold cannot be negative."
            )

        if self.high_threshold < self.moderate_threshold:
            raise ValueError(
                "high_threshold cannot be below "
                "moderate_threshold."
            )

        if self.recommend_quantile_above < self.high_threshold:
            raise ValueError(
                "recommend_quantile_above cannot be below "
                "high_threshold."
            )


class SkewnessDiagnosticAnalyzer:
    """Diagnose skewed numeric training features."""

    component_name = "skewness"

    def __init__(
        self,
        options: SkewnessDiagnosticOptions | None = None,
    ) -> None:
        """Initialize the skewness analyzer."""

        self.options = (
            options or SkewnessDiagnosticOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> SkewnessDiagnosticResult:
        """Analyze skewness for selected numeric features."""

        if not self.options.enabled:
            return SkewnessDiagnosticResult()

        frame = contract.training_features()

        diagnostics: dict[
            str,
            SkewnessDiagnostic,
        ] = {}

        moderate_columns: list[str] = []
        high_columns: list[str] = []

        for column in columns.numeric:
            if column not in frame.columns:
                continue

            diagnostic = self._analyze_column(
                column=column,
                series=frame[column],
            )

            diagnostics[column] = diagnostic

            if diagnostic.level == "moderate":
                moderate_columns.append(column)

            elif diagnostic.level == "high":
                high_columns.append(column)

        return SkewnessDiagnosticResult(
            diagnostics=diagnostics,
            moderately_skewed_columns=moderate_columns,
            highly_skewed_columns=high_columns,
            transformation_candidates={
                column: diagnostic.suggested_transformation
                for column, diagnostic in diagnostics.items()
                if diagnostic.suggested_transformation != "none"
            },
        )

    def _analyze_column(
        self,
        *,
        column: str,
        series: pd.Series,
    ) -> SkewnessDiagnostic:
        """Analyze one numeric feature."""

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid = numeric.dropna()

        valid_count = len(valid)
        unique_count = int(
            valid.nunique()
        )

        if (
            valid_count < self.options.minimum_valid_count
            or unique_count
            < self.options.minimum_unique_count
        ):
            return SkewnessDiagnostic(
                column=column,
                valid_count=valid_count,
                unique_count=unique_count,
                minimum=self._safe_float(
                    valid.min()
                ),
                maximum=self._safe_float(
                    valid.max()
                ),
                level="undefined",
                suggested_transformation="review",
                reason=(
                    "The feature contains too few valid or "
                    "unique values for reliable skewness analysis."
                ),
            )

        skewness = float(
            valid.skew()
        )

        if not np.isfinite(skewness):
            return SkewnessDiagnostic(
                column=column,
                valid_count=valid_count,
                unique_count=unique_count,
                minimum=self._safe_float(
                    valid.min()
                ),
                maximum=self._safe_float(
                    valid.max()
                ),
                level="undefined",
                suggested_transformation="review",
                reason=(
                    "A finite skewness value could not be calculated."
                ),
            )

        absolute_skewness = abs(skewness)

        contains_zero = bool(
            (valid == 0).any()
        )

        contains_negative = bool(
            (valid < 0).any()
        )

        level = self._level(
            absolute_skewness
        )

        transformation, reason = (
            self._recommend_transformation(
                absolute_skewness=absolute_skewness,
                contains_zero=contains_zero,
                contains_negative=contains_negative,
            )
        )

        return SkewnessDiagnostic(
            column=column,
            valid_count=valid_count,
            unique_count=unique_count,
            minimum=float(valid.min()),
            maximum=float(valid.max()),
            skewness=skewness,
            absolute_skewness=absolute_skewness,
            level=level,
            contains_zero=contains_zero,
            contains_negative=contains_negative,
            suggested_transformation=transformation,
            reason=reason,
        )

    def _level(
        self,
        absolute_skewness: float,
    ) -> SkewnessLevel:
        """Classify an absolute skewness value."""

        if (
            absolute_skewness
            < self.options.moderate_threshold
        ):
            return "symmetric"

        if absolute_skewness < self.options.high_threshold:
            return "moderate"

        return "high"

    def _recommend_transformation(
        self,
        *,
        absolute_skewness: float,
        contains_zero: bool,
        contains_negative: bool,
    ) -> tuple[
        TransformationSuggestion,
        str,
    ]:
        """Recommend a numeric transformation."""

        if (
            absolute_skewness
            < self.options.moderate_threshold
        ):
            return (
                "none",
                "The distribution is approximately symmetric.",
            )

        if (
            absolute_skewness
            >= self.options.recommend_quantile_above
        ):
            return (
                "quantile",
                "The distribution is extremely skewed.",
            )

        if contains_negative:
            return (
                "yeo_johnson",
                "The feature is skewed and contains "
                "negative values.",
            )

        if contains_zero:
            return (
                "log1p",
                "The feature is positively transformable but "
                "contains zero values.",
            )

        return (
            "box_cox",
            "The feature is skewed and contains strictly "
            "positive values.",
        )

    @staticmethod
    def _safe_float(
        value: object,
    ) -> float | None:
        """Convert a finite number safely."""

        if pd.isna(value):
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result