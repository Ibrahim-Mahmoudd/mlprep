"""Numeric feature-association analysis for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)


CorrelationMethod = Literal[
    "pearson",
    "spearman",
    "kendall",
]


class NumericAssociationPair(BaseModel):
    """Association between two numeric features."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    left_column: str
    right_column: str

    pearson: float | None = None
    spearman: float | None = None
    kendall: float | None = None

    absolute_primary_correlation: float | None = None

    valid_pair_count: int = Field(
        default=0,
        ge=0,
    )

    high_correlation: bool = False
    perfect_correlation: bool = False

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class NumericAssociationResult(BaseModel):
    """Complete numeric-association output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    analyzed_columns: list[str] = Field(
        default_factory=list
    )

    primary_method: CorrelationMethod

    correlation_matrices: dict[
        str,
        dict[str, dict[str, float | None]],
    ] = Field(default_factory=dict)

    pairs: list[NumericAssociationPair] = Field(
        default_factory=list
    )

    highly_correlated_pairs: list[
        NumericAssociationPair
    ] = Field(default_factory=list)

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class NumericAssociationOptions:
    """Options controlling numeric-association analysis."""

    enabled: bool = True

    methods: tuple[CorrelationMethod, ...] = (
        "pearson",
        "spearman",
    )

    primary_method: CorrelationMethod = "spearman"

    minimum_valid_pairs: int = 3

    high_correlation_threshold: float = 0.90
    perfect_correlation_threshold: float = 0.999999

    include_all_pairs: bool = True
    maximum_columns: int = 500

    def __post_init__(self) -> None:
        """Validate numeric-association options."""

        allowed_methods = {
            "pearson",
            "spearman",
            "kendall",
        }

        if not self.methods:
            raise ValueError(
                "At least one correlation method is required."
            )

        if not set(self.methods).issubset(
            allowed_methods
        ):
            raise ValueError(
                "Unsupported numeric correlation method."
            )

        if self.primary_method not in self.methods:
            raise ValueError(
                "primary_method must be included in methods."
            )

        if self.minimum_valid_pairs < 2:
            raise ValueError(
                "minimum_valid_pairs must be at least two."
            )

        if not (
            0.0
            <= self.high_correlation_threshold
            <= 1.0
        ):
            raise ValueError(
                "high_correlation_threshold must be between "
                "zero and one."
            )

        if not (
            self.high_correlation_threshold
            <= self.perfect_correlation_threshold
            <= 1.0
        ):
            raise ValueError(
                "perfect_correlation_threshold must be at "
                "least the high-correlation threshold."
            )

        if self.maximum_columns < 2:
            raise ValueError(
                "maximum_columns must be at least two."
            )


class NumericAssociationAnalyzer:
    """Analyze relationships between numeric training features."""

    component_name = "numeric_associations"

    def __init__(
        self,
        options: NumericAssociationOptions | None = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options or NumericAssociationOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> NumericAssociationResult:
        """Analyze numeric feature associations."""

        selected_columns = [
            column
            for column in columns.numeric
            if column in contract.train.X.columns
        ]

        if not self.options.enabled:
            return NumericAssociationResult(
                primary_method=self.options.primary_method,
            )

        if len(selected_columns) > self.options.maximum_columns:
            selected_columns = selected_columns[
                : self.options.maximum_columns
            ]

        if len(selected_columns) < 2:
            return NumericAssociationResult(
                analyzed_columns=selected_columns,
                primary_method=self.options.primary_method,
            )

        numeric_frame = contract.train.X.loc[
            :,
            selected_columns,
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        matrices: dict[
            str,
            dict[str, dict[str, float | None]],
        ] = {}

        raw_matrices: dict[
            str,
            pd.DataFrame,
        ] = {}

        for method in self.options.methods:
            matrix = numeric_frame.corr(
                method=method,
                min_periods=self.options.minimum_valid_pairs,
            )

            raw_matrices[method] = matrix

            matrices[method] = self._serialize_matrix(
                matrix
            )

        pairs: list[NumericAssociationPair] = []
        highly_correlated: list[
            NumericAssociationPair
        ] = []

        for left_position, left_column in enumerate(
            selected_columns
        ):
            for right_column in selected_columns[
                left_position + 1:
            ]:
                valid_mask = (
                    numeric_frame[left_column].notna()
                    & numeric_frame[right_column].notna()
                )

                values = {
                    method: self._safe_float(
                        raw_matrices[method].loc[
                            left_column,
                            right_column,
                        ]
                    )
                    for method in self.options.methods
                }

                primary_value = values.get(
                    self.options.primary_method
                )

                absolute_primary = (
                    abs(primary_value)
                    if primary_value is not None
                    else None
                )

                pair = NumericAssociationPair(
                    left_column=left_column,
                    right_column=right_column,
                    pearson=values.get("pearson"),
                    spearman=values.get("spearman"),
                    kendall=values.get("kendall"),
                    absolute_primary_correlation=(
                        absolute_primary
                    ),
                    valid_pair_count=int(
                        valid_mask.sum()
                    ),
                    high_correlation=bool(
                        absolute_primary is not None
                        and absolute_primary
                        >= self.options
                        .high_correlation_threshold
                    ),
                    perfect_correlation=bool(
                        absolute_primary is not None
                        and absolute_primary
                        >= self.options
                        .perfect_correlation_threshold
                    ),
                )

                if self.options.include_all_pairs:
                    pairs.append(pair)

                if pair.high_correlation:
                    highly_correlated.append(pair)

        highly_correlated.sort(
            key=lambda pair: (
                pair.absolute_primary_correlation
                or 0.0
            ),
            reverse=True,
        )

        return NumericAssociationResult(
            analyzed_columns=selected_columns,
            primary_method=self.options.primary_method,
            correlation_matrices=matrices,
            pairs=pairs,
            highly_correlated_pairs=highly_correlated,
        )

    @staticmethod
    def _serialize_matrix(
        matrix: pd.DataFrame,
    ) -> dict[str, dict[str, float | None]]:
        """Serialize a correlation matrix safely."""

        return {
            str(row): {
                str(column): (
                    None
                    if pd.isna(value)
                    else float(value)
                )
                for column, value
                in matrix.loc[row].items()
            }
            for row in matrix.index
        }

    @staticmethod
    def _safe_float(
        value: object,
    ) -> float | None:
        """Convert a finite numeric value safely."""

        if pd.isna(value):
            return None

        numeric_value = float(value)

        if not np.isfinite(numeric_value):
            return None

        return numeric_value