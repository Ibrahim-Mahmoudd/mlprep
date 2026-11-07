"""Numeric-categorical association analysis for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)


class MixedAssociationPair(BaseModel):
    """Association between numeric and categorical features."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    numeric_column: str
    categorical_column: str

    valid_pair_count: int = Field(
        default=0,
        ge=0,
    )

    category_count: int = Field(
        default=0,
        ge=0,
    )

    eta_squared: float | None = None

    between_group_variance: float | None = None
    total_variance: float | None = None

    group_statistics: dict[
        str,
        dict[str, float | int | None],
    ] = Field(default_factory=dict)

    strong_association: bool = False

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class MixedAssociationResult(BaseModel):
    """Complete numeric-categorical association output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    pairs: list[MixedAssociationPair] = Field(
        default_factory=list
    )

    strong_pairs: list[MixedAssociationPair] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class MixedAssociationOptions:
    """Options controlling mixed-type associations."""

    enabled: bool = True

    include_boolean_columns: bool = True

    minimum_valid_pairs: int = 5
    minimum_group_count: int = 2

    maximum_categories: int = 50

    strong_eta_squared_threshold: float = 0.14

    include_missing_as_category: bool = False

    def __post_init__(self) -> None:
        """Validate mixed-association options."""

        if self.minimum_valid_pairs < 2:
            raise ValueError(
                "minimum_valid_pairs must be at least two."
            )

        if self.minimum_group_count < 1:
            raise ValueError(
                "minimum_group_count must be positive."
            )

        if self.maximum_categories < 2:
            raise ValueError(
                "maximum_categories must be at least two."
            )

        if not (
            0.0
            <= self.strong_eta_squared_threshold
            <= 1.0
        ):
            raise ValueError(
                "strong_eta_squared_threshold must be "
                "between zero and one."
            )


class MixedAssociationAnalyzer:
    """Analyze numeric-to-categorical relationships."""

    component_name = "mixed_associations"

    def __init__(
        self,
        options: MixedAssociationOptions | None = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options or MixedAssociationOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> MixedAssociationResult:
        """Analyze numeric and categorical feature pairs."""

        if not self.options.enabled:
            return MixedAssociationResult()

        frame = contract.train.X

        numeric_columns = [
            column
            for column in columns.numeric
            if column in frame.columns
        ]

        categorical_columns = list(
            columns.categorical
        )

        if self.options.include_boolean_columns:
            categorical_columns.extend(
                columns.boolean
            )

        categorical_columns = [
            column
            for column in dict.fromkeys(
                categorical_columns
            )
            if column in frame.columns
        ]

        pairs: list[MixedAssociationPair] = []
        strong_pairs: list[
            MixedAssociationPair
        ] = []

        for numeric_column in numeric_columns:
            for categorical_column in categorical_columns:
                pair = self._analyze_pair(
                    numeric_column=numeric_column,
                    categorical_column=(
                        categorical_column
                    ),
                    numeric=frame[numeric_column],
                    categorical=frame[
                        categorical_column
                    ],
                )

                if pair is None:
                    continue

                pairs.append(pair)

                if pair.strong_association:
                    strong_pairs.append(pair)

        strong_pairs.sort(
            key=lambda pair: (
                pair.eta_squared or 0.0
            ),
            reverse=True,
        )

        return MixedAssociationResult(
            pairs=pairs,
            strong_pairs=strong_pairs,
        )

    def _analyze_pair(
        self,
        *,
        numeric_column: str,
        categorical_column: str,
        numeric: pd.Series,
        categorical: pd.Series,
    ) -> MixedAssociationPair | None:
        """Analyze one numeric-categorical pair."""

        frame = pd.DataFrame(
            {
                "numeric": pd.to_numeric(
                    numeric,
                    errors="coerce",
                ),
                "category": categorical,
            }
        )

        if self.options.include_missing_as_category:
            frame["category"] = (
                frame["category"].fillna(
                    "__MISSING__"
                )
            )

            frame = frame.dropna(
                subset=["numeric"]
            )
        else:
            frame = frame.dropna()

        frame = frame[
            np.isfinite(
                frame["numeric"].astype(float)
            )
        ]

        if len(frame) < self.options.minimum_valid_pairs:
            return None

        category_count = int(
            frame["category"].nunique()
        )

        if (
            category_count < 2
            or category_count
            > self.options.maximum_categories
        ):
            return None

        group_counts = frame.groupby(
            "category",
            dropna=False,
        ).size()

        valid_categories = group_counts[
            group_counts
            >= self.options.minimum_group_count
        ].index

        frame = frame[
            frame["category"].isin(
                valid_categories
            )
        ]

        category_count = int(
            frame["category"].nunique()
        )

        if category_count < 2:
            return None

        overall_mean = float(
            frame["numeric"].mean()
        )

        total_sum_squares = float(
            (
                (
                    frame["numeric"]
                    - overall_mean
                )
                ** 2
            ).sum()
        )

        between_sum_squares = 0.0

        group_statistics: dict[
            str,
            dict[str, float | int | None],
        ] = {}

        for category, group in frame.groupby(
            "category",
            dropna=False,
        ):
            group_values = group["numeric"]

            group_mean = float(
                group_values.mean()
            )

            between_sum_squares += (
                len(group_values)
                * (group_mean - overall_mean) ** 2
            )

            group_statistics[str(category)] = {
                "count": len(group_values),
                "mean": group_mean,
                "median": float(
                    group_values.median()
                ),
                "standard_deviation": (
                    float(
                        group_values.std(ddof=0)
                    )
                    if len(group_values) > 1
                    else 0.0
                ),
                "minimum": float(
                    group_values.min()
                ),
                "maximum": float(
                    group_values.max()
                ),
            }

        eta_squared = (
            between_sum_squares
            / total_sum_squares
            if total_sum_squares > 0
            else None
        )

        return MixedAssociationPair(
            numeric_column=numeric_column,
            categorical_column=categorical_column,
            valid_pair_count=len(frame),
            category_count=category_count,
            eta_squared=eta_squared,
            between_group_variance=(
                between_sum_squares
            ),
            total_variance=total_sum_squares,
            group_statistics=group_statistics,
            strong_association=bool(
                eta_squared is not None
                and eta_squared
                >= self.options
                .strong_eta_squared_threshold
            ),
        )
