"""Categorical feature-association analysis for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
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


class CategoricalAssociationPair(BaseModel):
    """Association between two categorical features."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    left_column: str
    right_column: str

    chi_square: float | None = None
    degrees_of_freedom: int | None = None

    cramers_v: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    valid_pair_count: int = Field(
        default=0,
        ge=0,
    )

    left_unique_count: int = Field(
        default=0,
        ge=0,
    )

    right_unique_count: int = Field(
        default=0,
        ge=0,
    )

    contingency_row_count: int = Field(
        default=0,
        ge=0,
    )

    contingency_column_count: int = Field(
        default=0,
        ge=0,
    )

    minimum_expected_frequency: float | None = None

    low_expected_frequency_cell_count: int = Field(
        default=0,
        ge=0,
    )

    strong_association: bool = False
    perfect_association: bool = False

    skipped: bool = False
    skip_reason: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class CategoricalAssociationResult(BaseModel):
    """Complete categorical-association analysis output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    analyzed_columns: list[str] = Field(
        default_factory=list
    )

    skipped_columns: dict[str, str] = Field(
        default_factory=dict
    )

    pairs: list[CategoricalAssociationPair] = Field(
        default_factory=list
    )

    strong_pairs: list[CategoricalAssociationPair] = Field(
        default_factory=list
    )

    perfect_pairs: list[CategoricalAssociationPair] = Field(
        default_factory=list
    )

    analyzed_pair_count: int = Field(
        default=0,
        ge=0,
    )

    skipped_pair_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class CategoricalAssociationOptions:
    """Options controlling categorical-association analysis."""

    enabled: bool = True

    include_boolean_columns: bool = True

    minimum_valid_pairs: int = 5

    maximum_categories_per_column: int = 100

    strong_cramers_v_threshold: float = 0.70
    perfect_cramers_v_threshold: float = 0.999999

    bias_correction: bool = True

    include_missing_as_category: bool = False

    missing_category_label: str = "__MISSING__"

    minimum_expected_frequency: float = 5.0

    report_low_expected_frequencies: bool = True
    include_contingency_metadata: bool = True

    maximum_pairs: int | None = None

    def __post_init__(self) -> None:
        """Validate categorical-association options."""

        if self.minimum_valid_pairs < 1:
            raise ValueError(
                "minimum_valid_pairs must be positive."
            )

        if self.maximum_categories_per_column < 2:
            raise ValueError(
                "maximum_categories_per_column must be "
                "at least two."
            )

        if not (
            0.0
            <= self.strong_cramers_v_threshold
            <= 1.0
        ):
            raise ValueError(
                "strong_cramers_v_threshold must be "
                "between zero and one."
            )

        if not (
            self.strong_cramers_v_threshold
            <= self.perfect_cramers_v_threshold
            <= 1.0
        ):
            raise ValueError(
                "perfect_cramers_v_threshold must be at least "
                "the strong association threshold and at most one."
            )

        if self.minimum_expected_frequency < 0.0:
            raise ValueError(
                "minimum_expected_frequency cannot be negative."
            )

        if not self.missing_category_label:
            raise ValueError(
                "missing_category_label cannot be empty."
            )

        if (
            self.maximum_pairs is not None
            and self.maximum_pairs < 1
        ):
            raise ValueError(
                "maximum_pairs must be positive when provided."
            )


class CategoricalAssociationAnalyzer:
    """Analyze relationships between categorical training features."""

    component_name = "categorical_associations"

    def __init__(
        self,
        options: (
            CategoricalAssociationOptions | None
        ) = None,
    ) -> None:
        """Initialize the categorical-association analyzer."""

        self.options = (
            options
            or CategoricalAssociationOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> CategoricalAssociationResult:
        """Analyze categorical feature associations."""

        if not self.options.enabled:
            return CategoricalAssociationResult(
                metadata={
                    "enabled": False,
                }
            )

        frame = contract.training_features()

        selected_columns = self._selected_columns(
            frame=frame,
            columns=columns,
        )

        (
            usable_columns,
            skipped_columns,
        ) = self._filter_columns(
            frame=frame,
            selected_columns=selected_columns,
        )

        pairs: list[
            CategoricalAssociationPair
        ] = []

        strong_pairs: list[
            CategoricalAssociationPair
        ] = []

        perfect_pairs: list[
            CategoricalAssociationPair
        ] = []

        skipped_pair_count = 0
        processed_pair_count = 0
        pair_limit_reached = False

        for left_position, left_column in enumerate(
            usable_columns
        ):
            if pair_limit_reached:
                break

            for right_column in usable_columns[
                left_position + 1:
            ]:
                if (
                    self.options.maximum_pairs
                    is not None
                    and processed_pair_count
                    >= self.options.maximum_pairs
                ):
                    pair_limit_reached = True
                    break

                processed_pair_count += 1

                pair = self._analyze_pair(
                    left_column=left_column,
                    right_column=right_column,
                    left=frame[left_column],
                    right=frame[right_column],
                )

                if pair.skipped:
                    skipped_pair_count += 1
                    continue

                pairs.append(pair)

                if pair.strong_association:
                    strong_pairs.append(pair)

                if pair.perfect_association:
                    perfect_pairs.append(pair)

        pairs.sort(
            key=lambda pair: (
                pair.cramers_v or 0.0
            ),
            reverse=True,
        )

        strong_pairs.sort(
            key=lambda pair: (
                pair.cramers_v or 0.0
            ),
            reverse=True,
        )

        perfect_pairs.sort(
            key=lambda pair: (
                pair.cramers_v or 0.0
            ),
            reverse=True,
        )

        return CategoricalAssociationResult(
            analyzed_columns=usable_columns,
            skipped_columns=skipped_columns,
            pairs=pairs,
            strong_pairs=strong_pairs,
            perfect_pairs=perfect_pairs,
            analyzed_pair_count=len(pairs),
            skipped_pair_count=skipped_pair_count,
            metadata={
                "enabled": True,
                "selected_column_count": len(
                    selected_columns
                ),
                "analyzed_column_count": len(
                    usable_columns
                ),
                "include_boolean_columns": (
                    self.options
                    .include_boolean_columns
                ),
                "include_missing_as_category": (
                    self.options
                    .include_missing_as_category
                ),
                "bias_correction": (
                    self.options.bias_correction
                ),
                "minimum_expected_frequency": (
                    self.options
                    .minimum_expected_frequency
                ),
                "pair_limit_reached": (
                    pair_limit_reached
                ),
            },
        )

    def _selected_columns(
        self,
        *,
        frame: pd.DataFrame,
        columns: AnalysisColumnGroups,
    ) -> list[object]:
        """Return categorical columns selected for analysis."""

        selected_columns = list(
            columns.categorical
        )

        if self.options.include_boolean_columns:
            selected_columns.extend(
                columns.boolean
            )

        return [
            column
            for column in dict.fromkeys(
                selected_columns
            )
            if column in frame.columns
        ]

    def _filter_columns(
        self,
        *,
        frame: pd.DataFrame,
        selected_columns: list[str],
    ) -> tuple[
        list[str],
        dict[str, str],
    ]:
        """Filter unusable categorical columns."""

        usable_columns: list[str] = []

        skipped_columns: dict[
            str,
            str,
        ] = {}

        for column in selected_columns:
            checked_series = self._prepare_series(
                frame[column]
            )

            valid_count = len(
                checked_series
            )

            if (
                valid_count
                < self.options.minimum_valid_pairs
            ):
                skipped_columns[column] = (
                    "The column contains too few valid values."
                )

                continue

            unique_count = int(
                checked_series.nunique(
                    dropna=False
                )
            )

            if unique_count < 2:
                skipped_columns[column] = (
                    "The column is constant or contains fewer "
                    "than two usable categories."
                )

                continue

            if (
                unique_count
                > self.options
                .maximum_categories_per_column
            ):
                skipped_columns[column] = (
                    "The column exceeds the configured maximum "
                    "number of categories."
                )

                continue

            usable_columns.append(column)

        return (
            usable_columns,
            skipped_columns,
        )

    def _analyze_pair(
        self,
        *,
        left_column: str,
        right_column: str,
        left: pd.Series,
        right: pd.Series,
    ) -> CategoricalAssociationPair:
        """Analyze one categorical feature pair."""

        pair_frame = pd.DataFrame(
            {
                "left": left,
                "right": right,
            }
        )

        if self.options.include_missing_as_category:
            pair_frame = (
                pair_frame
                .astype("object")
                .fillna(
                    self.options
                    .missing_category_label
                )
            )
        else:
            pair_frame = pair_frame.dropna(
                subset=[
                    "left",
                    "right",
                ]
            )

        valid_pair_count = len(
            pair_frame
        )

        if (
            valid_pair_count
            < self.options.minimum_valid_pairs
        ):
            return CategoricalAssociationPair(
                left_column=left_column,
                right_column=right_column,
                valid_pair_count=valid_pair_count,
                skipped=True,
                skip_reason=(
                    "The pair contains too few jointly "
                    "valid rows."
                ),
            )

        left_unique_count = int(
            pair_frame["left"].nunique(
                dropna=False
            )
        )

        right_unique_count = int(
            pair_frame["right"].nunique(
                dropna=False
            )
        )

        if (
            left_unique_count < 2
            or right_unique_count < 2
        ):
            return CategoricalAssociationPair(
                left_column=left_column,
                right_column=right_column,
                valid_pair_count=valid_pair_count,
                left_unique_count=left_unique_count,
                right_unique_count=right_unique_count,
                skipped=True,
                skip_reason=(
                    "At least one feature has fewer than "
                    "two usable categories."
                ),
            )

        if (
            left_unique_count
            > self.options
            .maximum_categories_per_column
            or right_unique_count
            > self.options
            .maximum_categories_per_column
        ):
            return CategoricalAssociationPair(
                left_column=left_column,
                right_column=right_column,
                valid_pair_count=valid_pair_count,
                left_unique_count=left_unique_count,
                right_unique_count=right_unique_count,
                skipped=True,
                skip_reason=(
                    "At least one feature exceeds the maximum "
                    "number of categories."
                ),
            )

        contingency = pd.crosstab(
            pair_frame["left"],
            pair_frame["right"],
            dropna=False,
        )

        observed = contingency.to_numpy(
            dtype=float
        )

        total = float(
            observed.sum()
        )

        if total <= 0.0:
            return CategoricalAssociationPair(
                left_column=left_column,
                right_column=right_column,
                valid_pair_count=valid_pair_count,
                left_unique_count=left_unique_count,
                right_unique_count=right_unique_count,
                skipped=True,
                skip_reason=(
                    "The contingency table contains "
                    "no observations."
                ),
            )

        row_totals = observed.sum(
            axis=1,
            keepdims=True,
        )

        column_totals = observed.sum(
            axis=0,
            keepdims=True,
        )

        expected = (
            row_totals
            @ column_totals
        ) / total

        valid_expected_cells = (
            expected > 0.0
        )

        if not bool(
            valid_expected_cells.any()
        ):
            return CategoricalAssociationPair(
                left_column=left_column,
                right_column=right_column,
                valid_pair_count=valid_pair_count,
                left_unique_count=left_unique_count,
                right_unique_count=right_unique_count,
                skipped=True,
                skip_reason=(
                    "The contingency table has no valid "
                    "expected frequencies."
                ),
            )

        observed_values = observed[
            valid_expected_cells
        ]

        expected_values = expected[
            valid_expected_cells
        ]

        chi_square = float(
            (
                (
                    observed_values
                    - expected_values
                )
                ** 2
                / expected_values
            ).sum()
        )

        row_count, column_count = (
            observed.shape
        )

        degrees_of_freedom = (
            (row_count - 1)
            * (column_count - 1)
        )

        cramers_v = self._cramers_v(
            chi_square=chi_square,
            sample_count=int(total),
            row_count=row_count,
            column_count=column_count,
        )

        minimum_expected_frequency = (
            float(
                expected_values.min()
            )
            if expected_values.size > 0
            else None
        )

        low_expected_frequency_count = int(
            (
                expected_values
                < self.options
                .minimum_expected_frequency
            ).sum()
        )

        metadata: dict[str, Any] = {}

        if (
            self.options
            .include_contingency_metadata
        ):
            metadata.update(
                {
                    "contingency_shape": [
                        row_count,
                        column_count,
                    ],
                    "observed_total": int(
                        total
                    ),
                    "nonzero_observed_cell_count": int(
                        (observed > 0).sum()
                    ),
                    "zero_observed_cell_count": int(
                        (observed == 0).sum()
                    ),
                }
            )

        if (
            self.options
            .report_low_expected_frequencies
            and low_expected_frequency_count > 0
        ):
            metadata[
                "low_expected_frequency_warning"
            ] = (
                "One or more contingency cells have "
                "expected frequencies below the configured "
                "threshold."
            )

        strong_association = bool(
            cramers_v is not None
            and cramers_v
            >= self.options
            .strong_cramers_v_threshold
        )

        perfect_association = bool(
            cramers_v is not None
            and cramers_v
            >= self.options
            .perfect_cramers_v_threshold
        )

        return CategoricalAssociationPair(
            left_column=left_column,
            right_column=right_column,
            chi_square=chi_square,
            degrees_of_freedom=(
                degrees_of_freedom
            ),
            cramers_v=cramers_v,
            valid_pair_count=valid_pair_count,
            left_unique_count=left_unique_count,
            right_unique_count=right_unique_count,
            contingency_row_count=row_count,
            contingency_column_count=(
                column_count
            ),
            minimum_expected_frequency=(
                minimum_expected_frequency
            ),
            low_expected_frequency_cell_count=(
                low_expected_frequency_count
            ),
            strong_association=(
                strong_association
            ),
            perfect_association=(
                perfect_association
            ),
            metadata=metadata,
        )

    def _prepare_series(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Prepare a categorical Series for eligibility checks."""

        if self.options.include_missing_as_category:
            return (
                series
                .astype("object")
                .fillna(
                    self.options
                    .missing_category_label
                )
            )

        return series.dropna()

    def _cramers_v(
        self,
        *,
        chi_square: float,
        sample_count: int,
        row_count: int,
        column_count: int,
    ) -> float | None:
        """Calculate Cramér's V with optional bias correction."""

        if sample_count <= 1:
            return None

        phi_squared = (
            chi_square / sample_count
        )

        if self.options.bias_correction:
            correction = (
                (column_count - 1)
                * (row_count - 1)
                / (sample_count - 1)
            )

            numerator = max(
                0.0,
                phi_squared - correction,
            )

            corrected_row_count = (
                row_count
                - (
                    (row_count - 1) ** 2
                    / (sample_count - 1)
                )
            )

            corrected_column_count = (
                column_count
                - (
                    (column_count - 1) ** 2
                    / (sample_count - 1)
                )
            )

            denominator = min(
                corrected_row_count - 1.0,
                corrected_column_count - 1.0,
            )

        else:
            numerator = phi_squared

            denominator = float(
                min(
                    row_count - 1,
                    column_count - 1,
                )
            )

        if denominator <= 0.0:
            return None

        value = sqrt(
            numerator / denominator
        )

        if not np.isfinite(value):
            return None

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )
