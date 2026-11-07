"""Missingness profiling for Phase 3."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)


class ColumnMissingnessProfile(BaseModel):
    """Missingness information for one feature."""

    model_config = ConfigDict(
        extra="forbid",
    )

    column: str

    missing_count: int = Field(ge=0)

    missing_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    completely_missing: bool = False
    has_missing_values: bool = False


class MissingnessPattern(BaseModel):
    """One observed row-level missingness pattern."""

    model_config = ConfigDict(
        extra="forbid",
    )

    pattern_hash: str
    missing_columns: list[str]

    row_count: int = Field(ge=0)

    row_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )


class MissingnessProfile(BaseModel):
    """Complete training missingness profile."""

    model_config = ConfigDict(
        extra="forbid",
    )

    total_missing_cells: int = Field(ge=0)

    total_missing_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    rows_with_missing_values: int = Field(ge=0)

    rows_with_missing_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    maximum_missing_values_per_row: int = Field(
        ge=0
    )

    average_missing_values_per_row: float = Field(
        ge=0.0
    )

    columns: dict[
        str,
        ColumnMissingnessProfile,
    ] = Field(default_factory=dict)

    patterns: list[
        MissingnessPattern
    ] = Field(default_factory=list)

    pairwise_missingness_correlation: dict[
        str,
        dict[str, float | None],
    ] = Field(default_factory=dict)

    validation_missingness_difference: dict[
        str,
        float,
    ] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class MissingnessProfilerOptions:
    """Options controlling missingness profiling."""

    enabled: bool = True

    maximum_patterns: int = 20

    calculate_pairwise_correlation: bool = True

    compare_validation: bool = True

    minimum_validation_difference: float = 0.0

    def __post_init__(self) -> None:
        """Validate missingness-profiler options."""

        if self.maximum_patterns < 0:
            raise ValueError(
                "maximum_patterns cannot be negative."
            )

        if self.minimum_validation_difference < 0.0:
            raise ValueError(
                "minimum_validation_difference cannot "
                "be negative."
            )


class MissingnessProfiler:
    """Analyze missingness in training and validation data."""

    component_name = "missingness_profile"

    def __init__(
        self,
        options: (
            MissingnessProfilerOptions | None
        ) = None,
    ) -> None:
        """Initialize the missingness profiler."""

        self.options = (
            options
            or MissingnessProfilerOptions()
        )

    def profile(
        self,
        *,
        contract: EDAContract,
    ) -> MissingnessProfile:
        """Build the complete missingness profile."""

        frame = contract.training_features()

        if not self.options.enabled:
            return MissingnessProfile(
                total_missing_cells=0,
                total_missing_percentage=0.0,
                rows_with_missing_values=0,
                rows_with_missing_percentage=0.0,
                maximum_missing_values_per_row=0,
                average_missing_values_per_row=0.0,
            )

        missing_mask = frame.isna()

        row_count = len(frame)
        cell_count = frame.shape[0] * frame.shape[1]

        total_missing = int(
            missing_mask.sum().sum()
        )

        row_missing_counts = missing_mask.sum(
            axis=1
        )

        rows_with_missing = int(
            (row_missing_counts > 0).sum()
        )

        column_profiles = {
            str(column): (
                ColumnMissingnessProfile(
                    column=str(column),
                    missing_count=int(
                        missing_mask[
                            column
                        ].sum()
                    ),
                    missing_percentage=(
                        float(
                            missing_mask[
                                column
                            ].mean()
                        )
                        * 100.0
                        if row_count > 0
                        else 0.0
                    ),
                    completely_missing=bool(
                        missing_mask[
                            column
                        ].all()
                    ),
                    has_missing_values=bool(
                        missing_mask[
                            column
                        ].any()
                    ),
                )
            )
            for column in frame.columns
        }

        patterns = self._patterns(
            missing_mask
        )

        correlations: dict[
            str,
            dict[str, float | None],
        ] = {}

        if (
            self.options
            .calculate_pairwise_correlation
            and len(frame.columns) > 1
        ):
            correlations = self._correlations(
                missing_mask
            )

        validation_difference = (
            self._validation_difference(
                contract=contract,
                training_mask=missing_mask,
            )
        )

        return MissingnessProfile(
            total_missing_cells=total_missing,
            total_missing_percentage=(
                total_missing / cell_count * 100.0
                if cell_count > 0
                else 0.0
            ),
            rows_with_missing_values=(
                rows_with_missing
            ),
            rows_with_missing_percentage=(
                rows_with_missing
                / row_count
                * 100.0
                if row_count > 0
                else 0.0
            ),
            maximum_missing_values_per_row=(
                int(row_missing_counts.max())
                if not row_missing_counts.empty
                else 0
            ),
            average_missing_values_per_row=(
                float(row_missing_counts.mean())
                if not row_missing_counts.empty
                else 0.0
            ),
            columns=column_profiles,
            patterns=patterns,
            pairwise_missingness_correlation=(
                correlations
            ),
            validation_missingness_difference=(
                validation_difference
            ),
        )

    def _patterns(
        self,
        missing_mask: pd.DataFrame,
    ) -> list[object]:
        """Build common row-level missingness patterns."""

        if (
            missing_mask.empty
            or self.options.maximum_patterns == 0
        ):
            return []

        pattern_series = missing_mask.apply(
            lambda row: tuple(
                column
                for column, missing
                in row.items()
                if bool(missing)
            ),
            axis=1,
        )

        counts = pattern_series.value_counts().head(
            self.options.maximum_patterns
        )

        total_rows = len(missing_mask)

        return [
            MissingnessPattern(
                pattern_hash=hashlib.sha256(
                    repr(pattern).encode(
                        "utf-8"
                    )
                ).hexdigest(),
                missing_columns=list(pattern),
                row_count=int(count),
                row_percentage=(
                    int(count)
                    / total_rows
                    * 100.0
                    if total_rows > 0
                    else 0.0
                ),
            )
            for pattern, count
            in counts.items()
        ]

    @staticmethod
    def _correlations(
        missing_mask: pd.DataFrame,
    ) -> dict[
        str,
        dict[str, float | None],
    ]:
        """Calculate missingness-indicator correlations."""

        correlation = (
            missing_mask
            .astype(int)
            .corr()
        )

        return {
            str(row_column): {
                str(column): (
                    None
                    if pd.isna(value)
                    else float(value)
                )
                for column, value
                in correlation.loc[
                    row_column
                ].items()
            }
            for row_column in correlation.index
        }

    def _validation_difference(
        self,
        *,
        contract: EDAContract,
        training_mask: pd.DataFrame,
    ) -> dict[str, float]:
        """Compare validation and training missingness."""

        if not (
            self.options.compare_validation
            and contract.validation is not None
        ):
            return {}

        validation_frame = (
            contract.validation_features()
        )

        if validation_frame is None:
            return {}

        differences: dict[str, float] = {}

        for column in training_mask.columns:
            if column not in validation_frame.columns:
                continue

            training_ratio = float(
                training_mask[column].mean()
            )

            validation_ratio = float(
                validation_frame[
                    column
                ].isna().mean()
            )

            difference = (
                validation_ratio
                - training_ratio
            )

            if (
                abs(difference)
                < self.options
                .minimum_validation_difference
            ):
                continue

            differences[
                str(column)
            ] = difference

        return differences
