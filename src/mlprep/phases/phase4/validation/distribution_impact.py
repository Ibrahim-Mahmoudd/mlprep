"""Imputation distribution-impact validation for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)


class ColumnDistributionImpact(BaseModel):
    """Distribution impact for one transformed feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    partition: PartitionName

    row_count: int = Field(
        default=0,
        ge=0,
    )

    missing_before: int = Field(
        default=0,
        ge=0,
    )

    missing_after: int = Field(
        default=0,
        ge=0,
    )

    replaced_count: int = Field(
        default=0,
        ge=0,
    )

    replaced_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    mean_before: float | None = None
    mean_after: float | None = None
    mean_relative_change: float | None = None

    median_before: float | None = None
    median_after: float | None = None
    median_relative_change: float | None = None

    dominant_value_percentage_before: float | None = None
    dominant_value_percentage_after: float | None = None
    dominant_value_percentage_increase: float | None = None

    excessive_replacement: bool = False
    excessive_numeric_shift: bool = False
    excessive_concentration: bool = False

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class DistributionImpactResult(BaseModel):
    """Complete imputation distribution-impact output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    impacts: dict[
        str,
        ColumnDistributionImpact,
    ] = Field(default_factory=dict)

    high_impact_columns: list[str] = Field(
        default_factory=list
    )

    excessive_replacement_columns: list[str] = Field(
        default_factory=list
    )

    excessive_concentration_columns: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class DistributionImpactOptions:
    """Options controlling imputation-impact validation."""

    enabled: bool = True

    analyze_training_only: bool = True

    maximum_replaced_percentage: float = 50.0

    maximum_mean_relative_change: float = 0.50

    maximum_median_relative_change: float = 0.50

    maximum_dominant_value_percentage_increase: float = 30.0

    block_excessive_replacement: bool = False
    block_excessive_numeric_shift: bool = False
    block_excessive_concentration: bool = False

    def __post_init__(self) -> None:
        """Validate distribution-impact options."""

        if not (
            0.0
            <= self.maximum_replaced_percentage
            <= 100.0
        ):
            raise ValueError(
                "maximum_replaced_percentage must be "
                "between zero and one hundred."
            )

        if self.maximum_mean_relative_change < 0.0:
            raise ValueError(
                "maximum_mean_relative_change cannot be negative."
            )

        if self.maximum_median_relative_change < 0.0:
            raise ValueError(
                "maximum_median_relative_change cannot be negative."
            )

        if not (
            0.0
            <= self.maximum_dominant_value_percentage_increase
            <= 100.0
        ):
            raise ValueError(
                "maximum_dominant_value_percentage_increase "
                "must be between zero and one hundred."
            )


class DistributionImpactValidator:
    """Measure the statistical impact of imputation."""

    component_name = "distribution_impact"

    def __init__(
        self,
        options: (
            DistributionImpactOptions | None
        ) = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or DistributionImpactOptions()
        )

    def validate(
        self,
        *,
        source: DatasetPartition,
        transformed: DatasetPartition,
    ) -> DistributionImpactResult:
        """Measure imputation impact for one partition."""

        if not self.options.enabled:
            return DistributionImpactResult(
                metadata={
                    "enabled": False,
                }
            )

        common_columns = [
            column
            for column in source.X.columns
            if column in transformed.X.columns
        ]

        impacts: dict[
            str,
            ColumnDistributionImpact,
        ] = {}

        for column in common_columns:
            impact = self._analyze_column(
                column=str(column),
                partition=source.name,
                source=source.X[column],
                transformed=(
                    transformed.X[column]
                ),
            )

            impacts[str(column)] = impact

        high_impact_columns = [
            column
            for column, impact
            in impacts.items()
            if (
                impact.excessive_replacement
                or impact.excessive_numeric_shift
                or impact.excessive_concentration
            )
        ]

        excessive_replacement_columns = [
            column
            for column, impact
            in impacts.items()
            if impact.excessive_replacement
        ]

        excessive_concentration_columns = [
            column
            for column, impact
            in impacts.items()
            if impact.excessive_concentration
        ]

        blocking_impact = any(
            (
                impact.excessive_replacement
                and self.options
                .block_excessive_replacement
            )
            or (
                impact.excessive_numeric_shift
                and self.options
                .block_excessive_numeric_shift
            )
            or (
                impact.excessive_concentration
                and self.options
                .block_excessive_concentration
            )
            for impact in impacts.values()
        )

        return DistributionImpactResult(
            valid=not blocking_impact,
            impacts=impacts,
            high_impact_columns=(
                high_impact_columns
            ),
            excessive_replacement_columns=(
                excessive_replacement_columns
            ),
            excessive_concentration_columns=(
                excessive_concentration_columns
            ),
            metadata={
                "enabled": True,
                "partition": source.name.value,
                "blocking_impact": (
                    blocking_impact
                ),
            },
        )

    def _analyze_column(
        self,
        *,
        column: str,
        partition: PartitionName,
        source: pd.Series,
        transformed: pd.Series,
    ) -> ColumnDistributionImpact:
        """Analyze imputation impact for one feature."""

        row_count = len(source)

        missing_before = int(
            source.isna().sum()
        )

        missing_after = int(
            transformed.isna().sum()
        )

        replaced_count = max(
            0,
            missing_before - missing_after,
        )

        replaced_percentage = (
            replaced_count
            / row_count
            * 100.0
            if row_count > 0
            else 0.0
        )

        source_numeric = pd.to_numeric(
            source,
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        transformed_numeric = pd.to_numeric(
            transformed,
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        numeric_analysis_available = bool(
            source_numeric.notna().any()
            and transformed_numeric
            .notna()
            .any()
        )

        mean_before = None
        mean_after = None
        mean_relative_change = None

        median_before = None
        median_after = None
        median_relative_change = None

        if numeric_analysis_available:
            mean_before = float(
                source_numeric.mean()
            )

            mean_after = float(
                transformed_numeric.mean()
            )

            mean_relative_change = (
                self._relative_change(
                    mean_before,
                    mean_after,
                )
            )

            median_before = float(
                source_numeric.median()
            )

            median_after = float(
                transformed_numeric.median()
            )

            median_relative_change = (
                self._relative_change(
                    median_before,
                    median_after,
                )
            )

        dominant_before = (
            self._dominant_percentage(
                source.dropna()
            )
        )

        dominant_after = (
            self._dominant_percentage(
                transformed.dropna()
            )
        )

        concentration_increase = (
            dominant_after - dominant_before
            if (
                dominant_before is not None
                and dominant_after is not None
            )
            else None
        )

        excessive_replacement = bool(
            replaced_percentage
            > self.options
            .maximum_replaced_percentage
        )

        excessive_numeric_shift = bool(
            (
                mean_relative_change is not None
                and mean_relative_change
                > self.options
                .maximum_mean_relative_change
            )
            or (
                median_relative_change is not None
                and median_relative_change
                > self.options
                .maximum_median_relative_change
            )
        )

        excessive_concentration = bool(
            concentration_increase is not None
            and concentration_increase
            > self.options
            .maximum_dominant_value_percentage_increase
        )

        return ColumnDistributionImpact(
            column=column,
            partition=partition,
            row_count=row_count,
            missing_before=missing_before,
            missing_after=missing_after,
            replaced_count=replaced_count,
            replaced_percentage=(
                replaced_percentage
            ),
            mean_before=mean_before,
            mean_after=mean_after,
            mean_relative_change=(
                mean_relative_change
            ),
            median_before=median_before,
            median_after=median_after,
            median_relative_change=(
                median_relative_change
            ),
            dominant_value_percentage_before=(
                dominant_before
            ),
            dominant_value_percentage_after=(
                dominant_after
            ),
            dominant_value_percentage_increase=(
                concentration_increase
            ),
            excessive_replacement=(
                excessive_replacement
            ),
            excessive_numeric_shift=(
                excessive_numeric_shift
            ),
            excessive_concentration=(
                excessive_concentration
            ),
        )

    @staticmethod
    def _relative_change(
        before: float,
        after: float,
    ) -> float:
        """Calculate a stable relative change."""

        denominator = max(
            abs(before),
            abs(after),
            1e-12,
        )

        return abs(
            after - before
        ) / denominator

    @staticmethod
    def _dominant_percentage(
        series: pd.Series,
    ) -> float | None:
        """Return the dominant-value percentage."""

        if series.empty:
            return None

        counts = series.value_counts(
            normalize=True,
            dropna=False,
        )

        if counts.empty:
            return None

        return float(
            counts.iloc[0]
            * 100.0
        )