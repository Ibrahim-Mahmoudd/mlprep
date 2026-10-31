"""Outlier-treatment distribution-impact validation."""

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
    """Distribution impact for one treated feature."""

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

    changed_value_count: int = Field(
        default=0,
        ge=0,
    )

    changed_value_percentage: float = Field(
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

    standard_deviation_before: float | None = None
    standard_deviation_after: float | None = None
    standard_deviation_relative_change: float | None = None

    boundary_percentage_after: float | None = None

    excessive_treatment: bool = False
    excessive_location_shift: bool = False
    excessive_scale_shift: bool = False
    excessive_boundary_concentration: bool = False


class DistributionImpactResult(BaseModel):
    """Complete treatment-impact validation output."""

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

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class DistributionImpactOptions:
    """Options controlling distribution-impact validation."""

    enabled: bool = True

    maximum_changed_percentage: float = 20.0

    maximum_mean_relative_change: float = 0.30
    maximum_median_relative_change: float = 0.30

    maximum_standard_deviation_relative_change: float = 0.50

    maximum_boundary_percentage_after: float = 30.0

    block_excessive_treatment: bool = False
    block_excessive_location_shift: bool = False
    block_excessive_scale_shift: bool = False
    block_excessive_boundary_concentration: bool = False

    def __post_init__(self) -> None:
        """Validate distribution-impact options."""

        percentage_values = {
            "maximum_changed_percentage": (
                self.maximum_changed_percentage
            ),
            "maximum_boundary_percentage_after": (
                self.maximum_boundary_percentage_after
            ),
        }

        for name, value in percentage_values.items():
            if not 0.0 <= value <= 100.0:
                raise ValueError(
                    f"{name} must be between zero "
                    "and one hundred."
                )

        relative_values = {
            "maximum_mean_relative_change": (
                self.maximum_mean_relative_change
            ),
            "maximum_median_relative_change": (
                self.maximum_median_relative_change
            ),
            "maximum_standard_deviation_relative_change": (
                self
                .maximum_standard_deviation_relative_change
            ),
        }

        for name, value in relative_values.items():
            if value < 0.0:
                raise ValueError(
                    f"{name} cannot be negative."
                )


class DistributionImpactValidator:
    """Measure the statistical impact of outlier treatment."""

    component_name = "distribution_impact"

    def __init__(
        self,
        options: DistributionImpactOptions | None = None,
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
        fitted_bounds: dict[
            str,
            dict[str, object],
        ],
    ) -> DistributionImpactResult:
        """Measure treatment impact on one partition."""

        if not self.options.enabled:
            return DistributionImpactResult(
                metadata={
                    "enabled": False,
                }
            )

        impacts: dict[
            str,
            ColumnDistributionImpact,
        ] = {}

        for column, state in (
            fitted_bounds.items()
        ):
            if (
                column not in source.X.columns
                or column
                not in transformed.X.columns
            ):
                continue

            impact = self._analyze_column(
                column=column,
                partition=source.name,
                source=source.X[column],
                transformed=(
                    transformed.X[column]
                ),
                lower_bound=(
                    self._optional_float(
                        state.get(
                            "lower_bound"
                        )
                    )
                ),
                upper_bound=(
                    self._optional_float(
                        state.get(
                            "upper_bound"
                        )
                    )
                ),
            )

            impacts[column] = impact

        high_impact_columns = [
            column
            for column, impact
            in impacts.items()
            if (
                impact.excessive_treatment
                or impact.excessive_location_shift
                or impact.excessive_scale_shift
                or impact
                .excessive_boundary_concentration
            )
        ]

        blocking = any(
            (
                impact.excessive_treatment
                and self.options
                .block_excessive_treatment
            )
            or (
                impact.excessive_location_shift
                and self.options
                .block_excessive_location_shift
            )
            or (
                impact.excessive_scale_shift
                and self.options
                .block_excessive_scale_shift
            )
            or (
                impact
                .excessive_boundary_concentration
                and self.options
                .block_excessive_boundary_concentration
            )
            for impact in impacts.values()
        )

        return DistributionImpactResult(
            valid=not blocking,
            impacts=impacts,
            high_impact_columns=(
                high_impact_columns
            ),
            metadata={
                "enabled": True,
                "partition": source.name.value,
                "blocking_impact": blocking,
            },
        )

    def _analyze_column(
        self,
        *,
        column: str,
        partition: PartitionName,
        source: pd.Series,
        transformed: pd.Series,
        lower_bound: float | None,
        upper_bound: float | None,
    ) -> ColumnDistributionImpact:
        """Analyze one feature before and after treatment."""

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

        changed_mask = (
            source.ne(
                transformed
            )
            .fillna(False)
        )

        both_missing = (
            source.isna()
            & transformed.isna()
        )

        changed_mask &= (
            ~both_missing
        )

        changed_count = int(
            changed_mask.sum()
        )

        row_count = len(
            source
        )

        changed_percentage = (
            changed_count
            / row_count
            * 100.0
            if row_count > 0
            else 0.0
        )

        mean_before = self._safe_statistic(
            source_numeric.mean()
        )

        mean_after = self._safe_statistic(
            transformed_numeric.mean()
        )

        median_before = self._safe_statistic(
            source_numeric.median()
        )

        median_after = self._safe_statistic(
            transformed_numeric.median()
        )

        std_before = self._safe_statistic(
            source_numeric.std()
        )

        std_after = self._safe_statistic(
            transformed_numeric.std()
        )

        mean_change = self._relative_change(
            mean_before,
            mean_after,
        )

        median_change = self._relative_change(
            median_before,
            median_after,
        )

        std_change = self._relative_change(
            std_before,
            std_after,
        )

        boundary_percentage = (
            self._boundary_percentage(
                series=transformed_numeric,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        )

        excessive_treatment = bool(
            changed_percentage
            > self.options
            .maximum_changed_percentage
        )

        excessive_location_shift = bool(
            (
                mean_change is not None
                and mean_change
                > self.options
                .maximum_mean_relative_change
            )
            or (
                median_change is not None
                and median_change
                > self.options
                .maximum_median_relative_change
            )
        )

        excessive_scale_shift = bool(
            std_change is not None
            and std_change
            > self.options
            .maximum_standard_deviation_relative_change
        )

        excessive_boundary_concentration = bool(
            boundary_percentage is not None
            and boundary_percentage
            > self.options
            .maximum_boundary_percentage_after
        )

        return ColumnDistributionImpact(
            column=column,
            partition=partition,
            row_count=row_count,
            changed_value_count=(
                changed_count
            ),
            changed_value_percentage=(
                changed_percentage
            ),
            mean_before=mean_before,
            mean_after=mean_after,
            mean_relative_change=(
                mean_change
            ),
            median_before=median_before,
            median_after=median_after,
            median_relative_change=(
                median_change
            ),
            standard_deviation_before=(
                std_before
            ),
            standard_deviation_after=(
                std_after
            ),
            standard_deviation_relative_change=(
                std_change
            ),
            boundary_percentage_after=(
                boundary_percentage
            ),
            excessive_treatment=(
                excessive_treatment
            ),
            excessive_location_shift=(
                excessive_location_shift
            ),
            excessive_scale_shift=(
                excessive_scale_shift
            ),
            excessive_boundary_concentration=(
                excessive_boundary_concentration
            ),
        )

    @staticmethod
    def _safe_statistic(
        value: object,
    ) -> float | None:
        """Convert a finite statistic to float."""

        try:
            converted = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

        if not np.isfinite(
            converted
        ):
            return None

        return converted

    @staticmethod
    def _relative_change(
        before: float | None,
        after: float | None,
    ) -> float | None:
        """Calculate a stable relative change."""

        if (
            before is None
            or after is None
        ):
            return None

        denominator = max(
            abs(before),
            abs(after),
            1e-12,
        )

        return abs(
            after - before
        ) / denominator

    @staticmethod
    def _boundary_percentage(
        *,
        series: pd.Series,
        lower_bound: float | None,
        upper_bound: float | None,
    ) -> float | None:
        """Calculate concentration at fitted boundaries."""

        valid = series.dropna()

        if valid.empty:
            return None

        boundary_mask = pd.Series(
            False,
            index=valid.index,
            dtype=bool,
        )

        if lower_bound is not None:
            boundary_mask |= (
                valid == lower_bound
            )

        if upper_bound is not None:
            boundary_mask |= (
                valid == upper_bound
            )

        return float(
            boundary_mask.mean()
            * 100.0
        )

    @staticmethod
    def _optional_float(
        value: object,
    ) -> float | None:
        """Convert an optional numeric value."""

        if value is None:
            return None

        return float(value)