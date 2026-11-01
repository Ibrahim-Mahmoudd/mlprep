"""Regression-target distribution validation for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    SplitFinding,
)


@dataclass(slots=True)
class TargetDistributionValidationOptions:
    """Options controlling regression distribution validation."""

    enabled: bool = True

    mean_relative_tolerance: float = 0.25
    median_relative_tolerance: float = 0.25
    standard_deviation_relative_tolerance: float = 0.40

    quantiles: tuple[float, ...] = (
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
    )

    quantile_relative_tolerance: float = 0.35

    require_numeric_target: bool = True
    block_nonfinite_values: bool = True

    def __post_init__(self) -> None:
        """Validate target-distribution options."""

        tolerance_values = [
            self.mean_relative_tolerance,
            self.median_relative_tolerance,
            (
                self
                .standard_deviation_relative_tolerance
            ),
            self.quantile_relative_tolerance,
        ]

        if any(
            value < 0.0
            for value in tolerance_values
        ):
            raise ValueError(
                "Distribution tolerances cannot be negative."
            )

        if any(
            not 0.0 <= quantile <= 1.0
            for quantile in self.quantiles
        ):
            raise ValueError(
                "Every quantile must be between zero and one."
            )


class TargetDistributionValidator:
    """Validate regression targets across partitions."""

    component_name = "target_distribution"

    def __init__(
        self,
        options: (
            TargetDistributionValidationOptions | None
        ) = None,
    ) -> None:
        """Initialize the target-distribution validator."""

        self.options = (
            options
            or TargetDistributionValidationOptions()
        )

    def validate(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
    ) -> list[object]:
        """Validate numeric target distributions."""

        if not self.options.enabled:
            return []

        findings: list[SplitFinding] = []

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        if (
            self.options.require_numeric_target
            and not pd.api.types.is_numeric_dtype(
                train.y.dtype
            )
        ):
            return [
                SplitFinding(
                    code="P2-TARGET-DIST-001",
                    category=(
                        SplitFindingCategory
                        .TARGET_DISTRIBUTION
                    ),
                    severity=SplitFindingSeverity.CRITICAL,
                    component=self.component_name,
                    message=(
                        "Regression target distribution "
                        "validation requires a numeric target."
                    ),
                    recommended_action=(
                        SplitFindingAction.RAISE
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            ]

        for name, partition in partitions.items():
            missing_count = int(
                partition.y.isna().sum()
            )

            if missing_count > 0:
                findings.append(
                    SplitFinding(
                        code="P2-TARGET-DIST-002",
                        category=(
                            SplitFindingCategory
                            .TARGET_DISTRIBUTION
                        ),
                        severity=SplitFindingSeverity.ERROR,
                        component=self.component_name,
                        message=(
                            f"Partition {name.value!r} contains "
                            "missing target values."
                        ),
                        partitions=[name],
                        affected_row_count=missing_count,
                        recommended_action=(
                            SplitFindingAction.RESPLIT
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=True,
                    )
                )

            nonfinite_count = int(
                partition.y
                .dropna()
                .map(
                    lambda value: not isfinite(
                        float(value)
                    )
                )
                .sum()
            )

            if (
                nonfinite_count > 0
                and self.options.block_nonfinite_values
            ):
                findings.append(
                    SplitFinding(
                        code="P2-TARGET-DIST-003",
                        category=(
                            SplitFindingCategory
                            .TARGET_DISTRIBUTION
                        ),
                        severity=SplitFindingSeverity.ERROR,
                        component=self.component_name,
                        message=(
                            f"Partition {name.value!r} contains "
                            "nonfinite target values."
                        ),
                        partitions=[name],
                        affected_row_count=nonfinite_count,
                        recommended_action=(
                            SplitFindingAction.RAISE
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=True,
                    )
                )

        reference = train.y.dropna().astype(float)
        reference = reference.loc[
            reference.map(isfinite)
        ]

        if reference.empty:
            return findings

        reference_statistics = self._statistics(
            reference
        )

        for name in (
            PartitionName.VALIDATION,
            PartitionName.TEST,
        ):
            target = partitions[
                name
            ].y.dropna().astype(float)
            target = target.loc[
                target.map(isfinite)
            ]

            if target.empty:
                continue

            statistics = self._statistics(
                target
            )

            deviations = self._deviations(
                reference=reference_statistics,
                candidate=statistics,
            )

            if deviations:
                findings.append(
                    SplitFinding(
                        code="P2-TARGET-DIST-004",
                        category=(
                            SplitFindingCategory
                            .TARGET_DISTRIBUTION
                        ),
                        severity=SplitFindingSeverity.WARNING,
                        component=self.component_name,
                        message=(
                            f"Partition {name.value!r} target "
                            "distribution differs from training."
                        ),
                        partitions=[
                            PartitionName.TRAIN,
                            name,
                        ],
                        recommended_action=(
                            SplitFindingAction.WARN
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=False,
                        details={
                            "deviations": deviations,
                            "training_statistics": (
                                reference_statistics
                            ),
                            "partition_statistics": (
                                statistics
                            ),
                        },
                    )
                )

        return findings

    def _statistics(
        self,
        target: pd.Series,
    ) -> dict[str, float]:
        """Calculate target distribution statistics."""

        statistics = {
            "mean": float(target.mean()),
            "median": float(target.median()),
            "standard_deviation": float(
                target.std(ddof=0)
            ),
        }

        for quantile in self.options.quantiles:
            statistics[
                f"quantile_{quantile}"
            ] = float(
                target.quantile(quantile)
            )

        return statistics

    def _deviations(
        self,
        *,
        reference: dict[str, float],
        candidate: dict[str, float],
    ) -> dict[str, float]:
        """Return statistics exceeding tolerance."""

        tolerances = {
            "mean": (
                self.options.mean_relative_tolerance
            ),
            "median": (
                self.options.median_relative_tolerance
            ),
            "standard_deviation": (
                self.options
                .standard_deviation_relative_tolerance
            ),
        }

        deviations: dict[str, float] = {}

        for statistic, tolerance in (
            tolerances.items()
        ):
            deviation = self._relative_difference(
                reference[statistic],
                candidate[statistic],
            )

            if deviation > tolerance:
                deviations[statistic] = deviation

        for quantile in self.options.quantiles:
            key = f"quantile_{quantile}"

            deviation = self._relative_difference(
                reference[key],
                candidate[key],
            )

            if (
                deviation
                > self.options
                .quantile_relative_tolerance
            ):
                deviations[key] = deviation

        return deviations

    @staticmethod
    def _relative_difference(
        left: float,
        right: float,
    ) -> float:
        """Calculate a stable relative difference."""

        denominator = max(
            abs(left),
            abs(right),
            1e-12,
        )

        return abs(left - right) / denominator
