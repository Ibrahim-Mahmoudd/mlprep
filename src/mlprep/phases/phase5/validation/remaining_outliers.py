"""Remaining-outlier validation for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase5.contract.treatment_plan import (
    TreatmentKind,
    TreatmentPlan,
)
from mlprep.phases.phase5.detection.detector_engine import (
    DetectorEngine,
)


class RemainingOutlierIssue(BaseModel):
    """Remaining outliers for one feature and partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    partition: PartitionName

    treatment: str

    row_count: int = Field(
        default=0,
        ge=0,
    )

    remaining_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    remaining_outlier_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    nonfinite_count: int = Field(
        default=0,
        ge=0,
    )

    allowed: bool = False
    blocks_readiness: bool = True

    reason: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class RemainingOutliersResult(BaseModel):
    """Complete remaining-outlier validation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    issues: list[
        RemainingOutlierIssue
    ] = Field(default_factory=list)

    remaining_outlier_columns: dict[
        PartitionName,
        list[str],
    ] = Field(default_factory=dict)

    blocking_columns: dict[
        PartitionName,
        list[str],
    ] = Field(default_factory=dict)

    nonfinite_columns: dict[
        PartitionName,
        list[str],
    ] = Field(default_factory=dict)

    total_remaining_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    total_nonfinite_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class RemainingOutliersOptions:
    """Options controlling remaining-outlier validation."""

    enabled: bool = True

    allow_report_only_outliers: bool = True

    allowed_columns: list[str] = field(
        default_factory=list
    )

    maximum_remaining_percentage_by_column: dict[
        str,
        float,
    ] = field(default_factory=dict)

    maximum_total_remaining_count: int | None = None

    prohibit_nonfinite_values: bool = True

    validate_training: bool = True
    validate_validation: bool = True
    validate_test: bool = True

    def __post_init__(self) -> None:
        """Validate remaining-outlier options."""

        if (
            self.maximum_total_remaining_count
            is not None
            and self.maximum_total_remaining_count < 0
        ):
            raise ValueError(
                "maximum_total_remaining_count cannot "
                "be negative."
            )

        for column, percentage in (
            self.maximum_remaining_percentage_by_column.items()
        ):
            if not 0.0 <= percentage <= 100.0:
                raise ValueError(
                    "Maximum remaining-outlier percentages "
                    "must be between zero and one hundred. "
                    f"Invalid feature: {column!r}"
                )


class RemainingOutliersValidator:
    """Validate outliers remaining after treatment."""

    component_name = "remaining_outliers"

    def __init__(
        self,
        options: RemainingOutliersOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or RemainingOutliersOptions()
        )

    def validate(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        plan: TreatmentPlan,
        detector_engine: DetectorEngine,
    ) -> RemainingOutliersResult:
        """Validate remaining outliers in every partition."""

        if not self.options.enabled:
            return RemainingOutliersResult(
                metadata={
                    "enabled": False,
                }
            )

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        validation_policy = {
            PartitionName.TRAIN: (
                self.options.validate_training
            ),
            PartitionName.VALIDATION: (
                self.options.validate_validation
            ),
            PartitionName.TEST: (
                self.options.validate_test
            ),
        }

        allowed_columns = set(
            self.options.allowed_columns
        )

        issues: list[
            RemainingOutlierIssue
        ] = []

        remaining_columns: dict[
            PartitionName,
            list[str],
        ] = {}

        blocking_columns: dict[
            PartitionName,
            list[str],
        ] = {}

        nonfinite_columns: dict[
            PartitionName,
            list[str],
        ] = {}

        total_remaining = 0
        total_nonfinite = 0

        for partition_name, partition in (
            partitions.items()
        ):
            if not validation_policy[
                partition_name
            ]:
                continue

            detection = detector_engine.detect(
                frame=partition.X,
                partition=partition_name,
            )

            partition_remaining: list[str] = []
            partition_blocking: list[str] = []
            partition_nonfinite: list[str] = []

            for column, entry in plan.entries.items():
                if column not in partition.X.columns:
                    continue

                if column not in detection.masks:
                    continue

                mask = detection.masks[
                    column
                ]

                remaining_count = int(
                    mask.sum()
                )

                row_count = len(
                    partition.X
                )

                remaining_percentage = (
                    remaining_count
                    / row_count
                    * 100.0
                    if row_count > 0
                    else 0.0
                )

                nonfinite_count = (
                    self._nonfinite_count(
                        partition.X[column]
                    )
                )

                total_remaining += (
                    remaining_count
                )

                total_nonfinite += (
                    nonfinite_count
                )

                if remaining_count > 0:
                    partition_remaining.append(
                        column
                    )

                if nonfinite_count > 0:
                    partition_nonfinite.append(
                        column
                    )

                (
                    allowed,
                    blocks_readiness,
                    reason,
                ) = self._column_policy(
                    column=column,
                    treatment=entry.treatment,
                    remaining_percentage=(
                        remaining_percentage
                    ),
                    nonfinite_count=(
                        nonfinite_count
                    ),
                    allowed_columns=(
                        allowed_columns
                    ),
                )

                if blocks_readiness:
                    partition_blocking.append(
                        column
                    )

                if (
                    remaining_count > 0
                    or nonfinite_count > 0
                ):
                    issues.append(
                        RemainingOutlierIssue(
                            column=column,
                            partition=partition_name,
                            treatment=(
                                entry.treatment.value
                            ),
                            row_count=row_count,
                            remaining_outlier_count=(
                                remaining_count
                            ),
                            remaining_outlier_percentage=(
                                remaining_percentage
                            ),
                            nonfinite_count=(
                                nonfinite_count
                            ),
                            allowed=allowed,
                            blocks_readiness=(
                                blocks_readiness
                            ),
                            reason=reason,
                        )
                    )

            remaining_columns[
                partition_name
            ] = partition_remaining

            blocking_columns[
                partition_name
            ] = partition_blocking

            nonfinite_columns[
                partition_name
            ] = partition_nonfinite

        total_limit_exceeded = bool(
            self.options.maximum_total_remaining_count
            is not None
            and total_remaining
            > self.options.maximum_total_remaining_count
        )

        valid = bool(
            not any(
                issue.blocks_readiness
                for issue in issues
            )
            and not total_limit_exceeded
        )

        return RemainingOutliersResult(
            valid=valid,
            issues=issues,
            remaining_outlier_columns=(
                remaining_columns
            ),
            blocking_columns=(
                blocking_columns
            ),
            nonfinite_columns=(
                nonfinite_columns
            ),
            total_remaining_outlier_count=(
                total_remaining
            ),
            total_nonfinite_count=(
                total_nonfinite
            ),
            metadata={
                "enabled": True,
                "total_limit_exceeded": (
                    total_limit_exceeded
                ),
            },
        )

    def _column_policy(
        self,
        *,
        column: str,
        treatment: TreatmentKind,
        remaining_percentage: float,
        nonfinite_count: int,
        allowed_columns: set[str],
    ) -> tuple[
        bool,
        bool,
        str,
    ]:
        """Resolve the remaining-outlier policy."""

        if (
            self.options.prohibit_nonfinite_values
            and nonfinite_count > 0
        ):
            return (
                False,
                True,
                "Nonfinite values remain after treatment.",
            )

        configured_maximum = (
            self.options
            .maximum_remaining_percentage_by_column
            .get(column)
        )

        if configured_maximum is not None:
            if (
                remaining_percentage
                <= configured_maximum
            ):
                return (
                    True,
                    False,
                    (
                        "Remaining outliers are within the "
                        "configured feature limit."
                    ),
                )

            return (
                False,
                True,
                (
                    "Remaining outliers exceed the "
                    "configured feature limit."
                ),
            )

        if column in allowed_columns:
            return (
                True,
                False,
                (
                    "Remaining outliers are explicitly "
                    "allowed for this feature."
                ),
            )

        if (
            treatment == TreatmentKind.REPORT_ONLY
            and self.options
            .allow_report_only_outliers
        ):
            return (
                True,
                False,
                (
                    "Remaining outliers are allowed because "
                    "the feature uses report-only treatment."
                ),
            )

        if remaining_percentage <= 0.0:
            return (
                True,
                False,
                "No outliers remain after treatment.",
            )

        return (
            False,
            True,
            (
                "Remaining outliers are not allowed for "
                "this feature."
            ),
        )

    @staticmethod
    def _nonfinite_count(
        series: pd.Series,
    ) -> int:
        """Count infinite numeric values."""

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        mask = (
            numeric.notna()
            & ~np.isfinite(
                numeric.astype(float)
            )
        )

        return int(
            mask.sum()
        )
