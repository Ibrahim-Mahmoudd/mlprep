"""Remaining-missingness validation for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)


class RemainingMissingnessIssue(BaseModel):
    """Remaining missingness for one feature and partition."""

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

    missing_count: int = Field(
        default=0,
        ge=0,
    )

    missing_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    allowed: bool = False
    blocks_readiness: bool = True

    reason: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class RemainingMissingnessResult(BaseModel):
    """Complete remaining-missingness validation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    issues: list[
        RemainingMissingnessIssue
    ] = Field(default_factory=list)

    remaining_missing_columns: dict[
        PartitionName,
        list[str],
    ] = Field(default_factory=dict)

    blocking_columns: dict[
        PartitionName,
        list[str],
    ] = Field(default_factory=dict)

    total_remaining_missing_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class RemainingMissingnessOptions:
    """Options controlling remaining-missingness validation."""

    enabled: bool = True

    allow_any_remaining_missingness: bool = False

    allowed_columns: list[str] = field(
        default_factory=list
    )

    report_only_columns: list[str] = field(
        default_factory=list
    )

    maximum_missing_percentage_by_column: dict[
        str,
        float,
    ] = field(default_factory=dict)

    maximum_total_missing_count: int | None = 0

    validate_training: bool = True
    validate_validation: bool = True
    validate_test: bool = True

    def __post_init__(self) -> None:
        """Validate remaining-missingness options."""

        if (
            self.maximum_total_missing_count is not None
            and self.maximum_total_missing_count < 0
        ):
            raise ValueError(
                "maximum_total_missing_count cannot be negative."
            )

        for column, percentage in (
            self.maximum_missing_percentage_by_column.items()
        ):
            if not 0.0 <= percentage <= 100.0:
                raise ValueError(
                    "Maximum missing percentages must be "
                    "between zero and one hundred. "
                    f"Invalid column: {column!r}"
                )


class RemainingMissingnessValidator:
    """Validate unresolved feature missingness."""

    component_name = "remaining_missingness"

    def __init__(
        self,
        options: (
            RemainingMissingnessOptions | None
        ) = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or RemainingMissingnessOptions()
        )

    def validate(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
    ) -> RemainingMissingnessResult:
        """Validate remaining missingness in every partition."""

        if not self.options.enabled:
            return RemainingMissingnessResult(
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

        issues: list[
            RemainingMissingnessIssue
        ] = []

        remaining_columns: dict[
            PartitionName,
            list[str],
        ] = {}

        blocking_columns: dict[
            PartitionName,
            list[str],
        ] = {}

        total_remaining_missing_count = 0

        allowed_columns = set(
            self.options.allowed_columns
        )

        report_only_columns = set(
            self.options.report_only_columns
        )

        for partition_name, partition in (
            partitions.items()
        ):
            if not validation_policy[
                partition_name
            ]:
                continue

            partition_remaining: list[str] = []
            partition_blocking: list[str] = []

            for column in partition.X.columns:
                missing_count = int(
                    partition.X[
                        column
                    ].isna().sum()
                )

                if missing_count == 0:
                    continue

                total_remaining_missing_count += (
                    missing_count
                )

                partition_remaining.append(
                    str(column)
                )

                row_count = len(
                    partition.X
                )

                missing_percentage = (
                    missing_count
                    / row_count
                    * 100.0
                    if row_count > 0
                    else 0.0
                )

                (
                    allowed,
                    blocks_readiness,
                    reason,
                ) = self._column_policy(
                    column=str(column),
                    missing_percentage=(
                        missing_percentage
                    ),
                    allowed_columns=allowed_columns,
                    report_only_columns=(
                        report_only_columns
                    ),
                )

                if blocks_readiness:
                    partition_blocking.append(
                        str(column)
                    )

                issues.append(
                    RemainingMissingnessIssue(
                        column=str(column),
                        partition=partition_name,
                        row_count=row_count,
                        missing_count=missing_count,
                        missing_percentage=(
                            missing_percentage
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

        total_limit_exceeded = bool(
            self.options.maximum_total_missing_count
            is not None
            and total_remaining_missing_count
            > self.options.maximum_total_missing_count
        )

        valid = not any(
            issue.blocks_readiness
            for issue in issues
        )

        if total_limit_exceeded:
            valid = False

        return RemainingMissingnessResult(
            valid=valid,
            issues=issues,
            remaining_missing_columns=(
                remaining_columns
            ),
            blocking_columns=blocking_columns,
            total_remaining_missing_count=(
                total_remaining_missing_count
            ),
            metadata={
                "enabled": True,
                "maximum_total_missing_count": (
                    self.options
                    .maximum_total_missing_count
                ),
                "total_limit_exceeded": (
                    total_limit_exceeded
                ),
            },
        )

    def _column_policy(
        self,
        *,
        column: str,
        missing_percentage: float,
        allowed_columns: set[str],
        report_only_columns: set[str],
    ) -> tuple[
        bool,
        bool,
        str,
    ]:
        """Resolve the remaining-missingness policy."""

        configured_maximum = (
            self.options
            .maximum_missing_percentage_by_column
            .get(column)
        )

        if configured_maximum is not None:
            if (
                missing_percentage
                <= configured_maximum
            ):
                return (
                    True,
                    False,
                    (
                        "Remaining missingness is within the "
                        "configured per-column limit."
                    ),
                )

            return (
                False,
                True,
                (
                    "Remaining missingness exceeds the "
                    "configured per-column limit."
                ),
            )

        if column in report_only_columns:
            return (
                True,
                False,
                (
                    "Remaining missingness is configured "
                    "for reporting only."
                ),
            )

        if column in allowed_columns:
            return (
                True,
                False,
                (
                    "Remaining missingness is explicitly "
                    "allowed for this feature."
                ),
            )

        if self.options.allow_any_remaining_missingness:
            return (
                True,
                False,
                (
                    "Remaining missingness is allowed by "
                    "the global policy."
                ),
            )

        return (
            False,
            True,
            (
                "Remaining missingness is not allowed for "
                "this feature."
            ),
        )