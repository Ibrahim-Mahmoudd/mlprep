"""Numeric-output integrity validation for Phase 6."""

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


class NumericIntegrityIssue(BaseModel):
    """Numeric-integrity issue for one output feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    partition: PartitionName

    dtype: str

    row_count: int = Field(
        default=0,
        ge=0,
    )

    missing_count: int = Field(
        default=0,
        ge=0,
    )

    positive_infinity_count: int = Field(
        default=0,
        ge=0,
    )

    negative_infinity_count: int = Field(
        default=0,
        ge=0,
    )

    nonfinite_count: int = Field(
        default=0,
        ge=0,
    )

    nonnumeric: bool = False

    constant_output: bool = False

    blocks_readiness: bool = True

    reason: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class NumericIntegrityResult(BaseModel):
    """Complete numeric-integrity validation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    issues: list[
        NumericIntegrityIssue
    ] = Field(default_factory=list)

    missing_columns: dict[
        PartitionName,
        list[str],
    ] = Field(default_factory=dict)

    nonfinite_columns: dict[
        PartitionName,
        list[str],
    ] = Field(default_factory=dict)

    nonnumeric_columns: dict[
        PartitionName,
        list[str],
    ] = Field(default_factory=dict)

    constant_columns: dict[
        PartitionName,
        list[str],
    ] = Field(default_factory=dict)

    total_missing_count: int = Field(
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
class NumericIntegrityOptions:
    """Options controlling numeric-output validation."""

    enabled: bool = True

    require_all_numeric_output: bool = True

    allow_missing_values: bool = False
    allow_nonfinite_values: bool = False

    allow_constant_features: bool = True

    allowed_missing_columns: list[str] = field(
        default_factory=list
    )

    allowed_nonfinite_columns: list[str] = field(
        default_factory=list
    )

    maximum_missing_percentage_by_column: dict[
        str,
        float,
    ] = field(default_factory=dict)

    maximum_absolute_value: float | None = None

    validate_training: bool = True
    validate_validation: bool = True
    validate_test: bool = True

    def __post_init__(self) -> None:
        """Validate numeric-integrity options."""

        if (
            self.maximum_absolute_value is not None
            and self.maximum_absolute_value <= 0.0
        ):
            raise ValueError(
                "maximum_absolute_value must be positive."
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


class NumericIntegrityValidator:
    """Validate transformed numeric feature matrices."""

    component_name = "numeric_integrity"

    def __init__(
        self,
        options: NumericIntegrityOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or NumericIntegrityOptions()
        )

    def validate(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
    ) -> NumericIntegrityResult:
        """Validate numeric integrity across all partitions."""

        if not self.options.enabled:
            return NumericIntegrityResult(
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

        allowed_missing = set(
            self.options.allowed_missing_columns
        )

        allowed_nonfinite = set(
            self.options.allowed_nonfinite_columns
        )

        issues: list[
            NumericIntegrityIssue
        ] = []

        missing_columns: dict[
            PartitionName,
            list[str],
        ] = {}

        nonfinite_columns: dict[
            PartitionName,
            list[str],
        ] = {}

        nonnumeric_columns: dict[
            PartitionName,
            list[str],
        ] = {}

        constant_columns: dict[
            PartitionName,
            list[str],
        ] = {}

        total_missing_count = 0
        total_nonfinite_count = 0

        for partition_name, partition in (
            partitions.items()
        ):
            if not validation_policy[
                partition_name
            ]:
                continue

            partition_missing: list[str] = []
            partition_nonfinite: list[str] = []
            partition_nonnumeric: list[str] = []
            partition_constant: list[str] = []

            for column in partition.X.columns:
                column_name = str(
                    column
                )

                series = partition.X[
                    column
                ]

                is_numeric = (
                    pd.api.types.is_numeric_dtype(
                        series.dtype
                    )
                )

                if not is_numeric:
                    partition_nonnumeric.append(
                        column_name
                    )

                    blocks_readiness = (
                        self.options
                        .require_all_numeric_output
                    )

                    issues.append(
                        NumericIntegrityIssue(
                            column=column_name,
                            partition=partition_name,
                            dtype=str(
                                series.dtype
                            ),
                            row_count=len(series),
                            nonnumeric=True,
                            blocks_readiness=(
                                blocks_readiness
                            ),
                            reason=(
                                "The output feature is not numeric."
                            ),
                        )
                    )

                    continue

                numeric = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                missing_count = int(
                    numeric.isna().sum()
                )

                positive_infinity_count = int(
                    np.isposinf(
                        numeric.astype(float)
                    ).sum()
                )

                negative_infinity_count = int(
                    np.isneginf(
                        numeric.astype(float)
                    ).sum()
                )

                nonfinite_count = (
                    positive_infinity_count
                    + negative_infinity_count
                )

                total_missing_count += (
                    missing_count
                )

                total_nonfinite_count += (
                    nonfinite_count
                )

                if missing_count > 0:
                    partition_missing.append(
                        column_name
                    )

                if nonfinite_count > 0:
                    partition_nonfinite.append(
                        column_name
                    )

                valid = numeric.replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                ).dropna()

                constant_output = bool(
                    not valid.empty
                    and valid.nunique(
                        dropna=True
                    )
                    <= 1
                )

                if constant_output:
                    partition_constant.append(
                        column_name
                    )

                missing_percentage = (
                    missing_count
                    / len(series)
                    * 100.0
                    if len(series) > 0
                    else 0.0
                )

                reasons: list[str] = []
                blocks_readiness = False

                if missing_count > 0:
                    configured_maximum = (
                        self.options
                        .maximum_missing_percentage_by_column
                        .get(
                            column_name
                        )
                    )

                    if configured_maximum is not None:
                        if (
                            missing_percentage
                            > configured_maximum
                        ):
                            reasons.append(
                                "Missing output values exceed "
                                "the configured feature limit."
                            )

                            blocks_readiness = True

                    elif (
                        not self.options.allow_missing_values
                        and column_name
                        not in allowed_missing
                    ):
                        reasons.append(
                            "Missing output values are not allowed."
                        )

                        blocks_readiness = True

                if (
                    nonfinite_count > 0
                    and not self.options
                    .allow_nonfinite_values
                    and column_name
                    not in allowed_nonfinite
                ):
                    reasons.append(
                        "Nonfinite output values are not allowed."
                    )

                    blocks_readiness = True

                if (
                    constant_output
                    and not self.options
                    .allow_constant_features
                ):
                    reasons.append(
                        "Constant transformed features are not allowed."
                    )

                    blocks_readiness = True

                extreme_count = 0

                if (
                    self.options.maximum_absolute_value
                    is not None
                    and not valid.empty
                ):
                    extreme_count = int(
                        valid.abs().gt(
                            self.options
                            .maximum_absolute_value
                        ).sum()
                    )

                    if extreme_count > 0:
                        reasons.append(
                            "Output values exceed the configured "
                            "absolute-value limit."
                        )

                        blocks_readiness = True

                if (
                    missing_count > 0
                    or nonfinite_count > 0
                    or constant_output
                    or extreme_count > 0
                ):
                    issues.append(
                        NumericIntegrityIssue(
                            column=column_name,
                            partition=partition_name,
                            dtype=str(
                                series.dtype
                            ),
                            row_count=len(series),
                            missing_count=(
                                missing_count
                            ),
                            positive_infinity_count=(
                                positive_infinity_count
                            ),
                            negative_infinity_count=(
                                negative_infinity_count
                            ),
                            nonfinite_count=(
                                nonfinite_count
                            ),
                            nonnumeric=False,
                            constant_output=(
                                constant_output
                            ),
                            blocks_readiness=(
                                blocks_readiness
                            ),
                            reason=(
                                " ".join(
                                    reasons
                                )
                                if reasons
                                else (
                                    "The feature is valid under "
                                    "the configured policy."
                                )
                            ),
                            metadata={
                                "missing_percentage": (
                                    missing_percentage
                                ),
                                "extreme_count": (
                                    extreme_count
                                ),
                            },
                        )
                    )

            missing_columns[
                partition_name
            ] = partition_missing

            nonfinite_columns[
                partition_name
            ] = partition_nonfinite

            nonnumeric_columns[
                partition_name
            ] = partition_nonnumeric

            constant_columns[
                partition_name
            ] = partition_constant

        return NumericIntegrityResult(
            valid=not any(
                issue.blocks_readiness
                for issue in issues
            ),
            issues=issues,
            missing_columns=(
                missing_columns
            ),
            nonfinite_columns=(
                nonfinite_columns
            ),
            nonnumeric_columns=(
                nonnumeric_columns
            ),
            constant_columns=(
                constant_columns
            ),
            total_missing_count=(
                total_missing_count
            ),
            total_nonfinite_count=(
                total_nonfinite_count
            ),
            metadata={
                "enabled": True,
            },
        )
