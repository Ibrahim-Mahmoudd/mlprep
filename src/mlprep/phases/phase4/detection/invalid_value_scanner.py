"""Invalid feature-value detection for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase4.contract.imputation_contract import (
    ImputationContract,
)


class InvalidValueRule(BaseModel):
    """Configured invalid-value rules for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    minimum_value: float | None = None
    maximum_value: float | None = None

    allowed_values: list[
        str | int | float | bool,
    ] | None = None

    forbidden_values: list[
        str | int | float | bool,
    ] = Field(default_factory=list)

    empty_string_is_invalid: bool = False
    whitespace_only_is_invalid: bool = False

    positive_infinity_is_invalid: bool = True
    negative_infinity_is_invalid: bool = True

    unparseable_datetime_is_invalid: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ColumnInvalidValueScan(BaseModel):
    """Invalid-value scan for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    partition: PartitionName

    invalid_count: int = Field(
        default=0,
        ge=0,
    )

    invalid_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    positive_infinity_count: int = Field(
        default=0,
        ge=0,
    )

    negative_infinity_count: int = Field(
        default=0,
        ge=0,
    )

    below_minimum_count: int = Field(
        default=0,
        ge=0,
    )

    above_maximum_count: int = Field(
        default=0,
        ge=0,
    )

    forbidden_value_count: int = Field(
        default=0,
        ge=0,
    )

    disallowed_value_count: int = Field(
        default=0,
        ge=0,
    )

    empty_string_count: int = Field(
        default=0,
        ge=0,
    )

    whitespace_only_count: int = Field(
        default=0,
        ge=0,
    )

    unparseable_datetime_count: int = Field(
        default=0,
        ge=0,
    )

    example_indices: list[str | int] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class InvalidValueScanResult(BaseModel):
    """Complete invalid-value scan output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    scans: dict[
        str,
        ColumnInvalidValueScan,
    ] = Field(default_factory=dict)

    columns_with_invalid_values: list[str] = Field(
        default_factory=list
    )

    total_invalid_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class InvalidValueScannerOptions:
    """Options controlling invalid-value detection."""

    enabled: bool = True

    detect_infinite_numeric_values: bool = True
    detect_unparseable_datetimes: bool = True

    default_empty_string_is_invalid: bool = False
    default_whitespace_only_is_invalid: bool = False

    convert_invalid_values_to_missing: bool = True

    maximum_example_indices: int = 10

    rules: dict[
        str,
        InvalidValueRule,
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate invalid-value scanner options."""

        if self.maximum_example_indices < 0:
            raise ValueError(
                "maximum_example_indices cannot be negative."
            )


class InvalidValueScanner:
    """Detect and optionally normalize configured invalid values."""

    component_name = "invalid_value_scanner"

    def __init__(
        self,
        options: InvalidValueScannerOptions | None = None,
    ) -> None:
        """Initialize the invalid-value scanner."""

        self.options = (
            options or InvalidValueScannerOptions()
        )

    def scan_partition(
        self,
        *,
        contract: ImputationContract,
        partition: PartitionName,
    ) -> tuple[
        InvalidValueScanResult,
        pd.DataFrame,
    ]:
        """Scan and normalize one partition feature frame."""

        source = contract.partition(
            partition
        ).X

        frame = source.copy(
            deep=True
        )

        if not self.options.enabled:
            return (
                InvalidValueScanResult(
                    metadata={
                        "enabled": False,
                    }
                ),
                frame,
            )

        scans: dict[
            str,
            ColumnInvalidValueScan,
        ] = {}

        total_invalid_count = 0

        for column in frame.columns:
            rule = self.options.rules.get(
                str(column),
                InvalidValueRule(
                    column=str(column),
                    empty_string_is_invalid=(
                        self.options
                        .default_empty_string_is_invalid
                    ),
                    whitespace_only_is_invalid=(
                        self.options
                        .default_whitespace_only_is_invalid
                    ),
                ),
            )

            invalid_mask, details = (
                self._invalid_mask(
                    contract=contract,
                    column=str(column),
                    series=frame[column],
                    rule=rule,
                )
            )

            invalid_count = int(
                invalid_mask.sum()
            )

            total_invalid_count += (
                invalid_count
            )

            if (
                invalid_count > 0
                and self.options
                .convert_invalid_values_to_missing
            ):
                frame.loc[
                    invalid_mask,
                    column,
                ] = pd.NA

            scans[str(column)] = (
                ColumnInvalidValueScan(
                    column=str(column),
                    partition=partition,
                    invalid_count=invalid_count,
                    invalid_percentage=(
                        invalid_count
                        / len(frame)
                        * 100.0
                        if len(frame) > 0
                        else 0.0
                    ),
                    positive_infinity_count=(
                        details[
                            "positive_infinity_count"
                        ]
                    ),
                    negative_infinity_count=(
                        details[
                            "negative_infinity_count"
                        ]
                    ),
                    below_minimum_count=(
                        details[
                            "below_minimum_count"
                        ]
                    ),
                    above_maximum_count=(
                        details[
                            "above_maximum_count"
                        ]
                    ),
                    forbidden_value_count=(
                        details[
                            "forbidden_value_count"
                        ]
                    ),
                    disallowed_value_count=(
                        details[
                            "disallowed_value_count"
                        ]
                    ),
                    empty_string_count=(
                        details[
                            "empty_string_count"
                        ]
                    ),
                    whitespace_only_count=(
                        details[
                            "whitespace_only_count"
                        ]
                    ),
                    unparseable_datetime_count=(
                        details[
                            "unparseable_datetime_count"
                        ]
                    ),
                    example_indices=[
                        self._safe_identifier(
                            value
                        )
                        for value
                        in frame.index[
                            invalid_mask
                        ].tolist()[
                            : self.options
                            .maximum_example_indices
                        ]
                    ],
                )
            )

        return (
            InvalidValueScanResult(
                scans=scans,
                columns_with_invalid_values=[
                    column
                    for column, scan
                    in scans.items()
                    if scan.invalid_count > 0
                ],
                total_invalid_count=(
                    total_invalid_count
                ),
                metadata={
                    "enabled": True,
                    "partition": (
                        partition.value
                    ),
                    "converted_to_missing": (
                        self.options
                        .convert_invalid_values_to_missing
                    ),
                },
            ),
            frame,
        )

    def _invalid_mask(
        self,
        *,
        contract: ImputationContract,
        column: str,
        series: pd.Series,
        rule: InvalidValueRule,
    ) -> tuple[
        pd.Series,
        dict[str, int],
    ]:
        """Build an invalid-value mask for one feature."""

        invalid_mask = pd.Series(
            False,
            index=series.index,
            dtype=bool,
        )

        details = {
            "positive_infinity_count": 0,
            "negative_infinity_count": 0,
            "below_minimum_count": 0,
            "above_maximum_count": 0,
            "forbidden_value_count": 0,
            "disallowed_value_count": 0,
            "empty_string_count": 0,
            "whitespace_only_count": 0,
            "unparseable_datetime_count": 0,
        }

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        numeric_family = (
            column
            in contract.phase3_state
            .columns
            .numeric
        )

        if (
            numeric_family
            and self.options
            .detect_infinite_numeric_values
        ):
            positive_infinity_mask = (
                numeric == np.inf
            )

            negative_infinity_mask = (
                numeric == -np.inf
            )

            if rule.positive_infinity_is_invalid:
                invalid_mask |= (
                    positive_infinity_mask
                )

                details[
                    "positive_infinity_count"
                ] = int(
                    positive_infinity_mask.sum()
                )

            if rule.negative_infinity_is_invalid:
                invalid_mask |= (
                    negative_infinity_mask
                )

                details[
                    "negative_infinity_count"
                ] = int(
                    negative_infinity_mask.sum()
                )

        if (
            numeric_family
            and rule.minimum_value is not None
        ):
            below_minimum_mask = (
                numeric.notna()
                & (
                    numeric
                    < rule.minimum_value
                )
            )

            invalid_mask |= (
                below_minimum_mask
            )

            details[
                "below_minimum_count"
            ] = int(
                below_minimum_mask.sum()
            )

        if (
            numeric_family
            and rule.maximum_value is not None
        ):
            above_maximum_mask = (
                numeric.notna()
                & (
                    numeric
                    > rule.maximum_value
                )
            )

            invalid_mask |= (
                above_maximum_mask
            )

            details[
                "above_maximum_count"
            ] = int(
                above_maximum_mask.sum()
            )

        if rule.forbidden_values:
            forbidden_mask = series.isin(
                rule.forbidden_values
            )

            invalid_mask |= forbidden_mask

            details[
                "forbidden_value_count"
            ] = int(
                forbidden_mask.sum()
            )

        if rule.allowed_values is not None:
            disallowed_mask = (
                series.notna()
                & ~series.isin(
                    rule.allowed_values
                )
            )

            invalid_mask |= (
                disallowed_mask
            )

            details[
                "disallowed_value_count"
            ] = int(
                disallowed_mask.sum()
            )

        string_like = (
            pd.api.types.is_object_dtype(
                series.dtype
            )
            or pd.api.types.is_string_dtype(
                series.dtype
            )
        )

        if string_like:
            string_values = series.astype(
                "string"
            )

            if rule.empty_string_is_invalid:
                empty_mask = (
                    string_values == ""
                ).fillna(False)

                invalid_mask |= empty_mask

                details[
                    "empty_string_count"
                ] = int(
                    empty_mask.sum()
                )

            if rule.whitespace_only_is_invalid:
                whitespace_mask = (
                    string_values.notna()
                    & (
                        string_values.str.strip()
                        == ""
                    )
                    & (
                        string_values != ""
                    )
                ).fillna(False)

                invalid_mask |= (
                    whitespace_mask
                )

                details[
                    "whitespace_only_count"
                ] = int(
                    whitespace_mask.sum()
                )

        datetime_family = (
            column
            in contract.phase3_state
            .columns
            .datetime
        )

        if (
            datetime_family
            and self.options
            .detect_unparseable_datetimes
            and rule
            .unparseable_datetime_is_invalid
        ):
            parsed = pd.to_datetime(
                series,
                errors="coerce",
            )

            unparseable_mask = (
                series.notna()
                & parsed.isna()
            )

            invalid_mask |= (
                unparseable_mask
            )

            details[
                "unparseable_datetime_count"
            ] = int(
                unparseable_mask.sum()
            )

        invalid_mask &= (
            series.notna()
            | invalid_mask
        )

        return invalid_mask, details

    @staticmethod
    def _safe_identifier(
        value: object,
    ) -> str | int:
        """Convert an index value into a report-safe identifier."""

        if isinstance(value, (str, int)):
            return value

        return str(value)
