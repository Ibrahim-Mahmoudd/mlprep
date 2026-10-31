"""Textual missing-token detection for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase4.contract.imputation_contract import (
    ImputationContract,
)


class ColumnMissingTokenScan(BaseModel):
    """Missing-token scan for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    partition: PartitionName

    matched_count: int = Field(
        default=0,
        ge=0,
    )

    matched_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    matched_tokens: dict[
        str,
        int,
    ] = Field(default_factory=dict)

    converted_to_missing: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class MissingTokenDetectionResult(BaseModel):
    """Complete missing-token detection output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    scans: dict[
        str,
        ColumnMissingTokenScan,
    ] = Field(default_factory=dict)

    columns_with_tokens: list[str] = Field(
        default_factory=list
    )

    total_matched_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class MissingTokenDetectorOptions:
    """Options controlling text missing-token detection."""

    enabled: bool = True

    default_tokens: set[str] = field(
        default_factory=lambda: {
            "",
            "na",
            "n/a",
            "null",
            "none",
            "nan",
            "missing",
            "-",
        }
    )

    column_tokens: dict[
        str,
        set[str],
    ] = field(default_factory=dict)

    excluded_columns: list[str] = field(
        default_factory=list
    )

    normalize_case: bool = True
    strip_whitespace: bool = True

    convert_tokens_to_missing: bool = True

    include_categorical_columns: bool = True
    include_text_columns: bool = True
    include_boolean_columns: bool = True

    preserve_original_token_counts: bool = True

    def __post_init__(self) -> None:
        """Validate missing-token detector options."""

        self.default_tokens = {
            str(token)
            for token in self.default_tokens
        }

        self.column_tokens = {
            str(column): {
                str(token)
                for token in tokens
            }
            for column, tokens
            in self.column_tokens.items()
        }


class MissingTokenDetector:
    """Detect and normalize configured textual missing tokens."""

    component_name = "missing_token_detector"

    def __init__(
        self,
        options: (
            MissingTokenDetectorOptions | None
        ) = None,
    ) -> None:
        """Initialize the missing-token detector."""

        self.options = (
            options
            or MissingTokenDetectorOptions()
        )

    def detect_partition(
        self,
        *,
        contract: ImputationContract,
        partition: PartitionName,
        frame: pd.DataFrame | None = None,
    ) -> tuple[
        MissingTokenDetectionResult,
        pd.DataFrame,
    ]:
        """Detect and normalize tokens in one partition."""

        source = (
            frame
            if frame is not None
            else contract.partition(
                partition
            ).X
        )

        normalized_frame = source.copy(
            deep=True
        )

        if not self.options.enabled:
            return (
                MissingTokenDetectionResult(
                    metadata={
                        "enabled": False,
                    }
                ),
                normalized_frame,
            )

        selected_columns = (
            self._selected_columns(
                contract=contract,
                frame=normalized_frame,
            )
        )

        scans: dict[
            str,
            ColumnMissingTokenScan,
        ] = {}

        total_matched_count = 0

        for column in selected_columns:
            tokens = self._tokens_for_column(
                column
            )

            token_mask, token_counts = (
                self._token_mask(
                    series=normalized_frame[
                        column
                    ],
                    tokens=tokens,
                )
            )

            matched_count = int(
                token_mask.sum()
            )

            total_matched_count += (
                matched_count
            )

            if (
                matched_count > 0
                and self.options
                .convert_tokens_to_missing
            ):
                normalized_frame.loc[
                    token_mask,
                    column,
                ] = pd.NA

            scans[column] = ColumnMissingTokenScan(
                column=column,
                partition=partition,
                matched_count=matched_count,
                matched_percentage=(
                    matched_count
                    / len(normalized_frame)
                    * 100.0
                    if len(normalized_frame) > 0
                    else 0.0
                ),
                matched_tokens=(
                    token_counts
                    if self.options
                    .preserve_original_token_counts
                    else {}
                ),
                converted_to_missing=bool(
                    matched_count > 0
                    and self.options
                    .convert_tokens_to_missing
                ),
            )

        return (
            MissingTokenDetectionResult(
                scans=scans,
                columns_with_tokens=[
                    column
                    for column, scan
                    in scans.items()
                    if scan.matched_count > 0
                ],
                total_matched_count=(
                    total_matched_count
                ),
                metadata={
                    "enabled": True,
                    "partition": (
                        partition.value
                    ),
                    "converted_to_missing": (
                        self.options
                        .convert_tokens_to_missing
                    ),
                },
            ),
            normalized_frame,
        )

    def _selected_columns(
        self,
        *,
        contract: ImputationContract,
        frame: pd.DataFrame,
    ) -> list[object]:
        """Return columns eligible for token detection."""

        groups = (
            contract.phase3_state.columns
        )

        selected: list[str] = []

        if self.options.include_categorical_columns:
            selected.extend(
                groups.categorical
            )

        if self.options.include_text_columns:
            selected.extend(
                groups.text
            )

        if self.options.include_boolean_columns:
            selected.extend(
                groups.boolean
            )

        excluded = set(
            self.options.excluded_columns
        )

        return [
            column
            for column in dict.fromkeys(
                selected
            )
            if (
                column in frame.columns
                and column not in excluded
            )
        ]

    def _tokens_for_column(
        self,
        column: str,
    ) -> set[object]:
        """Return normalized tokens for one feature."""

        tokens = set(
            self.options.default_tokens
        )

        tokens.update(
            self.options.column_tokens.get(
                column,
                set(),
            )
        )

        return {
            self._normalize_token(token)
            for token in tokens
        }

    def _token_mask(
        self,
        *,
        series: pd.Series,
        tokens: set[str],
    ) -> tuple[
        pd.Series,
        dict[str, int],
    ]:
        """Build a token-match mask and token counts."""

        string_values = series.astype(
            "string"
        )

        normalized = string_values.map(
            lambda value: (
                self._normalize_token(
                    str(value)
                )
                if pd.notna(value)
                else None
            )
        )

        token_mask = (
            normalized.isin(tokens)
            & series.notna()
        )

        matched_normalized = normalized.loc[
            token_mask
        ]

        token_counts = {
            str(token): int(count)
            for token, count
            in matched_normalized
            .value_counts(
                dropna=False
            )
            .items()
        }

        return (
            token_mask.astype(bool),
            token_counts,
        )

    def _normalize_token(
        self,
        value: str,
    ) -> str:
        """Normalize one configured or observed token."""

        normalized = value

        if self.options.strip_whitespace:
            normalized = normalized.strip()

        if self.options.normalize_case:
            normalized = normalized.casefold()

        return normalized