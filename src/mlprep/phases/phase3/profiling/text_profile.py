"""Text feature profiling for Phase 3."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)


class TextColumnProfile(BaseModel):
    """Analytical profile for one text feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    dtype: str

    row_count: int = Field(ge=0)
    nonmissing_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)

    missing_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    empty_string_count: int = Field(ge=0)
    whitespace_only_count: int = Field(ge=0)

    unique_count: int = Field(ge=0)
    uniqueness_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )

    minimum_character_count: int | None = None
    maximum_character_count: int | None = None
    average_character_count: float | None = None
    median_character_count: float | None = None

    minimum_word_count: int | None = None
    maximum_word_count: int | None = None
    average_word_count: float | None = None
    median_word_count: float | None = None

    values_with_digits_count: int = Field(ge=0)
    values_with_special_characters_count: int = Field(ge=0)
    values_with_line_breaks_count: int = Field(ge=0)

    duplicate_text_count: int = Field(ge=0)

    most_frequent_text_hashes: list[
        dict[str, Any]
    ] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class TextProfilerOptions:
    """Options controlling text profiling."""

    enabled: bool = True

    maximum_frequency_values: int = 10

    include_raw_values: bool = False
    hash_frequent_values: bool = True

    maximum_raw_value_length: int = 100

    def __post_init__(self) -> None:
        """Validate text-profiler options."""

        if self.maximum_frequency_values < 0:
            raise ValueError(
                "maximum_frequency_values cannot be negative."
            )

        if self.maximum_raw_value_length < 1:
            raise ValueError(
                "maximum_raw_value_length must be positive."
            )


class TextProfiler:
    """Profile free-text and short-text training features."""

    component_name = "text_profile"

    def __init__(
        self,
        options: TextProfilerOptions | None = None,
    ) -> None:
        """Initialize the text profiler."""

        self.options = (
            options or TextProfilerOptions()
        )

    def profile(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> dict[str, TextColumnProfile]:
        """Profile selected text features."""

        if not self.options.enabled:
            return {}

        frame = contract.training_features()

        profiles: dict[
            str,
            TextColumnProfile,
        ] = {}

        for column in columns.text:
            if column not in frame.columns:
                continue

            profiles[column] = (
                self._profile_column(
                    column=column,
                    series=frame[column],
                )
            )

        return profiles

    def _profile_column(
        self,
        *,
        column: str,
        series: pd.Series,
    ) -> TextColumnProfile:
        """Profile one text feature."""

        row_count = len(series)

        missing_count = int(
            series.isna().sum()
        )

        nonmissing = series.dropna().astype(
            "string"
        )

        nonmissing_count = len(nonmissing)

        missing_percentage = (
            missing_count / row_count * 100.0
            if row_count > 0
            else 0.0
        )

        stripped = nonmissing.str.strip()

        empty_string_count = int(
            (nonmissing == "").sum()
        )

        whitespace_only_count = int(
            (
                (stripped == "")
                & (nonmissing != "")
            ).sum()
        )

        unique_count = int(
            nonmissing.nunique()
        )

        uniqueness_ratio = (
            unique_count / nonmissing_count
            if nonmissing_count > 0
            else 0.0
        )

        character_counts = nonmissing.str.len()

        word_counts = nonmissing.map(
            lambda value: len(
                str(value).split()
            )
        )

        values_with_digits_count = int(
            nonmissing.str.contains(
                r"\d",
                regex=True,
                na=False,
            ).sum()
        )

        values_with_special_count = int(
            nonmissing.str.contains(
                r"[^\w\s]",
                regex=True,
                na=False,
            ).sum()
        )

        values_with_line_breaks = int(
            nonmissing.str.contains(
                r"[\r\n]",
                regex=True,
                na=False,
            ).sum()
        )

        duplicate_text_count = int(
            nonmissing.duplicated(
                keep=False
            ).sum()
        )

        frequency_records = self._frequency_records(
            nonmissing
        )

        return TextColumnProfile(
            column=column,
            dtype=str(series.dtype),
            row_count=row_count,
            nonmissing_count=nonmissing_count,
            missing_count=missing_count,
            missing_percentage=round(
                missing_percentage,
                6,
            ),
            empty_string_count=empty_string_count,
            whitespace_only_count=(
                whitespace_only_count
            ),
            unique_count=unique_count,
            uniqueness_ratio=uniqueness_ratio,
            minimum_character_count=(
                int(character_counts.min())
                if not character_counts.empty
                else None
            ),
            maximum_character_count=(
                int(character_counts.max())
                if not character_counts.empty
                else None
            ),
            average_character_count=(
                float(character_counts.mean())
                if not character_counts.empty
                else None
            ),
            median_character_count=(
                float(character_counts.median())
                if not character_counts.empty
                else None
            ),
            minimum_word_count=(
                int(word_counts.min())
                if not word_counts.empty
                else None
            ),
            maximum_word_count=(
                int(word_counts.max())
                if not word_counts.empty
                else None
            ),
            average_word_count=(
                float(word_counts.mean())
                if not word_counts.empty
                else None
            ),
            median_word_count=(
                float(word_counts.median())
                if not word_counts.empty
                else None
            ),
            values_with_digits_count=(
                values_with_digits_count
            ),
            values_with_special_characters_count=(
                values_with_special_count
            ),
            values_with_line_breaks_count=(
                values_with_line_breaks
            ),
            duplicate_text_count=(
                duplicate_text_count
            ),
            most_frequent_text_hashes=(
                frequency_records
            ),
        )

    def _frequency_records(
        self,
        series: pd.Series,
    ) -> list[dict[str, Any]]:
        """Build privacy-aware text frequency records."""

        counts = series.value_counts().head(
            self.options.maximum_frequency_values
        )

        records: list[
            dict[str, Any]
        ] = []

        for value, count in counts.items():
            text = str(value)

            record: dict[str, Any] = {
                "count": int(count),
                "length": len(text),
            }

            if self.options.include_raw_values:
                record["value"] = text[
                    : self.options
                    .maximum_raw_value_length
                ]

            elif self.options.hash_frequent_values:
                record["value_hash"] = str(
                    pd.util.hash_array(
                        pd.array([text])
                    )[0]
                )

            records.append(record)

        return records