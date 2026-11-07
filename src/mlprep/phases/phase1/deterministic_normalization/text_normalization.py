"""Deterministic text-value normalization for Phase 1."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
)

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    NormalizationEvent,
    Phase1Result,
)
from mlprep.types import (
    FindingAction,
    FindingCategory,
    Severity,
)

TextCaseStyle = Literal[
    "preserve",
    "lower",
    "upper",
    "casefold",
]


@dataclass(slots=True)
class TextColumnOptions:
    """Optional rules overriding global text normalization."""

    strip_whitespace: bool | None = None
    collapse_whitespace: bool | None = None
    normalize_unicode: bool | None = None
    remove_control_characters: bool | None = None
    case_style: TextCaseStyle | None = None

    aliases: dict[str, str] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class TextNormalizationOptions:
    """Options controlling text-value normalization."""

    enabled: bool = True

    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(
        default_factory=list
    )

    strip_whitespace: bool = True
    collapse_whitespace: bool = True
    normalize_unicode: bool = True
    remove_control_characters: bool = True

    case_style: TextCaseStyle = "preserve"

    global_aliases: dict[str, str] = field(
        default_factory=dict
    )

    column_options: dict[str, TextColumnOptions] = field(
        default_factory=dict
    )

    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate text-normalization options."""

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )

        if self.include_columns is not None:
            overlap = set(
                self.include_columns
            ).intersection(self.exclude_columns)

            if overlap:
                columns = ", ".join(sorted(overlap))

                raise ValueError(
                    "Columns cannot be both included and "
                    f"excluded: {columns}"
                )


class TextNormalizer:
    """Apply deterministic normalization to text columns."""

    component_name = "text_normalization"

    def __init__(
        self,
        options: TextNormalizationOptions | None = None,
    ) -> None:
        """Initialize the text normalizer."""

        self.options = (
            options or TextNormalizationOptions()
        )

    def normalize(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> pd.DataFrame:
        """Normalize selected text columns."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Text normalization requires a pandas DataFrame."
            )

        working_data = data.copy()

        if not self.options.enabled:
            return working_data

        columns = self._select_columns(
            working_data
        )

        self._report_missing_configured_columns(
            data=working_data,
            result=result,
        )

        for column in columns:
            self._normalize_column(
                data=working_data,
                column=column,
                result=result,
            )

        return working_data

    def _normalize_column(
        self,
        *,
        data: pd.DataFrame,
        column: str,
        result: Phase1Result,
    ) -> None:
        """Normalize one text column."""

        column_options = self.options.column_options.get(
            column,
            TextColumnOptions(),
        )

        original_series = data[column].copy()

        normalized_series = original_series.map(
            lambda value: self._normalize_value(
                value=value,
                column_options=column_options,
            ),
            na_action="ignore",
        )

        changed_mask = (
            original_series.notna()
            & normalized_series.notna()
            & (
                original_series.astype("string")
                != normalized_series.astype("string")
            )
        )

        changed_count = int(
            changed_mask.sum()
        )

        if changed_count == 0:
            return

        data[column] = normalized_series

        example_values = (
            original_series.loc[changed_mask]
            .head(self.options.maximum_examples)
            .tolist()
        )

        result.add_normalization_event(
            NormalizationEvent(
                component=self.component_name,
                operation="normalize_text_values",
                description=(
                    f"Normalized {changed_count} text value or "
                    f"values in column {column!r}."
                ),
                column=column,
                affected_row_count=changed_count,
                before_summary={
                    "examples": [
                        str(value)
                        for value in example_values
                    ],
                },
                after_summary={
                    "examples": [
                        str(value)
                        for value in normalized_series.loc[
                            changed_mask
                        ]
                        .head(self.options.maximum_examples)
                        .tolist()
                    ],
                },
                reversible=False,
            )
        )

        result.add_finding(
            Finding(
                code="P1-TEXT-001",
                category=(
                    FindingCategory.TEXT_NORMALIZATION
                ),
                severity=Severity.INFO,
                component=self.component_name,
                message=(
                    f"Text values were normalized in "
                    f"column {column!r}."
                ),
                columns=[column],
                affected_row_count=changed_count,
                affected_row_percentage=self._percentage(
                    affected_count=changed_count,
                    total_count=len(data),
                ),
                example_values=[
                    str(value)
                    for value in example_values
                ],
                recommended_action=FindingAction.COERCE,
                applied_action=FindingAction.COERCE,
                blocks_readiness=False,
            )
        )

    def _normalize_value(
        self,
        *,
        value: object,
        column_options: TextColumnOptions,
    ) -> object:
        """Normalize one nonmissing text value."""

        if not isinstance(value, str):
            return value

        strip_whitespace = self._resolve_option(
            local_value=column_options.strip_whitespace,
            global_value=self.options.strip_whitespace,
        )

        collapse_whitespace = self._resolve_option(
            local_value=column_options.collapse_whitespace,
            global_value=self.options.collapse_whitespace,
        )

        normalize_unicode = self._resolve_option(
            local_value=column_options.normalize_unicode,
            global_value=self.options.normalize_unicode,
        )

        remove_control_characters = self._resolve_option(
            local_value=(
                column_options.remove_control_characters
            ),
            global_value=(
                self.options.remove_control_characters
            ),
        )

        case_style = (
            column_options.case_style
            or self.options.case_style
        )

        normalized_value = value

        if normalize_unicode:
            normalized_value = unicodedata.normalize(
                "NFKC",
                normalized_value,
            )

        if remove_control_characters:
            normalized_value = "".join(
                character
                for character in normalized_value
                if (
                    unicodedata.category(character)
                    not in {"Cc", "Cf"}
                    or character in {"\t", "\n", "\r"}
                )
            )

        if strip_whitespace:
            normalized_value = normalized_value.strip()

        if collapse_whitespace:
            normalized_value = re.sub(
                r"\s+",
                " ",
                normalized_value,
            )

        normalized_value = self._apply_case_style(
            value=normalized_value,
            case_style=case_style,
        )

        aliases = {
            **self.options.global_aliases,
            **column_options.aliases,
        }

        return aliases.get(
            normalized_value,
            normalized_value,
        )

    def _select_columns(
        self,
        data: pd.DataFrame,
    ) -> list[object]:
        """Select eligible text columns."""

        excluded = set(
            self.options.exclude_columns
        )

        if self.options.include_columns is not None:
            return [
                column
                for column in self.options.include_columns
                if column in data.columns
                and column not in excluded
            ]

        return [
            str(column)
            for column in data.columns
            if str(column) not in excluded
            and (
                is_string_dtype(data[column].dtype)
                or is_object_dtype(data[column].dtype)
            )
        ]

    def _report_missing_configured_columns(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> None:
        """Report selected columns absent from the dataset."""

        configured_columns = set(
            self.options.column_options
        )

        if self.options.include_columns is not None:
            configured_columns.update(
                self.options.include_columns
            )

        missing_columns = sorted(
            configured_columns.difference(
                str(column)
                for column in data.columns
            )
        )

        if not missing_columns:
            return

        result.add_finding(
            Finding(
                code="P1-TEXT-002",
                category=(
                    FindingCategory.TEXT_NORMALIZATION
                ),
                severity=Severity.WARNING,
                component=self.component_name,
                message=(
                    "Text-normalization configuration references "
                    "columns that do not exist."
                ),
                columns=missing_columns,
                recommended_action=FindingAction.WARN,
                applied_action=FindingAction.REPORT,
                blocks_readiness=False,
            )
        )

    @staticmethod
    def _resolve_option(
        *,
        local_value: bool | None,
        global_value: bool,
    ) -> bool:
        """Resolve a local boolean override."""

        if local_value is None:
            return global_value

        return local_value

    @staticmethod
    def _apply_case_style(
        *,
        value: str,
        case_style: TextCaseStyle,
    ) -> str:
        """Apply the configured case style."""

        if case_style == "lower":
            return value.lower()

        if case_style == "upper":
            return value.upper()

        if case_style == "casefold":
            return value.casefold()

        return value

    @staticmethod
    def _percentage(
        *,
        affected_count: int,
        total_count: int,
    ) -> float:
        """Calculate a percentage safely."""

        if total_count <= 0:
            return 0.0

        return round(
            affected_count / total_count * 100.0,
            6,
        )