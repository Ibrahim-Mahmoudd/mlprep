"""Missing-token detection and normalization for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

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
from mlprep.phases.phase1.output_governance.state import (
    MissingTokenState,
    Phase1State,
)
from mlprep.types import (
    FindingAction,
    FindingCategory,
    Severity,
)

MissingToken: TypeAlias = str | int | float


@dataclass(slots=True)
class MissingTokenColumnOptions:
    """Missing-token rules overriding global behavior."""

    tokens: list[MissingToken] = field(
        default_factory=list
    )

    replace_global_tokens: bool = False

    case_sensitive: bool | None = None
    strip_before_comparison: bool | None = None

    treat_empty_string_as_missing: bool | None = None
    treat_whitespace_only_as_missing: bool | None = None


@dataclass(slots=True)
class MissingTokenOptions:
    """Options controlling missing-token normalization."""

    enabled: bool = True

    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(
        default_factory=list
    )

    global_tokens: list[MissingToken] = field(
        default_factory=lambda: [
            "",
            "NA",
            "N/A",
            "NULL",
            "null",
            "None",
            "?",
        ]
    )

    case_sensitive: bool = False
    strip_before_comparison: bool = True

    treat_empty_string_as_missing: bool = True
    treat_whitespace_only_as_missing: bool = True

    column_options: dict[
        str,
        MissingTokenColumnOptions,
    ] = field(default_factory=dict)

    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate missing-token options."""

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


class MissingTokenNormalizer:
    """Convert declared missing tokens to pandas.NA."""

    component_name = "missing_tokens"

    def __init__(
        self,
        options: MissingTokenOptions | None = None,
    ) -> None:
        """Initialize the missing-token normalizer."""

        self.options = (
            options or MissingTokenOptions()
        )

    def normalize(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Normalize configured missing tokens."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Missing-token normalization requires a "
                "pandas DataFrame."
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
                state=state,
            )

        result.dataset_summary.missing_cell_count = int(
            working_data.isna().sum().sum()
        )

        return working_data

    def _normalize_column(
        self,
        *,
        data: pd.DataFrame,
        column: str,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Normalize missing tokens in one column."""

        column_options = self.options.column_options.get(
            column,
            MissingTokenColumnOptions(),
        )

        tokens = self._resolve_tokens(
            column_options
        )

        case_sensitive = self._resolve_boolean(
            local_value=column_options.case_sensitive,
            global_value=self.options.case_sensitive,
        )

        strip_before_comparison = self._resolve_boolean(
            local_value=(
                column_options.strip_before_comparison
            ),
            global_value=(
                self.options.strip_before_comparison
            ),
        )

        treat_empty_string_as_missing = (
            self._resolve_boolean(
                local_value=(
                    column_options
                    .treat_empty_string_as_missing
                ),
                global_value=(
                    self.options
                    .treat_empty_string_as_missing
                ),
            )
        )

        treat_whitespace_only_as_missing = (
            self._resolve_boolean(
                local_value=(
                    column_options
                    .treat_whitespace_only_as_missing
                ),
                global_value=(
                    self.options
                    .treat_whitespace_only_as_missing
                ),
            )
        )

        normalized_tokens = {
            self._normalize_comparable_token(
                token=token,
                case_sensitive=case_sensitive,
                strip_before_comparison=(
                    strip_before_comparison
                ),
            )
            for token in tokens
        }

        series = data[column]

        token_counts: dict[str, int] = {}
        detected_tokens: list[MissingToken] = []
        replacement_mask = pd.Series(
            False,
            index=series.index,
            dtype=bool,
        )

        for index, value in series.items():
            if pd.isna(value):
                continue

            if not self._is_missing_token(
                value=value,
                normalized_tokens=normalized_tokens,
                case_sensitive=case_sensitive,
                strip_before_comparison=(
                    strip_before_comparison
                ),
                treat_empty_string_as_missing=(
                    treat_empty_string_as_missing
                ),
                treat_whitespace_only_as_missing=(
                    treat_whitespace_only_as_missing
                ),
            ):
                continue

            replacement_mask.loc[index] = True

            token_key = self._display_token(value)

            token_counts[token_key] = (
                token_counts.get(token_key, 0) + 1
            )

            if value not in detected_tokens:
                detected_tokens.append(value)

        replacement_count = int(
            replacement_mask.sum()
        )

        state.register_missing_token_state(
            MissingTokenState(
                column=column,
                configured_tokens=tokens,
                detected_tokens=detected_tokens,
                normalized_token_counts=token_counts,
                total_normalized_count=replacement_count,
                case_sensitive=case_sensitive,
                strip_before_comparison=(
                    strip_before_comparison
                ),
            )
        )

        if replacement_count == 0:
            return

        example_values = (
            series.loc[replacement_mask]
            .head(self.options.maximum_examples)
            .tolist()
        )

        data.loc[replacement_mask, column] = pd.NA

        result.add_normalization_event(
            NormalizationEvent(
                component=self.component_name,
                operation="normalize_missing_tokens",
                description=(
                    f"Converted {replacement_count} missing "
                    f"token or tokens to pandas.NA in "
                    f"column {column!r}."
                ),
                column=column,
                affected_row_count=replacement_count,
                before_summary={
                    "token_counts": token_counts,
                    "examples": [
                        self._display_token(value)
                        for value in example_values
                    ],
                },
                after_summary={
                    "normalized_value": "pandas.NA",
                },
                reversible=False,
            )
        )

        result.add_finding(
            Finding(
                code="P1-MISSING-001",
                category=FindingCategory.MISSING_TOKEN,
                severity=Severity.INFO,
                component=self.component_name,
                message=(
                    f"Missing tokens were normalized in "
                    f"column {column!r}."
                ),
                columns=[column],
                affected_row_count=replacement_count,
                affected_row_percentage=self._percentage(
                    affected_count=replacement_count,
                    total_count=len(data),
                ),
                example_values=[
                    self._display_token(value)
                    for value in example_values
                ],
                recommended_action=(
                    FindingAction.SET_MISSING
                ),
                applied_action=FindingAction.SET_MISSING,
                blocks_readiness=False,
                details={
                    "token_counts": token_counts,
                },
            )
        )

    def _select_columns(
        self,
        data: pd.DataFrame,
    ) -> list[object]:
        """Select columns eligible for token normalization."""

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

        selected_columns: list[str] = []

        numeric_tokens_exist = any(
            isinstance(token, (int, float))
            and not isinstance(token, bool)
            for token in self.options.global_tokens
        )

        for column in data.columns:
            column_name = str(column)

            if column_name in excluded:
                continue

            column_has_numeric_tokens = any(
                isinstance(token, (int, float))
                and not isinstance(token, bool)
                for token in self.options.column_options.get(
                    column_name,
                    MissingTokenColumnOptions(),
                ).tokens
            )

            if (
                is_string_dtype(data[column].dtype)
                or is_object_dtype(data[column].dtype)
                or numeric_tokens_exist
                or column_has_numeric_tokens
            ):
                selected_columns.append(column_name)

        return selected_columns

    def _resolve_tokens(
        self,
        column_options: MissingTokenColumnOptions,
    ) -> list[object]:
        """Resolve effective missing tokens for one column."""

        if column_options.replace_global_tokens:
            return list(column_options.tokens)

        return list(
            dict.fromkeys(
                [
                    *self.options.global_tokens,
                    *column_options.tokens,
                ]
            )
        )

    def _is_missing_token(
        self,
        *,
        value: object,
        normalized_tokens: set[object],
        case_sensitive: bool,
        strip_before_comparison: bool,
        treat_empty_string_as_missing: bool,
        treat_whitespace_only_as_missing: bool,
    ) -> bool:
        """Return whether one value represents missing data."""

        if isinstance(value, str):
            if (
                treat_empty_string_as_missing
                and value == ""
            ):
                return True

            if (
                treat_whitespace_only_as_missing
                and value.strip() == ""
            ):
                return True

        normalized_value = (
            self._normalize_comparable_token(
                token=value,
                case_sensitive=case_sensitive,
                strip_before_comparison=(
                    strip_before_comparison
                ),
            )
        )

        return normalized_value in normalized_tokens

    @staticmethod
    def _normalize_comparable_token(
        *,
        token: object,
        case_sensitive: bool,
        strip_before_comparison: bool,
    ) -> object:
        """Normalize a value for safe token comparison."""

        if not isinstance(token, str):
            return token

        value = token

        if strip_before_comparison:
            value = value.strip()

        if not case_sensitive:
            value = value.casefold()

        return value

    def _report_missing_configured_columns(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> None:
        """Report configured columns absent from the dataset."""

        configured_columns = set(
            self.options.column_options
        )

        if self.options.include_columns is not None:
            configured_columns.update(
                self.options.include_columns
            )

        available_columns = {
            str(column)
            for column in data.columns
        }

        missing_columns = sorted(
            configured_columns.difference(
                available_columns
            )
        )

        if not missing_columns:
            return

        result.add_finding(
            Finding(
                code="P1-MISSING-002",
                category=FindingCategory.MISSING_TOKEN,
                severity=Severity.WARNING,
                component=self.component_name,
                message=(
                    "Missing-token configuration references "
                    "columns that do not exist."
                ),
                columns=missing_columns,
                recommended_action=FindingAction.WARN,
                applied_action=FindingAction.REPORT,
                blocks_readiness=False,
            )
        )

    @staticmethod
    def _resolve_boolean(
        *,
        local_value: bool | None,
        global_value: bool,
    ) -> bool:
        """Resolve a local boolean override."""

        if local_value is None:
            return global_value

        return local_value

    @staticmethod
    def _display_token(
        value: object,
    ) -> str:
        """Create a safe token representation for reports."""

        return repr(value)

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
