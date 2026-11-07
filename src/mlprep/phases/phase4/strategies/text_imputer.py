"""Text imputation strategies for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from mlprep.phases.phase4.strategies.base import (
    BaseImputer,
)

TextImputationMethod = Literal[
    "constant",
    "empty_string",
    "missing_token",
    "most_frequent",
]


@dataclass(slots=True)
class TextImputerOptions:
    """Options controlling text imputation."""

    method: TextImputationMethod = "missing_token"

    constant_value: str = ""
    missing_token: str = "__MISSING_TEXT__"

    preserve_string_dtype: bool = True

    def __post_init__(self) -> None:
        """Validate text imputer options."""

        if (
            self.method == "missing_token"
            and not self.missing_token
        ):
            raise ValueError(
                "missing_token cannot be empty."
            )


class TextImputer(BaseImputer):
    """Fit and apply text imputation."""

    strategy_name = "text"

    def __init__(
        self,
        *,
        column: str,
        options: TextImputerOptions | None = None,
    ) -> None:
        """Initialize the text imputer."""

        super().__init__(
            column=column
        )

        self.options = (
            options or TextImputerOptions()
        )

        self.state.metadata[
            "method"
        ] = self.options.method

    def _fit(
        self,
        series: pd.Series,
    ) -> str:
        """Learn a text fill value."""

        if self.options.method == "constant":
            return self.options.constant_value

        if self.options.method == "empty_string":
            return ""

        if self.options.method == "missing_token":
            return self.options.missing_token

        valid = series.dropna().astype(
            "string"
        )

        if valid.empty:
            raise ValueError(
                f"Text feature {self.column!r} "
                "contains no valid training values."
            )

        counts = valid.value_counts()

        maximum_count = int(
            counts.max()
        )

        candidates = [
            str(value)
            for value
            in counts[
                counts == maximum_count
            ].index
        ]

        candidates.sort()

        return candidates[0]

    def _transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Fill missing text values."""

        transformed = series.astype(
            "string"
        ).fillna(
            str(
                self.state.learned_value
            )
        )

        if self.options.preserve_string_dtype:
            return transformed

        return transformed.astype(
            "object"
        )