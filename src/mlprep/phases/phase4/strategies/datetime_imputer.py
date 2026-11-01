"""Datetime imputation strategies for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from mlprep.phases.phase4.strategies.base import BaseImputer

DatetimeImputationMethod = Literal[
    "median",
    "most_frequent",
    "constant",
]


@dataclass(slots=True)
class DatetimeImputerOptions:
    """Options controlling datetime imputation."""

    method: DatetimeImputationMethod = "median"

    constant_value: str | pd.Timestamp | None = None

    day_first: bool = False
    year_first: bool = False

    def __post_init__(self) -> None:
        """Validate datetime imputer options."""

        if (
            self.method == "constant"
            and self.constant_value is None
        ):
            raise ValueError(
                "Constant datetime imputation requires "
                "constant_value."
            )


class DatetimeImputer(BaseImputer):
    """Fit and apply datetime imputation."""

    strategy_name = "datetime"

    def __init__(
        self,
        *,
        column: str,
        options: (
            DatetimeImputerOptions | None
        ) = None,
    ) -> None:
        """Initialize the datetime imputer."""

        super().__init__(
            column=column
        )

        self.options = (
            options or DatetimeImputerOptions()
        )

        self.state.metadata[
            "method"
        ] = self.options.method

    def _fit(
        self,
        series: pd.Series,
    ) -> str:
        """Learn a datetime fill value."""

        parsed = pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=self.options.day_first,
            yearfirst=self.options.year_first,
        )

        valid = parsed.dropna()

        if self.options.method == "constant":
            learned = pd.to_datetime(
                self.options.constant_value,
                errors="raise",
            )

        elif valid.empty:
            raise ValueError(
                f"Datetime feature {self.column!r} "
                "contains no valid training timestamps."
            )

        elif self.options.method == "median":
            learned = pd.Timestamp(valid.median())

        elif self.options.method == "most_frequent":
            counts = valid.value_counts()

            maximum_count = int(
                counts.max()
            )

            candidates = counts[
                counts == maximum_count
            ].index.tolist()

            candidates.sort()

            learned = pd.Timestamp(
                candidates[0]
            )

        else:
            raise ValueError(
                f"Unsupported datetime method: "
                f"{self.options.method!r}"
            )

        return pd.Timestamp(
            learned
        ).isoformat()

    def _transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Fill missing or invalid datetime values."""

        parsed = pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=self.options.day_first,
            yearfirst=self.options.year_first,
        )

        fill_value = pd.Timestamp(
            self.state.learned_value
        )

        return parsed.fillna(
            fill_value
        )
