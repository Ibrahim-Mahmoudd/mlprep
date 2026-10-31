"""Boolean imputation strategies for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from mlprep.phases.phase4.strategies.base import (
    BaseImputer,
)


BooleanImputationMethod = Literal[
    "most_frequent",
    "constant_true",
    "constant_false",
    "missing_category",
]


@dataclass(slots=True)
class BooleanImputerOptions:
    """Options controlling Boolean imputation."""

    method: BooleanImputationMethod = "most_frequent"

    missing_category_value: str = "__MISSING__"

    def __post_init__(self) -> None:
        """Validate Boolean imputer options."""

        if (
            self.method == "missing_category"
            and not self.missing_category_value
        ):
            raise ValueError(
                "missing_category_value cannot be empty."
            )


class BooleanImputer(BaseImputer):
    """Fit and apply Boolean imputation."""

    strategy_name = "boolean"

    def __init__(
        self,
        *,
        column: str,
        options: BooleanImputerOptions | None = None,
    ) -> None:
        """Initialize the Boolean imputer."""

        super().__init__(
            column=column
        )

        self.options = (
            options or BooleanImputerOptions()
        )

        self.state.metadata[
            "method"
        ] = self.options.method

    def _fit(
        self,
        series: pd.Series,
    ) -> bool | str:
        """Learn the Boolean fill value."""

        if self.options.method == "constant_true":
            return True

        if self.options.method == "constant_false":
            return False

        if self.options.method == "missing_category":
            return (
                self.options
                .missing_category_value
            )

        valid = series.dropna()

        if valid.empty:
            raise ValueError(
                f"Boolean feature {self.column!r} "
                "contains no valid training values."
            )

        normalized = valid.map(
            self._normalize_boolean
        )

        counts = normalized.value_counts()

        maximum_count = int(
            counts.max()
        )

        candidates = counts[
            counts == maximum_count
        ].index.tolist()

        return bool(
            sorted(
                candidates,
                key=lambda value: str(value),
            )[0]
        )

    def _transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Fill missing Boolean values."""

        fill_value = (
            self.state.learned_value
        )

        if isinstance(
            fill_value,
            bool,
        ):
            normalized = series.map(
                lambda value: (
                    pd.NA
                    if pd.isna(value)
                    else self._normalize_boolean(
                        value
                    )
                )
            )

            return normalized.fillna(
                fill_value
            ).astype("boolean")

        return series.astype(
            "object"
        ).fillna(
            fill_value
        )

    @staticmethod
    def _normalize_boolean(
        value: object,
    ) -> bool:
        """Normalize one Boolean-like value."""

        if isinstance(value, bool):
            return value

        if isinstance(
            value,
            (int, float),
        ) and value in {
            0,
            1,
            0.0,
            1.0,
        }:
            return bool(value)

        normalized = str(
            value
        ).strip().casefold()

        true_values = {
            "true",
            "yes",
            "y",
            "1",
            "t",
        }

        false_values = {
            "false",
            "no",
            "n",
            "0",
            "f",
        }

        if normalized in true_values:
            return True

        if normalized in false_values:
            return False

        raise ValueError(
            f"Value {value!r} is not Boolean-like."
        )