"""Categorical imputation strategies for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from mlprep.phases.phase4.strategies.base import (
    BaseImputer,
)

CategoricalImputationMethod = Literal[
    "most_frequent",
    "constant",
    "missing_category",
]


@dataclass(slots=True)
class CategoricalImputerOptions:
    """Options controlling categorical imputation."""

    method: CategoricalImputationMethod = (
        "missing_category"
    )

    constant_value: str = "__MISSING__"

    preserve_category_dtype: bool = True

    def __post_init__(self) -> None:
        """Validate categorical imputer options."""

        if (
            self.method
            in {
                "constant",
                "missing_category",
            }
            and not self.constant_value
        ):
            raise ValueError(
                "constant_value cannot be empty."
            )


class CategoricalImputer(BaseImputer):
    """Fit and apply categorical imputation."""

    strategy_name = "categorical"

    def __init__(
        self,
        *,
        column: str,
        options: (
            CategoricalImputerOptions | None
        ) = None,
    ) -> None:
        """Initialize the categorical imputer."""

        super().__init__(
            column=column
        )

        self.options = (
            options
            or CategoricalImputerOptions()
        )

        self.state.metadata[
            "method"
        ] = self.options.method

    def _fit(
        self,
        series: pd.Series,
    ) -> object:
        """Learn the categorical fill value."""

        if self.options.method in {
            "constant",
            "missing_category",
        }:
            return self.options.constant_value

        valid = series.dropna()

        if valid.empty:
            raise ValueError(
                f"Categorical feature {self.column!r} "
                "contains no valid training values."
            )

        counts = valid.value_counts(
            dropna=False
        )

        maximum_count = int(
            counts.max()
        )

        modes = counts[
            counts == maximum_count
        ].index.tolist()

        modes.sort(
            key=lambda value: str(value)
        )

        return modes[0]

    def _transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Fill missing categorical values."""

        fill_value = (
            self.state.learned_value
        )

        if isinstance(
            series.dtype,
            pd.CategoricalDtype,
        ):
            transformed = series.copy()

            if (
                fill_value
                not in transformed.cat.categories
            ):
                transformed = (
                    transformed
                    .cat
                    .add_categories(
                        [fill_value]
                    )
                )

            return transformed.fillna(
                fill_value
            )

        return series.fillna(
            fill_value
        )
