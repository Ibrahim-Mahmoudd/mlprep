"""Reusable constant imputation for Phase 4."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mlprep.phases.phase4.strategies.base import (
    BaseImputer,
)


class ConstantImputer(BaseImputer):
    """Fill missing values using a configured constant."""

    strategy_name = "constant"

    def __init__(
        self,
        *,
        column: str,
        value: Any,
    ) -> None:
        """Initialize the constant imputer."""

        super().__init__(
            column=column
        )

        if value is None or pd.isna(value):
            raise ValueError(
                "Constant imputation value cannot be missing."
            )

        self.value = value

        self.state.metadata[
            "configured_constant"
        ] = value

    def _fit(
        self,
        series: pd.Series,
    ) -> Any:
        """Return the configured constant."""

        return self.value

    def _transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Fill missing values with the constant."""

        if isinstance(
            series.dtype,
            pd.CategoricalDtype,
        ):
            transformed = series.copy()

            if (
                self.value
                not in transformed.cat.categories
            ):
                transformed = (
                    transformed
                    .cat
                    .add_categories(
                        [self.value]
                    )
                )

            return transformed.fillna(
                self.value
            )

        return series.fillna(
            self.value
        )
