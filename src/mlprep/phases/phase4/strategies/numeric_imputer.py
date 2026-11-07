"""Numeric imputation strategies for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from mlprep.phases.phase4.strategies.base import (
    BaseImputer,
)

NumericImputationMethod = Literal[
    "mean",
    "median",
    "constant",
    "quantile",
    "trimmed_mean",
]


@dataclass(slots=True)
class NumericImputerOptions:
    """Options controlling numeric imputation."""

    method: NumericImputationMethod = "median"

    constant_value: float = 0.0
    quantile: float = 0.50
    trim_fraction: float = 0.10

    preserve_nullable_dtype: bool = False

    def __post_init__(self) -> None:
        """Validate numeric imputer options."""

        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError(
                "quantile must be between zero and one."
            )

        if not 0.0 <= self.trim_fraction < 0.50:
            raise ValueError(
                "trim_fraction must be between zero and 0.5."
            )


class NumericImputer(BaseImputer):
    """Fit and apply numeric imputation."""

    strategy_name = "numeric"

    def __init__(
        self,
        *,
        column: str,
        options: NumericImputerOptions | None = None,
    ) -> None:
        """Initialize the numeric imputer."""

        super().__init__(
            column=column
        )

        self.options = (
            options or NumericImputerOptions()
        )

        self.state.metadata.update(
            {
                "method": self.options.method,
                "quantile": self.options.quantile,
                "trim_fraction": (
                    self.options.trim_fraction
                ),
            }
        )

    def _fit(
        self,
        series: pd.Series,
    ) -> float:
        """Learn a numeric fill value."""

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid = numeric.dropna()

        if self.options.method == "constant":
            return float(
                self.options.constant_value
            )

        if valid.empty:
            raise ValueError(
                f"Numeric feature {self.column!r} "
                "contains no valid training values."
            )

        if self.options.method == "mean":
            learned_value = valid.mean()

        elif self.options.method == "median":
            learned_value = valid.median()

        elif self.options.method == "quantile":
            learned_value = valid.quantile(
                self.options.quantile
            )

        elif self.options.method == "trimmed_mean":
            learned_value = self._trimmed_mean(
                valid
            )

        else:
            raise ValueError(
                f"Unsupported numeric method: "
                f"{self.options.method!r}"
            )

        learned_float = float(
            learned_value
        )

        if not np.isfinite(
            learned_float
        ):
            raise ValueError(
                "The learned numeric fill value is nonfinite."
            )

        return learned_float

    def _transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Fill missing numeric values."""

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        transformed = numeric.fillna(
            float(
                self.state.learned_value
            )
        )

        if self.options.preserve_nullable_dtype:
            original_dtype = str(
                series.dtype
            )

            if original_dtype.startswith(
                "Int"
            ):
                transformed = transformed.round().astype(
                    original_dtype
                )

            elif original_dtype.startswith(
                "Float"
            ):
                transformed = transformed.astype(
                    original_dtype
                )

        return transformed

    def _trimmed_mean(
        self,
        values: pd.Series,
    ) -> float:
        """Calculate a trimmed training mean."""

        ordered = values.sort_values()

        trim_count = int(
            len(ordered)
            * self.options.trim_fraction
        )

        if (
            trim_count == 0
            or trim_count * 2 >= len(
                ordered
            )
        ):
            return float(
                ordered.mean()
            )

        trimmed = ordered.iloc[
            trim_count:
            len(ordered) - trim_count
        ]

        return float(
            trimmed.mean()
        )
