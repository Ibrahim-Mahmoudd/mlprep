"""Standard scaling for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mlprep.phases.phase6.numeric.base import (
    BaseNumericTransformer,
)


@dataclass(slots=True)
class StandardScalerOptions:
    """Options controlling standard scaling."""

    with_mean: bool = True
    with_standard_deviation: bool = True
    degrees_of_freedom: int = 0
    constant_tolerance: float = 1e-12
    preserve_float32: bool = False

    def __post_init__(self) -> None:
        """Validate standard-scaler options."""

        if self.degrees_of_freedom < 0:
            raise ValueError(
                "degrees_of_freedom cannot be negative."
            )
        if self.constant_tolerance < 0.0:
            raise ValueError(
                "constant_tolerance cannot be negative."
            )


class StandardScaler(BaseNumericTransformer):
    """Scale values using training mean and standard deviation."""

    transformer_name = "standard_scaler"

    def __init__(
        self,
        *,
        column: str,
        options: StandardScalerOptions | None = None,
    ) -> None:
        """Initialize the standard scaler."""

        super().__init__(column=column)
        self.options = options or StandardScalerOptions()
        self.mean: float | None = None
        self.scale: float | None = None

    def _fit(self, valid: pd.Series) -> dict[str, Any]:
        """Learn training mean and standard deviation."""

        if valid.empty:
            raise ValueError(
                f"Feature {self.column!r} contains no "
                "valid training values."
            )

        self.mean = float(valid.mean())
        standard_deviation = float(
            valid.std(ddof=self.options.degrees_of_freedom)
        )

        if (
            pd.isna(standard_deviation)
            or abs(standard_deviation)
            <= self.options.constant_tolerance
        ):
            self.scale = 1.0
            self.state.constant_feature = True
        else:
            self.scale = standard_deviation

        return {
            "mean": self.mean,
            "standard_deviation": standard_deviation,
            "scale": self.scale,
            "with_mean": self.options.with_mean,
            "with_standard_deviation": (
                self.options.with_standard_deviation
            ),
            "constant_feature": self.state.constant_feature,
        }

    def _transform(self, numeric: pd.Series) -> pd.Series:
        """Apply the fitted standard scaling."""

        transformed = numeric.astype(float)
        if self.options.with_mean:
            transformed = transformed - self._parameter("mean", self.mean)
        if self.options.with_standard_deviation:
            transformed = transformed / self._parameter("scale", self.scale)
        dtype = "float32" if self.options.preserve_float32 else "float64"
        return transformed.astype(dtype)

    def _parameter(self, name: str, current: float | None) -> float:
        """Resolve a fitted parameter, including restored state."""

        if current is not None:
            return current
        value = self.state.learned_parameters.get(name)
        if not isinstance(value, (int, float)):
            raise RuntimeError(
                f"Fitted parameter {name!r} is unavailable "
                f"for feature {self.column!r}."
            )
        return float(value)
