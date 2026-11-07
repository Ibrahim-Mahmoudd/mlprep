"""Robust numeric scaling for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlprep.phases.phase6.numeric.base import (
    BaseNumericTransformer,
)


@dataclass(slots=True)
class RobustScalerOptions:
    """Options controlling robust scaling."""

    lower_quantile: float = 0.25
    upper_quantile: float = 0.75

    with_centering: bool = True
    with_scaling: bool = True

    constant_tolerance: float = 1e-12

    preserve_float32: bool = False

    def __post_init__(self) -> None:
        """Validate robust-scaler options."""

        if not (
            0.0
            <= self.lower_quantile
            < self.upper_quantile
            <= 1.0
        ):
            raise ValueError(
                "Robust-scaler quantiles must satisfy "
                "0 <= lower < upper <= 1."
            )

        if self.constant_tolerance < 0.0:
            raise ValueError(
                "constant_tolerance cannot be negative."
            )


class RobustScaler(BaseNumericTransformer):
    """Scale values using training median and quantile range."""

    transformer_name = "robust_scaler"

    def __init__(
        self,
        *,
        column: str,
        options: RobustScalerOptions | None = None,
    ) -> None:
        """Initialize the robust scaler."""

        super().__init__(
            column=column
        )

        self.options = (
            options
            or RobustScalerOptions()
        )

        self.center: float | None = None
        self.scale: float | None = None

    def _fit(
        self,
        valid: pd.Series,
    ) -> dict[str, object]:
        """Learn training median and quantile range."""

        if valid.empty:
            raise ValueError(
                f"Feature {self.column!r} contains no "
                "valid training values."
            )

        self.center = float(
            valid.median()
        )

        lower_value = float(
            valid.quantile(
                self.options.lower_quantile
            )
        )

        upper_value = float(
            valid.quantile(
                self.options.upper_quantile
            )
        )

        quantile_range = (
            upper_value - lower_value
        )

        if (
            abs(quantile_range)
            <= self.options.constant_tolerance
        ):
            self.scale = 1.0
            self.state.constant_feature = True

        else:
            self.scale = (
                quantile_range
            )

        return {
            "center": self.center,
            "lower_quantile_value": (
                lower_value
            ),
            "upper_quantile_value": (
                upper_value
            ),
            "quantile_range": (
                quantile_range
            ),
            "scale": self.scale,
            "with_centering": (
                self.options.with_centering
            ),
            "with_scaling": (
                self.options.with_scaling
            ),
            "constant_feature": (
                self.state.constant_feature
            ),
        }

    def _transform(
        self,
        numeric: pd.Series,
    ) -> pd.Series:
        """Apply robust scaling."""

        center = self._parameter(
            "center",
            self.center,
        )

        scale = self._parameter(
            "scale",
            self.scale,
        )

        transformed = numeric.astype(
            float
        )

        if self.options.with_centering:
            transformed = (
                transformed - center
            )

        if self.options.with_scaling:
            transformed = (
                transformed / scale
            )

        if self.options.preserve_float32:
            return transformed.astype(
                "float32"
            )

        return transformed.astype(
            "float64"
        )

    def _parameter(
        self,
        name: str,
        current: float | None,
    ) -> float:
        """Resolve one fitted parameter."""

        if current is not None:
            return current

        value = (
            self.state
            .learned_parameters
            .get(name)
        )

        if value is None:
            raise RuntimeError(
                f"Fitted parameter {name!r} "
                "is unavailable."
            )

        return float(value)
