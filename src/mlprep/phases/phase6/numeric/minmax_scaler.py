"""Min-max scaling for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlprep.phases.phase6.numeric.base import (
    BaseNumericTransformer,
)


@dataclass(slots=True)
class MinMaxScalerOptions:
    """Options controlling min-max scaling."""

    output_minimum: float = 0.0
    output_maximum: float = 1.0

    clip_to_output_range: bool = False

    constant_tolerance: float = 1e-12

    preserve_float32: bool = False

    def __post_init__(self) -> None:
        """Validate min-max options."""

        if (
            self.output_minimum
            >= self.output_maximum
        ):
            raise ValueError(
                "output_minimum must be smaller than "
                "output_maximum."
            )

        if self.constant_tolerance < 0.0:
            raise ValueError(
                "constant_tolerance cannot be negative."
            )


class MinMaxScaler(BaseNumericTransformer):
    """Scale values using training minimum and maximum."""

    transformer_name = "minmax_scaler"

    def __init__(
        self,
        *,
        column: str,
        options: MinMaxScalerOptions | None = None,
    ) -> None:
        """Initialize the min-max scaler."""

        super().__init__(
            column=column
        )

        self.options = (
            options
            or MinMaxScalerOptions()
        )

        self.training_minimum: float | None = None
        self.training_maximum: float | None = None
        self.training_range: float | None = None

    def _fit(
        self,
        valid: pd.Series,
    ) -> dict[str, object]:
        """Learn the training minimum and maximum."""

        if valid.empty:
            raise ValueError(
                f"Feature {self.column!r} contains no "
                "valid training values."
            )

        self.training_minimum = float(
            valid.min()
        )

        self.training_maximum = float(
            valid.max()
        )

        raw_range = (
            self.training_maximum
            - self.training_minimum
        )

        if (
            abs(raw_range)
            <= self.options.constant_tolerance
        ):
            self.training_range = 1.0
            self.state.constant_feature = True

        else:
            self.training_range = (
                raw_range
            )

        return {
            "training_minimum": (
                self.training_minimum
            ),
            "training_maximum": (
                self.training_maximum
            ),
            "training_range": (
                self.training_range
            ),
            "raw_training_range": (
                raw_range
            ),
            "output_minimum": (
                self.options.output_minimum
            ),
            "output_maximum": (
                self.options.output_maximum
            ),
            "clip_to_output_range": (
                self.options
                .clip_to_output_range
            ),
            "constant_feature": (
                self.state.constant_feature
            ),
        }

    def _transform(
        self,
        numeric: pd.Series,
    ) -> pd.Series:
        """Apply min-max scaling."""

        training_minimum = self._parameter(
            "training_minimum",
            self.training_minimum,
        )

        training_range = self._parameter(
            "training_range",
            self.training_range,
        )

        output_range = (
            self.options.output_maximum
            - self.options.output_minimum
        )

        transformed = (
            (
                numeric.astype(float)
                - training_minimum
            )
            / training_range
            * output_range
            + self.options.output_minimum
        )

        if self.state.constant_feature:
            transformed = pd.Series(
                self.options.output_minimum,
                index=numeric.index,
                name=numeric.name,
                dtype=float,
            )

        if self.options.clip_to_output_range:
            transformed = transformed.clip(
                lower=self.options.output_minimum,
                upper=self.options.output_maximum,
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
