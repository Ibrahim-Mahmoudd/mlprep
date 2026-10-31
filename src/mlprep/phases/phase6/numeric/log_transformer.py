"""Logarithmic numeric transformations for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from mlprep.phases.phase6.numeric.base import (
    BaseNumericTransformer,
)


LogMethod = Literal[
    "log1p",
    "signed_log1p",
]


@dataclass(slots=True)
class LogTransformerOptions:
    """Options controlling logarithmic transformations."""

    method: LogMethod = "log1p"

    domain_tolerance: float = 0.0

    preserve_float32: bool = False

    def __post_init__(self) -> None:
        """Validate logarithmic options."""

        if self.domain_tolerance < 0.0:
            raise ValueError(
                "domain_tolerance cannot be negative."
            )


class LogTransformer(BaseNumericTransformer):
    """Apply log1p or signed-log1p transformation."""

    transformer_name = "log"

    def __init__(
        self,
        *,
        column: str,
        options: LogTransformerOptions | None = None,
    ) -> None:
        """Initialize the log transformer."""

        super().__init__(
            column=column
        )

        self.options = (
            options
            or LogTransformerOptions()
        )

        self.state.metadata.update(
            {
                "method": self.options.method,
                "domain_tolerance": (
                    self.options.domain_tolerance
                ),
            }
        )

    def _fit(
        self,
        valid: pd.Series,
    ) -> dict[str, object]:
        """Validate training values and store domain metadata."""

        if valid.empty:
            raise ValueError(
                f"Feature {self.column!r} contains no "
                "valid numeric training values."
            )

        minimum = float(
            valid.min()
        )

        maximum = float(
            valid.max()
        )

        if self.options.method == "log1p":
            invalid_mask = (
                valid
                < (
                    -1.0
                    - self.options.domain_tolerance
                )
            )

            if bool(
                invalid_mask.any()
            ):
                raise ValueError(
                    f"log1p cannot transform feature "
                    f"{self.column!r} because training values "
                    "below -1 were found."
                )

        return {
            "method": self.options.method,
            "training_minimum": minimum,
            "training_maximum": maximum,
        }

    def _transform(
        self,
        numeric: pd.Series,
    ) -> pd.Series:
        """Apply the configured logarithmic transformation."""

        values = numeric.astype(
            float
        )

        if self.options.method == "log1p":
            invalid_mask = (
                values
                < (
                    -1.0
                    - self.options.domain_tolerance
                )
            )

            if bool(
                invalid_mask.fillna(
                    False
                ).any()
            ):
                raise ValueError(
                    f"log1p received values below -1 for "
                    f"feature {self.column!r}."
                )

            transformed = np.log1p(
                values
            )

        elif (
            self.options.method
            == "signed_log1p"
        ):
            transformed = (
                np.sign(values)
                * np.log1p(
                    np.abs(values)
                )
            )

        else:
            raise ValueError(
                f"Unsupported log method: "
                f"{self.options.method!r}"
            )

        result = pd.Series(
            transformed,
            index=numeric.index,
            name=numeric.name,
        )

        if self.options.preserve_float32:
            return result.astype(
                "float32"
            )

        return result.astype(
            "float64"
        )