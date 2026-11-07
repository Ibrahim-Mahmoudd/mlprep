"""Power transformations for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlprep.phases.phase6.numeric.base import (
    BaseNumericTransformer,
)


@dataclass(slots=True)
class PowerTransformerOptions:
    """Options controlling Yeo-Johnson transformation."""

    lambda_minimum: float = -5.0
    lambda_maximum: float = 5.0

    search_steps: int = 201

    constant_tolerance: float = 1e-12

    preserve_float32: bool = False

    def __post_init__(self) -> None:
        """Validate power-transformer options."""

        if (
            self.lambda_minimum
            >= self.lambda_maximum
        ):
            raise ValueError(
                "lambda_minimum must be smaller than "
                "lambda_maximum."
            )

        if self.search_steps < 3:
            raise ValueError(
                "search_steps must be at least three."
            )

        if self.constant_tolerance < 0.0:
            raise ValueError(
                "constant_tolerance cannot be negative."
            )


class PowerTransformer(BaseNumericTransformer):
    """Apply a fitted Yeo-Johnson transformation."""

    transformer_name = "yeo_johnson"

    def __init__(
        self,
        *,
        column: str,
        options: PowerTransformerOptions | None = None,
    ) -> None:
        """Initialize the power transformer."""

        super().__init__(
            column=column
        )

        self.options = (
            options
            or PowerTransformerOptions()
        )

        self.lambda_value: float | None = None

    def _fit(
        self,
        valid: pd.Series,
    ) -> dict[str, object]:
        """Learn the Yeo-Johnson lambda on training data."""

        if valid.empty:
            raise ValueError(
                f"Feature {self.column!r} contains no "
                "valid numeric training values."
            )

        values = valid.to_numpy(
            dtype=float
        )

        variance = float(
            np.var(values)
        )

        if (
            variance
            <= self.options.constant_tolerance
        ):
            self.lambda_value = 1.0

            self.state.constant_feature = True

            return {
                "lambda": self.lambda_value,
                "constant_feature": True,
            }

        candidate_lambdas = np.linspace(
            self.options.lambda_minimum,
            self.options.lambda_maximum,
            self.options.search_steps,
        )

        best_lambda = 1.0
        best_score = float(
            "-inf"
        )

        for candidate in candidate_lambdas:
            transformed = self._yeo_johnson(
                values,
                float(candidate),
            )

            transformed_variance = float(
                np.var(
                    transformed
                )
            )

            if (
                transformed_variance
                <= self.options.constant_tolerance
                or not np.isfinite(
                    transformed_variance
                )
            ):
                continue

            log_likelihood = (
                -0.5
                * len(values)
                * np.log(
                    transformed_variance
                )
                + (
                    float(candidate)
                    - 1.0
                )
                * np.sum(
                    np.sign(values)
                    * np.log1p(
                        np.abs(values)
                    )
                )
            )

            if (
                np.isfinite(
                    log_likelihood
                )
                and log_likelihood
                > best_score
            ):
                best_score = float(
                    log_likelihood
                )

                best_lambda = float(
                    candidate
                )

        self.lambda_value = best_lambda

        self.state.metadata[
            "search_score"
        ] = best_score

        return {
            "lambda": best_lambda,
            "constant_feature": False,
            "search_steps": (
                self.options.search_steps
            ),
        }

    def _transform(
        self,
        numeric: pd.Series,
    ) -> pd.Series:
        """Apply the fitted Yeo-Johnson transformation."""

        if self.lambda_value is None:
            learned_lambda = (
                self.state
                .learned_parameters
                .get(
                    "lambda"
                )
            )

            if learned_lambda is None:
                raise RuntimeError(
                    "The fitted Yeo-Johnson lambda "
                    "is unavailable."
                )

            self.lambda_value = float(
                learned_lambda
            )

        values = numeric.to_numpy(
            dtype=float
        )

        transformed = self._yeo_johnson(
            values,
            self.lambda_value,
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

    @staticmethod
    def _yeo_johnson(
        values: np.ndarray,
        lambda_value: float,
    ) -> np.ndarray:
        """Apply the Yeo-Johnson formula."""

        result = np.empty_like(
            values,
            dtype=float,
        )

        positive_mask = (
            values >= 0.0
        )

        negative_mask = (
            ~positive_mask
        )

        positive_values = values[
            positive_mask
        ]

        negative_values = values[
            negative_mask
        ]

        if abs(lambda_value) < 1e-12:
            result[
                positive_mask
            ] = np.log1p(
                positive_values
            )

        else:
            result[
                positive_mask
            ] = (
                np.power(
                    positive_values + 1.0,
                    lambda_value,
                )
                - 1.0
            ) / lambda_value

        if abs(
            lambda_value - 2.0
        ) < 1e-12:
            result[
                negative_mask
            ] = -np.log1p(
                -negative_values
            )

        else:
            result[
                negative_mask
            ] = -(
                (
                    np.power(
                        1.0 - negative_values,
                        2.0 - lambda_value,
                    )
                    - 1.0
                )
                / (
                    2.0
                    - lambda_value
                )
            )

        return result
