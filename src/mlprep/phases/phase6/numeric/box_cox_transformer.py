"""Box-Cox numeric transformations for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlprep.phases.phase6.numeric.base import (
    BaseNumericTransformer,
)


@dataclass(slots=True)
class BoxCoxTransformerOptions:
    """Options controlling Box-Cox lambda fitting."""

    lambda_minimum: float = -5.0
    lambda_maximum: float = 5.0
    search_steps: int = 201
    constant_tolerance: float = 1e-12
    preserve_float32: bool = False

    def __post_init__(self) -> None:
        """Validate Box-Cox options."""

        if self.lambda_minimum >= self.lambda_maximum:
            raise ValueError(
                "lambda_minimum must be smaller than "
                "lambda_maximum."
            )
        if self.search_steps < 3:
            raise ValueError("search_steps must be at least three.")
        if self.constant_tolerance < 0.0:
            raise ValueError("constant_tolerance cannot be negative.")


class BoxCoxTransformer(BaseNumericTransformer):
    """Fit Box-Cox on strictly positive training values."""

    transformer_name = "box_cox"

    def __init__(
        self,
        *,
        column: str,
        options: BoxCoxTransformerOptions | None = None,
    ) -> None:
        """Initialize the Box-Cox transformer."""

        super().__init__(column=column)
        self.options = options or BoxCoxTransformerOptions()
        self.lambda_value: float | None = None

    def _fit(self, valid: pd.Series) -> dict[str, object]:
        """Fit lambda using positive training values only."""

        if valid.empty:
            raise ValueError(
                f"Feature {self.column!r} contains no "
                "valid numeric training values."
            )
        if bool((valid <= 0.0).any()):
            raise ValueError(
                f"Box-Cox requires strictly positive values "
                f"for feature {self.column!r}."
            )

        values = valid.to_numpy(dtype=float)
        variance = float(np.var(values))
        if variance <= self.options.constant_tolerance:
            self.lambda_value = 1.0
            self.state.constant_feature = True
            return {
                "lambda": self.lambda_value,
                "constant_feature": True,
            }

        log_values = np.log(values)
        best_lambda: float | None = None
        best_score = float("-inf")
        for candidate in np.linspace(
            self.options.lambda_minimum,
            self.options.lambda_maximum,
            self.options.search_steps,
        ):
            candidate_lambda = float(candidate)
            transformed = self._box_cox(values, candidate_lambda)
            transformed_variance = float(np.var(transformed))
            if (
                transformed_variance <= self.options.constant_tolerance
                or not np.isfinite(transformed_variance)
            ):
                continue
            score = (
                -0.5 * len(values) * np.log(transformed_variance)
                + (candidate_lambda - 1.0) * float(log_values.sum())
            )
            if np.isfinite(score) and score > best_score:
                best_lambda = candidate_lambda
                best_score = float(score)

        if best_lambda is None:
            raise ValueError(
                f"Could not fit a finite Box-Cox transformation "
                f"for feature {self.column!r}."
            )

        self.lambda_value = best_lambda
        self.state.metadata["search_score"] = best_score
        return {
            "lambda": best_lambda,
            "constant_feature": False,
            "search_steps": self.options.search_steps,
        }

    def _transform(self, numeric: pd.Series) -> pd.Series:
        """Apply the fitted Box-Cox transformation."""

        if bool((numeric.dropna() <= 0.0).any()):
            raise ValueError(
                f"Box-Cox requires strictly positive values "
                f"for feature {self.column!r}."
            )
        if self.lambda_value is None:
            learned_lambda = self.state.learned_parameters.get("lambda")
            if learned_lambda is None:
                raise RuntimeError(
                    "The fitted Box-Cox lambda is unavailable."
                )
            self.lambda_value = float(learned_lambda)

        transformed = self._box_cox(
            numeric.to_numpy(dtype=float),
            self.lambda_value,
        )
        result = pd.Series(
            transformed,
            index=numeric.index,
            name=numeric.name,
        )
        dtype = "float32" if self.options.preserve_float32 else "float64"
        return result.astype(dtype)

    @staticmethod
    def _box_cox(values: np.ndarray, lambda_value: float) -> np.ndarray:
        """Apply the Box-Cox formula to positive values."""

        with np.errstate(over="ignore", invalid="ignore"):
            if abs(lambda_value) < 1e-12:
                return np.log(values)
            return np.expm1(lambda_value * np.log(values)) / lambda_value
