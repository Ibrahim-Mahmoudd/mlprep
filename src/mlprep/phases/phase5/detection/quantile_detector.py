"""Quantile-based outlier detection for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class QuantileDetectorState(BaseModel):
    """Fitted state of a quantile detector."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    fitted: bool = False

    lower_quantile: float = Field(
        ge=0.0,
        le=1.0,
    )

    upper_quantile: float = Field(
        ge=0.0,
        le=1.0,
    )

    lower_bound: float | None = None
    upper_bound: float | None = None

    training_valid_count: int = Field(
        default=0,
        ge=0,
    )

    training_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class QuantileDetectorOptions:
    """Options controlling quantile detection."""

    lower_quantile: float = 0.01
    upper_quantile: float = 0.99

    minimum_valid_count: int = 10

    allow_equal_bounds: bool = True

    def __post_init__(self) -> None:
        """Validate quantile-detector options."""

        if not (
            0.0
            <= self.lower_quantile
            < self.upper_quantile
            <= 1.0
        ):
            raise ValueError(
                "Quantiles must satisfy "
                "0 <= lower < upper <= 1."
            )

        if self.minimum_valid_count < 2:
            raise ValueError(
                "minimum_valid_count must be at least two."
            )


class QuantileDetector:
    """Learn quantile bounds from training data."""

    detector_name = "quantile"

    def __init__(
        self,
        *,
        column: str,
        options: QuantileDetectorOptions | None = None,
    ) -> None:
        """Initialize the quantile detector."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        self.column = column

        self.options = (
            options or QuantileDetectorOptions()
        )

        self.state = QuantileDetectorState(
            column=column,
            lower_quantile=(
                self.options.lower_quantile
            ),
            upper_quantile=(
                self.options.upper_quantile
            ),
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether the detector is fitted."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
    ) -> QuantileDetector:
        """Learn lower and upper training quantile bounds."""

        self._validate_series(
            series
        )

        if self.is_fitted:
            raise RuntimeError(
                "The quantile detector is already fitted."
            )

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid = numeric.dropna()

        if (
            len(valid)
            < self.options.minimum_valid_count
        ):
            raise ValueError(
                f"Feature {self.column!r} has too few "
                "valid training values."
            )

        lower_bound = float(
            valid.quantile(
                self.options.lower_quantile
            )
        )

        upper_bound = float(
            valid.quantile(
                self.options.upper_quantile
            )
        )

        if (
            not np.isfinite(lower_bound)
            or not np.isfinite(upper_bound)
        ):
            raise ValueError(
                "Fitted quantile bounds are nonfinite."
            )

        if lower_bound > upper_bound:
            raise ValueError(
                "The fitted lower bound exceeds "
                "the fitted upper bound."
            )

        if (
            lower_bound == upper_bound
            and not self.options.allow_equal_bounds
        ):
            raise ValueError(
                "The fitted quantile bounds are equal."
            )

        outlier_mask = (
            (valid < lower_bound)
            | (valid > upper_bound)
        )

        self.state = QuantileDetectorState(
            column=self.column,
            fitted=True,
            lower_quantile=(
                self.options.lower_quantile
            ),
            upper_quantile=(
                self.options.upper_quantile
            ),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            training_valid_count=len(
                valid
            ),
            training_outlier_count=int(
                outlier_mask.sum()
            ),
            metadata={
                "equal_bounds": (
                    lower_bound == upper_bound
                ),
            },
        )

        return self

    def detect(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Detect values outside the fitted quantile bounds."""

        self._validate_series(
            series
        )

        self._assert_fitted()

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        mask = (
            numeric.notna()
            & (
                (numeric < self.state.lower_bound)
                | (
                    numeric
                    > self.state.upper_bound
                )
            )
        )

        return mask.astype(
            bool
        )

    def fit_detect(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Fit and detect training outliers."""

        self.fit(
            series
        )

        return self.detect(
            series
        )

    def get_state(
        self,
    ) -> QuantileDetectorState:
        """Return a copy of fitted detector state."""

        return self.state.model_copy(
            deep=True
        )

    def _assert_fitted(
        self,
    ) -> None:
        """Raise when the detector is not fitted."""

        if not self.state.fitted:
            raise RuntimeError(
                "The quantile detector is not fitted."
            )

    def _validate_series(
        self,
        series: pd.Series,
    ) -> None:
        """Validate an input Series."""

        if not isinstance(
            series,
            pd.Series,
        ):
            raise TypeError(
                "Detector input must be a pandas Series."
            )

        if (
            series.name is not None
            and str(series.name) != self.column
        ):
            raise ValueError(
                f"Expected feature {self.column!r}, "
                f"received {series.name!r}."
            )
