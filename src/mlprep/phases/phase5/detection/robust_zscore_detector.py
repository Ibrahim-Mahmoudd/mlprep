"""Robust Z-score outlier detection for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

ZeroMADPolicy = Literal[
    "no_outliers",
    "raise",
]


class RobustZScoreDetectorState(BaseModel):
    """Fitted state of a robust Z-score detector."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    fitted: bool = False

    median: float | None = None

    median_absolute_deviation: float | None = None

    threshold: float = Field(
        gt=0.0,
    )

    consistency_constant: float = Field(
        gt=0.0,
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
class RobustZScoreDetectorOptions:
    """Options controlling robust Z-score detection."""

    threshold: float = 3.5

    consistency_constant: float = (
        0.6744897501960817
    )

    minimum_valid_count: int = 5

    zero_mad_policy: ZeroMADPolicy = (
        "no_outliers"
    )

    def __post_init__(self) -> None:
        """Validate robust Z-score options."""

        if self.threshold <= 0.0:
            raise ValueError(
                "threshold must be positive."
            )

        if self.consistency_constant <= 0.0:
            raise ValueError(
                "consistency_constant must be positive."
            )

        if self.minimum_valid_count < 2:
            raise ValueError(
                "minimum_valid_count must be at least two."
            )

        if self.zero_mad_policy not in {
            "no_outliers",
            "raise",
        }:
            raise ValueError(
                "zero_mad_policy must be "
                "'no_outliers' or 'raise'."
            )


class RobustZScoreDetector:
    """Detect extremes using the training median and MAD."""

    detector_name = "robust_zscore"

    def __init__(
        self,
        *,
        column: str,
        options: (
            RobustZScoreDetectorOptions | None
        ) = None,
    ) -> None:
        """Initialize the robust Z-score detector."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        self.column = column

        self.options = (
            options
            or RobustZScoreDetectorOptions()
        )

        self.state = RobustZScoreDetectorState(
            column=column,
            threshold=self.options.threshold,
            consistency_constant=(
                self.options
                .consistency_constant
            ),
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether the detector is fitted."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
    ) -> RobustZScoreDetector:
        """Learn robust statistics from training data."""

        self._validate_series(
            series
        )

        if self.is_fitted:
            raise RuntimeError(
                "The robust Z-score detector is "
                "already fitted."
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

        median = float(
            valid.median()
        )

        median_absolute_deviation = float(
            (
                valid - median
            ).abs().median()
        )

        if not np.isfinite(
            median_absolute_deviation
        ):
            raise ValueError(
                "The fitted MAD is nonfinite."
            )

        zero_mad = (
            median_absolute_deviation == 0.0
        )

        if zero_mad:
            if (
                self.options.zero_mad_policy
                == "raise"
            ):
                raise ValueError(
                    "The fitted MAD is zero."
                )

            lower_bound = median
            upper_bound = median
            training_outlier_count = 0

        else:
            radius = (
                self.options.threshold
                * median_absolute_deviation
                / self.options
                .consistency_constant
            )

            lower_bound = median - radius
            upper_bound = median + radius

            training_outlier_count = int(
                (
                    (valid < lower_bound)
                    | (valid > upper_bound)
                ).sum()
            )

        self.state = RobustZScoreDetectorState(
            column=self.column,
            fitted=True,
            median=median,
            median_absolute_deviation=(
                median_absolute_deviation
            ),
            threshold=self.options.threshold,
            consistency_constant=(
                self.options
                .consistency_constant
            ),
            lower_bound=float(
                lower_bound
            ),
            upper_bound=float(
                upper_bound
            ),
            training_valid_count=len(
                valid
            ),
            training_outlier_count=(
                training_outlier_count
            ),
            metadata={
                "zero_mad": zero_mad,
                "zero_mad_policy": (
                    self.options
                    .zero_mad_policy
                ),
            },
        )

        return self

    def detect(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Detect values outside fitted robust bounds."""

        self._validate_series(
            series
        )

        self._assert_fitted()

        if (
            self.state
            .median_absolute_deviation
            == 0.0
        ):
            return pd.Series(
                False,
                index=series.index,
                dtype=bool,
            )

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

    def robust_zscores(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Calculate robust Z-scores using fitted state."""

        self._assert_fitted()

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        mad = (
            self.state
            .median_absolute_deviation
        )

        if mad == 0.0:
            return pd.Series(
                0.0,
                index=series.index,
                dtype=float,
            )

        return (
            self.options.consistency_constant
            * (
                numeric - self.state.median
            )
            / mad
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
    ) -> RobustZScoreDetectorState:
        """Return a copy of fitted state."""

        return self.state.model_copy(
            deep=True
        )

    def _assert_fitted(
        self,
    ) -> None:
        """Raise when the detector is not fitted."""

        if not self.state.fitted:
            raise RuntimeError(
                "The robust Z-score detector is not fitted."
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