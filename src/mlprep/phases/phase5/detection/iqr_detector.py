"""IQR-based outlier detection for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class IQRDetectorState(BaseModel):
    """Fitted state of an IQR detector."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    fitted: bool = False

    first_quartile: float | None = None
    third_quartile: float | None = None

    interquartile_range: float | None = None

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
class IQRDetectorOptions:
    """Options controlling IQR detection."""

    multiplier: float = 1.5

    minimum_valid_count: int = 4

    allow_zero_iqr: bool = True

    def __post_init__(self) -> None:
        """Validate IQR-detector options."""

        if self.multiplier <= 0.0:
            raise ValueError(
                "multiplier must be positive."
            )

        if self.minimum_valid_count < 2:
            raise ValueError(
                "minimum_valid_count must be at least two."
            )


class IQRDetector:
    """Learn IQR bounds from training data and detect outliers."""

    detector_name = "iqr"

    def __init__(
        self,
        *,
        column: str,
        options: IQRDetectorOptions | None = None,
    ) -> None:
        """Initialize the IQR detector."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        self.column = column

        self.options = (
            options or IQRDetectorOptions()
        )

        self.state = IQRDetectorState(
            column=column
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether the detector is fitted."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
    ) -> IQRDetector:
        """Learn IQR bounds from training values."""

        self._validate_series(
            series
        )

        if self.is_fitted:
            raise RuntimeError(
                "The IQR detector is already fitted."
            )

        numeric = self._numeric(
            series
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

        first_quartile = float(
            valid.quantile(0.25)
        )

        third_quartile = float(
            valid.quantile(0.75)
        )

        interquartile_range = (
            third_quartile - first_quartile
        )

        if not np.isfinite(
            interquartile_range
        ):
            raise ValueError(
                "The fitted IQR is nonfinite."
            )

        if (
            interquartile_range == 0.0
            and not self.options.allow_zero_iqr
        ):
            raise ValueError(
                "The fitted IQR is zero."
            )

        lower_bound = (
            first_quartile
            - self.options.multiplier
            * interquartile_range
        )

        upper_bound = (
            third_quartile
            + self.options.multiplier
            * interquartile_range
        )

        outlier_mask = (
            (valid < lower_bound)
            | (valid > upper_bound)
        )

        self.state = IQRDetectorState(
            column=self.column,
            fitted=True,
            first_quartile=first_quartile,
            third_quartile=third_quartile,
            interquartile_range=(
                interquartile_range
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
            training_outlier_count=int(
                outlier_mask.sum()
            ),
            metadata={
                "multiplier": (
                    self.options.multiplier
                ),
                "zero_iqr": (
                    interquartile_range == 0.0
                ),
            },
        )

        return self

    def detect(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Detect values outside the fitted IQR bounds."""

        self._validate_series(
            series
        )

        self._assert_fitted()

        numeric = self._numeric(
            series
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
        """Fit the detector and detect training outliers."""

        self.fit(
            series
        )

        return self.detect(
            series
        )

    def get_state(
        self,
    ) -> IQRDetectorState:
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
                "The IQR detector is not fitted."
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

    @staticmethod
    def _numeric(
        series: pd.Series,
    ) -> pd.Series:
        """Convert a Series to finite numeric values."""

        return pd.to_numeric(
            series,
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )