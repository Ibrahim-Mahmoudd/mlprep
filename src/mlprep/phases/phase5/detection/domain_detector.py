"""Explicit domain-bound detection for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DomainDetectorState(BaseModel):
    """Fitted state of a domain detector."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    fitted: bool = False

    lower_bound: float | None = None
    upper_bound: float | None = None

    training_valid_count: int = Field(
        default=0,
        ge=0,
    )

    training_violation_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class DomainDetectorOptions:
    """Options controlling explicit domain detection."""

    minimum: float | None = None
    maximum: float | None = None

    inclusive_minimum: bool = True
    inclusive_maximum: bool = True

    treat_nonfinite_as_violation: bool = True

    def __post_init__(self) -> None:
        """Validate domain-detector options."""

        if (
            self.minimum is None
            and self.maximum is None
        ):
            raise ValueError(
                "At least one domain bound must be configured."
            )

        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(
                "minimum cannot exceed maximum."
            )


class DomainDetector:
    """Detect values outside explicit permitted bounds."""

    detector_name = "domain"

    def __init__(
        self,
        *,
        column: str,
        options: DomainDetectorOptions,
    ) -> None:
        """Initialize the domain detector."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        self.column = column
        self.options = options

        self.state = DomainDetectorState(
            column=column,
            lower_bound=options.minimum,
            upper_bound=options.maximum,
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether the detector is initialized."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
    ) -> DomainDetector:
        """Validate training data against configured bounds."""

        self._validate_series(
            series
        )

        if self.is_fitted:
            raise RuntimeError(
                "The domain detector is already fitted."
            )

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        violation_mask = self._mask(
            numeric
        )

        self.state = DomainDetectorState(
            column=self.column,
            fitted=True,
            lower_bound=self.options.minimum,
            upper_bound=self.options.maximum,
            training_valid_count=int(
                numeric.notna().sum()
            ),
            training_violation_count=int(
                violation_mask.sum()
            ),
            metadata={
                "inclusive_minimum": (
                    self.options
                    .inclusive_minimum
                ),
                "inclusive_maximum": (
                    self.options
                    .inclusive_maximum
                ),
                "treat_nonfinite_as_violation": (
                    self.options
                    .treat_nonfinite_as_violation
                ),
            },
        )

        return self

    def detect(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Detect values outside the configured domain."""

        self._validate_series(
            series
        )

        self._assert_fitted()

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        return self._mask(
            numeric
        )

    def fit_detect(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Initialize and detect training violations."""

        self.fit(
            series
        )

        return self.detect(
            series
        )

    def get_state(
        self,
    ) -> DomainDetectorState:
        """Return a copy of detector state."""

        return self.state.model_copy(
            deep=True
        )

    def _mask(
        self,
        numeric: pd.Series,
    ) -> pd.Series:
        """Build a domain-violation mask."""

        mask = pd.Series(
            False,
            index=numeric.index,
            dtype=bool,
        )

        if self.options.minimum is not None:
            if self.options.inclusive_minimum:
                mask |= (
                    numeric
                    < self.options.minimum
                )
            else:
                mask |= (
                    numeric
                    <= self.options.minimum
                )

        if self.options.maximum is not None:
            if self.options.inclusive_maximum:
                mask |= (
                    numeric
                    > self.options.maximum
                )
            else:
                mask |= (
                    numeric
                    >= self.options.maximum
                )

        if (
            self.options
            .treat_nonfinite_as_violation
        ):
            mask |= (
                numeric.notna()
                & ~np.isfinite(
                    numeric.astype(float)
                )
            )

        return mask.fillna(
            False
        ).astype(
            bool
        )

    def _assert_fitted(
        self,
    ) -> None:
        """Raise when the detector is not fitted."""

        if not self.state.fitted:
            raise RuntimeError(
                "The domain detector is not fitted."
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