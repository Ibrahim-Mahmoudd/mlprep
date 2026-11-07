"""Boundary clipping strategy for Phase 5."""

from __future__ import annotations

import pandas as pd

from mlprep.phases.phase5.strategies.base import (
    BaseTreatmentStrategy,
)


class ClipStrategy(BaseTreatmentStrategy):
    """Clip detected values to fitted lower and upper bounds."""

    strategy_name = "clip"

    def _fit(
        self,
        *,
        series: pd.Series,
        outlier_mask: pd.Series,
    ) -> tuple[
        float | None,
        float | None,
    ]:
        """Validate the configured clipping bounds."""

        del series
        del outlier_mask

        if (
            self.lower_bound is None
            and self.upper_bound is None
        ):
            raise ValueError(
                "Clipping requires at least one fitted bound."
            )

        return (
            self.lower_bound,
            self.upper_bound,
        )

    def _transform(
        self,
        *,
        series: pd.Series,
        outlier_mask: pd.Series,
    ) -> pd.Series:
        """Clip only values identified by the detector."""

        transformed = series.copy(
            deep=True
        )

        detected_values = transformed.loc[
            outlier_mask
        ]

        clipped_values = detected_values.clip(
            lower=self.lower_bound,
            upper=self.upper_bound,
        )

        transformed.loc[
            outlier_mask
        ] = clipped_values

        return transformed
