"""Set-missing outlier treatment for Phase 5."""

from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np
import pandas as pd

from mlprep.phases.phase5.strategies.base import (
    BaseTreatmentStrategy,
)


class SetMissingStrategy(BaseTreatmentStrategy):
    """Replace detected values through an explicit fallback policy."""

    strategy_name = "set_missing"

    def __init__(
        self,
        *,
        column: str,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        fallback_value: Any,
    ) -> None:
        """Initialize the set-missing strategy."""

        if fallback_value is None:
            raise ValueError(
                "fallback_value cannot be None."
            )

        try:
            fallback_is_missing = bool(
                pd.isna(
                    fallback_value
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            fallback_is_missing = False

        if fallback_is_missing:
            raise ValueError(
                "fallback_value cannot be missing."
            )

        if isinstance(
            fallback_value,
            Number,
        ) and not np.isfinite(
            float(fallback_value)
        ):
            raise ValueError(
                "A numeric fallback must be finite."
            )

        super().__init__(
            column=column,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        self.fallback_value = fallback_value

        self.state.fallback_value = (
            fallback_value
        )

    def _fit(
        self,
        *,
        series: pd.Series,
        outlier_mask: pd.Series,
    ) -> tuple[
        float | None,
        float | None,
    ]:
        """Validate the explicit fallback."""

        del series
        del outlier_mask

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
        """Replace detected values using the fitted fallback."""

        transformed = series.copy(
            deep=True
        )

        transformed.loc[
            outlier_mask
        ] = pd.NA

        transformed = transformed.fillna(
            self.fallback_value
        )

        return transformed
