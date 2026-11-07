"""Winsorization treatment strategy for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlprep.phases.phase5.strategies.base import (
    BaseTreatmentStrategy,
)


@dataclass(slots=True)
class WinsorizeStrategyOptions:
    """Options controlling winsorization."""

    learn_bounds_from_training: bool = False

    lower_quantile: float = 0.01
    upper_quantile: float = 0.99

    minimum_valid_count: int = 2

    def __post_init__(self) -> None:
        """Validate winsorization options."""

        if not (
            0.0
            <= self.lower_quantile
            < self.upper_quantile
            <= 1.0
        ):
            raise ValueError(
                "Winsorization quantiles must satisfy "
                "0 <= lower < upper <= 1."
            )

        if self.minimum_valid_count < 2:
            raise ValueError(
                "minimum_valid_count must be at least two."
            )


class WinsorizeStrategy(BaseTreatmentStrategy):
    """Replace extreme values with fitted boundary values."""

    strategy_name = "winsorize"

    def __init__(
        self,
        *,
        column: str,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        options: WinsorizeStrategyOptions | None = None,
    ) -> None:
        """Initialize the winsorization strategy."""

        super().__init__(
            column=column,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        self.options = (
            options
            or WinsorizeStrategyOptions()
        )

        self.state.metadata.update(
            {
                "lower_quantile": (
                    self.options.lower_quantile
                ),
                "upper_quantile": (
                    self.options.upper_quantile
                ),
                "learn_bounds_from_training": (
                    self.options
                    .learn_bounds_from_training
                ),
            }
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
        """Resolve winsorization boundaries."""

        del outlier_mask

        if not self.options.learn_bounds_from_training:
            if (
                self.lower_bound is None
                and self.upper_bound is None
            ):
                raise ValueError(
                    "Winsorization requires detector bounds "
                    "when bound learning is disabled."
                )

            return (
                self.lower_bound,
                self.upper_bound,
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
                "Too few valid training values for "
                "winsorization."
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

        return (
            lower_bound,
            upper_bound,
        )

    def _transform(
        self,
        *,
        series: pd.Series,
        outlier_mask: pd.Series,
    ) -> pd.Series:
        """Winsorize detected values."""

        transformed = series.copy(
            deep=True
        )

        numeric = pd.to_numeric(
            transformed.loc[
                outlier_mask
            ],
            errors="coerce",
        )

        winsorized = numeric.clip(
            lower=self.lower_bound,
            upper=self.upper_bound,
        )

        transformed.loc[
            outlier_mask
        ] = winsorized

        return transformed
