"""Report-only outlier strategy for Phase 5."""

from __future__ import annotations

import pandas as pd

from mlprep.phases.phase5.strategies.base import (
    BaseTreatmentStrategy,
)


class ReportOnlyStrategy(BaseTreatmentStrategy):
    """Report detected outliers without changing values."""

    strategy_name = "report_only"

    def _fit(
        self,
        *,
        series: pd.Series,
        outlier_mask: pd.Series,
    ) -> tuple[
        float | None,
        float | None,
    ]:
        """Store detector bounds without learning treatment state."""

        del series
        del outlier_mask

        self.state.metadata[
            "values_modified"
        ] = False

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
        """Return an unchanged copy of the feature."""

        del outlier_mask

        return series.copy(
            deep=True
        )