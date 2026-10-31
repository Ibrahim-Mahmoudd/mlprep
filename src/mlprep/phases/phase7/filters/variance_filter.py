"""Low-variance feature filtering for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class VarianceFeatureResult(BaseModel):
    """Variance-filter result for one feature."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    column: str
    variance: float = Field(default=0.0, ge=0.0)
    threshold: float = Field(default=0.0, ge=0.0)
    unique_count: int = Field(default=0, ge=0)
    binary_feature: bool = False
    low_variance: bool = False
    candidate_for_removal: bool = False
    protected_override: bool = False
    required_override: bool = False
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VarianceFilterResult(BaseModel):
    """Complete variance-filter output."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    features: dict[str, VarianceFeatureResult] = Field(default_factory=dict)
    variance_scores: dict[str, float] = Field(default_factory=dict)
    low_variance_features: list[str] = Field(default_factory=list)
    removal_candidates: list[str] = Field(default_factory=list)
    retained_features: list[str] = Field(default_factory=list)
    overridden_features: list[str] = Field(default_factory=list)
    fitted_on_training_only: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class VarianceFilterOptions:
    """Options controlling variance filtering."""

    enabled: bool = True
    minimum_variance: float = 0.0001
    binary_minimum_variance: float = 0.0
    use_population_variance: bool = True
    require_finite_values: bool = True
    allow_missing_values: bool = False

    def __post_init__(self) -> None:
        """Validate variance-filter options."""

        if self.minimum_variance < 0.0:
            raise ValueError("minimum_variance cannot be negative.")
        if self.binary_minimum_variance < 0.0:
            raise ValueError("binary_minimum_variance cannot be negative.")


class VarianceFilter:
    """Detect low-variance numeric training features."""

    component_name = "variance_filter"

    def __init__(
        self,
        options: VarianceFilterOptions | None = None,
    ) -> None:
        """Initialize the filter."""

        self.options = options or VarianceFilterOptions()
        self.result: VarianceFilterResult | None = None

    def fit(
        self,
        *,
        frame: pd.DataFrame,
        candidate_features: list[str] | None = None,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
    ) -> VarianceFilter:
        """Calculate variance from training features only."""

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame.")
        candidates = list(
            candidate_features
            if candidate_features is not None
            else frame.columns.map(str)
        )
        missing = [column for column in candidates if column not in frame.columns]
        if missing:
            raise KeyError(f"Candidate columns are missing: {missing}")
        required = set(required_features or [])
        protected = set(protected_features or [])

        feature_results: dict[str, VarianceFeatureResult] = {}
        scores: dict[str, float] = {}
        low_variance: list[str] = []
        removal_candidates: list[str] = []
        retained: list[str] = []
        overridden: list[str] = []

        for column in candidates:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            nonfinite = numeric.notna() & ~np.isfinite(numeric.astype(float))
            if self.options.require_finite_values and bool(nonfinite.any()):
                raise ValueError(
                    f"Feature {column!r} contains nonfinite values."
                )
            missing_mask = numeric.isna() | nonfinite
            if (
                not self.options.allow_missing_values
                and bool(missing_mask.any())
            ):
                raise ValueError(
                    f"Feature {column!r} contains missing or nonnumeric values."
                )
            values = numeric.loc[~missing_mask].astype(float)
            variance = (
                float(
                    values.var(
                        ddof=0 if self.options.use_population_variance else 1
                    )
                )
                if len(values) > int(not self.options.use_population_variance)
                else 0.0
            )
            if not np.isfinite(variance):
                variance = 0.0
            unique_count = int(values.nunique(dropna=True))
            binary = unique_count <= 2
            threshold = (
                self.options.binary_minimum_variance
                if binary
                else self.options.minimum_variance
            )
            is_low = variance <= threshold
            required_override = column in required and is_low
            protected_override = column in protected and is_low
            removable = (
                is_low
                and not required_override
                and not protected_override
                and self.options.enabled
            )
            reason = (
                "low_variance"
                if is_low
                else "variance_above_threshold"
            )
            feature_results[column] = VarianceFeatureResult(
                column=column,
                variance=variance,
                threshold=threshold,
                unique_count=unique_count,
                binary_feature=binary,
                low_variance=is_low,
                candidate_for_removal=removable,
                protected_override=protected_override,
                required_override=required_override,
                reason=reason,
                metadata={"valid_count": len(values)},
            )
            scores[column] = variance
            if is_low:
                low_variance.append(column)
            if removable:
                removal_candidates.append(column)
            else:
                retained.append(column)
            if required_override or protected_override:
                overridden.append(column)

        self.result = VarianceFilterResult(
            features=feature_results,
            variance_scores=scores,
            low_variance_features=low_variance,
            removal_candidates=removal_candidates,
            retained_features=retained,
            overridden_features=overridden,
            fitted_on_training_only=True,
            metadata={"enabled": self.options.enabled},
        )
        return self

    def get_result(self) -> VarianceFilterResult:
        """Return the fitted variance-filter result."""

        if self.result is None:
            raise RuntimeError("VarianceFilter.fit() must be called first.")
        return self.result.model_copy(deep=True)
