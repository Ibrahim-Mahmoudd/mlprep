"""High-correlation feature filtering for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

CorrelationMethod = Literal["pearson", "spearman", "kendall"]


class CorrelatedPairResult(BaseModel):
    """One highly correlated feature pair."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    left_feature: str
    right_feature: str

    correlation: float
    absolute_correlation: float = Field(
        ge=0.0,
        le=1.0,
    )

    retained_feature: str
    removal_candidate: str | None = None

    retention_reason: str

    protected_override: bool = False
    required_override: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class CorrelationFilterResult(BaseModel):
    """Complete high-correlation filter output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    pairs: list[
        CorrelatedPairResult
    ] = Field(default_factory=list)

    removal_candidates: list[str] = Field(
        default_factory=list
    )

    retained_features: list[str] = Field(
        default_factory=list
    )

    overridden_features: list[str] = Field(
        default_factory=list
    )

    correlation_matrix: dict[
        str,
        dict[str, float],
    ] = Field(default_factory=dict)

    fitted_on_training_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class CorrelationFilterOptions:
    """Options controlling correlation filtering."""

    enabled: bool = True

    maximum_absolute_correlation: float = 0.95

    method: CorrelationMethod = "pearson"

    minimum_periods: int = 2

    include_correlation_matrix: bool = False

    def __post_init__(self) -> None:
        """Validate correlation-filter options."""

        if not 0.0 <= (
            self.maximum_absolute_correlation
        ) <= 1.0:
            raise ValueError(
                "maximum_absolute_correlation must be "
                "between zero and one."
            )

        if self.method not in {
            "pearson",
            "spearman",
            "kendall",
        }:
            raise ValueError(
                "method must be 'pearson', "
                "'spearman', or 'kendall'."
            )

        if self.minimum_periods < 2:
            raise ValueError(
                "minimum_periods must be at least two."
            )


class CorrelationFilter:
    """Detect and resolve highly correlated feature pairs."""

    component_name = "correlation_filter"

    def __init__(
        self,
        options: CorrelationFilterOptions | None = None,
    ) -> None:
        """Initialize the filter."""

        self.options = (
            options
            or CorrelationFilterOptions()
        )

        self.result: CorrelationFilterResult | None = None

    def fit(
        self,
        *,
        frame: pd.DataFrame,
        candidate_features: list[str] | None = None,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
        feature_scores: dict[str, float] | None = None,
        variance_scores: dict[str, float] | None = None,
        source_feature_order: list[str] | None = None,
    ) -> CorrelationFilter:
        """Calculate training correlations."""

        candidates = (
            [
                str(column)
                for column in frame.columns
            ]
            if candidate_features is None
            else list(
                candidate_features
            )
        )

        self._validate_features(
            frame=frame,
            features=candidates,
        )

        required = set(
            required_features or []
        )

        protected = set(
            protected_features or []
        )

        scores = dict(
            feature_scores or {}
        )

        variances = dict(
            variance_scores or {}
        )

        source_order = (
            [
                str(column)
                for column in frame.columns
            ]
            if source_feature_order is None
            else list(
                source_feature_order
            )
        )

        order_index = {
            feature: index
            for index, feature
            in enumerate(
                source_order
            )
        }

        if not self.options.enabled:
            self.result = CorrelationFilterResult(
                retained_features=(
                    candidates
                ),
                metadata={
                    "enabled": False,
                },
            )

            return self

        correlation_matrix = frame.loc[
            :,
            candidates,
        ].corr(
            method=self.options.method,
            min_periods=(
                self.options.minimum_periods
            ),
        )

        active_features = list(
            candidates
        )

        removed: set[str] = set()

        pairs: list[
            CorrelatedPairResult
        ] = []

        overridden_features: list[str] = []

        for left_index, left_feature in enumerate(
            candidates
        ):
            if left_feature in removed:
                continue

            for right_feature in candidates[
                left_index + 1:
            ]:
                if right_feature in removed:
                    continue

                raw_correlation = correlation_matrix.at[
                    left_feature,
                    right_feature,
                ]

                if not isinstance(
                    raw_correlation,
                    (int, float, np.integer, np.floating),
                ):
                    continue

                correlation = float(
                    raw_correlation
                )
                if not np.isfinite(correlation):
                    continue

                absolute_correlation = abs(
                    correlation
                )

                if (
                    absolute_correlation
                    <= self.options
                    .maximum_absolute_correlation
                ):
                    continue

                (
                    retained,
                    removal_candidate,
                    reason,
                    override,
                ) = self._resolve_pair(
                    left_feature=left_feature,
                    right_feature=right_feature,
                    required=required,
                    protected=protected,
                    scores=scores,
                    variances=variances,
                    frame=frame,
                    order_index=order_index,
                )

                if removal_candidate is not None:
                    removed.add(
                        removal_candidate
                    )

                    if removal_candidate in active_features:
                        active_features.remove(
                            removal_candidate
                        )

                if override:
                    overridden_features.extend(
                        [
                            left_feature,
                            right_feature,
                        ]
                    )

                pairs.append(
                    CorrelatedPairResult(
                        left_feature=left_feature,
                        right_feature=right_feature,
                        correlation=correlation,
                        absolute_correlation=(
                            absolute_correlation
                        ),
                        retained_feature=retained,
                        removal_candidate=(
                            removal_candidate
                        ),
                        retention_reason=reason,
                        protected_override=bool(
                            left_feature in protected
                            and right_feature in protected
                        ),
                        required_override=bool(
                            left_feature in required
                            and right_feature in required
                        ),
                    )
                )

        matrix_payload: dict[
            str,
            dict[str, float],
        ] = {}

        if self.options.include_correlation_matrix:
            matrix_payload = {
                str(column): {
                    str(index): float(value)
                    for index, value
                    in values.items()
                    if not pd.isna(
                        value
                    )
                }
                for column, values
                in correlation_matrix
                .to_dict()
                .items()
            }

        self.result = CorrelationFilterResult(
            pairs=pairs,
            removal_candidates=[
                feature
                for feature in candidates
                if feature in removed
            ],
            retained_features=(
                active_features
            ),
            overridden_features=list(
                dict.fromkeys(
                    overridden_features
                )
            ),
            correlation_matrix=(
                matrix_payload
            ),
            fitted_on_training_only=True,
            metadata={
                "enabled": True,
                "candidate_feature_count": len(
                    candidates
                ),
                "high_correlation_pair_count": len(
                    pairs
                ),
                "removal_candidate_count": len(
                    removed
                ),
            },
        )

        return self

    def get_result(
        self,
    ) -> CorrelationFilterResult:
        """Return the fitted result."""

        if self.result is None:
            raise RuntimeError(
                "The correlation filter is not fitted."
            )

        return self.result.model_copy(
            deep=True
        )

    @staticmethod
    def _resolve_pair(
        *,
        left_feature: str,
        right_feature: str,
        required: set[str],
        protected: set[str],
        scores: dict[str, float],
        variances: dict[str, float],
        frame: pd.DataFrame,
        order_index: dict[str, int],
    ) -> tuple[
        str,
        str | None,
        str,
        bool,
    ]:
        """Resolve one correlated pair deterministically."""

        left_required = (
            left_feature in required
        )

        right_required = (
            right_feature in required
        )

        if left_required and right_required:
            return (
                left_feature,
                None,
                (
                    "Both correlated features are required; "
                    "neither feature was removed."
                ),
                True,
            )

        if left_required:
            return (
                left_feature,
                right_feature,
                "The left feature is required.",
                False,
            )

        if right_required:
            return (
                right_feature,
                left_feature,
                "The right feature is required.",
                False,
            )

        left_protected = (
            left_feature in protected
        )

        right_protected = (
            right_feature in protected
        )

        if left_protected and right_protected:
            return (
                left_feature,
                None,
                (
                    "Both correlated features are protected; "
                    "neither feature was removed."
                ),
                True,
            )

        if left_protected:
            return (
                left_feature,
                right_feature,
                "The left feature is protected.",
                False,
            )

        if right_protected:
            return (
                right_feature,
                left_feature,
                "The right feature is protected.",
                False,
            )

        left_priority = (
            float(
                scores.get(
                    left_feature,
                    0.0,
                )
            ),
            -int(
                frame[
                    left_feature
                ].isna().sum()
            ),
            float(
                variances.get(
                    left_feature,
                    0.0,
                )
            ),
            -order_index.get(
                left_feature,
                len(
                    order_index
                ),
            ),
        )

        right_priority = (
            float(
                scores.get(
                    right_feature,
                    0.0,
                )
            ),
            -int(
                frame[
                    right_feature
                ].isna().sum()
            ),
            float(
                variances.get(
                    right_feature,
                    0.0,
                )
            ),
            -order_index.get(
                right_feature,
                len(
                    order_index
                ),
            ),
        )

        if left_priority >= right_priority:
            return (
                left_feature,
                right_feature,
                (
                    "The left feature has the higher "
                    "deterministic retention priority."
                ),
                False,
            )

        return (
            right_feature,
            left_feature,
            (
                "The right feature has the higher "
                "deterministic retention priority."
            ),
            False,
        )

    @staticmethod
    def _validate_features(
        *,
        frame: pd.DataFrame,
        features: list[str],
    ) -> None:
        """Validate correlation-filter features."""

        missing = [
            feature
            for feature in features
            if feature not in frame.columns
        ]

        if missing:
            raise KeyError(
                "Correlation-filter features are missing: "
                f"{missing}"
            )

        nonnumeric = [
            feature
            for feature in features
            if not pd.api.types.is_numeric_dtype(
                frame[
                    feature
                ].dtype
            )
        ]

        if nonnumeric:
            raise TypeError(
                "Correlation filtering requires numeric "
                f"features: {nonnumeric}"
            )
