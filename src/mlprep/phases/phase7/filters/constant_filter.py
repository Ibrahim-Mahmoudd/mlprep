"""Constant-feature filtering for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class ConstantFeatureResult(BaseModel):
    """Constant-filter result for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    unique_count: int = Field(
        default=0,
        ge=0,
    )

    missing_count: int = Field(
        default=0,
        ge=0,
    )

    constant: bool = False

    candidate_for_removal: bool = False
    protected_override: bool = False
    required_override: bool = False

    reason: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ConstantFilterResult(BaseModel):
    """Complete constant-feature filter output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    features: dict[
        str,
        ConstantFeatureResult,
    ] = Field(default_factory=dict)

    constant_features: list[str] = Field(
        default_factory=list
    )

    removal_candidates: list[str] = Field(
        default_factory=list
    )

    retained_features: list[str] = Field(
        default_factory=list
    )

    overridden_features: list[str] = Field(
        default_factory=list
    )

    fitted_on_training_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class ConstantFilterOptions:
    """Options controlling constant-feature filtering."""

    enabled: bool = True

    minimum_unique_count: int = 2

    count_missing_as_category: bool = False

    def __post_init__(self) -> None:
        """Validate constant-filter options."""

        if self.minimum_unique_count < 1:
            raise ValueError(
                "minimum_unique_count must be positive."
            )


class ConstantFilter:
    """Detect constant training features."""

    component_name = "constant_filter"

    def __init__(
        self,
        options: ConstantFilterOptions | None = None,
    ) -> None:
        """Initialize the filter."""

        self.options = (
            options
            or ConstantFilterOptions()
        )

        self.result: ConstantFilterResult | None = None

    def fit(
        self,
        *,
        frame: pd.DataFrame,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
    ) -> ConstantFilter:
        """Detect constant features using training data."""

        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                "frame must be a pandas DataFrame."
            )

        if frame.columns.duplicated().any():
            raise ValueError(
                "Constant filtering requires unique "
                "feature names."
            )

        required = set(
            required_features or []
        )

        protected = set(
            protected_features or []
        )

        feature_results: dict[
            str,
            ConstantFeatureResult,
        ] = {}

        constant_features: list[str] = []
        removal_candidates: list[str] = []
        retained_features: list[str] = []
        overridden_features: list[str] = []

        if not self.options.enabled:
            self.result = ConstantFilterResult(
                retained_features=[
                    str(column)
                    for column in frame.columns
                ],
                metadata={
                    "enabled": False,
                },
            )

            return self

        for column in frame.columns:
            column_name = str(
                column
            )

            series = frame[
                column
            ]

            unique_count = int(
                series.nunique(
                    dropna=(
                        not self.options
                        .count_missing_as_category
                    )
                )
            )

            missing_count = int(
                series.isna().sum()
            )

            constant = bool(
                unique_count
                < self.options.minimum_unique_count
            )

            required_override = bool(
                constant
                and column_name in required
            )

            protected_override = bool(
                constant
                and column_name in protected
            )

            candidate_for_removal = bool(
                constant
                and not required_override
                and not protected_override
            )

            if constant:
                constant_features.append(
                    column_name
                )

            if candidate_for_removal:
                removal_candidates.append(
                    column_name
                )

                reason = (
                    "The feature has fewer unique training "
                    "values than the configured minimum."
                )

            else:
                retained_features.append(
                    column_name
                )

                if required_override:
                    overridden_features.append(
                        column_name
                    )

                    reason = (
                        "The feature is constant but remains "
                        "because it is required."
                    )

                elif protected_override:
                    overridden_features.append(
                        column_name
                    )

                    reason = (
                        "The feature is constant but remains "
                        "because it is protected."
                    )

                else:
                    reason = (
                        "The feature satisfies the configured "
                        "unique-value requirement."
                    )

            feature_results[
                column_name
            ] = ConstantFeatureResult(
                column=column_name,
                unique_count=unique_count,
                missing_count=missing_count,
                constant=constant,
                candidate_for_removal=(
                    candidate_for_removal
                ),
                protected_override=(
                    protected_override
                ),
                required_override=(
                    required_override
                ),
                reason=reason,
            )

        self.result = ConstantFilterResult(
            features=feature_results,
            constant_features=constant_features,
            removal_candidates=removal_candidates,
            retained_features=retained_features,
            overridden_features=overridden_features,
            fitted_on_training_only=True,
            metadata={
                "enabled": True,
                "source_feature_count": len(
                    frame.columns
                ),
                "constant_feature_count": len(
                    constant_features
                ),
                "removal_candidate_count": len(
                    removal_candidates
                ),
            },
        )

        return self

    def get_result(
        self,
    ) -> ConstantFilterResult:
        """Return the fitted filter result."""

        if self.result is None:
            raise RuntimeError(
                "The constant filter is not fitted."
            )

        return self.result.model_copy(
            deep=True
        )