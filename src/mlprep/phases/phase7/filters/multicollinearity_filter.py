"""Multicollinearity filtering for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class MulticollinearityIteration(BaseModel):
    """One iterative multicollinearity-filter step."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    iteration: int = Field(
        ge=1
    )

    feature_count: int = Field(
        ge=1
    )

    maximum_vif: float
    maximum_vif_feature: str

    condition_number: float | None = None

    removed_feature: str | None = None
    removal_reason: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class MulticollinearityFilterResult(BaseModel):
    """Complete multicollinearity-filter output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    vif_scores: dict[
        str,
        float,
    ] = Field(default_factory=dict)

    removed_features: list[str] = Field(
        default_factory=list
    )

    retained_features: list[str] = Field(
        default_factory=list
    )

    overridden_features: list[str] = Field(
        default_factory=list
    )

    condition_number: float | None = None

    condition_number_exceeded: bool = False

    iterations: list[
        MulticollinearityIteration
    ] = Field(default_factory=list)

    converged: bool = True
    fitted_on_training_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class MulticollinearityFilterOptions:
    """Options controlling VIF filtering."""

    enabled: bool = False

    maximum_vif: float = 10.0

    maximum_condition_number: float | None = None

    maximum_iterations: int = 100

    numerical_tolerance: float = 1e-12

    add_intercept: bool = True

    def __post_init__(self) -> None:
        """Validate multicollinearity options."""

        if self.maximum_vif <= 1.0:
            raise ValueError(
                "maximum_vif must exceed one."
            )

        if (
            self.maximum_condition_number
            is not None
            and self.maximum_condition_number <= 1.0
        ):
            raise ValueError(
                "maximum_condition_number must exceed one."
            )

        if self.maximum_iterations < 1:
            raise ValueError(
                "maximum_iterations must be positive."
            )

        if self.numerical_tolerance <= 0.0:
            raise ValueError(
                "numerical_tolerance must be positive."
            )


class MulticollinearityFilter:
    """Iteratively remove features with excessive VIF."""

    component_name = "multicollinearity_filter"

    def __init__(
        self,
        options: MulticollinearityFilterOptions | None = None,
    ) -> None:
        """Initialize the filter."""

        self.options = (
            options
            or MulticollinearityFilterOptions()
        )

        self.result: (
            MulticollinearityFilterResult | None
        ) = None

    def fit(
        self,
        *,
        frame: pd.DataFrame,
        candidate_features: list[str] | None = None,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
        source_feature_order: list[str] | None = None,
    ) -> MulticollinearityFilter:
        """Fit iterative VIF filtering on training data."""

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

        self._validate_frame(
            frame=frame,
            features=candidates,
        )

        if not self.options.enabled:
            self.result = (
                MulticollinearityFilterResult(
                    retained_features=(
                        candidates
                    ),
                    metadata={
                        "enabled": False,
                    },
                )
            )

            return self

        required = set(
            required_features or []
        )

        protected = set(
            protected_features or []
        )

        source_order = (
            candidates
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

        active = list(
            candidates
        )

        removed: list[str] = []
        overridden: list[str] = []
        iterations: list[
            MulticollinearityIteration
        ] = []

        converged = False

        for iteration_number in range(
            1,
            self.options.maximum_iterations + 1,
        ):
            if len(active) <= 1:
                converged = True
                break

            active_frame = frame.loc[
                :,
                active,
            ].astype(float)

            vif_scores = self._calculate_vif(
                active_frame
            )

            condition_number = (
                self._condition_number(
                    active_frame
                )
            )

            maximum_feature = max(
                active,
                key=lambda feature: (
                    vif_scores.get(
                        feature,
                        0.0,
                    ),
                    order_index.get(
                        feature,
                        -1,
                    ),
                ),
            )

            maximum_vif = float(
                vif_scores[
                    maximum_feature
                ]
            )

            if maximum_vif <= self.options.maximum_vif:
                iterations.append(
                    MulticollinearityIteration(
                        iteration=(
                            iteration_number
                        ),
                        feature_count=len(
                            active
                        ),
                        maximum_vif=(
                            maximum_vif
                        ),
                        maximum_vif_feature=(
                            maximum_feature
                        ),
                        condition_number=(
                            condition_number
                        ),
                        removal_reason=(
                            "All VIF values satisfy the "
                            "configured threshold."
                        ),
                    )
                )

                converged = True
                break

            removable_features = [
                feature
                for feature in active
                if (
                    vif_scores[
                        feature
                    ]
                    > self.options.maximum_vif
                    and feature not in required
                    and feature not in protected
                )
            ]

            if not removable_features:
                overridden.extend(
                    [
                        feature
                        for feature in active
                        if vif_scores[
                            feature
                        ]
                        > self.options.maximum_vif
                    ]
                )

                iterations.append(
                    MulticollinearityIteration(
                        iteration=(
                            iteration_number
                        ),
                        feature_count=len(
                            active
                        ),
                        maximum_vif=(
                            maximum_vif
                        ),
                        maximum_vif_feature=(
                            maximum_feature
                        ),
                        condition_number=(
                            condition_number
                        ),
                        removal_reason=(
                            "All excessive-VIF features are "
                            "required or protected."
                        ),
                    )
                )

                converged = False
                break

            feature_to_remove = max(
                removable_features,
                key=lambda feature: (
                    vif_scores[
                        feature
                    ],
                    order_index.get(
                        feature,
                        -1,
                    ),
                ),
            )

            iterations.append(
                MulticollinearityIteration(
                    iteration=iteration_number,
                    feature_count=len(
                        active
                    ),
                    maximum_vif=maximum_vif,
                    maximum_vif_feature=(
                        maximum_feature
                    ),
                    condition_number=(
                        condition_number
                    ),
                    removed_feature=(
                        feature_to_remove
                    ),
                    removal_reason=(
                        "The feature has the highest "
                        "removable VIF."
                    ),
                )
            )

            active.remove(
                feature_to_remove
            )

            removed.append(
                feature_to_remove
            )

        final_frame = frame.loc[
            :,
            active,
        ].astype(float)

        final_vif_scores = (
            self._calculate_vif(
                final_frame
            )
            if active
            else {}
        )

        final_condition_number = (
            self._condition_number(
                final_frame
            )
            if active
            else None
        )

        condition_exceeded = bool(
            self.options.maximum_condition_number
            is not None
            and final_condition_number is not None
            and final_condition_number
            > self.options.maximum_condition_number
        )

        self.result = (
            MulticollinearityFilterResult(
                vif_scores=final_vif_scores,
                removed_features=removed,
                retained_features=active,
                overridden_features=list(
                    dict.fromkeys(
                        overridden
                    )
                ),
                condition_number=(
                    final_condition_number
                ),
                condition_number_exceeded=(
                    condition_exceeded
                ),
                iterations=iterations,
                converged=converged,
                fitted_on_training_only=True,
                metadata={
                    "enabled": True,
                    "source_feature_count": len(
                        candidates
                    ),
                    "retained_feature_count": len(
                        active
                    ),
                    "removed_feature_count": len(
                        removed
                    ),
                },
            )
        )

        return self

    def get_result(
        self,
    ) -> MulticollinearityFilterResult:
        """Return the fitted result."""

        if self.result is None:
            raise RuntimeError(
                "The multicollinearity filter is not fitted."
            )

        return self.result.model_copy(
            deep=True
        )

    def _calculate_vif(
        self,
        frame: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate VIF for each feature."""

        if frame.empty:
            return {}

        values = frame.to_numpy(
            dtype=float
        )

        scores: dict[
            str,
            float,
        ] = {}

        for index, column in enumerate(
            frame.columns
        ):
            target = values[
                :,
                index,
            ]

            predictors = np.delete(
                values,
                index,
                axis=1,
            )

            if predictors.shape[1] == 0:
                scores[
                    str(column)
                ] = 1.0

                continue

            if self.options.add_intercept:
                predictors = np.column_stack(
                    [
                        np.ones(
                            predictors.shape[0]
                        ),
                        predictors,
                    ]
                )

            coefficients, _, _, _ = (
                np.linalg.lstsq(
                    predictors,
                    target,
                    rcond=None,
                )
            )

            predicted = (
                predictors
                @ coefficients
            )

            residual_sum = float(
                np.sum(
                    np.square(
                        target - predicted
                    )
                )
            )

            centered = (
                target
                - np.mean(
                    target
                )
            )

            total_sum = float(
                np.sum(
                    np.square(
                        centered
                    )
                )
            )

            if (
                total_sum
                <= self.options
                .numerical_tolerance
            ):
                scores[
                    str(column)
                ] = float(
                    "inf"
                )

                continue

            r_squared = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - residual_sum
                    / total_sum,
                ),
            )

            denominator = (
                1.0
                - r_squared
            )

            scores[
                str(column)
            ] = (
                float(
                    "inf"
                )
                if denominator
                <= self.options
                .numerical_tolerance
                else 1.0
                / denominator
            )

        return scores

    def _condition_number(
        self,
        frame: pd.DataFrame,
    ) -> float | None:
        """Calculate a standardized matrix condition number."""

        if frame.empty:
            return None

        values = frame.to_numpy(
            dtype=float
        )

        means = np.mean(
            values,
            axis=0,
        )

        standard_deviations = np.std(
            values,
            axis=0,
        )

        standard_deviations = np.where(
            standard_deviations
            <= self.options.numerical_tolerance,
            1.0,
            standard_deviations,
        )

        standardized = (
            values - means
        ) / standard_deviations

        singular_values = np.linalg.svd(
            standardized,
            compute_uv=False,
        )

        if singular_values.size == 0:
            return None

        smallest = float(
            singular_values.min()
        )

        largest = float(
            singular_values.max()
        )

        if (
            smallest
            <= self.options.numerical_tolerance
        ):
            return float(
                "inf"
            )

        return (
            largest
            / smallest
        )

    @staticmethod
    def _validate_frame(
        *,
        frame: pd.DataFrame,
        features: list[str],
    ) -> None:
        """Validate VIF-filter input."""

        missing = [
            feature
            for feature in features
            if feature not in frame.columns
        ]

        if missing:
            raise KeyError(
                "Multicollinearity features are missing: "
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
                "Multicollinearity filtering requires "
                f"numeric features: {nonnumeric}"
            )

        invalid = [
            feature
            for feature in features
            if (
                frame[
                    feature
                ].isna().any()
                or not np.isfinite(
                    frame[
                        feature
                    ].astype(float)
                ).all()
            )
        ]

        if invalid:
            raise ValueError(
                "Multicollinearity filtering requires "
                "complete finite features: "
                f"{invalid}"
            )
