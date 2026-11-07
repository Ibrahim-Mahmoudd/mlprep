"""Outlier-treatment recommendations for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.profiling.numeric_profile import (
    NumericColumnProfile,
)

OutlierTreatment = Literal[
    "none",
    "report_only",
    "clip",
    "winsorize",
    "set_missing",
    "robust_scale",
    "indicator",
    "domain_review",
]


class OutlierRecommendation(BaseModel):
    """Outlier recommendation for one numeric feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    treatment: OutlierTreatment

    priority: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str

    suggested_lower_bound: float | None = None
    suggested_upper_bound: float | None = None

    add_indicator: bool = False

    requires_training_fit: bool = True

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class OutlierRecommendationResult(BaseModel):
    """Complete outlier-treatment recommendations."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    recommendations: dict[
        str,
        OutlierRecommendation,
    ] = Field(default_factory=dict)

    treatment_columns: list[str] = Field(
        default_factory=list
    )

    domain_review_columns: list[str] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class OutlierRecommendationOptions:
    """Options controlling outlier recommendations."""

    enabled: bool = True

    iqr_multiplier: float = 1.5

    high_skewness_threshold: float = 1.0
    extreme_skewness_threshold: float = 3.0

    recommend_indicator: bool = True

    def __post_init__(self) -> None:
        """Validate outlier recommendation options."""

        if self.iqr_multiplier <= 0.0:
            raise ValueError(
                "iqr_multiplier must be positive."
            )

        if self.high_skewness_threshold < 0.0:
            raise ValueError(
                "high_skewness_threshold cannot be negative."
            )

        if (
            self.extreme_skewness_threshold
            < self.high_skewness_threshold
        ):
            raise ValueError(
                "extreme_skewness_threshold cannot be below "
                "high_skewness_threshold."
            )


class OutlierRecommendationBuilder:
    """Build Phase 5 outlier-treatment recommendations."""

    component_name = "outlier_recommendations"

    def __init__(
        self,
        options: (
            OutlierRecommendationOptions | None
        ) = None,
    ) -> None:
        """Initialize the recommendation builder."""

        self.options = (
            options
            or OutlierRecommendationOptions()
        )

    def build(
        self,
        *,
        numeric_profiles: dict[
            str,
            NumericColumnProfile,
        ],
    ) -> OutlierRecommendationResult:
        """Build recommendations from numeric profiles."""

        if not self.options.enabled:
            return OutlierRecommendationResult()

        recommendations: dict[
            str,
            OutlierRecommendation,
        ] = {}

        domain_review_columns: list[str] = []

        for column, profile in (
            numeric_profiles.items()
        ):
            if (
                profile.first_quartile is None
                or profile.third_quartile is None
                or profile.interquartile_range is None
            ):
                continue

            iqr = profile.interquartile_range

            lower_bound = (
                profile.first_quartile
                - self.options.iqr_multiplier
                * iqr
            )

            upper_bound = (
                profile.third_quartile
                + self.options.iqr_multiplier
                * iqr
            )

            absolute_skewness = abs(
                profile.skewness or 0.0
            )

            warnings: list[str] = []

            if profile.infinite_count > 0:
                treatment: OutlierTreatment = (
                    "set_missing"
                )

                priority = 1.0

                reason = (
                    "The feature contains infinite values, "
                    "which should be treated as invalid."
                )

                domain_review_columns.append(
                    column
                )

            elif (
                absolute_skewness
                >= self.options
                .extreme_skewness_threshold
            ):
                treatment = "winsorize"

                priority = min(
                    1.0,
                    absolute_skewness / 5.0,
                )

                reason = (
                    "The feature is extremely skewed. "
                    "Winsorization or transformation should "
                    "be evaluated on training data."
                )

                warnings.append(
                    "Do not remove validation or test rows."
                )

            elif (
                absolute_skewness
                >= self.options
                .high_skewness_threshold
            ):
                treatment = "robust_scale"

                priority = min(
                    1.0,
                    absolute_skewness / 3.0,
                )

                reason = (
                    "The feature is skewed and may benefit from "
                    "robust scaling."
                )

            else:
                treatment = "report_only"
                priority = 0.30

                reason = (
                    "No aggressive automatic outlier treatment "
                    "is recommended from univariate statistics."
                )

            recommendations[column] = (
                OutlierRecommendation(
                    column=column,
                    treatment=treatment,
                    priority=priority,
                    reason=reason,
                    suggested_lower_bound=(
                        float(lower_bound)
                    ),
                    suggested_upper_bound=(
                        float(upper_bound)
                    ),
                    add_indicator=(
                        self.options
                        .recommend_indicator
                        and treatment
                        not in {
                            "none",
                            "report_only",
                        }
                    ),
                    requires_training_fit=True,
                    warnings=warnings,
                    metadata={
                        "skewness": profile.skewness,
                        "iqr": iqr,
                        "infinite_count": (
                            profile.infinite_count
                        ),
                    },
                )
            )

        return OutlierRecommendationResult(
            recommendations=recommendations,
            treatment_columns=[
                column
                for column, recommendation
                in recommendations.items()
                if recommendation.treatment
                not in {
                    "none",
                    "report_only",
                }
            ],
            domain_review_columns=list(
                dict.fromkeys(
                    domain_review_columns
                )
            ),
        )