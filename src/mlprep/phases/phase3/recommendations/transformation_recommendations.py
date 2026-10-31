"""Numeric transformation recommendations for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.diagnostics.skewness import (
    SkewnessDiagnostic,
    SkewnessDiagnosticResult,
)


TransformationMethod = Literal[
    "none",
    "log",
    "log1p",
    "box_cox",
    "yeo_johnson",
    "quantile",
    "robust_scale",
    "standard_scale",
    "review",
]


VALID_TRANSFORMATION_METHODS: frozenset[str] = frozenset(
    {
        "none",
        "log",
        "log1p",
        "box_cox",
        "yeo_johnson",
        "quantile",
        "robust_scale",
        "standard_scale",
        "review",
    }
)


class TransformationRecommendation(BaseModel):
    """Transformation recommendation for one numeric feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    method: TransformationMethod

    priority: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str

    requires_training_fit: bool = True

    preserve_original_feature: bool = False

    prerequisites: list[str] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class TransformationRecommendationResult(BaseModel):
    """Complete numeric transformation recommendations."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    recommendations: dict[
        str,
        TransformationRecommendation,
    ] = Field(
        default_factory=dict
    )

    transformation_columns: list[str] = Field(
        default_factory=list
    )

    review_columns: list[str] = Field(
        default_factory=list
    )

    no_transformation_columns: list[str] = Field(
        default_factory=list
    )

    fitted_transformation_columns: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class TransformationRecommendationOptions:
    """Options controlling transformation recommendations."""

    enabled: bool = True

    recommend_scaling_for_symmetric_numeric: bool = False

    symmetric_scaling_method: Literal[
        "standard_scale",
        "robust_scale",
    ] = "standard_scale"

    preserve_original_features: bool = False

    include_no_transformation_recommendations: bool = True

    minimum_priority: float = 0.0

    symmetric_priority: float = 0.10
    scaling_priority: float = 0.30
    moderate_skewness_priority: float = 0.50
    high_skewness_priority: float = 0.80
    undefined_skewness_priority: float = 0.40

    extreme_skewness_reference: float = 3.0

    def __post_init__(self) -> None:
        """Validate transformation recommendation options."""

        priority_values = {
            "minimum_priority": (
                self.minimum_priority
            ),
            "symmetric_priority": (
                self.symmetric_priority
            ),
            "scaling_priority": (
                self.scaling_priority
            ),
            "moderate_skewness_priority": (
                self.moderate_skewness_priority
            ),
            "high_skewness_priority": (
                self.high_skewness_priority
            ),
            "undefined_skewness_priority": (
                self.undefined_skewness_priority
            ),
        }

        for name, value in priority_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between zero and one."
                )

        if self.extreme_skewness_reference <= 0.0:
            raise ValueError(
                "extreme_skewness_reference must be positive."
            )


class TransformationRecommendationBuilder:
    """Build numeric transformation recommendations."""

    component_name = "transformation_recommendations"

    def __init__(
        self,
        options: (
            TransformationRecommendationOptions | None
        ) = None,
    ) -> None:
        """Initialize the transformation recommendation builder."""

        self.options = (
            options
            or TransformationRecommendationOptions()
        )

    def build(
        self,
        *,
        skewness: SkewnessDiagnosticResult,
    ) -> TransformationRecommendationResult:
        """Build recommendations from skewness diagnostics."""

        if not self.options.enabled:
            return TransformationRecommendationResult(
                metadata={
                    "enabled": False,
                }
            )

        recommendations: dict[
            str,
            TransformationRecommendation,
        ] = {}

        transformation_columns: list[str] = []
        review_columns: list[str] = []
        no_transformation_columns: list[str] = []
        fitted_transformation_columns: list[str] = []

        for column, diagnostic in (
            skewness.diagnostics.items()
        ):
            recommendation = (
                self._build_column_recommendation(
                    column=column,
                    diagnostic=diagnostic,
                )
            )

            if recommendation is None:
                continue

            recommendations[column] = recommendation

            if recommendation.method == "none":
                no_transformation_columns.append(
                    column
                )

            elif recommendation.method == "review":
                review_columns.append(column)

            else:
                transformation_columns.append(
                    column
                )

            if recommendation.requires_training_fit:
                fitted_transformation_columns.append(
                    column
                )

        return TransformationRecommendationResult(
            recommendations=recommendations,
            transformation_columns=list(
                dict.fromkeys(
                    transformation_columns
                )
            ),
            review_columns=list(
                dict.fromkeys(
                    review_columns
                )
            ),
            no_transformation_columns=list(
                dict.fromkeys(
                    no_transformation_columns
                )
            ),
            fitted_transformation_columns=list(
                dict.fromkeys(
                    fitted_transformation_columns
                )
            ),
            metadata={
                "enabled": True,
                "recommendation_count": len(
                    recommendations
                ),
                "transformation_count": len(
                    transformation_columns
                ),
                "review_count": len(
                    review_columns
                ),
                "no_transformation_count": len(
                    no_transformation_columns
                ),
            },
        )

    def _build_column_recommendation(
        self,
        *,
        column: str,
        diagnostic: SkewnessDiagnostic,
    ) -> TransformationRecommendation | None:
        """Build one feature transformation recommendation."""

        (
            method,
            priority,
            reason,
        ) = self._resolve_recommendation(
            diagnostic=diagnostic
        )

        if priority < self.options.minimum_priority:
            return None

        if (
            method == "none"
            and not self.options
            .include_no_transformation_recommendations
        ):
            return None

        return TransformationRecommendation(
            column=column,
            method=method,
            priority=priority,
            reason=reason,
            requires_training_fit=(
                self._requires_training_fit(
                    method
                )
            ),
            preserve_original_feature=(
                self.options
                .preserve_original_features
            ),
            prerequisites=self._prerequisites(
                method=method
            ),
            warnings=self._warnings(
                method=method
            ),
            metadata={
                "skewness": diagnostic.skewness,
                "absolute_skewness": (
                    diagnostic.absolute_skewness
                ),
                "skewness_level": (
                    diagnostic.level
                ),
                "minimum": diagnostic.minimum,
                "maximum": diagnostic.maximum,
                "contains_zero": (
                    diagnostic.contains_zero
                ),
                "contains_negative": (
                    diagnostic.contains_negative
                ),
                "original_suggestion": (
                    diagnostic
                    .suggested_transformation
                ),
            },
        )

    def _resolve_recommendation(
        self,
        *,
        diagnostic: SkewnessDiagnostic,
    ) -> tuple[
        TransformationMethod,
        float,
        str,
    ]:
        """Resolve the final transformation method and priority."""

        if diagnostic.level == "undefined":
            return (
                "review",
                self.options
                .undefined_skewness_priority,
                diagnostic.reason,
            )

        suggested_method = self._normalize_method(
            diagnostic.suggested_transformation
        )

        if suggested_method == "review":
            return (
                "review",
                self.options
                .undefined_skewness_priority,
                diagnostic.reason,
            )

        if suggested_method == "none":
            if (
                self.options
                .recommend_scaling_for_symmetric_numeric
            ):
                return (
                    self.options
                    .symmetric_scaling_method,
                    self.options.scaling_priority,
                    (
                        "The feature is approximately symmetric. "
                        "Scaling may help models that are sensitive "
                        "to feature magnitude."
                    ),
                )

            return (
                "none",
                self.options.symmetric_priority,
                diagnostic.reason,
            )

        priority = self._transformation_priority(
            level=diagnostic.level,
            absolute_skewness=(
                diagnostic.absolute_skewness
            ),
        )

        return (
            suggested_method,
            priority,
            diagnostic.reason,
        )

    def _transformation_priority(
        self,
        *,
        level: str,
        absolute_skewness: float | None,
    ) -> float:
        """Calculate transformation priority."""

        if level == "high":
            base_priority = (
                self.options
                .high_skewness_priority
            )

        elif level == "moderate":
            base_priority = (
                self.options
                .moderate_skewness_priority
            )

        else:
            base_priority = (
                self.options.symmetric_priority
            )

        if absolute_skewness is None:
            return base_priority

        magnitude_priority = min(
            1.0,
            absolute_skewness
            / self.options
            .extreme_skewness_reference,
        )

        return min(
            1.0,
            max(
                base_priority,
                magnitude_priority,
            ),
        )

    @staticmethod
    def _normalize_method(
        method: str,
    ) -> TransformationMethod:
        """Validate and normalize a transformation method."""

        if method not in VALID_TRANSFORMATION_METHODS:
            return "review"

        return cast(
            TransformationMethod,
            method,
        )

    @staticmethod
    def _requires_training_fit(
        method: TransformationMethod,
    ) -> bool:
        """Return whether a method learns state from training data."""

        fitted_methods: set[
            TransformationMethod
        ] = {
            "box_cox",
            "yeo_johnson",
            "quantile",
            "robust_scale",
            "standard_scale",
        }

        return method in fitted_methods

    @staticmethod
    def _prerequisites(
        *,
        method: TransformationMethod,
    ) -> list[object]:
        """Return prerequisites for one transformation."""

        prerequisites: dict[
            TransformationMethod,
            list[str],
        ] = {
            "none": [],
            "review": [],
            "log": [
                (
                    "All transformed values must be "
                    "strictly positive."
                ),
                (
                    "The same deterministic transformation "
                    "must be applied to training, validation, "
                    "test, and production data."
                ),
            ],
            "log1p": [
                (
                    "All transformed values must be greater "
                    "than or equal to -1."
                ),
                (
                    "Nonfinite outputs must be rejected or "
                    "handled explicitly."
                ),
            ],
            "box_cox": [
                (
                    "All transformed values must be "
                    "strictly positive."
                ),
                (
                    "The Box-Cox parameter must be learned "
                    "from training data only."
                ),
            ],
            "yeo_johnson": [
                (
                    "The Yeo-Johnson parameter must be learned "
                    "from training data only."
                ),
            ],
            "quantile": [
                (
                    "The empirical quantile mapping must be "
                    "learned from training data only."
                ),
                (
                    "Validation, test, and production data must "
                    "use the mapping learned from training."
                ),
            ],
            "standard_scale": [
                (
                    "The mean and standard deviation must be "
                    "learned from training data only."
                ),
            ],
            "robust_scale": [
                (
                    "The median and scale statistics must be "
                    "learned from training data only."
                ),
            ],
        }

        return list(
            prerequisites[method]
        )

    @staticmethod
    def _warnings(
        *,
        method: TransformationMethod,
    ) -> list[object]:
        """Return warnings for one transformation."""

        warnings: dict[
            TransformationMethod,
            list[str],
        ] = {
            "none": [],
            "log": [
                (
                    "A logarithmic transformation changes "
                    "feature interpretation."
                ),
            ],
            "log1p": [
                (
                    "A log1p transformation changes feature "
                    "interpretation."
                ),
            ],
            "box_cox": [
                (
                    "Box-Cox cannot process zero or negative "
                    "values without an explicit shift policy."
                ),
            ],
            "yeo_johnson": [
                (
                    "Yeo-Johnson is nonlinear and may reduce "
                    "direct coefficient interpretability."
                ),
            ],
            "quantile": [
                (
                    "Quantile transformation may distort "
                    "distances and reduce interpretability."
                ),
                (
                    "Values outside the training range require "
                    "an explicit extrapolation policy."
                ),
            ],
            "standard_scale": [
                (
                    "Standard scaling is sensitive to extreme "
                    "values."
                ),
            ],
            "robust_scale": [
                (
                    "Robust scaling changes feature units."
                ),
            ],
            "review": [
                (
                    "No automatic transformation should be "
                    "applied until the feature is reviewed."
                ),
            ],
        }

        return list(
            warnings[method]
        )