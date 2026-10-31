"""Categorical and text encoding recommendations for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.diagnostics.high_cardinality import (
    HighCardinalityDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.rare_categories import (
    RareCategoryDiagnosticResult,
)


EncodingMethod = Literal[
    "one_hot",
    "ordinal",
    "frequency",
    "count",
    "hashing",
    "target_encoding",
    "tfidf",
    "text_hashing",
    "exclude",
    "review",
]


class EncodingRecommendation(BaseModel):
    """Encoding recommendation for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    method: EncodingMethod

    priority: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str

    group_rare_categories: bool = False
    rare_category_label: str = "__RARE__"

    unknown_category_policy: str = "use_unknown"

    requires_cross_fitting: bool = False
    requires_training_fit: bool = True

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class EncodingRecommendationResult(BaseModel):
    """Complete encoding recommendations."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    recommendations: dict[
        str,
        EncodingRecommendation,
    ] = Field(default_factory=dict)

    cross_fitting_columns: list[str] = Field(
        default_factory=list
    )

    rare_grouping_columns: list[str] = Field(
        default_factory=list
    )

    excluded_columns: list[str] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class EncodingRecommendationOptions:
    """Options controlling encoding recommendations."""

    enabled: bool = True

    allow_target_encoding: bool = True

    prefer_frequency_over_target_encoding: bool = False

    rare_category_label: str = "__RARE__"

    unknown_category_policy: str = "use_unknown"

    def __post_init__(self) -> None:
        """Validate encoding recommendation options."""

        if not self.rare_category_label:
            raise ValueError(
                "rare_category_label cannot be empty."
            )

        if not self.unknown_category_policy:
            raise ValueError(
                "unknown_category_policy cannot be empty."
            )


class EncodingRecommendationBuilder:
    """Build Phase 7 encoding recommendations."""

    component_name = "encoding_recommendations"

    def __init__(
        self,
        options: (
            EncodingRecommendationOptions | None
        ) = None,
    ) -> None:
        """Initialize the recommendation builder."""

        self.options = (
            options
            or EncodingRecommendationOptions()
        )

    def build(
        self,
        *,
        cardinality: HighCardinalityDiagnosticResult,
        rare_categories: RareCategoryDiagnosticResult,
    ) -> EncodingRecommendationResult:
        """Build categorical and text encoding recommendations."""

        if not self.options.enabled:
            return EncodingRecommendationResult()

        recommendations: dict[
            str,
            EncodingRecommendation,
        ] = {}

        cross_fitting_columns: list[str] = []
        rare_grouping_columns: list[str] = []
        excluded_columns: list[str] = []

        for column, diagnostic in (
            cardinality.diagnostics.items()
        ):
            suggested = diagnostic.suggested_encoding

            warnings: list[str] = []
            requires_cross_fitting = False

            if suggested == "target_encoding":
                if not self.options.allow_target_encoding:
                    method: EncodingMethod = "frequency"

                    reason = (
                        "Target encoding is disabled, so frequency "
                        "encoding is recommended."
                    )

                elif (
                    self.options
                    .prefer_frequency_over_target_encoding
                ):
                    method = "frequency"

                    reason = (
                        "Frequency encoding is preferred to avoid "
                        "target-dependent transformations."
                    )

                else:
                    method = "target_encoding"

                    reason = diagnostic.reason

                    requires_cross_fitting = True

                    warnings.append(
                        "Phase 7 must use cross-fitted target "
                        "encoding for training rows."
                    )

                    cross_fitting_columns.append(
                        column
                    )

            elif suggested == "text_vectorization":
                method = "tfidf"
                reason = diagnostic.reason

            else:
                method = suggested
                reason = diagnostic.reason

            rare_diagnostic = (
                rare_categories
                .diagnostics
                .get(column)
            )

            group_rare = bool(
                rare_diagnostic is not None
                and rare_diagnostic
                .recommend_grouping
            )

            if group_rare:
                rare_grouping_columns.append(
                    column
                )

            if method == "exclude":
                excluded_columns.append(
                    column
                )

            priority = self._priority(
                level=diagnostic.level
            )

            recommendations[column] = (
                EncodingRecommendation(
                    column=column,
                    method=method,
                    priority=priority,
                    reason=reason,
                    group_rare_categories=(
                        group_rare
                    ),
                    rare_category_label=(
                        self.options
                        .rare_category_label
                    ),
                    unknown_category_policy=(
                        self.options
                        .unknown_category_policy
                    ),
                    requires_cross_fitting=(
                        requires_cross_fitting
                    ),
                    requires_training_fit=(
                        method != "exclude"
                    ),
                    warnings=warnings,
                    metadata={
                        "unique_count": (
                            diagnostic.unique_count
                        ),
                        "uniqueness_ratio": (
                            diagnostic
                            .uniqueness_ratio
                        ),
                        "cardinality_level": (
                            diagnostic.level
                        ),
                    },
                )
            )

        return EncodingRecommendationResult(
            recommendations=recommendations,
            cross_fitting_columns=list(
                dict.fromkeys(
                    cross_fitting_columns
                )
            ),
            rare_grouping_columns=list(
                dict.fromkeys(
                    rare_grouping_columns
                )
            ),
            excluded_columns=list(
                dict.fromkeys(
                    excluded_columns
                )
            ),
        )

    @staticmethod
    def _priority(
        *,
        level: str,
    ) -> float:
        """Map cardinality level to priority."""

        priorities = {
            "low": 0.40,
            "medium": 0.65,
            "high": 0.85,
            "identifier_like": 1.0,
        }

        return priorities.get(
            level,
            0.50,
        )