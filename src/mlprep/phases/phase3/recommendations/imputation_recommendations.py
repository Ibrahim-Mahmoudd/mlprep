"""Missing-value imputation recommendations for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.profiling.missingness_profile import (
    MissingnessProfile,
)
from mlprep.phases.phase3.profiling.numeric_profile import (
    NumericColumnProfile,
)
from mlprep.phases.phase3.relationships.missingness_associations import (
    MissingnessAssociationResult,
)

ImputationMethod = Literal[
    "none",
    "median",
    "mean",
    "most_frequent",
    "constant",
    "missing_category",
    "time_aware",
    "group_based",
    "model_based",
    "drop_feature",
    "review",
]


class ImputationRecommendation(BaseModel):
    """Imputation recommendation for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    method: ImputationMethod

    priority: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str

    add_missing_indicator: bool = False

    constant_value: str | float | int | None = None

    requires_training_fit: bool = True

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class ImputationRecommendationResult(BaseModel):
    """Complete imputation recommendations."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    recommendations: dict[
        str,
        ImputationRecommendation,
    ] = Field(default_factory=dict)

    missing_indicator_columns: list[str] = Field(
        default_factory=list
    )

    drop_review_columns: list[str] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class ImputationRecommendationOptions:
    """Options controlling imputation recommendations."""

    enabled: bool = True

    high_missingness_threshold: float = 0.80
    indicator_missingness_threshold: float = 0.05

    skewness_median_threshold: float = 1.0

    missing_category_value: str = "__MISSING__"

    add_indicator_when_target_associated: bool = True

    def __post_init__(self) -> None:
        """Validate imputation recommendation options."""

        for value in (
            self.high_missingness_threshold,
            self.indicator_missingness_threshold,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Missingness thresholds must be "
                    "between zero and one."
                )

        if self.skewness_median_threshold < 0.0:
            raise ValueError(
                "skewness_median_threshold cannot be negative."
            )

        if not self.missing_category_value:
            raise ValueError(
                "missing_category_value cannot be empty."
            )


class ImputationRecommendationBuilder:
    """Build Phase 4 imputation recommendations."""

    component_name = "imputation_recommendations"

    def __init__(
        self,
        options: (
            ImputationRecommendationOptions | None
        ) = None,
    ) -> None:
        """Initialize the recommendation builder."""

        self.options = (
            options
            or ImputationRecommendationOptions()
        )

    def build(
        self,
        *,
        columns: AnalysisColumnGroups,
        missingness: MissingnessProfile,
        numeric_profiles: dict[
            str,
            NumericColumnProfile,
        ],
        missingness_associations: (
            MissingnessAssociationResult
        ),
    ) -> ImputationRecommendationResult:
        """Build imputation recommendations."""

        if not self.options.enabled:
            return ImputationRecommendationResult()

        target_associated = set(
            missingness_associations
            .strongly_associated_features
        )

        recommendations: dict[
            str,
            ImputationRecommendation,
        ] = {}

        indicator_columns: list[str] = []
        drop_review_columns: list[str] = []

        for column, profile in (
            missingness.columns.items()
        ):
            missing_ratio = (
                profile.missing_percentage
                / 100.0
            )

            if missing_ratio <= 0.0:
                continue

            add_indicator = bool(
                missing_ratio
                >= self.options
                .indicator_missingness_threshold
                or (
                    self.options
                    .add_indicator_when_target_associated
                    and column in target_associated
                )
            )

            if profile.completely_missing:
                method: ImputationMethod = (
                    "drop_feature"
                )

                reason = (
                    "The training feature is completely missing."
                )

                priority = 1.0
                drop_review_columns.append(column)

            elif (
                missing_ratio
                >= self.options
                .high_missingness_threshold
            ):
                method = "review"

                reason = (
                    "The feature has extremely high missingness. "
                    "Review data availability before imputation."
                )

                priority = 0.95
                drop_review_columns.append(column)

            elif column in columns.numeric:
                numeric_profile = (
                    numeric_profiles.get(column)
                )

                if (
                    numeric_profile is not None
                    and numeric_profile.skewness is not None
                    and abs(
                        numeric_profile.skewness
                    )
                    >= self.options
                    .skewness_median_threshold
                ):
                    method = "median"

                    reason = (
                        "Median imputation is recommended because "
                        "the numeric feature is skewed."
                    )
                else:
                    method = "median"

                    reason = (
                        "Median imputation is robust to potential "
                        "extreme values."
                    )

                priority = min(
                    1.0,
                    0.5 + missing_ratio,
                )

            elif column in columns.datetime:
                method = "time_aware"

                reason = (
                    "The feature is temporal. Phase 4 should "
                    "evaluate time-aware or constant imputation."
                )

                priority = min(
                    1.0,
                    0.5 + missing_ratio,
                )

            elif (
                column in columns.categorical
                or column in columns.boolean
            ):
                method = "missing_category"

                reason = (
                    "An explicit missing category preserves "
                    "missingness information."
                )

                priority = min(
                    1.0,
                    0.5 + missing_ratio,
                )

            elif column in columns.text:
                method = "constant"

                reason = (
                    "Text missingness can be represented using "
                    "an empty or explicit missing token."
                )

                priority = min(
                    1.0,
                    0.4 + missing_ratio,
                )

            else:
                method = "review"
                reason = (
                    "The feature type is unresolved."
                )
                priority = 0.50

            if add_indicator:
                indicator_columns.append(column)

            recommendations[column] = (
                ImputationRecommendation(
                    column=column,
                    method=method,
                    priority=priority,
                    reason=reason,
                    add_missing_indicator=(
                        add_indicator
                    ),
                    constant_value=(
                        self.options
                        .missing_category_value
                        if method
                        in {
                            "missing_category",
                            "constant",
                        }
                        else None
                    ),
                    requires_training_fit=(
                        method
                        not in {
                            "constant",
                            "missing_category",
                            "drop_feature",
                            "review",
                        }
                    ),
                    metadata={
                        "missing_percentage": (
                            profile
                            .missing_percentage
                        ),
                        "target_associated": (
                            column
                            in target_associated
                        ),
                    },
                )
            )

        return ImputationRecommendationResult(
            recommendations=recommendations,
            missing_indicator_columns=list(
                dict.fromkeys(
                    indicator_columns
                )
            ),
            drop_review_columns=list(
                dict.fromkeys(
                    drop_review_columns
                )
            ),
        )