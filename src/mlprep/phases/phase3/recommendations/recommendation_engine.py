"""Unified recommendation engine for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.diagnostics.high_cardinality import (
    HighCardinalityDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.rare_categories import (
    RareCategoryDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.skewness import (
    SkewnessDiagnosticResult,
)
from mlprep.phases.phase3.profiling.missingness_profile import (
    MissingnessProfile,
)
from mlprep.phases.phase3.profiling.numeric_profile import (
    NumericColumnProfile,
)
from mlprep.phases.phase3.recommendations.encoding_recommendations import (
    EncodingRecommendationBuilder,
    EncodingRecommendationOptions,
    EncodingRecommendationResult,
)
from mlprep.phases.phase3.recommendations.imputation_recommendations import (
    ImputationRecommendationBuilder,
    ImputationRecommendationOptions,
    ImputationRecommendationResult,
)
from mlprep.phases.phase3.recommendations.interaction_recommendations import (
    InteractionRecommendationBuilder,
    InteractionRecommendationOptions,
    InteractionRecommendationResult,
)
from mlprep.phases.phase3.recommendations.outlier_recommendations import (
    OutlierRecommendationBuilder,
    OutlierRecommendationOptions,
    OutlierRecommendationResult,
)
from mlprep.phases.phase3.recommendations.transformation_recommendations import (
    TransformationRecommendationBuilder,
    TransformationRecommendationOptions,
    TransformationRecommendationResult,
)
from mlprep.phases.phase3.relationships.interaction_candidates import (
    InteractionCandidateResult,
)
from mlprep.phases.phase3.relationships.missingness_associations import (
    MissingnessAssociationResult,
)


class Phase3Recommendations(BaseModel):
    """Complete recommendation output from Phase 3."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    transformations: TransformationRecommendationResult

    imputation: ImputationRecommendationResult

    outliers: OutlierRecommendationResult

    encoding: EncodingRecommendationResult

    interactions: InteractionRecommendationResult

    feature_exclusion_candidates: list[str] = Field(
        default_factory=list
    )

    conflicts: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class RecommendationEngineOptions:
    """Complete recommendation-engine configuration."""

    enabled: bool = True

    transformations: (
        TransformationRecommendationOptions
    ) = field(
        default_factory=(
            TransformationRecommendationOptions
        )
    )

    imputation: ImputationRecommendationOptions = field(
        default_factory=(
            ImputationRecommendationOptions
        )
    )

    outliers: OutlierRecommendationOptions = field(
        default_factory=(
            OutlierRecommendationOptions
        )
    )

    encoding: EncodingRecommendationOptions = field(
        default_factory=(
            EncodingRecommendationOptions
        )
    )

    interactions: (
        InteractionRecommendationOptions
    ) = field(
        default_factory=(
            InteractionRecommendationOptions
        )
    )


class RecommendationEngine:
    """Build and reconcile all Phase 3 recommendations."""

    component_name = "recommendation_engine"

    def __init__(
        self,
        options: RecommendationEngineOptions | None = None,
    ) -> None:
        """Initialize the recommendation engine."""

        self.options = (
            options or RecommendationEngineOptions()
        )

    def build(
        self,
        *,
        columns: AnalysisColumnGroups,
        numeric_profiles: dict[
            str,
            NumericColumnProfile,
        ],
        missingness: MissingnessProfile,
        skewness: SkewnessDiagnosticResult,
        rare_categories: RareCategoryDiagnosticResult,
        cardinality: HighCardinalityDiagnosticResult,
        missingness_associations: (
            MissingnessAssociationResult
        ),
        interaction_candidates: (
            InteractionCandidateResult
        ),
    ) -> Phase3Recommendations:
        """Build all downstream recommendations."""

        if not self.options.enabled:
            return self._empty_result()

        transformations = (
            TransformationRecommendationBuilder(
                self.options.transformations
            ).build(
                skewness=skewness
            )
        )

        imputation = (
            ImputationRecommendationBuilder(
                self.options.imputation
            ).build(
                columns=columns,
                missingness=missingness,
                numeric_profiles=(
                    numeric_profiles
                ),
                missingness_associations=(
                    missingness_associations
                ),
            )
        )

        outliers = OutlierRecommendationBuilder(
            self.options.outliers
        ).build(
            numeric_profiles=numeric_profiles
        )

        encoding = EncodingRecommendationBuilder(
            self.options.encoding
        ).build(
            cardinality=cardinality,
            rare_categories=rare_categories,
        )

        interactions = (
            InteractionRecommendationBuilder(
                self.options.interactions
            ).build(
                candidates=interaction_candidates
            )
        )

        exclusion_candidates = list(
            dict.fromkeys(
                [
                    *encoding.excluded_columns,
                    *imputation.drop_review_columns,
                ]
            )
        )

        conflicts = self._find_conflicts(
            transformations=transformations,
            imputation=imputation,
            outliers=outliers,
            encoding=encoding,
        )

        return Phase3Recommendations(
            transformations=transformations,
            imputation=imputation,
            outliers=outliers,
            encoding=encoding,
            interactions=interactions,
            feature_exclusion_candidates=(
                exclusion_candidates
            ),
            conflicts=conflicts,
            metadata={
                "transformation_recommendation_count": len(
                    transformations.recommendations
                ),
                "imputation_recommendation_count": len(
                    imputation.recommendations
                ),
                "outlier_recommendation_count": len(
                    outliers.recommendations
                ),
                "encoding_recommendation_count": len(
                    encoding.recommendations
                ),
                "interaction_recommendation_count": len(
                    interactions.recommendations
                ),
            },
        )

    @staticmethod
    def _find_conflicts(
        *,
        transformations: TransformationRecommendationResult,
        imputation: ImputationRecommendationResult,
        outliers: OutlierRecommendationResult,
        encoding: EncodingRecommendationResult,
    ) -> list[object]:
        """Detect conflicting downstream recommendations."""

        conflicts: list[str] = []

        excluded_columns = set(
            encoding.excluded_columns
        )

        excluded_columns.update(
            imputation.drop_review_columns
        )

        for column in excluded_columns:
            if (
                column
                in transformations.recommendations
            ):
                conflicts.append(
                    f"Feature {column!r} is recommended for "
                    "both exclusion and numeric transformation."
                )

            if column in outliers.recommendations:
                conflicts.append(
                    f"Feature {column!r} is recommended for "
                    "both exclusion and outlier treatment."
                )

        for column, recommendation in (
            imputation.recommendations.items()
        ):
            if (
                recommendation.method
                == "drop_feature"
                and column
                in encoding.recommendations
            ):
                conflicts.append(
                    f"Feature {column!r} is marked for dropping "
                    "but also has an encoding recommendation."
                )

        return conflicts

    @staticmethod
    def _empty_result(
    ) -> Phase3Recommendations:
        """Return an empty recommendation result."""

        return Phase3Recommendations(
            transformations=(
                TransformationRecommendationResult()
            ),
            imputation=(
                ImputationRecommendationResult()
            ),
            outliers=(
                OutlierRecommendationResult()
            ),
            encoding=(
                EncodingRecommendationResult()
            ),
            interactions=(
                InteractionRecommendationResult()
            ),
        )
