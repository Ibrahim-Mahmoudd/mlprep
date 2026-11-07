"""Recommendation components for Phase 3."""

from mlprep.phases.phase3.recommendations.encoding_recommendations import (
    EncodingRecommendation,
    EncodingRecommendationBuilder,
    EncodingRecommendationOptions,
    EncodingRecommendationResult,
)
from mlprep.phases.phase3.recommendations.imputation_recommendations import (
    ImputationRecommendation,
    ImputationRecommendationBuilder,
    ImputationRecommendationOptions,
    ImputationRecommendationResult,
)
from mlprep.phases.phase3.recommendations.interaction_recommendations import (
    InteractionRecommendation,
    InteractionRecommendationBuilder,
    InteractionRecommendationOptions,
    InteractionRecommendationResult,
)
from mlprep.phases.phase3.recommendations.outlier_recommendations import (
    OutlierRecommendation,
    OutlierRecommendationBuilder,
    OutlierRecommendationOptions,
    OutlierRecommendationResult,
)
from mlprep.phases.phase3.recommendations.recommendation_engine import (
    Phase3Recommendations,
    RecommendationEngine,
    RecommendationEngineOptions,
)
from mlprep.phases.phase3.recommendations.transformation_recommendations import (
    TransformationRecommendation,
    TransformationRecommendationBuilder,
    TransformationRecommendationOptions,
    TransformationRecommendationResult,
)

__all__ = [
    "EncodingRecommendation",
    "EncodingRecommendationBuilder",
    "EncodingRecommendationOptions",
    "EncodingRecommendationResult",
    "ImputationRecommendation",
    "ImputationRecommendationBuilder",
    "ImputationRecommendationOptions",
    "ImputationRecommendationResult",
    "InteractionRecommendation",
    "InteractionRecommendationBuilder",
    "InteractionRecommendationOptions",
    "InteractionRecommendationResult",
    "OutlierRecommendation",
    "OutlierRecommendationBuilder",
    "OutlierRecommendationOptions",
    "OutlierRecommendationResult",
    "Phase3Recommendations",
    "RecommendationEngine",
    "RecommendationEngineOptions",
    "TransformationRecommendation",
    "TransformationRecommendationBuilder",
    "TransformationRecommendationOptions",
    "TransformationRecommendationResult",
]
