"""Feature-interaction recommendations for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.relationships.interaction_candidates import (
    InteractionCandidateResult,
)


class InteractionRecommendation(BaseModel):
    """Recommendation for one feature interaction."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str
    kind: str

    source_columns: list[str] = Field(
        min_length=2
    )

    priority: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str

    create_in_phase6: bool = True

    safety_requirements: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class InteractionRecommendationResult(BaseModel):
    """Complete feature-interaction recommendations."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    recommendations: list[
        InteractionRecommendation
    ] = Field(default_factory=list)

    approved_candidate_names: list[str] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class InteractionRecommendationOptions:
    """Options controlling interaction recommendations."""

    enabled: bool = True

    minimum_priority: float = 0.10

    maximum_recommendations: int = 100

    allow_ratio_features: bool = True
    allow_categorical_crosses: bool = True

    def __post_init__(self) -> None:
        """Validate interaction recommendation options."""

        if not 0.0 <= self.minimum_priority <= 1.0:
            raise ValueError(
                "minimum_priority must be between zero and one."
            )

        if self.maximum_recommendations < 1:
            raise ValueError(
                "maximum_recommendations must be positive."
            )


class InteractionRecommendationBuilder:
    """Convert interaction candidates into Phase 6 recommendations."""

    component_name = "interaction_recommendations"

    def __init__(
        self,
        options: (
            InteractionRecommendationOptions | None
        ) = None,
    ) -> None:
        """Initialize the recommendation builder."""

        self.options = (
            options
            or InteractionRecommendationOptions()
        )

    def build(
        self,
        *,
        candidates: InteractionCandidateResult,
    ) -> InteractionRecommendationResult:
        """Build interaction recommendations."""

        if not self.options.enabled:
            return InteractionRecommendationResult()

        recommendations: list[
            InteractionRecommendation
        ] = []

        for candidate in candidates.candidates:
            if (
                candidate.priority
                < self.options.minimum_priority
            ):
                continue

            if (
                candidate.kind == "numeric_ratio"
                and not self.options.allow_ratio_features
            ):
                continue

            if (
                candidate.kind == "categorical_cross"
                and not (
                    self.options
                    .allow_categorical_crosses
                )
            ):
                continue

            safety_requirements: list[str] = []

            if candidate.kind == "numeric_ratio":
                safety_requirements.extend(
                    [
                        "Protect against division by zero.",
                        "Replace nonfinite outputs according "
                        "to the Phase 6 policy.",
                    ]
                )

            if candidate.kind == "categorical_cross":
                safety_requirements.append(
                    "Apply cardinality controls before encoding."
                )

            recommendations.append(
                InteractionRecommendation(
                    name=candidate.name,
                    kind=candidate.kind,
                    source_columns=list(
                        candidate.source_columns
                    ),
                    priority=candidate.priority,
                    reason=candidate.reason,
                    create_in_phase6=True,
                    safety_requirements=(
                        safety_requirements
                    ),
                    metadata=dict(
                        candidate.metadata
                    ),
                )
            )

        recommendations.sort(
            key=lambda recommendation: (
                recommendation.priority
            ),
            reverse=True,
        )

        recommendations = recommendations[
            : self.options.maximum_recommendations
        ]

        return InteractionRecommendationResult(
            recommendations=recommendations,
            approved_candidate_names=[
                recommendation.name
                for recommendation
                in recommendations
            ],
        )