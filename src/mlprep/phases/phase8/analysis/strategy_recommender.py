"""Nonbinding balancing-strategy recommendations for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase8.analysis.class_distribution import (
    ClassDistributionResult,
)
from mlprep.phases.phase8.analysis.imbalance_metrics import (
    ImbalanceMetricsResult,
)
from mlprep.phases.phase8.contract.balancing_plan import (
    BalancingStrategy,
    NeighborShortagePolicy,
)


class StrategyCandidate(BaseModel):
    """One candidate balancing strategy."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    strategy: BalancingStrategy

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    recommended: bool = False
    feasible: bool = True

    advantages: list[str] = Field(
        default_factory=list,
    )

    risks: list[str] = Field(
        default_factory=list,
    )

    constraints: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class StrategyRecommendationResult(BaseModel):
    """Complete nonbinding balancing recommendation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    configured_strategy: BalancingStrategy

    recommended_strategy: BalancingStrategy

    recommendation_applied: bool = False
    configuration_has_authority: bool = True

    recommendation_conflicts_with_configuration: bool = False

    candidates: dict[
        str,
        StrategyCandidate,
    ] = Field(
        default_factory=dict,
    )

    nearest_neighbor_synthesis_safe: bool = False

    recommended_neighbor_count: int | None = Field(
        default=None,
        ge=1,
    )

    fallback_strategy: BalancingStrategy | None = None

    reasons: list[str] = Field(
        default_factory=list,
    )

    warnings: list[str] = Field(
        default_factory=list,
    )

    analyzed_on_training_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class StrategyRecommenderOptions:
    """Options controlling balancing-strategy recommendations."""

    enabled: bool = True

    minimum_rows_for_synthetic_sampling: int = 100

    minimum_minority_count_for_synthetic_sampling: int = 6

    preferred_neighbor_count: int = 5

    maximum_feature_count_for_neighbor_sampling: int = 500

    maximum_binary_feature_ratio_for_synthetic_sampling: float = 0.80

    sparse_zero_rate_threshold: float = 0.90

    mild_imbalance_strategy: BalancingStrategy = (
        BalancingStrategy.CLASS_WEIGHTS_ONLY
    )

    moderate_imbalance_strategy: BalancingStrategy = (
        BalancingStrategy.RANDOM_OVERSAMPLING
    )

    severe_imbalance_strategy: BalancingStrategy = (
        BalancingStrategy.SMOTE
    )

    extreme_imbalance_strategy: BalancingStrategy = (
        BalancingStrategy.RANDOM_OVERSAMPLING
    )

    neighbor_shortage_fallback: BalancingStrategy = (
        BalancingStrategy.RANDOM_OVERSAMPLING
    )

    prefer_class_weights_for_high_dimensional_data: bool = True

    prefer_random_oversampling_for_binary_dominant_data: bool = True

    def __post_init__(self) -> None:
        """Validate strategy-recommender options."""

        if (
            self.minimum_rows_for_synthetic_sampling
            < 1
        ):
            raise ValueError(
                "minimum_rows_for_synthetic_sampling "
                "must be positive."
            )

        if (
            self.minimum_minority_count_for_synthetic_sampling
            < 2
        ):
            raise ValueError(
                "minimum_minority_count_for_synthetic_sampling "
                "must be at least two."
            )

        if self.preferred_neighbor_count < 1:
            raise ValueError(
                "preferred_neighbor_count must be positive."
            )

        if (
            self.maximum_feature_count_for_neighbor_sampling
            < 1
        ):
            raise ValueError(
                "maximum_feature_count_for_neighbor_sampling "
                "must be positive."
            )

        if not 0.0 <= (
            self.maximum_binary_feature_ratio_for_synthetic_sampling
        ) <= 1.0:
            raise ValueError(
                "maximum_binary_feature_ratio_for_synthetic_sampling "
                "must be between zero and one."
            )

        if not 0.0 <= (
            self.sparse_zero_rate_threshold
        ) <= 1.0:
            raise ValueError(
                "sparse_zero_rate_threshold must be "
                "between zero and one."
            )


class StrategyRecommender:
    """Produce a nonbinding balancing recommendation."""

    component_name = "strategy_recommender"

    def __init__(
        self,
        options: StrategyRecommenderOptions | None = None,
    ) -> None:
        """Initialize the recommender."""

        self.options = (
            options
            or StrategyRecommenderOptions()
        )

        self.result: (
            StrategyRecommendationResult | None
        ) = None

    def recommend(
        self,
        *,
        features: pd.DataFrame,
        distribution: ClassDistributionResult,
        metrics: ImbalanceMetricsResult,
        configured_strategy: BalancingStrategy,
        neighbor_shortage_policy: NeighborShortagePolicy = (
            NeighborShortagePolicy.REDUCE_NEIGHBORS
        ),
    ) -> StrategyRecommendationResult:
        """Recommend a strategy from training diagnostics."""

        self._validate_inputs(
            features=features,
            distribution=distribution,
            metrics=metrics,
        )

        resolved_configured_strategy = (
            configured_strategy
            if isinstance(
                configured_strategy,
                BalancingStrategy,
            )
            else BalancingStrategy(
                configured_strategy
            )
        )

        resolved_shortage_policy = (
            neighbor_shortage_policy
            if isinstance(
                neighbor_shortage_policy,
                NeighborShortagePolicy,
            )
            else NeighborShortagePolicy(
                neighbor_shortage_policy
            )
        )

        if not self.options.enabled:
            result = StrategyRecommendationResult(
                configured_strategy=(
                    resolved_configured_strategy
                ),
                recommended_strategy=(
                    resolved_configured_strategy
                ),
                recommendation_applied=False,
                configuration_has_authority=True,
                metadata={
                    "enabled": False,
                },
            )

            self.result = result

            return result.model_copy(
                deep=True
            )

        row_count = len(
            features
        )

        feature_count = len(
            features.columns
        )

        binary_feature_count = sum(
            self._is_binary_feature(
                features[
                    column
                ]
            )
            for column
            in features.columns
        )

        binary_feature_ratio = (
            binary_feature_count
            / feature_count
            if feature_count > 0
            else 0.0
        )

        sparse_feature_count = sum(
            self._zero_rate(
                features[
                    column
                ]
            )
            >= self.options
            .sparse_zero_rate_threshold
            for column
            in features.columns
        )

        sparse_feature_ratio = (
            sparse_feature_count
            / feature_count
            if feature_count > 0
            else 0.0
        )

        effective_neighbor_count = min(
            self.options
            .preferred_neighbor_count,
            max(
                distribution.smallest_class_count
                - 1,
                0,
            ),
        )

        nearest_neighbor_safe = bool(
            row_count
            >= self.options
            .minimum_rows_for_synthetic_sampling
            and distribution.smallest_class_count
            >= self.options
            .minimum_minority_count_for_synthetic_sampling
            and effective_neighbor_count >= 1
            and feature_count
            <= self.options
            .maximum_feature_count_for_neighbor_sampling
            and binary_feature_ratio
            <= self.options
            .maximum_binary_feature_ratio_for_synthetic_sampling
        )

        candidates = self._candidate_strategies(
            metrics=metrics,
            nearest_neighbor_safe=(
                nearest_neighbor_safe
            ),
            binary_feature_ratio=(
                binary_feature_ratio
            ),
            sparse_feature_ratio=(
                sparse_feature_ratio
            ),
            feature_count=feature_count,
        )

        recommended_strategy = (
            self._base_recommendation(
                metrics=metrics
            )
        )

        reasons: list[str] = []
        warnings: list[str] = []

        if metrics.balanced:
            recommended_strategy = (
                BalancingStrategy.NONE
            )

            reasons.append(
                "The training target satisfies the configured "
                "balance threshold."
            )

        elif (
            recommended_strategy
            in self._synthetic_strategies()
            and not nearest_neighbor_safe
        ):
            recommended_strategy = (
                self.options
                .neighbor_shortage_fallback
            )

            reasons.append(
                "Nearest-neighbor synthesis is not considered "
                "safe for the current training data."
            )

            warnings.append(
                "A non-synthetic fallback strategy is "
                "recommended."
            )

        if (
            self.options
            .prefer_class_weights_for_high_dimensional_data
            and feature_count
            > self.options
            .maximum_feature_count_for_neighbor_sampling
        ):
            recommended_strategy = (
                BalancingStrategy.CLASS_WEIGHTS_ONLY
            )

            reasons.append(
                "The selected feature matrix is too "
                "high-dimensional for the configured "
                "neighbor-sampling preference."
            )

        if (
            self.options
            .prefer_random_oversampling_for_binary_dominant_data
            and binary_feature_ratio
            > self.options
            .maximum_binary_feature_ratio_for_synthetic_sampling
            and not metrics.balanced
        ):
            recommended_strategy = (
                BalancingStrategy.RANDOM_OVERSAMPLING
            )

            reasons.append(
                "The selected feature matrix is dominated by "
                "binary features, so random oversampling is "
                "preferred over interpolation."
            )

        if (
            resolved_shortage_policy
            == NeighborShortagePolicy.BLOCK
            and not nearest_neighbor_safe
            and resolved_configured_strategy
            in self._synthetic_strategies()
        ):
            warnings.append(
                "The configured synthetic strategy may be "
                "blocked because nearest-neighbor requirements "
                "are not satisfied."
            )

        if (
            resolved_shortage_policy
            == NeighborShortagePolicy.RANDOM_OVERSAMPLING
            and not nearest_neighbor_safe
            and resolved_configured_strategy
            in self._synthetic_strategies()
        ):
            warnings.append(
                "The configured synthetic strategy should "
                "fall back to random oversampling."
            )

        for _strategy, candidate in (
            candidates.items()
        ):
            candidate.recommended = bool(
                candidate.strategy
                == recommended_strategy
            )

        recommendation_conflict = bool(
            recommended_strategy
            != resolved_configured_strategy
        )

        if recommendation_conflict:
            warnings.append(
                "The nonbinding recommendation differs from "
                "the configured strategy. The configured plan "
                "retains authority."
            )

        fallback_strategy = (
            self.options
            .neighbor_shortage_fallback
            if not nearest_neighbor_safe
            else None
        )

        if not reasons:
            reasons.append(
                "The recommendation follows the diagnosed "
                f"{metrics.severity!r} imbalance severity."
            )

        result = StrategyRecommendationResult(
            configured_strategy=(
                resolved_configured_strategy
            ),
            recommended_strategy=(
                recommended_strategy
            ),
            recommendation_applied=False,
            configuration_has_authority=True,
            recommendation_conflicts_with_configuration=(
                recommendation_conflict
            ),
            candidates={
                strategy.value: candidate
                for strategy, candidate
                in candidates.items()
            },
            nearest_neighbor_synthesis_safe=(
                nearest_neighbor_safe
            ),
            recommended_neighbor_count=(
                effective_neighbor_count
                if effective_neighbor_count >= 1
                else None
            ),
            fallback_strategy=(
                fallback_strategy
            ),
            reasons=reasons,
            warnings=warnings,
            analyzed_on_training_only=True,
            metadata={
                "enabled": True,
                "row_count": row_count,
                "feature_count": feature_count,
                "binary_feature_count": (
                    binary_feature_count
                ),
                "binary_feature_ratio": (
                    binary_feature_ratio
                ),
                "sparse_feature_count": (
                    sparse_feature_count
                ),
                "sparse_feature_ratio": (
                    sparse_feature_ratio
                ),
                "smallest_class_count": (
                    distribution
                    .smallest_class_count
                ),
                "imbalance_severity": (
                    metrics.severity
                ),
            },
        )

        self.result = result

        return result.model_copy(
            deep=True
        )

    def get_result(
        self,
    ) -> StrategyRecommendationResult:
        """Return the latest recommendation."""

        if self.result is None:
            raise RuntimeError(
                "A balancing strategy has not been recommended."
            )

        return self.result.model_copy(
            deep=True
        )

    def _base_recommendation(
        self,
        *,
        metrics: ImbalanceMetricsResult,
    ) -> BalancingStrategy:
        """Map imbalance severity to a base recommendation."""

        if metrics.severity == "none":
            return BalancingStrategy.NONE

        if metrics.severity == "mild":
            return self.options.mild_imbalance_strategy

        if metrics.severity == "moderate":
            return (
                self.options
                .moderate_imbalance_strategy
            )

        if metrics.severity == "severe":
            return (
                self.options
                .severe_imbalance_strategy
            )

        return (
            self.options
            .extreme_imbalance_strategy
        )

    def _candidate_strategies(
        self,
        *,
        metrics: ImbalanceMetricsResult,
        nearest_neighbor_safe: bool,
        binary_feature_ratio: float,
        sparse_feature_ratio: float,
        feature_count: int,
    ) -> dict[
        BalancingStrategy,
        StrategyCandidate,
    ]:
        """Build scored strategy candidates."""

        severity_score = (
            metrics.imbalance_score
        )

        candidates = {
            BalancingStrategy.NONE: StrategyCandidate(
                strategy=BalancingStrategy.NONE,
                score=max(
                    0.0,
                    1.0
                    - severity_score,
                ),
                feasible=True,
                advantages=[
                    "Preserves all original training rows.",
                    "Introduces no synthetic values.",
                ],
                risks=[
                    "May leave the diagnosed imbalance unchanged."
                ],
            ),
            (
                BalancingStrategy.CLASS_WEIGHTS_ONLY
            ): StrategyCandidate(
                strategy=(
                    BalancingStrategy.CLASS_WEIGHTS_ONLY
                ),
                score=min(
                    1.0,
                    0.50
                    + severity_score
                    * 0.35,
                ),
                feasible=True,
                advantages=[
                    "Preserves the original training distribution.",
                    "Avoids synthetic interpolation.",
                ],
                risks=[
                    "Requires downstream estimators that support weights."
                ],
            ),
            (
                BalancingStrategy.RANDOM_OVERSAMPLING
            ): StrategyCandidate(
                strategy=(
                    BalancingStrategy.RANDOM_OVERSAMPLING
                ),
                score=min(
                    1.0,
                    0.45
                    + severity_score
                    * 0.50,
                ),
                feasible=True,
                advantages=[
                    "Supports very small minority classes.",
                    "Preserves observed feature values.",
                ],
                risks=[
                    "May increase overfitting through duplicates."
                ],
            ),
            (
                BalancingStrategy.RANDOM_UNDERSAMPLING
            ): StrategyCandidate(
                strategy=(
                    BalancingStrategy.RANDOM_UNDERSAMPLING
                ),
                score=min(
                    1.0,
                    0.30
                    + severity_score
                    * 0.40,
                ),
                feasible=bool(
                    metrics.majority_count
                    > metrics.minority_count
                ),
                advantages=[
                    "Reduces training size.",
                    "Avoids synthetic feature generation.",
                ],
                risks=[
                    "Removes original majority-class information."
                ],
            ),
            BalancingStrategy.SMOTE: StrategyCandidate(
                strategy=BalancingStrategy.SMOTE,
                score=(
                    min(
                        1.0,
                        0.55
                        + severity_score
                        * 0.40,
                    )
                    if nearest_neighbor_safe
                    else 0.0
                ),
                feasible=nearest_neighbor_safe,
                advantages=[
                    "Generates new minority-class samples.",
                    "Can reduce exact duplication.",
                ],
                risks=[
                    "May create unrealistic interpolated rows.",
                    "Can be unsuitable for binary-dominant data.",
                ],
            ),
            (
                BalancingStrategy.BORDERLINE_SMOTE
            ): StrategyCandidate(
                strategy=(
                    BalancingStrategy.BORDERLINE_SMOTE
                ),
                score=(
                    min(
                        1.0,
                        0.50
                        + severity_score
                        * 0.35,
                    )
                    if nearest_neighbor_safe
                    else 0.0
                ),
                feasible=nearest_neighbor_safe,
                advantages=[
                    "Focuses synthesis near class boundaries."
                ],
                risks=[
                    "May amplify noisy boundary observations."
                ],
            ),
            BalancingStrategy.ADASYN: StrategyCandidate(
                strategy=BalancingStrategy.ADASYN,
                score=(
                    min(
                        1.0,
                        0.45
                        + severity_score
                        * 0.35,
                    )
                    if nearest_neighbor_safe
                    else 0.0
                ),
                feasible=nearest_neighbor_safe,
                advantages=[
                    "Allocates more synthetic rows to difficult regions."
                ],
                risks=[
                    "May emphasize noise and outliers."
                ],
            ),
            (
                BalancingStrategy.SMOTE_TOMEK
            ): StrategyCandidate(
                strategy=(
                    BalancingStrategy.SMOTE_TOMEK
                ),
                score=(
                    min(
                        1.0,
                        0.45
                        + severity_score
                        * 0.30,
                    )
                    if nearest_neighbor_safe
                    else 0.0
                ),
                feasible=nearest_neighbor_safe,
                advantages=[
                    "Combines synthesis with boundary cleaning."
                ],
                risks=[
                    "May remove original training rows."
                ],
            ),
            (
                BalancingStrategy.SMOTE_ENN
            ): StrategyCandidate(
                strategy=(
                    BalancingStrategy.SMOTE_ENN
                ),
                score=(
                    min(
                        1.0,
                        0.40
                        + severity_score
                        * 0.30,
                    )
                    if nearest_neighbor_safe
                    else 0.0
                ),
                feasible=nearest_neighbor_safe,
                advantages=[
                    "Combines synthesis with neighborhood cleaning."
                ],
                risks=[
                    "May remove a substantial number of original rows."
                ],
            ),
        }

        synthetic_constraints: list[str] = []

        if not nearest_neighbor_safe:
            synthetic_constraints.append(
                "Nearest-neighbor synthesis safety "
                "requirements are not satisfied."
            )

        if binary_feature_ratio > (
            self.options
            .maximum_binary_feature_ratio_for_synthetic_sampling
        ):
            synthetic_constraints.append(
                "The binary-feature ratio exceeds the "
                "configured synthetic-sampling limit."
            )

        if feature_count > (
            self.options
            .maximum_feature_count_for_neighbor_sampling
        ):
            synthetic_constraints.append(
                "The feature count exceeds the configured "
                "neighbor-sampling limit."
            )

        if sparse_feature_ratio > 0.50:
            synthetic_constraints.append(
                "More than half of the selected features "
                "are highly sparse."
            )

        for strategy in self._synthetic_strategies():
            candidates[
                strategy
            ].constraints.extend(
                synthetic_constraints
            )

        return candidates

    @staticmethod
    def _synthetic_strategies(
    ) -> set[object]:
        """Return strategies that synthesize rows."""

        return {
            BalancingStrategy.SMOTE,
            BalancingStrategy.BORDERLINE_SMOTE,
            BalancingStrategy.ADASYN,
            BalancingStrategy.SMOTE_TOMEK,
            BalancingStrategy.SMOTE_ENN,
        }

    @staticmethod
    def _is_binary_feature(
        series: pd.Series,
    ) -> bool:
        """Return whether a feature contains only zero and one."""

        values = set(
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .dropna()
            .unique()
        )

        return bool(
            values
            and values.issubset(
                {
                    0,
                    1,
                }
            )
        )

    @staticmethod
    def _zero_rate(
        series: pd.Series,
    ) -> float:
        """Calculate the zero rate of one numeric feature."""

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        valid = numeric.dropna()

        if valid.empty:
            return 0.0

        return float(
            valid.eq(
                0.0
            ).mean()
        )

    @staticmethod
    def _validate_inputs(
        *,
        features: pd.DataFrame,
        distribution: ClassDistributionResult,
        metrics: ImbalanceMetricsResult,
    ) -> None:
        """Validate recommendation inputs."""

        if not isinstance(
            features,
            pd.DataFrame,
        ):
            raise TypeError(
                "features must be a pandas DataFrame."
            )

        if features.empty:
            raise ValueError(
                "Training features cannot be empty."
            )

        if not isinstance(
            distribution,
            ClassDistributionResult,
        ):
            raise TypeError(
                "distribution must be a "
                "ClassDistributionResult."
            )

        if not isinstance(
            metrics,
            ImbalanceMetricsResult,
        ):
            raise TypeError(
                "metrics must be an ImbalanceMetricsResult."
            )

        if len(
            features
        ) != distribution.row_count:
            raise ValueError(
                "Training features and class-distribution "
                "row counts differ."
            )
