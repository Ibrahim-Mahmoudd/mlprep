"""Target-imbalance metrics for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase8.analysis.class_distribution import (
    ClassDistributionResult,
)


class ImbalanceMetricsResult(BaseModel):
    """Complete target-imbalance metrics."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    class_count: int = Field(
        default=0,
        ge=0,
    )

    total_count: int = Field(
        default=0,
        ge=0,
    )

    majority_count: int = Field(
        default=0,
        ge=0,
    )

    minority_count: int = Field(
        default=0,
        ge=0,
    )

    majority_to_minority_ratio: float = Field(
        default=1.0,
        ge=1.0,
    )

    minority_to_majority_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    imbalance_ratio: float = Field(
        default=1.0,
        ge=1.0,
    )

    maximum_class_proportion: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    minimum_class_proportion: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    class_proportion_range: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    normalized_entropy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    entropy_deficit: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    gini_impurity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    normalized_gini_impurity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    effective_number_of_classes: float = Field(
        default=0.0,
        ge=0.0,
    )

    multiclass_dispersion: float = Field(
        default=0.0,
        ge=0.0,
    )

    coefficient_of_variation: float = Field(
        default=0.0,
        ge=0.0,
    )

    imbalance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    balanced: bool = False
    severity: str = "none"

    analyzed_on_training_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class ImbalanceMetricsOptions:
    """Options controlling target-imbalance metrics."""

    enabled: bool = True

    balanced_ratio_threshold: float = 1.20

    mild_ratio_threshold: float = 2.00
    moderate_ratio_threshold: float = 4.00
    severe_ratio_threshold: float = 10.00

    entropy_epsilon: float = 1e-12

    use_combined_imbalance_score: bool = True

    ratio_score_weight: float = 0.50
    entropy_score_weight: float = 0.25
    dispersion_score_weight: float = 0.25

    def __post_init__(self) -> None:
        """Validate imbalance-metric options."""

        thresholds = [
            self.balanced_ratio_threshold,
            self.mild_ratio_threshold,
            self.moderate_ratio_threshold,
            self.severe_ratio_threshold,
        ]

        if any(
            threshold < 1.0
            for threshold in thresholds
        ):
            raise ValueError(
                "Imbalance-ratio thresholds must be "
                "at least one."
            )

        if thresholds != sorted(
            thresholds
        ):
            raise ValueError(
                "Imbalance-ratio thresholds must be "
                "nondecreasing."
            )

        if self.entropy_epsilon <= 0.0:
            raise ValueError(
                "entropy_epsilon must be positive."
            )

        weights = [
            self.ratio_score_weight,
            self.entropy_score_weight,
            self.dispersion_score_weight,
        ]

        if any(
            weight < 0.0
            for weight in weights
        ):
            raise ValueError(
                "Imbalance-score weights cannot be negative."
            )

        if sum(
            weights
        ) <= 0.0:
            raise ValueError(
                "At least one imbalance-score weight "
                "must be positive."
            )


class ImbalanceMetricsAnalyzer:
    """Calculate diagnostic imbalance metrics."""

    component_name = "imbalance_metrics"

    def __init__(
        self,
        options: ImbalanceMetricsOptions | None = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options
            or ImbalanceMetricsOptions()
        )

        self.result: ImbalanceMetricsResult | None = None

    def analyze(
        self,
        *,
        distribution: ClassDistributionResult,
    ) -> ImbalanceMetricsResult:
        """Calculate metrics from a class distribution."""

        if not isinstance(
            distribution,
            ClassDistributionResult,
        ):
            raise TypeError(
                "distribution must be a "
                "ClassDistributionResult."
            )

        if not self.options.enabled:
            result = ImbalanceMetricsResult(
                class_count=(
                    distribution.class_count
                ),
                total_count=(
                    distribution.valid_target_count
                ),
                majority_count=(
                    distribution.largest_class_count
                ),
                minority_count=(
                    distribution.smallest_class_count
                ),
                balanced=distribution.balanced,
                metadata={
                    "enabled": False,
                },
            )

            self.result = result

            return result.model_copy(
                deep=True
            )

        class_count = (
            distribution.class_count
        )

        total_count = sum(
            entry.count
            for entry
            in distribution.classes.values()
        )

        if class_count < 1:
            raise ValueError(
                "At least one class is required to "
                "calculate imbalance metrics."
            )

        if total_count < 1:
            raise ValueError(
                "At least one valid target value is required."
            )

        proportions = np.asarray(
            [
                entry.proportion
                for entry
                in distribution.classes.values()
            ],
            dtype=float,
        )

        proportions = proportions[
            proportions > 0.0
        ]

        if proportions.size < 1:
            raise ValueError(
                "The class distribution contains no "
                "positive class proportions."
            )

        majority_count = (
            distribution.largest_class_count
        )

        minority_count = (
            distribution.smallest_class_count
        )

        if minority_count <= 0:
            majority_to_minority_ratio = float(
                "inf"
            )

            minority_to_majority_ratio = 0.0

        else:
            majority_to_minority_ratio = (
                majority_count
                / minority_count
            )

            minority_to_majority_ratio = (
                minority_count
                / majority_count
                if majority_count > 0
                else 0.0
            )

        maximum_class_proportion = float(
            proportions.max()
        )

        minimum_class_proportion = float(
            proportions.min()
        )

        class_proportion_range = (
            maximum_class_proportion
            - minimum_class_proportion
        )

        entropy = float(
            -np.sum(
                proportions
                * np.log(
                    np.maximum(
                        proportions,
                        self.options.entropy_epsilon,
                    )
                )
            )
        )

        maximum_entropy = (
            float(
                np.log(
                    class_count
                )
            )
            if class_count > 1
            else 0.0
        )

        normalized_entropy = (
            entropy
            / maximum_entropy
            if maximum_entropy > 0.0
            else 1.0
        )

        normalized_entropy = float(
            np.clip(
                normalized_entropy,
                0.0,
                1.0,
            )
        )

        entropy_deficit = (
            1.0
            - normalized_entropy
        )

        gini_impurity = float(
            1.0
            - np.sum(
                np.square(
                    proportions
                )
            )
        )

        maximum_gini = (
            1.0
            - 1.0
            / class_count
            if class_count > 0
            else 0.0
        )

        normalized_gini_impurity = (
            gini_impurity
            / maximum_gini
            if maximum_gini > 0.0
            else 1.0
        )

        normalized_gini_impurity = float(
            np.clip(
                normalized_gini_impurity,
                0.0,
                1.0,
            )
        )

        effective_number_of_classes = float(
            np.exp(
                entropy
            )
        )

        uniform_proportion = (
            1.0
            / class_count
        )

        multiclass_dispersion = float(
            np.sqrt(
                np.mean(
                    np.square(
                        proportions
                        - uniform_proportion
                    )
                )
            )
        )

        class_counts = np.asarray(
            [
                entry.count
                for entry
                in distribution.classes.values()
            ],
            dtype=float,
        )

        mean_class_count = float(
            class_counts.mean()
        )

        coefficient_of_variation = (
            float(
                class_counts.std(
                    ddof=0
                )
                / mean_class_count
            )
            if mean_class_count > 0.0
            else 0.0
        )

        ratio_score = self._ratio_score(
            majority_to_minority_ratio
        )

        maximum_possible_dispersion = (
            self._maximum_dispersion(
                class_count
            )
        )

        normalized_dispersion = (
            multiclass_dispersion
            / maximum_possible_dispersion
            if maximum_possible_dispersion > 0.0
            else 0.0
        )

        normalized_dispersion = float(
            np.clip(
                normalized_dispersion,
                0.0,
                1.0,
            )
        )

        if (
            self.options
            .use_combined_imbalance_score
        ):
            weight_total = (
                self.options.ratio_score_weight
                + self.options.entropy_score_weight
                + self.options
                .dispersion_score_weight
            )

            imbalance_score = (
                ratio_score
                * self.options
                .ratio_score_weight
                + entropy_deficit
                * self.options
                .entropy_score_weight
                + normalized_dispersion
                * self.options
                .dispersion_score_weight
            ) / weight_total

        else:
            imbalance_score = ratio_score

        imbalance_score = float(
            np.clip(
                imbalance_score,
                0.0,
                1.0,
            )
        )

        balanced = bool(
            majority_to_minority_ratio
            <= self.options
            .balanced_ratio_threshold
        )

        severity = self._severity(
            majority_to_minority_ratio
        )

        result = ImbalanceMetricsResult(
            class_count=class_count,
            total_count=total_count,
            majority_count=(
                majority_count
            ),
            minority_count=(
                minority_count
            ),
            majority_to_minority_ratio=(
                self._finite_ratio(
                    majority_to_minority_ratio
                )
            ),
            minority_to_majority_ratio=(
                minority_to_majority_ratio
            ),
            imbalance_ratio=(
                self._finite_ratio(
                    majority_to_minority_ratio
                )
            ),
            maximum_class_proportion=(
                maximum_class_proportion
            ),
            minimum_class_proportion=(
                minimum_class_proportion
            ),
            class_proportion_range=(
                class_proportion_range
            ),
            normalized_entropy=(
                normalized_entropy
            ),
            entropy_deficit=(
                entropy_deficit
            ),
            gini_impurity=(
                gini_impurity
            ),
            normalized_gini_impurity=(
                normalized_gini_impurity
            ),
            effective_number_of_classes=(
                effective_number_of_classes
            ),
            multiclass_dispersion=(
                multiclass_dispersion
            ),
            coefficient_of_variation=(
                coefficient_of_variation
            ),
            imbalance_score=(
                imbalance_score
            ),
            balanced=balanced,
            severity=severity,
            analyzed_on_training_only=True,
            metadata={
                "enabled": True,
                "raw_entropy": entropy,
                "maximum_entropy": (
                    maximum_entropy
                ),
                "normalized_dispersion": (
                    normalized_dispersion
                ),
                "ratio_score": (
                    ratio_score
                ),
                "distribution_balanced_flag": (
                    distribution.balanced
                ),
            },
        )

        self.result = result

        return result.model_copy(
            deep=True
        )

    def get_result(
        self,
    ) -> ImbalanceMetricsResult:
        """Return the latest metrics result."""

        if self.result is None:
            raise RuntimeError(
                "Imbalance metrics have not been analyzed."
            )

        return self.result.model_copy(
            deep=True
        )

    def _severity(
        self,
        ratio: float,
    ) -> str:
        """Classify imbalance severity."""

        if (
            ratio
            <= self.options
            .balanced_ratio_threshold
        ):
            return "none"

        if ratio <= self.options.mild_ratio_threshold:
            return "mild"

        if (
            ratio
            <= self.options
            .moderate_ratio_threshold
        ):
            return "moderate"

        if ratio <= self.options.severe_ratio_threshold:
            return "severe"

        return "extreme"

    def _ratio_score(
        self,
        ratio: float,
    ) -> float:
        """Normalize an imbalance ratio to zero through one."""

        if not np.isfinite(
            ratio
        ):
            return 1.0

        if (
            ratio
            <= self.options
            .balanced_ratio_threshold
        ):
            return 0.0

        denominator = max(
            self.options.severe_ratio_threshold
            - self.options.balanced_ratio_threshold,
            self.options.entropy_epsilon,
        )

        return float(
            np.clip(
                (
                    ratio
                    - self.options
                    .balanced_ratio_threshold
                )
                / denominator,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _maximum_dispersion(
        class_count: int,
    ) -> float:
        """Return maximum dispersion from a uniform distribution."""

        if class_count <= 1:
            return 0.0

        extreme = np.zeros(
            class_count,
            dtype=float,
        )

        extreme[
            0
        ] = 1.0

        uniform = np.full(
            class_count,
            1.0
            / class_count,
            dtype=float,
        )

        return float(
            np.sqrt(
                np.mean(
                    np.square(
                        extreme
                        - uniform
                    )
                )
            )
        )

    @staticmethod
    def _finite_ratio(
        value: float,
    ) -> float:
        """Convert an infinite ratio to a finite report value."""

        if np.isfinite(
            value
        ):
            return max(
                1.0,
                float(
                    value
                ),
            )

        return float(
            np.finfo(
                np.float64
            ).max
        )
