"""Distribution-stability evaluation for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class FeatureDistributionStability(BaseModel):
    """Distribution-stability metrics for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    training_row_count: int = Field(
        default=0,
        ge=0,
    )

    comparison_row_count: int = Field(
        default=0,
        ge=0,
    )

    population_stability_index: float = Field(
        default=0.0,
        ge=0.0,
    )

    standardized_mean_shift: float = Field(
        default=0.0,
        ge=0.0,
    )

    median_shift: float = Field(
        default=0.0,
        ge=0.0,
    )

    zero_rate_training: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    zero_rate_comparison: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    zero_rate_shift: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    missing_rate_training: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    missing_rate_comparison: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    missing_rate_shift: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    unstable: bool = False
    blocks_readiness: bool = False

    instability_reasons: list[str] = Field(
        default_factory=list,
    )

    bin_edges: list[float] = Field(
        default_factory=list,
    )

    training_bin_rates: list[float] = Field(
        default_factory=list,
    )

    comparison_bin_rates: list[float] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class DistributionStabilityResult(BaseModel):
    """Complete distribution-stability output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    features: dict[
        str,
        FeatureDistributionStability,
    ] = Field(
        default_factory=dict,
    )

    stable_features: list[str] = Field(
        default_factory=list,
    )

    unstable_features: list[str] = Field(
        default_factory=list,
    )

    blocking_features: list[str] = Field(
        default_factory=list,
    )

    fitted_on_training_only: bool = True

    comparison_used_for_evaluation_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class DistributionStabilityOptions:
    """Options controlling distribution-stability evaluation."""

    enabled: bool = True

    number_of_bins: int = 10

    population_stability_warning_threshold: float = 0.10
    population_stability_blocking_threshold: float = 0.25

    standardized_mean_shift_warning_threshold: float = 0.20
    standardized_mean_shift_blocking_threshold: float = 0.50

    median_shift_warning_threshold: float = 0.20
    median_shift_blocking_threshold: float = 0.50

    zero_rate_shift_warning_threshold: float = 0.10
    zero_rate_shift_blocking_threshold: float = 0.25

    missing_rate_shift_warning_threshold: float = 0.05
    missing_rate_shift_blocking_threshold: float = 0.20

    epsilon: float = 1e-6
    standard_deviation_epsilon: float = 1e-12

    require_finite_values: bool = True
    allow_missing_values: bool = False

    block_unstable_features: bool = False

    def __post_init__(self) -> None:
        """Validate distribution-stability options."""

        if self.number_of_bins < 2:
            raise ValueError(
                "number_of_bins must be at least two."
            )

        if self.epsilon <= 0.0:
            raise ValueError(
                "epsilon must be positive."
            )

        if self.standard_deviation_epsilon <= 0.0:
            raise ValueError(
                "standard_deviation_epsilon must be positive."
            )

        threshold_values = {
            "population_stability_warning_threshold": (
                self.population_stability_warning_threshold
            ),
            "population_stability_blocking_threshold": (
                self.population_stability_blocking_threshold
            ),
            "standardized_mean_shift_warning_threshold": (
                self.standardized_mean_shift_warning_threshold
            ),
            "standardized_mean_shift_blocking_threshold": (
                self.standardized_mean_shift_blocking_threshold
            ),
            "median_shift_warning_threshold": (
                self.median_shift_warning_threshold
            ),
            "median_shift_blocking_threshold": (
                self.median_shift_blocking_threshold
            ),
            "zero_rate_shift_warning_threshold": (
                self.zero_rate_shift_warning_threshold
            ),
            "zero_rate_shift_blocking_threshold": (
                self.zero_rate_shift_blocking_threshold
            ),
            "missing_rate_shift_warning_threshold": (
                self.missing_rate_shift_warning_threshold
            ),
            "missing_rate_shift_blocking_threshold": (
                self.missing_rate_shift_blocking_threshold
            ),
        }

        for name, value in threshold_values.items():
            if value < 0.0:
                raise ValueError(
                    f"{name} cannot be negative."
                )

        threshold_pairs = [
            (
                self.population_stability_warning_threshold,
                self.population_stability_blocking_threshold,
                "population stability",
            ),
            (
                self.standardized_mean_shift_warning_threshold,
                self.standardized_mean_shift_blocking_threshold,
                "standardized mean shift",
            ),
            (
                self.median_shift_warning_threshold,
                self.median_shift_blocking_threshold,
                "median shift",
            ),
            (
                self.zero_rate_shift_warning_threshold,
                self.zero_rate_shift_blocking_threshold,
                "zero-rate shift",
            ),
            (
                self.missing_rate_shift_warning_threshold,
                self.missing_rate_shift_blocking_threshold,
                "missing-rate shift",
            ),
        ]

        for warning, blocking, metric in threshold_pairs:
            if warning > blocking:
                raise ValueError(
                    f"The {metric} warning threshold cannot "
                    "exceed its blocking threshold."
                )


class DistributionStabilityEvaluator:
    """Compare training distributions with another partition."""

    component_name = "distribution_stability"

    def __init__(
        self,
        options: DistributionStabilityOptions | None = None,
    ) -> None:
        """Initialize the evaluator."""

        self.options = (
            options
            or DistributionStabilityOptions()
        )

    def evaluate(
        self,
        *,
        training: pd.DataFrame,
        comparison: pd.DataFrame,
        features: list[str] | None = None,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
    ) -> DistributionStabilityResult:
        """Evaluate feature distributions without fitting on comparison."""

        if not isinstance(
            training,
            pd.DataFrame,
        ):
            raise TypeError(
                "training must be a pandas DataFrame."
            )

        if not isinstance(
            comparison,
            pd.DataFrame,
        ):
            raise TypeError(
                "comparison must be a pandas DataFrame."
            )

        selected_features = (
            [
                str(column)
                for column in training.columns
            ]
            if features is None
            else [
                str(column)
                for column in features
            ]
        )

        self._validate_frames(
            training=training,
            comparison=comparison,
            features=selected_features,
        )

        if not self.options.enabled:
            return DistributionStabilityResult(
                stable_features=selected_features,
                metadata={
                    "enabled": False,
                },
            )

        required = set(
            required_features or []
        )

        protected = set(
            protected_features or []
        )

        feature_results: dict[
            str,
            FeatureDistributionStability,
        ] = {}

        stable_features: list[str] = []
        unstable_features: list[str] = []
        blocking_features: list[str] = []

        for column in selected_features:
            result = self._evaluate_feature(
                column=column,
                training=training[column],
                comparison=comparison[column],
                required=(
                    column in required
                ),
                protected=(
                    column in protected
                ),
            )

            feature_results[
                column
            ] = result

            if result.unstable:
                unstable_features.append(
                    column
                )

            else:
                stable_features.append(
                    column
                )

            if result.blocks_readiness:
                blocking_features.append(
                    column
                )

        return DistributionStabilityResult(
            valid=not blocking_features,
            features=feature_results,
            stable_features=stable_features,
            unstable_features=unstable_features,
            blocking_features=blocking_features,
            fitted_on_training_only=True,
            comparison_used_for_evaluation_only=True,
            metadata={
                "enabled": True,
                "feature_count": len(
                    selected_features
                ),
                "stable_feature_count": len(
                    stable_features
                ),
                "unstable_feature_count": len(
                    unstable_features
                ),
                "blocking_feature_count": len(
                    blocking_features
                ),
            },
        )

    def _evaluate_feature(
        self,
        *,
        column: str,
        training: pd.Series,
        comparison: pd.Series,
        required: bool,
        protected: bool,
    ) -> FeatureDistributionStability:
        """Calculate stability metrics for one feature."""

        training_numeric = pd.to_numeric(
            training,
            errors="coerce",
        )

        comparison_numeric = pd.to_numeric(
            comparison,
            errors="coerce",
        )

        training_missing_rate = self._rate(
            int(
                training_numeric.isna().sum()
            ),
            len(
                training_numeric
            ),
        )

        comparison_missing_rate = self._rate(
            int(
                comparison_numeric.isna().sum()
            ),
            len(
                comparison_numeric
            ),
        )

        missing_rate_shift = abs(
            comparison_missing_rate
            - training_missing_rate
        )

        training_nonfinite = self._nonfinite_mask(
            training_numeric
        )

        comparison_nonfinite = self._nonfinite_mask(
            comparison_numeric
        )

        if (
            self.options.require_finite_values
            and bool(
                training_nonfinite.any()
            )
        ):
            raise ValueError(
                f"Training feature {column!r} contains "
                "nonfinite values."
            )

        if (
            self.options.require_finite_values
            and bool(
                comparison_nonfinite.any()
            )
        ):
            raise ValueError(
                f"Comparison feature {column!r} contains "
                "nonfinite values."
            )

        if (
            not self.options.allow_missing_values
            and training_numeric.isna().any()
        ):
            raise ValueError(
                f"Training feature {column!r} contains "
                "missing values."
            )

        if (
            not self.options.allow_missing_values
            and comparison_numeric.isna().any()
        ):
            raise ValueError(
                f"Comparison feature {column!r} contains "
                "missing values."
            )

        training_valid = training_numeric.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).dropna()

        comparison_valid = comparison_numeric.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).dropna()

        if training_valid.empty:
            raise ValueError(
                f"Training feature {column!r} contains no "
                "valid numeric values."
            )

        bin_edges = self._training_bin_edges(
            training_valid
        )

        training_rates = self._bin_rates(
            values=training_valid,
            bin_edges=bin_edges,
        )

        comparison_rates = self._bin_rates(
            values=comparison_valid,
            bin_edges=bin_edges,
        )

        population_stability_index = self._psi(
            training_rates=training_rates,
            comparison_rates=comparison_rates,
        )

        training_mean = float(
            training_valid.mean()
        )

        comparison_mean = (
            float(
                comparison_valid.mean()
            )
            if not comparison_valid.empty
            else training_mean
        )

        training_standard_deviation = float(
            training_valid.std(
                ddof=0
            )
        )

        standardized_mean_shift = (
            abs(
                comparison_mean
                - training_mean
            )
            / max(
                abs(
                    training_standard_deviation
                ),
                self.options
                .standard_deviation_epsilon,
            )
        )

        training_median = float(
            training_valid.median()
        )

        comparison_median = (
            float(
                comparison_valid.median()
            )
            if not comparison_valid.empty
            else training_median
        )

        training_iqr = float(
            training_valid.quantile(
                0.75
            )
            - training_valid.quantile(
                0.25
            )
        )

        median_shift = (
            abs(
                comparison_median
                - training_median
            )
            / max(
                abs(
                    training_iqr
                ),
                self.options
                .standard_deviation_epsilon,
            )
        )

        zero_rate_training = self._rate(
            int(
                training_valid.eq(
                    0.0
                ).sum()
            ),
            len(
                training_valid
            ),
        )

        zero_rate_comparison = self._rate(
            int(
                comparison_valid.eq(
                    0.0
                ).sum()
            ),
            len(
                comparison_valid
            ),
        )

        zero_rate_shift = abs(
            zero_rate_comparison
            - zero_rate_training
        )

        reasons: list[str] = []
        blocking_metric = False

        metric_policies = [
            (
                population_stability_index,
                self.options
                .population_stability_warning_threshold,
                self.options
                .population_stability_blocking_threshold,
                "population stability index",
            ),
            (
                standardized_mean_shift,
                self.options
                .standardized_mean_shift_warning_threshold,
                self.options
                .standardized_mean_shift_blocking_threshold,
                "standardized mean shift",
            ),
            (
                median_shift,
                self.options
                .median_shift_warning_threshold,
                self.options
                .median_shift_blocking_threshold,
                "median shift",
            ),
            (
                zero_rate_shift,
                self.options
                .zero_rate_shift_warning_threshold,
                self.options
                .zero_rate_shift_blocking_threshold,
                "zero-rate shift",
            ),
            (
                missing_rate_shift,
                self.options
                .missing_rate_shift_warning_threshold,
                self.options
                .missing_rate_shift_blocking_threshold,
                "missing-rate shift",
            ),
        ]

        for (
            value,
            warning_threshold,
            blocking_threshold,
            metric_name,
        ) in metric_policies:
            if value > blocking_threshold:
                reasons.append(
                    f"{metric_name} exceeds the blocking threshold."
                )

                blocking_metric = True

            elif value > warning_threshold:
                reasons.append(
                    f"{metric_name} exceeds the warning threshold."
                )

        unstable = bool(
            reasons
        )

        exempt_from_blocking = bool(
            required
            or protected
        )

        blocks_readiness = bool(
            blocking_metric
            and self.options.block_unstable_features
            and not exempt_from_blocking
        )

        return FeatureDistributionStability(
            column=column,
            training_row_count=len(
                training
            ),
            comparison_row_count=len(
                comparison
            ),
            population_stability_index=(
                population_stability_index
            ),
            standardized_mean_shift=(
                standardized_mean_shift
            ),
            median_shift=median_shift,
            zero_rate_training=(
                zero_rate_training
            ),
            zero_rate_comparison=(
                zero_rate_comparison
            ),
            zero_rate_shift=(
                zero_rate_shift
            ),
            missing_rate_training=(
                training_missing_rate
            ),
            missing_rate_comparison=(
                comparison_missing_rate
            ),
            missing_rate_shift=(
                missing_rate_shift
            ),
            unstable=unstable,
            blocks_readiness=(
                blocks_readiness
            ),
            instability_reasons=(
                reasons
            ),
            bin_edges=[
                float(value)
                for value in bin_edges
            ],
            training_bin_rates=[
                float(value)
                for value in training_rates
            ],
            comparison_bin_rates=[
                float(value)
                for value in comparison_rates
            ],
            metadata={
                "required_feature": required,
                "protected_feature": protected,
                "blocking_exemption": (
                    exempt_from_blocking
                ),
                "training_mean": (
                    training_mean
                ),
                "comparison_mean": (
                    comparison_mean
                ),
                "training_median": (
                    training_median
                ),
                "comparison_median": (
                    comparison_median
                ),
                "training_standard_deviation": (
                    training_standard_deviation
                ),
                "training_iqr": (
                    training_iqr
                ),
            },
        )

    def _training_bin_edges(
        self,
        training: pd.Series,
    ) -> np.ndarray:
        """Learn deterministic bin edges from training only."""

        unique_count = int(
            training.nunique(
                dropna=True
            )
        )

        if unique_count <= 1:
            value = float(
                training.iloc[
                    0
                ]
            )

            width = max(
                abs(
                    value
                )
                * 1e-9,
                self.options.epsilon,
            )

            return np.array(
                [
                    value - width,
                    value + width,
                ],
                dtype=float,
            )

        quantiles = np.linspace(
            0.0,
            1.0,
            self.options.number_of_bins + 1,
        )

        edges = np.quantile(
            training.to_numpy(
                dtype=float
            ),
            quantiles,
        )

        edges = np.unique(
            edges.astype(
                float
            )
        )

        if len(edges) < 2:
            minimum = float(
                training.min()
            )

            maximum = float(
                training.max()
            )

            return np.array(
                [
                    minimum,
                    maximum,
                ],
                dtype=float,
            )

        edges[
            0
        ] = -np.inf

        edges[
            -1
        ] = np.inf

        return edges

    def _bin_rates(
        self,
        *,
        values: pd.Series,
        bin_edges: np.ndarray,
    ) -> np.ndarray:
        """Calculate rates using training-derived bins."""

        bin_count = max(
            len(
                bin_edges
            )
            - 1,
            1,
        )

        if values.empty:
            return np.full(
                bin_count,
                self.options.epsilon,
                dtype=float,
            )

        counts, _ = np.histogram(
            values.to_numpy(
                dtype=float
            ),
            bins=bin_edges,
        )

        rates = counts.astype(
            float
        ) / max(
            int(
                counts.sum()
            ),
            1,
        )

        rates = np.maximum(
            rates,
            self.options.epsilon,
        )

        return rates / float(rates.sum())

    def _psi(
        self,
        *,
        training_rates: np.ndarray,
        comparison_rates: np.ndarray,
    ) -> float:
        """Calculate the population stability index."""

        training_safe = np.maximum(
            training_rates,
            self.options.epsilon,
        )

        comparison_safe = np.maximum(
            comparison_rates,
            self.options.epsilon,
        )

        value = np.sum(
            (
                comparison_safe
                - training_safe
            )
            * np.log(
                comparison_safe
                / training_safe
            )
        )

        return max(
            0.0,
            float(
                value
            ),
        )

    @staticmethod
    def _rate(
        count: int,
        total: int,
    ) -> float:
        """Calculate a safe rate."""

        if total <= 0:
            return 0.0

        return count / total

    @staticmethod
    def _nonfinite_mask(
        values: pd.Series,
    ) -> pd.Series:
        """Return the nonfinite numeric-value mask."""

        return (
            values.notna()
            & ~np.isfinite(
                values.astype(
                    float
                )
            )
        )

    @staticmethod
    def _validate_frames(
        *,
        training: pd.DataFrame,
        comparison: pd.DataFrame,
        features: list[str],
    ) -> None:
        """Validate distribution-stability inputs."""

        if not features:
            raise ValueError(
                "At least one feature is required."
            )

        missing_training = [
            feature
            for feature in features
            if feature not in training.columns
        ]

        missing_comparison = [
            feature
            for feature in features
            if feature not in comparison.columns
        ]

        if missing_training:
            raise KeyError(
                "Training stability features are missing: "
                f"{missing_training}"
            )

        if missing_comparison:
            raise KeyError(
                "Comparison stability features are missing: "
                f"{missing_comparison}"
            )

        nonnumeric = [
            feature
            for feature in features
            if (
                not pd.api.types.is_numeric_dtype(
                    training[
                        feature
                    ].dtype
                )
                or not pd.api.types.is_numeric_dtype(
                    comparison[
                        feature
                    ].dtype
                )
            )
        ]

        if nonnumeric:
            raise TypeError(
                "Distribution stability requires numeric "
                f"features: {nonnumeric}"
            )
