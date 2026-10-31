"""Training-validation distribution-shift diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import EDAContract


ShiftLevel = Literal[
    "none",
    "low",
    "moderate",
    "high",
    "unavailable",
]


class DistributionShiftDiagnostic(BaseModel):
    """Shift diagnostic for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    feature_family: str

    training_count: int = Field(
        default=0,
        ge=0,
    )

    validation_count: int = Field(
        default=0,
        ge=0,
    )

    psi: float | None = None
    ks_statistic: float | None = None
    total_variation_distance: float | None = None
    jensen_shannon_divergence: float | None = None

    missingness_difference: float | None = None

    validation_only_category_count: int = Field(
        default=0,
        ge=0,
    )

    level: ShiftLevel
    shifted: bool = False

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class DistributionShiftResult(BaseModel):
    """Complete training-validation shift output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    diagnostics: dict[
        str,
        DistributionShiftDiagnostic,
    ] = Field(default_factory=dict)

    shifted_columns: list[str] = Field(
        default_factory=list
    )

    high_shift_columns: list[str] = Field(
        default_factory=list
    )

    validation_available: bool = False


@dataclass(slots=True)
class DistributionShiftOptions:
    """Options controlling distribution-shift diagnostics."""

    enabled: bool = True

    numeric_bin_count: int = 10
    smoothing: float = 1e-6

    moderate_psi_threshold: float = 0.10
    high_psi_threshold: float = 0.25

    moderate_ks_threshold: float = 0.10
    high_ks_threshold: float = 0.20

    moderate_total_variation_threshold: float = 0.10
    high_total_variation_threshold: float = 0.25

    include_missing_as_category: bool = True

    maximum_categories: int = 500

    def __post_init__(self) -> None:
        """Validate shift options."""

        if self.numeric_bin_count < 2:
            raise ValueError(
                "numeric_bin_count must be at least two."
            )

        if self.smoothing <= 0.0:
            raise ValueError(
                "smoothing must be positive."
            )

        threshold_pairs = [
            (
                self.moderate_psi_threshold,
                self.high_psi_threshold,
            ),
            (
                self.moderate_ks_threshold,
                self.high_ks_threshold,
            ),
            (
                self.moderate_total_variation_threshold,
                self.high_total_variation_threshold,
            ),
        ]

        for moderate, high in threshold_pairs:
            if moderate < 0.0 or high < moderate:
                raise ValueError(
                    "Shift thresholds are invalid."
                )


class DistributionShiftAnalyzer:
    """Compare training and validation feature distributions."""

    component_name = "distribution_shift"

    def __init__(
        self,
        options: DistributionShiftOptions | None = None,
    ) -> None:
        """Initialize the shift analyzer."""

        self.options = (
            options or DistributionShiftOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> DistributionShiftResult:
        """Compare training and validation partitions."""

        if not self.options.enabled:
            return DistributionShiftResult(
                validation_available=False
            )

        training = contract.training_features()

        validation = contract.validation_features()

        if validation is None or validation.empty:
            return DistributionShiftResult(
                validation_available=False
            )

        diagnostics: dict[
            str,
            DistributionShiftDiagnostic,
        ] = {}

        for column in columns.numeric:
            if (
                column not in training.columns
                or column not in validation.columns
            ):
                continue

            diagnostics[column] = (
                self._numeric_shift(
                    column=column,
                    training=training[column],
                    validation=validation[column],
                )
            )

        categorical_columns = list(
            dict.fromkeys(
                [
                    *columns.categorical,
                    *columns.boolean,
                ]
            )
        )

        for column in categorical_columns:
            if (
                column not in training.columns
                or column not in validation.columns
            ):
                continue

            diagnostics[column] = (
                self._categorical_shift(
                    column=column,
                    training=training[column],
                    validation=validation[column],
                )
            )

        return DistributionShiftResult(
            diagnostics=diagnostics,
            shifted_columns=[
                column
                for column, diagnostic
                in diagnostics.items()
                if diagnostic.shifted
            ],
            high_shift_columns=[
                column
                for column, diagnostic
                in diagnostics.items()
                if diagnostic.level == "high"
            ],
            validation_available=True,
        )

    def _numeric_shift(
        self,
        *,
        column: str,
        training: pd.Series,
        validation: pd.Series,
    ) -> DistributionShiftDiagnostic:
        """Calculate numeric distribution shift."""

        training_numeric = pd.to_numeric(
            training,
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        validation_numeric = pd.to_numeric(
            validation,
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        training_valid = training_numeric.dropna()
        validation_valid = validation_numeric.dropna()

        missingness_difference = float(
            validation_numeric.isna().mean()
            - training_numeric.isna().mean()
        )

        if (
            training_valid.empty
            or validation_valid.empty
        ):
            return DistributionShiftDiagnostic(
                column=column,
                feature_family="numeric",
                training_count=len(training_valid),
                validation_count=len(
                    validation_valid
                ),
                missingness_difference=(
                    missingness_difference
                ),
                level="unavailable",
            )

        quantiles = np.linspace(
            0.0,
            1.0,
            self.options.numeric_bin_count + 1,
        )

        boundaries = np.unique(
            training_valid.quantile(
                quantiles
            ).to_numpy(
                dtype=float
            )
        )

        if len(boundaries) < 2:
            boundaries = np.array(
                [
                    float(training_valid.min()),
                    float(training_valid.max())
                    + self.options.smoothing,
                ]
            )

        boundaries[0] = -np.inf
        boundaries[-1] = np.inf

        train_counts, _ = np.histogram(
            training_valid,
            bins=boundaries,
        )

        validation_counts, _ = np.histogram(
            validation_valid,
            bins=boundaries,
        )

        train_distribution = self._normalize(
            train_counts
        )

        validation_distribution = self._normalize(
            validation_counts
        )

        psi = self._psi(
            train_distribution,
            validation_distribution,
        )

        ks_statistic = self._ks_statistic(
            training_valid,
            validation_valid,
        )

        level = self._numeric_level(
            psi=psi,
            ks_statistic=ks_statistic,
        )

        return DistributionShiftDiagnostic(
            column=column,
            feature_family="numeric",
            training_count=len(training_valid),
            validation_count=len(
                validation_valid
            ),
            psi=psi,
            ks_statistic=ks_statistic,
            missingness_difference=(
                missingness_difference
            ),
            level=level,
            shifted=level in {
                "moderate",
                "high",
            },
        )

    def _categorical_shift(
        self,
        *,
        column: str,
        training: pd.Series,
        validation: pd.Series,
    ) -> DistributionShiftDiagnostic:
        """Calculate categorical distribution shift."""

        if self.options.include_missing_as_category:
            train_values = training.astype(
                "object"
            ).fillna("__MISSING__")

            validation_values = validation.astype(
                "object"
            ).fillna("__MISSING__")
        else:
            train_values = training.dropna()
            validation_values = validation.dropna()

        categories = list(
            set(
                train_values.unique().tolist()
            ).union(
                validation_values.unique().tolist()
            )
        )

        if (
            len(categories)
            > self.options.maximum_categories
        ):
            return DistributionShiftDiagnostic(
                column=column,
                feature_family="categorical",
                training_count=len(train_values),
                validation_count=len(
                    validation_values
                ),
                level="unavailable",
                metadata={
                    "reason": (
                        "The category count exceeds the "
                        "configured analysis limit."
                    ),
                },
            )

        train_counts = train_values.value_counts()

        validation_counts = (
            validation_values.value_counts()
        )

        train_distribution = np.array(
            [
                int(
                    train_counts.get(
                        category,
                        0,
                    )
                )
                for category in categories
            ],
            dtype=float,
        )

        validation_distribution = np.array(
            [
                int(
                    validation_counts.get(
                        category,
                        0,
                    )
                )
                for category in categories
            ],
            dtype=float,
        )

        train_distribution = self._normalize(
            train_distribution
        )

        validation_distribution = self._normalize(
            validation_distribution
        )

        total_variation = float(
            0.5
            * np.abs(
                train_distribution
                - validation_distribution
            ).sum()
        )

        js_divergence = self._jensen_shannon(
            train_distribution,
            validation_distribution,
        )

        training_categories = set(
            train_values.unique().tolist()
        )

        validation_categories = set(
            validation_values.unique().tolist()
        )

        validation_only_count = len(
            validation_categories.difference(
                training_categories
            )
        )

        level = self._categorical_level(
            total_variation
        )

        return DistributionShiftDiagnostic(
            column=column,
            feature_family="categorical",
            training_count=len(train_values),
            validation_count=len(
                validation_values
            ),
            total_variation_distance=(
                total_variation
            ),
            jensen_shannon_divergence=(
                js_divergence
            ),
            missingness_difference=float(
                validation.isna().mean()
                - training.isna().mean()
            ),
            validation_only_category_count=(
                validation_only_count
            ),
            level=level,
            shifted=level in {
                "moderate",
                "high",
            },
        )

    def _normalize(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Normalize counts with additive smoothing."""

        smoothed = (
            values.astype(float)
            + self.options.smoothing
        )

        return smoothed / smoothed.sum()

    @staticmethod
    def _psi(
        expected: np.ndarray,
        actual: np.ndarray,
    ) -> float:
        """Calculate population stability index."""

        return float(
            (
                (actual - expected)
                * np.log(actual / expected)
            ).sum()
        )

    @staticmethod
    def _ks_statistic(
        training: pd.Series,
        validation: pd.Series,
    ) -> float:
        """Calculate a two-sample KS statistic."""

        values = np.sort(
            np.unique(
                np.concatenate(
                    [
                        training.to_numpy(
                            dtype=float
                        ),
                        validation.to_numpy(
                            dtype=float
                        ),
                    ]
                )
            )
        )

        train_sorted = np.sort(
            training.to_numpy(
                dtype=float
            )
        )

        validation_sorted = np.sort(
            validation.to_numpy(
                dtype=float
            )
        )

        train_cdf = np.searchsorted(
            train_sorted,
            values,
            side="right",
        ) / len(train_sorted)

        validation_cdf = np.searchsorted(
            validation_sorted,
            values,
            side="right",
        ) / len(validation_sorted)

        return float(
            np.max(
                np.abs(
                    train_cdf
                    - validation_cdf
                )
            )
        )

    @staticmethod
    def _jensen_shannon(
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        """Calculate Jensen-Shannon divergence."""

        midpoint = (
            left + right
        ) / 2.0

        left_divergence = float(
            (
                left
                * np.log(
                    left / midpoint
                )
            ).sum()
        )

        right_divergence = float(
            (
                right
                * np.log(
                    right / midpoint
                )
            ).sum()
        )

        return (
            left_divergence
            + right_divergence
        ) / 2.0

    def _numeric_level(
        self,
        *,
        psi: float,
        ks_statistic: float,
    ) -> ShiftLevel:
        """Classify numeric shift."""

        if (
            psi >= self.options.high_psi_threshold
            or ks_statistic
            >= self.options.high_ks_threshold
        ):
            return "high"

        if (
            psi
            >= self.options.moderate_psi_threshold
            or ks_statistic
            >= self.options.moderate_ks_threshold
        ):
            return "moderate"

        if psi > 0.0 or ks_statistic > 0.0:
            return "low"

        return "none"

    def _categorical_level(
        self,
        total_variation: float,
    ) -> ShiftLevel:
        """Classify categorical shift."""

        if (
            total_variation
            >= self.options
            .high_total_variation_threshold
        ):
            return "high"

        if (
            total_variation
            >= self.options
            .moderate_total_variation_threshold
        ):
            return "moderate"

        if total_variation > 0.0:
            return "low"

        return "none"