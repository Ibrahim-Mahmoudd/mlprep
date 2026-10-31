"""Target profiling for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)
from mlprep.types import TaskType


TargetProfileKind = Literal[
    "classification",
    "regression",
]


class TargetProfile(BaseModel):
    """Complete analytical profile of the training target."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    kind: TargetProfileKind
    dtype: str

    row_count: int = Field(ge=0)
    nonmissing_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)

    unique_count: int = Field(ge=0)

    class_counts: dict[str, int] = Field(
        default_factory=dict
    )

    class_percentages: dict[str, float] = Field(
        default_factory=dict
    )

    majority_class: str | None = None
    minority_class: str | None = None

    imbalance_ratio: float | None = None
    entropy: float | None = None

    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None

    zero_count: int = Field(default=0, ge=0)
    negative_count: int = Field(default=0, ge=0)
    infinite_count: int = Field(default=0, ge=0)

    quantiles: dict[str, float | None] = Field(
        default_factory=dict
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class TargetProfilerOptions:
    """Options controlling target profiling."""

    enabled: bool = True

    quantiles: tuple[float, ...] = (
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99,
    )

    include_validation_comparison: bool = True

    def __post_init__(self) -> None:
        """Validate target-profiler options."""

        if any(
            not 0.0 <= quantile <= 1.0
            for quantile in self.quantiles
        ):
            raise ValueError(
                "Every quantile must be between zero and one."
            )


class TargetProfiler:
    """Profile a classification or regression target."""

    component_name = "target_profile"

    def __init__(
        self,
        options: TargetProfilerOptions | None = None,
    ) -> None:
        """Initialize the target profiler."""

        self.options = (
            options or TargetProfilerOptions()
        )

    def profile(
        self,
        *,
        contract: EDAContract,
    ) -> TargetProfile:
        """Profile the training target."""

        target = contract.training_target()

        task_type = (
            contract.phase1_state
            .problem_contract
            .task_type
            if contract.phase1_state
            .problem_contract
            is not None
            else None
        )

        if task_type == TaskType.REGRESSION:
            return self._profile_regression(
                column=contract.target_column,
                target=target,
                contract=contract,
            )

        return self._profile_classification(
            column=contract.target_column,
            target=target,
            contract=contract,
        )

    def _profile_classification(
        self,
        *,
        column: str,
        target: pd.Series,
        contract: EDAContract,
    ) -> TargetProfile:
        """Profile a classification target."""

        row_count = len(target)
        missing_count = int(
            target.isna().sum()
        )

        nonmissing = target.dropna()
        nonmissing_count = len(nonmissing)

        counts = nonmissing.value_counts(
            dropna=False
        )

        percentages = (
            counts / nonmissing_count * 100.0
            if nonmissing_count > 0
            else counts.astype(float)
        )

        class_counts = {
            str(value): int(count)
            for value, count
            in counts.items()
        }

        class_percentages = {
            str(value): float(percentage)
            for value, percentage
            in percentages.items()
        }

        majority_class = None
        minority_class = None
        imbalance_ratio = None

        if not counts.empty:
            majority_class = str(
                counts.index[0]
            )

            minority_class = str(
                counts.index[-1]
            )

            minority_count = int(
                counts.iloc[-1]
            )

            imbalance_ratio = (
                int(counts.iloc[0])
                / minority_count
                if minority_count > 0
                else None
            )

        probabilities = (
            counts / nonmissing_count
            if nonmissing_count > 0
            else counts.astype(float)
        )

        entropy = -sum(
            float(probability)
            * log2(float(probability))
            for probability
            in probabilities.tolist()
            if probability > 0
        )

        metadata = self._validation_metadata(
            contract=contract,
            training_target=target,
            classification=True,
        )

        return TargetProfile(
            column=column,
            kind="classification",
            dtype=str(target.dtype),
            row_count=row_count,
            nonmissing_count=nonmissing_count,
            missing_count=missing_count,
            unique_count=int(
                nonmissing.nunique()
            ),
            class_counts=class_counts,
            class_percentages=(
                class_percentages
            ),
            majority_class=majority_class,
            minority_class=minority_class,
            imbalance_ratio=imbalance_ratio,
            entropy=float(entropy),
            metadata=metadata,
        )

    def _profile_regression(
        self,
        *,
        column: str,
        target: pd.Series,
        contract: EDAContract,
    ) -> TargetProfile:
        """Profile a regression target."""

        numeric = pd.to_numeric(
            target,
            errors="coerce",
        )

        row_count = len(numeric)
        missing_count = int(
            numeric.isna().sum()
        )

        nonmissing = numeric.dropna()
        nonmissing_count = len(nonmissing)

        finite = nonmissing[
            np.isfinite(
                nonmissing.astype(float)
            )
        ]

        infinite_count = (
            nonmissing_count - len(finite)
        )

        metadata = self._validation_metadata(
            contract=contract,
            training_target=numeric,
            classification=False,
        )

        return TargetProfile(
            column=column,
            kind="regression",
            dtype=str(target.dtype),
            row_count=row_count,
            nonmissing_count=nonmissing_count,
            missing_count=missing_count,
            unique_count=int(
                nonmissing.nunique()
            ),
            minimum=self._safe_float(
                finite.min()
            )
            if not finite.empty
            else None,
            maximum=self._safe_float(
                finite.max()
            )
            if not finite.empty
            else None,
            mean=self._safe_float(
                finite.mean()
            )
            if not finite.empty
            else None,
            median=self._safe_float(
                finite.median()
            )
            if not finite.empty
            else None,
            standard_deviation=(
                self._safe_float(
                    finite.std(ddof=0)
                )
                if not finite.empty
                else None
            ),
            skewness=(
                self._safe_float(
                    finite.skew()
                )
                if len(finite) >= 3
                else None
            ),
            kurtosis=(
                self._safe_float(
                    finite.kurtosis()
                )
                if len(finite) >= 4
                else None
            ),
            zero_count=int(
                (finite == 0).sum()
            ),
            negative_count=int(
                (finite < 0).sum()
            ),
            infinite_count=infinite_count,
            quantiles={
                str(quantile): (
                    self._safe_float(
                        finite.quantile(
                            quantile
                        )
                    )
                    if not finite.empty
                    else None
                )
                for quantile
                in self.options.quantiles
            },
            metadata=metadata,
        )

    def _validation_metadata(
        self,
        *,
        contract: EDAContract,
        training_target: pd.Series,
        classification: bool,
    ) -> dict[str, Any]:
        """Build optional validation-target comparison metadata."""

        if not (
            self.options
            .include_validation_comparison
            and contract.validation is not None
        ):
            return {}

        validation_target = (
            contract.validation_target()
        )

        if validation_target is None:
            return {}

        if classification:
            training_distribution = (
                training_target.value_counts(
                    normalize=True,
                    dropna=False,
                )
            )

            validation_distribution = (
                validation_target.value_counts(
                    normalize=True,
                    dropna=False,
                )
            )

            classes = set(
                training_distribution.index.tolist()
            ).union(
                validation_distribution.index.tolist()
            )

            return {
                "validation_class_percentage_difference": {
                    str(value): (
                        float(
                            validation_distribution.get(
                                value,
                                0.0,
                            )
                        )
                        - float(
                            training_distribution.get(
                                value,
                                0.0,
                            )
                        )
                    )
                    for value in classes
                }
            }

        validation_numeric = pd.to_numeric(
            validation_target,
            errors="coerce",
        ).dropna()

        training_numeric = pd.to_numeric(
            training_target,
            errors="coerce",
        ).dropna()

        return {
            "validation_mean_difference": (
                float(
                    validation_numeric.mean()
                    - training_numeric.mean()
                )
                if (
                    not validation_numeric.empty
                    and not training_numeric.empty
                )
                else None
            ),
            "validation_median_difference": (
                float(
                    validation_numeric.median()
                    - training_numeric.median()
                )
                if (
                    not validation_numeric.empty
                    and not training_numeric.empty
                )
                else None
            ),
        }

    @staticmethod
    def _safe_float(
        value: object,
    ) -> float | None:
        """Convert a numeric value safely."""

        if pd.isna(value):
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result