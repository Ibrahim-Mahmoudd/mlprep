"""Feature-to-target association analysis for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)
from mlprep.types import TaskType


TargetAssociationKind = Literal[
    "numeric_classification",
    "categorical_classification",
    "numeric_regression",
    "categorical_regression",
]


class TargetAssociation(BaseModel):
    """Association between one feature and the target."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    feature: str
    kind: TargetAssociationKind

    valid_pair_count: int = Field(
        default=0,
        ge=0,
    )

    association_score: float | None = None

    pearson: float | None = None
    spearman: float | None = None

    eta_squared: float | None = None
    cramers_v: float | None = None

    suspiciously_strong: bool = False

    details: dict[str, object] = Field(
        default_factory=dict
    )


class TargetAssociationResult(BaseModel):
    """Complete feature-to-target association output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    target_column: str
    task_type: str

    associations: list[TargetAssociation] = Field(
        default_factory=list
    )

    strongest_features: list[str] = Field(
        default_factory=list
    )

    suspicious_features: list[str] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class TargetAssociationOptions:
    """Options controlling feature-to-target analysis."""

    enabled: bool = True

    minimum_valid_pairs: int = 5

    maximum_categories: int = 100

    suspicious_association_threshold: float = 0.999

    maximum_ranked_features: int = 50

    def __post_init__(self) -> None:
        """Validate target-association options."""

        if self.minimum_valid_pairs < 2:
            raise ValueError(
                "minimum_valid_pairs must be at least two."
            )

        if self.maximum_categories < 2:
            raise ValueError(
                "maximum_categories must be at least two."
            )

        if not (
            0.0
            <= self.suspicious_association_threshold
            <= 1.0
        ):
            raise ValueError(
                "suspicious_association_threshold must be "
                "between zero and one."
            )

        if self.maximum_ranked_features < 1:
            raise ValueError(
                "maximum_ranked_features must be positive."
            )


class TargetAssociationAnalyzer:
    """Analyze every selected feature against the target."""

    component_name = "target_associations"

    def __init__(
        self,
        options: TargetAssociationOptions | None = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options or TargetAssociationOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> TargetAssociationResult:
        """Analyze feature-to-target associations."""

        task_type = self._task_type(
            contract
        )

        if not self.options.enabled:
            return TargetAssociationResult(
                target_column=contract.target_column,
                task_type=task_type.value,
            )

        frame = contract.train.X
        target = contract.training_target()

        associations: list[
            TargetAssociation
        ] = []

        classification = task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }

        for column in columns.numeric:
            if column not in frame.columns:
                continue

            if classification:
                association = (
                    self._numeric_classification(
                        feature=column,
                        values=frame[column],
                        target=target,
                    )
                )
            else:
                association = self._numeric_regression(
                    feature=column,
                    values=frame[column],
                    target=target,
                )

            if association is not None:
                associations.append(
                    association
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
            if column not in frame.columns:
                continue

            if classification:
                association = (
                    self._categorical_classification(
                        feature=column,
                        values=frame[column],
                        target=target,
                    )
                )
            else:
                association = (
                    self._categorical_regression(
                        feature=column,
                        values=frame[column],
                        target=target,
                    )
                )

            if association is not None:
                associations.append(
                    association
                )

        ranked_associations = sorted(
            associations,
            key=lambda association: (
                association.association_score
                or 0.0
            ),
            reverse=True,
        )

        return TargetAssociationResult(
            target_column=contract.target_column,
            task_type=task_type.value,
            associations=ranked_associations,
            strongest_features=[
                association.feature
                for association
                in ranked_associations[
                    : self.options.maximum_ranked_features
                ]
            ],
            suspicious_features=[
                association.feature
                for association
                in ranked_associations
                if association.suspiciously_strong
            ],
        )

    def _numeric_classification(
        self,
        *,
        feature: str,
        values: pd.Series,
        target: pd.Series,
    ) -> TargetAssociation | None:
        """Analyze numeric feature versus classification target."""

        frame = pd.DataFrame(
            {
                "feature": pd.to_numeric(
                    values,
                    errors="coerce",
                ),
                "target": target,
            }
        ).dropna()

        if len(frame) < self.options.minimum_valid_pairs:
            return None

        if frame["target"].nunique() < 2:
            return None

        overall_mean = float(
            frame["feature"].mean()
        )

        total_sum_squares = float(
            (
                frame["feature"]
                - overall_mean
            ).pow(2).sum()
        )

        between_sum_squares = 0.0
        group_means: dict[str, float] = {}

        for label, group in frame.groupby(
            "target",
            dropna=False,
        ):
            group_mean = float(
                group["feature"].mean()
            )

            group_means[str(label)] = group_mean

            between_sum_squares += (
                len(group)
                * (group_mean - overall_mean) ** 2
            )

        eta_squared = (
            between_sum_squares
            / total_sum_squares
            if total_sum_squares > 0
            else None
        )

        return TargetAssociation(
            feature=feature,
            kind="numeric_classification",
            valid_pair_count=len(frame),
            association_score=eta_squared,
            eta_squared=eta_squared,
            suspiciously_strong=bool(
                eta_squared is not None
                and eta_squared
                >= self.options
                .suspicious_association_threshold
            ),
            details={
                "target_group_means": group_means,
            },
        )

    def _categorical_classification(
        self,
        *,
        feature: str,
        values: pd.Series,
        target: pd.Series,
    ) -> TargetAssociation | None:
        """Analyze categorical feature versus classification target."""

        frame = pd.DataFrame(
            {
                "feature": values,
                "target": target,
            }
        ).dropna()

        if len(frame) < self.options.minimum_valid_pairs:
            return None

        if (
            frame["feature"].nunique()
            > self.options.maximum_categories
        ):
            return None

        contingency = pd.crosstab(
            frame["feature"],
            frame["target"],
        )

        if min(contingency.shape) < 2:
            return None

        observed = contingency.to_numpy(
            dtype=float
        )

        total = float(observed.sum())

        expected = (
            observed.sum(axis=1, keepdims=True)
            @ observed.sum(axis=0, keepdims=True)
        ) / total

        valid_cells = expected > 0

        chi_square = float(
            (
                (
                    observed[valid_cells]
                    - expected[valid_cells]
                )
                ** 2
                / expected[valid_cells]
            ).sum()
        )

        denominator = min(
            contingency.shape[0] - 1,
            contingency.shape[1] - 1,
        )

        cramers_v = (
            sqrt(
                (chi_square / total)
                / denominator
            )
            if denominator > 0
            else None
        )

        if cramers_v is not None:
            cramers_v = min(
                1.0,
                cramers_v,
            )

        return TargetAssociation(
            feature=feature,
            kind="categorical_classification",
            valid_pair_count=len(frame),
            association_score=cramers_v,
            cramers_v=cramers_v,
            suspiciously_strong=bool(
                cramers_v is not None
                and cramers_v
                >= self.options
                .suspicious_association_threshold
            ),
        )

    def _numeric_regression(
        self,
        *,
        feature: str,
        values: pd.Series,
        target: pd.Series,
    ) -> TargetAssociation | None:
        """Analyze numeric feature versus regression target."""

        frame = pd.DataFrame(
            {
                "feature": pd.to_numeric(
                    values,
                    errors="coerce",
                ),
                "target": pd.to_numeric(
                    target,
                    errors="coerce",
                ),
            }
        ).dropna()

        if len(frame) < self.options.minimum_valid_pairs:
            return None

        pearson = self._safe_correlation(
            frame["feature"],
            frame["target"],
            method="pearson",
        )

        spearman = self._safe_correlation(
            frame["feature"],
            frame["target"],
            method="spearman",
        )

        score_candidates = [
            abs(value)
            for value in (
                pearson,
                spearman,
            )
            if value is not None
        ]

        score = (
            max(score_candidates)
            if score_candidates
            else None
        )

        return TargetAssociation(
            feature=feature,
            kind="numeric_regression",
            valid_pair_count=len(frame),
            association_score=score,
            pearson=pearson,
            spearman=spearman,
            suspiciously_strong=bool(
                score is not None
                and score
                >= self.options
                .suspicious_association_threshold
            ),
        )

    def _categorical_regression(
        self,
        *,
        feature: str,
        values: pd.Series,
        target: pd.Series,
    ) -> TargetAssociation | None:
        """Analyze categorical feature versus regression target."""

        frame = pd.DataFrame(
            {
                "feature": values,
                "target": pd.to_numeric(
                    target,
                    errors="coerce",
                ),
            }
        ).dropna()

        if len(frame) < self.options.minimum_valid_pairs:
            return None

        if (
            frame["feature"].nunique()
            > self.options.maximum_categories
        ):
            return None

        if frame["feature"].nunique() < 2:
            return None

        overall_mean = float(
            frame["target"].mean()
        )

        total_sum_squares = float(
            (
                frame["target"]
                - overall_mean
            ).pow(2).sum()
        )

        between_sum_squares = 0.0

        for _, group in frame.groupby(
            "feature",
            dropna=False,
        ):
            group_mean = float(
                group["target"].mean()
            )

            between_sum_squares += (
                len(group)
                * (group_mean - overall_mean) ** 2
            )

        eta_squared = (
            between_sum_squares
            / total_sum_squares
            if total_sum_squares > 0
            else None
        )

        return TargetAssociation(
            feature=feature,
            kind="categorical_regression",
            valid_pair_count=len(frame),
            association_score=eta_squared,
            eta_squared=eta_squared,
            suspiciously_strong=bool(
                eta_squared is not None
                and eta_squared
                >= self.options
                .suspicious_association_threshold
            ),
        )

    @staticmethod
    def _safe_correlation(
        left: pd.Series,
        right: pd.Series,
        *,
        method: str,
    ) -> float | None:
        """Calculate a finite correlation value."""

        value = left.corr(
            right,
            method=method,
        )

        if pd.isna(value):
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result

    @staticmethod
    def _task_type(
        contract: EDAContract,
    ) -> TaskType:
        """Resolve the Phase 1 task type."""

        problem_contract = (
            contract.phase1_state
            .problem_contract
        )

        if problem_contract is None:
            raise ValueError(
                "Phase 3 target analysis requires the "
                "Phase 1 problem contract."
            )

        return problem_contract.task_type