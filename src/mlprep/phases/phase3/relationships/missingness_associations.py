"""Missingness-association analysis for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)
from mlprep.types import TaskType


class MissingnessTargetAssociation(BaseModel):
    """Association between feature missingness and target."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    feature: str

    missing_count: int = Field(
        default=0,
        ge=0,
    )

    nonmissing_count: int = Field(
        default=0,
        ge=0,
    )

    missing_target_mean: float | None = None
    nonmissing_target_mean: float | None = None

    target_mean_difference: float | None = None

    missing_positive_rate: float | None = None
    nonmissing_positive_rate: float | None = None

    positive_rate_difference: float | None = None

    association_score: float | None = None
    strong_association: bool = False


class MissingnessAssociationResult(BaseModel):
    """Complete missingness-association output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    target_associations: list[
        MissingnessTargetAssociation
    ] = Field(default_factory=list)

    strongly_associated_features: list[str] = Field(
        default_factory=list
    )

    missingness_correlations: dict[
        str,
        dict[str, float | None],
    ] = Field(default_factory=dict)


@dataclass(slots=True)
class MissingnessAssociationOptions:
    """Options controlling missingness associations."""

    enabled: bool = True

    minimum_missing_count: int = 5
    minimum_nonmissing_count: int = 5

    strong_association_threshold: float = 0.10

    calculate_missingness_correlations: bool = True

    def __post_init__(self) -> None:
        """Validate missingness-association options."""

        if self.minimum_missing_count < 1:
            raise ValueError(
                "minimum_missing_count must be positive."
            )

        if self.minimum_nonmissing_count < 1:
            raise ValueError(
                "minimum_nonmissing_count must be positive."
            )

        if self.strong_association_threshold < 0:
            raise ValueError(
                "strong_association_threshold cannot be "
                "negative."
            )


class MissingnessAssociationAnalyzer:
    """Analyze relationships involving missing-value indicators."""

    component_name = "missingness_associations"

    def __init__(
        self,
        options: (
            MissingnessAssociationOptions | None
        ) = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options
            or MissingnessAssociationOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
    ) -> MissingnessAssociationResult:
        """Analyze missingness relationships."""

        if not self.options.enabled:
            return MissingnessAssociationResult()

        frame = contract.train.X
        target = contract.training_target()

        problem_contract = (
            contract.phase1_state
            .problem_contract
        )

        if problem_contract is None:
            raise ValueError(
                "Missingness-target analysis requires "
                "the Phase 1 problem contract."
            )

        classification = (
            problem_contract.task_type
            in {
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTICLASS_CLASSIFICATION,
            }
        )

        positive_class = (
            problem_contract.positive_class
        )

        associations: list[
            MissingnessTargetAssociation
        ] = []

        for column in frame.columns:
            missing_mask = frame[column].isna()

            missing_count = int(
                missing_mask.sum()
            )

            nonmissing_count = int(
                (~missing_mask).sum()
            )

            if (
                missing_count
                < self.options.minimum_missing_count
                or nonmissing_count
                < self.options
                .minimum_nonmissing_count
            ):
                continue

            if (
                classification
                and positive_class is not None
            ):
                association = (
                    self._classification_association(
                        feature=str(column),
                        missing_mask=missing_mask,
                        target=target,
                        positive_class=positive_class,
                    )
                )
            else:
                association = (
                    self._numeric_target_association(
                        feature=str(column),
                        missing_mask=missing_mask,
                        target=target,
                    )
                )

            if association is not None:
                associations.append(
                    association
                )

        correlations: dict[
            str,
            dict[str, float | None],
        ] = {}

        if (
            self.options
            .calculate_missingness_correlations
        ):
            correlations = self._correlations(
                frame.isna()
            )

        strong_features = [
            association.feature
            for association in associations
            if association.strong_association
        ]

        associations.sort(
            key=lambda association: (
                association.association_score
                or 0.0
            ),
            reverse=True,
        )

        return MissingnessAssociationResult(
            target_associations=associations,
            strongly_associated_features=(
                strong_features
            ),
            missingness_correlations=correlations,
        )

    def _classification_association(
        self,
        *,
        feature: str,
        missing_mask: pd.Series,
        target: pd.Series,
        positive_class: object,
    ) -> MissingnessTargetAssociation:
        """Analyze missingness against binary positive rate."""

        target_is_positive = (
            target == positive_class
        )

        missing_rate = float(
            target_is_positive.loc[
                missing_mask
            ].mean()
        )

        nonmissing_rate = float(
            target_is_positive.loc[
                ~missing_mask
            ].mean()
        )

        difference = abs(
            missing_rate - nonmissing_rate
        )

        return MissingnessTargetAssociation(
            feature=feature,
            missing_count=int(
                missing_mask.sum()
            ),
            nonmissing_count=int(
                (~missing_mask).sum()
            ),
            missing_positive_rate=missing_rate,
            nonmissing_positive_rate=(
                nonmissing_rate
            ),
            positive_rate_difference=(
                difference
            ),
            association_score=difference,
            strong_association=(
                difference
                >= self.options
                .strong_association_threshold
            ),
        )

    def _numeric_target_association(
        self,
        *,
        feature: str,
        missing_mask: pd.Series,
        target: pd.Series,
    ) -> MissingnessTargetAssociation | None:
        """Analyze missingness against a numeric target."""

        numeric_target = pd.to_numeric(
            target,
            errors="coerce",
        )

        missing_values = numeric_target.loc[
            missing_mask
        ].dropna()

        nonmissing_values = numeric_target.loc[
            ~missing_mask
        ].dropna()

        if (
            missing_values.empty
            or nonmissing_values.empty
        ):
            return None

        missing_mean = float(
            missing_values.mean()
        )

        nonmissing_mean = float(
            nonmissing_values.mean()
        )

        denominator = max(
            abs(missing_mean),
            abs(nonmissing_mean),
            1e-12,
        )

        difference = (
            abs(
                missing_mean
                - nonmissing_mean
            )
            / denominator
        )

        return MissingnessTargetAssociation(
            feature=feature,
            missing_count=int(
                missing_mask.sum()
            ),
            nonmissing_count=int(
                (~missing_mask).sum()
            ),
            missing_target_mean=missing_mean,
            nonmissing_target_mean=(
                nonmissing_mean
            ),
            target_mean_difference=(
                difference
            ),
            association_score=difference,
            strong_association=(
                difference
                >= self.options
                .strong_association_threshold
            ),
        )

    @staticmethod
    def _correlations(
        missing_mask: pd.DataFrame,
    ) -> dict[
        str,
        dict[str, float | None],
    ]:
        """Calculate missingness-indicator correlations."""

        useful_columns = [
            column
            for column in missing_mask.columns
            if missing_mask[column].nunique() > 1
        ]

        if len(useful_columns) < 2:
            return {}

        matrix = (
            missing_mask.loc[
                :,
                useful_columns,
            ]
            .astype(int)
            .corr()
        )

        return {
            str(row): {
                str(column): (
                    None
                    if pd.isna(value)
                    else float(value)
                )
                for column, value
                in matrix.loc[row].items()
            }
            for row in matrix.index
        }