"""Rare-category diagnostics for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import EDAContract


class RareCategoryDiagnostic(BaseModel):
    """Rare-category diagnostic for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    valid_count: int = Field(
        default=0,
        ge=0,
    )

    unique_count: int = Field(
        default=0,
        ge=0,
    )

    rare_category_count: int = Field(
        default=0,
        ge=0,
    )

    rare_observation_count: int = Field(
        default=0,
        ge=0,
    )

    rare_observation_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    rare_categories: list[str] = Field(
        default_factory=list
    )

    validation_only_categories: list[str] = Field(
        default_factory=list
    )

    training_only_categories: list[str] = Field(
        default_factory=list
    )

    recommend_grouping: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class RareCategoryDiagnosticResult(BaseModel):
    """Complete rare-category diagnostic output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    diagnostics: dict[
        str,
        RareCategoryDiagnostic,
    ] = Field(default_factory=dict)

    grouping_candidates: list[str] = Field(
        default_factory=list
    )

    unknown_category_risk_columns: list[str] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class RareCategoryDiagnosticOptions:
    """Options controlling rare-category diagnostics."""

    enabled: bool = True

    minimum_count: int = 5
    maximum_ratio: float = 0.01

    grouping_category_count_threshold: int = 2
    grouping_observation_percentage_threshold: float = 0.0

    compare_validation: bool = True

    maximum_reported_categories: int = 100

    include_boolean_columns: bool = False

    def __post_init__(self) -> None:
        """Validate rare-category options."""

        if self.minimum_count < 1:
            raise ValueError(
                "minimum_count must be positive."
            )

        if not 0.0 <= self.maximum_ratio <= 1.0:
            raise ValueError(
                "maximum_ratio must be between zero and one."
            )

        if self.grouping_category_count_threshold < 1:
            raise ValueError(
                "grouping_category_count_threshold must be "
                "positive."
            )

        if not (
            0.0
            <= self.grouping_observation_percentage_threshold
            <= 100.0
        ):
            raise ValueError(
                "grouping_observation_percentage_threshold "
                "must be between zero and one hundred."
            )


class RareCategoryDiagnosticAnalyzer:
    """Detect rare and validation-only categories."""

    component_name = "rare_categories"

    def __init__(
        self,
        options: (
            RareCategoryDiagnosticOptions | None
        ) = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options or RareCategoryDiagnosticOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> RareCategoryDiagnosticResult:
        """Analyze selected categorical features."""

        if not self.options.enabled:
            return RareCategoryDiagnosticResult()

        training = contract.training_features()

        validation = (
            contract.validation_features()
            if self.options.compare_validation
            else None
        )

        selected_columns = list(
            columns.categorical
        )

        if self.options.include_boolean_columns:
            selected_columns.extend(
                columns.boolean
            )

        diagnostics: dict[
            str,
            RareCategoryDiagnostic,
        ] = {}

        for column in dict.fromkeys(
            selected_columns
        ):
            if column not in training.columns:
                continue

            diagnostic = self._analyze_column(
                column=column,
                training=training[column],
                validation=(
                    validation[column]
                    if (
                        validation is not None
                        and column in validation.columns
                    )
                    else None
                ),
            )

            diagnostics[column] = diagnostic

        return RareCategoryDiagnosticResult(
            diagnostics=diagnostics,
            grouping_candidates=[
                column
                for column, diagnostic
                in diagnostics.items()
                if diagnostic.recommend_grouping
            ],
            unknown_category_risk_columns=[
                column
                for column, diagnostic
                in diagnostics.items()
                if diagnostic.validation_only_categories
            ],
        )

    def _analyze_column(
        self,
        *,
        column: str,
        training: pd.Series,
        validation: pd.Series | None,
    ) -> RareCategoryDiagnostic:
        """Analyze one categorical feature."""

        training_values = training.dropna()

        counts = training_values.value_counts(
            dropna=False
        )

        valid_count = int(
            counts.sum()
        )

        unique_count = len(counts)

        rare_values: list[object] = []

        rare_observation_count = 0

        for value, count in counts.items():
            ratio = (
                int(count) / valid_count
                if valid_count > 0
                else 0.0
            )

            if (
                int(count) < self.options.minimum_count
                or ratio <= self.options.maximum_ratio
            ):
                rare_values.append(value)

                rare_observation_count += int(
                    count
                )

        rare_percentage = (
            rare_observation_count
            / valid_count
            * 100.0
            if valid_count > 0
            else 0.0
        )

        validation_only: list[str] = []
        training_only: list[str] = []

        if validation is not None:
            training_categories = set(
                training_values.unique().tolist()
            )

            validation_categories = set(
                validation
                .dropna()
                .unique()
                .tolist()
            )

            validation_only = sorted(
                str(value)
                for value
                in validation_categories.difference(
                    training_categories
                )
            )

            training_only = sorted(
                str(value)
                for value
                in training_categories.difference(
                    validation_categories
                )
            )

        recommend_grouping = bool(
            len(rare_values)
            >= self.options
            .grouping_category_count_threshold
            and rare_percentage
            >= self.options
            .grouping_observation_percentage_threshold
        )

        return RareCategoryDiagnostic(
            column=column,
            valid_count=valid_count,
            unique_count=unique_count,
            rare_category_count=len(
                rare_values
            ),
            rare_observation_count=(
                rare_observation_count
            ),
            rare_observation_percentage=round(
                rare_percentage,
                6,
            ),
            rare_categories=[
                str(value)
                for value in rare_values[
                    : self.options
                    .maximum_reported_categories
                ]
            ],
            validation_only_categories=(
                validation_only[
                    : self.options
                    .maximum_reported_categories
                ]
            ),
            training_only_categories=(
                training_only[
                    : self.options
                    .maximum_reported_categories
                ]
            ),
            recommend_grouping=recommend_grouping,
            metadata={
                "reported_rare_categories_truncated": (
                    len(rare_values)
                    > self.options
                    .maximum_reported_categories
                ),
            },
        )