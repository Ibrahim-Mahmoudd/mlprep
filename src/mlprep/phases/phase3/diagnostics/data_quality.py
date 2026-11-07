"""General analytical data-quality diagnostics for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import EDAContract

QualitySeverity = Literal[
    "info",
    "warning",
    "error",
]


class DataQualityIssue(BaseModel):
    """One analytical data-quality issue."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    code: str
    severity: QualitySeverity

    message: str

    columns: list[str] = Field(
        default_factory=list
    )

    affected_row_count: int = Field(
        default=0,
        ge=0,
    )

    recommendation: str | None = None

    details: dict[str, object] = Field(
        default_factory=dict
    )


class DataQualityDiagnosticResult(BaseModel):
    """Complete data-quality diagnostic output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    issues: list[DataQualityIssue] = Field(
        default_factory=list
    )

    constant_columns: list[str] = Field(
        default_factory=list
    )

    near_constant_columns: list[str] = Field(
        default_factory=list
    )

    completely_missing_columns: list[str] = Field(
        default_factory=list
    )

    highly_missing_columns: list[str] = Field(
        default_factory=list
    )

    sparse_columns: list[str] = Field(
        default_factory=list
    )

    identifier_risk_columns: list[str] = Field(
        default_factory=list
    )

    insufficient_information_columns: list[str] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class DataQualityDiagnosticOptions:
    """Options controlling general quality diagnostics."""

    enabled: bool = True

    near_constant_threshold: float = 0.995

    high_missingness_threshold: float = 0.50

    numeric_zero_sparsity_threshold: float = 0.95

    identifier_uniqueness_threshold: float = 0.95

    minimum_nonmissing_count: int = 2

    def __post_init__(self) -> None:
        """Validate quality-diagnostic options."""

        bounded_values = {
            "near_constant_threshold": (
                self.near_constant_threshold
            ),
            "high_missingness_threshold": (
                self.high_missingness_threshold
            ),
            "numeric_zero_sparsity_threshold": (
                self.numeric_zero_sparsity_threshold
            ),
            "identifier_uniqueness_threshold": (
                self.identifier_uniqueness_threshold
            ),
        }

        for name, value in bounded_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between zero and one."
                )

        if self.minimum_nonmissing_count < 1:
            raise ValueError(
                "minimum_nonmissing_count must be positive."
            )


class DataQualityDiagnosticAnalyzer:
    """Detect broad quality risks in training features."""

    component_name = "data_quality"

    def __init__(
        self,
        options: (
            DataQualityDiagnosticOptions | None
        ) = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options or DataQualityDiagnosticOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> DataQualityDiagnosticResult:
        """Run general analytical data-quality checks."""

        if not self.options.enabled:
            return DataQualityDiagnosticResult()

        frame = contract.training_features()

        issues: list[DataQualityIssue] = []

        constant_columns: list[str] = []
        near_constant_columns: list[str] = []
        completely_missing_columns: list[str] = []
        highly_missing_columns: list[str] = []
        sparse_columns: list[str] = []
        identifier_risk_columns: list[str] = []
        insufficient_columns: list[str] = []

        for column in frame.columns:
            series = frame[column]

            missing_count = int(
                series.isna().sum()
            )

            missing_ratio = float(
                series.isna().mean()
            )

            nonmissing = series.dropna()

            if nonmissing.empty:
                completely_missing_columns.append(
                    str(column)
                )

                issues.append(
                    DataQualityIssue(
                        code="P3-QUALITY-001",
                        severity="error",
                        message=(
                            f"Feature {column!r} is completely missing."
                        ),
                        columns=[str(column)],
                        affected_row_count=len(series),
                        recommendation=(
                            "Review whether the feature should be "
                            "removed or supplied by the data source."
                        ),
                    )
                )

                continue

            if (
                len(nonmissing)
                < self.options.minimum_nonmissing_count
            ):
                insufficient_columns.append(
                    str(column)
                )

            unique_count = int(
                nonmissing.nunique()
            )

            if unique_count <= 1:
                constant_columns.append(
                    str(column)
                )

                issues.append(
                    DataQualityIssue(
                        code="P3-QUALITY-002",
                        severity="warning",
                        message=(
                            f"Feature {column!r} is constant."
                        ),
                        columns=[str(column)],
                        recommendation=(
                            "Consider removing the feature in Phase 8."
                        ),
                    )
                )

            else:
                dominant_ratio = float(
                    nonmissing.value_counts(
                        normalize=True,
                        dropna=False,
                    ).iloc[0]
                )

                if (
                    dominant_ratio
                    >= self.options
                    .near_constant_threshold
                ):
                    near_constant_columns.append(
                        str(column)
                    )

                    issues.append(
                        DataQualityIssue(
                            code="P3-QUALITY-003",
                            severity="warning",
                            message=(
                                f"Feature {column!r} is near-constant."
                            ),
                            columns=[str(column)],
                            recommendation=(
                                "Evaluate low-variance removal "
                                "during Phase 8."
                            ),
                            details={
                                "dominant_ratio": (
                                    dominant_ratio
                                ),
                            },
                        )
                    )

            if (
                missing_ratio
                >= self.options
                .high_missingness_threshold
            ):
                highly_missing_columns.append(
                    str(column)
                )

                issues.append(
                    DataQualityIssue(
                        code="P3-QUALITY-004",
                        severity="warning",
                        message=(
                            f"Feature {column!r} has high missingness."
                        ),
                        columns=[str(column)],
                        affected_row_count=missing_count,
                        recommendation=(
                            "Phase 4 should review imputation and "
                            "missing-indicator strategies."
                        ),
                        details={
                            "missing_ratio": (
                                missing_ratio
                            ),
                        },
                    )
                )

            uniqueness_ratio = (
                unique_count / len(nonmissing)
                if len(nonmissing) > 0
                else 0.0
            )

            if (
                uniqueness_ratio
                >= self.options
                .identifier_uniqueness_threshold
                and column
                not in columns.numeric
            ):
                identifier_risk_columns.append(
                    str(column)
                )

            if column in columns.numeric:
                numeric = pd.to_numeric(
                    nonmissing,
                    errors="coerce",
                )

                zero_ratio = float(
                    (numeric == 0).mean()
                )

                if (
                    zero_ratio
                    >= self.options
                    .numeric_zero_sparsity_threshold
                ):
                    sparse_columns.append(
                        str(column)
                    )

                    issues.append(
                        DataQualityIssue(
                            code="P3-QUALITY-005",
                            severity="info",
                            message=(
                                f"Numeric feature {column!r} is "
                                "highly zero-sparse."
                            ),
                            columns=[str(column)],
                            recommendation=(
                                "Consider sparse representations "
                                "or indicator features."
                            ),
                            details={
                                "zero_ratio": zero_ratio,
                            },
                        )
                    )

        return DataQualityDiagnosticResult(
            issues=issues,
            constant_columns=constant_columns,
            near_constant_columns=(
                near_constant_columns
            ),
            completely_missing_columns=(
                completely_missing_columns
            ),
            highly_missing_columns=(
                highly_missing_columns
            ),
            sparse_columns=sparse_columns,
            identifier_risk_columns=(
                identifier_risk_columns
            ),
            insufficient_information_columns=(
                insufficient_columns
            ),
        )
