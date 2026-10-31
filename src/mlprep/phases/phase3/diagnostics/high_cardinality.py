"""High-cardinality diagnostics for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import EDAContract


CardinalityLevel = Literal[
    "low",
    "medium",
    "high",
    "identifier_like",
]

EncodingSuggestion = Literal[
    "one_hot",
    "ordinal",
    "frequency",
    "hashing",
    "target_encoding",
    "text_vectorization",
    "exclude",
    "review",
]


class HighCardinalityDiagnostic(BaseModel):
    """Cardinality diagnostic for one categorical feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    row_count: int = Field(
        default=0,
        ge=0,
    )

    nonmissing_count: int = Field(
        default=0,
        ge=0,
    )

    unique_count: int = Field(
        default=0,
        ge=0,
    )

    uniqueness_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    level: CardinalityLevel

    identifier_like: bool = False

    suggested_encoding: EncodingSuggestion

    reason: str


class HighCardinalityDiagnosticResult(BaseModel):
    """Complete high-cardinality output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    diagnostics: dict[
        str,
        HighCardinalityDiagnostic,
    ] = Field(default_factory=dict)

    high_cardinality_columns: list[str] = Field(
        default_factory=list
    )

    identifier_like_columns: list[str] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class HighCardinalityDiagnosticOptions:
    """Options controlling cardinality diagnostics."""

    enabled: bool = True

    low_cardinality_limit: int = 15
    medium_cardinality_limit: int = 100

    identifier_uniqueness_threshold: float = 0.95

    target_encoding_minimum_cardinality: int = 20

    prefer_hashing_above: int = 1_000

    def __post_init__(self) -> None:
        """Validate cardinality options."""

        if self.low_cardinality_limit < 2:
            raise ValueError(
                "low_cardinality_limit must be at least two."
            )

        if (
            self.medium_cardinality_limit
            < self.low_cardinality_limit
        ):
            raise ValueError(
                "medium_cardinality_limit cannot be below "
                "low_cardinality_limit."
            )

        if not (
            0.0
            <= self.identifier_uniqueness_threshold
            <= 1.0
        ):
            raise ValueError(
                "identifier_uniqueness_threshold must be "
                "between zero and one."
            )


class HighCardinalityDiagnosticAnalyzer:
    """Diagnose categorical-cardinality risks."""

    component_name = "high_cardinality"

    def __init__(
        self,
        options: (
            HighCardinalityDiagnosticOptions | None
        ) = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options or HighCardinalityDiagnosticOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> HighCardinalityDiagnosticResult:
        """Analyze categorical and text cardinality."""

        if not self.options.enabled:
            return HighCardinalityDiagnosticResult()

        frame = contract.training_features()

        selected_columns = list(
            dict.fromkeys(
                [
                    *columns.categorical,
                    *columns.text,
                ]
            )
        )

        diagnostics: dict[
            str,
            HighCardinalityDiagnostic,
        ] = {}

        for column in selected_columns:
            if column not in frame.columns:
                continue

            series = frame[column]

            nonmissing = series.dropna()

            nonmissing_count = len(
                nonmissing
            )

            unique_count = int(
                nonmissing.nunique()
            )

            uniqueness_ratio = (
                unique_count / nonmissing_count
                if nonmissing_count > 0
                else 0.0
            )

            identifier_like = bool(
                uniqueness_ratio
                >= self.options
                .identifier_uniqueness_threshold
            )

            level = self._level(
                unique_count=unique_count,
                identifier_like=identifier_like,
            )

            encoding, reason = self._recommend(
                unique_count=unique_count,
                identifier_like=identifier_like,
                is_text=column in columns.text,
            )

            diagnostics[column] = (
                HighCardinalityDiagnostic(
                    column=column,
                    row_count=len(series),
                    nonmissing_count=(
                        nonmissing_count
                    ),
                    unique_count=unique_count,
                    uniqueness_ratio=(
                        uniqueness_ratio
                    ),
                    level=level,
                    identifier_like=(
                        identifier_like
                    ),
                    suggested_encoding=encoding,
                    reason=reason,
                )
            )

        return HighCardinalityDiagnosticResult(
            diagnostics=diagnostics,
            high_cardinality_columns=[
                column
                for column, diagnostic
                in diagnostics.items()
                if diagnostic.level
                in {
                    "high",
                    "identifier_like",
                }
            ],
            identifier_like_columns=[
                column
                for column, diagnostic
                in diagnostics.items()
                if diagnostic.identifier_like
            ],
        )

    def _level(
        self,
        *,
        unique_count: int,
        identifier_like: bool,
    ) -> CardinalityLevel:
        """Classify feature cardinality."""

        if identifier_like:
            return "identifier_like"

        if (
            unique_count
            <= self.options.low_cardinality_limit
        ):
            return "low"

        if (
            unique_count
            <= self.options.medium_cardinality_limit
        ):
            return "medium"

        return "high"

    def _recommend(
        self,
        *,
        unique_count: int,
        identifier_like: bool,
        is_text: bool,
    ) -> tuple[
        EncodingSuggestion,
        str,
    ]:
        """Recommend an encoding family."""

        if identifier_like:
            return (
                "exclude",
                "The feature behaves like an identifier and "
                "may encourage memorization.",
            )

        if is_text:
            return (
                "text_vectorization",
                "The feature is classified as text.",
            )

        if (
            unique_count
            <= self.options.low_cardinality_limit
        ):
            return (
                "one_hot",
                "The feature has low cardinality.",
            )

        if (
            unique_count
            >= self.options.prefer_hashing_above
        ):
            return (
                "hashing",
                "The feature has extremely high cardinality.",
            )

        if (
            unique_count
            >= self.options
            .target_encoding_minimum_cardinality
        ):
            return (
                "target_encoding",
                "The feature has high cardinality. Phase 7 "
                "must use cross-fitted target encoding.",
            )

        return (
            "frequency",
            "Frequency encoding may control dimensionality.",
        )