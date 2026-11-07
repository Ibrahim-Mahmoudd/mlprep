"""Semantic type inference for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.types import (
    ColumnRole,
    FindingAction,
    FindingCategory,
    SemanticType,
    Severity,
)


@dataclass(slots=True)
class SemanticTypeOverride:
    """Explicit semantic type for one column."""

    semantic_type: SemanticType
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate the semantic-type override."""

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between zero and one."
            )


@dataclass(slots=True)
class SemanticTypeInferenceOptions:
    """Options controlling semantic type inference."""

    enabled: bool = True

    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(
        default_factory=list
    )

    overrides: dict[
        str,
        SemanticTypeOverride,
    ] = field(default_factory=dict)

    binary_unique_value_limit: int = 2
    nominal_unique_value_limit: int = 100

    identifier_uniqueness_threshold: float = 0.95

    free_text_average_length_threshold: float = 40.0
    free_text_unique_ratio_threshold: float = 0.50

    percentage_name_terms: tuple[str, ...] = (
        "percentage",
        "percent",
        "pct",
        "ratio",
        "rate",
    )

    currency_name_terms: tuple[str, ...] = (
        "amount",
        "price",
        "cost",
        "salary",
        "income",
        "revenue",
        "balance",
        "currency",
    )

    identifier_name_terms: tuple[str, ...] = (
        "id",
        "identifier",
        "uuid",
        "guid",
        "key",
        "code",
    )

    confidence_threshold: float = 0.70

    def __post_init__(self) -> None:
        """Validate semantic inference options."""

        if self.binary_unique_value_limit < 2:
            raise ValueError(
                "binary_unique_value_limit must be at least two."
            )

        if self.nominal_unique_value_limit < 2:
            raise ValueError(
                "nominal_unique_value_limit must be at least two."
            )

        if (
            self.nominal_unique_value_limit
            < self.binary_unique_value_limit
        ):
            raise ValueError(
                "nominal_unique_value_limit cannot be smaller "
                "than binary_unique_value_limit."
            )

        bounded_values = {
            "identifier_uniqueness_threshold": (
                self.identifier_uniqueness_threshold
            ),
            "free_text_unique_ratio_threshold": (
                self.free_text_unique_ratio_threshold
            ),
            "confidence_threshold": (
                self.confidence_threshold
            ),
        }

        for name, value in bounded_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between zero and one."
                )

        if self.free_text_average_length_threshold < 0:
            raise ValueError(
                "free_text_average_length_threshold cannot "
                "be negative."
            )


class SemanticTypeInferer:
    """Infer the real-world semantic type of each column."""

    component_name = "semantic_types"

    def __init__(
        self,
        options: SemanticTypeInferenceOptions | None = None,
    ) -> None:
        """Initialize the semantic-type inferer."""

        self.options = (
            options or SemanticTypeInferenceOptions()
        )

    def inspect(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> dict[str, SemanticType]:
        """Infer semantic types for selected columns."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Semantic-type inference requires a "
                "pandas DataFrame."
            )

        if not self.options.enabled:
            return {}

        resolved_types: dict[str, SemanticType] = {}

        for column in self._select_columns(data):
            semantic_type, confidence, evidence = (
                self._resolve_semantic_type(
                    column=column,
                    series=data[column],
                    state=state,
                )
            )

            resolved_types[column] = semantic_type

            resolution = (
                state.type_resolution_states.get(column)
            )

            if resolution is not None:
                resolution.semantic_type = semantic_type
                resolution.semantic_type_confidence = (
                    confidence
                )

                resolution.type_was_declared = (
                    column in self.options.overrides
                )

                resolution.type_was_inferred = (
                    column not in self.options.overrides
                )

                resolution.inference_evidence.update(
                    evidence
                )

                state.register_type_resolution(
                    resolution
                )

            column_state = state.column_states.get(column)

            if column_state is not None:
                column_state.semantic_type = semantic_type
                column_state.semantic_type_confidence = (
                    confidence
                )

                if (
                    semantic_type
                    == SemanticType.IDENTIFIER
                ):
                    column_state.is_identifier_like = True
                    column_state.use_as_model_feature = False

                state.register_column_state(
                    column_state
                )

            if confidence < self.options.confidence_threshold:
                result.add_finding(
                    Finding(
                        code="P1-SEMANTIC-001",
                        category=(
                            FindingCategory.SEMANTIC_TYPE
                        ),
                        severity=Severity.WARNING,
                        component=self.component_name,
                        message=(
                            f"Semantic type inference for "
                            f"column {column!r} has low confidence."
                        ),
                        columns=[column],
                        recommended_action=FindingAction.WARN,
                        applied_action=FindingAction.REPORT,
                        blocks_readiness=False,
                        details={
                            "semantic_type": (
                                semantic_type.value
                            ),
                            "confidence": confidence,
                            "evidence": evidence,
                        },
                    )
                )

        return resolved_types

    def _resolve_semantic_type(
        self,
        *,
        column: str,
        series: pd.Series,
        state: Phase1State,
    ) -> tuple[
        SemanticType,
        float,
        dict[str, object],
    ]:
        """Resolve the semantic type for one column."""

        override = self.options.overrides.get(column)

        if override is not None:
            return (
                override.semantic_type,
                override.confidence,
                {
                    "source": "explicit_override",
                },
            )

        column_state = state.column_states.get(column)

        if column_state is not None:
            if ColumnRole.TARGET in column_state.roles:
                return self._infer_target_type(series)

            if ColumnRole.IDENTIFIER in column_state.roles:
                return (
                    SemanticType.IDENTIFIER,
                    1.0,
                    {
                        "source": "declared_identifier_role",
                    },
                )

            if (
                ColumnRole.TIMESTAMP in column_state.roles
                or ColumnRole.PREDICTION_TIMESTAMP
                in column_state.roles
                or ColumnRole.OUTCOME_TIMESTAMP
                in column_state.roles
            ):
                return (
                    SemanticType.DATETIME,
                    1.0,
                    {
                        "source": "declared_timestamp_role",
                    },
                )

        resolution = (
            state.type_resolution_states.get(column)
        )

        if (
            resolution is not None
            and resolution.semantic_type
            != SemanticType.UNKNOWN
        ):
            return (
                resolution.semantic_type,
                resolution.semantic_type_confidence,
                {
                    "source": "previous_inspection",
                },
            )

        non_missing = series.dropna()

        if non_missing.empty:
            return (
                SemanticType.UNKNOWN,
                0.0,
                {
                    "source": "all_values_missing",
                },
            )

        unique_count = int(
            non_missing.nunique(dropna=True)
        )

        uniqueness_ratio = (
            unique_count / len(non_missing)
        )

        normalized_name = column.casefold()

        if is_bool_dtype(series.dtype):
            return (
                SemanticType.BINARY,
                1.0,
                {
                    "source": "boolean_dtype",
                    "unique_count": unique_count,
                },
            )

        if is_datetime64_any_dtype(series.dtype):
            return (
                SemanticType.DATETIME,
                1.0,
                {
                    "source": "datetime_dtype",
                },
            )

        if is_timedelta64_dtype(series.dtype):
            return (
                SemanticType.DURATION,
                1.0,
                {
                    "source": "timedelta_dtype",
                },
            )

        if self._name_contains_term(
            normalized_name,
            self.options.identifier_name_terms,
        ) and (
            uniqueness_ratio
            >= self.options.identifier_uniqueness_threshold
        ):
            return (
                SemanticType.IDENTIFIER,
                0.90,
                {
                    "source": "name_and_uniqueness",
                    "uniqueness_ratio": uniqueness_ratio,
                },
            )

        if is_numeric_dtype(series.dtype):
            if (
                unique_count
                <= self.options.binary_unique_value_limit
            ):
                return (
                    SemanticType.BINARY,
                    0.95,
                    {
                        "source": "numeric_binary_cardinality",
                        "unique_count": unique_count,
                    },
                )

            if self._name_contains_term(
                normalized_name,
                self.options.percentage_name_terms,
            ):
                return (
                    SemanticType.PERCENTAGE,
                    0.85,
                    {
                        "source": "numeric_name_signal",
                    },
                )

            if self._name_contains_term(
                normalized_name,
                self.options.currency_name_terms,
            ):
                return (
                    SemanticType.CURRENCY,
                    0.80,
                    {
                        "source": "numeric_name_signal",
                    },
                )

            if is_integer_dtype(series.dtype):
                return (
                    SemanticType.DISCRETE,
                    0.85,
                    {
                        "source": "integer_dtype",
                        "unique_count": unique_count,
                    },
                )

            if is_float_dtype(series.dtype):
                return (
                    SemanticType.CONTINUOUS,
                    0.85,
                    {
                        "source": "float_dtype",
                        "unique_count": unique_count,
                    },
                )

        if (
            is_string_dtype(series.dtype)
            or series.dtype == object
        ):
            string_values = non_missing.astype(
                "string"
            )

            average_length = float(
                string_values.str.len().mean()
            )

            if (
                uniqueness_ratio
                >= self.options.identifier_uniqueness_threshold
                and average_length <= 40.0
            ):
                return (
                    SemanticType.IDENTIFIER,
                    0.75,
                    {
                        "source": "string_uniqueness",
                        "uniqueness_ratio": uniqueness_ratio,
                        "average_length": average_length,
                    },
                )

            if (
                average_length
                >= self.options
                .free_text_average_length_threshold
                and uniqueness_ratio
                >= self.options.free_text_unique_ratio_threshold
            ):
                return (
                    SemanticType.FREE_TEXT,
                    0.85,
                    {
                        "source": "text_length_and_uniqueness",
                        "uniqueness_ratio": uniqueness_ratio,
                        "average_length": average_length,
                    },
                )

            if (
                unique_count
                <= self.options.binary_unique_value_limit
            ):
                return (
                    SemanticType.BINARY,
                    0.85,
                    {
                        "source": "text_binary_cardinality",
                        "unique_count": unique_count,
                    },
                )

            if (
                unique_count
                <= self.options.nominal_unique_value_limit
            ):
                return (
                    SemanticType.NOMINAL,
                    0.80,
                    {
                        "source": "categorical_cardinality",
                        "unique_count": unique_count,
                    },
                )

            return (
                SemanticType.SHORT_TEXT,
                0.70,
                {
                    "source": "string_fallback",
                    "unique_count": unique_count,
                    "average_length": average_length,
                },
            )

        return (
            SemanticType.UNKNOWN,
            0.25,
            {
                "source": "unsupported_pattern",
            },
        )

    @staticmethod
    def _infer_target_type(
        series: pd.Series,
    ) -> tuple[
        SemanticType,
        float,
        dict[str, object],
    ]:
        """Infer the semantic type of a target column."""

        unique_count = int(
            series.nunique(dropna=True)
        )

        if unique_count == 2:
            return (
                SemanticType.BINARY,
                1.0,
                {
                    "source": "target_binary_cardinality",
                    "unique_count": unique_count,
                },
            )

        if is_numeric_dtype(series.dtype):
            return (
                SemanticType.CONTINUOUS,
                0.80,
                {
                    "source": "numeric_target",
                    "unique_count": unique_count,
                },
            )

        return (
            SemanticType.NOMINAL,
            0.85,
            {
                "source": "categorical_target",
                "unique_count": unique_count,
            },
        )

    def _select_columns(
        self,
        data: pd.DataFrame,
    ) -> list[object]:
        """Select columns eligible for semantic inference."""

        excluded = set(
            self.options.exclude_columns
        )

        if self.options.include_columns is not None:
            return [
                column
                for column in self.options.include_columns
                if column in data.columns
                and column not in excluded
            ]

        return [
            str(column)
            for column in data.columns
            if str(column) not in excluded
        ]

    @staticmethod
    def _name_contains_term(
        normalized_name: str,
        terms: tuple[str, ...],
    ) -> bool:
        """Return whether a column name contains a semantic term."""

        name_parts = {
            part
            for part in normalized_name.replace(
                "-",
                "_",
            ).split("_")
            if part
        }

        return any(
            term in name_parts
            or normalized_name.endswith(
                f"_{term}"
            )
            for term in terms
        )
