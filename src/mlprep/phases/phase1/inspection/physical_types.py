"""Physical dtype inspection for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
    TypeResolutionState,
)
from mlprep.types import (
    FindingAction,
    FindingCategory,
    PhysicalType,
    SemanticType,
    Severity,
)


@dataclass(slots=True)
class PhysicalTypeInspectionOptions:
    """Options controlling physical dtype inspection."""

    enabled: bool = True

    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(
        default_factory=list
    )

    report_object_columns: bool = True
    report_unknown_types: bool = True

    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate physical-type inspection options."""

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )

        if self.include_columns is None:
            return

        overlap = set(
            self.include_columns
        ).intersection(self.exclude_columns)

        if overlap:
            columns = ", ".join(sorted(overlap))

            raise ValueError(
                "Columns cannot be both included and "
                f"excluded: {columns}"
            )


class PhysicalTypeInspector:
    """Inspect the physical storage type of each column."""

    component_name = "physical_types"

    def __init__(
        self,
        options: PhysicalTypeInspectionOptions | None = None,
    ) -> None:
        """Initialize the physical-type inspector."""

        self.options = (
            options or PhysicalTypeInspectionOptions()
        )

    def inspect(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> dict[str, PhysicalType]:
        """Inspect selected columns and register their physical types."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Physical-type inspection requires a "
                "pandas DataFrame."
            )

        if not self.options.enabled:
            return {}

        columns = self._select_columns(data)

        detected_types: dict[str, PhysicalType] = {}

        self._report_missing_selected_columns(
            data=data,
            result=result,
        )

        for column in columns:
            series = data[column]

            physical_type = self.detect_type(series)

            detected_types[column] = physical_type

            existing_resolution = (
                state.type_resolution_states.get(column)
            )

            semantic_type = SemanticType.UNKNOWN
            confidence = 0.0

            if existing_resolution is not None:
                semantic_type = (
                    existing_resolution.semantic_type
                )

                confidence = (
                    existing_resolution
                    .semantic_type_confidence
                )

            state.register_type_resolution(
                TypeResolutionState(
                    column=column,
                    original_physical_type=physical_type,
                    resolved_physical_type=physical_type,
                    semantic_type=semantic_type,
                    semantic_type_confidence=confidence,
                    type_was_declared=False,
                    type_was_inferred=False,
                    coercion_was_applied=False,
                )
            )

            column_state = state.column_states.get(column)

            if column_state is not None:
                column_state.physical_type = physical_type

                if column_state.dtype_before is None:
                    column_state.dtype_before = str(
                        series.dtype
                    )

                column_state.dtype_after = str(
                    series.dtype
                )

                state.register_column_state(
                    column_state
                )

            self._add_type_finding(
                column=column,
                series=series,
                physical_type=physical_type,
                result=result,
            )

        return detected_types

    @staticmethod
    def detect_type(
        series: pd.Series,
    ) -> PhysicalType:
        """Resolve a pandas Series into an MLPrep physical type."""

        dtype = series.dtype

        if is_bool_dtype(dtype):
            return PhysicalType.BOOLEAN

        if is_integer_dtype(dtype):
            return PhysicalType.INTEGER

        if is_float_dtype(dtype):
            return PhysicalType.FLOAT

        if is_complex_dtype(dtype):
            return PhysicalType.COMPLEX

        if is_datetime64_any_dtype(dtype):
            return PhysicalType.DATETIME

        if is_timedelta64_dtype(dtype):
            return PhysicalType.TIMEDELTA

        if isinstance(dtype, pd.CategoricalDtype):
            return PhysicalType.CATEGORY

        if is_string_dtype(dtype):
            return PhysicalType.STRING

        if is_object_dtype(dtype):
            return PhysicalType.OBJECT

        return PhysicalType.UNKNOWN

    def _select_columns(
        self,
        data: pd.DataFrame,
    ) -> list[str]:
        """Select columns eligible for physical-type inspection."""

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

    def _add_type_finding(
        self,
        *,
        column: str,
        series: pd.Series,
        physical_type: PhysicalType,
        result: Phase1Result,
    ) -> None:
        """Add findings for ambiguous physical storage types."""

        if (
            physical_type == PhysicalType.OBJECT
            and self.options.report_object_columns
        ):
            result.add_finding(
                Finding(
                    code="P1-PHYSICAL-001",
                    category=FindingCategory.PHYSICAL_TYPE,
                    severity=Severity.INFO,
                    component=self.component_name,
                    message=(
                        f"Column {column!r} uses the ambiguous "
                        "object dtype."
                    ),
                    columns=[column],
                    recommended_action=FindingAction.REPORT,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=False,
                    details={
                        "pandas_dtype": str(series.dtype),
                    },
                )
            )

        if (
            physical_type == PhysicalType.UNKNOWN
            and self.options.report_unknown_types
        ):
            result.add_finding(
                Finding(
                    code="P1-PHYSICAL-002",
                    category=FindingCategory.PHYSICAL_TYPE,
                    severity=Severity.WARNING,
                    component=self.component_name,
                    message=(
                        f"Column {column!r} has an unsupported "
                        "or unknown physical dtype."
                    ),
                    columns=[column],
                    recommended_action=FindingAction.WARN,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=False,
                    details={
                        "pandas_dtype": str(series.dtype),
                    },
                )
            )

    def _report_missing_selected_columns(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> None:
        """Report requested columns absent from the dataset."""

        if self.options.include_columns is None:
            return

        available_columns = {
            str(column)
            for column in data.columns
        }

        missing_columns = [
            column
            for column in self.options.include_columns
            if column not in available_columns
        ]

        if not missing_columns:
            return

        result.add_finding(
            Finding(
                code="P1-PHYSICAL-003",
                category=FindingCategory.PHYSICAL_TYPE,
                severity=Severity.WARNING,
                component=self.component_name,
                message=(
                    "Physical-type inspection references columns "
                    "that do not exist."
                ),
                columns=missing_columns,
                recommended_action=FindingAction.WARN,
                applied_action=FindingAction.REPORT,
                blocks_readiness=False,
            )
        )
