"""Safe DataFrame memory optimization for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
)

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    NormalizationEvent,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    MemoryChangeState,
    Phase1State,
)
from mlprep.types import (
    FindingAction,
    FindingCategory,
    SemanticType,
    Severity,
)

MemoryOptimizationLevel = Literal[
    "disabled",
    "safe",
    "balanced",
    "aggressive",
]


@dataclass(slots=True)
class MemoryOptimizerOptions:
    """Options controlling physical memory optimization."""

    level: MemoryOptimizationLevel = "safe"

    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(
        default_factory=list
    )

    downcast_integers: bool = True
    downcast_floats: bool = False

    preserve_identifier_width: bool = True
    preserve_currency_precision: bool = True

    float_relative_tolerance: float = 1e-6
    float_absolute_tolerance: float = 1e-8

    minimum_memory_saving_ratio: float = 0.10

    def __post_init__(self) -> None:
        """Validate memory optimizer options."""

        if self.float_relative_tolerance < 0:
            raise ValueError(
                "float_relative_tolerance cannot be negative."
            )

        if self.float_absolute_tolerance < 0:
            raise ValueError(
                "float_absolute_tolerance cannot be negative."
            )

        if not 0.0 <= self.minimum_memory_saving_ratio <= 1.0:
            raise ValueError(
                "minimum_memory_saving_ratio must be between "
                "zero and one."
            )


class MemoryOptimizer:
    """Safely reduce DataFrame storage memory."""

    component_name = "memory"

    def __init__(
        self,
        options: MemoryOptimizerOptions | None = None,
    ) -> None:
        """Initialize the memory optimizer."""

        self.options = (
            options or MemoryOptimizerOptions()
        )

    def optimize(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Optimize selected physical dtypes safely."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Memory optimization requires a pandas DataFrame."
            )

        working_data = data.copy()

        memory_before = self._dataframe_memory(
            working_data
        )

        result.dataset_summary.memory_usage_before_bytes = (
            memory_before
        )

        if self.options.level == "disabled":
            result.dataset_summary.memory_usage_after_bytes = (
                memory_before
            )

            return working_data

        for column in self._select_columns(working_data):
            self._optimize_column(
                data=working_data,
                column=column,
                result=result,
                state=state,
            )

        memory_after = self._dataframe_memory(
            working_data
        )

        result.dataset_summary.memory_usage_after_bytes = (
            memory_after
        )

        saved_bytes = max(
            0,
            memory_before - memory_after,
        )

        if saved_bytes > 0:
            result.add_finding(
                Finding(
                    code="P1-MEMORY-001",
                    category=FindingCategory.MEMORY,
                    severity=Severity.INFO,
                    component=self.component_name,
                    message=(
                        "DataFrame memory usage was reduced safely."
                    ),
                    recommended_action=FindingAction.COERCE,
                    applied_action=FindingAction.COERCE,
                    blocks_readiness=False,
                    details={
                        "memory_before_bytes": (
                            memory_before
                        ),
                        "memory_after_bytes": (
                            memory_after
                        ),
                        "memory_saved_bytes": saved_bytes,
                        "memory_reduction_ratio": (
                            saved_bytes / memory_before
                            if memory_before > 0
                            else 0.0
                        ),
                    },
                )
            )

        return working_data

    def _optimize_column(
        self,
        *,
        data: pd.DataFrame,
        column: str,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Attempt safe optimization of one column."""

        series = data[column]

        column_state = state.column_states.get(column)

        if column_state is not None:
            if (
                self.options.preserve_identifier_width
                and column_state.semantic_type
                == SemanticType.IDENTIFIER
            ):
                return

            if (
                self.options.preserve_currency_precision
                and column_state.semantic_type
                == SemanticType.CURRENCY
            ):
                return

        optimized_series: pd.Series | None = None

        if (
            self.options.downcast_integers
            and is_integer_dtype(series.dtype)
        ):
            optimized_series = pd.to_numeric(
                series,
                downcast="integer",
            )

        elif (
            self.options.downcast_floats
            and is_float_dtype(series.dtype)
        ):
            candidate = pd.to_numeric(
                series,
                downcast="float",
            )

            if self._float_conversion_is_safe(
                original=series,
                candidate=candidate,
            ):
                optimized_series = candidate

        if optimized_series is None:
            return

        if optimized_series.dtype == series.dtype:
            return

        memory_before = int(
            series.memory_usage(
                index=False,
                deep=True,
            )
        )

        memory_after = int(
            optimized_series.memory_usage(
                index=False,
                deep=True,
            )
        )

        if memory_before == 0:
            return

        saving_ratio = (
            memory_before - memory_after
        ) / memory_before

        if (
            saving_ratio
            < self.options.minimum_memory_saving_ratio
        ):
            return

        dtype_before = str(series.dtype)
        dtype_after = str(optimized_series.dtype)

        data[column] = optimized_series

        memory_change = MemoryChangeState(
            column=column,
            dtype_before=dtype_before,
            dtype_after=dtype_after,
            memory_before_bytes=memory_before,
            memory_after_bytes=memory_after,
            semantic_type_preserved=True,
            precision_verified=True,
            reversible=True,
        )

        state.register_memory_change(memory_change)

        if column_state is not None:
            column_state.dtype_before = dtype_before
            column_state.dtype_after = dtype_after

            state.register_column_state(column_state)

        result.add_normalization_event(
            NormalizationEvent(
                component=self.component_name,
                operation="downcast_dtype",
                description=(
                    f"Changed column {column!r} from "
                    f"{dtype_before!r} to {dtype_after!r}."
                ),
                column=column,
                affected_row_count=len(data),
                before_summary={
                    "dtype": dtype_before,
                    "memory_bytes": memory_before,
                },
                after_summary={
                    "dtype": dtype_after,
                    "memory_bytes": memory_after,
                },
                reversible=True,
                metadata={
                    "memory_saved_bytes": (
                        memory_before - memory_after
                    ),
                    "memory_saving_ratio": saving_ratio,
                },
            )
        )

    def _float_conversion_is_safe(
        self,
        *,
        original: pd.Series,
        candidate: pd.Series,
    ) -> bool:
        """Verify that float downcasting preserves values."""

        original_values = original.to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        candidate_values = candidate.to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        return bool(
            np.allclose(
                original_values,
                candidate_values,
                rtol=(
                    self.options.float_relative_tolerance
                ),
                atol=(
                    self.options.float_absolute_tolerance
                ),
                equal_nan=True,
            )
        )

    def _select_columns(
        self,
        data: pd.DataFrame,
    ) -> list[object]:
        """Select columns eligible for optimization."""

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
    def _dataframe_memory(
        data: pd.DataFrame,
    ) -> int:
        """Calculate deep DataFrame memory usage."""

        return int(
            data.memory_usage(
                index=True,
                deep=True,
            ).sum()
        )
