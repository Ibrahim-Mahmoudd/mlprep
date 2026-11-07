"""Time-aware imputation for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from mlprep.phases.phase4.strategies.base import (
    BaseImputer,
    ImputerExecutionMode,
)

TemporalImputationMethod = Literal[
    "forward_fill",
    "forward_fill_then_constant",
]


@dataclass(slots=True)
class TemporalImputerOptions:
    """Options controlling temporal imputation."""

    method: TemporalImputationMethod = (
        "forward_fill_then_constant"
    )

    fallback_value: object | None = None

    require_monotonic_time: bool = True

    allow_equal_timestamps: bool = True
    allow_backward_fill: bool = False

    def __post_init__(self) -> None:
        """Validate temporal-imputer options."""

        if self.allow_backward_fill:
            raise ValueError(
                "Backward filling is disabled because it may "
                "introduce future information."
            )


class TemporalImputer(BaseImputer):
    """Forward-fill values without using future observations."""

    strategy_name = "time_aware"

    def __init__(
        self,
        *,
        column: str,
        timestamp_column: str,
        options: TemporalImputerOptions | None = None,
    ) -> None:
        """Initialize the temporal imputer."""

        super().__init__(
            column=column
        )

        if not timestamp_column.strip():
            raise ValueError(
                "timestamp_column cannot be empty."
            )

        self.timestamp_column = (
            timestamp_column
        )

        self.options = (
            options or TemporalImputerOptions()
        )

        self._fallback_value: object = None

        self.state.metadata.update(
            {
                "method": self.options.method,
                "timestamp_column": (
                    timestamp_column
                ),
                "future_values_allowed": False,
            }
        )

    def fit_frame(
        self,
        frame: pd.DataFrame,
    ) -> TemporalImputer:
        """Fit temporal fallback state from training data."""

        self._validate_frame(
            frame
        )

        if self.is_fitted:
            raise RuntimeError(
                "Temporal imputer is already fitted."
            )

        values = frame[self.column]

        self.state.input_dtype = str(
            values.dtype
        )

        self.state.training_row_count = len(
            values
        )

        self.state.training_missing_count = int(
            values.isna().sum()
        )

        if self.options.fallback_value is not None:
            self._fallback_value = (
                self.options.fallback_value
            )

        else:
            valid = values.dropna()

            if valid.empty:
                raise ValueError(
                    f"Temporal feature {self.column!r} "
                    "contains no valid training values."
                )

            counts = valid.value_counts()

            maximum_count = int(
                counts.max()
            )

            candidates = counts[
                counts == maximum_count
            ].index.tolist()

            candidates.sort(
                key=lambda value: str(value)
            )

            self._fallback_value = (
                candidates[0]
            )

        self.state.learned_value = (
            self._fallback_value
        )

        self.state.fitted = True

        return self

    def transform_frame(
        self,
        frame: pd.DataFrame,
        *,
        mode: ImputerExecutionMode,
    ) -> pd.Series:
        """Forward-fill one partition in timestamp order."""

        if mode == ImputerExecutionMode.FIT:
            raise ValueError(
                "Use fit_frame() for fitting."
            )

        if not self.is_fitted:
            raise RuntimeError(
                "Temporal imputer is not fitted."
            )

        self._validate_frame(
            frame
        )

        timestamps = pd.to_datetime(
            frame[self.timestamp_column],
            errors="raise",
        )

        order_frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "original_order": range(
                    len(frame)
                ),
            },
            index=frame.index,
        ).sort_values(
            by=[
                "timestamp",
                "original_order",
            ],
            kind="mergesort",
        )

        ordered_index = (
            order_frame.index
        )

        ordered_values = frame.loc[
            ordered_index,
            self.column,
        ]

        transformed = (
            ordered_values.ffill()
        )

        if (
            self.options.method
            == "forward_fill_then_constant"
        ):
            transformed = transformed.fillna(
                self._fallback_value
            )

        transformed = transformed.reindex(
            frame.index
        )

        if not transformed.index.equals(
            frame.index
        ):
            raise ValueError(
                "Temporal imputation changed row order."
            )

        self.state.transformed_missing_count = int(
            transformed.isna().sum()
        )

        return transformed

    def _fit(
        self,
        series: pd.Series,
    ) -> object:
        """Disable one-Series fitting."""

        raise RuntimeError(
            "TemporalImputer requires fit_frame()."
        )

    def _transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Disable one-Series transformation."""

        raise RuntimeError(
            "TemporalImputer requires transform_frame()."
        )

    def _validate_frame(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """Validate temporal imputer input."""

        required_columns = {
            self.column,
            self.timestamp_column,
        }

        missing_columns = required_columns.difference(
            frame.columns
        )

        if missing_columns:
            raise ValueError(
                "Temporal-imputer frame is missing columns: "
                f"{sorted(missing_columns)}"
            )

        timestamps = pd.to_datetime(
            frame[self.timestamp_column],
            errors="coerce",
        )

        if timestamps.isna().any():
            raise ValueError(
                "Temporal imputation requires valid timestamps."
            )

        if self.options.require_monotonic_time:
            ordered = timestamps.sort_values(
                kind="mergesort"
            )

            if not ordered.is_monotonic_increasing:
                raise ValueError(
                    "Timestamps cannot be ordered safely."
                )

            if (
                not self.options
                .allow_equal_timestamps
                and ordered.duplicated().any()
            ):
                raise ValueError(
                    "Equal timestamps are not allowed."
                )