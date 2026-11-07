"""Group-based imputation for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from mlprep.phases.phase4.strategies.base import (
    BaseImputer,
    ImputerExecutionMode,
)

GroupAggregationMethod = Literal[
    "mean",
    "median",
    "most_frequent",
]


@dataclass(slots=True)
class GroupImputerOptions:
    """Options controlling group-based imputation."""

    method: GroupAggregationMethod = "median"

    minimum_group_nonmissing_count: int = 1

    fallback_method: GroupAggregationMethod = "median"

    def __post_init__(self) -> None:
        """Validate group-imputer options."""

        if self.minimum_group_nonmissing_count < 1:
            raise ValueError(
                "minimum_group_nonmissing_count must be positive."
            )


class GroupImputer(BaseImputer):
    """Impute values using training-group statistics."""

    strategy_name = "group_based"

    def __init__(
        self,
        *,
        column: str,
        group_columns: list[str],
        options: GroupImputerOptions | None = None,
    ) -> None:
        """Initialize the group imputer."""

        super().__init__(
            column=column
        )

        if not group_columns:
            raise ValueError(
                "GroupImputer requires group columns."
            )

        self.group_columns = list(
            group_columns
        )

        self.options = (
            options or GroupImputerOptions()
        )

        self._group_mapping: dict[
            tuple[object, ...],
            object,
        ] = {}

        self._global_fallback: object = None

        self.state.metadata.update(
            {
                "method": self.options.method,
                "group_columns": (
                    self.group_columns
                ),
            }
        )

    def fit_frame(
        self,
        frame: pd.DataFrame,
    ) -> GroupImputer:
        """Fit using target and group columns."""

        required_columns = {
            self.column,
            *self.group_columns,
        }

        missing_columns = required_columns.difference(
            frame.columns
        )

        if missing_columns:
            raise ValueError(
                "Group-imputer fit frame is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if self.is_fitted:
            raise RuntimeError(
                "Group imputer is already fitted."
            )

        target = frame[self.column]

        self.state.input_dtype = str(
            target.dtype
        )

        self.state.training_row_count = len(
            target
        )

        self.state.training_missing_count = int(
            target.isna().sum()
        )

        self._global_fallback = self._aggregate(
            target.dropna(),
            self.options.fallback_method,
        )

        grouped = frame.groupby(
            self.group_columns,
            dropna=False,
            sort=False,
        )

        for key, group in grouped:
            values = group[
                self.column
            ].dropna()

            if (
                len(values)
                < self.options
                .minimum_group_nonmissing_count
            ):
                continue

            normalized_key = self._normalize_key(
                key
            )

            self._group_mapping[
                normalized_key
            ] = self._aggregate(
                values,
                self.options.method,
            )

        self.state.learned_value = (
            self._global_fallback
        )

        self.state.metadata[
            "group_mapping_count"
        ] = len(
            self._group_mapping
        )

        self.state.fitted = True

        return self

    def transform_frame(
        self,
        frame: pd.DataFrame,
        *,
        mode: ImputerExecutionMode,
    ) -> pd.Series:
        """Transform one frame using fitted group state."""

        if mode == ImputerExecutionMode.FIT:
            raise ValueError(
                "Use fit_frame() for fitting."
            )

        if not self.is_fitted:
            raise RuntimeError(
                "Group imputer is not fitted."
            )

        required_columns = {
            self.column,
            *self.group_columns,
        }

        missing_columns = required_columns.difference(
            frame.columns
        )

        if missing_columns:
            raise ValueError(
                "Group-imputer transform frame is missing "
                f"columns: {sorted(missing_columns)}"
            )

        result = frame[self.column].copy()

        missing_mask = result.isna()

        for index in result.index[
            missing_mask
        ]:
            key = tuple(
                frame.at[
                    index,
                    group_column,
                ]
                for group_column
                in self.group_columns
            )

            value = self._group_mapping.get(
                key,
                self._global_fallback,
            )

            result.at[index] = value

        if not result.index.equals(
            frame.index
        ):
            raise ValueError(
                "Group imputation changed the index."
            )

        self.state.transformed_missing_count = int(
            result.isna().sum()
        )

        return result

    def _fit(
        self,
        series: pd.Series,
    ) -> object:
        """Disable one-Series fitting."""

        raise RuntimeError(
            "GroupImputer requires fit_frame()."
        )

    def _transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Disable one-Series transformation."""

        raise RuntimeError(
            "GroupImputer requires transform_frame()."
        )

    @staticmethod
    def _normalize_key(
        key: object,
    ) -> tuple[object, ...]:
        """Normalize a pandas group key."""

        if isinstance(key, tuple):
            return key

        return (key,)

    @staticmethod
    def _aggregate(
        values: pd.Series,
        method: GroupAggregationMethod,
    ) -> object:
        """Calculate one group statistic."""

        if values.empty:
            raise ValueError(
                "Cannot aggregate an empty Series."
            )

        if method == "mean":
            numeric = pd.to_numeric(
                values,
                errors="coerce",
            ).dropna()

            if numeric.empty:
                raise ValueError(
                    "Mean aggregation requires numeric values."
                )

            value = float(
                numeric.mean()
            )

        elif method == "median":
            numeric = pd.to_numeric(
                values,
                errors="coerce",
            ).dropna()

            if numeric.empty:
                raise ValueError(
                    "Median aggregation requires numeric values."
                )

            value = float(
                numeric.median()
            )

        elif method == "most_frequent":
            counts = values.value_counts(
                dropna=False
            )

            maximum_count = int(
                counts.max()
            )

            candidates = counts[
                counts == maximum_count
            ].index.tolist()

            candidates.sort(
                key=lambda item: str(item)
            )

            value = candidates[0]

        else:
            raise ValueError(
                f"Unsupported group method: {method!r}"
            )

        if isinstance(
            value,
            float,
        ) and not np.isfinite(value):
            raise ValueError(
                "Group aggregation produced a nonfinite value."
            )

        return value