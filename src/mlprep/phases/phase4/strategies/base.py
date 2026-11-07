"""Base interfaces for fitted Phase 4 imputers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class ImputerExecutionMode(StrEnum):
    """Execution modes supported by fitted imputers."""

    FIT = "fit"
    TRAINING_TRANSFORM = "training_transform"
    VALIDATION_TRANSFORM = "validation_transform"
    TEST_TRANSFORM = "test_transform"
    PRODUCTION_TRANSFORM = "production_transform"


class ImputerState(BaseModel):
    """Serializable fitted state of one imputer."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    column: str
    strategy: str

    fitted: bool = False

    learned_value: Any = None

    input_dtype: str | None = None
    output_dtype: str | None = None

    training_row_count: int = Field(
        default=0,
        ge=0,
    )

    training_missing_count: int = Field(
        default=0,
        ge=0,
    )

    transformed_missing_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class BaseImputer(ABC):
    """Common interface for fitted Phase 4 imputers."""

    strategy_name = "base"

    def __init__(
        self,
        *,
        column: str,
    ) -> None:
        """Initialize the imputer."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        self.column = column

        self.state = ImputerState(
            column=column,
            strategy=self.strategy_name,
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether the imputer is fitted."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
        *,
        mode: ImputerExecutionMode = (
            ImputerExecutionMode.FIT
        ),
    ) -> BaseImputer:
        """Fit the imputer using training data only."""

        self._validate_series(series)

        if mode != ImputerExecutionMode.FIT:
            raise PermissionError(
                "Imputers may only be fitted in FIT mode."
            )

        if self.is_fitted:
            raise RuntimeError(
                f"Imputer for {self.column!r} is already fitted."
            )

        self.state.input_dtype = str(
            series.dtype
        )

        self.state.training_row_count = len(
            series
        )

        self.state.training_missing_count = int(
            series.isna().sum()
        )

        learned_value = self._fit(
            series.copy(deep=True)
        )

        self.state.learned_value = (
            learned_value
        )

        self.state.fitted = True

        return self

    def transform(
        self,
        series: pd.Series,
        *,
        mode: ImputerExecutionMode,
    ) -> pd.Series:
        """Transform one Series using fitted state."""

        self._validate_series(series)

        if mode == ImputerExecutionMode.FIT:
            raise ValueError(
                "Use fit() or fit_transform() in FIT mode."
            )

        if not self.is_fitted:
            raise RuntimeError(
                f"Imputer for {self.column!r} is not fitted."
            )

        original_index = series.index.copy()
        original_name = series.name

        transformed = self._transform(
            series.copy(deep=True)
        )

        if not isinstance(
            transformed,
            pd.Series,
        ):
            raise TypeError(
                "Imputer transform must return a pandas Series."
            )

        if len(transformed) != len(series):
            raise ValueError(
                "Imputation changed the number of rows."
            )

        if not transformed.index.equals(
            original_index
        ):
            raise ValueError(
                "Imputation changed the Series index."
            )

        transformed.name = original_name

        self.state.output_dtype = str(
            transformed.dtype
        )

        self.state.transformed_missing_count = int(
            transformed.isna().sum()
        )

        return transformed

    def fit_transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Fit and transform training data."""

        self.fit(
            series,
            mode=ImputerExecutionMode.FIT,
        )

        return self.transform(
            series,
            mode=(
                ImputerExecutionMode
                .TRAINING_TRANSFORM
            ),
        )

    def get_state(
        self,
    ) -> ImputerState:
        """Return a deep copy of fitted state."""

        return self.state.model_copy(
            deep=True
        )

    @abstractmethod
    def _fit(
        self,
        series: pd.Series,
    ) -> Any:
        """Learn fitted state from training data."""

    @abstractmethod
    def _transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Transform one Series."""

    def _validate_series(
        self,
        series: pd.Series,
    ) -> None:
        """Validate one input Series."""

        if not isinstance(
            series,
            pd.Series,
        ):
            raise TypeError(
                "Imputer input must be a pandas Series."
            )

        if (
            series.name is not None
            and str(series.name) != self.column
        ):
            raise ValueError(
                f"Expected column {self.column!r}, "
                f"received {series.name!r}."
            )
