"""Base numeric transformers for Phase 6."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class NumericExecutionMode(StrEnum):
    """Supported numeric transformation modes."""

    FIT = "fit"
    TRAINING_TRANSFORM = "training_transform"
    VALIDATION_TRANSFORM = "validation_transform"
    TEST_TRANSFORM = "test_transform"
    PRODUCTION_TRANSFORM = "production_transform"


class NumericTransformerState(BaseModel):
    """Serializable state for one numeric transformer."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    column: str
    transformer: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    input_dtype: str | None = None
    output_dtype: str | None = None

    training_row_count: int = Field(
        default=0,
        ge=0,
    )

    training_valid_count: int = Field(
        default=0,
        ge=0,
    )

    training_missing_count: int = Field(
        default=0,
        ge=0,
    )

    training_nonfinite_count: int = Field(
        default=0,
        ge=0,
    )

    transformed_missing_count: int = Field(
        default=0,
        ge=0,
    )

    transformed_nonfinite_count: int = Field(
        default=0,
        ge=0,
    )

    constant_feature: bool = False

    learned_parameters: dict[str, Any] = Field(
        default_factory=dict
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class BaseNumericTransformer(ABC):
    """Common interface for fitted numeric transformers."""

    transformer_name = "base_numeric"

    def __init__(
        self,
        *,
        column: str,
        require_finite_input: bool = True,
        allow_missing_input: bool = False,
        require_finite_output: bool = True,
        allow_missing_output: bool = False,
    ) -> None:
        """Initialize the numeric transformer."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        self.column = column

        self.require_finite_input = (
            require_finite_input
        )

        self.allow_missing_input = (
            allow_missing_input
        )

        self.require_finite_output = (
            require_finite_output
        )

        self.allow_missing_output = (
            allow_missing_output
        )

        self.state = NumericTransformerState(
            column=column,
            transformer=self.transformer_name,
        )

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the transformer is fitted."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
        *,
        mode: NumericExecutionMode = NumericExecutionMode.FIT,
    ) -> BaseNumericTransformer:
        """Fit the transformer on training data only."""

        self._validate_series(
            series
        )

        if mode != NumericExecutionMode.FIT:
            raise PermissionError(
                "Numeric transformers may only be fitted "
                "in FIT mode."
            )

        if self.is_fitted:
            raise RuntimeError(
                f"Transformer for {self.column!r} "
                "is already fitted."
            )

        numeric = self._numeric_series(
            series
        )

        self._validate_input_values(
            numeric
        )

        finite_mask = self._finite_mask(
            numeric
        )

        self.state.input_dtype = str(
            series.dtype
        )

        self.state.training_row_count = len(
            series
        )

        self.state.training_valid_count = int(
            finite_mask.sum()
        )

        self.state.training_missing_count = int(
            numeric.isna().sum()
        )

        self.state.training_nonfinite_count = (
            self._nonfinite_count(
                numeric
            )
        )

        valid = numeric.loc[
            finite_mask
        ]

        learned_parameters = self._fit(
            valid.copy(
                deep=True
            )
        )

        if not isinstance(
            learned_parameters,
            dict,
        ):
            raise TypeError(
                "Numeric transformer fitting must return "
                "a dictionary of learned parameters."
            )

        self.state.learned_parameters = (
            learned_parameters
        )

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        return self

    def transform(
        self,
        series: pd.Series,
        *,
        mode: NumericExecutionMode,
    ) -> pd.Series:
        """Transform one numeric Series."""

        self._validate_series(
            series
        )

        if mode == NumericExecutionMode.FIT:
            raise ValueError(
                "Use fit() or fit_transform() in FIT mode."
            )

        if not self.is_fitted:
            raise RuntimeError(
                f"Transformer for {self.column!r} "
                "is not fitted."
            )

        numeric = self._numeric_series(
            series
        )

        self._validate_input_values(
            numeric
        )

        original_index = numeric.index.copy()
        original_name = numeric.name

        transformed = self._transform(
            numeric.copy(
                deep=True
            )
        )

        if not isinstance(
            transformed,
            pd.Series,
        ):
            raise TypeError(
                "Numeric transformation must return "
                "a pandas Series."
            )

        if len(transformed) != len(numeric):
            raise ValueError(
                "Numeric transformation changed row count."
            )

        if not transformed.index.equals(
            original_index
        ):
            raise ValueError(
                "Numeric transformation changed the index."
            )

        transformed.name = original_name

        self._validate_output_values(
            transformed
        )

        self.state.output_dtype = str(
            transformed.dtype
        )

        self.state.transformed_missing_count = int(
            transformed.isna().sum()
        )

        self.state.transformed_nonfinite_count = (
            self._nonfinite_count(
                transformed
            )
        )

        return transformed

    def fit_transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Fit and transform the training Series."""

        self.fit(
            series,
            mode=NumericExecutionMode.FIT,
        )

        return self.transform(
            series,
            mode=(
                NumericExecutionMode
                .TRAINING_TRANSFORM
            ),
        )

    def get_state(
        self,
    ) -> NumericTransformerState:
        """Return a deep copy of fitted state."""

        return self.state.model_copy(
            deep=True
        )

    @abstractmethod
    def _fit(
        self,
        valid: pd.Series,
    ) -> dict[str, Any]:
        """Learn numeric transformation state."""

    @abstractmethod
    def _transform(
        self,
        numeric: pd.Series,
    ) -> pd.Series:
        """Apply numeric transformation state."""

    def _validate_series(
        self,
        series: pd.Series,
    ) -> None:
        """Validate the input Series."""

        if not isinstance(
            series,
            pd.Series,
        ):
            raise TypeError(
                "Numeric transformer input must be "
                "a pandas Series."
            )

        if (
            series.name is not None
            and str(series.name) != self.column
        ):
            raise ValueError(
                f"Expected feature {self.column!r}, "
                f"received {series.name!r}."
            )

    def _validate_input_values(
        self,
        numeric: pd.Series,
    ) -> None:
        """Validate numeric input values."""

        missing_count = int(
            numeric.isna().sum()
        )

        if (
            not self.allow_missing_input
            and missing_count > 0
        ):
            raise ValueError(
                f"Feature {self.column!r} contains "
                f"{missing_count} missing numeric values."
            )

        nonfinite_count = self._nonfinite_count(
            numeric
        )

        if (
            self.require_finite_input
            and nonfinite_count > 0
        ):
            raise ValueError(
                f"Feature {self.column!r} contains "
                f"{nonfinite_count} nonfinite values."
            )

    def _validate_output_values(
        self,
        transformed: pd.Series,
    ) -> None:
        """Validate numeric output values."""

        missing_count = int(
            transformed.isna().sum()
        )

        if (
            not self.allow_missing_output
            and missing_count > 0
        ):
            raise ValueError(
                f"Transformer {self.transformer_name!r} "
                f"produced {missing_count} missing values "
                f"for feature {self.column!r}."
            )

        nonfinite_count = self._nonfinite_count(
            transformed
        )

        if (
            self.require_finite_output
            and nonfinite_count > 0
        ):
            raise ValueError(
                f"Transformer {self.transformer_name!r} "
                f"produced {nonfinite_count} nonfinite values "
                f"for feature {self.column!r}."
            )

    @staticmethod
    def _numeric_series(
        series: pd.Series,
    ) -> pd.Series:
        """Convert a Series to numeric values."""

        return pd.to_numeric(
            series,
            errors="coerce",
        )

    @staticmethod
    def _finite_mask(
        series: pd.Series,
    ) -> pd.Series:
        """Return a finite-value mask."""

        return (
            series.notna()
            & np.isfinite(
                series.astype(float)
            )
        )

    @classmethod
    def _nonfinite_count(
        cls,
        series: pd.Series,
    ) -> int:
        """Count positive and negative infinite values."""

        mask = (
            series.notna()
            & ~np.isfinite(
                series.astype(float)
            )
        )

        return int(
            mask.sum()
        )
