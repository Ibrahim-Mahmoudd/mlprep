"""Base outlier-treatment strategies for Phase 5."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class TreatmentExecutionMode(StrEnum):
    """Supported treatment execution modes."""

    FIT = "fit"
    TRAINING_TRANSFORM = "training_transform"
    VALIDATION_TRANSFORM = "validation_transform"
    TEST_TRANSFORM = "test_transform"
    PRODUCTION_TRANSFORM = "production_transform"


class TreatmentState(BaseModel):
    """Serializable fitted state for one treatment strategy."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    column: str
    strategy: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    lower_bound: float | None = None
    upper_bound: float | None = None

    fallback_value: Any = None

    input_dtype: str | None = None
    output_dtype: str | None = None

    training_row_count: int = Field(
        default=0,
        ge=0,
    )

    training_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    transformed_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    remaining_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class TreatmentValidationOptions:
    """Common validation options for treatment strategies."""

    preserve_index: bool = True
    preserve_row_count: bool = True
    reject_missing_mask_values: bool = True


class BaseTreatmentStrategy(ABC):
    """Common interface for Phase 5 treatment strategies."""

    strategy_name = "base"

    def __init__(
        self,
        *,
        column: str,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        validation: TreatmentValidationOptions | None = None,
    ) -> None:
        """Initialize the treatment strategy."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise ValueError(
                "lower_bound cannot exceed upper_bound."
            )

        self.column = column
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.validation = (
            validation
            or TreatmentValidationOptions()
        )

        self.state = TreatmentState(
            column=column,
            strategy=self.strategy_name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the strategy is fitted."""

        return self.state.fitted

    def fit(
        self,
        *,
        series: pd.Series,
        outlier_mask: pd.Series,
        mode: TreatmentExecutionMode = (
            TreatmentExecutionMode.FIT
        ),
    ) -> BaseTreatmentStrategy:
        """Fit treatment state using training data only."""

        self._validate_inputs(
            series=series,
            outlier_mask=outlier_mask,
        )

        if mode != TreatmentExecutionMode.FIT:
            raise PermissionError(
                "Treatment strategies may only be fitted "
                "in FIT mode."
            )

        if self.is_fitted:
            raise RuntimeError(
                f"Treatment strategy for {self.column!r} "
                "is already fitted."
            )

        self.state.input_dtype = str(
            series.dtype
        )

        self.state.training_row_count = len(
            series
        )

        self.state.training_outlier_count = int(
            outlier_mask.sum()
        )

        fitted_values = self._fit(
            series=series.copy(deep=True),
            outlier_mask=outlier_mask.copy(
                deep=True
            ),
        )

        if fitted_values is not None:
            lower_bound, upper_bound = (
                fitted_values
            )

            if lower_bound is not None:
                self.lower_bound = float(
                    lower_bound
                )

            if upper_bound is not None:
                self.upper_bound = float(
                    upper_bound
                )

        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound
            > self.upper_bound
        ):
            raise ValueError(
                "Fitted lower bound exceeds the upper bound."
            )

        self.state.lower_bound = (
            self.lower_bound
        )

        self.state.upper_bound = (
            self.upper_bound
        )

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        return self

    def transform(
        self,
        *,
        series: pd.Series,
        outlier_mask: pd.Series,
        mode: TreatmentExecutionMode,
    ) -> pd.Series:
        """Apply fitted treatment to one feature."""

        self._validate_inputs(
            series=series,
            outlier_mask=outlier_mask,
        )

        if mode == TreatmentExecutionMode.FIT:
            raise ValueError(
                "Use fit() or fit_transform() in FIT mode."
            )

        if not self.is_fitted:
            raise RuntimeError(
                f"Treatment strategy for {self.column!r} "
                "is not fitted."
            )

        original_index = series.index.copy()
        original_name = series.name

        transformed = self._transform(
            series=series.copy(deep=True),
            outlier_mask=outlier_mask.copy(
                deep=True
            ),
        )

        if not isinstance(
            transformed,
            pd.Series,
        ):
            raise TypeError(
                "Treatment transformation must return "
                "a pandas Series."
            )

        if (
            self.validation.preserve_row_count
            and len(transformed) != len(series)
        ):
            raise ValueError(
                "Outlier treatment changed the row count."
            )

        if (
            self.validation.preserve_index
            and not transformed.index.equals(
                original_index
            )
        ):
            raise ValueError(
                "Outlier treatment changed the index."
            )

        transformed.name = original_name

        self.state.output_dtype = str(
            transformed.dtype
        )

        self.state.transformed_outlier_count = int(
            outlier_mask.sum()
        )

        return transformed

    def fit_transform(
        self,
        *,
        series: pd.Series,
        outlier_mask: pd.Series,
    ) -> pd.Series:
        """Fit and transform a training feature."""

        self.fit(
            series=series,
            outlier_mask=outlier_mask,
            mode=TreatmentExecutionMode.FIT,
        )

        return self.transform(
            series=series,
            outlier_mask=outlier_mask,
            mode=(
                TreatmentExecutionMode
                .TRAINING_TRANSFORM
            ),
        )

    def get_state(
        self,
    ) -> TreatmentState:
        """Return a deep copy of fitted state."""

        return self.state.model_copy(
            deep=True
        )

    @abstractmethod
    def _fit(
        self,
        *,
        series: pd.Series,
        outlier_mask: pd.Series,
    ) -> tuple[
        float | None,
        float | None,
    ] | None:
        """Learn strategy-specific state."""

    @abstractmethod
    def _transform(
        self,
        *,
        series: pd.Series,
        outlier_mask: pd.Series,
    ) -> pd.Series:
        """Apply strategy-specific treatment."""

    def _validate_inputs(
        self,
        *,
        series: pd.Series,
        outlier_mask: pd.Series,
    ) -> None:
        """Validate treatment inputs."""

        if not isinstance(
            series,
            pd.Series,
        ):
            raise TypeError(
                "series must be a pandas Series."
            )

        if not isinstance(
            outlier_mask,
            pd.Series,
        ):
            raise TypeError(
                "outlier_mask must be a pandas Series."
            )

        if (
            series.name is not None
            and str(series.name) != self.column
        ):
            raise ValueError(
                f"Expected feature {self.column!r}, "
                f"received {series.name!r}."
            )

        if len(series) != len(outlier_mask):
            raise ValueError(
                "The feature and outlier mask have "
                "different row counts."
            )

        if not series.index.equals(
            outlier_mask.index
        ):
            raise ValueError(
                "The feature and outlier mask have "
                "different indices."
            )

        if (
            self.validation
            .reject_missing_mask_values
            and outlier_mask.isna().any()
        ):
            raise ValueError(
                "The outlier mask contains missing values."
            )

        if not pd.api.types.is_bool_dtype(
            outlier_mask.dtype
        ):
            unique_values = set(
                outlier_mask.dropna().unique()
            )

            if not unique_values.issubset(
                {False, True}
            ):
                raise TypeError(
                    "The outlier mask must be Boolean-like."
                )
