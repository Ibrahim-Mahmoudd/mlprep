"""Optional model-based imputation for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from mlprep.phases.phase4.strategies.base import (
    BaseImputer,
    ImputerExecutionMode,
)


ModelImputationTask = Literal[
    "regression",
    "classification",
]


@dataclass(slots=True)
class ModelImputerOptions:
    """Options controlling model-based imputation."""

    enabled: bool = False

    task: ModelImputationTask = "regression"

    predictor_columns: list[str] | None = None

    minimum_training_rows: int = 50

    ridge_alpha: float = 1.0

    categorical_fallback_value: object = (
        "__MISSING__"
    )

    def __post_init__(self) -> None:
        """Validate model-imputer options."""

        if self.minimum_training_rows < 2:
            raise ValueError(
                "minimum_training_rows must be at least two."
            )

        if self.ridge_alpha < 0.0:
            raise ValueError(
                "ridge_alpha cannot be negative."
            )


class ModelImputer(BaseImputer):
    """Optional lightweight model-based imputer.

    This implementation supports numeric predictors only.
    It remains disabled by default.
    """

    strategy_name = "model_based"

    def __init__(
        self,
        *,
        column: str,
        options: ModelImputerOptions | None = None,
    ) -> None:
        """Initialize the model imputer."""

        super().__init__(
            column=column
        )

        self.options = (
            options or ModelImputerOptions()
        )

        self._predictor_columns: list[str] = []
        self._feature_means: dict[
            str,
            float,
        ] = {}

        self._coefficients: np.ndarray | None = None
        self._classes: list[object] = []

        self.state.metadata.update(
            {
                "enabled": self.options.enabled,
                "task": self.options.task,
            }
        )

    def fit_frame(
        self,
        frame: pd.DataFrame,
    ) -> ModelImputer:
        """Fit a model using observed training target rows."""

        if not self.options.enabled:
            raise PermissionError(
                "Model-based imputation is disabled."
            )

        if self.column not in frame.columns:
            raise ValueError(
                f"Column {self.column!r} does not exist."
            )

        predictors = (
            list(
                self.options.predictor_columns
            )
            if self.options.predictor_columns
            is not None
            else [
                str(column)
                for column in frame.columns
                if column != self.column
                and pd.api.types
                .is_numeric_dtype(
                    frame[column].dtype
                )
            ]
        )

        if not predictors:
            raise ValueError(
                "Model imputation requires numeric predictors."
            )

        missing_predictors = set(
            predictors
        ).difference(
            frame.columns
        )

        if missing_predictors:
            raise ValueError(
                "Predictor columns do not exist: "
                f"{sorted(missing_predictors)}"
            )

        observed_mask = frame[
            self.column
        ].notna()

        training_frame = frame.loc[
            observed_mask,
            predictors,
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        target = frame.loc[
            observed_mask,
            self.column,
        ]

        if (
            len(training_frame)
            < self.options.minimum_training_rows
        ):
            raise ValueError(
                "Too few observed training rows for "
                "model-based imputation."
            )

        self._predictor_columns = (
            predictors
        )

        for predictor in predictors:
            mean_value = float(
                training_frame[
                    predictor
                ].mean()
            )

            if not np.isfinite(
                mean_value
            ):
                mean_value = 0.0

            self._feature_means[
                predictor
            ] = mean_value

            training_frame[
                predictor
            ] = training_frame[
                predictor
            ].fillna(
                mean_value
            )

        X = training_frame.to_numpy(
            dtype=float
        )

        X = np.column_stack(
            [
                np.ones(len(X)),
                X,
            ]
        )

        if self.options.task == "regression":
            y = pd.to_numeric(
                target,
                errors="coerce",
            )

            valid_target = y.notna()

            X = X[
                valid_target.to_numpy()
            ]

            y_values = y.loc[
                valid_target
            ].to_numpy(
                dtype=float
            )

            penalty = (
                self.options.ridge_alpha
                * np.eye(
                    X.shape[1]
                )
            )

            penalty[0, 0] = 0.0

            self._coefficients = (
                np.linalg.pinv(
                    X.T @ X + penalty
                )
                @ X.T
                @ y_values
            )

            fallback = float(
                np.median(
                    y_values
                )
            )

        else:
            counts = target.value_counts()

            self._classes = (
                counts.index.tolist()
            )

            fallback = (
                self._classes[0]
            )

            self._coefficients = None

        self.state.input_dtype = str(
            frame[self.column].dtype
        )

        self.state.training_row_count = len(
            frame
        )

        self.state.training_missing_count = int(
            frame[self.column].isna().sum()
        )

        self.state.learned_value = fallback

        self.state.metadata.update(
            {
                "predictor_columns": (
                    predictors
                ),
                "training_observed_count": (
                    int(observed_mask.sum())
                ),
            }
        )

        self.state.fitted = True

        return self

    def transform_frame(
        self,
        frame: pd.DataFrame,
        *,
        mode: ImputerExecutionMode,
    ) -> pd.Series:
        """Predict missing values in one partition."""

        if mode == ImputerExecutionMode.FIT:
            raise ValueError(
                "Use fit_frame() for fitting."
            )

        if not self.is_fitted:
            raise RuntimeError(
                "Model imputer is not fitted."
            )

        result = frame[
            self.column
        ].copy()

        missing_mask = result.isna()

        if not bool(
            missing_mask.any()
        ):
            return result

        if self.options.task == "classification":
            result.loc[
                missing_mask
            ] = self.state.learned_value

            return result

        if self._coefficients is None:
            raise RuntimeError(
                "Regression coefficients are unavailable."
            )

        predictor_frame = frame.loc[
            missing_mask,
            self._predictor_columns,
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        for predictor, mean_value in (
            self._feature_means.items()
        ):
            predictor_frame[
                predictor
            ] = predictor_frame[
                predictor
            ].fillna(
                mean_value
            )

        X = predictor_frame.to_numpy(
            dtype=float
        )

        X = np.column_stack(
            [
                np.ones(len(X)),
                X,
            ]
        )

        predictions = (
            X @ self._coefficients
        )

        result.loc[
            missing_mask
        ] = predictions

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
            "ModelImputer requires fit_frame()."
        )

    def _transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Disable one-Series transformation."""

        raise RuntimeError(
            "ModelImputer requires transform_frame()."
        )