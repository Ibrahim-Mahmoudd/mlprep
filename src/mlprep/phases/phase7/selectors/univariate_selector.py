"""Univariate supervised feature selection for Phase 7."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    chi2,
    f_classif,
    f_regression,
)

from mlprep.phases.phase7.contract.selection_plan import (
    UnivariateMethod,
)
from mlprep.phases.phase7.selectors.base import (
    BaseFeatureSelector,
)


class UnivariateSelector(BaseFeatureSelector):
    """Rank features using task-aware univariate methods."""

    selector_name = "univariate_selector"

    def __init__(
        self,
        *,
        task: str,
        method: UnivariateMethod,
        top_k: int | None = None,
        minimum_score: float | None = None,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
    ) -> None:
        """Initialize the univariate selector."""

        super().__init__(
            task=task,
            top_k=top_k,
            minimum_score=minimum_score,
            required_features=required_features,
            protected_features=protected_features,
        )

        self.method = (
            method
            if isinstance(
                method,
                UnivariateMethod,
            )
            else UnivariateMethod(
                method
            )
        )

        self._validate_method()

    def _calculate_scores(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[
        dict[str, float],
        dict[str, Any],
    ]:
        """Calculate univariate scores."""

        values = features.to_numpy(
            dtype=float
        )

        target_values = target.to_numpy()

        if self.method == UnivariateMethod.ANOVA:
            raw_scores, p_values = f_classif(
                values,
                target_values,
            )

        elif (
            self.method
            == UnivariateMethod.CHI_SQUARE
        ):
            minimum_value = float(
                np.min(
                    values
                )
            )

            if minimum_value < 0.0:
                raise ValueError(
                    "Chi-square selection requires "
                    "nonnegative feature values."
                )

            raw_scores, p_values = chi2(
                values,
                target_values,
            )

        elif (
            self.method
            == UnivariateMethod.F_REGRESSION
        ):
            raw_scores, p_values = f_regression(
                values,
                target_values,
            )

        elif (
            self.method
            == UnivariateMethod.CORRELATION
        ):
            numeric_target = pd.to_numeric(
                target,
                errors="raise",
            )

            raw_scores = np.array(
                [
                    abs(
                        float(
                            features[
                                column
                            ].corr(
                                numeric_target
                            )
                            or 0.0
                        )
                    )
                    for column
                    in features.columns
                ],
                dtype=float,
            )

            p_values = np.full(
                len(
                    features.columns
                ),
                np.nan,
            )

        else:
            raise ValueError(
                f"Unsupported univariate method: "
                f"{self.method!r}"
            )

        scores = {
            str(column): self._safe_score(
                raw_scores[
                    index
                ]
            )
            for index, column
            in enumerate(
                features.columns
            )
        }

        p_value_mapping = {
            str(column): (
                None
                if not np.isfinite(
                    p_values[
                        index
                    ]
                )
                else float(
                    p_values[
                        index
                    ]
                )
            )
            for index, column
            in enumerate(
                features.columns
            )
        }

        return (
            scores,
            {
                "method": self.method.value,
                "p_values": p_value_mapping,
            },
        )

    def _validate_method(
        self,
    ) -> None:
        """Validate task and method compatibility."""

        classification_methods = {
            UnivariateMethod.ANOVA,
            UnivariateMethod.CHI_SQUARE,
        }

        regression_methods = {
            UnivariateMethod.F_REGRESSION,
            UnivariateMethod.CORRELATION,
        }

        if (
            self.task == "classification"
            and self.method
            not in classification_methods
        ):
            raise ValueError(
                "The selected univariate method is not "
                "valid for classification."
            )

        if (
            self.task == "regression"
            and self.method
            not in regression_methods
        ):
            raise ValueError(
                "The selected univariate method is not "
                "valid for regression."
            )
