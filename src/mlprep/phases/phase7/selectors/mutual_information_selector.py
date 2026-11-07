"""Mutual-information feature selection for Phase 7."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

from mlprep.phases.phase7.selectors.base import (
    BaseFeatureSelector,
)


class MutualInformationSelector(
    BaseFeatureSelector
):
    """Rank features using estimated mutual information."""

    selector_name = "mutual_information_selector"

    def __init__(
        self,
        *,
        task: str,
        top_k: int | None = None,
        minimum_score: float | None = None,
        discrete_features: str = "auto",
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
        random_seed: int = 42,
    ) -> None:
        """Initialize the selector."""

        super().__init__(
            task=task,
            top_k=top_k,
            minimum_score=minimum_score,
            required_features=required_features,
            protected_features=protected_features,
            random_seed=random_seed,
        )

        if discrete_features not in {
            "auto",
            "all",
            "none",
        }:
            raise ValueError(
                "discrete_features must be 'auto', "
                "'all', or 'none'."
            )

        self.discrete_features = (
            discrete_features
        )

    def _calculate_scores(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[
        dict[str, float],
        dict[str, Any],
    ]:
        """Calculate mutual-information scores."""

        discrete_mask = (
            self._discrete_mask(
                features
            )
        )

        values = features.to_numpy(
            dtype=float
        )

        target_values = target.to_numpy()

        if self.task == "classification":
            raw_scores = mutual_info_classif(
                values,
                target_values,
                discrete_features=(
                    discrete_mask
                ),
                random_state=(
                    self.random_seed
                ),
            )

        else:
            raw_scores = mutual_info_regression(
                values,
                target_values,
                discrete_features=(
                    discrete_mask
                ),
                random_state=(
                    self.random_seed
                ),
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

        return (
            scores,
            {
                "discrete_features": (
                    self.discrete_features
                ),
                "discrete_feature_columns": [
                    str(column)
                    for index, column
                    in enumerate(
                        features.columns
                    )
                    if bool(
                        discrete_mask[
                            index
                        ]
                    )
                ],
            },
        )

    def _discrete_mask(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Resolve discrete-feature indicators."""

        if self.discrete_features == "all":
            return np.ones(
                len(
                    features.columns
                ),
                dtype=bool,
            )

        if self.discrete_features == "none":
            return np.zeros(
                len(
                    features.columns
                ),
                dtype=bool,
            )

        return np.array(
            [
                self._is_discrete(
                    features[
                        column
                    ]
                )
                for column
                in features.columns
            ],
            dtype=bool,
        )

    @staticmethod
    def _is_discrete(
        series: pd.Series,
    ) -> bool:
        """Infer whether a numeric feature is discrete."""

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        if pd.api.types.is_bool_dtype(
            series.dtype
        ):
            return True

        if pd.api.types.is_integer_dtype(
            series.dtype
        ):
            return bool(
                unique_count
                <= max(
                    20,
                    int(
                        len(series)
                        * 0.05
                    ),
                )
            )

        unique_values = set(
            series.dropna().unique()
        )

        return unique_values.issubset(
            {
                0,
                1,
            }
        )
