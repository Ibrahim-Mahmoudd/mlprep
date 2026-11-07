"""Model-importance feature selection for Phase 7."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from mlprep.phases.phase7.contract.selection_plan import (
    ModelImportanceEstimator,
)
from mlprep.phases.phase7.selectors.base import (
    BaseFeatureSelector,
)


class ModelImportanceSelector(
    BaseFeatureSelector
):
    """Rank features with a lightweight tree estimator."""

    selector_name = "model_importance_selector"

    def __init__(
        self,
        *,
        task: str,
        estimator: ModelImportanceEstimator,
        number_of_estimators: int = 200,
        maximum_depth: int | None = None,
        top_k: int | None = None,
        minimum_score: float | None = None,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
        random_seed: int = 42,
        number_of_jobs: int = 1,
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

        if number_of_estimators < 1:
            raise ValueError(
                "number_of_estimators must be positive."
            )

        if number_of_jobs == 0:
            raise ValueError(
                "number_of_jobs cannot be zero."
            )

        self.estimator_kind = (
            estimator
            if isinstance(
                estimator,
                ModelImportanceEstimator,
            )
            else ModelImportanceEstimator(
                estimator
            )
        )

        self.number_of_estimators = (
            number_of_estimators
        )

        self.maximum_depth = maximum_depth
        self.number_of_jobs = number_of_jobs

    def _calculate_scores(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[
        dict[str, float],
        dict[str, Any],
    ]:
        """Fit the ranking estimator and read importances."""

        estimator = self._build_estimator()

        estimator.fit(
            features.to_numpy(
                dtype=float
            ),
            target.to_numpy(),
        )

        importances = (
            estimator.feature_importances_
        )

        scores = {
            str(column): self._safe_score(
                importances[
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
                "estimator": (
                    self.estimator_kind.value
                ),
                "number_of_estimators": (
                    self.number_of_estimators
                ),
                "maximum_depth": (
                    self.maximum_depth
                ),
                "number_of_jobs": (
                    self.number_of_jobs
                ),
            },
        )

    def _build_estimator(
        self,
    ) -> object:
        """Build the configured lightweight estimator."""

        common = {
            "n_estimators": (
                self.number_of_estimators
            ),
            "max_depth": (
                self.maximum_depth
            ),
            "random_state": (
                self.random_seed
            ),
            "n_jobs": (
                self.number_of_jobs
            ),
        }

        if (
            self.estimator_kind
            == ModelImportanceEstimator.RANDOM_FOREST
        ):
            if self.task == "classification":
                return RandomForestClassifier(
                    **common
                )

            return RandomForestRegressor(
                **common
            )

        if self.task == "classification":
            return ExtraTreesClassifier(
                **common
            )

        return ExtraTreesRegressor(
            **common
        )
