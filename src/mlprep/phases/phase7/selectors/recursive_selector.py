"""Recursive feature elimination for Phase 7."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)

from mlprep.phases.phase7.contract.selection_plan import (
    RecursiveEstimator,
)
from mlprep.phases.phase7.selectors.base import (
    BaseFeatureSelector,
)


class RecursiveSelector(BaseFeatureSelector):
    """Apply deterministic recursive feature elimination."""

    selector_name = "recursive_selector"

    def __init__(
        self,
        *,
        task: str,
        estimator: RecursiveEstimator,
        target_feature_count: int | None = None,
        minimum_feature_count: int = 1,
        removal_step: int = 1,
        maximum_iterations: int = 100,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
        random_seed: int = 42,
    ) -> None:
        """Initialize the recursive selector."""

        super().__init__(
            task=task,
            top_k=target_feature_count,
            minimum_score=None,
            required_features=required_features,
            protected_features=protected_features,
            random_seed=random_seed,
        )

        if minimum_feature_count < 1:
            raise ValueError(
                "minimum_feature_count must be positive."
            )

        if removal_step < 1:
            raise ValueError(
                "removal_step must be positive."
            )

        if maximum_iterations < 1:
            raise ValueError(
                "maximum_iterations must be positive."
            )

        self.estimator_kind = (
            estimator
            if isinstance(
                estimator,
                RecursiveEstimator,
            )
            else RecursiveEstimator(
                estimator
            )
        )

        self.target_feature_count = (
            target_feature_count
        )

        self.minimum_feature_count = (
            minimum_feature_count
        )

        self.removal_step = removal_step

        self.maximum_iterations = (
            maximum_iterations
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
        """Perform recursive elimination and produce rankings."""

        source_order = [
            str(column)
            for column in features.columns
        ]

        protected = {
            *self.required_features,
            *self.protected_features,
        }

        target_count = (
            self.target_feature_count
            if self.target_feature_count is not None
            else self.minimum_feature_count
        )

        target_count = max(
            target_count,
            self.minimum_feature_count,
            len(
                protected.intersection(
                    source_order
                )
            ),
        )

        target_count = min(
            target_count,
            len(
                source_order
            ),
        )

        active = list(
            source_order
        )

        elimination_order: list[str] = []

        iterations: list[
            dict[str, Any]
        ] = []

        for iteration in range(
            1,
            self.maximum_iterations + 1,
        ):
            if len(active) <= target_count:
                break

            estimator = self._build_estimator()

            estimator.fit(
                features.loc[
                    :,
                    active,
                ].to_numpy(
                    dtype=float
                ),
                target.to_numpy(),
            )

            importances = self._importances(
                estimator
            )

            removable = [
                column
                for column in active
                if column not in protected
            ]

            if not removable:
                break

            importance_mapping = {
                column: float(
                    importances[
                        index
                    ]
                )
                for index, column
                in enumerate(
                    active
                )
            }

            removable.sort(
                key=lambda column: (
                    importance_mapping[
                        column
                    ],
                    -source_order.index(
                        column
                    ),
                )
            )

            maximum_removal = (
                len(active)
                - target_count
            )

            remove_count = min(
                self.removal_step,
                maximum_removal,
                len(
                    removable
                ),
            )

            if remove_count <= 0:
                break

            removed = removable[
                : remove_count
            ]

            for column in removed:
                active.remove(
                    column
                )

                elimination_order.append(
                    column
                )

            iterations.append(
                {
                    "iteration": iteration,
                    "removed": removed,
                    "remaining_count": len(
                        active
                    ),
                }
            )

        final_estimator = self._build_estimator()

        final_estimator.fit(
            features.loc[
                :,
                active,
            ].to_numpy(
                dtype=float
            ),
            target.to_numpy(),
        )

        final_importances = self._importances(
            final_estimator
        )

        scores = {
            column: 0.0
            for column in source_order
        }

        for index, column in enumerate(
            active
        ):
            scores[
                column
            ] = self._safe_score(
                final_importances[
                    index
                ]
            ) + 1.0

        elimination_total = len(
            elimination_order
        )

        for index, column in enumerate(
            elimination_order
        ):
            scores[
                column
            ] = (
                index + 1
            ) / max(
                elimination_total + 1,
                1,
            )

        self.top_k = len(
            active
        )

        return (
            scores,
            {
                "estimator": (
                    self.estimator_kind.value
                ),
                "target_feature_count": (
                    target_count
                ),
                "iterations": iterations,
                "elimination_order": (
                    elimination_order
                ),
                "final_features": (
                    active
                ),
                "converged": bool(
                    len(active)
                    <= target_count
                ),
            },
        )

    def _build_estimator(
        self,
    ) -> object:
        """Build the recursive ranking estimator."""

        if (
            self.estimator_kind
            == RecursiveEstimator.LINEAR
        ):
            if self.task == "classification":
                return LogisticRegression(
                    max_iter=1000,
                    random_state=(
                        self.random_seed
                    ),
                )

            return LinearRegression()

        if self.task == "classification":
            return RandomForestClassifier(
                n_estimators=150,
                random_state=(
                    self.random_seed
                ),
                n_jobs=1,
            )

        return RandomForestRegressor(
            n_estimators=150,
            random_state=(
                self.random_seed
            ),
            n_jobs=1,
        )

    @staticmethod
    def _importances(
        estimator: object,
    ) -> np.ndarray:
        """Extract nonnegative estimator importances."""

        if hasattr(
            estimator,
            "feature_importances_",
        ):
            return np.asarray(
                estimator.feature_importances_,
                dtype=float,
            )

        if hasattr(
            estimator,
            "coef_",
        ):
            coefficients = np.asarray(
                estimator.coef_,
                dtype=float,
            )

            if coefficients.ndim == 1:
                return np.abs(
                    coefficients
                )

            return np.mean(
                np.abs(
                    coefficients
                ),
                axis=0,
            )

        raise TypeError(
            "The recursive estimator does not expose "
            "feature importances or coefficients."
        )
