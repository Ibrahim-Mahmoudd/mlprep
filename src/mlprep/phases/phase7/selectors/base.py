"""Base supervised feature selectors for Phase 7."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class SelectorFitContext(StrEnum):
    """Supported selector fitting contexts."""

    TRAINING_UNSUPERVISED = "training_unsupervised"
    TRAINING_SUPERVISED = "training_supervised"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"


class FeatureScore(BaseModel):
    """Score and ranking information for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    score: float

    rank: int = Field(
        ge=1,
    )

    selected: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SelectorState(BaseModel):
    """Serializable fitted state for one selector."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    selector: str
    task: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    target_access_used: bool = False
    target_access_authorized: bool = False

    source_features: list[str] = Field(
        default_factory=list,
    )

    selected_features: list[str] = Field(
        default_factory=list,
    )

    rejected_features: list[str] = Field(
        default_factory=list,
    )

    feature_scores: dict[
        str,
        FeatureScore,
    ] = Field(
        default_factory=dict,
    )

    random_seed: int | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class BaseFeatureSelector(ABC):
    """Common interface for Phase 7 supervised selectors."""

    selector_name = "base_selector"

    def __init__(
        self,
        *,
        task: str,
        top_k: int | None = None,
        minimum_score: float | None = None,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the feature selector."""

        if task not in {
            "classification",
            "regression",
        }:
            raise ValueError(
                "task must be 'classification' or 'regression'."
            )

        if top_k is not None and top_k < 1:
            raise ValueError(
                "top_k must be positive when provided."
            )

        if random_seed is not None and not isinstance(
            random_seed,
            int,
        ):
            raise TypeError(
                "random_seed must be an integer or None."
            )

        self.task = task
        self.top_k = top_k
        self.minimum_score = minimum_score

        self.required_features = self._normalize_feature_list(
            required_features or [],
            name="required_features",
        )

        self.protected_features = self._normalize_feature_list(
            protected_features or [],
            name="protected_features",
        )

        self.random_seed = random_seed

        self.state = SelectorState(
            selector=self.selector_name,
            task=task,
            random_seed=random_seed,
        )

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the selector is fitted."""

        return self.state.fitted

    def fit(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        context: SelectorFitContext | str,
        target_access_authorized: bool,
    ) -> Self:
        """Fit the selector on authorized training data."""

        self._validate_inputs(
            features=features,
            target=target,
        )

        if not isinstance(
            context,
            SelectorFitContext,
        ):
            try:
                context = SelectorFitContext(
                    context
                )

            except ValueError as error:
                raise ValueError(
                    f"Unsupported selector fit context: {context!r}"
                ) from error

        if (
            context
            != SelectorFitContext.TRAINING_SUPERVISED
        ):
            raise PermissionError(
                "Supervised selectors may only be fitted "
                "in the training-supervised context."
            )

        if not isinstance(
            target_access_authorized,
            bool,
        ):
            raise TypeError(
                "target_access_authorized must be a Boolean."
            )

        if not target_access_authorized:
            raise PermissionError(
                "Supervised feature selection requires "
                "explicit training-target access authorization."
            )

        if self.is_fitted:
            raise RuntimeError(
                f"Selector {self.selector_name!r} "
                "is already fitted."
            )

        source_features = [
            str(column)
            for column in features.columns
        ]

        self._validate_configured_features(
            source_features=source_features,
        )

        self.state.source_features = list(
            source_features
        )

        raw_scores, metadata = self._calculate_scores(
            features=features,
            target=target,
        )

        if not isinstance(
            raw_scores,
            dict,
        ):
            raise TypeError(
                "Selector score calculation must return "
                "a dictionary of feature scores."
            )

        if not isinstance(
            metadata,
            dict,
        ):
            raise TypeError(
                "Selector score calculation must return "
                "a metadata dictionary."
            )

        unknown_score_columns = sorted(
            set(
                str(column)
                for column in raw_scores
            ).difference(
                source_features
            )
        )

        if unknown_score_columns:
            raise ValueError(
                "Selector scores contain unknown features: "
                f"{unknown_score_columns}"
            )

        normalized_scores = {
            column: self._safe_score(
                raw_scores.get(
                    column,
                    0.0,
                )
            )
            for column in source_features
        }

        rankings = self._rank_scores(
            scores=normalized_scores,
            source_order=source_features,
        )

        selected = self._select(
            rankings=rankings,
            scores=normalized_scores,
        )

        selected = self._restore_protected(
            selected=selected,
            source_order=source_features,
        )

        selected_set = set(
            selected
        )

        feature_scores = {
            column: FeatureScore(
                column=column,
                score=normalized_scores[
                    column
                ],
                rank=rankings[
                    column
                ],
                selected=(
                    column in selected_set
                ),
            )
            for column in source_features
        }

        self.state.selected_features = list(
            selected
        )

        self.state.rejected_features = [
            column
            for column in source_features
            if column not in selected_set
        ]

        self.state.feature_scores = (
            feature_scores
        )

        self.state.target_access_used = True
        self.state.target_access_authorized = True

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        self.state.metadata.update(
            metadata
        )

        self.state.metadata.update(
            {
                "source_feature_count": len(
                    source_features
                ),
                "selected_feature_count": len(
                    selected
                ),
                "rejected_feature_count": len(
                    self.state.rejected_features
                ),
                "top_k": self.top_k,
                "minimum_score": (
                    self.minimum_score
                ),
                "required_features": list(
                    self.required_features
                ),
                "protected_features": list(
                    self.protected_features
                ),
            }
        )

        return self

    def rank_features(
        self,
    ) -> list[str]:
        """Return features ordered by fitted rank."""

        self._assert_fitted()

        source_order = {
            column: index
            for index, column
            in enumerate(
                self.state.source_features
            )
        }

        return sorted(
            self.state.source_features,
            key=lambda column: (
                self.state
                .feature_scores[
                    column
                ]
                .rank,
                source_order[
                    column
                ],
            ),
        )

    def selected_features(
        self,
    ) -> list[str]:
        """Return selected features in source order."""

        self._assert_fitted()

        return list(
            self.state.selected_features
        )

    def rejected_features(
        self,
    ) -> list[str]:
        """Return rejected features in source order."""

        self._assert_fitted()

        return list(
            self.state.rejected_features
        )

    def feature_scores(
        self,
    ) -> dict[str, FeatureScore]:
        """Return fitted feature scores."""

        self._assert_fitted()

        return {
            column: score.model_copy(
                deep=True
            )
            for column, score
            in self.state.feature_scores.items()
        }

    def get_state(
        self,
    ) -> SelectorState:
        """Return a deep copy of fitted selector state."""

        self._assert_fitted()

        return self.state.model_copy(
            deep=True
        )

    @abstractmethod
    def _calculate_scores(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[
        dict[str, float],
        dict[str, Any],
    ]:
        """Calculate selector-specific feature scores."""

    def _select(
        self,
        *,
        rankings: dict[str, int],
        scores: dict[str, float],
    ) -> list[str]:
        """Apply score and feature-count selection policies."""

        source_order = {
            column: index
            for index, column
            in enumerate(
                self.state.source_features
            )
        }

        ranked_features = sorted(
            scores,
            key=lambda column: (
                rankings[
                    column
                ],
                source_order.get(
                    column,
                    len(
                        source_order
                    ),
                ),
                column,
            ),
        )

        if self.minimum_score is not None:
            ranked_features = [
                column
                for column in ranked_features
                if scores[
                    column
                ] >= self.minimum_score
            ]

        if self.top_k is not None:
            ranked_features = (
                ranked_features[
                    : self.top_k
                ]
            )

        selected_set = set(
            ranked_features
        )

        return [
            column
            for column
            in self.state.source_features
            if column in selected_set
        ]

    def _restore_protected(
        self,
        *,
        selected: list[str],
        source_order: list[str],
    ) -> list[str]:
        """Restore required and protected features."""

        selected_set = set(
            selected
        )

        selected_set.update(
            feature
            for feature
            in self.required_features
            if feature in source_order
        )

        selected_set.update(
            feature
            for feature
            in self.protected_features
            if feature in source_order
        )

        return [
            column
            for column in source_order
            if column in selected_set
        ]

    @staticmethod
    def _rank_scores(
        *,
        scores: dict[str, float],
        source_order: list[str],
    ) -> dict[str, int]:
        """Rank feature scores deterministically."""

        order_index = {
            column: index
            for index, column
            in enumerate(
                source_order
            )
        }

        ordered_features = sorted(
            scores,
            key=lambda column: (
                -scores[
                    column
                ],
                order_index.get(
                    column,
                    len(
                        order_index
                    ),
                ),
                column,
            ),
        )

        return {
            column: rank
            for rank, column
            in enumerate(
                ordered_features,
                start=1,
            )
        }

    @staticmethod
    def _safe_score(
        value: object,
    ) -> float:
        """Convert a selector score to a finite float."""

        if not isinstance(value, (int, float, np.number)):
            return 0.0

        try:
            score = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not np.isfinite(
            score
        ):
            return 0.0

        return score

    @staticmethod
    def _validate_inputs(
        *,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        """Validate selector inputs."""

        if not isinstance(
            features,
            pd.DataFrame,
        ):
            raise TypeError(
                "features must be a pandas DataFrame."
            )

        if not isinstance(
            target,
            pd.Series,
        ):
            raise TypeError(
                "target must be a pandas Series."
            )

        if features.empty:
            raise ValueError(
                "Training features cannot be empty."
            )

        if len(features) != len(target):
            raise ValueError(
                "Features and target have different row counts."
            )

        if not features.index.equals(
            target.index
        ):
            raise ValueError(
                "Features and target have different indices."
            )

        if features.columns.duplicated().any():
            duplicate_columns = [
                str(column)
                for column
                in features.columns[
                    features.columns.duplicated(
                        keep=False
                    )
                ].unique()
            ]

            raise ValueError(
                "Feature names must be unique. "
                f"Duplicate features: {duplicate_columns}"
            )

        nonnumeric_columns = [
            str(column)
            for column in features.columns
            if not pd.api.types.is_numeric_dtype(
                features[
                    column
                ].dtype
            )
        ]

        if nonnumeric_columns:
            raise TypeError(
                "Feature selectors require numeric features: "
                f"{nonnumeric_columns}"
            )

        missing_feature_columns = [
            str(column)
            for column in features.columns
            if features[
                column
            ].isna().any()
        ]

        if missing_feature_columns:
            raise ValueError(
                "Feature selectors require complete features. "
                "Missing values were found in: "
                f"{missing_feature_columns}"
            )

        nonfinite_feature_columns: list[str] = []

        for column in features.columns:
            numeric = pd.to_numeric(
                features[
                    column
                ],
                errors="coerce",
            )

            nonfinite_mask = (
                numeric.notna()
                & ~np.isfinite(
                    numeric.astype(float)
                )
            )

            if bool(
                nonfinite_mask.any()
            ):
                nonfinite_feature_columns.append(
                    str(column)
                )

        if nonfinite_feature_columns:
            raise ValueError(
                "Feature selectors require finite features. "
                "Nonfinite values were found in: "
                f"{nonfinite_feature_columns}"
            )

        if target.isna().any():
            raise ValueError(
                "Feature selectors require a complete target."
            )

        if target.nunique(
            dropna=True
        ) < 2:
            raise ValueError(
                "The training target must contain at least "
                "two distinct values."
            )

    def _validate_configured_features(
        self,
        *,
        source_features: list[str],
    ) -> None:
        """Validate required and protected feature names."""

        available = set(
            source_features
        )

        missing_required = sorted(
            set(
                self.required_features
            ).difference(
                available
            )
        )

        if missing_required:
            raise KeyError(
                "Required selector features are unavailable: "
                f"{missing_required}"
            )

        missing_protected = sorted(
            set(
                self.protected_features
            ).difference(
                available
            )
        )

        if missing_protected:
            raise KeyError(
                "Protected selector features are unavailable: "
                f"{missing_protected}"
            )

    @staticmethod
    def _normalize_feature_list(
        features: list[str],
        *,
        name: str,
    ) -> list[str]:
        """Normalize and validate one feature-name list."""

        normalized = [
            str(feature).strip()
            for feature in features
        ]

        if any(
            not feature
            for feature in normalized
        ):
            raise ValueError(
                f"{name} cannot contain empty feature names."
            )

        if len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(
                f"{name} contains duplicate feature names."
            )

        return normalized

    def _assert_fitted(
        self,
    ) -> None:
        """Raise when fitted selector state is unavailable."""

        if not self.is_fitted:
            raise RuntimeError(
                f"Selector {self.selector_name!r} "
                "is not fitted."
            )
