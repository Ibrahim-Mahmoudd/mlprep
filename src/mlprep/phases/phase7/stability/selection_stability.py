"""Bootstrap selection-stability evaluation for Phase 7."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase7.selectors.base import (
    SelectorFitContext,
    SelectorState,
)


SelectorFactory = Callable[
    [],
    object,
]


class FeatureSelectionStability(BaseModel):
    """Selection stability for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    selection_count: int = Field(
        default=0,
        ge=0,
    )

    bootstrap_count: int = Field(
        default=0,
        ge=1,
    )

    selection_frequency: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    mean_rank: float | None = None
    rank_standard_deviation: float | None = None

    minimum_rank: int | None = Field(
        default=None,
        ge=1,
    )

    maximum_rank: int | None = Field(
        default=None,
        ge=1,
    )

    stable: bool = True
    blocks_readiness: bool = False

    required_override: bool = False
    protected_override: bool = False

    instability_reasons: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SelectionStabilityResult(BaseModel):
    """Complete bootstrap selection-stability output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    features: dict[
        str,
        FeatureSelectionStability,
    ] = Field(
        default_factory=dict,
    )

    stable_features: list[str] = Field(
        default_factory=list,
    )

    unstable_features: list[str] = Field(
        default_factory=list,
    )

    blocking_features: list[str] = Field(
        default_factory=list,
    )

    bootstrap_selected_features: list[
        list[str]
    ] = Field(
        default_factory=list,
    )

    bootstrap_success_count: int = Field(
        default=0,
        ge=0,
    )

    bootstrap_failure_count: int = Field(
        default=0,
        ge=0,
    )

    fitted_on_training_only: bool = True
    target_access_authorized: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class SelectionStabilityOptions:
    """Options controlling bootstrap selection stability."""

    enabled: bool = False

    bootstrap_iterations: int = 30
    sample_fraction: float = 0.8

    sample_with_replacement: bool = True

    minimum_selection_frequency: float = 0.60

    maximum_rank_standard_deviation: float | None = None

    random_seed: int = 42

    stratify_classification_target: bool = True

    block_unstable_features: bool = False

    fail_on_bootstrap_error: bool = True

    store_bootstrap_selections: bool = True

    def __post_init__(self) -> None:
        """Validate selection-stability options."""

        if self.bootstrap_iterations < 1:
            raise ValueError(
                "bootstrap_iterations must be positive."
            )

        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError(
                "sample_fraction must be greater than zero "
                "and at most one."
            )

        if not 0.0 <= (
            self.minimum_selection_frequency
        ) <= 1.0:
            raise ValueError(
                "minimum_selection_frequency must be "
                "between zero and one."
            )

        if (
            self.maximum_rank_standard_deviation
            is not None
            and self.maximum_rank_standard_deviation < 0.0
        ):
            raise ValueError(
                "maximum_rank_standard_deviation cannot "
                "be negative."
            )


class SelectionStabilityEvaluator:
    """Evaluate selector stability over training resamples."""

    component_name = "selection_stability"

    def __init__(
        self,
        options: SelectionStabilityOptions | None = None,
    ) -> None:
        """Initialize the evaluator."""

        self.options = (
            options
            or SelectionStabilityOptions()
        )

    def evaluate(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        selector_factory: SelectorFactory,
        task: str,
        target_access_authorized: bool,
        evaluated_features: list[str] | None = None,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
    ) -> SelectionStabilityResult:
        """Evaluate stability using bootstrap training samples."""

        self._validate_inputs(
            features=features,
            target=target,
            task=task,
        )

        source_features = (
            [
                str(column)
                for column in features.columns
            ]
            if evaluated_features is None
            else [
                str(column)
                for column in evaluated_features
            ]
        )

        missing_features = [
            feature
            for feature in source_features
            if feature not in features.columns
        ]

        if missing_features:
            raise KeyError(
                "Selection-stability features are missing: "
                f"{missing_features}"
            )

        if not self.options.enabled:
            return SelectionStabilityResult(
                stable_features=source_features,
                target_access_authorized=(
                    target_access_authorized
                ),
                metadata={
                    "enabled": False,
                },
            )

        if not target_access_authorized:
            raise PermissionError(
                "Selection-stability evaluation requires "
                "authorized training-target access."
            )

        required = set(
            required_features or []
        )

        protected = set(
            protected_features or []
        )

        selection_counts = {
            column: 0
            for column in source_features
        }

        rank_values: dict[
            str,
            list[int],
        ] = {
            column: []
            for column in source_features
        }

        bootstrap_selections: list[
            list[str]
        ] = []

        success_count = 0
        failure_count = 0

        random_generator = (
            np.random.default_rng(
                self.options.random_seed
            )
        )

        for iteration in range(
            self.options.bootstrap_iterations
        ):
            try:
                sampled_positions = self._sample_positions(
                    target=target,
                    task=task,
                    random_generator=random_generator,
                )

                sampled_features = (
                    features.iloc[
                        sampled_positions
                    ].copy(
                        deep=True
                    )
                )

                sampled_target = (
                    target.iloc[
                        sampled_positions
                    ].copy(
                        deep=True
                    )
                )

                sampled_target.index = (
                    sampled_features.index
                )

                selector = selector_factory()

                if not hasattr(
                    selector,
                    "fit",
                ):
                    raise TypeError(
                        "selector_factory must return an object "
                        "with a fit() method."
                    )

                selector.fit(
                    features=sampled_features,
                    target=sampled_target,
                    context=(
                        SelectorFitContext
                        .TRAINING_SUPERVISED
                    ),
                    target_access_authorized=True,
                )

                if not hasattr(
                    selector,
                    "get_state",
                ):
                    raise TypeError(
                        "The bootstrap selector must expose "
                        "get_state()."
                    )

                state = selector.get_state()

                if not isinstance(
                    state,
                    SelectorState,
                ):
                    raise TypeError(
                        "The bootstrap selector state must be "
                        "a SelectorState."
                    )

                selected = [
                    column
                    for column
                    in source_features
                    if column
                    in set(
                        state.selected_features
                    )
                ]

                if self.options.store_bootstrap_selections:
                    bootstrap_selections.append(
                        selected
                    )

                for column in selected:
                    selection_counts[
                        column
                    ] += 1

                for column in source_features:
                    feature_score = (
                        state
                        .feature_scores
                        .get(
                            column
                        )
                    )

                    if feature_score is not None:
                        rank_values[
                            column
                        ].append(
                            feature_score.rank
                        )

                success_count += 1

            except Exception:
                failure_count += 1

                if self.options.fail_on_bootstrap_error:
                    raise

        if success_count < 1:
            raise RuntimeError(
                "Every bootstrap selector execution failed."
            )

        feature_results: dict[
            str,
            FeatureSelectionStability,
        ] = {}

        stable_features: list[str] = []
        unstable_features: list[str] = []
        blocking_features: list[str] = []

        for column in source_features:
            selection_frequency = (
                selection_counts[
                    column
                ]
                / success_count
            )

            ranks = rank_values[
                column
            ]

            mean_rank = (
                float(
                    np.mean(
                        ranks
                    )
                )
                if ranks
                else None
            )

            rank_standard_deviation = (
                float(
                    np.std(
                        ranks,
                        ddof=0,
                    )
                )
                if ranks
                else None
            )

            minimum_rank = (
                min(
                    ranks
                )
                if ranks
                else None
            )

            maximum_rank = (
                max(
                    ranks
                )
                if ranks
                else None
            )

            reasons: list[str] = []

            if (
                selection_frequency
                < self.options
                .minimum_selection_frequency
            ):
                reasons.append(
                    "Selection frequency is below the "
                    "configured minimum."
                )

            if (
                self.options
                .maximum_rank_standard_deviation
                is not None
                and rank_standard_deviation is not None
                and rank_standard_deviation
                > self.options
                .maximum_rank_standard_deviation
            ):
                reasons.append(
                    "Rank variability exceeds the "
                    "configured maximum."
                )

            required_override = bool(
                column in required
                and reasons
            )

            protected_override = bool(
                column in protected
                and reasons
            )

            stable = not reasons

            blocks_readiness = bool(
                reasons
                and self.options.block_unstable_features
                and not required_override
                and not protected_override
            )

            if stable:
                stable_features.append(
                    column
                )

            else:
                unstable_features.append(
                    column
                )

            if blocks_readiness:
                blocking_features.append(
                    column
                )

            feature_results[
                column
            ] = FeatureSelectionStability(
                column=column,
                selection_count=(
                    selection_counts[
                        column
                    ]
                ),
                bootstrap_count=(
                    success_count
                ),
                selection_frequency=(
                    selection_frequency
                ),
                mean_rank=mean_rank,
                rank_standard_deviation=(
                    rank_standard_deviation
                ),
                minimum_rank=minimum_rank,
                maximum_rank=maximum_rank,
                stable=stable,
                blocks_readiness=(
                    blocks_readiness
                ),
                required_override=(
                    required_override
                ),
                protected_override=(
                    protected_override
                ),
                instability_reasons=(
                    reasons
                ),
            )

        return SelectionStabilityResult(
            valid=not blocking_features,
            features=feature_results,
            stable_features=stable_features,
            unstable_features=unstable_features,
            blocking_features=blocking_features,
            bootstrap_selected_features=(
                bootstrap_selections
            ),
            bootstrap_success_count=(
                success_count
            ),
            bootstrap_failure_count=(
                failure_count
            ),
            fitted_on_training_only=True,
            target_access_authorized=True,
            metadata={
                "enabled": True,
                "requested_bootstrap_count": (
                    self.options
                    .bootstrap_iterations
                ),
                "successful_bootstrap_count": (
                    success_count
                ),
                "failed_bootstrap_count": (
                    failure_count
                ),
                "sample_fraction": (
                    self.options.sample_fraction
                ),
                "sample_with_replacement": (
                    self.options
                    .sample_with_replacement
                ),
                "stable_feature_count": len(
                    stable_features
                ),
                "unstable_feature_count": len(
                    unstable_features
                ),
            },
        )

    def _sample_positions(
        self,
        *,
        target: pd.Series,
        task: str,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        """Create one deterministic bootstrap sample."""

        row_count = len(
            target
        )

        sample_count = max(
            2,
            int(
                round(
                    row_count
                    * self.options
                    .sample_fraction
                )
            ),
        )

        if (
            task == "classification"
            and self.options
            .stratify_classification_target
        ):
            sampled_positions: list[int] = []

            target_values = target.to_numpy()

            for class_value in pd.unique(
                target_values
            ):
                class_positions = np.flatnonzero(
                    target_values
                    == class_value
                )

                class_sample_count = max(
                    1,
                    int(
                        round(
                            len(
                                class_positions
                            )
                            * self.options
                            .sample_fraction
                        )
                    ),
                )

                class_sample = (
                    random_generator.choice(
                        class_positions,
                        size=class_sample_count,
                        replace=(
                            self.options
                            .sample_with_replacement
                        ),
                    )
                )

                sampled_positions.extend(
                    int(position)
                    for position
                    in class_sample
                )

            sampled = np.asarray(
                sampled_positions,
                dtype=int,
            )

            random_generator.shuffle(
                sampled
            )

            return sampled

        return random_generator.choice(
            np.arange(
                row_count
            ),
            size=sample_count,
            replace=(
                self.options
                .sample_with_replacement
            ),
        ).astype(
            int
        )

    @staticmethod
    def _validate_inputs(
        *,
        features: pd.DataFrame,
        target: pd.Series,
        task: str,
    ) -> None:
        """Validate bootstrap stability inputs."""

        if task not in {
            "classification",
            "regression",
        }:
            raise ValueError(
                "task must be 'classification' or 'regression'."
            )

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
                "Training features and target have different "
                "row counts."
            )

        if not features.index.equals(
            target.index
        ):
            raise ValueError(
                "Training features and target have different "
                "indices."
            )

        if target.nunique(
            dropna=True
        ) < 2:
            raise ValueError(
                "Selection stability requires at least two "
                "distinct target values."
            )

        if target.isna().any():
            raise ValueError(
                "Selection stability requires a complete target."
            )

        if features.isna().any().any():
            raise ValueError(
                "Selection stability requires complete features."
            )

        nonnumeric = [
            str(column)
            for column in features.columns
            if not pd.api.types.is_numeric_dtype(
                features[
                    column
                ].dtype
            )
        ]

        if nonnumeric:
            raise TypeError(
                "Selection stability requires numeric "
                f"features: {nonnumeric}"
            )