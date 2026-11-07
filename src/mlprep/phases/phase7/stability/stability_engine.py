"""Unified feature-stability engine for Phase 7."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase7.stability.distribution_stability import (
    DistributionStabilityEvaluator,
    DistributionStabilityOptions,
    DistributionStabilityResult,
)
from mlprep.phases.phase7.stability.selection_stability import (
    SelectionStabilityEvaluator,
    SelectionStabilityOptions,
    SelectionStabilityResult,
)

SelectorFactory = Callable[
    [],
    object,
]


class AggregatedStabilityDecision(BaseModel):
    """Combined stability decision for one selected feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    distribution_stable: bool = True
    selection_stable: bool = True

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


class StabilityEngineResult(BaseModel):
    """Complete Phase 7 stability-engine output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    valid: bool = True

    distribution: DistributionStabilityResult | None = None

    selection: SelectionStabilityResult | None = None

    decisions: dict[
        str,
        AggregatedStabilityDecision,
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

    retained_features: list[str] = Field(
        default_factory=list,
    )

    removed_unstable_features: list[str] = Field(
        default_factory=list,
    )

    fitted_on_training_only: bool = True

    validation_used_for_evaluation_only: bool = True

    target_access_authorized: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class StabilityEngineOptions:
    """Options controlling the stability engine."""

    enabled: bool = True

    distribution: DistributionStabilityOptions = field(
        default_factory=DistributionStabilityOptions,
    )

    selection: SelectionStabilityOptions = field(
        default_factory=SelectionStabilityOptions,
    )

    remove_unstable_features: bool = False

    require_at_least_one_feature: bool = True

    require_distribution_stability: bool = False
    require_selection_stability: bool = False

    def __post_init__(self) -> None:
        """Validate stability-engine options."""

        boolean_options = {
            "enabled": self.enabled,
            "remove_unstable_features": (
                self.remove_unstable_features
            ),
            "require_at_least_one_feature": (
                self.require_at_least_one_feature
            ),
            "require_distribution_stability": (
                self.require_distribution_stability
            ),
            "require_selection_stability": (
                self.require_selection_stability
            ),
        }

        for name, value in boolean_options.items():
            if not isinstance(
                value,
                bool,
            ):
                raise TypeError(
                    f"{name} must be a Boolean."
                )


class StabilityEngine:
    """Evaluate distribution and bootstrap-selection stability."""

    component_name = "stability_engine"

    def __init__(
        self,
        options: StabilityEngineOptions | None = None,
    ) -> None:
        """Initialize the stability engine."""

        self.options = (
            options
            or StabilityEngineOptions()
        )

        self.result: StabilityEngineResult | None = None

    def evaluate(
        self,
        *,
        training_features: pd.DataFrame,
        validation_features: pd.DataFrame,
        training_target: pd.Series,
        selected_features: list[str],
        task: str,
        target_access_authorized: bool,
        selector_factory: SelectorFactory | None = None,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
    ) -> StabilityEngine:
        """Evaluate all configured stability components."""

        selected = [
            str(column)
            for column in selected_features
        ]

        self._validate_features(
            training=training_features,
            validation=validation_features,
            selected_features=selected,
        )

        required = set(
            required_features or []
        )

        protected = set(
            protected_features or []
        )

        if not self.options.enabled:
            self.result = StabilityEngineResult(
                stable_features=selected,
                retained_features=selected,
                target_access_authorized=(
                    target_access_authorized
                ),
                metadata={
                    "enabled": False,
                },
            )

            return self

        distribution_result = None
        selection_result = None

        if self.options.distribution.enabled:
            distribution_result = (
                DistributionStabilityEvaluator(
                    options=self.options.distribution
                ).evaluate(
                    training=training_features,
                    comparison=validation_features,
                    features=selected,
                    required_features=list(
                        required
                    ),
                    protected_features=list(
                        protected
                    ),
                )
            )

        elif (
            self.options
            .require_distribution_stability
        ):
            raise ValueError(
                "Distribution stability is required but disabled."
            )

        if self.options.selection.enabled:
            if selector_factory is None:
                raise ValueError(
                    "selector_factory is required when "
                    "selection stability is enabled."
                )

            selection_result = (
                SelectionStabilityEvaluator(
                    options=self.options.selection
                ).evaluate(
                    features=training_features.loc[
                        :,
                        selected,
                    ],
                    target=training_target,
                    selector_factory=selector_factory,
                    task=task,
                    target_access_authorized=(
                        target_access_authorized
                    ),
                    evaluated_features=selected,
                    required_features=list(
                        required
                    ),
                    protected_features=list(
                        protected
                    ),
                )
            )

        elif (
            self.options
            .require_selection_stability
        ):
            raise ValueError(
                "Selection stability is required but disabled."
            )

        decisions: dict[
            str,
            AggregatedStabilityDecision,
        ] = {}

        stable_features: list[str] = []
        unstable_features: list[str] = []
        blocking_features: list[str] = []
        removed_unstable_features: list[str] = []

        retained_features: list[str] = []

        for column in selected:
            distribution_feature = (
                distribution_result
                .features
                .get(
                    column
                )
                if distribution_result
                is not None
                else None
            )

            selection_feature = (
                selection_result
                .features
                .get(
                    column
                )
                if selection_result
                is not None
                else None
            )

            distribution_stable = bool(
                distribution_feature is None
                or not distribution_feature.unstable
            )

            selection_stable = bool(
                selection_feature is None
                or selection_feature.stable
            )

            reasons: list[str] = []

            if (
                distribution_feature is not None
                and distribution_feature.unstable
            ):
                reasons.extend(
                    [
                        (
                            "distribution: "
                            f"{reason}"
                        )
                        for reason
                        in distribution_feature
                        .instability_reasons
                    ]
                )

            if (
                selection_feature is not None
                and not selection_feature.stable
            ):
                reasons.extend(
                    [
                        (
                            "selection: "
                            f"{reason}"
                        )
                        for reason
                        in selection_feature
                        .instability_reasons
                    ]
                )

            stable = bool(
                distribution_stable
                and selection_stable
            )

            required_override = bool(
                not stable
                and column in required
            )

            protected_override = bool(
                not stable
                and column in protected
            )

            component_blocks = bool(
                (
                    distribution_feature is not None
                    and distribution_feature
                    .blocks_readiness
                )
                or (
                    selection_feature is not None
                    and selection_feature
                    .blocks_readiness
                )
            )

            blocks_readiness = bool(
                component_blocks
                and not required_override
                and not protected_override
            )

            remove_feature = bool(
                not stable
                and self.options
                .remove_unstable_features
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

            if remove_feature:
                removed_unstable_features.append(
                    column
                )

            else:
                retained_features.append(
                    column
                )

            decisions[
                column
            ] = AggregatedStabilityDecision(
                column=column,
                distribution_stable=(
                    distribution_stable
                ),
                selection_stable=(
                    selection_stable
                ),
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
                metadata={
                    "removed_for_instability": (
                        remove_feature
                    ),
                },
            )

        if (
            self.options.require_at_least_one_feature
            and not retained_features
        ):
            raise ValueError(
                "Stability filtering removed every feature."
            )

        valid = bool(
            not blocking_features
            and (
                distribution_result is None
                or distribution_result.valid
                or not self.options
                .require_distribution_stability
            )
            and (
                selection_result is None
                or selection_result.valid
                or not self.options
                .require_selection_stability
            )
        )

        self.result = StabilityEngineResult(
            valid=valid,
            distribution=distribution_result,
            selection=selection_result,
            decisions=decisions,
            stable_features=stable_features,
            unstable_features=unstable_features,
            blocking_features=blocking_features,
            retained_features=retained_features,
            removed_unstable_features=(
                removed_unstable_features
            ),
            fitted_on_training_only=True,
            validation_used_for_evaluation_only=True,
            target_access_authorized=(
                target_access_authorized
            ),
            metadata={
                "enabled": True,
                "selected_feature_count": len(
                    selected
                ),
                "stable_feature_count": len(
                    stable_features
                ),
                "unstable_feature_count": len(
                    unstable_features
                ),
                "retained_feature_count": len(
                    retained_features
                ),
                "removed_unstable_count": len(
                    removed_unstable_features
                ),
            },
        )

        return self

    def get_result(
        self,
    ) -> StabilityEngineResult:
        """Return a copy of the stability-engine result."""

        if self.result is None:
            raise RuntimeError(
                "The stability engine has not been evaluated."
            )

        return self.result.model_copy(
            deep=True
        )

    @staticmethod
    def _validate_features(
        *,
        training: pd.DataFrame,
        validation: pd.DataFrame,
        selected_features: list[str],
    ) -> None:
        """Validate stability-engine feature schemas."""

        if not isinstance(
            training,
            pd.DataFrame,
        ):
            raise TypeError(
                "training_features must be a pandas DataFrame."
            )

        if not isinstance(
            validation,
            pd.DataFrame,
        ):
            raise TypeError(
                "validation_features must be a pandas DataFrame."
            )

        if not selected_features:
            raise ValueError(
                "selected_features cannot be empty."
            )

        missing_training = [
            feature
            for feature in selected_features
            if feature not in training.columns
        ]

        missing_validation = [
            feature
            for feature in selected_features
            if feature not in validation.columns
        ]

        if missing_training:
            raise KeyError(
                "Training stability features are missing: "
                f"{missing_training}"
            )

        if missing_validation:
            raise KeyError(
                "Validation stability features are missing: "
                f"{missing_validation}"
            )

        duplicate_features = [
            feature
            for feature in selected_features
            if selected_features.count(
                feature
            )
            > 1
        ]

        if duplicate_features:
            raise ValueError(
                "selected_features contains duplicates: "
                f"{sorted(set(duplicate_features))}"
            )