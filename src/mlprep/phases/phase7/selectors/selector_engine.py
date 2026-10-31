"""Unified supervised selector engine for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase7.contract.selection_plan import (
    SelectionComponentStatus,
    SelectionPlan,
    SelectorAggregationKind,
)
from mlprep.phases.phase7.selectors.base import (
    SelectorFitContext,
    SelectorState,
)
from mlprep.phases.phase7.selectors.model_importance_selector import (
    ModelImportanceSelector,
)
from mlprep.phases.phase7.selectors.mutual_information_selector import (
    MutualInformationSelector,
)
from mlprep.phases.phase7.selectors.recursive_selector import (
    RecursiveSelector,
)
from mlprep.phases.phase7.selectors.univariate_selector import (
    UnivariateSelector,
)


class AggregatedFeatureScore(BaseModel):
    """Aggregated selector decision for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    vote_count: int = Field(
        default=0,
        ge=0,
    )

    weighted_score: float = 0.0

    normalized_score: float = 0.0

    selected: bool = False

    selected_by: list[str] = Field(
        default_factory=list
    )

    required_override: bool = False
    protected_override: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class SelectorEngineResult(BaseModel):
    """Complete supervised selector-engine output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    source_features: list[str] = Field(
        default_factory=list
    )

    selected_features: list[str] = Field(
        default_factory=list
    )

    rejected_features: list[str] = Field(
        default_factory=list
    )

    selector_states: dict[
        str,
        SelectorState,
    ] = Field(default_factory=dict)

    aggregated_scores: dict[
        str,
        AggregatedFeatureScore,
    ] = Field(default_factory=dict)

    aggregation: str

    target_access_used: bool = False
    target_access_authorized: bool = False

    fitted_on_training_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class SelectorEngineOptions:
    """Options controlling selector aggregation."""

    require_nonempty_selection: bool = True

    weighted_score_threshold: float = 0.0

    normalize_selector_scores: bool = True


class SelectorEngine:
    """Fit and aggregate authorized supervised selectors."""

    component_name = "selector_engine"

    def __init__(
        self,
        options: SelectorEngineOptions | None = None,
    ) -> None:
        """Initialize the selector engine."""

        self.options = (
            options
            or SelectorEngineOptions()
        )

        self.result: SelectorEngineResult | None = None

    def fit(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        plan: SelectionPlan,
    ) -> SelectorEngine:
        """Fit enabled selectors on training data."""

        source_features = [
            str(column)
            for column in features.columns
        ]

        enabled_selectors = (
            plan.enabled_selectors()
        )

        if (
            enabled_selectors
            and not plan
            .supervised_selection_authorized
        ):
            raise PermissionError(
                "The selection plan contains enabled "
                "supervised selectors without authorization."
            )

        states: dict[
            str,
            SelectorState,
        ] = {}

        for selector_name in enabled_selectors:
            selector = self._build_selector(
                selector_name=selector_name,
                plan=plan,
            )

            selector.fit(
                features=features,
                target=target,
                context=(
                    SelectorFitContext
                    .TRAINING_SUPERVISED
                ),
                target_access_authorized=(
                    plan
                    .supervised_selection_authorized
                ),
            )

            states[
                selector_name
            ] = selector.get_state()

        if states:
            (
                selected,
                aggregated_scores,
            ) = self._aggregate(
                states=states,
                plan=plan,
                source_features=source_features,
            )

        else:
            selected = list(
                source_features
            )

            aggregated_scores = {
                column: AggregatedFeatureScore(
                    column=column,
                    selected=True,
                    metadata={
                        "reason": (
                            "No supervised selector was enabled."
                        ),
                    },
                )
                for column in source_features
            }

        selected = self._restore_features(
            selected=selected,
            source_features=source_features,
            required_features=(
                plan.required_features
            ),
            protected_features=(
                plan.protected_features
            ),
            aggregated_scores=(
                aggregated_scores
            ),
        )

        if (
            self.options.require_nonempty_selection
            and not selected
        ):
            raise ValueError(
                "Supervised selection rejected every feature."
            )

        selected_set = set(
            selected
        )

        self.result = SelectorEngineResult(
            source_features=source_features,
            selected_features=selected,
            rejected_features=[
                column
                for column in source_features
                if column not in selected_set
            ],
            selector_states=states,
            aggregated_scores=(
                aggregated_scores
            ),
            aggregation=(
                plan.aggregation.value
            ),
            target_access_used=bool(
                states
            ),
            target_access_authorized=(
                plan
                .supervised_selection_authorized
            ),
            fitted_on_training_only=True,
            metadata={
                "enabled_selector_count": len(
                    states
                ),
                "source_feature_count": len(
                    source_features
                ),
                "selected_feature_count": len(
                    selected
                ),
            },
        )

        return self

    def get_result(
        self,
    ) -> SelectorEngineResult:
        """Return fitted selector-engine output."""

        if self.result is None:
            raise RuntimeError(
                "The selector engine is not fitted."
            )

        return self.result.model_copy(
            deep=True
        )

    def _build_selector(
        self,
        *,
        selector_name: str,
        plan: SelectionPlan,
    ) -> object:
        """Build one selector from the plan."""

        common = {
            "task": plan.task.value,
            "required_features": (
                plan.required_features
            ),
            "protected_features": (
                plan.protected_features
            ),
        }

        if selector_name == "univariate_selector":
            specification = (
                plan.univariate_selector
            )

            return UnivariateSelector(
                method=specification.method,
                top_k=specification.top_k,
                minimum_score=(
                    specification.minimum_score
                ),
                **common,
            )

        if (
            selector_name
            == "mutual_information_selector"
        ):
            specification = (
                plan
                .mutual_information_selector
            )

            return MutualInformationSelector(
                top_k=specification.top_k,
                minimum_score=(
                    specification.minimum_score
                ),
                discrete_features=(
                    specification
                    .discrete_features
                ),
                random_seed=(
                    specification.random_seed
                ),
                **common,
            )

        if (
            selector_name
            == "model_importance_selector"
        ):
            specification = (
                plan.model_importance_selector
            )

            return ModelImportanceSelector(
                estimator=(
                    specification.estimator
                ),
                number_of_estimators=(
                    specification
                    .number_of_estimators
                ),
                maximum_depth=(
                    specification.maximum_depth
                ),
                top_k=specification.top_k,
                minimum_score=(
                    specification
                    .minimum_importance
                ),
                random_seed=(
                    specification.random_seed
                ),
                number_of_jobs=(
                    specification.number_of_jobs
                ),
                **common,
            )

        if selector_name == "recursive_selector":
            specification = (
                plan.recursive_selector
            )

            return RecursiveSelector(
                estimator=(
                    specification.estimator
                ),
                target_feature_count=(
                    specification
                    .target_feature_count
                ),
                minimum_feature_count=(
                    specification
                    .minimum_feature_count
                ),
                removal_step=(
                    specification.removal_step
                ),
                maximum_iterations=(
                    specification
                    .maximum_iterations
                ),
                random_seed=(
                    specification.random_seed
                ),
                **common,
            )

        raise ValueError(
            f"Unsupported selector: {selector_name!r}"
        )

    def _aggregate(
        self,
        *,
        states: dict[
            str,
            SelectorState,
        ],
        plan: SelectionPlan,
        source_features: list[str],
    ) -> tuple[
        list[str],
        dict[str, AggregatedFeatureScore],
    ]:
        """Aggregate selector decisions."""

        aggregated: dict[
            str,
            AggregatedFeatureScore,
        ] = {}

        for column in source_features:
            selected_by: list[str] = []
            vote_count = 0
            weighted_score = 0.0
            score_weight_total = 0.0

            for selector_name, state in states.items():
                weight = float(
                    plan
                    .selector_weights
                    .get(
                        selector_name,
                        1.0,
                    )
                )

                feature_score = (
                    state.feature_scores[
                        column
                    ]
                )

                normalized_score = (
                    self._normalized_score(
                        feature_score.score,
                        state=state,
                    )
                    if self.options
                    .normalize_selector_scores
                    else feature_score.score
                )

                weighted_score += (
                    normalized_score
                    * weight
                )

                score_weight_total += weight

                if feature_score.selected:
                    vote_count += 1

                    selected_by.append(
                        selector_name
                    )

            combined_score = (
                weighted_score
                / score_weight_total
                if score_weight_total > 0.0
                else 0.0
            )

            selected = self._aggregated_selection(
                column=column,
                selected_by=selected_by,
                vote_count=vote_count,
                combined_score=combined_score,
                states=states,
                plan=plan,
            )

            aggregated[
                column
            ] = AggregatedFeatureScore(
                column=column,
                vote_count=vote_count,
                weighted_score=(
                    weighted_score
                ),
                normalized_score=(
                    combined_score
                ),
                selected=selected,
                selected_by=selected_by,
            )

        selected_features = [
            column
            for column in source_features
            if aggregated[
                column
            ].selected
        ]

        return (
            selected_features,
            aggregated,
        )

    def _aggregated_selection(
        self,
        *,
        column: str,
        selected_by: list[str],
        vote_count: int,
        combined_score: float,
        states: dict[
            str,
            SelectorState,
        ],
        plan: SelectionPlan,
    ) -> bool:
        """Apply the configured aggregation method."""

        selector_count = len(
            states
        )

        if (
            plan.aggregation
            == SelectorAggregationKind.UNION
        ):
            return vote_count > 0

        if (
            plan.aggregation
            == SelectorAggregationKind.INTERSECTION
        ):
            return (
                selector_count > 0
                and vote_count
                == selector_count
            )

        if (
            plan.aggregation
            == SelectorAggregationKind.VOTING
        ):
            return (
                vote_count
                >= plan.minimum_votes
            )

        if (
            plan.aggregation
            == SelectorAggregationKind.WEIGHTED_SCORE
        ):
            return (
                combined_score
                >= self.options
                .weighted_score_threshold
            )

        for selector_name in (
            "recursive_selector",
            "model_importance_selector",
            "mutual_information_selector",
            "univariate_selector",
        ):
            if selector_name in states:
                return (
                    column
                    in states[
                        selector_name
                    ].selected_features
                )

        return bool(
            selected_by
        )

    @staticmethod
    def _normalized_score(
        score: float,
        *,
        state: SelectorState,
    ) -> float:
        """Normalize one score within a selector."""

        scores = [
            item.score
            for item
            in state.feature_scores.values()
        ]

        minimum = min(
            scores
        )

        maximum = max(
            scores
        )

        if maximum <= minimum:
            return (
                1.0
                if score > 0.0
                else 0.0
            )

        return (
            score - minimum
        ) / (
            maximum - minimum
        )

    @staticmethod
    def _restore_features(
        *,
        selected: list[str],
        source_features: list[str],
        required_features: list[str],
        protected_features: list[str],
        aggregated_scores: dict[
            str,
            AggregatedFeatureScore,
        ],
    ) -> list[object]:
        """Restore required and protected features."""

        selected_set = set(
            selected
        )

        for column in required_features:
            if column not in source_features:
                continue

            selected_set.add(
                column
            )

            aggregated_scores[
                column
            ].selected = True

            aggregated_scores[
                column
            ].required_override = True

        for column in protected_features:
            if column not in source_features:
                continue

            selected_set.add(
                column
            )

            aggregated_scores[
                column
            ].selected = True

            aggregated_scores[
                column
            ].protected_override = True

        return [
            column
            for column in source_features
            if column in selected_set
        ]