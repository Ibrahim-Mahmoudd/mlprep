"""Unified fitted feature-selection pipeline for Phase 7."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase6.feature_engineering.feature_lineage import (
    FeatureLineageEntry,
    FeatureLineageState,
)
from mlprep.phases.phase7.contract.selection_contract import (
    SelectionContract,
)
from mlprep.phases.phase7.contract.selection_plan import (
    SelectionPlan,
)
from mlprep.phases.phase7.filters.filter_engine import (
    FilterEngine,
    FilterEngineOptions,
    FilterEngineResult,
)
from mlprep.phases.phase7.orchestration.fit_context import (
    SelectionFitContext,
)
from mlprep.phases.phase7.orchestration.partition_transformer import (
    PartitionSelectionMetadata,
    PartitionTransformer,
    PartitionTransformerOptions,
)
from mlprep.phases.phase7.selectors.selector_engine import (
    SelectorEngine,
    SelectorEngineOptions,
    SelectorEngineResult,
)
from mlprep.phases.phase7.stability.stability_engine import (
    StabilityEngine,
    StabilityEngineOptions,
    StabilityEngineResult,
)


SelectorFactory = Callable[
    [],
    object,
]


class FeatureSelectionDecision(BaseModel):
    """Final Phase 7 decision for one source feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    selected: bool = False

    selected_by_filter_engine: bool = False
    selected_by_selector_engine: bool = False
    selected_by_stability_engine: bool = False

    required: bool = False
    protected: bool = False
    prohibited: bool = False

    restored_for_minimum_count: bool = False
    removed_for_maximum_count: bool = False

    selection_score: float = 0.0

    selection_reasons: list[str] = Field(
        default_factory=list,
    )

    removal_reasons: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class PartitionSelectionResult(BaseModel):
    """Selection result for one transformed partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    partition: PartitionName

    selection: PartitionSelectionMetadata

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SelectionPipelineResult(BaseModel):
    """Complete fitted Phase 7 pipeline output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    partitions: dict[
        PartitionName,
        PartitionSelectionResult,
    ] = Field(
        default_factory=dict,
    )

    source_features: list[str] = Field(
        default_factory=list,
    )

    selected_features: list[str] = Field(
        default_factory=list,
    )

    removed_features: list[str] = Field(
        default_factory=list,
    )

    decisions: dict[
        str,
        FeatureSelectionDecision,
    ] = Field(
        default_factory=dict,
    )

    filter_result: FilterEngineResult

    selector_result: SelectorEngineResult

    stability_result: StabilityEngineResult | None = None

    selected_lineage: FeatureLineageState | None = None

    fitted_on_training_only: bool = True

    supervised_target_access_used: bool = False
    supervised_target_access_authorized: bool = False

    validation_used_for_evaluation_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class SelectionPipelineOptions:
    """Options controlling the Phase 7 selection pipeline."""

    filters: FilterEngineOptions = field(
        default_factory=FilterEngineOptions,
    )

    selectors: SelectorEngineOptions = field(
        default_factory=SelectorEngineOptions,
    )

    stability: StabilityEngineOptions = field(
        default_factory=StabilityEngineOptions,
    )

    partition_transformer: PartitionTransformerOptions = field(
        default_factory=PartitionTransformerOptions,
    )

    test_transform_authorized: bool = True

    combine_filter_and_selector_results: str = "intersection"

    require_nonempty_selection: bool = True

    preserve_source_order: bool = True

    enforce_minimum_feature_count: bool = True
    enforce_maximum_feature_count: bool = True

    use_selector_scores_for_count_policy: bool = True

    def __post_init__(self) -> None:
        """Validate selection-pipeline options."""

        supported_combination_policies = {
            "intersection",
            "union",
            "filters_then_selectors",
            "selectors_then_filters",
        }

        if (
            self.combine_filter_and_selector_results
            not in supported_combination_policies
        ):
            raise ValueError(
                "combine_filter_and_selector_results must be "
                "'intersection', 'union', "
                "'filters_then_selectors', or "
                "'selectors_then_filters'."
            )

        boolean_options = {
            "test_transform_authorized": (
                self.test_transform_authorized
            ),
            "require_nonempty_selection": (
                self.require_nonempty_selection
            ),
            "preserve_source_order": (
                self.preserve_source_order
            ),
            "enforce_minimum_feature_count": (
                self.enforce_minimum_feature_count
            ),
            "enforce_maximum_feature_count": (
                self.enforce_maximum_feature_count
            ),
            "use_selector_scores_for_count_policy": (
                self.use_selector_scores_for_count_policy
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


class SelectionPipeline:
    """Fit feature selection on training and apply its schema."""

    component_name = "selection_pipeline"

    def __init__(
        self,
        options: SelectionPipelineOptions | None = None,
    ) -> None:
        """Initialize the selection pipeline."""

        self.options = (
            options
            or SelectionPipelineOptions()
        )

        self.filter_engine = FilterEngine(
            options=self.options.filters,
        )

        self.selector_engine = SelectorEngine(
            options=self.options.selectors,
        )

        self.stability_engine = StabilityEngine(
            options=self.options.stability,
        )

        self.partition_transformer = PartitionTransformer(
            options=(
                self.options.partition_transformer
            ),
        )

        self.plan: SelectionPlan | None = None

        self.filter_result: FilterEngineResult | None = None
        self.selector_result: SelectorEngineResult | None = None
        self.stability_result: StabilityEngineResult | None = None

        self.source_features: list[str] = []
        self.selected_features: list[str] = []

        self.decisions: dict[
            str,
            FeatureSelectionDecision,
        ] = {}

        self.selected_lineage: FeatureLineageState | None = None

        self._fitted = False

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the pipeline is fitted."""

        return self._fitted

    def fit(
        self,
        *,
        contract: SelectionContract,
        plan: SelectionPlan,
        selector_factory: SelectorFactory | None = None,
    ) -> SelectionPipeline:
        """Fit all selection state on training data."""

        if self._fitted:
            raise RuntimeError(
                "The feature-selection pipeline is already fitted."
            )

        if not isinstance(
            contract,
            SelectionContract,
        ):
            raise TypeError(
                "contract must be a SelectionContract."
            )

        if not isinstance(
            plan,
            SelectionPlan,
        ):
            raise TypeError(
                "plan must be a SelectionPlan."
            )

        context = SelectionFitContext.training_fit(
            supervised_selection_authorized=(
                plan.supervised_selection_authorized
            ),
        )

        supervised_required = bool(
            plan.enabled_selectors()
        )

        context.validate_for_fit(
            supervised_selection_required=(
                supervised_required
            ),
        )

        training_features = contract.train.X

        source_features = list(
            plan.source_feature_order
        )

        if source_features != contract.feature_columns:
            raise ValueError(
                "The plan source schema differs from the "
                "Phase 7 contract schema."
            )

        prohibited_features = set(
            plan.prohibited_features
        )

        supervised_candidate_features = [
            column
            for column in plan.source_feature_order
            if column not in prohibited_features
        ]

        if not supervised_candidate_features:
            raise ValueError(
                "No eligible features remain after excluding "
                "prohibited features."
            )

        missing_supervised_candidates = [
            column
            for column in supervised_candidate_features
            if column not in training_features.columns
        ]

        if missing_supervised_candidates:
            raise KeyError(
                "Supervised selector candidates are unavailable: "
                f"{missing_supervised_candidates}"
            )

        self.selector_engine.fit(
            features=training_features.loc[
                :,
                supervised_candidate_features,
            ],
            target=contract.train.y,
            plan=plan,
        )

        selector_result = (
            self.selector_engine.get_result()
        )

        selector_scores = {
            column: score.normalized_score
            for column, score
            in selector_result
            .aggregated_scores
            .items()
        }

        self.filter_engine.fit(
            frame=training_features,
            plan=plan,
            feature_scores=selector_scores,
        )

        filter_result = (
            self.filter_engine.get_result()
        )

        initial_selected = self._combine_results(
            plan=plan,
            filter_result=filter_result,
            selector_result=selector_result,
        )

        initial_selected = (
            self._restore_required_and_protected(
                selected=initial_selected,
                source_order=source_features,
                required_features=(
                    plan.required_features
                ),
                protected_features=(
                    plan.protected_features
                ),
            )
        )

        initial_selected = [
            column
            for column in initial_selected
            if column not in prohibited_features
        ]

        initial_selected = self._enforce_feature_counts(
            selected=initial_selected,
            source_order=source_features,
            plan=plan,
            selector_scores=selector_scores,
            filter_result=filter_result,
        )

        if (
            self.options.require_nonempty_selection
            and not initial_selected
        ):
            raise ValueError(
                "Feature selection produced an empty schema."
            )

        stability_result = None

        if self.options.stability.enabled:
            stability_factory = selector_factory

            if (
                self.options.stability.selection.enabled
                and stability_factory is None
            ):
                stability_factory = (
                    self._default_stability_factory(
                        plan=plan,
                    )
                )

            self.stability_engine.evaluate(
                training_features=(
                    contract.train.X
                ),
                validation_features=(
                    contract.validation.X
                ),
                training_target=(
                    contract.train.y
                ),
                selected_features=(
                    initial_selected
                ),
                task=plan.task.value,
                target_access_authorized=(
                    plan.supervised_selection_authorized
                ),
                selector_factory=(
                    stability_factory
                ),
                required_features=(
                    plan.required_features
                ),
                protected_features=(
                    plan.protected_features
                ),
            )

            stability_result = (
                self.stability_engine.get_result()
            )

            final_selected = list(
                stability_result.retained_features
            )

        else:
            final_selected = list(
                initial_selected
            )

        final_selected = (
            self._restore_required_and_protected(
                selected=final_selected,
                source_order=source_features,
                required_features=(
                    plan.required_features
                ),
                protected_features=(
                    plan.protected_features
                ),
            )
        )

        final_selected = [
            column
            for column in final_selected
            if column not in prohibited_features
        ]

        final_selected = self._enforce_feature_counts(
            selected=final_selected,
            source_order=source_features,
            plan=plan,
            selector_scores=selector_scores,
            filter_result=filter_result,
        )

        if (
            self.options.require_nonempty_selection
            and not final_selected
        ):
            raise ValueError(
                "The final selected feature schema is empty."
            )

        self.source_features = list(
            source_features
        )

        self.selected_features = list(
            final_selected
        )

        self.filter_result = filter_result
        self.selector_result = selector_result
        self.stability_result = stability_result

        self.decisions = self._build_decisions(
            plan=plan,
            filter_result=filter_result,
            selector_result=selector_result,
            stability_result=stability_result,
            selected_features=(
                final_selected
            ),
        )

        self._register_count_policy_decisions(
            decisions=self.decisions,
            initially_selected=(
                initial_selected
            ),
            finally_selected=(
                final_selected
            ),
            minimum_feature_count=(
                plan.minimum_feature_count
            ),
            maximum_feature_count=(
                plan.maximum_feature_count
            ),
        )

        self.selected_lineage = self._build_selected_lineage(
            source_lineage=(
                contract.phase6_state.lineage
            ),
            selected_features=(
                final_selected
            ),
        )

        self.plan = plan
        self._fitted = True

        return self

    def fit_transform(
        self,
        *,
        contract: SelectionContract,
        plan: SelectionPlan,
        selector_factory: SelectorFactory | None = None,
    ) -> tuple[
        DatasetPartition,
        DatasetPartition,
        DatasetPartition,
        SelectionPipelineResult,
    ]:
        """Fit on training and apply the selected schema."""

        self.fit(
            contract=contract,
            plan=plan,
            selector_factory=selector_factory,
        )

        transformed_train, training_result = (
            self._transform_partition(
                partition=contract.train,
                context=(
                    SelectionFitContext
                    .training_transform()
                ),
            )
        )

        (
            transformed_validation,
            validation_result,
        ) = self._transform_partition(
            partition=contract.validation,
            context=(
                SelectionFitContext
                .validation_transform()
            ),
        )

        transformed_test, test_result = (
            self._transform_partition(
                partition=contract.test,
                context=(
                    SelectionFitContext
                    .test_transform(
                        authorized=(
                            self.options
                            .test_transform_authorized
                        ),
                    )
                ),
            )
        )

        if self.filter_result is None:
            raise RuntimeError(
                "The fitted filter result is unavailable."
            )

        if self.selector_result is None:
            raise RuntimeError(
                "The fitted selector result is unavailable."
            )

        selected_set = set(
            self.selected_features
        )

        pipeline_result = SelectionPipelineResult(
            partitions={
                PartitionName.TRAIN: (
                    training_result
                ),
                PartitionName.VALIDATION: (
                    validation_result
                ),
                PartitionName.TEST: (
                    test_result
                ),
            },
            source_features=list(
                self.source_features
            ),
            selected_features=list(
                self.selected_features
            ),
            removed_features=[
                column
                for column in self.source_features
                if column not in selected_set
            ],
            decisions={
                column: decision.model_copy(
                    deep=True
                )
                for column, decision
                in self.decisions.items()
            },
            filter_result=(
                self.filter_result.model_copy(
                    deep=True
                )
            ),
            selector_result=(
                self.selector_result.model_copy(
                    deep=True
                )
            ),
            stability_result=(
                self.stability_result.model_copy(
                    deep=True
                )
                if self.stability_result is not None
                else None
            ),
            selected_lineage=(
                self.selected_lineage.model_copy(
                    deep=True
                )
                if self.selected_lineage is not None
                else None
            ),
            fitted_on_training_only=True,
            supervised_target_access_used=(
                self.selector_result
                .target_access_used
            ),
            supervised_target_access_authorized=(
                self.selector_result
                .target_access_authorized
            ),
            validation_used_for_evaluation_only=True,
            metadata={
                "source_feature_count": len(
                    self.source_features
                ),
                "selected_feature_count": len(
                    self.selected_features
                ),
                "removed_feature_count": (
                    len(
                        self.source_features
                    )
                    - len(
                        self.selected_features
                    )
                ),
                "combination_policy": (
                    self.options
                    .combine_filter_and_selector_results
                ),
            },
        )

        return (
            transformed_train,
            transformed_validation,
            transformed_test,
            pipeline_result,
        )

    def transform_partition(
        self,
        *,
        partition: DatasetPartition,
        context: SelectionFitContext,
    ) -> tuple[
        DatasetPartition,
        PartitionSelectionResult,
    ]:
        """Apply fitted selection to another partition."""

        if not self._fitted:
            raise RuntimeError(
                "The feature-selection pipeline must be fitted "
                "before transforming another partition."
            )

        return self._transform_partition(
            partition=partition,
            context=context,
        )

    def get_selected_features(
        self,
    ) -> list[object]:
        """Return the fitted selected feature schema."""

        if not self._fitted:
            raise RuntimeError(
                "The selection pipeline is not fitted."
            )

        return list(
            self.selected_features
        )

    def get_decisions(
        self,
    ) -> dict[
        str,
        FeatureSelectionDecision,
    ]:
        """Return fitted feature-selection decisions."""

        if not self._fitted:
            raise RuntimeError(
                "The selection pipeline is not fitted."
            )

        return {
            column: decision.model_copy(
                deep=True
            )
            for column, decision
            in self.decisions.items()
        }

    def _transform_partition(
        self,
        *,
        partition: DatasetPartition,
        context: SelectionFitContext,
    ) -> tuple[
        DatasetPartition,
        PartitionSelectionResult,
    ]:
        """Apply the fitted selected schema."""

        if not self._fitted:
            raise RuntimeError(
                "The feature-selection pipeline is not fitted."
            )

        if self.plan is None:
            raise RuntimeError(
                "The fitted selection plan is unavailable."
            )

        selection_reasons = {
            column: list(
                decision.selection_reasons
            )
            for column, decision
            in self.decisions.items()
        }

        transformed, metadata = (
            self.partition_transformer.transform(
                partition=partition,
                context=context,
                source_feature_order=(
                    self.source_features
                ),
                selected_features=(
                    self.selected_features
                ),
                required_features=(
                    self.plan.required_features
                ),
                protected_features=(
                    self.plan.protected_features
                ),
                selection_reasons=(
                    selection_reasons
                ),
            )
        )

        result = PartitionSelectionResult(
            partition=partition.name,
            selection=metadata,
            metadata={
                "context_operation": (
                    context.operation.value
                ),
                "training_data": (
                    context.training_data
                ),
                "test_authorized": (
                    context.test_partition_authorized
                ),
            },
        )

        return (
            transformed,
            result,
        )

    def _combine_results(
        self,
        *,
        plan: SelectionPlan,
        filter_result: FilterEngineResult,
        selector_result: SelectorEngineResult,
    ) -> list[object]:
        """Combine filter and selector decisions."""

        filter_selected = set(
            filter_result.retained_features
        )

        selector_selected = set(
            selector_result.selected_features
        )

        method = (
            self.options
            .combine_filter_and_selector_results
        )

        if method == "intersection":
            selected_set = (
                filter_selected.intersection(
                    selector_selected
                )
            )

        elif method == "union":
            selected_set = (
                filter_selected.union(
                    selector_selected
                )
            )

        elif method == "filters_then_selectors":
            selected_set = {
                column
                for column
                in selector_result.selected_features
                if column in filter_selected
            }

        else:
            selected_set = {
                column
                for column
                in filter_result.retained_features
                if column in selector_selected
            }

        return [
            column
            for column in plan.source_feature_order
            if column in selected_set
        ]

    def _enforce_feature_counts(
        self,
        *,
        selected: list[str],
        source_order: list[str],
        plan: SelectionPlan,
        selector_scores: dict[str, float],
        filter_result: FilterEngineResult,
    ) -> list[object]:
        """Enforce minimum and maximum feature counts."""

        selected_set = set(
            selected
        )

        prohibited = set(
            plan.prohibited_features
        )

        required_or_protected = {
            *plan.required_features,
            *plan.protected_features,
        }

        selected_set.difference_update(
            prohibited
        )

        selected_set.update(
            feature
            for feature
            in required_or_protected
            if feature not in prohibited
        )

        if (
            self.options
            .enforce_minimum_feature_count
            and len(selected_set)
            < plan.minimum_feature_count
        ):
            candidates = [
                column
                for column in source_order
                if (
                    column not in selected_set
                    and column not in prohibited
                )
            ]

            if (
                self.options
                .use_selector_scores_for_count_policy
            ):
                candidates.sort(
                    key=lambda column: (
                        -selector_scores.get(
                            column,
                            0.0,
                        ),
                        0
                        if column
                        in filter_result.retained_features
                        else 1,
                        source_order.index(
                            column
                        ),
                    )
                )

            else:
                candidates.sort(
                    key=source_order.index
                )

            needed = (
                plan.minimum_feature_count
                - len(
                    selected_set
                )
            )

            selected_set.update(
                candidates[
                    :needed
                ]
            )

        if (
            self.options
            .enforce_minimum_feature_count
            and len(selected_set)
            < plan.minimum_feature_count
        ):
            raise ValueError(
                "The minimum feature count cannot be reached "
                "without selecting prohibited features."
            )

        if (
            self.options
            .enforce_maximum_feature_count
            and plan.maximum_feature_count
            is not None
            and len(selected_set)
            > plan.maximum_feature_count
        ):
            mandatory = {
                column
                for column in selected_set
                if column in required_or_protected
            }

            if (
                len(mandatory)
                > plan.maximum_feature_count
            ):
                raise ValueError(
                    "The number of required and protected "
                    "features exceeds maximum_feature_count."
                )

            optional = [
                column
                for column in source_order
                if (
                    column in selected_set
                    and column not in mandatory
                )
            ]

            if (
                self.options
                .use_selector_scores_for_count_policy
            ):
                optional.sort(
                    key=lambda column: (
                        -selector_scores.get(
                            column,
                            0.0,
                        ),
                        0
                        if column
                        in filter_result.retained_features
                        else 1,
                        source_order.index(
                            column
                        ),
                    )
                )

            else:
                optional.sort(
                    key=source_order.index
                )

            optional_limit = (
                plan.maximum_feature_count
                - len(
                    mandatory
                )
            )

            selected_set = (
                mandatory
                | set(
                    optional[
                        :optional_limit
                    ]
                )
            )

        result = [
            column
            for column in source_order
            if column in selected_set
        ]

        if (
            self.options.require_nonempty_selection
            and not result
        ):
            raise ValueError(
                "Feature-count enforcement produced "
                "an empty selection."
            )

        return result

    @staticmethod
    def _restore_required_and_protected(
        *,
        selected: list[str],
        source_order: list[str],
        required_features: list[str],
        protected_features: list[str],
    ) -> list[object]:
        """Restore mandatory features."""

        selected_set = set(
            selected
        )

        selected_set.update(
            required_features
        )

        selected_set.update(
            protected_features
        )

        return [
            column
            for column in source_order
            if column in selected_set
        ]

    def _build_decisions(
        self,
        *,
        plan: SelectionPlan,
        filter_result: FilterEngineResult,
        selector_result: SelectorEngineResult,
        stability_result: StabilityEngineResult | None,
        selected_features: list[str],
    ) -> dict[
        str,
        FeatureSelectionDecision,
    ]:
        """Build final per-feature selection decisions."""

        selected_set = set(
            selected_features
        )

        filter_selected = set(
            filter_result.retained_features
        )

        selector_selected = set(
            selector_result.selected_features
        )

        stability_selected = (
            set(
                stability_result.retained_features
            )
            if stability_result is not None
            else set(
                selected_features
            )
        )

        required = set(
            plan.required_features
        )

        protected = set(
            plan.protected_features
        )

        prohibited = set(
            plan.prohibited_features
        )

        decisions: dict[
            str,
            FeatureSelectionDecision,
        ] = {}

        for column in plan.source_feature_order:
            filter_decision = (
                filter_result.decisions.get(
                    column
                )
            )

            selector_score = (
                selector_result
                .aggregated_scores
                .get(
                    column
                )
            )

            stability_decision = (
                stability_result
                .decisions
                .get(
                    column
                )
                if stability_result is not None
                else None
            )

            selection_reasons: list[str] = []
            removal_reasons: list[str] = []

            if column in filter_selected:
                selection_reasons.append(
                    "retained_by_filter_engine"
                )

            else:
                removal_reasons.append(
                    "rejected_by_filter_engine"
                )

            if column in selector_selected:
                selection_reasons.append(
                    "selected_by_selector_engine"
                )

            else:
                removal_reasons.append(
                    "rejected_by_selector_engine"
                )

            if column in stability_selected:
                selection_reasons.append(
                    "retained_by_stability_engine"
                )

            elif stability_result is not None:
                removal_reasons.append(
                    "rejected_by_stability_engine"
                )

            if column in required:
                selection_reasons.append(
                    "required_feature"
                )

            if column in protected:
                selection_reasons.append(
                    "protected_feature"
                )

            if column in prohibited:
                removal_reasons.append(
                    "prohibited_feature"
                )

            if filter_decision is not None:
                for reason in (
                    filter_decision.removal_reasons
                ):
                    if reason not in removal_reasons:
                        removal_reasons.append(
                            reason
                        )

            if stability_decision is not None:
                for reason in (
                    stability_decision
                    .instability_reasons
                ):
                    if reason not in removal_reasons:
                        removal_reasons.append(
                            reason
                        )

            if column in selected_set:
                if "final_selection" not in selection_reasons:
                    selection_reasons.append(
                        "final_selection"
                    )

            else:
                if "not_in_final_selection" not in removal_reasons:
                    removal_reasons.append(
                        "not_in_final_selection"
                    )

            decisions[
                column
            ] = FeatureSelectionDecision(
                column=column,
                selected=(
                    column in selected_set
                ),
                selected_by_filter_engine=(
                    column in filter_selected
                ),
                selected_by_selector_engine=(
                    column in selector_selected
                ),
                selected_by_stability_engine=(
                    column in stability_selected
                ),
                required=(
                    column in required
                ),
                protected=(
                    column in protected
                ),
                prohibited=(
                    column in prohibited
                ),
                selection_score=(
                    selector_score.normalized_score
                    if selector_score is not None
                    else 0.0
                ),
                selection_reasons=(
                    selection_reasons
                ),
                removal_reasons=(
                    removal_reasons
                ),
            )

        return decisions

    @staticmethod
    def _register_count_policy_decisions(
        *,
        decisions: dict[
            str,
            FeatureSelectionDecision,
        ],
        initially_selected: list[str],
        finally_selected: list[str],
        minimum_feature_count: int,
        maximum_feature_count: int | None,
    ) -> None:
        """Register minimum and maximum count-policy effects."""

        initial_set = set(
            initially_selected
        )

        final_set = set(
            finally_selected
        )

        for column, decision in decisions.items():
            if (
                column in final_set
                and column not in initial_set
            ):
                decision.restored_for_minimum_count = True

                if (
                    "restored_for_feature_count_policy"
                    not in decision.selection_reasons
                ):
                    decision.selection_reasons.append(
                        "restored_for_feature_count_policy"
                    )

            if (
                column in initial_set
                and column not in final_set
                and maximum_feature_count is not None
            ):
                decision.removed_for_maximum_count = True

                if (
                    "removed_for_maximum_feature_count"
                    not in decision.removal_reasons
                ):
                    decision.removal_reasons.append(
                        "removed_for_maximum_feature_count"
                    )

            decision.metadata.update(
                {
                    "minimum_feature_count": (
                        minimum_feature_count
                    ),
                    "maximum_feature_count": (
                        maximum_feature_count
                    ),
                }
            )

    @staticmethod
    def _build_selected_lineage(
        *,
        source_lineage: FeatureLineageState | None,
        selected_features: list[str],
    ) -> FeatureLineageState | None:
        """Restrict Phase 6 lineage to selected features."""

        if source_lineage is None:
            return None

        selected_set = set(
            selected_features
        )

        entries: dict[
            str,
            FeatureLineageEntry,
        ] = {}

        output_to_sources: dict[
            str,
            list[str],
        ] = {}

        source_to_outputs: dict[
            str,
            list[str],
        ] = {}

        for output_column in selected_features:
            entry = source_lineage.entries.get(
                output_column
            )

            if entry is None:
                raise KeyError(
                    "Phase 6 lineage is unavailable for "
                    f"selected feature {output_column!r}."
                )

            entries[
                output_column
            ] = entry.model_copy(
                deep=True
            )

            sources = list(
                entry.source_columns
            )

            output_to_sources[
                output_column
            ] = sources

            for source in sources:
                source_outputs = (
                    source_to_outputs.setdefault(
                        source,
                        [],
                    )
                )

                if output_column not in source_outputs:
                    source_outputs.append(
                        output_column
                    )

        output_columns = [
            column
            for column in source_lineage.output_columns
            if column in selected_set
        ]

        if output_columns != selected_features:
            raise ValueError(
                "Selected lineage output order differs from "
                "the selected feature schema."
            )

        return FeatureLineageState(
            entries=entries,
            source_to_outputs=(
                source_to_outputs
            ),
            output_to_sources=(
                output_to_sources
            ),
            source_columns=[
                source
                for source
                in source_lineage.source_columns
                if source in source_to_outputs
            ],
            output_columns=(
                output_columns
            ),
            metadata={
                **dict(
                    source_lineage.metadata
                ),
                "phase7_selected_feature_count": (
                    len(
                        selected_features
                    )
                ),
            },
        )

    def _default_stability_factory(
        self,
        *,
        plan: SelectionPlan,
    ) -> SelectorFactory:
        """Build a selector factory for bootstrap stability."""

        enabled_selectors = (
            plan.enabled_selectors()
        )

        if not enabled_selectors:
            raise ValueError(
                "Selection stability requires at least one "
                "enabled supervised selector."
            )

        primary_selector_name = (
            enabled_selectors[
                0
            ]
        )

        def factory() -> object:
            """Build a fresh selector instance."""

            return self.selector_engine.build_selector(
                selector_name=(
                    primary_selector_name
                ),
                plan=plan,
            )

        return factory