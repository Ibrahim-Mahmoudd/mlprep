"""Unified unsupervised filter engine for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase7.contract.selection_plan import (
    SelectionPlan,
)
from mlprep.phases.phase7.filters.constant_filter import (
    ConstantFilter,
    ConstantFilterOptions,
    ConstantFilterResult,
)
from mlprep.phases.phase7.filters.correlation_filter import (
    CorrelationFilter,
    CorrelationFilterOptions,
    CorrelationFilterResult,
)
from mlprep.phases.phase7.filters.duplicate_filter import (
    DuplicateFilter,
    DuplicateFilterOptions,
    DuplicateFilterResult,
)
from mlprep.phases.phase7.filters.multicollinearity_filter import (
    MulticollinearityFilter,
    MulticollinearityFilterOptions,
    MulticollinearityFilterResult,
)
from mlprep.phases.phase7.filters.variance_filter import (
    VarianceFilter,
    VarianceFilterOptions,
    VarianceFilterResult,
)


class FeatureRemovalDecision(BaseModel):
    """Final unsupervised removal decision for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    removal_reasons: list[str] = Field(
        default_factory=list
    )

    removed: bool = False

    required_override: bool = False
    protected_override: bool = False

    restored: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class FilterEngineResult(BaseModel):
    """Complete unsupervised filter-engine output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    source_features: list[str] = Field(
        default_factory=list
    )

    retained_features: list[str] = Field(
        default_factory=list
    )

    removed_features: list[str] = Field(
        default_factory=list
    )

    decisions: dict[
        str,
        FeatureRemovalDecision,
    ] = Field(default_factory=dict)

    constant_filter: ConstantFilterResult | None = None
    variance_filter: VarianceFilterResult | None = None
    duplicate_filter: DuplicateFilterResult | None = None
    correlation_filter: CorrelationFilterResult | None = None
    multicollinearity_filter: (
        MulticollinearityFilterResult | None
    ) = None

    required_overrides: list[str] = Field(
        default_factory=list
    )

    protected_overrides: list[str] = Field(
        default_factory=list
    )

    prohibited_removals: list[str] = Field(
        default_factory=list
    )

    fitted_on_training_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class FilterEngineOptions:
    """Options controlling the filter engine."""

    apply_filters_sequentially: bool = True

    preserve_source_order: bool = True

    require_at_least_one_feature: bool = True

    constant: ConstantFilterOptions = field(
        default_factory=ConstantFilterOptions
    )

    variance: VarianceFilterOptions = field(
        default_factory=VarianceFilterOptions
    )

    duplicate: DuplicateFilterOptions = field(
        default_factory=DuplicateFilterOptions
    )

    correlation: CorrelationFilterOptions = field(
        default_factory=CorrelationFilterOptions
    )

    multicollinearity: (
        MulticollinearityFilterOptions
    ) = field(
        default_factory=(
            MulticollinearityFilterOptions
        )
    )


class FilterEngine:
    """Run all unsupervised feature filters."""

    component_name = "filter_engine"

    def __init__(
        self,
        options: FilterEngineOptions | None = None,
    ) -> None:
        """Initialize the filter engine."""

        self.options = (
            options
            or FilterEngineOptions()
        )

        self.result: FilterEngineResult | None = None

    def fit(
        self,
        *,
        frame: pd.DataFrame,
        plan: SelectionPlan,
        feature_scores: dict[str, float] | None = None,
    ) -> FilterEngine:
        """Fit filters using training features only."""

        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                "frame must be a pandas DataFrame."
            )

        source_features = list(
            plan.candidate_features
        )

        missing = [
            feature
            for feature in source_features
            if feature not in frame.columns
        ]

        if missing:
            raise KeyError(
                "Filter-engine features are missing: "
                f"{missing}"
            )

        required = list(
            plan.required_features
        )

        protected = list(
            plan.protected_features
        )

        active_features = list(
            source_features
        )

        decisions = {
            feature: FeatureRemovalDecision(
                column=feature
            )
            for feature in source_features
        }

        prohibited_removals = [
            feature
            for feature
            in plan.prohibited_features
            if feature in frame.columns
        ]

        for feature in prohibited_removals:
            if feature not in decisions:
                decisions[
                    feature
                ] = FeatureRemovalDecision(
                    column=feature
                )

            decisions[
                feature
            ].removal_reasons.append(
                "The feature is prohibited."
            )

            decisions[
                feature
            ].removed = True

        constant_result = None
        variance_result = None
        duplicate_result = None
        correlation_result = None
        multicollinearity_result = None

        if plan.constant_filter.enabled:
            constant_filter = ConstantFilter(
                options=ConstantFilterOptions(
                    enabled=True,
                    minimum_unique_count=(
                        plan
                        .constant_filter
                        .minimum_unique_count
                    ),
                    count_missing_as_category=(
                        self.options
                        .constant
                        .count_missing_as_category
                    ),
                )
            )

            constant_filter.fit(
                frame=frame,
                required_features=required,
                protected_features=protected,
            )

            constant_result = (
                constant_filter.get_result()
            )

            self._apply_removals(
                active_features=active_features,
                decisions=decisions,
                removal_candidates=(
                    constant_result
                    .removal_candidates
                ),
                reason="constant_feature",
            )

        variance_candidates = self._candidates(
            source_features=source_features,
            active_features=active_features,
        )

        if plan.variance_filter.enabled:
            variance_filter = VarianceFilter(
                options=VarianceFilterOptions(
                    enabled=True,
                    minimum_variance=(
                        plan
                        .variance_filter
                        .minimum_variance
                    ),
                    binary_minimum_variance=(
                        plan
                        .variance_filter
                        .binary_minimum_variance
                    ),
                    use_population_variance=(
                        plan
                        .variance_filter
                        .use_population_variance
                    ),
                    require_finite_values=(
                        self.options
                        .variance
                        .require_finite_values
                    ),
                    allow_missing_values=(
                        self.options
                        .variance
                        .allow_missing_values
                    ),
                )
            )

            variance_filter.fit(
                frame=frame,
                candidate_features=(
                    variance_candidates
                ),
                required_features=required,
                protected_features=protected,
            )

            variance_result = (
                variance_filter.get_result()
            )

            self._apply_removals(
                active_features=active_features,
                decisions=decisions,
                removal_candidates=(
                    variance_result
                    .removal_candidates
                ),
                reason="low_variance",
            )

        duplicate_candidates = self._candidates(
            source_features=source_features,
            active_features=active_features,
        )

        if plan.duplicate_filter.enabled:
            duplicate_filter = DuplicateFilter(
                options=DuplicateFilterOptions(
                    enabled=True,
                    verify_dtypes=(
                        plan
                        .duplicate_filter
                        .verify_dtypes
                    ),
                    verify_exact_values=(
                        self.options
                        .duplicate
                        .verify_exact_values
                    ),
                )
            )

            duplicate_filter.fit(
                frame=frame,
                candidate_features=(
                    duplicate_candidates
                ),
                required_features=required,
                protected_features=protected,
                source_feature_order=(
                    plan.source_feature_order
                ),
            )

            duplicate_result = (
                duplicate_filter.get_result()
            )

            self._apply_removals(
                active_features=active_features,
                decisions=decisions,
                removal_candidates=(
                    duplicate_result
                    .removal_candidates
                ),
                reason="duplicate_feature",
            )

        correlation_candidates = self._candidates(
            source_features=source_features,
            active_features=active_features,
        )

        variance_scores = (
            variance_result.variance_scores
            if variance_result is not None
            else {}
        )

        if plan.correlation_filter.enabled:
            correlation_filter = (
                CorrelationFilter(
                    options=CorrelationFilterOptions(
                        enabled=True,
                        maximum_absolute_correlation=(
                            plan
                            .correlation_filter
                            .maximum_absolute_correlation
                        ),
                        method=(
                            plan
                            .correlation_filter
                            .method
                        ),
                        minimum_periods=(
                            plan
                            .correlation_filter
                            .minimum_periods
                        ),
                        include_correlation_matrix=(
                            self.options
                            .correlation
                            .include_correlation_matrix
                        ),
                    )
                )
            )

            correlation_filter.fit(
                frame=frame,
                candidate_features=(
                    correlation_candidates
                ),
                required_features=required,
                protected_features=protected,
                feature_scores=feature_scores,
                variance_scores=variance_scores,
                source_feature_order=(
                    plan.source_feature_order
                ),
            )

            correlation_result = (
                correlation_filter.get_result()
            )

            self._apply_removals(
                active_features=active_features,
                decisions=decisions,
                removal_candidates=(
                    correlation_result
                    .removal_candidates
                ),
                reason="high_correlation",
            )

        multicollinearity_candidates = self._candidates(
            source_features=source_features,
            active_features=active_features,
        )

        if plan.multicollinearity_filter.enabled:
            multicollinearity_filter = (
                MulticollinearityFilter(
                    options=(
                        MulticollinearityFilterOptions(
                            enabled=True,
                            maximum_vif=(
                                plan
                                .multicollinearity_filter
                                .maximum_vif
                            ),
                            maximum_condition_number=(
                                plan
                                .multicollinearity_filter
                                .maximum_condition_number
                            ),
                            maximum_iterations=(
                                plan
                                .multicollinearity_filter
                                .maximum_iterations
                            ),
                            numerical_tolerance=(
                                plan
                                .multicollinearity_filter
                                .numerical_tolerance
                            ),
                            add_intercept=(
                                self.options
                                .multicollinearity
                                .add_intercept
                            ),
                        )
                    )
                )
            )

            multicollinearity_filter.fit(
                frame=frame,
                candidate_features=(
                    multicollinearity_candidates
                ),
                required_features=required,
                protected_features=protected,
                source_feature_order=(
                    plan.source_feature_order
                ),
            )

            multicollinearity_result = (
                multicollinearity_filter
                .get_result()
            )

            self._apply_removals(
                active_features=active_features,
                decisions=decisions,
                removal_candidates=(
                    multicollinearity_result
                    .removed_features
                ),
                reason="excessive_vif",
            )

        required_overrides = self._restore_features(
            active_features=active_features,
            decisions=decisions,
            features=required,
            source_order=(
                plan.source_feature_order
            ),
            override_type="required",
        )

        protected_overrides = self._restore_features(
            active_features=active_features,
            decisions=decisions,
            features=protected,
            source_order=(
                plan.source_feature_order
            ),
            override_type="protected",
        )

        active_features = [
            feature
            for feature
            in plan.source_feature_order
            if feature in set(
                active_features
            )
        ]

        if (
            self.options.require_at_least_one_feature
            and not active_features
        ):
            raise ValueError(
                "Unsupervised filtering removed every feature."
            )

        removed_features = [
            feature
            for feature in source_features
            if feature not in set(
                active_features
            )
        ]

        self.result = FilterEngineResult(
            source_features=source_features,
            retained_features=active_features,
            removed_features=removed_features,
            decisions=decisions,
            constant_filter=constant_result,
            variance_filter=variance_result,
            duplicate_filter=duplicate_result,
            correlation_filter=correlation_result,
            multicollinearity_filter=(
                multicollinearity_result
            ),
            required_overrides=(
                required_overrides
            ),
            protected_overrides=(
                protected_overrides
            ),
            prohibited_removals=(
                prohibited_removals
            ),
            fitted_on_training_only=True,
            metadata={
                "source_feature_count": len(
                    source_features
                ),
                "retained_feature_count": len(
                    active_features
                ),
                "removed_feature_count": len(
                    removed_features
                ),
                "sequential": (
                    self.options
                    .apply_filters_sequentially
                ),
            },
        )

        return self

    def get_result(
        self,
    ) -> FilterEngineResult:
        """Return the fitted engine result."""

        if self.result is None:
            raise RuntimeError(
                "The filter engine is not fitted."
            )

        return self.result.model_copy(
            deep=True
        )

    def _candidates(
        self,
        *,
        source_features: list[str],
        active_features: list[str],
    ) -> list[object]:
        """Resolve candidates for the next filter."""

        if self.options.apply_filters_sequentially:
            return list(
                active_features
            )

        return list(
            source_features
        )

    @staticmethod
    def _apply_removals(
        *,
        active_features: list[str],
        decisions: dict[
            str,
            FeatureRemovalDecision,
        ],
        removal_candidates: list[str],
        reason: str,
    ) -> None:
        """Apply filter removal candidates."""

        for feature in removal_candidates:
            if feature not in decisions:
                decisions[
                    feature
                ] = FeatureRemovalDecision(
                    column=feature
                )

            decision = decisions[
                feature
            ]

            if reason not in (
                decision.removal_reasons
            ):
                decision.removal_reasons.append(
                    reason
                )

            decision.removed = True

            if feature in active_features:
                active_features.remove(
                    feature
                )

    @staticmethod
    def _restore_features(
        *,
        active_features: list[str],
        decisions: dict[
            str,
            FeatureRemovalDecision,
        ],
        features: list[str],
        source_order: list[str],
        override_type: str,
    ) -> list[object]:
        """Restore required or protected features."""

        restored: list[str] = []

        for feature in features:
            if feature not in decisions:
                continue

            decision = decisions[
                feature
            ]

            if feature not in active_features:
                active_features.append(
                    feature
                )

                decision.restored = True

                restored.append(
                    feature
                )

            decision.removed = False

            if override_type == "required":
                decision.required_override = True

            elif override_type == "protected":
                decision.protected_override = True

        active_set = set(
            active_features
        )

        active_features[:] = [
            feature
            for feature in source_order
            if feature in active_set
        ]

        return restored
