"""Executable feature-selection plan for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from mlprep.phases.phase7.contract.selection_contract import (
    SelectionContract,
)


class LearningTask(StrEnum):
    """Supported supervised learning tasks."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class SelectorAggregationKind(StrEnum):
    """Supported selector-aggregation policies."""

    UNION = "union"
    INTERSECTION = "intersection"
    VOTING = "voting"
    WEIGHTED_SCORE = "weighted_score"
    PRIORITY = "priority"


class UnivariateMethod(StrEnum):
    """Supported univariate selection methods."""

    ANOVA = "anova"
    CHI_SQUARE = "chi_square"
    F_REGRESSION = "f_regression"
    CORRELATION = "correlation"


class ModelImportanceEstimator(StrEnum):
    """Supported feature-importance estimators."""

    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"


class RecursiveEstimator(StrEnum):
    """Supported recursive selector estimators."""

    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"


class SelectionComponentStatus(StrEnum):
    """Status of one selection-plan component."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    BLOCKED = "blocked"


class ConstantFilterPlan(BaseModel):
    """Constant-filter plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    enabled: bool = True

    minimum_unique_count: int = Field(
        default=2,
        ge=1,
    )

    status: SelectionComponentStatus = (
        SelectionComponentStatus.ENABLED
    )


class VarianceFilterPlan(BaseModel):
    """Low-variance filter plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    enabled: bool = True

    minimum_variance: float = Field(
        default=0.0001,
        ge=0.0,
    )

    binary_minimum_variance: float = Field(
        default=0.0,
        ge=0.0,
    )

    use_population_variance: bool = True

    status: SelectionComponentStatus = (
        SelectionComponentStatus.ENABLED
    )


class DuplicateFilterPlan(BaseModel):
    """Duplicate-feature filter plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    enabled: bool = True

    verify_dtypes: bool = False

    status: SelectionComponentStatus = (
        SelectionComponentStatus.ENABLED
    )


class CorrelationFilterPlan(BaseModel):
    """High-correlation filter plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    enabled: bool = True

    maximum_absolute_correlation: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
    )

    method: str = "pearson"

    minimum_periods: int = Field(
        default=2,
        ge=2,
    )

    status: SelectionComponentStatus = (
        SelectionComponentStatus.ENABLED
    )

    @model_validator(mode="after")
    def validate_method(
        self,
    ) -> CorrelationFilterPlan:
        """Validate correlation method."""

        if self.method not in {
            "pearson",
            "spearman",
            "kendall",
        }:
            raise ValueError(
                "Correlation method must be 'pearson', "
                "'spearman', or 'kendall'."
            )

        return self


class MulticollinearityFilterPlan(BaseModel):
    """Multicollinearity-filter plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    enabled: bool = False

    maximum_vif: float = Field(
        default=10.0,
        gt=1.0,
    )

    maximum_condition_number: float | None = Field(
        default=None,
        gt=1.0,
    )

    maximum_iterations: int = Field(
        default=100,
        ge=1,
    )

    numerical_tolerance: float = Field(
        default=1e-12,
        gt=0.0,
    )

    status: SelectionComponentStatus = (
        SelectionComponentStatus.DISABLED
    )


class UnivariateSelectorPlan(BaseModel):
    """Univariate selector plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    enabled: bool = False

    method: UnivariateMethod = (
        UnivariateMethod.ANOVA
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
    )

    minimum_score: float | None = None

    requires_target_access: bool = True

    status: SelectionComponentStatus = (
        SelectionComponentStatus.DISABLED
    )


class MutualInformationSelectorPlan(BaseModel):
    """Mutual-information selector plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    enabled: bool = False

    top_k: int | None = Field(
        default=None,
        ge=1,
    )

    minimum_score: float | None = Field(
        default=None,
        ge=0.0,
    )

    discrete_features: str = "auto"

    random_seed: int = 42

    requires_target_access: bool = True

    status: SelectionComponentStatus = (
        SelectionComponentStatus.DISABLED
    )

    @model_validator(mode="after")
    def validate_discrete_features(
        self,
    ) -> MutualInformationSelectorPlan:
        """Validate discrete-feature policy."""

        if self.discrete_features not in {
            "auto",
            "all",
            "none",
        }:
            raise ValueError(
                "discrete_features must be 'auto', "
                "'all', or 'none'."
            )

        return self


class ModelImportanceSelectorPlan(BaseModel):
    """Model-importance selector plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    enabled: bool = False

    estimator: ModelImportanceEstimator = (
        ModelImportanceEstimator.RANDOM_FOREST
    )

    number_of_estimators: int = Field(
        default=200,
        ge=1,
    )

    maximum_depth: int | None = Field(
        default=None,
        ge=1,
    )

    minimum_importance: float | None = Field(
        default=None,
        ge=0.0,
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
    )

    random_seed: int = 42

    number_of_jobs: int = 1

    requires_target_access: bool = True

    status: SelectionComponentStatus = (
        SelectionComponentStatus.DISABLED
    )


class RecursiveSelectorPlan(BaseModel):
    """Recursive feature-selection plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    enabled: bool = False

    estimator: RecursiveEstimator = (
        RecursiveEstimator.LINEAR
    )

    target_feature_count: int | None = Field(
        default=None,
        ge=1,
    )

    minimum_feature_count: int = Field(
        default=1,
        ge=1,
    )

    removal_step: int = Field(
        default=1,
        ge=1,
    )

    maximum_iterations: int = Field(
        default=100,
        ge=1,
    )

    random_seed: int = 42

    requires_target_access: bool = True

    status: SelectionComponentStatus = (
        SelectionComponentStatus.DISABLED
    )

    @model_validator(mode="after")
    def validate_feature_counts(
        self,
    ) -> RecursiveSelectorPlan:
        """Validate recursive feature-count settings."""

        if (
            self.target_feature_count is not None
            and self.target_feature_count
            < self.minimum_feature_count
        ):
            raise ValueError(
                "target_feature_count cannot be smaller "
                "than minimum_feature_count."
            )

        return self


class SelectionPlan(BaseModel):
    """Complete executable Phase 7 selection plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    task: LearningTask

    constant_filter: ConstantFilterPlan
    variance_filter: VarianceFilterPlan
    duplicate_filter: DuplicateFilterPlan
    correlation_filter: CorrelationFilterPlan
    multicollinearity_filter: (
        MulticollinearityFilterPlan
    )

    univariate_selector: UnivariateSelectorPlan
    mutual_information_selector: (
        MutualInformationSelectorPlan
    )
    model_importance_selector: (
        ModelImportanceSelectorPlan
    )
    recursive_selector: RecursiveSelectorPlan

    aggregation: SelectorAggregationKind

    minimum_votes: int = Field(
        default=1,
        ge=1,
    )

    selector_weights: dict[
        str,
        float,
    ] = Field(default_factory=dict)

    required_features: list[str] = Field(
        default_factory=list
    )

    protected_features: list[str] = Field(
        default_factory=list
    )

    prohibited_features: list[str] = Field(
        default_factory=list
    )

    candidate_features: list[str] = Field(
        default_factory=list
    )

    minimum_feature_count: int = Field(
        default=1,
        ge=1,
    )

    maximum_feature_count: int | None = Field(
        default=None,
        ge=1,
    )

    supervised_selection_authorized: bool = False

    source_feature_order: list[str] = Field(
        default_factory=list
    )

    blocked_components: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @property
    def has_blocked_components(
        self,
    ) -> bool:
        """Return whether blocked components exist."""

        return bool(
            self.blocked_components
        )

    def enabled_filters(
        self,
    ) -> list[object]:
        """Return enabled filter names."""

        components = {
            "constant_filter": (
                self.constant_filter
            ),
            "variance_filter": (
                self.variance_filter
            ),
            "duplicate_filter": (
                self.duplicate_filter
            ),
            "correlation_filter": (
                self.correlation_filter
            ),
            "multicollinearity_filter": (
                self.multicollinearity_filter
            ),
        }

        return [
            name
            for name, component
            in components.items()
            if (
                component.enabled
                and component.status
                == SelectionComponentStatus.ENABLED
            )
        ]

    def enabled_selectors(
        self,
    ) -> list[object]:
        """Return enabled selector names."""

        components = {
            "univariate_selector": (
                self.univariate_selector
            ),
            "mutual_information_selector": (
                self.mutual_information_selector
            ),
            "model_importance_selector": (
                self.model_importance_selector
            ),
            "recursive_selector": (
                self.recursive_selector
            ),
        }

        return [
            name
            for name, component
            in components.items()
            if (
                component.enabled
                and component.status
                == SelectionComponentStatus.ENABLED
            )
        ]

    def requires_target_access(
        self,
    ) -> bool:
        """Return whether any enabled selector uses the target."""

        components = [
            self.univariate_selector,
            self.mutual_information_selector,
            self.model_importance_selector,
            self.recursive_selector,
        ]

        return any(
            component.enabled
            and component.status
            == SelectionComponentStatus.ENABLED
            and component.requires_target_access
            for component in components
        )


@dataclass(slots=True)
class SelectionPlanOptions:
    """Options controlling Phase 7 plan construction."""

    enabled: bool = True

    task: LearningTask = (
        LearningTask.CLASSIFICATION
    )

    constant_filter_enabled: bool = True

    variance_filter_enabled: bool = True
    minimum_variance: float = 0.0001
    binary_minimum_variance: float = 0.0

    duplicate_filter_enabled: bool = True

    correlation_filter_enabled: bool = True
    maximum_absolute_correlation: float = 0.95
    correlation_method: str = "pearson"

    multicollinearity_filter_enabled: bool = False
    maximum_vif: float = 10.0
    maximum_condition_number: float | None = None
    multicollinearity_maximum_iterations: int = 100

    univariate_selector_enabled: bool = False
    univariate_method: UnivariateMethod | None = None
    univariate_top_k: int | None = None
    univariate_minimum_score: float | None = None

    mutual_information_enabled: bool = False
    mutual_information_top_k: int | None = None
    mutual_information_minimum_score: float | None = None
    mutual_information_discrete_features: str = "auto"

    model_importance_enabled: bool = False
    model_importance_estimator: (
        ModelImportanceEstimator
    ) = ModelImportanceEstimator.RANDOM_FOREST
    model_importance_estimators: int = 200
    model_importance_maximum_depth: int | None = None
    model_importance_minimum_score: float | None = None
    model_importance_top_k: int | None = None

    recursive_selector_enabled: bool = False
    recursive_estimator: RecursiveEstimator = (
        RecursiveEstimator.LINEAR
    )
    recursive_target_feature_count: int | None = None
    recursive_minimum_feature_count: int = 1
    recursive_removal_step: int = 1
    recursive_maximum_iterations: int = 100

    aggregation: SelectorAggregationKind = (
        SelectorAggregationKind.UNION
    )

    minimum_votes: int = 1

    selector_weights: dict[
        str,
        float,
    ] = field(default_factory=dict)

    required_features: list[str] = field(
        default_factory=list
    )

    protected_features: list[str] = field(
        default_factory=list
    )

    prohibited_features: list[str] = field(
        default_factory=list
    )

    minimum_feature_count: int = 1
    maximum_feature_count: int | None = None

    supervised_selection_authorized: bool = False

    random_seed: int = 42
    number_of_jobs: int = 1

    def __post_init__(self) -> None:
        """Validate plan options."""

        if not isinstance(
            self.task,
            LearningTask,
        ):
            self.task = LearningTask(
                self.task
            )

        if not isinstance(
            self.aggregation,
            SelectorAggregationKind,
        ):
            self.aggregation = (
                SelectorAggregationKind(
                    self.aggregation
                )
            )

        if not isinstance(
            self.model_importance_estimator,
            ModelImportanceEstimator,
        ):
            self.model_importance_estimator = (
                ModelImportanceEstimator(
                    self
                    .model_importance_estimator
                )
            )

        if not isinstance(
            self.recursive_estimator,
            RecursiveEstimator,
        ):
            self.recursive_estimator = (
                RecursiveEstimator(
                    self.recursive_estimator
                )
            )

        if (
            self.univariate_method is not None
            and not isinstance(
                self.univariate_method,
                UnivariateMethod,
            )
        ):
            self.univariate_method = (
                UnivariateMethod(
                    self.univariate_method
                )
            )

        if self.minimum_variance < 0.0:
            raise ValueError(
                "minimum_variance cannot be negative."
            )

        if self.binary_minimum_variance < 0.0:
            raise ValueError(
                "binary_minimum_variance cannot be negative."
            )

        if not 0.0 <= (
            self.maximum_absolute_correlation
        ) <= 1.0:
            raise ValueError(
                "maximum_absolute_correlation must be "
                "between zero and one."
            )

        if self.maximum_vif <= 1.0:
            raise ValueError(
                "maximum_vif must exceed one."
            )

        if self.minimum_feature_count < 1:
            raise ValueError(
                "minimum_feature_count must be positive."
            )

        if (
            self.maximum_feature_count is not None
            and self.maximum_feature_count
            < self.minimum_feature_count
        ):
            raise ValueError(
                "maximum_feature_count cannot be smaller "
                "than minimum_feature_count."
            )

        if self.minimum_votes < 1:
            raise ValueError(
                "minimum_votes must be positive."
            )

        if self.number_of_jobs == 0:
            raise ValueError(
                "number_of_jobs cannot be zero."
            )

        for selector, weight in (
            self.selector_weights.items()
        ):
            if weight < 0.0:
                raise ValueError(
                    f"Selector weight for {selector!r} "
                    "cannot be negative."
                )

        self.required_features = self._normalize_features(
            self.required_features,
            "required_features",
        )

        self.protected_features = self._normalize_features(
            self.protected_features,
            "protected_features",
        )

        self.prohibited_features = self._normalize_features(
            self.prohibited_features,
            "prohibited_features",
        )

        prohibited = set(
            self.prohibited_features
        )

        required_conflicts = set(
            self.required_features
        ).intersection(
            prohibited
        )

        protected_conflicts = set(
            self.protected_features
        ).intersection(
            prohibited
        )

        if required_conflicts:
            raise ValueError(
                "Required features cannot be prohibited: "
                f"{sorted(required_conflicts)}"
            )

        if protected_conflicts:
            raise ValueError(
                "Protected features cannot be prohibited: "
                f"{sorted(protected_conflicts)}"
            )

    @staticmethod
    def _normalize_features(
        features: list[str],
        name: str,
    ) -> list[object]:
        """Normalize a feature list."""

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


class SelectionPlanBuilder:
    """Build an executable Phase 7 selection plan."""

    component_name = "selection_plan"

    def __init__(
        self,
        options: SelectionPlanOptions | None = None,
    ) -> None:
        """Initialize the plan builder."""

        self.options = (
            options
            or SelectionPlanOptions()
        )

    def build(
        self,
        *,
        contract: SelectionContract,
    ) -> SelectionPlan:
        """Build the complete selection plan."""

        source_features = list(
            contract.feature_columns
        )

        if not self.options.enabled:
            return SelectionPlan(
                task=self.options.task,
                constant_filter=ConstantFilterPlan(
                    enabled=False,
                    status=(
                        SelectionComponentStatus
                        .DISABLED
                    ),
                ),
                variance_filter=VarianceFilterPlan(
                    enabled=False,
                    status=(
                        SelectionComponentStatus
                        .DISABLED
                    ),
                ),
                duplicate_filter=DuplicateFilterPlan(
                    enabled=False,
                    status=(
                        SelectionComponentStatus
                        .DISABLED
                    ),
                ),
                correlation_filter=CorrelationFilterPlan(
                    enabled=False,
                    status=(
                        SelectionComponentStatus
                        .DISABLED
                    ),
                ),
                multicollinearity_filter=(
                    MulticollinearityFilterPlan(
                        enabled=False,
                        status=(
                            SelectionComponentStatus
                            .DISABLED
                        ),
                    )
                ),
                univariate_selector=(
                    UnivariateSelectorPlan(
                        enabled=False,
                        status=(
                            SelectionComponentStatus
                            .DISABLED
                        ),
                    )
                ),
                mutual_information_selector=(
                    MutualInformationSelectorPlan(
                        enabled=False,
                        status=(
                            SelectionComponentStatus
                            .DISABLED
                        ),
                    )
                ),
                model_importance_selector=(
                    ModelImportanceSelectorPlan(
                        enabled=False,
                        status=(
                            SelectionComponentStatus
                            .DISABLED
                        ),
                    )
                ),
                recursive_selector=(
                    RecursiveSelectorPlan(
                        enabled=False,
                        status=(
                            SelectionComponentStatus
                            .DISABLED
                        ),
                    )
                ),
                aggregation=self.options.aggregation,
                required_features=(
                    self._merged_required_features(
                        contract
                    )
                ),
                protected_features=(
                    self._merged_protected_features(
                        contract
                    )
                ),
                prohibited_features=(
                    self._merged_prohibited_features(
                        contract
                    )
                ),
                candidate_features=source_features,
                minimum_feature_count=(
                    self.options
                    .minimum_feature_count
                ),
                maximum_feature_count=(
                    self.options
                    .maximum_feature_count
                ),
                supervised_selection_authorized=(
                    self.options
                    .supervised_selection_authorized
                ),
                source_feature_order=source_features,
                metadata={
                    "enabled": False,
                },
            )

        required_features = (
            self._merged_required_features(
                contract
            )
        )

        protected_features = (
            self._merged_protected_features(
                contract
            )
        )

        prohibited_features = (
            self._merged_prohibited_features(
                contract
            )
        )

        prohibited_set = set(
            prohibited_features
        )

        candidate_features = [
            feature
            for feature in source_features
            if feature not in prohibited_set
        ]

        if (
            self.options.minimum_feature_count
            > len(candidate_features)
        ):
            raise ValueError(
                "minimum_feature_count exceeds the number "
                "of eligible candidate features."
            )

        constant_filter = ConstantFilterPlan(
            enabled=(
                self.options
                .constant_filter_enabled
            ),
            status=self._status(
                self.options
                .constant_filter_enabled
            ),
        )

        variance_filter = VarianceFilterPlan(
            enabled=(
                self.options
                .variance_filter_enabled
            ),
            minimum_variance=(
                self.options.minimum_variance
            ),
            binary_minimum_variance=(
                self.options
                .binary_minimum_variance
            ),
            status=self._status(
                self.options
                .variance_filter_enabled
            ),
        )

        duplicate_filter = DuplicateFilterPlan(
            enabled=(
                self.options
                .duplicate_filter_enabled
            ),
            status=self._status(
                self.options
                .duplicate_filter_enabled
            ),
        )

        correlation_filter = CorrelationFilterPlan(
            enabled=(
                self.options
                .correlation_filter_enabled
            ),
            maximum_absolute_correlation=(
                self.options
                .maximum_absolute_correlation
            ),
            method=(
                self.options
                .correlation_method
            ),
            status=self._status(
                self.options
                .correlation_filter_enabled
            ),
        )

        multicollinearity_filter = (
            MulticollinearityFilterPlan(
                enabled=(
                    self.options
                    .multicollinearity_filter_enabled
                ),
                maximum_vif=(
                    self.options.maximum_vif
                ),
                maximum_condition_number=(
                    self.options
                    .maximum_condition_number
                ),
                maximum_iterations=(
                    self.options
                    .multicollinearity_maximum_iterations
                ),
                status=self._status(
                    self.options
                    .multicollinearity_filter_enabled
                ),
            )
        )

        blocked_components: list[str] = []

        univariate_selector = (
            self._build_univariate_selector(
                blocked_components
            )
        )

        mutual_information_selector = (
            self._build_mutual_information_selector(
                blocked_components
            )
        )

        model_importance_selector = (
            self._build_model_importance_selector(
                blocked_components
            )
        )

        recursive_selector = (
            self._build_recursive_selector(
                blocked_components
            )
        )

        enabled_selector_count = sum(
            component.enabled
            and component.status
            == SelectionComponentStatus.ENABLED
            for component in (
                univariate_selector,
                mutual_information_selector,
                model_importance_selector,
                recursive_selector,
            )
        )

        if (
            self.options.aggregation
            == SelectorAggregationKind.VOTING
            and enabled_selector_count > 0
            and self.options.minimum_votes
            > enabled_selector_count
        ):
            raise ValueError(
                "minimum_votes exceeds the number of "
                "enabled selectors."
            )

        return SelectionPlan(
            task=self.options.task,
            constant_filter=constant_filter,
            variance_filter=variance_filter,
            duplicate_filter=duplicate_filter,
            correlation_filter=correlation_filter,
            multicollinearity_filter=(
                multicollinearity_filter
            ),
            univariate_selector=(
                univariate_selector
            ),
            mutual_information_selector=(
                mutual_information_selector
            ),
            model_importance_selector=(
                model_importance_selector
            ),
            recursive_selector=(
                recursive_selector
            ),
            aggregation=self.options.aggregation,
            minimum_votes=(
                self.options.minimum_votes
            ),
            selector_weights=dict(
                self.options.selector_weights
            ),
            required_features=(
                required_features
            ),
            protected_features=(
                protected_features
            ),
            prohibited_features=(
                prohibited_features
            ),
            candidate_features=(
                candidate_features
            ),
            minimum_feature_count=(
                self.options.minimum_feature_count
            ),
            maximum_feature_count=(
                self.options.maximum_feature_count
            ),
            supervised_selection_authorized=(
                self.options
                .supervised_selection_authorized
            ),
            source_feature_order=(
                source_features
            ),
            blocked_components=(
                blocked_components
            ),
            metadata={
                "enabled": True,
                "source_feature_count": len(
                    source_features
                ),
                "candidate_feature_count": len(
                    candidate_features
                ),
                "required_feature_count": len(
                    required_features
                ),
                "protected_feature_count": len(
                    protected_features
                ),
                "prohibited_feature_count": len(
                    prohibited_features
                ),
                "enabled_filter_count": sum(
                    (
                        constant_filter.enabled,
                        variance_filter.enabled,
                        duplicate_filter.enabled,
                        correlation_filter.enabled,
                        multicollinearity_filter.enabled,
                    )
                ),
                "enabled_selector_count": (
                    enabled_selector_count
                ),
            },
        )

    def _build_univariate_selector(
        self,
        blocked_components: list[str],
    ) -> UnivariateSelectorPlan:
        """Build the univariate selector plan."""

        enabled = (
            self.options
            .univariate_selector_enabled
        )

        if not enabled:
            return UnivariateSelectorPlan(
                enabled=False,
                status=(
                    SelectionComponentStatus
                    .DISABLED
                ),
            )

        method = (
            self.options.univariate_method
            or self._default_univariate_method()
        )

        status = self._supervised_status(
            enabled=True,
            component="univariate_selector",
            blocked_components=(
                blocked_components
            ),
        )

        return UnivariateSelectorPlan(
            enabled=True,
            method=method,
            top_k=(
                self.options
                .univariate_top_k
            ),
            minimum_score=(
                self.options
                .univariate_minimum_score
            ),
            requires_target_access=True,
            status=status,
        )

    def _build_mutual_information_selector(
        self,
        blocked_components: list[str],
    ) -> MutualInformationSelectorPlan:
        """Build the mutual-information selector plan."""

        enabled = (
            self.options
            .mutual_information_enabled
        )

        if not enabled:
            return MutualInformationSelectorPlan(
                enabled=False,
                status=(
                    SelectionComponentStatus
                    .DISABLED
                ),
            )

        status = self._supervised_status(
            enabled=True,
            component=(
                "mutual_information_selector"
            ),
            blocked_components=(
                blocked_components
            ),
        )

        return MutualInformationSelectorPlan(
            enabled=True,
            top_k=(
                self.options
                .mutual_information_top_k
            ),
            minimum_score=(
                self.options
                .mutual_information_minimum_score
            ),
            discrete_features=(
                self.options
                .mutual_information_discrete_features
            ),
            random_seed=(
                self.options.random_seed
            ),
            requires_target_access=True,
            status=status,
        )

    def _build_model_importance_selector(
        self,
        blocked_components: list[str],
    ) -> ModelImportanceSelectorPlan:
        """Build the model-importance selector plan."""

        enabled = (
            self.options
            .model_importance_enabled
        )

        if not enabled:
            return ModelImportanceSelectorPlan(
                enabled=False,
                status=(
                    SelectionComponentStatus
                    .DISABLED
                ),
            )

        status = self._supervised_status(
            enabled=True,
            component=(
                "model_importance_selector"
            ),
            blocked_components=(
                blocked_components
            ),
        )

        return ModelImportanceSelectorPlan(
            enabled=True,
            estimator=(
                self.options
                .model_importance_estimator
            ),
            number_of_estimators=(
                self.options
                .model_importance_estimators
            ),
            maximum_depth=(
                self.options
                .model_importance_maximum_depth
            ),
            minimum_importance=(
                self.options
                .model_importance_minimum_score
            ),
            top_k=(
                self.options
                .model_importance_top_k
            ),
            random_seed=(
                self.options.random_seed
            ),
            number_of_jobs=(
                self.options.number_of_jobs
            ),
            requires_target_access=True,
            status=status,
        )

    def _build_recursive_selector(
        self,
        blocked_components: list[str],
    ) -> RecursiveSelectorPlan:
        """Build the recursive selector plan."""

        enabled = (
            self.options
            .recursive_selector_enabled
        )

        if not enabled:
            return RecursiveSelectorPlan(
                enabled=False,
                status=(
                    SelectionComponentStatus
                    .DISABLED
                ),
            )

        status = self._supervised_status(
            enabled=True,
            component="recursive_selector",
            blocked_components=(
                blocked_components
            ),
        )

        return RecursiveSelectorPlan(
            enabled=True,
            estimator=(
                self.options
                .recursive_estimator
            ),
            target_feature_count=(
                self.options
                .recursive_target_feature_count
            ),
            minimum_feature_count=(
                self.options
                .recursive_minimum_feature_count
            ),
            removal_step=(
                self.options
                .recursive_removal_step
            ),
            maximum_iterations=(
                self.options
                .recursive_maximum_iterations
            ),
            random_seed=(
                self.options.random_seed
            ),
            requires_target_access=True,
            status=status,
        )

    def _supervised_status(
        self,
        *,
        enabled: bool,
        component: str,
        blocked_components: list[str],
    ) -> SelectionComponentStatus:
        """Resolve a supervised component status."""

        if not enabled:
            return (
                SelectionComponentStatus.DISABLED
            )

        if (
            not self.options
            .supervised_selection_authorized
        ):
            blocked_components.append(
                component
            )

            return (
                SelectionComponentStatus.BLOCKED
            )

        return SelectionComponentStatus.ENABLED

    def _default_univariate_method(
        self,
    ) -> UnivariateMethod:
        """Resolve the task-aware univariate method."""

        if (
            self.options.task
            == LearningTask.CLASSIFICATION
        ):
            return UnivariateMethod.ANOVA

        return UnivariateMethod.F_REGRESSION

    def _merged_required_features(
        self,
        contract: SelectionContract,
    ) -> list[object]:
        """Merge required feature policies."""

        configured = {
            *contract.required_features,
            *self.options.required_features,
        }

        return [
            feature
            for feature
            in contract.feature_columns
            if feature in configured
        ]

    def _merged_protected_features(
        self,
        contract: SelectionContract,
    ) -> list[object]:
        """Merge protected feature policies."""

        configured = {
            *contract.protected_features,
            *self.options.protected_features,
        }

        return [
            feature
            for feature
            in contract.feature_columns
            if feature in configured
        ]

    def _merged_prohibited_features(
        self,
        contract: SelectionContract,
    ) -> list[object]:
        """Merge prohibited feature policies."""

        configured = {
            *contract.prohibited_features,
            *self.options.prohibited_features,
        }

        return [
            feature
            for feature
            in contract.feature_columns
            if feature in configured
        ]

    @staticmethod
    def _status(
        enabled: bool,
    ) -> SelectionComponentStatus:
        """Return enabled or disabled status."""

        if enabled:
            return SelectionComponentStatus.ENABLED

        return SelectionComponentStatus.DISABLED
