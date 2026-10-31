"""Executable balancing-plan construction for Phase 8."""

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

from mlprep.phases.phase8.contract.balancing_contract import (
    BalancingContract,
)
from mlprep.phases.phase8.utils.class_identity import (
    ClassIdentityMapping,
)


class BalancingTask(StrEnum):
    """Supported Phase 8 balancing tasks."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class BalancingStrategy(StrEnum):
    """Supported balancing strategies."""

    NONE = "none"

    CLASS_WEIGHTS_ONLY = "class_weights_only"

    RANDOM_OVERSAMPLING = "random_oversampling"
    RANDOM_UNDERSAMPLING = "random_undersampling"

    SMOTE = "smote"
    BORDERLINE_SMOTE = "borderline_smote"
    ADASYN = "adasyn"

    SMOTE_TOMEK = "smote_tomek"
    SMOTE_ENN = "smote_enn"


class SamplingStrategyKind(StrEnum):
    """Supported sampling-target policies."""

    AUTO = "auto"
    MINORITY = "minority"
    NOT_MINORITY = "not_minority"
    NOT_MAJORITY = "not_majority"
    ALL = "all"
    CUSTOM = "custom"


class NeighborShortagePolicy(StrEnum):
    """Policy for insufficient nearest-neighbor classes."""

    REDUCE_NEIGHBORS = "reduce_neighbors"
    RANDOM_OVERSAMPLING = "random_oversampling"
    BLOCK = "block"


class ClassWeightMethod(StrEnum):
    """Supported class-weight methods."""

    NONE = "none"
    BALANCED = "balanced"
    INVERSE_FREQUENCY = "inverse_frequency"
    SQRT_INVERSE_FREQUENCY = "sqrt_inverse_frequency"
    EFFECTIVE_NUMBER = "effective_number"
    CUSTOM = "custom"


class WeightNormalizationKind(StrEnum):
    """Supported weight-normalization policies."""

    NONE = "none"
    MEAN_ONE = "mean_one"
    SUM_ONE = "sum_one"
    MAX_ONE = "max_one"


class HybridCleaningKind(StrEnum):
    """Supported hybrid cleaning methods."""

    TOMEK = "tomek"
    ENN = "enn"


class PlanComponentStatus(StrEnum):
    """Status of one balancing-plan component."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    BLOCKED = "blocked"


class SamplingPlan(BaseModel):
    """Executable training sampling plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    strategy: BalancingStrategy = (
        BalancingStrategy.NONE
    )

    sampling_strategy: SamplingStrategyKind = (
        SamplingStrategyKind.AUTO
    )

    custom_target_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    random_seed: int = 42

    minimum_class_count: int = Field(
        default=2,
        ge=1,
    )

    maximum_expansion_ratio: float | None = Field(
        default=None,
        ge=1.0,
    )

    maximum_contraction_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    preserve_original_rows: bool = True

    allow_original_row_removal: bool = False

    status: PlanComponentStatus = (
        PlanComponentStatus.ENABLED
    )

    @model_validator(mode="after")
    def validate_sampling_plan(
        self,
    ) -> SamplingPlan:
        """Validate the sampling plan."""

        if (
            self.sampling_strategy
            == SamplingStrategyKind.CUSTOM
            and not self.custom_target_counts
        ):
            raise ValueError(
                "custom_target_counts is required when "
                "sampling_strategy is 'custom'."
            )

        if (
            self.sampling_strategy
            != SamplingStrategyKind.CUSTOM
            and self.custom_target_counts
        ):
            raise ValueError(
                "custom_target_counts can only be used with "
                "the 'custom' sampling strategy."
            )

        nonpositive_counts = {
            class_label: count
            for class_label, count
            in self.custom_target_counts.items()
            if count < 1
        }

        if nonpositive_counts:
            raise ValueError(
                "Custom class target counts must be positive: "
                f"{nonpositive_counts}"
            )

        undersampling_strategies = {
            BalancingStrategy.RANDOM_UNDERSAMPLING,
            BalancingStrategy.SMOTE_TOMEK,
            BalancingStrategy.SMOTE_ENN,
        }

        if (
            self.strategy in undersampling_strategies
            and self.preserve_original_rows
            and not self.allow_original_row_removal
        ):
            raise ValueError(
                "The configured strategy may remove original "
                "training rows, but original-row removal is "
                "disabled."
            )

        return self


class NeighborSamplingPlan(BaseModel):
    """Nearest-neighbor synthetic sampling plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    nearest_neighbors: int = Field(
        default=5,
        ge=1,
    )

    neighborhood_size: int = Field(
        default=10,
        ge=1,
    )

    shortage_policy: NeighborShortagePolicy = (
        NeighborShortagePolicy.REDUCE_NEIGHBORS
    )

    minimum_effective_neighbors: int = Field(
        default=1,
        ge=1,
    )

    maximum_synthetic_expansion_ratio: float | None = Field(
        default=None,
        ge=1.0,
    )

    status: PlanComponentStatus = (
        PlanComponentStatus.DISABLED
    )

    @model_validator(mode="after")
    def validate_neighbor_plan(
        self,
    ) -> NeighborSamplingPlan:
        """Validate neighbor-sampling settings."""

        if (
            self.minimum_effective_neighbors
            > self.nearest_neighbors
        ):
            raise ValueError(
                "minimum_effective_neighbors cannot exceed "
                "nearest_neighbors."
            )

        return self


class BorderlineSmotePlan(BaseModel):
    """Borderline-SMOTE-specific options."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    method: str = "borderline-1"

    danger_neighbors: int = Field(
        default=10,
        ge=1,
    )

    status: PlanComponentStatus = (
        PlanComponentStatus.DISABLED
    )

    @model_validator(mode="after")
    def validate_method(
        self,
    ) -> BorderlineSmotePlan:
        """Validate the Borderline-SMOTE method."""

        if self.method not in {
            "borderline-1",
            "borderline-2",
        }:
            raise ValueError(
                "Borderline-SMOTE method must be "
                "'borderline-1' or 'borderline-2'."
            )

        return self


class AdasynPlan(BaseModel):
    """ADASYN-specific options."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    density_neighbors: int = Field(
        default=5,
        ge=1,
    )

    maximum_generation_attempts: int = Field(
        default=100,
        ge=1,
    )

    status: PlanComponentStatus = (
        PlanComponentStatus.DISABLED
    )


class HybridSamplingPlan(BaseModel):
    """Hybrid sampling and cleaning options."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    cleaning: HybridCleaningKind = (
        HybridCleaningKind.TOMEK
    )

    enn_neighbors: int = Field(
        default=3,
        ge=1,
    )

    allow_original_row_removal: bool = False

    status: PlanComponentStatus = (
        PlanComponentStatus.DISABLED
    )


class ClassWeightPlan(BaseModel):
    """Class-weight generation plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    enabled: bool = True

    method: ClassWeightMethod = (
        ClassWeightMethod.BALANCED
    )

    custom_weights: dict[
        str,
        float,
    ] = Field(
        default_factory=dict
    )

    normalization: WeightNormalizationKind = (
        WeightNormalizationKind.MEAN_ONE
    )

    effective_number_beta: float = Field(
        default=0.9999,
        gt=0.0,
        lt=1.0,
    )

    generate_sample_weights: bool = True

    status: PlanComponentStatus = (
        PlanComponentStatus.ENABLED
    )

    @model_validator(mode="after")
    def validate_weight_plan(
        self,
    ) -> ClassWeightPlan:
        """Validate class-weight settings."""

        if (
            self.enabled
            and self.method
            == ClassWeightMethod.NONE
        ):
            raise ValueError(
                "Enabled class weighting requires a method "
                "other than 'none'."
            )

        if (
            not self.enabled
            and self.method
            != ClassWeightMethod.NONE
        ):
            self.method = ClassWeightMethod.NONE

        if (
            self.method == ClassWeightMethod.CUSTOM
            and not self.custom_weights
        ):
            raise ValueError(
                "custom_weights is required for the custom "
                "class-weight method."
            )

        if (
            self.method != ClassWeightMethod.CUSTOM
            and self.custom_weights
        ):
            raise ValueError(
                "custom_weights can only be used with the "
                "custom class-weight method."
            )

        invalid_weights = {
            class_label: weight
            for class_label, weight
            in self.custom_weights.items()
            if weight <= 0.0
        }

        if invalid_weights:
            raise ValueError(
                "Custom class weights must be positive: "
                f"{invalid_weights}"
            )

        return self


class RowIdentityPlan(BaseModel):
    """Output row-ID generation policy."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    preserve_original_row_ids: bool = True

    synthetic_prefix: str = "phase8__synthetic"
    duplicate_prefix: str = "phase8__duplicate"

    include_strategy_in_identifier: bool = True

    deterministic: bool = True

    reject_original_id_collisions: bool = True

    @model_validator(mode="after")
    def validate_prefixes(
        self,
    ) -> RowIdentityPlan:
        """Validate row-ID prefixes."""

        if not self.synthetic_prefix.strip():
            raise ValueError(
                "synthetic_prefix cannot be empty."
            )

        if not self.duplicate_prefix.strip():
            raise ValueError(
                "duplicate_prefix cannot be empty."
            )

        if (
            self.synthetic_prefix
            == self.duplicate_prefix
        ):
            raise ValueError(
                "synthetic_prefix and duplicate_prefix "
                "must be different."
            )

        return self


class PartitionProtectionPlan(BaseModel):
    """Validation and test protection policy."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    protect_validation: bool = True
    protect_test: bool = True

    require_identical_validation_fingerprint: bool = True
    require_identical_test_fingerprint: bool = True

    reject_validation_resampling: bool = True
    reject_test_resampling: bool = True


class BalancingPlan(BaseModel):
    """Complete executable Phase 8 balancing plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    task: BalancingTask

    sampling: SamplingPlan
    neighbors: NeighborSamplingPlan

    borderline_smote: BorderlineSmotePlan
    adasyn: AdasynPlan
    hybrid: HybridSamplingPlan

    class_weights: ClassWeightPlan
    row_identity: RowIdentityPlan

    partition_protection: PartitionProtectionPlan

    source_feature_columns: list[str] = Field(
        default_factory=list
    )

    source_class_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict
    )

    class_identity_mapping: ClassIdentityMapping

    resampling_authorized: bool = False
    target_access_authorized: bool = False
    synthetic_generation_authorized: bool = False

    blocked_components: list[str] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @property
    def resampling_enabled(
        self,
    ) -> bool:
        """Return whether training rows may change."""

        return self.sampling.strategy not in {
            BalancingStrategy.NONE,
            BalancingStrategy.CLASS_WEIGHTS_ONLY,
        }

    @property
    def synthetic_sampling_enabled(
        self,
    ) -> bool:
        """Return whether synthetic rows may be generated."""

        return self.sampling.strategy in {
            BalancingStrategy.SMOTE,
            BalancingStrategy.BORDERLINE_SMOTE,
            BalancingStrategy.ADASYN,
            BalancingStrategy.SMOTE_TOMEK,
            BalancingStrategy.SMOTE_ENN,
        }

    @property
    def class_weighting_enabled(
        self,
    ) -> bool:
        """Return whether class weights are enabled."""

        return bool(
            self.class_weights.enabled
            and self.class_weights.method
            != ClassWeightMethod.NONE
        )

    @property
    def has_blocked_components(
        self,
    ) -> bool:
        """Return whether blocked components exist."""

        return bool(
            self.blocked_components
        )


@dataclass(slots=True)
class BalancingPlanOptions:
    """Options controlling Phase 8 plan construction."""

    enabled: bool = True

    task: BalancingTask = (
        BalancingTask.CLASSIFICATION
    )

    strategy: BalancingStrategy = (
        BalancingStrategy.NONE
    )

    sampling_strategy: SamplingStrategyKind = (
        SamplingStrategyKind.AUTO
    )

    custom_target_counts: dict[
        str,
        int,
    ] = field(
        default_factory=dict
    )

    random_seed: int = 42

    minimum_class_count: int = 2

    maximum_expansion_ratio: float | None = None
    maximum_contraction_ratio: float | None = None

    preserve_original_rows: bool = True
    allow_original_row_removal: bool = False

    nearest_neighbors: int = 5
    neighborhood_size: int = 10

    neighbor_shortage_policy: NeighborShortagePolicy = (
        NeighborShortagePolicy.REDUCE_NEIGHBORS
    )

    minimum_effective_neighbors: int = 1

    maximum_synthetic_expansion_ratio: float | None = None

    borderline_method: str = "borderline-1"
    borderline_danger_neighbors: int = 10

    adasyn_density_neighbors: int = 5
    adasyn_maximum_generation_attempts: int = 100

    hybrid_cleaning: HybridCleaningKind = (
        HybridCleaningKind.TOMEK
    )

    hybrid_enn_neighbors: int = 3

    class_weights_enabled: bool = True

    class_weight_method: ClassWeightMethod = (
        ClassWeightMethod.BALANCED
    )

    custom_class_weights: dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )

    class_weight_normalization: WeightNormalizationKind = (
        WeightNormalizationKind.MEAN_ONE
    )

    effective_number_beta: float = 0.9999

    generate_sample_weights: bool = True

    preserve_original_row_ids: bool = True

    synthetic_row_id_prefix: str = (
        "phase8__synthetic"
    )

    duplicate_row_id_prefix: str = (
        "phase8__duplicate"
    )

    include_strategy_in_row_id: bool = True
    deterministic_row_ids: bool = True
    reject_original_id_collisions: bool = True

    protect_validation: bool = True
    protect_test: bool = True

    require_identical_validation_fingerprint: bool = True
    require_identical_test_fingerprint: bool = True

    reject_validation_resampling: bool = True
    reject_test_resampling: bool = True

    resampling_authorized: bool = False
    target_access_authorized: bool = True
    synthetic_generation_authorized: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize plan options."""

        enum_fields = {
            "task": BalancingTask,
            "strategy": BalancingStrategy,
            "sampling_strategy": SamplingStrategyKind,
            "neighbor_shortage_policy": (
                NeighborShortagePolicy
            ),
            "hybrid_cleaning": HybridCleaningKind,
            "class_weight_method": (
                ClassWeightMethod
            ),
            "class_weight_normalization": (
                WeightNormalizationKind
            ),
        }

        for field_name, enum_type in enum_fields.items():
            value = getattr(
                self,
                field_name
            )

            if not isinstance(
                value,
                enum_type,
            ):
                setattr(
                    self,
                    field_name,
                    enum_type(
                        value
                    ),
                )

        if self.minimum_class_count < 1:
            raise ValueError(
                "minimum_class_count must be positive."
            )

        if self.nearest_neighbors < 1:
            raise ValueError(
                "nearest_neighbors must be positive."
            )

        if self.neighborhood_size < 1:
            raise ValueError(
                "neighborhood_size must be positive."
            )

        if self.minimum_effective_neighbors < 1:
            raise ValueError(
                "minimum_effective_neighbors must be positive."
            )

        if (
            self.minimum_effective_neighbors
            > self.nearest_neighbors
        ):
            raise ValueError(
                "minimum_effective_neighbors cannot exceed "
                "nearest_neighbors."
            )

        if self.borderline_danger_neighbors < 1:
            raise ValueError(
                "borderline_danger_neighbors must be positive."
            )

        if self.adasyn_density_neighbors < 1:
            raise ValueError(
                "adasyn_density_neighbors must be positive."
            )

        if (
            self.adasyn_maximum_generation_attempts
            < 1
        ):
            raise ValueError(
                "adasyn_maximum_generation_attempts must "
                "be positive."
            )

        if self.hybrid_enn_neighbors < 1:
            raise ValueError(
                "hybrid_enn_neighbors must be positive."
            )

        if (
            self.maximum_expansion_ratio is not None
            and self.maximum_expansion_ratio < 1.0
        ):
            raise ValueError(
                "maximum_expansion_ratio must be at least one."
            )

        if (
            self.maximum_contraction_ratio is not None
            and not 0.0
            <= self.maximum_contraction_ratio
            <= 1.0
        ):
            raise ValueError(
                "maximum_contraction_ratio must be between "
                "zero and one."
            )

        if (
            self.maximum_synthetic_expansion_ratio
            is not None
            and self.maximum_synthetic_expansion_ratio
            < 1.0
        ):
            raise ValueError(
                "maximum_synthetic_expansion_ratio must be "
                "at least one."
            )

        if not 0.0 < self.effective_number_beta < 1.0:
            raise ValueError(
                "effective_number_beta must be between "
                "zero and one."
            )

        for class_label, count in (
            self.custom_target_counts.items()
        ):
            if count < 1:
                raise ValueError(
                    "Custom target counts must be positive. "
                    f"Invalid class: {class_label!r}"
                )

        for class_label, weight in (
            self.custom_class_weights.items()
        ):
            if weight <= 0.0:
                raise ValueError(
                    "Custom class weights must be positive. "
                    f"Invalid class: {class_label!r}"
                )


class BalancingPlanBuilder:
    """Build an executable Phase 8 balancing plan."""

    component_name = "balancing_plan"

    def __init__(
        self,
        options: BalancingPlanOptions | None = None,
    ) -> None:
        """Initialize the plan builder."""

        self.options = (
            options
            or BalancingPlanOptions()
        )

    def build(
        self,
        *,
        contract: BalancingContract,
    ) -> BalancingPlan:
        """Build the complete training-balancing plan."""

        if not isinstance(
            contract,
            BalancingContract,
        ):
            raise TypeError(
                "contract must be a BalancingContract."
            )

        if not self.options.enabled:
            return self._disabled_plan(
                contract
            )

        blocked_components: list[str] = []
        warnings: list[str] = []
        normalized_custom_target_counts = (
            self._validate_custom_target_counts(
                contract=contract,
            )
        )
        normalized_custom_class_weights = (
            self._normalize_class_mapping(
                configured=self.options.custom_class_weights,
                identity_mapping=contract.class_identity_mapping,
            )
            if (
                self.options.class_weights_enabled
                and self.options.class_weight_method
                == ClassWeightMethod.CUSTOM
            )
            else {}
        )

        self._validate_task(
            contract=contract,
            blocked_components=(
                blocked_components
            ),
        )

        strategy = self.options.strategy

        resampling_enabled = strategy not in {
            BalancingStrategy.NONE,
            BalancingStrategy.CLASS_WEIGHTS_ONLY,
        }

        synthetic_enabled = strategy in {
            BalancingStrategy.SMOTE,
            BalancingStrategy.BORDERLINE_SMOTE,
            BalancingStrategy.ADASYN,
            BalancingStrategy.SMOTE_TOMEK,
            BalancingStrategy.SMOTE_ENN,
        }

        if (
            resampling_enabled
            and not self.options.resampling_authorized
        ):
            blocked_components.append(
                "training_resampling"
            )

        if (
            resampling_enabled
            and not self.options
            .target_access_authorized
        ):
            blocked_components.append(
                "training_target_access"
            )

        if (
            synthetic_enabled
            and not self.options
            .synthetic_generation_authorized
        ):
            blocked_components.append(
                "synthetic_generation"
            )

        if (
            strategy
            == BalancingStrategy.RANDOM_UNDERSAMPLING
            and not self.options
            .allow_original_row_removal
        ):
            blocked_components.append(
                "original_row_removal"
            )

        if (
            strategy
            in {
                BalancingStrategy.SMOTE_TOMEK,
                BalancingStrategy.SMOTE_ENN,
            }
            and not self.options
            .allow_original_row_removal
        ):
            blocked_components.append(
                "hybrid_original_row_removal"
            )

        effective_neighbor_count = (
            self._resolve_neighbor_count(
                contract=contract,
                synthetic_enabled=(
                    synthetic_enabled
                ),
                blocked_components=(
                    blocked_components
                ),
                warnings=warnings,
            )
        )

        sampling_plan = SamplingPlan(
            strategy=strategy,
            sampling_strategy=(
                self.options.sampling_strategy
            ),
            custom_target_counts=dict(
                normalized_custom_target_counts
            ),
            random_seed=(
                self.options.random_seed
            ),
            minimum_class_count=(
                self.options.minimum_class_count
            ),
            maximum_expansion_ratio=(
                self.options
                .maximum_expansion_ratio
            ),
            maximum_contraction_ratio=(
                self.options
                .maximum_contraction_ratio
            ),
            preserve_original_rows=(
                self.options.preserve_original_rows
            ),
            allow_original_row_removal=(
                self.options
                .allow_original_row_removal
            ),
            status=(
                PlanComponentStatus.BLOCKED
                if blocked_components
                and resampling_enabled
                else (
                    PlanComponentStatus.ENABLED
                    if resampling_enabled
                    else PlanComponentStatus.DISABLED
                )
            ),
        )

        neighbors = NeighborSamplingPlan(
            nearest_neighbors=(
                effective_neighbor_count
            ),
            neighborhood_size=(
                self.options.neighborhood_size
            ),
            shortage_policy=(
                self.options
                .neighbor_shortage_policy
            ),
            minimum_effective_neighbors=(
                self.options
                .minimum_effective_neighbors
            ),
            maximum_synthetic_expansion_ratio=(
                self.options
                .maximum_synthetic_expansion_ratio
            ),
            status=(
                PlanComponentStatus.ENABLED
                if synthetic_enabled
                and "synthetic_generation"
                not in blocked_components
                else (
                    PlanComponentStatus.BLOCKED
                    if synthetic_enabled
                    else PlanComponentStatus.DISABLED
                )
            ),
        )

        borderline_smote = BorderlineSmotePlan(
            method=(
                self.options.borderline_method
            ),
            danger_neighbors=(
                self.options
                .borderline_danger_neighbors
            ),
            status=(
                PlanComponentStatus.ENABLED
                if strategy
                == BalancingStrategy.BORDERLINE_SMOTE
                else PlanComponentStatus.DISABLED
            ),
        )

        adasyn = AdasynPlan(
            density_neighbors=(
                self.options
                .adasyn_density_neighbors
            ),
            maximum_generation_attempts=(
                self.options
                .adasyn_maximum_generation_attempts
            ),
            status=(
                PlanComponentStatus.ENABLED
                if strategy
                == BalancingStrategy.ADASYN
                else PlanComponentStatus.DISABLED
            ),
        )

        hybrid = HybridSamplingPlan(
            cleaning=(
                self._hybrid_cleaning_for_strategy(
                    strategy
                )
            ),
            enn_neighbors=(
                self.options.hybrid_enn_neighbors
            ),
            allow_original_row_removal=(
                self.options
                .allow_original_row_removal
            ),
            status=(
                PlanComponentStatus.ENABLED
                if strategy
                in {
                    BalancingStrategy.SMOTE_TOMEK,
                    BalancingStrategy.SMOTE_ENN,
                }
                else PlanComponentStatus.DISABLED
            ),
        )

        class_weight_method = (
            self.options.class_weight_method
            if self.options.class_weights_enabled
            else ClassWeightMethod.NONE
        )

        class_weights = ClassWeightPlan(
            enabled=(
                self.options.class_weights_enabled
            ),
            method=class_weight_method,
            custom_weights=dict(
                normalized_custom_class_weights
            ),
            normalization=(
                self.options
                .class_weight_normalization
            ),
            effective_number_beta=(
                self.options
                .effective_number_beta
            ),
            generate_sample_weights=(
                self.options
                .generate_sample_weights
            ),
            status=(
                PlanComponentStatus.ENABLED
                if self.options
                .class_weights_enabled
                else PlanComponentStatus.DISABLED
            ),
        )

        row_identity = RowIdentityPlan(
            preserve_original_row_ids=(
                self.options
                .preserve_original_row_ids
            ),
            synthetic_prefix=(
                self.options
                .synthetic_row_id_prefix
            ),
            duplicate_prefix=(
                self.options
                .duplicate_row_id_prefix
            ),
            include_strategy_in_identifier=(
                self.options
                .include_strategy_in_row_id
            ),
            deterministic=(
                self.options
                .deterministic_row_ids
            ),
            reject_original_id_collisions=(
                self.options
                .reject_original_id_collisions
            ),
        )

        partition_protection = (
            PartitionProtectionPlan(
                protect_validation=(
                    self.options
                    .protect_validation
                ),
                protect_test=(
                    self.options.protect_test
                ),
                require_identical_validation_fingerprint=(
                    self.options
                    .require_identical_validation_fingerprint
                ),
                require_identical_test_fingerprint=(
                    self.options
                    .require_identical_test_fingerprint
                ),
                reject_validation_resampling=(
                    self.options
                    .reject_validation_resampling
                ),
                reject_test_resampling=(
                    self.options
                    .reject_test_resampling
                ),
            )
        )

        return BalancingPlan(
            task=self.options.task,
            sampling=sampling_plan,
            neighbors=neighbors,
            borderline_smote=(
                borderline_smote
            ),
            adasyn=adasyn,
            hybrid=hybrid,
            class_weights=class_weights,
            row_identity=row_identity,
            partition_protection=(
                partition_protection
            ),
            source_feature_columns=(
                contract.feature_columns
            ),
            source_class_counts=dict(
                contract.training_class_counts
            ),
            class_identity_mapping=(
                contract.class_identity_mapping.model_copy(
                    deep=True
                )
            ),
            resampling_authorized=(
                self.options
                .resampling_authorized
            ),
            target_access_authorized=(
                self.options
                .target_access_authorized
            ),
            synthetic_generation_authorized=(
                self.options
                .synthetic_generation_authorized
            ),
            blocked_components=list(
                dict.fromkeys(
                    blocked_components
                )
            ),
            warnings=warnings,
            metadata={
                "source_row_count": (
                    contract.train.row_count()
                ),
                "source_feature_count": len(
                    contract.feature_columns
                ),
                "class_count": (
                    contract.class_count
                ),
                "smallest_class_count": (
                    contract
                    .smallest_class_count
                ),
                "largest_class_count": (
                    contract
                    .largest_class_count
                ),
                "configured_neighbor_count": (
                    self.options.nearest_neighbors
                ),
                "effective_neighbor_count": (
                    effective_neighbor_count
                ),
            },
        )

    def _disabled_plan(
        self,
        contract: BalancingContract,
    ) -> BalancingPlan:
        """Build a disabled balancing plan."""

        return BalancingPlan(
            task=self.options.task,
            sampling=SamplingPlan(
                strategy=BalancingStrategy.NONE,
                preserve_original_rows=True,
                allow_original_row_removal=False,
                status=(
                    PlanComponentStatus.DISABLED
                ),
            ),
            neighbors=NeighborSamplingPlan(
                status=(
                    PlanComponentStatus.DISABLED
                ),
            ),
            borderline_smote=BorderlineSmotePlan(
                status=(
                    PlanComponentStatus.DISABLED
                ),
            ),
            adasyn=AdasynPlan(
                status=(
                    PlanComponentStatus.DISABLED
                ),
            ),
            hybrid=HybridSamplingPlan(
                status=(
                    PlanComponentStatus.DISABLED
                ),
            ),
            class_weights=ClassWeightPlan(
                enabled=False,
                method=ClassWeightMethod.NONE,
                generate_sample_weights=False,
                status=(
                    PlanComponentStatus.DISABLED
                ),
            ),
            row_identity=RowIdentityPlan(),
            partition_protection=(
                PartitionProtectionPlan()
            ),
            source_feature_columns=(
                contract.feature_columns
            ),
            source_class_counts=dict(
                contract.training_class_counts
            ),
            class_identity_mapping=(
                contract.class_identity_mapping.model_copy(
                    deep=True
                )
            ),
            resampling_authorized=False,
            target_access_authorized=False,
            synthetic_generation_authorized=False,
            metadata={
                "enabled": False,
            },
        )

    def _resolve_neighbor_count(
        self,
        *,
        contract: BalancingContract,
        synthetic_enabled: bool,
        blocked_components: list[str],
        warnings: list[str],
    ) -> int:
        """Resolve an effective neighbor count."""

        configured = (
            self.options.nearest_neighbors
        )

        if not synthetic_enabled:
            return configured

        maximum_supported = (
            contract.smallest_class_count
            - 1
        )

        if maximum_supported >= configured:
            return configured

        policy = (
            self.options
            .neighbor_shortage_policy
        )

        if (
            policy
            == NeighborShortagePolicy.REDUCE_NEIGHBORS
        ):
            effective = max(
                self.options
                .minimum_effective_neighbors,
                maximum_supported,
            )

            if effective > maximum_supported:
                blocked_components.append(
                    "nearest_neighbors"
                )

                return configured

            warnings.append(
                "The nearest-neighbor count was reduced "
                "because the smallest class is too small."
            )

            return effective

        if (
            policy
            == NeighborShortagePolicy
            .RANDOM_OVERSAMPLING
        ):
            warnings.append(
                "Synthetic sampling must fall back to random "
                "oversampling because the smallest class does "
                "not support the configured neighbor count."
            )

            return configured

        blocked_components.append(
            "nearest_neighbors"
        )

        return configured

    def _validate_task(
        self,
        *,
        contract: BalancingContract,
        blocked_components: list[str],
    ) -> None:
        """Validate task compatibility."""

        if (
            self.options.task
            == BalancingTask.REGRESSION
            and self.options.strategy
            not in {
                BalancingStrategy.NONE,
                BalancingStrategy.CLASS_WEIGHTS_ONLY,
            }
        ):
            blocked_components.append(
                "regression_resampling"
            )

        if (
            self.options.task
            == BalancingTask.CLASSIFICATION
            and contract.class_count < 2
        ):
            blocked_components.append(
                "classification_target"
            )

    def _validate_custom_target_counts(
        self,
        *,
        contract: BalancingContract,
    ) -> dict[str, int]:
        """Resolve custom sampling counts to canonical class keys."""

        if (
            self.options.sampling_strategy
            != SamplingStrategyKind.CUSTOM
        ):
            return {}

        try:
            resolved = self._normalize_class_mapping(
                configured=self.options.custom_target_counts,
                identity_mapping=contract.class_identity_mapping,
            )
        except KeyError as error:
            raise ValueError(
                "Custom target counts contain an unknown class label."
            ) from error

        unknown_classes = set(resolved).difference(
            contract.training_class_counts
        )
        if unknown_classes:
            raise ValueError(
                "Custom target counts contain classes without "
                f"training rows: {sorted(unknown_classes)}"
            )
        return resolved

    @staticmethod
    def _normalize_class_mapping(
        *,
        configured: dict[str, Any],
        identity_mapping: ClassIdentityMapping,
    ) -> dict[str, Any]:
        """Resolve legacy labels while rejecting ambiguous aliases."""

        resolved: dict[str, Any] = {}
        for label, value in configured.items():
            key = identity_mapping.resolve_configured_key(label)
            if key in resolved:
                raise ValueError(
                    "Multiple configured class labels resolve to "
                    f"class key {key!r}."
                )
            resolved[key] = value
        return resolved

    def _hybrid_cleaning_for_strategy(
        self,
        strategy: BalancingStrategy,
    ) -> HybridCleaningKind:
        """Resolve hybrid cleaning from strategy."""

        if strategy == BalancingStrategy.SMOTE_ENN:
            return HybridCleaningKind.ENN

        if strategy == BalancingStrategy.SMOTE_TOMEK:
            return HybridCleaningKind.TOMEK

        return self.options.hybrid_cleaning
