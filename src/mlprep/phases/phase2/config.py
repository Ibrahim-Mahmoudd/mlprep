"""Unified configuration for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlprep.phases.phase2.contract.split_contract import (
    SplitContractOptions,
)
from mlprep.phases.phase2.contract.split_roles import (
    SplitRoleOptions,
)
from mlprep.phases.phase2.leakage_firewall.firewall import (
    LeakageFirewallOptions,
)
from mlprep.phases.phase2.output_governance.fingerprint import (
    Phase2FingerprintOptions,
)
from mlprep.phases.phase2.output_governance.metadata import (
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.readiness import (
    Phase2ReadinessOptions,
)
from mlprep.phases.phase2.output_governance.report import (
    Phase2ReportOptions,
)
from mlprep.phases.phase2.output_governance.test_lock import (
    TestLockOptions,
)
from mlprep.phases.phase2.strategies.group_split import (
    GroupSplitOptions,
)
from mlprep.phases.phase2.strategies.group_stratified_split import (
    GroupStratifiedSplitOptions,
)
from mlprep.phases.phase2.strategies.predefined_split import (
    PredefinedSplitOptions,
)
from mlprep.phases.phase2.strategies.purged_time_split import (
    PurgedTimeSplitOptions,
)
from mlprep.phases.phase2.strategies.random_split import (
    RandomSplitOptions,
)
from mlprep.phases.phase2.strategies.regression_stratified_split import (
    RegressionStratifiedSplitOptions,
)
from mlprep.phases.phase2.strategies.stratified_split import (
    StratifiedSplitOptions,
)
from mlprep.phases.phase2.strategies.time_split import (
    TimeSplitOptions,
)
from mlprep.phases.phase2.validation.class_distribution import (
    ClassDistributionValidationOptions,
)
from mlprep.phases.phase2.validation.group_distribution import (
    GroupDistributionValidationOptions,
)
from mlprep.phases.phase2.validation.partition_integrity import (
    PartitionIntegrityValidationOptions,
)
from mlprep.phases.phase2.validation.split_sizes import (
    SplitSizeValidationOptions,
)
from mlprep.phases.phase2.validation.target_distribution import (
    TargetDistributionValidationOptions,
)
from mlprep.phases.phase2.validation.temporal_boundaries import (
    TemporalBoundaryValidationOptions,
)


@dataclass(slots=True)
class Phase2ExecutionOptions:
    """General execution options for Phase 2."""

    enabled: bool = True

    stop_on_contract_failure: bool = True
    stop_on_component_exception: bool = True

    copy_phase1_data: bool = True
    store_execution_metadata: bool = True

    def __post_init__(self) -> None:
        """Validate execution options."""

        boolean_values = {
            "enabled": self.enabled,
            "stop_on_contract_failure": (
                self.stop_on_contract_failure
            ),
            "stop_on_component_exception": (
                self.stop_on_component_exception
            ),
            "copy_phase1_data": (
                self.copy_phase1_data
            ),
            "store_execution_metadata": (
                self.store_execution_metadata
            ),
        }

        for name, value in boolean_values.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a boolean."
                )


@dataclass(slots=True)
class Phase2ValidationOptions:
    """Complete split-validation configuration."""

    split_sizes: SplitSizeValidationOptions = field(
        default_factory=SplitSizeValidationOptions
    )

    class_distribution: (
        ClassDistributionValidationOptions
    ) = field(
        default_factory=(
            ClassDistributionValidationOptions
        )
    )

    target_distribution: (
        TargetDistributionValidationOptions
    ) = field(
        default_factory=(
            TargetDistributionValidationOptions
        )
    )

    group_distribution: (
        GroupDistributionValidationOptions
    ) = field(
        default_factory=(
            GroupDistributionValidationOptions
        )
    )

    temporal_boundaries: (
        TemporalBoundaryValidationOptions
    ) = field(
        default_factory=(
            TemporalBoundaryValidationOptions
        )
    )

    partition_integrity: (
        PartitionIntegrityValidationOptions
    ) = field(
        default_factory=(
            PartitionIntegrityValidationOptions
        )
    )


@dataclass(slots=True)
class Phase2StrategyOptions:
    """Configuration for every supported split strategy."""

    random: RandomSplitOptions

    stratified: StratifiedSplitOptions

    group: GroupSplitOptions

    group_stratified: (
        GroupStratifiedSplitOptions
    )

    time: TimeSplitOptions

    purged_time: PurgedTimeSplitOptions

    regression_stratified: (
        RegressionStratifiedSplitOptions
    )

    predefined: PredefinedSplitOptions | None = None

    @classmethod
    def create_defaults(
        cls,
        *,
        ratios: SplitRatios,
        random_seed: int,
        predefined_split_column: str | None = None,
    ) -> Phase2StrategyOptions:
        """Create consistent options for every strategy."""

        predefined_options = None

        if predefined_split_column is not None:
            predefined_options = (
                PredefinedSplitOptions(
                    split_column=(
                        predefined_split_column
                    )
                )
            )

        return cls(
            random=RandomSplitOptions(
                ratios=ratios.model_copy(
                    deep=True
                ),
                random_seed=random_seed,
            ),
            stratified=StratifiedSplitOptions(
                ratios=ratios.model_copy(
                    deep=True
                ),
                random_seed=random_seed,
            ),
            group=GroupSplitOptions(
                ratios=ratios.model_copy(
                    deep=True
                ),
                random_seed=random_seed,
            ),
            group_stratified=(
                GroupStratifiedSplitOptions(
                    ratios=ratios.model_copy(
                        deep=True
                    ),
                    random_seed=random_seed,
                )
            ),
            time=TimeSplitOptions(
                ratios=ratios.model_copy(
                    deep=True
                ),
            ),
            purged_time=PurgedTimeSplitOptions(
                ratios=ratios.model_copy(
                    deep=True
                ),
            ),
            regression_stratified=(
                RegressionStratifiedSplitOptions(
                    ratios=ratios.model_copy(
                        deep=True
                    ),
                    random_seed=random_seed,
                )
            ),
            predefined=predefined_options,
        )

    def get(
        self,
        strategy: SplitStrategyName,
    ) -> object:
        """Return options for one split strategy."""

        strategy_options = {
            SplitStrategyName.RANDOM: (
                self.random
            ),
            SplitStrategyName.STRATIFIED: (
                self.stratified
            ),
            SplitStrategyName.GROUP: (
                self.group
            ),
            SplitStrategyName.GROUP_STRATIFIED: (
                self.group_stratified
            ),
            SplitStrategyName.TIME: (
                self.time
            ),
            SplitStrategyName.PURGED_TIME: (
                self.purged_time
            ),
            SplitStrategyName.REGRESSION_STRATIFIED: (
                self.regression_stratified
            ),
            SplitStrategyName.PREDEFINED: (
                self.predefined
            ),
        }

        resolved_options = strategy_options[
            strategy
        ]

        if resolved_options is None:
            raise ValueError(
                f"Strategy {strategy.value!r} does not "
                "have configured options."
            )

        return resolved_options


@dataclass(slots=True)
class Phase2Config:
    """Complete configuration for Phase 2."""

    strategy: SplitStrategyName

    ratios: SplitRatios = field(
        default_factory=SplitRatios
    )

    random_seed: int = 42

    execution: Phase2ExecutionOptions = field(
        default_factory=Phase2ExecutionOptions
    )

    split_roles: SplitRoleOptions = field(
        default_factory=SplitRoleOptions
    )

    contract: SplitContractOptions | None = None

    strategies: Phase2StrategyOptions | None = None

    leakage_firewall: LeakageFirewallOptions = field(
        default_factory=LeakageFirewallOptions
    )

    validation: Phase2ValidationOptions = field(
        default_factory=Phase2ValidationOptions
    )

    fingerprint: Phase2FingerprintOptions = field(
        default_factory=Phase2FingerprintOptions
    )

    test_lock: TestLockOptions = field(
        default_factory=TestLockOptions
    )

    readiness: Phase2ReadinessOptions = field(
        default_factory=Phase2ReadinessOptions
    )

    reporting: Phase2ReportOptions = field(
        default_factory=Phase2ReportOptions
    )

    def __post_init__(self) -> None:
        """Validate and synchronize Phase 2 configuration."""

        self.ratios.validate_total()

        if self.strategies is None:
            self.strategies = (
                Phase2StrategyOptions.create_defaults(
                    ratios=self.ratios,
                    random_seed=self.random_seed,
                )
            )

        if self.contract is None:
            self.contract = SplitContractOptions(
                strategy=self.strategy,
                ratios=self.ratios.model_copy(
                    deep=True
                ),
                random_seed=self.random_seed,
                shuffle=self._strategy_shuffles(),
                target_column=(
                    self.split_roles.target_column
                ),
                feature_columns=(
                    list(
                        self.split_roles
                        .feature_columns
                    )
                    if self.split_roles
                    .feature_columns
                    is not None
                    else None
                ),
                identifier_columns=list(
                    self.split_roles
                    .identifier_columns
                ),
                group_columns=list(
                    self.split_roles.group_columns
                ),
                timestamp_column=(
                    self.split_roles
                    .timestamp_column
                ),
                observation_key=list(
                    self.split_roles
                    .observation_key
                ),
                excluded_feature_columns=list(
                    self.split_roles
                    .excluded_feature_columns
                ),
                internal_row_id_column=(
                    self.split_roles
                    .internal_row_id_column
                ),
                lock_test_set=(
                    self.test_lock.enabled
                ),
            )

        self._validate_contract_consistency()
        self._synchronize_options()

    def selected_strategy_options(
        self,
    ) -> object:
        """Return options for the selected strategy."""

        if self.strategies is None:
            raise RuntimeError(
                "Strategy options are not initialized."
            )

        return self.strategies.get(
            self.strategy
        )

    def _validate_contract_consistency(
        self,
    ) -> None:
        """Ensure strategy and contract settings agree."""

        if self.contract is None:
            raise RuntimeError(
                "Split contract is not initialized."
            )

        if self.contract.strategy != self.strategy:
            raise ValueError(
                "Phase2Config.strategy and "
                "contract.strategy must match."
            )

        if (
            self.contract.internal_row_id_column
            != self.split_roles
            .internal_row_id_column
        ):
            raise ValueError(
                "The internal row-ID column must match "
                "between split_roles and contract."
            )

        group_strategies = {
            SplitStrategyName.GROUP,
            SplitStrategyName.GROUP_STRATIFIED,
        }

        if (
            self.strategy in group_strategies
            and not (
                self.split_roles.group_columns
                or self.split_roles
                .allow_phase1_role_fallback
            )
        ):
            raise ValueError(
                f"Strategy {self.strategy.value!r} "
                "requires group columns."
            )

        temporal_strategies = {
            SplitStrategyName.TIME,
            SplitStrategyName.PURGED_TIME,
        }

        if (
            self.strategy in temporal_strategies
            and self.split_roles.timestamp_column
            is None
            and not self.split_roles
            .allow_phase1_role_fallback
        ):
            raise ValueError(
                f"Strategy {self.strategy.value!r} "
                "requires a timestamp column."
            )

        if (
            self.strategy
            == SplitStrategyName.PREDEFINED
            and (
                self.strategies is None
                or self.strategies.predefined
                is None
            )
        ):
            raise ValueError(
                "The predefined strategy requires "
                "PredefinedSplitOptions."
            )

    def _synchronize_options(
        self,
    ) -> None:
        """Synchronize settings used by multiple components."""

        if self.contract is None:
            raise RuntimeError(
                "Split contract is not initialized."
            )

        self.contract.strategy = self.strategy
        self.contract.ratios = (
            self.ratios.model_copy(
                deep=True
            )
        )

        self.contract.random_seed = self.random_seed

        self.contract.target_column = (
            self.split_roles.target_column
        )

        self.contract.feature_columns = (
            list(
                self.split_roles.feature_columns
            )
            if self.split_roles
            .feature_columns
            is not None
            else None
        )

        self.contract.identifier_columns = list(
            self.split_roles.identifier_columns
        )

        self.contract.group_columns = list(
            self.split_roles.group_columns
        )

        self.contract.timestamp_column = (
            self.split_roles.timestamp_column
        )

        self.contract.observation_key = list(
            self.split_roles.observation_key
        )

        self.contract.excluded_feature_columns = list(
            self.split_roles
            .excluded_feature_columns
        )

        self.contract.internal_row_id_column = (
            self.split_roles
            .internal_row_id_column
        )

        self.contract.lock_test_set = (
            self.test_lock.enabled
        )

        self.validation.split_sizes.requested_ratios = (
            self.ratios.model_copy(
                deep=True
            )
        )

        post_outcome_options = (
            self.leakage_firewall.post_outcome
        )

        post_outcome_options.declared_inference_unavailable_columns = list(
            dict.fromkeys(
                [
                    *post_outcome_options
                    .declared_inference_unavailable_columns,
                    *self.split_roles
                    .excluded_feature_columns,
                ]
            )
        )

    def _strategy_shuffles(
        self,
    ) -> bool:
        """Return whether the selected strategy shuffles."""

        return self.strategy not in {SplitStrategyName.TIME, SplitStrategyName.PURGED_TIME, SplitStrategyName.PREDEFINED}

    @classmethod
    def create(
        cls,
        *,
        strategy: SplitStrategyName,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        target_column: str | None = None,
        feature_columns: list[str] | None = None,
        identifier_columns: list[str] | None = None,
        group_columns: list[str] | None = None,
        observation_key: list[str] | None = None,
        timestamp_column: str | None = None,
        excluded_feature_columns: list[str] | None = None,
        internal_row_id_column: str = (
            "__mlprep_row_id__"
        ),
        predefined_split_column: str | None = None,
    ) -> Phase2Config:
        """Create a common Phase 2 configuration."""

        ratios = SplitRatios(
            train=train_ratio,
            validation=validation_ratio,
            test=test_ratio,
        )

        split_roles = SplitRoleOptions(
            target_column=target_column,
            feature_columns=feature_columns,
            identifier_columns=list(
                identifier_columns or []
            ),
            group_columns=list(
                group_columns or []
            ),
            observation_key=list(
                observation_key or []
            ),
            timestamp_column=timestamp_column,
            excluded_feature_columns=list(
                excluded_feature_columns or []
            ),
            internal_row_id_column=(
                internal_row_id_column
            ),
        )

        strategies = (
            Phase2StrategyOptions.create_defaults(
                ratios=ratios,
                random_seed=random_seed,
                predefined_split_column=(
                    predefined_split_column
                ),
            )
        )

        return cls(
            strategy=strategy,
            ratios=ratios,
            random_seed=random_seed,
            split_roles=split_roles,
            strategies=strategies,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert configuration into a dictionary."""

        return self._serialize_value(self)

    @classmethod
    def _serialize_value(
        cls,
        value: Any,
    ) -> Any:
        """Recursively serialize configuration values."""

        if hasattr(
            value,
            "__dataclass_fields__",
        ):
            return {
                name: cls._serialize_value(
                    getattr(value, name)
                )
                for name
                in value.__dataclass_fields__
            }

        if hasattr(value, "model_dump"):
            return cls._serialize_value(
                value.model_dump(
                    mode="json"
                )
            )

        if isinstance(value, dict):
            return {
                cls._serialize_value(key): (
                    cls._serialize_value(item)
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                cls._serialize_value(item)
                for item in value
            ]

        if hasattr(value, "value"):
            return value.value

        return value