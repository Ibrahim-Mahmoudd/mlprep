"""Training-only balancing pipeline for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase8.analysis import (
    ClassDistributionAnalyzer,
    ClassDistributionOptions,
    ClassDistributionResult,
    ImbalanceMetricsAnalyzer,
    ImbalanceMetricsOptions,
    ImbalanceMetricsResult,
    StrategyRecommendationResult,
    StrategyRecommender,
    StrategyRecommenderOptions,
)
from mlprep.phases.phase8.contract import (
    BalancingContract,
    BalancingPlan,
    BalancingStrategy,
    HybridCleaningKind,
    NeighborShortagePolicy,
    SamplingStrategyKind,
)
from mlprep.phases.phase8.lineage import (
    RowLineageState,
    RowLineageTracker,
)
from mlprep.phases.phase8.orchestration.fit_context import (
    BalancingFitContext,
)
from mlprep.phases.phase8.orchestration.partition_guard import (
    PartitionGuard,
    PartitionGuardOptions,
    PartitionGuardResult,
)
from mlprep.phases.phase8.samplers import (
    AdasynSampler,
    BorderlineSmoteSampler,
    HybridSampler,
    RandomOverSampler,
    RandomUnderSampler,
    ResampledRowType,
    SamplerExecutionContext,
    SamplerResult,
    SamplerRowRecord,
    SamplerState,
    SmoteSampler,
)
from mlprep.phases.phase8.utils.class_identity import (
    ClassIdentityMapping,
    build_class_identity_mapping,
    normalize_class_value,
)
from mlprep.phases.phase8.weighting import (
    ClassWeightCalculator,
    ClassWeightOptions,
    ClassWeightResult,
    SampleWeightGenerator,
    SampleWeightOptions,
    SampleWeightResult,
)


class BalancingPipelineResult(BaseModel):
    """Complete output produced by the balancing pipeline."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    distribution_before: ClassDistributionResult
    distribution_after: ClassDistributionResult

    metrics_before: ImbalanceMetricsResult
    metrics_after: ImbalanceMetricsResult

    recommendation: StrategyRecommendationResult

    configured_strategy: BalancingStrategy
    executed_strategy: BalancingStrategy

    sampler_state: SamplerState

    class_weights: ClassWeightResult | None = None
    sample_weights: SampleWeightResult | None = None

    row_lineage: RowLineageState

    validation_guard: PartitionGuardResult
    test_guard: PartitionGuardResult

    fitted_on_training_only: bool = True

    validation_resampled: bool = False
    test_resampled: bool = False

    validation_remained_protected: bool = True
    test_remained_protected: bool = True

    target_access_authorized: bool = False
    resampling_authorized: bool = False
    synthetic_generation_authorized: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class BalancingPipelineOptions:
    """Options controlling the Phase 8 balancing pipeline."""

    distribution: ClassDistributionOptions = field(
        default_factory=ClassDistributionOptions,
    )

    metrics: ImbalanceMetricsOptions = field(
        default_factory=ImbalanceMetricsOptions,
    )

    recommender: StrategyRecommenderOptions = field(
        default_factory=StrategyRecommenderOptions,
    )

    class_weights: ClassWeightOptions = field(
        default_factory=ClassWeightOptions,
    )

    sample_weights: SampleWeightOptions = field(
        default_factory=SampleWeightOptions,
    )

    partition_guard: PartitionGuardOptions = field(
        default_factory=PartitionGuardOptions,
    )

    require_nonempty_training_output: bool = True

    allow_sampler_fallback: bool = True

    copy_validation_partition: bool = True
    copy_test_partition: bool = True


class BalancingPipeline:
    """Analyze and balance the training partition only."""

    component_name = "balancing_pipeline"

    def __init__(
        self,
        options: BalancingPipelineOptions | None = None,
    ) -> None:
        """Initialize the balancing pipeline."""

        self.options = (
            options
            or BalancingPipelineOptions()
        )

        self.execution_id = str(
            uuid4()
        )

        self.partition_guard = PartitionGuard(
            options=self.options.partition_guard
        )

        self.result: BalancingPipelineResult | None = None
        self.executed_strategy = BalancingStrategy.NONE

    def fit_resample(
        self,
        *,
        contract: BalancingContract,
        plan: BalancingPlan,
    ) -> tuple[
        DatasetPartition,
        DatasetPartition,
        DatasetPartition,
        BalancingPipelineResult,
    ]:
        """Execute the complete training-only balancing workflow."""

        if not isinstance(
            contract,
            BalancingContract,
        ):
            raise TypeError(
                "contract must be a BalancingContract."
            )

        if not isinstance(
            plan,
            BalancingPlan,
        ):
            raise TypeError(
                "plan must be a BalancingPlan."
            )

        if plan.has_blocked_components:
            raise PermissionError(
                "The balancing plan contains blocked "
                f"components: {plan.blocked_components}"
            )

        if (
            plan.class_identity_mapping.class_order
            != contract.class_identity_mapping.class_order
        ):
            raise ValueError(
                "The balancing plan class identities do not "
                "match the training contract."
            )

        analysis_context = (
            BalancingFitContext.training_analysis(
                target_access_authorized=(
                    plan.target_access_authorized
                )
            )
        )

        analysis_context.assert_analysis_allowed()

        validation_context = (
            BalancingFitContext.validation_guard()
        )

        test_context = (
            BalancingFitContext.test_guard()
        )

        self.partition_guard.capture(
            partition=contract.validation,
            context=validation_context,
        )

        self.partition_guard.capture(
            partition=contract.test,
            context=test_context,
        )

        validation_output = contract.validation.copy(
            deep=self.options.copy_validation_partition
        )

        test_output = contract.test.copy(
            deep=self.options.copy_test_partition
        )

        distribution_before = (
            ClassDistributionAnalyzer(
                options=self.options.distribution
            ).analyze(
                target=contract.train.y
            )
        )

        metrics_before = ImbalanceMetricsAnalyzer(
            options=self.options.metrics
        ).analyze(
            distribution=distribution_before
        )

        recommendation = StrategyRecommender(
            options=self.options.recommender
        ).recommend(
            features=contract.train.X,
            distribution=distribution_before,
            metrics=metrics_before,
            configured_strategy=(
                plan.sampling.strategy
            ),
            neighbor_shortage_policy=(
                plan.neighbors.shortage_policy
            ),
        )

        class_weight_result = (
            self._calculate_class_weights(
                source_target=contract.train.y,
                class_identity_mapping=(
                    contract.class_identity_mapping
                ),
                plan=plan,
            )
        )

        sampler_result, executed_strategy = (
            self._execute_sampling(
                contract=contract,
                plan=plan,
            )
        )

        training_output = DatasetPartition(
            name=PartitionName.TRAIN,
            X=sampler_result.features,
            y=sampler_result.target,
            row_ids=sampler_result.row_ids,
            metadata=(
                contract.train.metadata.model_copy(
                    deep=True
                )
            ),
        )

        if (
            self.options.require_nonempty_training_output
            and training_output.row_count() == 0
        ):
            raise ValueError(
                "Balancing produced an empty training partition."
            )

        lineage = RowLineageTracker(
            strategy=executed_strategy.value
        ).build(
            source_features=contract.train.X,
            source_target=contract.train.y,
            source_row_ids=contract.train.row_ids,
            sampler_result=sampler_result,
        )

        sample_weight_result = (
            self._calculate_sample_weights(
                training=training_output,
                lineage=sampler_result.row_lineage,
                class_weights=class_weight_result,
                plan=plan,
            )
        )

        distribution_after = (
            ClassDistributionAnalyzer(
                options=self.options.distribution
            ).analyze(
                target=training_output.y
            )
        )

        metrics_after = ImbalanceMetricsAnalyzer(
            options=self.options.metrics
        ).analyze(
            distribution=distribution_after
        )

        validation_guard_result = (
            self.partition_guard.verify(
                partition=validation_output,
                context=validation_context,
            )
        )

        test_guard_result = (
            self.partition_guard.verify(
                partition=test_output,
                context=test_context,
            )
        )

        if (
            plan.partition_protection.protect_validation
            and not validation_guard_result.valid
        ):
            raise ValueError(
                "The validation partition changed during "
                "Phase 8."
            )

        if (
            plan.partition_protection.protect_test
            and not test_guard_result.valid
        ):
            raise ValueError(
                "The test partition changed during Phase 8."
            )

        pipeline_result = BalancingPipelineResult(
            distribution_before=(
                distribution_before
            ),
            distribution_after=(
                distribution_after
            ),
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            recommendation=recommendation,
            configured_strategy=(
                plan.sampling.strategy
            ),
            executed_strategy=(
                executed_strategy
            ),
            sampler_state=(
                sampler_result.state
            ),
            class_weights=(
                class_weight_result
            ),
            sample_weights=(
                sample_weight_result
            ),
            row_lineage=lineage,
            validation_guard=(
                validation_guard_result
            ),
            test_guard=test_guard_result,
            fitted_on_training_only=True,
            validation_resampled=False,
            test_resampled=False,
            validation_remained_protected=(
                validation_guard_result.valid
            ),
            test_remained_protected=(
                test_guard_result.valid
            ),
            target_access_authorized=(
                plan.target_access_authorized
            ),
            resampling_authorized=(
                plan.resampling_authorized
            ),
            synthetic_generation_authorized=(
                plan
                .synthetic_generation_authorized
            ),
            metadata={
                "execution_id": self.execution_id,
                "source_training_row_count": (
                    contract.train.row_count()
                ),
                "output_training_row_count": (
                    training_output.row_count()
                ),
                "source_feature_count": (
                    contract.train.feature_count()
                ),
                "output_feature_count": (
                    training_output.feature_count()
                ),
                "fallback_used": (
                    executed_strategy
                    != plan.sampling.strategy
                ),
            },
        )

        self.result = pipeline_result
        self.executed_strategy = executed_strategy

        return (
            training_output,
            validation_output,
            test_output,
            pipeline_result.model_copy(
                deep=True
            ),
        )

    def get_result(
        self,
    ) -> BalancingPipelineResult:
        """Return the latest balancing-pipeline result."""

        if self.result is None:
            raise RuntimeError(
                "The balancing pipeline has not executed."
            )

        return self.result.model_copy(
            deep=True
        )

    def _execute_sampling(
        self,
        *,
        contract: BalancingContract,
        plan: BalancingPlan,
    ) -> tuple[
        SamplerResult,
        BalancingStrategy,
    ]:
        """Execute the configured training sampler."""

        strategy = plan.sampling.strategy

        if strategy in {
            BalancingStrategy.NONE,
            BalancingStrategy.CLASS_WEIGHTS_ONLY,
        }:
            return (
                self._identity_result(
                    partition=contract.train,
                    strategy=strategy,
                ),
                strategy,
            )

        sampling_strategy = (
            self._resolve_sampling_strategy(
                plan
            )
        )

        sampler = self._build_sampler(
            plan=plan,
            strategy=strategy,
            sampling_strategy=sampling_strategy,
        )

        try:
            result = sampler.fit_resample(
                features=contract.train.X,
                target=contract.train.y,
                row_ids=contract.train.row_ids,
                context=(
                    SamplerExecutionContext
                    .TRAINING_RESAMPLING
                ),
                resampling_authorized=(
                    plan.resampling_authorized
                ),
                target_access_authorized=(
                    plan.target_access_authorized
                ),
                synthetic_generation_authorized=(
                    plan
                    .synthetic_generation_authorized
                ),
            )

            return (
                result,
                strategy,
            )

        except (
            ValueError,
            RuntimeError,
        ):
            if not self.options.allow_sampler_fallback:
                raise

            if (
                not plan.synthetic_sampling_enabled
                or plan.neighbors.shortage_policy
                != NeighborShortagePolicy
                .RANDOM_OVERSAMPLING
            ):
                raise

            fallback = RandomOverSampler(
                sampling_strategy=(
                    sampling_strategy
                ),
                random_seed=(
                    plan.sampling.random_seed
                ),
                synthetic_prefix=(
                    plan.row_identity
                    .synthetic_prefix
                ),
                duplicate_prefix=(
                    plan.row_identity
                    .duplicate_prefix
                ),
                include_strategy_in_identifier=(
                    plan.row_identity
                    .include_strategy_in_identifier
                ),
                deterministic_row_ids=(
                    plan.row_identity.deterministic
                ),
            )

            fallback_result = (
                fallback.fit_resample(
                    features=contract.train.X,
                    target=contract.train.y,
                    row_ids=contract.train.row_ids,
                    context=(
                        SamplerExecutionContext
                        .TRAINING_RESAMPLING
                    ),
                    resampling_authorized=(
                        plan.resampling_authorized
                    ),
                    target_access_authorized=(
                        plan.target_access_authorized
                    ),
                    synthetic_generation_authorized=False,
                )
            )

            fallback_result.metadata[
                "fallback_from"
            ] = strategy.value

            return (
                fallback_result,
                BalancingStrategy
                .RANDOM_OVERSAMPLING,
            )

    def _build_sampler(
        self,
        *,
        plan: BalancingPlan,
        strategy: BalancingStrategy,
        sampling_strategy: str | dict[Any, int],
    ) -> object:
        """Build the configured sampler."""

        common = {
            "sampling_strategy": (
                sampling_strategy
            ),
            "random_seed": (
                plan.sampling.random_seed
            ),
            "synthetic_prefix": (
                plan.row_identity.synthetic_prefix
            ),
            "duplicate_prefix": (
                plan.row_identity.duplicate_prefix
            ),
            "include_strategy_in_identifier": (
                plan.row_identity
                .include_strategy_in_identifier
            ),
            "deterministic_row_ids": (
                plan.row_identity.deterministic
            ),
        }

        if (
            strategy
            == BalancingStrategy.RANDOM_OVERSAMPLING
        ):
            return RandomOverSampler(
                **common
            )

        if (
            strategy
            == BalancingStrategy.RANDOM_UNDERSAMPLING
        ):
            return RandomUnderSampler(
                **common
            )

        if strategy == BalancingStrategy.SMOTE:
            return SmoteSampler(
                nearest_neighbors=(
                    plan.neighbors.nearest_neighbors
                ),
                **common,
            )

        if (
            strategy
            == BalancingStrategy.BORDERLINE_SMOTE
        ):
            return BorderlineSmoteSampler(
                nearest_neighbors=(
                    plan.neighbors.nearest_neighbors
                ),
                method=(
                    plan.borderline_smote.method
                ),
                danger_neighbors=(
                    plan
                    .borderline_smote
                    .danger_neighbors
                ),
                **common,
            )

        if strategy == BalancingStrategy.ADASYN:
            return AdasynSampler(
                density_neighbors=(
                    plan.adasyn.density_neighbors
                ),
                **common,
            )

        if strategy in {
            BalancingStrategy.SMOTE_TOMEK,
            BalancingStrategy.SMOTE_ENN,
        }:
            cleaning = (
                HybridCleaningKind.TOMEK
                if strategy
                == BalancingStrategy.SMOTE_TOMEK
                else HybridCleaningKind.ENN
            )

            return HybridSampler(
                cleaning=cleaning,
                nearest_neighbors=(
                    plan.neighbors.nearest_neighbors
                ),
                enn_neighbors=(
                    plan.hybrid.enn_neighbors
                ),
                **common,
            )

        raise ValueError(
            f"Unsupported balancing strategy: "
            f"{strategy.value!r}"
        )

    def _calculate_class_weights(
        self,
        *,
        source_target: pd.Series,
        class_identity_mapping: ClassIdentityMapping,
        plan: BalancingPlan,
    ) -> ClassWeightResult | None:
        """Calculate weights from the original training target."""

        if not plan.class_weighting_enabled:
            return None

        context = BalancingFitContext.training_weighting(
            target_access_authorized=(
                plan.target_access_authorized
            )
        )

        context.assert_weighting_allowed()

        options = ClassWeightOptions(
            enabled=True,
            method=plan.class_weights.method,
            normalization=(
                plan.class_weights.normalization
            ),
            custom_weights=dict(
                plan.class_weights.custom_weights
            ),
            effective_number_beta=(
                plan.class_weights
                .effective_number_beta
            ),
        )

        calculator = ClassWeightCalculator(
            options=options
        )

        calculator.fit(
            target=source_target,
            target_access_authorized=(
                plan.target_access_authorized
            ),
            class_identity_mapping=class_identity_mapping,
        )

        return calculator.get_result()

    def _calculate_sample_weights(
        self,
        *,
        training: DatasetPartition,
        lineage: list[SamplerRowRecord],
        class_weights: ClassWeightResult | None,
        plan: BalancingPlan,
    ) -> SampleWeightResult | None:
        """Map class weights to final training rows."""

        if (
            class_weights is None
            or not plan.class_weights
            .generate_sample_weights
        ):
            return None

        generator = SampleWeightGenerator(
            options=self.options.sample_weights
        )

        return generator.generate(
            target=training.y,
            row_ids=training.row_ids,
            class_weights=class_weights,
            row_lineage=lineage,
            target_access_authorized=(
                plan.target_access_authorized
            ),
        )

    @staticmethod
    def _resolve_sampling_strategy(
        plan: BalancingPlan,
    ) -> str | dict[Any, int]:
        """Convert the plan sampling policy."""

        if (
            plan.sampling.sampling_strategy
            == SamplingStrategyKind.CUSTOM
        ):
            source_values = (
                plan.source_class_counts
            )

            unknown = set(
                plan.sampling.custom_target_counts
            ).difference(
                source_values
            )

            if unknown:
                raise ValueError(
                    "Custom target counts contain unknown "
                    f"class keys: {sorted(unknown)}"
                )

            return dict(
                plan.sampling.custom_target_counts
            )

        return (
            plan.sampling
            .sampling_strategy
            .value
        )

    @staticmethod
    def _identity_result(
        *,
        partition: DatasetPartition,
        strategy: BalancingStrategy,
    ) -> SamplerResult:
        """Build an unchanged training sampler result."""

        features = partition.X.copy(
            deep=True
        ).reset_index(
            drop=True
        )

        target = partition.y.copy(
            deep=True
        ).reset_index(
            drop=True
        )

        target.name = partition.y.name

        row_ids = pd.Series(
            [
                str(value)
                for value
                in partition.row_ids.tolist()
            ],
            name=partition.row_ids.name,
            dtype="string",
        )

        identity_mapping = build_class_identity_mapping(
            target,
            reject_missing=False,
        )
        class_counts: dict[str, int] = {}
        for value in target:
            normalized = normalize_class_value(value)
            if normalized is None:
                continue
            key = identity_mapping.key_for(normalized)
            class_counts[key] = class_counts.get(key, 0) + 1

        lineage = [
            SamplerRowRecord(
                output_position=position,
                output_row_id=str(
                    partition.row_ids.iloc[
                        position
                    ]
                ),
                row_type=(
                    ResampledRowType.ORIGINAL
                ),
                class_value=(
                    partition.y.iloc[
                        position
                    ]
                ),
                source_row_ids=[
                    str(
                        partition.row_ids.iloc[
                            position
                        ]
                    )
                ],
                source_positions=[
                    position
                ],
                original=True,
            )
            for position in range(
                partition.row_count()
            )
        ]

        state = SamplerState(
            sampler="identity_sampler",
            strategy=strategy.value,
            fitted=True,
            fitted_on_training_only=True,
            resampling_authorized=True,
            target_access_authorized=True,
            source_feature_columns=(
                partition.feature_names()
            ),
            source_feature_dtypes={
                str(column): str(
                    partition.X[
                        column
                    ].dtype
                )
                for column
                in partition.X.columns
            },
            source_row_count=(
                partition.row_count()
            ),
            output_row_count=(
                partition.row_count()
            ),
            class_counts_before=dict(class_counts),
            class_counts_after=dict(class_counts),
            original_output_count=(
                partition.row_count()
            ),
        )

        return SamplerResult(
            features=features,
            target=target,
            row_ids=row_ids,
            row_lineage=lineage,
            removed_rows=[],
            state=state,
            metadata={
                "identity_result": True,
            },
        )
