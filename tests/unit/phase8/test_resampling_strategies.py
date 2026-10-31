from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase2.output_governance.result import DatasetPartition
from mlprep.phases.phase8.contract.balancing_contract import (
    BalancingContract,
    BalancingContractOptions,
)
from mlprep.phases.phase8.contract.balancing_plan import (
    BalancingPlanBuilder,
    BalancingPlanOptions,
    BalancingStrategy,
    HybridCleaningKind,
    NeighborShortagePolicy,
    SamplingStrategyKind,
)
from mlprep.phases.phase8.orchestration.balancing_pipeline import (
    BalancingPipeline,
)
from mlprep.phases.phase8.samplers import (
    AdasynSampler,
    BorderlineSmoteSampler,
    HybridSampler,
    RandomOverSampler,
    RandomUnderSampler,
    ResampledRowType,
    SamplerExecutionContext,
    SmoteSampler,
)
from mlprep.phases.phase8.utils.class_identity import (
    build_class_identity_mapping,
    class_key,
)


def _training_data(
    class_counts: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(42)
    target_values = np.concatenate(
        [
            np.full(count, class_id, dtype=np.int64)
            for class_id, count in enumerate(class_counts)
        ]
    )
    centers = np.linspace(0.0, 0.3, len(class_counts))
    features = np.vstack(
        [
            rng.normal(
                loc=center,
                scale=1.0,
                size=(count, 2),
            )
            for center, count in zip(centers, class_counts, strict=True)
        ]
    )
    target = pd.Series(target_values, name="target")
    frame = pd.DataFrame(
        features,
        columns=["feature_a", "feature_b"],
    )
    row_ids = pd.Series(
        [f"train-{position}" for position in range(len(target))],
        dtype="string",
        name="row_id",
    )
    return frame, target, row_ids


def _samplers(
    *,
    random_seed: int,
) -> list[tuple[str, object]]:
    return [
        (
            "random_oversampling",
            RandomOverSampler(random_seed=random_seed),
        ),
        (
            "random_undersampling",
            RandomUnderSampler(random_seed=random_seed),
        ),
        (
            "smote",
            SmoteSampler(nearest_neighbors=3, random_seed=random_seed),
        ),
        (
            "borderline_smote",
            BorderlineSmoteSampler(
                nearest_neighbors=3,
                danger_neighbors=5,
                random_seed=random_seed,
            ),
        ),
        (
            "adasyn",
            AdasynSampler(
                density_neighbors=3,
                random_seed=random_seed,
            ),
        ),
        (
            "smote_tomek",
            HybridSampler(
                cleaning=HybridCleaningKind.TOMEK,
                nearest_neighbors=3,
                random_seed=random_seed,
            ),
        ),
        (
            "smote_enn",
            HybridSampler(
                cleaning=HybridCleaningKind.ENN,
                nearest_neighbors=3,
                enn_neighbors=3,
                random_seed=random_seed,
            ),
        ),
    ]


def _sample(sampler: object, *, features: pd.DataFrame, target: pd.Series,
            row_ids: pd.Series):
    return sampler.fit_resample(
        features=features,
        target=target,
        row_ids=row_ids,
        context=SamplerExecutionContext.TRAINING_RESAMPLING,
        resampling_authorized=True,
        target_access_authorized=True,
        synthetic_generation_authorized=True,
    )


@pytest.mark.parametrize(
    "class_counts",
    [
        pytest.param((60, 20), id="binary"),
        pytest.param((60, 30, 10), id="multiclass"),
    ],
)
@pytest.mark.parametrize(
    "sampler_name",
    [
        "random_oversampling",
        "random_undersampling",
        "smote",
        "borderline_smote",
        "adasyn",
        "smote_tomek",
        "smote_enn",
    ],
)
def test_resampling_strategies_support_binary_and_multiclass(
    class_counts: tuple[int, ...],
    sampler_name: str,
) -> None:
    features, target, row_ids = _training_data(class_counts)
    sampler = dict(_samplers(random_seed=17))[sampler_name]

    result = _sample(
        sampler,
        features=features,
        target=target,
        row_ids=row_ids,
    )

    assert len(result.features) == len(result.target) == len(result.row_ids)
    assert result.state.fitted
    assert result.state.fitted_on_training_only
    assert len(result.state.class_counts_before) == len(class_counts)
    assert len(result.row_lineage) == len(result.features)
    assert result.row_ids.is_unique
    assert {
        record.output_position for record in result.row_lineage
    } == set(range(len(result.features)))
    if sampler_name == "random_undersampling":
        assert result.removed_rows
        assert result.state.removed_original_count == len(result.removed_rows)
    elif sampler_name == "random_oversampling":
        assert any(
            record.row_type == ResampledRowType.DUPLICATE
            for record in result.row_lineage
        )
    else:
        assert any(record.synthetic for record in result.row_lineage)


@pytest.mark.parametrize(
    "sampler_name",
    [
        "random_oversampling",
        "random_undersampling",
        "smote",
        "borderline_smote",
        "adasyn",
        "smote_tomek",
        "smote_enn",
    ],
)
def test_resampling_strategy_outputs_are_deterministic_for_same_seed(
    sampler_name: str,
) -> None:
    features, target, row_ids = _training_data((60, 30, 10))
    first_sampler = dict(_samplers(random_seed=23))[sampler_name]
    second_sampler = dict(_samplers(random_seed=23))[sampler_name]

    first = _sample(
        first_sampler,
        features=features,
        target=target,
        row_ids=row_ids,
    )
    second = _sample(
        second_sampler,
        features=features,
        target=target,
        row_ids=row_ids,
    )

    pd.testing.assert_frame_equal(first.features, second.features)
    pd.testing.assert_series_equal(first.target, second.target)
    assert first.row_ids.tolist() == second.row_ids.tolist()
    assert [
        record.model_dump(mode="json") for record in first.row_lineage
    ] == [
        record.model_dump(mode="json") for record in second.row_lineage
    ]
    assert [
        record.row_id for record in first.removed_rows
    ] == [
        record.row_id for record in second.removed_rows
    ]


@pytest.mark.parametrize(
    "cleaning",
    [HybridCleaningKind.TOMEK, HybridCleaningKind.ENN],
)
def test_hybrid_sampling_tracks_synthetic_and_removed_row_lineage(
    cleaning: HybridCleaningKind,
) -> None:
    features, target, row_ids = _training_data((60, 30, 10))
    sampler = HybridSampler(
        cleaning=cleaning,
        nearest_neighbors=3,
        enn_neighbors=3,
        random_seed=11,
    )

    result = _sample(
        sampler,
        features=features,
        target=target,
        row_ids=row_ids,
    )

    generated_ids = {
        record.output_row_id
        for record in result.row_lineage
        if record.synthetic
    }
    removed_ids = {record.row_id for record in result.removed_rows}

    assert generated_ids
    assert result.row_ids.is_unique
    assert generated_ids.isdisjoint(set(row_ids))
    assert removed_ids
    assert removed_ids.isdisjoint(set(result.row_ids))
    assert result.state.synthetic_output_count == len(generated_ids)
    assert result.state.removed_original_count == len(removed_ids)


def test_random_oversampling_respects_custom_target_counts() -> None:
    features, target, row_ids = _training_data((60, 20))
    sampler = RandomOverSampler(
        sampling_strategy={class_key(1): 25},
        random_seed=19,
    )

    result = _sample(
        sampler,
        features=features,
        target=target,
        row_ids=row_ids,
    )

    assert Counter(result.target.tolist()) == {0: 60, 1: 25}


def _pipeline_contract(
    class_counts: tuple[int, ...],
) -> BalancingContract:
    features, target, row_ids = _training_data(class_counts)

    def partition(
        name: PartitionName,
        row_prefix: str,
        row_count: int,
    ) -> DatasetPartition:
        indices = pd.RangeIndex(row_count)
        return DatasetPartition(
            name=name,
            X=pd.DataFrame(
                {
                    "feature_a": np.linspace(0.1, 0.9, row_count),
                    "feature_b": np.linspace(1.0, 2.0, row_count),
                },
                index=indices,
            ),
            y=pd.Series(
                np.tile([0, 1], (row_count + 1) // 2)[:row_count],
                index=indices,
                name="target",
            ),
            row_ids=pd.Series(
                [f"{row_prefix}-{position}" for position in range(row_count)],
                index=indices,
                dtype="string",
                name="row_id",
            ),
        )

    identity_mapping = build_class_identity_mapping(target)
    class_counts_by_identity: Counter[str] = Counter(
        identity_mapping.key_for(value) for value in target
    )
    return BalancingContract(
        train=DatasetPartition(
            name=PartitionName.TRAIN,
            X=features,
            y=target,
            row_ids=row_ids,
        ),
        validation=partition(PartitionName.VALIDATION, "validation", 12),
        test=partition(PartitionName.TEST, "test", 12),
        phase7_state=object(),
        training_class_counts=dict(class_counts_by_identity),
        class_identity_mapping=identity_mapping,
        target_dtype=str(target.dtype),
        issues=[],
        options=BalancingContractOptions(),
    )


def _pipeline_plan(
    contract: BalancingContract,
    *,
    strategy: BalancingStrategy = BalancingStrategy.SMOTE,
    nearest_neighbors: int = 5,
    shortage_policy: NeighborShortagePolicy = (
        NeighborShortagePolicy.REDUCE_NEIGHBORS
    ),
    sampling_strategy: SamplingStrategyKind = SamplingStrategyKind.AUTO,
    custom_target_counts: dict[str, int] | None = None,
) :
    return BalancingPlanBuilder(
        BalancingPlanOptions(
            strategy=strategy,
            sampling_strategy=sampling_strategy,
            custom_target_counts=custom_target_counts or {},
            nearest_neighbors=nearest_neighbors,
            minimum_effective_neighbors=1,
            neighbor_shortage_policy=shortage_policy,
            class_weights_enabled=False,
            allow_original_row_removal=True,
            resampling_authorized=True,
            target_access_authorized=True,
            synthetic_generation_authorized=True,
            random_seed=31,
        )
    ).build(contract=contract)


def test_small_minority_class_reduces_neighbor_count() -> None:
    contract = _pipeline_contract((12, 3))
    plan = _pipeline_plan(contract, nearest_neighbors=5)

    assert plan.neighbors.nearest_neighbors == 2
    assert any("nearest-neighbor count was reduced" in note for note in plan.warnings)

    training, _, _, result = BalancingPipeline().fit_resample(
        contract=contract,
        plan=plan,
    )

    assert len(training.X) > contract.train.row_count()
    assert result.executed_strategy == BalancingStrategy.SMOTE


def test_small_minority_class_falls_back_to_random_oversampling() -> None:
    contract = _pipeline_contract((12, 2))
    plan = _pipeline_plan(
        contract,
        nearest_neighbors=5,
        shortage_policy=NeighborShortagePolicy.RANDOM_OVERSAMPLING,
    )

    training, _, _, result = BalancingPipeline().fit_resample(
        contract=contract,
        plan=plan,
    )

    assert result.executed_strategy == BalancingStrategy.RANDOM_OVERSAMPLING
    assert result.sampler_state.sampler == "random_oversampler"
    assert result.metadata["fallback_used"] is True
    assert len(training.X) > contract.train.row_count()


def test_small_minority_class_blocks_when_fallback_policy_is_block() -> None:
    contract = _pipeline_contract((12, 2))
    plan = _pipeline_plan(
        contract,
        nearest_neighbors=5,
        shortage_policy=NeighborShortagePolicy.BLOCK,
    )

    assert "nearest_neighbors" in plan.blocked_components
    with pytest.raises(PermissionError, match="blocked"):
        BalancingPipeline().fit_resample(
            contract=contract,
            plan=plan,
        )


def test_balancing_pipeline_preserves_validation_and_test_partitions() -> None:
    contract = _pipeline_contract((60, 20))
    validation_before = contract.validation.copy(deep=True)
    test_before = contract.test.copy(deep=True)
    plan = _pipeline_plan(contract)

    _, validation_output, test_output, result = (
        BalancingPipeline().fit_resample(
            contract=contract,
            plan=plan,
        )
    )

    pd.testing.assert_frame_equal(contract.validation.X, validation_before.X)
    pd.testing.assert_series_equal(contract.validation.y, validation_before.y)
    pd.testing.assert_series_equal(
        contract.validation.row_ids,
        validation_before.row_ids,
    )
    pd.testing.assert_frame_equal(contract.test.X, test_before.X)
    pd.testing.assert_series_equal(contract.test.y, test_before.y)
    pd.testing.assert_series_equal(contract.test.row_ids, test_before.row_ids)
    pd.testing.assert_frame_equal(validation_output.X, validation_before.X)
    pd.testing.assert_frame_equal(test_output.X, test_before.X)
    assert result.validation_remained_protected
    assert result.test_remained_protected
    assert not result.validation_resampled
    assert not result.test_resampled


@pytest.mark.parametrize(
    ("context", "resampling_authorized", "target_access_authorized",
     "synthetic_generation_authorized", "message"),
    [
        (
            SamplerExecutionContext.VALIDATION,
            True,
            True,
            True,
            "training-resampling context",
        ),
        (
            SamplerExecutionContext.TRAINING_RESAMPLING,
            False,
            True,
            True,
            "not authorized",
        ),
        (
            SamplerExecutionContext.TRAINING_RESAMPLING,
            True,
            False,
            True,
            "Training-target access",
        ),
        (
            SamplerExecutionContext.TRAINING_RESAMPLING,
            True,
            True,
            False,
            "Synthetic-row generation",
        ),
    ],
)
def test_synthetic_sampler_rejects_unauthorized_execution(
    context: SamplerExecutionContext,
    resampling_authorized: bool,
    target_access_authorized: bool,
    synthetic_generation_authorized: bool,
    message: str,
) -> None:
    features, target, row_ids = _training_data((60, 20))
    with pytest.raises(PermissionError, match=message):
        SmoteSampler(nearest_neighbors=3).fit_resample(
            features=features,
            target=target,
            row_ids=row_ids,
            context=context,
            resampling_authorized=resampling_authorized,
            target_access_authorized=target_access_authorized,
            synthetic_generation_authorized=synthetic_generation_authorized,
        )
