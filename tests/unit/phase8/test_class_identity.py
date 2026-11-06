from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase2.output_governance.result import DatasetPartition
from mlprep.phases.phase8.analysis.class_distribution import (
    ClassDistributionAnalyzer,
)
from mlprep.phases.phase8.contract.balancing_contract import (
    BalancingContract,
    BalancingContractOptions,
    BalancingContractResolver,
)
from mlprep.phases.phase8.contract.balancing_plan import (
    BalancingPlanBuilder,
    BalancingPlanOptions,
    BalancingStrategy,
    ClassWeightMethod,
    SamplingStrategyKind,
    WeightNormalizationKind,
)
from mlprep.phases.phase8.lineage import (
    RowLineageEntry,
    RowLineageState,
    SyntheticIdGenerator,
)
from mlprep.phases.phase8.orchestration.balancing_pipeline import (
    BalancingPipeline,
)
from mlprep.phases.phase8.output_governance.fingerprint import (
    Phase8FingerprintManager,
)
from mlprep.phases.phase8.output_governance.state import Phase8State
from mlprep.phases.phase8.phase import Phase8
from mlprep.phases.phase8.samplers import (
    RandomOverSampler,
    ResampledRowType,
    SamplerExecutionContext,
    SamplerRowRecord,
)
from mlprep.phases.phase8.utils.class_identity import (
    build_class_identity,
    build_class_identity_mapping,
    class_display_value,
    class_key,
    class_type_name,
    class_values_equal,
    normalize_class_value,
    resolve_class_count_mapping,
    resolve_class_weight_mapping,
    serialize_class_value,
    validate_class_identity_mapping,
)
from mlprep.phases.phase8.weighting import (
    ClassWeightCalculator,
    SampleWeightEntry,
    SampleWeightGenerator,
    SampleWeightResult,
)


def _mixed_target() -> pd.Series:
    return pd.Series(
        [1, 1, "1", 1.0, 1.0, 1.0, True],
        dtype=object,
        name="target",
    )


def _identity_counts(
    target: pd.Series,
    identity_mapping,
) -> dict[str, int]:
    counts: Counter[str] = Counter(
        identity_mapping.key_for(value)
        for value in target
    )
    return dict(counts)


def test_class_distribution_preserves_typed_class_identities() -> None:
    target = _mixed_target()
    result = ClassDistributionAnalyzer().analyze(target=target)
    identities = build_class_identity_mapping(target)

    assert result.class_count == 4
    assert BalancingContractResolver._training_class_counts(
        target=target,
        class_identity_mapping=identities,
    ) == _identity_counts(target, identities)
    assert {
        key: result.classes[key].count
        for key in result.class_order
    } == _identity_counts(target, identities)
    assert len(set(identities.class_order)) == 4


def test_class_weights_and_sample_weights_use_canonical_identity_keys() -> None:
    target = _mixed_target()
    identities = build_class_identity_mapping(target)
    row_ids = pd.Series(
        [f"row-{position}" for position in range(len(target))],
        dtype="string",
    )
    weights = ClassWeightCalculator().fit(
        target=target,
        target_access_authorized=True,
        class_identity_mapping=identities,
    ).get_result()

    assert weights.class_counts == _identity_counts(target, identities)
    assert set(weights.class_weights) == set(identities.class_order)

    lineage = [
        SamplerRowRecord(
            output_position=position,
            output_row_id=str(row_ids.iloc[position]),
            row_type=ResampledRowType.ORIGINAL,
            class_value=value,
            source_row_ids=[str(row_ids.iloc[position])],
            source_positions=[position],
            original=True,
        )
        for position, value in enumerate(target)
    ]
    sample_weights = SampleWeightGenerator().generate(
        target=target,
        row_ids=row_ids,
        class_weights=weights,
        row_lineage=lineage,
        target_access_authorized=True,
    )

    assert len(sample_weights.entries) == len(target)
    for position, value in enumerate(target):
        assert sample_weights.entries[position].class_key == class_key(value)
        assert type(sample_weights.entries[position].class_value) is type(value)


def test_random_oversampler_round_trips_typed_classes_and_custom_targets() -> None:
    target = _mixed_target()
    identities = build_class_identity_mapping(target)
    features = pd.DataFrame(
        {"feature": range(len(target))},
        index=target.index,
    )
    row_ids = pd.Series(
        [f"row-{position}" for position in range(len(target))],
        index=target.index,
        dtype="string",
    )
    sampler = RandomOverSampler(
        sampling_strategy={
            class_key(True): 2,
            class_key("1"): 2,
        },
        random_seed=17,
    )

    result = sampler.fit_resample(
        features=features,
        target=target,
        row_ids=row_ids,
        context=SamplerExecutionContext.TRAINING_RESAMPLING,
        resampling_authorized=True,
        target_access_authorized=True,
    )

    assert _identity_counts(result.target, identities) == {
        class_key(1): 2,
        class_key("1"): 2,
        class_key(1.0): 3,
        class_key(True): 2,
    }
    assert result.state.class_counts_after == _identity_counts(
        result.target,
        identities,
    )
    assert any(
        record.row_type == ResampledRowType.DUPLICATE
        and type(record.class_value) is bool
        for record in result.row_lineage
    )


def test_random_oversampler_ids_are_deterministic_for_same_seed_and_input() -> None:
    target = pd.Series(
        ["minority", "majority", "majority", "majority"],
        dtype=object,
        name="target",
    )
    features = pd.DataFrame(
        {"feature": [0.0, 1.0, 2.0, 3.0]}
    )
    row_ids = pd.Series(
        ["row-a", "row-b", "row-c", "row-d"],
        dtype="string",
    )

    def sample(seed: int):
        return RandomOverSampler(
            sampling_strategy="auto",
            random_seed=seed,
        ).fit_resample(
            features=features,
            target=target,
            row_ids=row_ids,
            context=SamplerExecutionContext.TRAINING_RESAMPLING,
            resampling_authorized=True,
            target_access_authorized=True,
        )

    first = sample(23)
    repeat = sample(23)
    different_seed = sample(24)
    nondeterministic_first = RandomOverSampler(
        sampling_strategy="auto",
        random_seed=23,
        deterministic_row_ids=False,
    ).fit_resample(
        features=features,
        target=target,
        row_ids=row_ids,
        context=SamplerExecutionContext.TRAINING_RESAMPLING,
        resampling_authorized=True,
        target_access_authorized=True,
    )
    nondeterministic_repeat = RandomOverSampler(
        sampling_strategy="auto",
        random_seed=23,
        deterministic_row_ids=False,
    ).fit_resample(
        features=features,
        target=target,
        row_ids=row_ids,
        context=SamplerExecutionContext.TRAINING_RESAMPLING,
        resampling_authorized=True,
        target_access_authorized=True,
    )

    assert first.row_ids.tolist() == repeat.row_ids.tolist()
    assert first.row_ids.tolist() != different_seed.row_ids.tolist()
    id_state = first.state.metadata["synthetic_id_state"]
    assert id_state["deterministic"] is True
    nondeterministic_state = nondeterministic_first.state.metadata[
        "synthetic_id_state"
    ]
    assert nondeterministic_state["deterministic"] is False
    assert (
        nondeterministic_first.row_ids.tolist()
        != nondeterministic_repeat.row_ids.tolist()
    )
    assert len(set(first.row_ids.tolist())) == len(first.row_ids)


def test_generated_row_id_collision_uses_deterministic_retry() -> None:
    initial = SyntheticIdGenerator(
        execution_id="stable-execution",
        random_seed=31,
        existing_row_ids=[],
    )
    colliding_id = initial.next_synthetic_id(
        strategy="smote",
    )
    generator = SyntheticIdGenerator(
        execution_id="stable-execution",
        random_seed=31,
        existing_row_ids=[colliding_id],
    )

    generated_id = generator.next_synthetic_id(
        strategy="smote",
    )

    assert generated_id != colliding_id
    assert generator.get_state().collision_count == 1
    assert generator.get_state().generated_records[0].collision_attempts == 1
    assert generator.get_state().generated_records[0].deterministic is True


def test_identity_sampler_state_keeps_typed_class_counts() -> None:
    target = _mixed_target()
    identities = build_class_identity_mapping(target)
    partition = DatasetPartition(
        name=PartitionName.TRAIN,
        X=pd.DataFrame({"feature": range(len(target))}),
        y=target,
        row_ids=pd.Series(
            [f"row-{position}" for position in range(len(target))],
            dtype="string",
        ),
    )

    result = BalancingPipeline._identity_result(
        partition=partition,
        strategy=BalancingStrategy.NONE,
    )

    assert result.state.class_counts_before == _identity_counts(
        target,
        identities,
    )
    assert result.state.class_counts_after == result.state.class_counts_before


def test_balancing_pipeline_preserves_custom_typed_class_targets() -> None:
    target = _mixed_target()
    identities = build_class_identity_mapping(target)

    def partition(
        name: PartitionName,
        row_prefix: str,
        labels: pd.Series,
    ) -> DatasetPartition:
        return DatasetPartition(
            name=name,
            X=pd.DataFrame(
                {"feature": range(len(labels))},
                index=labels.index,
            ),
            y=labels,
            row_ids=pd.Series(
                [f"{row_prefix}-{position}" for position in range(len(labels))],
                index=labels.index,
                dtype="string",
            ),
        )

    train = partition(PartitionName.TRAIN, "train", target)
    validation = partition(
        PartitionName.VALIDATION,
        "validation",
        pd.Series([1, "1", 1.0, True], dtype=object, name="target"),
    )
    test = partition(
        PartitionName.TEST,
        "test",
        pd.Series([1, "1", 1.0, True], dtype=object, name="target"),
    )
    contract = BalancingContract(
        train=train,
        validation=validation,
        test=test,
        phase7_state=object(),
        training_class_counts=_identity_counts(target, identities),
        class_identity_mapping=identities,
        target_dtype=str(target.dtype),
        issues=[],
        options=BalancingContractOptions(),
    )
    plan = BalancingPlanBuilder(
        BalancingPlanOptions(
            strategy=BalancingStrategy.RANDOM_OVERSAMPLING,
            sampling_strategy=SamplingStrategyKind.CUSTOM,
            custom_target_counts={
                class_key(True): 2,
                class_key("1"): 2,
            },
            class_weight_method=ClassWeightMethod.CUSTOM,
            custom_class_weights={
                class_key(1): 1.0,
                class_key("1"): 2.0,
                class_key(1.0): 3.0,
                class_key(True): 4.0,
            },
            class_weight_normalization=WeightNormalizationKind.NONE,
            resampling_authorized=True,
            target_access_authorized=True,
        )
    ).build(contract=contract)

    output_train, _, _, pipeline_result = BalancingPipeline().fit_resample(
        contract=contract,
        plan=plan,
    )

    assert _identity_counts(output_train.y, identities) == {
        class_key(1): 2,
        class_key("1"): 2,
        class_key(1.0): 3,
        class_key(True): 2,
    }
    assert pipeline_result.class_weights is not None
    assert pipeline_result.class_weights.class_counts == _identity_counts(
        target,
        identities,
    )
    assert pipeline_result.class_weights.class_weights == {
        class_key(1): 1.0,
        class_key("1"): 2.0,
        class_key(1.0): 3.0,
        class_key(True): 4.0,
    }
    assert pipeline_result.sample_weights is not None
    assert [
        type(entry.class_value)
        for entry in pipeline_result.sample_weights.entries
    ] == [
        type(value)
        for value in output_train.y
    ]
    assert [
        entry.weight
        for entry in pipeline_result.sample_weights.entries
    ] == [
        pipeline_result.class_weights.class_weights[class_key(value)]
        for value in output_train.y
    ]
    phase_state = Phase8State()
    Phase8._register_pipeline_state(
        state=phase_state,
        pipeline_result=pipeline_result,
    )
    assert phase_state.synthetic_id_state is not None
    assert (
        phase_state.synthetic_id_state.generated_ids
        == pipeline_result.sampler_state.metadata[
            "synthetic_id_state"
        ]["generated_ids"]
    )


def test_ambiguous_custom_class_alias_is_rejected() -> None:
    identities = build_class_identity_mapping(_mixed_target())

    with pytest.raises(ValueError, match="ambiguous"):
        BalancingPlanBuilder._normalize_class_mapping(
            configured={"1": 2},
            identity_mapping=identities,
        )


def test_original_value_mapping_rejects_python_equal_class_collisions() -> None:
    identities = build_class_identity_mapping([1, True])

    with pytest.raises(ValueError, match="cannot be represented"):
        identities.resolve_count_mapping(
            {
                class_key(1): 3,
                class_key(True): 4,
            }
        )


def test_class_identity_mapping_survives_phase_state_round_trip() -> None:
    identities = build_class_identity_mapping(_mixed_target())
    state = Phase8State(class_identity_mapping=identities)

    restored = Phase8State.model_validate(
        state.model_dump(mode="json")
    )

    assert restored.class_identity_mapping is not None
    assert restored.class_identity_mapping.class_order == identities.class_order
    assert {
        key: type(restored.class_identity_mapping.value_for(key))
        for key in identities.class_order
    } == {
        class_key(1): int,
        class_key("1"): str,
        class_key(1.0): float,
        class_key(True): bool,
    }


class _TargetClass(Enum):
    ACTIVE = "active"


@pytest.mark.parametrize(
    ("value", "expected_type", "expected_display"),
    [
        (None, "none", "<missing>"),
        (False, "bool", "False"),
        (12, "int", "12"),
        (1.25, "float", "1.25"),
        (Decimal("2.50"), "decimal", "2.50"),
        ("label", "str", "label"),
        (b"label", "bytes", "bGFiZWw="),
        (date(2024, 2, 3), "date", "2024-02-03"),
        (
            datetime(2024, 2, 3, 4, 5, tzinfo=UTC),
            "datetime",
            "2024-02-03T04:05:00+00:00",
        ),
    ],
)
def test_class_identity_supports_builtin_scalar_types(
    value,
    expected_type: str,
    expected_display: str,
) -> None:
    identity = build_class_identity(value)

    assert identity.type_name == expected_type
    assert identity.display_value == expected_display
    assert identity.key == class_key(value)
    assert identity.serialized_value == serialize_class_value(value)
    assert class_type_name(value) == expected_type
    assert class_display_value(value) == expected_display


def test_class_identity_supports_enums_and_normalizes_numpy_or_missing_values() -> None:
    enum_identity = build_class_identity(_TargetClass.ACTIVE)

    assert enum_identity.type_name.endswith("._TargetClass")
    assert enum_identity.display_value == "_TargetClass.ACTIVE"
    assert '"kind":"enum"' in enum_identity.serialized_value
    assert normalize_class_value(np.int32(7)) == 7
    assert type(normalize_class_value(np.int32(7))) is int
    assert normalize_class_value(pd.NA) is None
    assert normalize_class_value(pd.NaT) is None
    assert normalize_class_value(float("nan")) is None
    assert normalize_class_value(Decimal("NaN")) is None
    assert normalize_class_value(bytearray(b"abc")) == b"abc"


@pytest.mark.parametrize(
    "value",
    [float("inf"), Decimal("Infinity"), object()],
)
def test_class_identity_rejects_non_finite_or_unsupported_values(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        serialize_class_value(value)


def test_class_identity_mapping_resolves_typed_and_legacy_labels() -> None:
    identities = build_class_identity_mapping([1, "1", 1.0, True])

    assert identities.key_for(np.int64(1)) == class_key(1)
    assert identities.value_for(class_key("1")) == "1"
    assert identities.identity_for_value(True).key == class_key(True)
    assert identities.identity_for_key(class_key(1)).value == 1
    assert identities.resolve_configured_key(class_key(1.0)) == class_key(1.0)
    assert identities.resolve_configured_key("int::1") == class_key(1)
    assert identities.contains_key(class_key(True))
    assert identities.contains_value(np.bool_(True))
    assert not identities.contains_value("unknown")
    assert identities.class_key_to_value[class_key("1")] == "1"
    assert identities.class_key_to_display_value[class_key(1.0)] == "1.0"

    with pytest.raises(KeyError, match="Unknown configured"):
        identities.resolve_configured_key("absent")
    with pytest.raises(KeyError, match="Unknown Phase 8 class key"):
        identities.value_for("absent")
    with pytest.raises(KeyError, match="not registered"):
        identities.key_for("absent")


def test_class_identity_mapping_validates_counts_and_weights() -> None:
    identities = build_class_identity_mapping(["minority", "majority"])
    keys = identities.class_order

    assert resolve_class_count_mapping(
        {keys[0]: 4, keys[1]: 6},
        identity_mapping=identities,
        require_all_classes=True,
    ) == {"minority": 4, "majority": 6}
    assert resolve_class_weight_mapping(
        {keys[0]: 2.0, keys[1]: 1.0},
        identity_mapping=identities,
        require_all_classes=True,
    ) == {"minority": 2.0, "majority": 1.0}
    assert identities.resolve_count_mapping(
        {"unknown": 2},
        reject_unknown_classes=False,
    ) == {}
    assert identities.resolve_weight_mapping(
        {"unknown": 2.0},
        reject_unknown_classes=False,
    ) == {}

    with pytest.raises(ValueError, match="positive"):
        identities.resolve_count_mapping({keys[0]: 0})
    with pytest.raises(KeyError, match="unknown keys"):
        identities.resolve_count_mapping({"unknown": 2})
    with pytest.raises(KeyError, match="missing keys"):
        identities.resolve_count_mapping({keys[0]: 2}, require_all_classes=True)
    with pytest.raises(ValueError, match="finite and positive"):
        identities.resolve_weight_mapping({keys[0]: float("inf")})
    with pytest.raises(KeyError, match="unknown keys"):
        identities.resolve_weight_mapping({"unknown": 2.0})
    with pytest.raises(KeyError, match="missing keys"):
        identities.resolve_weight_mapping({keys[0]: 2.0}, require_all_classes=True)


def test_class_identity_mapping_rejects_missing_single_class_and_corruption() -> None:
    with pytest.raises(ValueError, match="missing values"):
        build_class_identity_mapping([1, None])
    assert build_class_identity_mapping(
        [1, None],
        reject_missing=False,
    ).class_order == [class_key(1)]
    with pytest.raises(ValueError, match="At least two"):
        build_class_identity_mapping([1, 1], require_at_least_two_classes=True)

    mapping = build_class_identity_mapping([1, 2])
    validate_class_identity_mapping(mapping)
    mapping.class_order = [class_key(1), class_key(1)]
    with pytest.raises(ValueError, match="duplicate"):
        validate_class_identity_mapping(mapping)

    mapping = build_class_identity_mapping([1, 2])
    mapping.class_order = ["missing"]
    with pytest.raises(ValueError, match="unknown identity"):
        validate_class_identity_mapping(mapping)

    mapping = build_class_identity_mapping([1])
    mapping.identities["extra"] = mapping.identities[class_key(1)].model_copy(
        update={"key": "extra"}
    )
    with pytest.raises(ValueError, match="absent from class_order"):
        validate_class_identity_mapping(mapping)

    mapping = build_class_identity_mapping([1])
    invalid_identity = mapping.identities[class_key(1)].model_copy(
        update={"key": "invalid"}
    )
    mapping.class_order = ["invalid"]
    mapping.identities = {"invalid": invalid_identity}
    with pytest.raises(ValueError, match="invalid key"):
        validate_class_identity_mapping(mapping)


def test_class_value_equality_requires_matching_types_and_scalar_comparison() -> None:
    assert class_values_equal(np.int32(1), 1)
    assert not class_values_equal(1, True)
    assert not class_values_equal(1, 1.0)
    assert class_values_equal(None, pd.NA)
    assert class_values_equal(float("nan"), float("nan"))
    assert class_values_equal(_TargetClass.ACTIVE, _TargetClass.ACTIVE)


def test_phase8_state_json_round_trip_preserves_numpy_class_identities() -> None:
    values = [
        np.int8(1),
        "1",
        np.float32(1.0),
        np.bool_(True),
    ]
    identities = build_class_identity_mapping(values)
    index = pd.Index(
        [10, 11, 12, 13],
        dtype="int32",
        name="training_row",
    )
    weights = pd.Series(
        np.asarray([0.25, 0.5, 0.75, 1.0], dtype="float32"),
        index=index,
        dtype="float32",
        name="sample_weight",
    )
    sample_weights = SampleWeightResult(
        weights=weights,
        row_count=len(values),
        class_weight_method="custom",
        entries=[
            SampleWeightEntry(
                output_position=position,
                output_row_id=f"row-{position}",
                class_key=class_key(value),
                class_value=value,
                weight=float(weights.iloc[position]),
                row_type=ResampledRowType.ORIGINAL,
            )
            for position, value in enumerate(values)
        ],
    )
    row_lineage = RowLineageState(
        strategy="identity",
        source_row_count=len(values),
        output_row_count=len(values),
        entries=[
            RowLineageEntry(
                output_position=position,
                output_row_id=f"row-{position}",
                row_type=ResampledRowType.ORIGINAL,
                class_value=value,
                strategy="identity",
            )
            for position, value in enumerate(values)
        ],
    )
    state = Phase8State(
        class_identity_mapping=identities,
        sample_weights=sample_weights,
        row_lineage=row_lineage,
        fitted_state_fingerprint="fitted-state-fingerprint",
        output_fingerprint="output-fingerprint",
    )

    serialized = state.model_dump(mode="json")
    restored = Phase8State.model_validate(serialized)

    assert restored.class_identity_mapping is not None
    assert restored.class_identity_mapping.class_order == identities.class_order
    assert [
        type(restored.class_identity_mapping.value_for(key))
        for key in identities.class_order
    ] == [int, str, float, bool]
    assert restored.sample_weights is not None
    assert restored.sample_weights.weights.equals(weights)
    assert restored.sample_weights.weights.dtype == weights.dtype
    assert restored.sample_weights.weights.index.equals(weights.index)
    assert restored.sample_weights.weights.index.dtype == weights.index.dtype
    assert restored.sample_weights.weights.index.name == weights.index.name
    assert restored.sample_weights.weights.name == weights.name
    assert restored.row_lineage is not None
    assert [
        type(entry.class_value)
        for entry in restored.row_lineage.entries
    ] == [int, str, float, bool]
    assert restored.fitted_state_fingerprint == state.fitted_state_fingerprint
    assert restored.output_fingerprint == state.output_fingerprint
    manager_payload = Phase8FingerprintManager.serialize_model(state)
    manager_restored = Phase8State.model_validate(manager_payload)
    assert manager_restored.row_lineage == restored.row_lineage
    assert manager_restored.sample_weights is not None
    assert manager_restored.sample_weights.weights.equals(weights)


def test_sample_weight_result_json_round_trip_preserves_numpy_scalars() -> None:
    values = [
        np.int32(1),
        "1",
        np.float64(1.0),
        np.bool_(True),
    ]
    index = pd.Index(
        [21, 22, 23, 24],
        dtype="int64",
        name="row",
    )
    weights = pd.Series(
        [0.2, 0.4, 0.6, 0.8],
        index=index,
        dtype="float64",
        name="weight",
    )
    result = SampleWeightResult(
        weights=weights,
        row_count=len(values),
        class_weight_method="custom",
        entries=[
            SampleWeightEntry(
                output_position=position,
                output_row_id=f"row-{position}",
                class_key=class_key(value),
                class_value=value,
                weight=float(weights.iloc[position]),
                row_type=ResampledRowType.ORIGINAL,
            )
            for position, value in enumerate(values)
        ],
    )

    serialized = result.model_dump(mode="json")
    restored = SampleWeightResult.model_validate(serialized)

    assert restored.weights.equals(weights)
    assert restored.weights.dtype == weights.dtype
    assert restored.weights.index.equals(weights.index)
    assert restored.weights.index.dtype == weights.index.dtype
    assert restored.weights.index.name == weights.index.name
    assert restored.weights.name == weights.name
    assert [type(entry.class_value) for entry in restored.entries] == [
        int,
        str,
        float,
        bool,
    ]
    assert [entry.class_key for entry in restored.entries] == [
        class_key(value)
        for value in (1, "1", 1.0, True)
    ]
