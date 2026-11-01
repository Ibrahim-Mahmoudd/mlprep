"""Tests for Phase 9 model handoff governance."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase2.output_governance.result import DatasetPartition
from mlprep.phases.phase8.output_governance.fingerprint import (
    Phase8FingerprintManager,
)
from mlprep.phases.phase8.output_governance.metadata import (
    PartitionName as Phase8PartitionName,
)
from mlprep.phases.phase8.output_governance.metadata import (
    Phase8Status,
)
from mlprep.phases.phase8.output_governance.result import (
    Phase8ReadinessDecision,
    Phase8Result,
)
from mlprep.phases.phase8.output_governance.state import Phase8State
from mlprep.phases.phase9 import Phase9
from mlprep.phases.phase9.config import Phase9Config


def build_partition(
    name: PartitionName,
    row_ids: list[str],
) -> DatasetPartition:
    index = pd.Index(range(len(row_ids)))
    return DatasetPartition(
        name=name,
        X=pd.DataFrame(
            {"feature": range(len(row_ids))},
            index=index,
        ),
        y=pd.Series(
            [0] * len(row_ids),
            index=index,
            name="target",
        ),
        row_ids=pd.Series(
            row_ids,
            index=index,
            name="row_id",
        ),
    )


def build_phase8_result(
    *,
    ready: bool = True,
    overlapping_row_ids: bool = False,
) -> tuple[Phase8Result, Phase8State]:
    train_ids = ["train-1", "train-2"]
    validation_ids = ["validation-1"]
    if overlapping_row_ids:
        validation_ids = ["train-1"]

    result = Phase8Result(
        train=build_partition(PartitionName.TRAIN, train_ids),
        validation=build_partition(
            PartitionName.VALIDATION,
            [*validation_ids, "validation-2"],
        ),
        test=build_partition(PartitionName.TEST, ["test-1"]),
    )
    result.readiness = Phase8ReadinessDecision(
        ready=ready,
        status=(
            Phase8Status.PASSED
            if ready
            else Phase8Status.FAILED
        ),
    )
    result.class_weights = {"0": 1.0}
    result.sample_weights = pd.Series(
        [1.0, 1.0],
        index=result.train.X.index,
        name="sample_weight",
    )

    state = Phase8State(
        ready_for_phase_9=ready,
        fitted_on_training_only=True,
        output_partition_fingerprints={
            Phase8PartitionName.TRAIN:
                Phase8FingerprintManager.partition_fingerprint(
                    result.train
                ),
            Phase8PartitionName.VALIDATION:
                Phase8FingerprintManager.partition_fingerprint(
                    result.validation
                ),
            Phase8PartitionName.TEST:
                Phase8FingerprintManager.partition_fingerprint(
                    result.test
                ),
        },
        output_fingerprint="output-fingerprint",
        fitted_state_fingerprint="state-fingerprint",
    )
    result.phase_state = state.model_dump(mode="json")
    return result, state


def test_handoff_copies_partitions_and_serializes_summary() -> None:
    source, _ = build_phase8_result()

    handoff = Phase9().run(phase8_result=source)
    handoff.train.X.loc[0, "feature"] = 99

    assert source.train.X.loc[0, "feature"] == 0
    assert handoff.target_column == "target"
    assert handoff.feature_columns == ["feature"]
    assert handoff.sample_weights is not None
    assert handoff.sample_weights.tolist() == [1.0, 1.0]
    assert json.dumps(handoff.to_report_dict())


def test_handoff_rejects_phase8_not_ready() -> None:
    source, _ = build_phase8_result(ready=False)

    with pytest.raises(PermissionError, match="not ready"):
        Phase9().run(phase8_result=source)


def test_handoff_rejects_overlapping_partition_rows() -> None:
    source, _ = build_phase8_result(overlapping_row_ids=True)

    with pytest.raises(ValueError, match="row IDs overlap"):
        Phase9().run(phase8_result=source)


def test_handoff_rejects_invalid_training_weights() -> None:
    source, _ = build_phase8_result()
    source.sample_weights.iloc[0] = 0.0

    with pytest.raises(ValueError, match="finite and positive"):
        Phase9().run(phase8_result=source)


@pytest.mark.parametrize(
    ("partition_name", "mutation"),
    [
        ("train", "value"),
        ("validation", "value"),
        ("test", "value"),
        ("validation", "dtype"),
        ("test", "index"),
        ("validation", "row_order"),
        ("train", "row_id"),
    ],
)
def test_handoff_rejects_partitions_changed_after_phase8(
    partition_name: str,
    mutation: str,
) -> None:
    source, _ = build_phase8_result()
    partition = getattr(source, partition_name)

    if mutation == "value":
        partition.X.iloc[0, 0] += 1
    elif mutation == "dtype":
        partition.X["feature"] = partition.X["feature"].astype("float64")
    elif mutation == "index":
        changed_index = pd.Index(partition.X.index + 10)
        partition.X.index = changed_index
        partition.y.index = changed_index
        partition.row_ids.index = changed_index
    elif mutation == "row_id":
        partition.row_ids.iloc[0] = "changed-row-id"
    else:
        reordered = partition.X.iloc[::-1].copy()
        partition.X = reordered
        partition.y = partition.y.iloc[::-1].copy()
        partition.row_ids = partition.row_ids.iloc[::-1].copy()

    with pytest.raises(
        ValueError,
        match=f"{partition_name} partition changed after Phase 8",
    ):
        Phase9().run(phase8_result=source)


def test_handoff_accepts_unmodified_phase8_partitions() -> None:
    source, _ = build_phase8_result()

    handoff = Phase9().run(phase8_result=source)

    assert handoff.readiness.ready


def test_handoff_accepts_phase8_state_after_json_round_trip() -> None:
    source, state = build_phase8_result()
    serialized_state = state.model_dump(mode="json")
    restored_state = Phase8State.model_validate(serialized_state)

    handoff = Phase9().run(
        phase8_result=source,
        phase8_state=restored_state,
    )

    assert handoff.readiness.ready
    assert handoff.phase8_state.output_fingerprint == state.output_fingerprint
    assert handoff.phase8_state.fitted_state_fingerprint == (
        state.fitted_state_fingerprint
    )


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("ready_for_phase_9", False, PermissionError, "does not authorize"),
        ("fitted_on_training_only", False, ValueError, "training data only"),
        ("validation_remained_protected", False, ValueError, "validation partition"),
        ("test_remained_protected", False, ValueError, "test partition"),
        ("output_fingerprint", None, ValueError, "output fingerprint"),
        ("fitted_state_fingerprint", None, ValueError, "fitted-state fingerprint"),
    ],
)
def test_handoff_rejects_phase8_state_without_required_governance(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    source, state = build_phase8_result()
    setattr(state, field, value)

    with pytest.raises(error, match=message):
        Phase9().run(phase8_result=source, phase8_state=state)


def test_handoff_rejects_missing_readiness_and_state_fingerprints() -> None:
    source, state = build_phase8_result()
    source.readiness = None

    with pytest.raises(ValueError, match="no readiness decision"):
        Phase9().run(phase8_result=source, phase8_state=state)

    source, state = build_phase8_result()
    del state.output_partition_fingerprints[Phase8PartitionName.TEST]

    with pytest.raises(ValueError, match="test output fingerprint is missing"):
        Phase9().run(phase8_result=source, phase8_state=state)


def test_handoff_rejects_partition_names_and_changed_fingerprints() -> None:
    source, state = build_phase8_result()
    source.validation.name = PartitionName.TRAIN

    with pytest.raises(ValueError, match="Expected the validation partition"):
        Phase9().run(phase8_result=source, phase8_state=state)

    source, state = build_phase8_result()
    source.test.X.iloc[0, 0] += 1

    with pytest.raises(ValueError, match="test partition changed after Phase 8"):
        Phase9().run(phase8_result=source, phase8_state=state)


def test_handoff_rejects_invalid_feature_and_target_schemas() -> None:
    source, state = build_phase8_result()
    source.train.X = source.train.X.iloc[:, :0]
    state.output_partition_fingerprints[Phase8PartitionName.TRAIN] = (
        Phase8FingerprintManager.partition_fingerprint(source.train)
    )

    with pytest.raises(ValueError, match="no model features"):
        Phase9().run(phase8_result=source, phase8_state=state)

    source, state = build_phase8_result()
    source.validation.X["extra"] = 1
    state.output_partition_fingerprints[Phase8PartitionName.VALIDATION] = (
        Phase8FingerprintManager.partition_fingerprint(source.validation)
    )

    with pytest.raises(ValueError, match="feature schema differs"):
        Phase9().run(phase8_result=source, phase8_state=state)

    source, state = build_phase8_result()
    source.validation.y.name = "other_target"
    state.output_partition_fingerprints[Phase8PartitionName.VALIDATION] = (
        Phase8FingerprintManager.partition_fingerprint(source.validation)
    )

    with pytest.raises(ValueError, match="same non-empty column name"):
        Phase9().run(phase8_result=source, phase8_state=state)


def test_handoff_checks_target_features_and_can_disable_row_disjointness() -> None:
    source, state = build_phase8_result()
    for partition in (source.train, source.validation, source.test):
        partition.X.columns = ["target"]
        state.output_partition_fingerprints[partition.name] = (
            Phase8FingerprintManager.partition_fingerprint(partition)
        )

    with pytest.raises(ValueError, match="target is incorrectly included"):
        Phase9().run(phase8_result=source, phase8_state=state)

    source, state = build_phase8_result(overlapping_row_ids=True)
    handoff = Phase9(
        Phase9Config(require_disjoint_rows=False)
    ).run(phase8_result=source, phase8_state=state)

    assert handoff.readiness.ready


def test_handoff_validates_types_state_availability_and_copy_option() -> None:
    source, _ = build_phase8_result()
    with pytest.raises(TypeError, match="phase8_result"):
        Phase9().run(phase8_result=object())  # type: ignore[arg-type]

    source.phase_state = {}
    with pytest.raises(ValueError, match="state is unavailable"):
        Phase9().run(phase8_result=source)

    source, state = build_phase8_result()
    with pytest.raises(TypeError, match="phase8_state"):
        Phase9().run(
            phase8_result=source,
            phase8_state=object(),  # type: ignore[arg-type]
        )

    handoff = Phase9(
        Phase9Config(copy_partitions=False)
    ).run(phase8_result=source)

    assert handoff.train is source.train
    assert handoff.validation is source.validation
    assert handoff.test is source.test
    assert state.output_fingerprint == handoff.metadata.phase8_output_fingerprint


@pytest.mark.parametrize(
    ("weights", "error", "message"),
    [
        ([1.0, 1.0], TypeError, "must be a pandas Series"),
        (pd.Series([1.0]), ValueError, "do not align with training rows"),
        (
            pd.Series([1.0, 1.0], index=[10, 11]),
            ValueError,
            "indices do not align",
        ),
        (
            pd.Series([float("inf"), 1.0]),
            ValueError,
            "must be finite and positive",
        ),
    ],
)
def test_handoff_rejects_misaligned_or_invalid_sample_weights(
    weights: object,
    error: type[Exception],
    message: str,
) -> None:
    source, _ = build_phase8_result()
    source.sample_weights = weights

    with pytest.raises(error, match=message):
        Phase9().run(phase8_result=source)


@pytest.mark.parametrize("weight", [0.0, float("nan"), float("inf")])
def test_handoff_rejects_invalid_class_weights(weight: float) -> None:
    source, _ = build_phase8_result()
    source.class_weights = {"0": weight}

    with pytest.raises(ValueError, match="must be finite and positive"):
        Phase9().run(phase8_result=source)
