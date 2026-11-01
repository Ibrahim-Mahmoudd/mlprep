from __future__ import annotations

import pytest

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase4.output_governance.state import Phase4State
from mlprep.phases.phase4.strategies.base import ImputerState


def test_component_status_transitions_are_idempotent() -> None:
    state = Phase4State()

    state.mark_failed("numeric")
    state.mark_failed("numeric")
    state.mark_skipped("numeric")
    state.mark_completed("numeric")
    state.mark_completed("numeric")

    assert state.component_completed("numeric")
    assert state.completed_components == ["numeric"]
    assert state.failed_components == []
    assert state.skipped_components == []

    state.mark_failed("numeric")
    assert not state.component_completed("numeric")
    assert state.failed_components == ["numeric"]
    assert state.completed_components == []


@pytest.mark.parametrize(
    "mark",
    [
        "mark_completed",
        "mark_failed",
        "mark_skipped",
    ],
)
def test_component_status_rejects_empty_names(mark: str) -> None:
    state = Phase4State()

    with pytest.raises(ValueError, match="component cannot be empty"):
        getattr(state, mark)(" ")


def test_registered_imputer_state_is_copied_and_key_is_validated() -> None:
    state = Phase4State()
    imputer_state = ImputerState(
        column="amount",
        strategy="numeric",
        fitted=True,
        learned_value=2.5,
        metadata={"method": "median"},
    )

    state.register_imputer_state(column="amount", state=imputer_state)
    imputer_state.metadata["method"] = "mean"

    assert state.imputer_states["amount"].metadata == {"method": "median"}
    with pytest.raises(ValueError, match="does not match"):
        state.register_imputer_state(
            column="other",
            state=imputer_state,
        )


def test_partition_fingerprints_and_nested_state_round_trip() -> None:
    state = Phase4State(
        fitted_state_fingerprint="fit-sha",
        output_fingerprint="output-sha",
        source_feature_columns=["amount"],
        output_feature_columns=["amount", "amount__missing"],
    )
    state.register_partition_fingerprint(
        partition=PartitionName.TRAIN,
        input_fingerprint="train-in",
        output_fingerprint="train-out",
    )
    state.register_partition_fingerprint(
        partition=PartitionName.TRAIN,
        output_fingerprint="train-out-updated",
    )
    state.register_imputer_state(
        column="amount",
        state=ImputerState(
            column="amount",
            strategy="numeric",
            fitted=True,
            learned_value=2.5,
            metadata={"quantile": 0.5},
        ),
    )

    restored = Phase4State.model_validate(state.model_dump(mode="json"))

    assert restored.input_partition_fingerprints == {
        PartitionName.TRAIN: "train-in"
    }
    assert restored.output_partition_fingerprints == {
        PartitionName.TRAIN: "train-out-updated"
    }
    assert restored.fitted_state_fingerprint == "fit-sha"
    assert restored.output_fingerprint == "output-sha"
    assert restored.source_feature_columns == ["amount"]
    assert restored.output_feature_columns == [
        "amount",
        "amount__missing",
    ]
    assert restored.imputer_states["amount"].learned_value == 2.5
    assert restored.imputer_states["amount"].metadata == {"quantile": 0.5}
