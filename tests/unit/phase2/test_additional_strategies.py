from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
    SplitContractOptions,
)
from mlprep.phases.phase2.output_governance.metadata import (
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.state import SplitColumnState
from mlprep.phases.phase2.strategies.group_stratified_split import (
    GroupStratifiedSplitOptions,
    GroupStratifiedSplitStrategy,
)
from mlprep.phases.phase2.strategies.predefined_split import (
    PredefinedSplitOptions,
    PredefinedSplitStrategy,
)
from mlprep.phases.phase2.strategies.time_split import (
    TimeSplitOptions,
    TimeSplitStrategy,
)

_RATIOS = SplitRatios(train=0.5, validation=0.25, test=0.25)


def _contract(
    *,
    strategy: SplitStrategyName,
    data: pd.DataFrame,
    group_columns: list[str] | None = None,
    timestamp_column: str | None = None,
) -> SplitContract:
    return SplitContract(
        data=data,
        phase1_result=None,
        phase1_state=None,
        columns=SplitColumnState(
            target_column="target",
            feature_columns=["feature"],
            group_columns=group_columns or [],
            timestamp_column=timestamp_column,
        ),
        options=SplitContractOptions(
            strategy=strategy,
            ratios=_RATIOS,
            group_columns=group_columns or [],
            timestamp_column=timestamp_column,
        ),
        findings=[],
    )


def _group_data(group_count: int = 12) -> pd.DataFrame:
    rows = []
    for group_number in range(group_count):
        for class_value in (0, 0, 1, 1):
            rows.append(
                {
                    "feature": len(rows),
                    "target": class_value,
                    "group": f"group-{group_number}",
                    "__mlprep_row_id__": f"row-{len(rows)}",
                }
            )
    return pd.DataFrame(rows, index=pd.Index(range(len(rows)), name="row"))


def test_group_stratified_split_preserves_groups_and_metadata() -> None:
    data = _group_data()
    contract = _contract(
        strategy=SplitStrategyName.GROUP_STRATIFIED,
        data=data,
        group_columns=["group"],
    )
    strategy = GroupStratifiedSplitStrategy(
        GroupStratifiedSplitOptions(
            ratios=_RATIOS,
            random_seed=11,
            shuffle_groups=False,
            distribution_tolerance=1.0,
            maximum_ratio_deviation=1.0,
        )
    )

    result = strategy.split(contract)

    train_groups = set(data.loc[result.indices.train_indices, "group"])
    validation_groups = set(data.loc[result.indices.validation_indices, "group"])
    test_groups = set(data.loc[result.indices.test_indices, "group"])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
    assert (
        len(result.train.X)
        + len(result.validation.X)
        + len(result.test.X)
        == len(data)
    )
    assert result.metadata["stratified"] is True
    assert result.metadata["grouped"] is True
    assert set(result.metadata["unique_group_counts"]) == {
        "train",
        "validation",
        "test",
    }


def test_group_stratified_split_rejects_missing_or_insufficient_groups() -> None:
    contract = _contract(
        strategy=SplitStrategyName.GROUP_STRATIFIED,
        data=_group_data(group_count=2),
        group_columns=["group"],
    )
    strategy = GroupStratifiedSplitStrategy(
        GroupStratifiedSplitOptions(ratios=_RATIOS)
    )
    with pytest.raises(ValueError, match="too few groups"):
        strategy.split(contract)

    missing_groups = _group_data()
    missing_groups.loc[0, "group"] = None
    contract = _contract(
        strategy=SplitStrategyName.GROUP_STRATIFIED,
        data=missing_groups,
        group_columns=["group"],
    )
    with pytest.raises(ValueError, match="missing group values"):
        strategy.split(contract)


def test_time_split_keeps_tied_timestamps_together_and_orders_partitions() -> None:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="D").repeat(2)
    data = pd.DataFrame(
        {
            "feature": range(12),
            "target": [index % 2 for index in range(12)],
            "timestamp": timestamps,
            "__mlprep_row_id__": [f"row-{index}" for index in range(12)],
        },
        index=pd.Index(range(12), name="row"),
    )
    contract = _contract(
        strategy=SplitStrategyName.TIME,
        data=data,
        timestamp_column="timestamp",
    )
    result = TimeSplitStrategy(
        TimeSplitOptions(ratios=_RATIOS)
    ).split(contract)

    train_times = data.loc[result.indices.train_indices, "timestamp"]
    validation_times = data.loc[result.indices.validation_indices, "timestamp"]
    test_times = data.loc[result.indices.test_indices, "timestamp"]
    assert train_times.max() < validation_times.min()
    assert validation_times.max() < test_times.min()
    assert result.metadata["temporal"] is True


def test_time_split_rejects_missing_invalid_and_unconfigured_timestamps() -> None:
    data = pd.DataFrame(
        {
            "feature": range(8),
            "target": [0, 1] * 4,
            "timestamp": pd.date_range("2024-01-01", periods=8),
            "__mlprep_row_id__": [f"row-{index}" for index in range(8)],
        }
    )
    with pytest.raises(ValueError, match="requires a timestamp column"):
        _contract(strategy=SplitStrategyName.TIME, data=data)

    contract = _contract(
        strategy=SplitStrategyName.TIME,
        data=data,
        timestamp_column="unavailable",
    )
    with pytest.raises(ValueError, match="does not exist"):
        TimeSplitStrategy(TimeSplitOptions(ratios=_RATIOS)).split(contract)

    contract = _contract(
        strategy=SplitStrategyName.TIME,
        data=data,
        timestamp_column="timestamp",
    )

    invalid = data.copy()
    invalid["timestamp"] = invalid["timestamp"].astype(object)
    invalid.loc[0, "timestamp"] = "not-a-date"
    contract = _contract(
        strategy=SplitStrategyName.TIME,
        data=invalid,
        timestamp_column="timestamp",
    )
    with pytest.raises(ValueError, match="unparseable"):
        TimeSplitStrategy(TimeSplitOptions(ratios=_RATIOS)).split(contract)

    missing = data.copy()
    missing.loc[0, "timestamp"] = None
    contract = _contract(
        strategy=SplitStrategyName.TIME,
        data=missing,
        timestamp_column="timestamp",
    )
    with pytest.raises(ValueError, match="missing values"):
        TimeSplitStrategy(TimeSplitOptions(ratios=_RATIOS)).split(contract)


def test_predefined_split_normalizes_values_and_requires_complete_assignment() -> None:
    data = pd.DataFrame(
        {
            "feature": range(6),
            "target": [0, 1, 0, 1, 0, 1],
            "partition": [" Train ", "train", "VAL", "validation", "TEST", "test"],
            "__mlprep_row_id__": [f"row-{index}" for index in range(6)],
        },
        index=pd.Index(range(6), name="row"),
    )
    contract = _contract(
        strategy=SplitStrategyName.PREDEFINED,
        data=data,
    )
    result = PredefinedSplitStrategy(
        PredefinedSplitOptions(split_column="partition")
    ).split(contract)

    assert result.indices.train_indices.tolist() == [0, 1]
    assert result.indices.validation_indices.tolist() == [2, 3]
    assert result.indices.test_indices.tolist() == [4, 5]
    assert result.metadata["predefined"] is True

    data.loc[0, "partition"] = "holdout"
    contract = _contract(
        strategy=SplitStrategyName.PREDEFINED,
        data=data,
    )
    with pytest.raises(ValueError, match="unassigned values"):
        PredefinedSplitStrategy(
            PredefinedSplitOptions(split_column="partition")
        ).split(contract)


def test_predefined_split_rejects_overlap_and_reports_empty_partitions() -> None:
    with pytest.raises(ValueError, match="overlap"):
        PredefinedSplitOptions(
            split_column="partition",
            train_values={"train"},
            validation_values={"train"},
        )

    data = pd.DataFrame(
        {
            "feature": range(4),
            "target": [0, 1, 0, 1],
            "partition": ["train", "train", "test", "test"],
            "__mlprep_row_id__": [f"row-{index}" for index in range(4)],
        }
    )
    contract = _contract(
        strategy=SplitStrategyName.PREDEFINED,
        data=data,
    )
    result = PredefinedSplitStrategy(
        PredefinedSplitOptions(
            split_column="partition",
            require_complete_assignment=False,
        )
    ).split(contract)
    assert any(finding.code == "P2-PREDEFINED-001" for finding in result.findings)
