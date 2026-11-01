from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase1.output_governance.state import Phase1State
from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
    SplitContractOptions,
)
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.state import SplitColumnState
from mlprep.phases.phase2.strategies.base import StrategyPartitions
from mlprep.phases.phase2.validation.group_distribution import (
    GroupDistributionValidationOptions,
    GroupDistributionValidator,
)
from mlprep.phases.phase2.validation.temporal_boundaries import (
    TemporalBoundaryValidationOptions,
    TemporalBoundaryValidator,
)


def _contract(
    data: pd.DataFrame,
    *,
    group_columns: list[str] | None = None,
    timestamp_column: str | None = None,
) -> SplitContract:
    groups = group_columns or []
    strategy = (
        SplitStrategyName.GROUP
        if groups
        else SplitStrategyName.TIME
        if timestamp_column is not None
        else SplitStrategyName.RANDOM
    )
    return SplitContract(
        data=data,
        phase1_result=None,
        phase1_state=Phase1State(),
        columns=SplitColumnState(
            target_column="target",
            feature_columns=["feature"],
            group_columns=groups,
            timestamp_column=timestamp_column,
        ),
        options=SplitContractOptions(
            strategy=strategy,
            group_columns=groups,
            timestamp_column=timestamp_column,
        ),
        findings=[],
    )


def _indices(
    train: list[int],
    validation: list[int],
    test: list[int],
) -> StrategyPartitions:
    return StrategyPartitions(
        train_indices=pd.Index(train),
        validation_indices=pd.Index(validation),
        test_indices=pd.Index(test),
    )


def test_group_distribution_reports_overlap_and_concentration() -> None:
    data = pd.DataFrame(
        {
            "feature": range(6),
            "target": range(6),
            "group": ["a", "b", "a", "b", "c", "c"],
        }
    )
    findings = GroupDistributionValidator(
        GroupDistributionValidationOptions(
            maximum_largest_group_fraction=0.5,
        )
    ).validate(
        contract=_contract(data, group_columns=["group"]),
        indices=_indices([0, 1], [2, 3], [4, 5]),
    )

    overlaps = [
        finding
        for finding in findings
        if finding.code == "P2-GROUP-DIST-001"
    ]
    concentrated = [
        finding
        for finding in findings
        if finding.code == "P2-GROUP-DIST-003"
    ]
    assert len(overlaps) == 1
    assert overlaps[0].partitions == [
        PartitionName.TRAIN,
        PartitionName.VALIDATION,
    ]
    assert overlaps[0].affected_row_count == 2
    assert overlaps[0].blocks_readiness
    assert [finding.partitions for finding in concentrated] == [
        [PartitionName.TEST]
    ]
    assert concentrated[0].details["largest_group_fraction"] == 1.0
    assert not concentrated[0].blocks_readiness


def test_group_distribution_handles_minimum_groups_empty_parts_and_options() -> None:
    data = pd.DataFrame(
        {
            "feature": range(3),
            "target": range(3),
            "group": ["a", "b", "c"],
        }
    )
    validator = GroupDistributionValidator(
        GroupDistributionValidationOptions(
            minimum_groups_per_partition=2,
            require_no_group_overlap=False,
        )
    )
    findings = validator.validate(
        contract=_contract(data, group_columns=["group"]),
        indices=_indices([0], [], [1, 2]),
    )
    disabled = GroupDistributionValidator(
        GroupDistributionValidationOptions(enabled=False)
    ).validate(
        contract=_contract(data, group_columns=["group"]),
        indices=_indices([0], [1], [2]),
    )
    no_groups = GroupDistributionValidator().validate(
        contract=_contract(data),
        indices=_indices([0], [1], [2]),
    )

    assert [
        (finding.partitions, finding.details["unique_group_count"])
        for finding in findings
    ] == [
        ([PartitionName.TRAIN], 1),
        ([PartitionName.VALIDATION], 0),
    ]
    assert disabled == []
    assert no_groups == []


def test_group_distribution_combines_multiple_group_columns() -> None:
    data = pd.DataFrame(
        {
            "feature": range(6),
            "target": range(6),
            "region": ["north"] * 6,
            "entity": list("abcdef"),
        }
    )
    findings = GroupDistributionValidator().validate(
        contract=_contract(
            data,
            group_columns=["region", "entity"],
        ),
        indices=_indices([0, 1], [2, 3], [4, 5]),
    )

    assert findings == []
    with pytest.raises(ValueError, match="at least one"):
        GroupDistributionValidationOptions(
            minimum_groups_per_partition=0
        )
    with pytest.raises(ValueError, match="greater than zero"):
        GroupDistributionValidationOptions(
            maximum_largest_group_fraction=0.0
        )


def test_temporal_boundaries_serialize_ranges_and_validate_gaps() -> None:
    data = pd.DataFrame(
        {
            "feature": range(6),
            "target": range(6),
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="D"),
        }
    )
    findings, boundaries = TemporalBoundaryValidator(
        TemporalBoundaryValidationOptions(
            minimum_train_validation_gap="2D",
            minimum_validation_test_gap="2D",
        )
    ).validate(
        contract=_contract(data, timestamp_column="timestamp"),
        indices=_indices([0, 1], [2, 3], [4, 5]),
    )

    assert [finding.code for finding in findings] == [
        "P2-TIME-BOUNDARY-003",
        "P2-TIME-BOUNDARY-003",
    ]
    assert [finding.details["actual_gap"] for finding in findings] == [
        "1 days 00:00:00",
        "1 days 00:00:00",
    ]
    assert boundaries.train_start == "2024-01-01T00:00:00"
    assert boundaries.train_end == "2024-01-02T00:00:00"
    assert boundaries.validation_start == "2024-01-03T00:00:00"
    assert boundaries.test_end == "2024-01-06T00:00:00"


def test_temporal_boundaries_handle_equal_times_and_overlap_policy() -> None:
    data = pd.DataFrame(
        {
            "feature": range(5),
            "target": range(5),
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
        }
    )
    contract = _contract(data, timestamp_column="timestamp")
    partitions = _indices([0, 1], [1, 2], [3, 4])
    blocked, _ = TemporalBoundaryValidator().validate(
        contract=contract,
        indices=partitions,
    )
    allowed, _ = TemporalBoundaryValidator(
        TemporalBoundaryValidationOptions(
            allow_equal_boundary_timestamps=True,
            minimum_train_validation_gap="0D",
            minimum_validation_test_gap="0D",
        )
    ).validate(
        contract=contract,
        indices=partitions,
    )
    nonblocking, _ = TemporalBoundaryValidator(
        TemporalBoundaryValidationOptions(
            block_on_overlap=False,
        )
    ).validate(
        contract=contract,
        indices=partitions,
    )

    assert [finding.code for finding in blocked] == [
        "P2-TIME-BOUNDARY-002"
    ]
    assert blocked[0].blocks_readiness
    assert allowed == []
    assert len(nonblocking) == 1
    assert not nonblocking[0].blocks_readiness


def test_temporal_boundaries_cover_missing_disabled_and_unconfigured_inputs() -> None:
    data = pd.DataFrame(
        {
            "feature": range(6),
            "target": range(6),
            "timestamp": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "invalid",
                "2024-01-05",
                "2024-01-06",
            ],
        }
    )
    contract = _contract(data, timestamp_column="timestamp")
    partitions = _indices([0, 1], [2, 3], [4, 5])
    missing, empty_boundaries = TemporalBoundaryValidator().validate(
        contract=contract,
        indices=partitions,
    )
    tolerated, boundaries = TemporalBoundaryValidator(
        TemporalBoundaryValidationOptions(
            require_nonmissing_timestamps=False,
        )
    ).validate(
        contract=contract,
        indices=partitions,
    )
    disabled, disabled_boundaries = TemporalBoundaryValidator(
        TemporalBoundaryValidationOptions(enabled=False)
    ).validate(
        contract=contract,
        indices=partitions,
    )
    unconfigured, _ = TemporalBoundaryValidator().validate(
        contract=_contract(data),
        indices=partitions,
    )

    assert [finding.code for finding in missing] == [
        "P2-TIME-BOUNDARY-001"
    ]
    assert missing[0].affected_row_count == 1
    assert empty_boundaries.train_start is None
    assert tolerated == []
    assert boundaries.validation_start == "2024-01-03T00:00:00"
    assert disabled == [] and disabled_boundaries.train_start is None
    assert unconfigured == []
