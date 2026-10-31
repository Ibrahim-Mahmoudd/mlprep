"""Tests for Phase 2 group-aware splitting."""

from __future__ import annotations

import pandas as pd

from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
    SplitContractOptions,
)
from mlprep.phases.phase2.output_governance.metadata import (
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.state import (
    SplitColumnState,
)
from mlprep.phases.phase2.strategies.group_split import (
    GroupSplitOptions,
    GroupSplitStrategy,
)


def build_group_contract() -> SplitContract:
    """Create a valid split contract with independent groups."""

    group_count = 12
    rows_per_group = 3
    row_count = group_count * rows_per_group
    data = pd.DataFrame(
        {
            "feature": range(row_count),
            "target": [index % 2 for index in range(row_count)],
            "group": [
                f"group-{index // rows_per_group}"
                for index in range(row_count)
            ],
            "__mlprep_row_id__": [
                f"row-{index}" for index in range(row_count)
            ],
        },
        index=pd.Index(range(row_count), name="row"),
    )
    ratios = SplitRatios(
        train=0.4,
        validation=0.3,
        test=0.3,
    )
    options = SplitContractOptions(
        strategy=SplitStrategyName.GROUP,
        ratios=ratios,
        group_columns=["group"],
    )
    columns = SplitColumnState(
        target_column="target",
        feature_columns=["feature"],
        group_columns=["group"],
    )
    return SplitContract(
        data=data,
        phase1_result=None,
        phase1_state=None,
        columns=columns,
        options=options,
        findings=[],
    )


def build_strategy() -> GroupSplitStrategy:
    """Create the seeded group split strategy."""

    return GroupSplitStrategy(
        GroupSplitOptions(
            ratios=SplitRatios(
                train=0.4,
                validation=0.3,
                test=0.3,
            ),
            random_seed=42,
            maximum_ratio_deviation=0.0,
        )
    )


def test_group_split_returns_complete_partitions() -> None:
    contract = build_group_contract()

    result = build_strategy().split(contract)

    assert len(result.train.X) + len(result.validation.X) + len(
        result.test.X
    ) == len(contract.data)
    assert not result.findings or all(
        finding.code == "P2-GROUP-002"
        for finding in result.findings
    )


def test_group_split_keeps_groups_disjoint() -> None:
    contract = build_group_contract()

    result = build_strategy().split(contract)

    train_groups = set(
        contract.data.loc[result.indices.train_indices, "group"]
    )
    validation_groups = set(
        contract.data.loc[result.indices.validation_indices, "group"]
    )
    test_groups = set(
        contract.data.loc[result.indices.test_indices, "group"]
    )
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_group_split_calculates_ratio_deviations_without_name_error() -> None:
    contract = build_group_contract()

    findings = build_strategy().split(contract).findings

    ratio_findings = [
        finding
        for finding in findings
        if finding.code == "P2-GROUP-002"
    ]
    assert ratio_findings
    assert all(
        "deviation" in finding.details
        for finding in ratio_findings
    )


def test_group_split_is_repeatable_with_same_seed() -> None:
    contract = build_group_contract()

    first = build_strategy().split(contract)
    second = build_strategy().split(contract)

    assert first.indices.train_indices.equals(
        second.indices.train_indices
    )
    assert first.indices.validation_indices.equals(
        second.indices.validation_indices
    )
    assert first.indices.test_indices.equals(
        second.indices.test_indices
    )
