from __future__ import annotations

import numpy as np
import pandas as pd

from mlprep.phases.phase1.output_governance.state import Phase1State
from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
    SplitContractOptions,
)
from mlprep.phases.phase2.leakage_firewall.duplicate_overlap import (
    DuplicateOverlapChecker,
)
from mlprep.phases.phase2.leakage_firewall.group_overlap import (
    GroupOverlapChecker,
)
from mlprep.phases.phase2.leakage_firewall.post_outcome import (
    PostOutcomeLeakageChecker,
    PostOutcomeLeakageOptions,
)
from mlprep.phases.phase2.leakage_firewall.row_overlap import (
    RowOverlapChecker,
)
from mlprep.phases.phase2.leakage_firewall.target_leakage import (
    TargetLeakageChecker,
)
from mlprep.phases.phase2.leakage_firewall.temporal_leakage import (
    TemporalLeakageChecker,
)
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase2.output_governance.state import SplitColumnState
from mlprep.phases.phase2.strategies.base import StrategyPartitions
from mlprep.phases.phase2.strategies.random_split import (
    RandomSplitOptions,
    RandomSplitStrategy,
)
from mlprep.phases.phase2.strategies.regression_stratified_split import (
    RegressionStratifiedSplitOptions,
    RegressionStratifiedSplitStrategy,
)
from mlprep.phases.phase2.strategies.stratified_split import (
    StratifiedSplitOptions,
    StratifiedSplitStrategy,
)

_RATIOS = SplitRatios(
    train=0.6,
    validation=0.2,
    test=0.2,
)


def _contract(
    data: pd.DataFrame,
    *,
    strategy: SplitStrategyName = SplitStrategyName.RANDOM,
    groups: list[str] | None = None,
    timestamp: str | None = None,
    features: list[str] | None = None,
) -> SplitContract:
    group_columns = groups or []
    return SplitContract(
        data=data,
        phase1_result=None,
        phase1_state=Phase1State(),
        columns=SplitColumnState(
            target_column="target",
            feature_columns=features or ["signal"],
            group_columns=group_columns,
            timestamp_column=timestamp,
        ),
        options=SplitContractOptions(
            strategy=strategy,
            ratios=_RATIOS,
            group_columns=group_columns,
            timestamp_column=timestamp,
        ),
        findings=[],
    )


def test_random_split_assigns_every_row_deterministically() -> None:
    data = pd.DataFrame(
        {
            "signal": range(40),
            "target": [index % 2 for index in range(40)],
            "__mlprep_row_id__": [f"row-{index}" for index in range(40)],
        },
        index=pd.Index(range(40), name="row"),
    )
    contract = _contract(data)
    strategy = RandomSplitStrategy(
        RandomSplitOptions(
            ratios=_RATIOS,
            random_seed=29,
        )
    )

    first = strategy.split(contract)
    second = RandomSplitStrategy(
        RandomSplitOptions(
            ratios=_RATIOS,
            random_seed=29,
        )
    ).split(contract)

    assert first.indices.all_indices().sort_values().tolist() == data.index.tolist()
    assert first.indices.train_indices.equals(second.indices.train_indices)
    assert first.indices.validation_indices.equals(second.indices.validation_indices)
    assert first.indices.test_indices.equals(second.indices.test_indices)
    assert first.metadata["stratified"] is False
    assert len(first.train.X) == 24
    assert len(first.validation.X) == 8
    assert len(first.test.X) == 8


def test_random_split_reports_when_shuffle_is_disabled() -> None:
    data = pd.DataFrame(
        {
            "signal": range(15),
            "target": [index % 2 for index in range(15)],
            "__mlprep_row_id__": [f"row-{index}" for index in range(15)],
        }
    )
    result = RandomSplitStrategy(
        RandomSplitOptions(
            ratios=_RATIOS,
            shuffle=False,
        )
    ).split(_contract(data))

    assert result.indices.train_indices.tolist() == list(range(9))
    assert any(finding.code == "P2-RANDOM-002" for finding in result.findings)
    assert not any(
        finding.blocks_readiness
        for finding in result.findings
        if finding.code == "P2-RANDOM-002"
    )


def test_stratified_split_preserves_class_proportions_and_input() -> None:
    data = pd.DataFrame(
        {
            "signal": range(100),
            "target": [0] * 50 + [1] * 50,
            "__mlprep_row_id__": [f"row-{index}" for index in range(100)],
        },
        index=pd.Index(range(100), name="row"),
    )
    original = data.copy(deep=True)
    result = StratifiedSplitStrategy(
        StratifiedSplitOptions(
            ratios=_RATIOS,
            random_seed=15,
        )
    ).split(
        _contract(
            data,
            strategy=SplitStrategyName.STRATIFIED,
        )
    )

    for partition in (
        result.train,
        result.validation,
        result.test,
    ):
        assert partition.y.value_counts(normalize=True).to_dict() == {
            0: 0.5,
            1: 0.5,
        }
    assert result.metadata["stratified"] is True
    pd.testing.assert_frame_equal(data, original)


def test_regression_stratified_split_restores_contract_and_source_data() -> None:
    data = pd.DataFrame(
        {
            "signal": np.arange(80, dtype=float),
            "target": np.arange(80, dtype=float) ** 1.2,
            "__mlprep_row_id__": [f"row-{index}" for index in range(80)],
        },
        index=pd.Index(range(80), name="row"),
    )
    original = data.copy(deep=True)
    contract = _contract(
        data,
        strategy=SplitStrategyName.REGRESSION_STRATIFIED,
    )
    strategy = RegressionStratifiedSplitStrategy(
        RegressionStratifiedSplitOptions(
            ratios=_RATIOS,
            number_of_bins=4,
            random_seed=5,
        )
    )

    result = strategy.split(contract)

    assert len(result.indices.all_indices()) == len(data)
    assert result.metadata["regression_stratified"] is True
    assert contract.columns.target_column == "target"
    assert "__mlprep_regression_bin__" not in contract.data.columns
    pd.testing.assert_frame_equal(contract.data, original)


def test_target_and_post_outcome_checks_report_blocking_features() -> None:
    data = pd.DataFrame(
        {
            "signal": [1, 2, 3, 4],
            "target_copy": [0, 1, 0, 1],
            "final_decision": [0, 0, 1, 1],
            "target": [0, 1, 0, 1],
            "__mlprep_row_id__": [f"row-{index}" for index in range(4)],
        }
    )
    contract = _contract(
        data,
        features=["signal", "target_copy", "final_decision"],
    )
    target_findings = TargetLeakageChecker().inspect(contract)
    outcome_findings, suspicious = PostOutcomeLeakageChecker(
        PostOutcomeLeakageOptions(
            declared_post_outcome_columns=["final_decision"],
        )
    ).inspect(contract)

    assert any(
        finding.code == "P2-TARGET-LEAKAGE-002"
        and finding.blocks_readiness
        for finding in target_findings
    )
    assert any(
        finding.code == "P2-POST-OUTCOME-001"
        and finding.blocks_readiness
        for finding in outcome_findings
    )
    assert "final_decision" in suspicious


def test_group_duplicate_row_and_temporal_overlap_guards() -> None:
    grouped_data = pd.DataFrame(
        {
            "signal": range(6),
            "target": [0, 1, 0, 1, 0, 1],
            "group": ["a", "a", "b", "b", "c", "c"],
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="D"),
            "__mlprep_row_id__": [f"row-{index}" for index in range(6)],
        },
        index=pd.Index(range(6), name="row"),
    )
    grouped_contract = _contract(
        grouped_data,
        strategy=SplitStrategyName.GROUP,
        groups=["group"],
        timestamp="timestamp",
    )
    indices = StrategyPartitions(
        train_indices=pd.Index([0, 1, 2]),
        validation_indices=pd.Index([1, 3]),
        test_indices=pd.Index([4, 5]),
    )
    group_findings, overlapping_groups = GroupOverlapChecker().inspect(
        contract=grouped_contract,
        indices=indices,
    )
    temporal_findings, violations = TemporalLeakageChecker().inspect(
        contract=grouped_contract,
        indices=StrategyPartitions(
            train_indices=pd.Index([0, 1]),
            validation_indices=pd.Index([4]),
            test_indices=pd.Index([2, 3, 5]),
        ),
    )
    row_findings, overlapping_rows = RowOverlapChecker().inspect(
        train=DatasetPartition(
            name=PartitionName.TRAIN,
            X=pd.DataFrame({"signal": [1, 2]}, index=[0, 1]),
            y=pd.Series([0, 1], index=[0, 1]),
            row_ids=pd.Series(["r0", "r1"], index=[0, 1]),
        ),
        validation=DatasetPartition(
            name=PartitionName.VALIDATION,
            X=pd.DataFrame({"signal": [2, 3]}, index=[2, 3]),
            y=pd.Series([1, 0], index=[2, 3]),
            row_ids=pd.Series(["r1", "r3"], index=[2, 3]),
        ),
        test=DatasetPartition(
            name=PartitionName.TEST,
            X=pd.DataFrame({"signal": [4]}, index=[4]),
            y=pd.Series([0], index=[4]),
            row_ids=pd.Series(["r4"], index=[4]),
        ),
    )
    duplicate_findings = DuplicateOverlapChecker().inspect(
        train=DatasetPartition(
            name=PartitionName.TRAIN,
            X=pd.DataFrame({"signal": [1, 2]}, index=[0, 1]),
            y=pd.Series([0, 1], index=[0, 1]),
            row_ids=pd.Series(["a", "b"], index=[0, 1]),
        ),
        validation=DatasetPartition(
            name=PartitionName.VALIDATION,
            X=pd.DataFrame({"signal": [2, 3]}, index=[2, 3]),
            y=pd.Series([0, 1], index=[2, 3]),
            row_ids=pd.Series(["c", "d"], index=[2, 3]),
        ),
        test=DatasetPartition(
            name=PartitionName.TEST,
            X=pd.DataFrame({"signal": [5]}, index=[4]),
            y=pd.Series([0], index=[4]),
            row_ids=pd.Series(["e"], index=[4]),
        ),
    )

    assert any(finding.blocks_readiness for finding in group_findings)
    assert "train__validation" in overlapping_groups
    assert any(finding.blocks_readiness for finding in temporal_findings)
    assert violations
    assert any(finding.blocks_readiness for finding in row_findings)
    assert overlapping_rows == {"train__validation": ["r1"]}
    assert any(
        finding.code == "P2-DUPLICATE-OVERLAP-001"
        for finding in duplicate_findings
    )
