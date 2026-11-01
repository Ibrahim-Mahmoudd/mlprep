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
from mlprep.phases.phase2.output_governance.state import SplitColumnState
from mlprep.phases.phase2.strategies.purged_time_split import (
    PurgedTimeSplitOptions,
    PurgedTimeSplitStrategy,
)


def test_purged_time_split_excludes_gap_rows_and_preserves_order() -> None:
    data = pd.DataFrame(
        {
            "feature": range(40),
            "target": [index % 2 for index in range(40)],
            "timestamp": pd.date_range("2024-01-01", periods=40, freq="D"),
            "__mlprep_row_id__": [f"row-{index}" for index in range(40)],
        },
        index=pd.Index(range(40), name="row"),
    )
    contract = SplitContract(
        data=data,
        phase1_result=None,
        phase1_state=None,
        columns=SplitColumnState(
            target_column="target",
            feature_columns=["feature"],
            timestamp_column="timestamp",
        ),
        options=SplitContractOptions(
            strategy=SplitStrategyName.PURGED_TIME,
            ratios=SplitRatios(
                train=0.6,
                validation=0.2,
                test=0.2,
            ),
            timestamp_column="timestamp",
        ),
        findings=[],
    )
    result = PurgedTimeSplitStrategy(
        PurgedTimeSplitOptions(
            ratios=SplitRatios(
                train=0.6,
                validation=0.2,
                test=0.2,
            ),
            purge_gap="2 days",
            embargo_gap="2 days",
        )
    ).split(contract)

    train_timestamps = data.loc[
        result.indices.train_indices,
        "timestamp",
    ]
    validation_timestamps = data.loc[
        result.indices.validation_indices,
        "timestamp",
    ]
    test_timestamps = data.loc[
        result.indices.test_indices,
        "timestamp",
    ]
    assigned = result.indices.all_indices()
    dropped_rows = set(data.index).difference(
        assigned.tolist()
    )

    assert train_timestamps.max() < validation_timestamps.min()
    assert validation_timestamps.max() < test_timestamps.min()
    assert len(assigned) < len(data)
    assert len(assigned.unique()) == len(assigned)
    assert dropped_rows == {22, 23, 32, 33}
    assert result.metadata["purged"] is True
    assert result.metadata["purge_gap"] == "2 days 00:00:00"
    assert not any(finding.blocks_readiness for finding in result.findings)
