from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase2.leakage_firewall.row_overlap import (
    RowOverlapChecker,
)
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)


def _partition(
    name: PartitionName,
    row_ids: list[str],
) -> DatasetPartition:
    index = pd.RangeIndex(len(row_ids))
    return DatasetPartition(
        name=name,
        X=pd.DataFrame({"feature": range(len(row_ids))}, index=index),
        y=pd.Series([0] * len(row_ids), index=index, name="target"),
        row_ids=pd.Series(row_ids, index=index, dtype="string"),
    )


def test_row_overlap_checker_reports_shared_ids_across_partitions() -> None:
    findings, overlaps = RowOverlapChecker().inspect(
        train=_partition(PartitionName.TRAIN, ["a", "b"]),
        validation=_partition(PartitionName.VALIDATION, ["b", "c"]),
        test=_partition(PartitionName.TEST, ["d"]),
    )

    assert len(findings) == 1
    assert findings[0].code == "P2-ROW-OVERLAP-001"
    assert findings[0].blocks_readiness
    assert overlaps == {"train__validation": ["b"]}


def test_row_overlap_checker_accepts_disjoint_partitions() -> None:
    findings, overlaps = RowOverlapChecker().inspect(
        train=_partition(PartitionName.TRAIN, ["a"]),
        validation=_partition(PartitionName.VALIDATION, ["b"]),
        test=_partition(PartitionName.TEST, ["c"]),
    )

    assert findings == []
    assert overlaps == {}


def test_dataset_partition_rejects_misaligned_indices() -> None:
    with pytest.raises(ValueError, match="indices are not aligned"):
        DatasetPartition(
            name=PartitionName.TRAIN,
            X=pd.DataFrame({"feature": [1]}, index=[4]),
            y=pd.Series([0], index=[5]),
            row_ids=pd.Series(["row-1"], index=[4]),
        )
