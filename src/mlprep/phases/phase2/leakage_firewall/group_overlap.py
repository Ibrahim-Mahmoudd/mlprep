"""Cross-partition group-overlap detection for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
)
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
)
from mlprep.phases.phase2.output_governance.result import (
    SplitFinding,
)
from mlprep.phases.phase2.strategies.base import (
    StrategyPartitions,
)


@dataclass(slots=True)
class GroupOverlapOptions:
    """Options controlling group-overlap checks."""

    enabled: bool = True

    block_on_overlap: bool = True
    maximum_examples: int = 10

    def __post_init__(self) -> None:
        """Validate group-overlap options."""

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )


class GroupOverlapChecker:
    """Detect groups shared across dataset partitions."""

    component_name = "group_overlap"

    def __init__(
        self,
        options: GroupOverlapOptions | None = None,
    ) -> None:
        """Initialize the group-overlap checker."""

        self.options = options or GroupOverlapOptions()

    def inspect(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> tuple[
        list[SplitFinding],
        dict[str, list[str | int]],
    ]:
        """Inspect group isolation across all partitions."""

        if (
            not self.options.enabled
            or not contract.columns.group_columns
        ):
            return [], {}

        group_keys = self._group_keys(contract)

        partition_indices = {
            PartitionName.TRAIN: indices.train_indices,
            PartitionName.VALIDATION: (
                indices.validation_indices
            ),
            PartitionName.TEST: indices.test_indices,
        }

        partition_groups = {
            partition_name: set(
                group_keys.loc[
                    partition_index
                ].tolist()
            )
            for partition_name, partition_index
            in partition_indices.items()
        }

        findings: list[SplitFinding] = []
        overlap_state: dict[
            str,
            list[str | int],
        ] = {}

        names = list(partition_groups)

        for position, left_name in enumerate(names):
            for right_name in names[position + 1:]:
                overlap = partition_groups[
                    left_name
                ].intersection(
                    partition_groups[right_name]
                )

                if not overlap:
                    continue

                serialized_overlap = [
                    self._serialize_group(group)
                    for group in overlap
                ]

                pair_name = (
                    f"{left_name.value}__"
                    f"{right_name.value}"
                )

                overlap_state[pair_name] = (
                    serialized_overlap
                )

                findings.append(
                    SplitFinding(
                        code="P2-GROUP-OVERLAP-001",
                        category=(
                            SplitFindingCategory.GROUP_OVERLAP
                        ),
                        severity=SplitFindingSeverity.CRITICAL,
                        component=self.component_name,
                        message=(
                            f"Partitions {left_name.value!r} and "
                            f"{right_name.value!r} share one or "
                            "more groups."
                        ),
                        partitions=[
                            left_name,
                            right_name,
                        ],
                        columns=list(
                            contract.columns.group_columns
                        ),
                        affected_row_count=len(overlap),
                        example_row_ids=serialized_overlap[
                            : self.options.maximum_examples
                        ],
                        recommended_action=(
                            SplitFindingAction.RESPLIT
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=(
                            self.options.block_on_overlap
                        ),
                    )
                )

        return findings, overlap_state

    @staticmethod
    def _group_keys(
        contract: SplitContract,
    ) -> pd.Series:
        """Build one hashable group key per source row."""

        group_columns = contract.columns.group_columns

        if len(group_columns) == 1:
            return contract.data[
                group_columns[0]
            ].copy()

        group_frame = contract.data.loc[
            :,
            group_columns,
        ]

        return pd.Series(
            list(
                group_frame.itertuples(
                    index=False,
                    name=None,
                )
            ),
            index=contract.data.index,
            dtype="object",
        )

    @staticmethod
    def _serialize_group(
        value: Any,
    ) -> str | int:
        """Convert a group key into a report-safe value."""

        if isinstance(value, (str, int)):
            return value

        return str(value)
