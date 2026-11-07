"""Group-distribution validation for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

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
class GroupDistributionValidationOptions:
    """Options controlling group distribution validation."""

    enabled: bool = True

    require_no_group_overlap: bool = True
    minimum_groups_per_partition: int = 1

    maximum_largest_group_fraction: float = 1.0

    def __post_init__(self) -> None:
        """Validate group-distribution options."""

        if self.minimum_groups_per_partition < 1:
            raise ValueError(
                "minimum_groups_per_partition must be "
                "at least one."
            )

        if not (
            0.0
            < self.maximum_largest_group_fraction
            <= 1.0
        ):
            raise ValueError(
                "maximum_largest_group_fraction must be "
                "greater than zero and at most one."
            )


class GroupDistributionValidator:
    """Validate group isolation and concentration."""

    component_name = "group_distribution"

    def __init__(
        self,
        options: (
            GroupDistributionValidationOptions | None
        ) = None,
    ) -> None:
        """Initialize the group-distribution validator."""

        self.options = (
            options
            or GroupDistributionValidationOptions()
        )

    def validate(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> list[object]:
        """Validate groups across all partitions."""

        if (
            not self.options.enabled
            or not contract.columns.group_columns
        ):
            return []

        findings: list[SplitFinding] = []

        group_keys = self._group_keys(
            contract
        )

        partition_indices = {
            PartitionName.TRAIN: indices.train_indices,
            PartitionName.VALIDATION: (
                indices.validation_indices
            ),
            PartitionName.TEST: indices.test_indices,
        }

        partition_groups = {
            name: set(
                group_keys.loc[
                    partition_index
                ].tolist()
            )
            for name, partition_index
            in partition_indices.items()
        }

        names = list(partition_groups)

        if self.options.require_no_group_overlap:
            for position, left_name in enumerate(
                names
            ):
                for right_name in names[
                    position + 1:
                ]:
                    overlap = partition_groups[
                        left_name
                    ].intersection(
                        partition_groups[right_name]
                    )

                    if not overlap:
                        continue

                    findings.append(
                        SplitFinding(
                            code="P2-GROUP-DIST-001",
                            category=(
                                SplitFindingCategory
                                .GROUP_DISTRIBUTION
                            ),
                            severity=(
                                SplitFindingSeverity.CRITICAL
                            ),
                            component=self.component_name,
                            message=(
                                f"Partitions "
                                f"{left_name.value!r} and "
                                f"{right_name.value!r} share "
                                "one or more groups."
                            ),
                            partitions=[
                                left_name,
                                right_name,
                            ],
                            columns=list(
                                contract.columns
                                .group_columns
                            ),
                            affected_row_count=len(
                                overlap
                            ),
                            recommended_action=(
                                SplitFindingAction.RESPLIT
                            ),
                            applied_action=(
                                SplitFindingAction.REPORT
                            ),
                            blocks_readiness=True,
                        )
                    )

        for name, partition_index in (
            partition_indices.items()
        ):
            partition_group_keys = group_keys.loc[
                partition_index
            ]

            unique_group_count = int(
                partition_group_keys.nunique(
                    dropna=False
                )
            )

            if (
                unique_group_count
                < self.options
                .minimum_groups_per_partition
            ):
                findings.append(
                    SplitFinding(
                        code="P2-GROUP-DIST-002",
                        category=(
                            SplitFindingCategory
                            .GROUP_DISTRIBUTION
                        ),
                        severity=SplitFindingSeverity.ERROR,
                        component=self.component_name,
                        message=(
                            f"Partition {name.value!r} "
                            "contains too few unique groups."
                        ),
                        partitions=[name],
                        recommended_action=(
                            SplitFindingAction.RESPLIT
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=True,
                        details={
                            "unique_group_count": (
                                unique_group_count
                            ),
                            "minimum_group_count": (
                                self.options
                                .minimum_groups_per_partition
                            ),
                        },
                    )
                )

            group_counts = (
                partition_group_keys.value_counts(
                    dropna=False
                )
            )

            if group_counts.empty:
                continue

            largest_group_fraction = (
                int(group_counts.iloc[0])
                / len(partition_group_keys)
            )

            if (
                largest_group_fraction
                > self.options
                .maximum_largest_group_fraction
            ):
                findings.append(
                    SplitFinding(
                        code="P2-GROUP-DIST-003",
                        category=(
                            SplitFindingCategory
                            .GROUP_DISTRIBUTION
                        ),
                        severity=SplitFindingSeverity.WARNING,
                        component=self.component_name,
                        message=(
                            f"Partition {name.value!r} is "
                            "dominated by one group."
                        ),
                        partitions=[name],
                        recommended_action=(
                            SplitFindingAction.WARN
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=False,
                        details={
                            "largest_group_fraction": (
                                largest_group_fraction
                            ),
                            "maximum_fraction": (
                                self.options
                                .maximum_largest_group_fraction
                            ),
                        },
                    )
                )

        return findings

    @staticmethod
    def _group_keys(
        contract: SplitContract,
    ) -> pd.Series:
        """Build one group key per row."""

        group_columns = (
            contract.columns.group_columns
        )

        if len(group_columns) == 1:
            return contract.data[
                group_columns[0]
            ].copy()

        return pd.Series(
            list(
                contract.data.loc[
                    :,
                    group_columns,
                ].itertuples(
                    index=False,
                    name=None,
                )
            ),
            index=contract.data.index,
            dtype="object",
        )
