"""Group-aware dataset splitting for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
)
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.result import (
    SplitFinding,
)
from mlprep.phases.phase2.strategies.base import (
    BaseSplitStrategy,
    StrategyPartitions,
)


@dataclass(slots=True)
class GroupSplitOptions:
    """Options controlling group-aware splitting."""

    ratios: SplitRatios
    random_seed: int = 42

    shuffle_groups: bool = True

    balance_by_row_count: bool = True

    minimum_groups_per_partition: int = 1

    maximum_ratio_deviation: float = 0.10

    def __post_init__(self) -> None:
        """Validate group-split options."""

        self.ratios.validate_total()

        if self.minimum_groups_per_partition < 1:
            raise ValueError(
                "minimum_groups_per_partition must be "
                "at least one."
            )

        if not 0.0 <= self.maximum_ratio_deviation <= 1.0:
            raise ValueError(
                "maximum_ratio_deviation must be between "
                "zero and one."
            )


class GroupSplitStrategy(BaseSplitStrategy):
    """Split data while keeping each group in one partition."""

    strategy_name = SplitStrategyName.GROUP

    def __init__(
        self,
        options: GroupSplitOptions,
    ) -> None:
        """Initialize the group split strategy."""

        super().__init__(
            ratios=options.ratios,
            random_seed=options.random_seed,
        )

        self.options = options

    def validate_contract(
        self,
        contract: SplitContract,
    ) -> None:
        """Validate requirements for group splitting."""

        super().validate_contract(contract)

        group_columns = (
            contract.columns.group_columns
        )

        if not group_columns:
            raise ValueError(
                "Group splitting requires at least one "
                "group column."
            )

        missing_columns = [
            column
            for column in group_columns
            if column not in contract.data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Group columns are missing from the dataset: "
                f"{missing_columns}"
            )

        missing_group_mask = contract.data[
            group_columns
        ].isna().any(axis=1)

        if bool(missing_group_mask.any()):
            raise ValueError(
                "Group splitting cannot run while group "
                "columns contain missing values."
            )

        group_keys = self._build_group_keys(
            contract
        )

        unique_group_count = int(
            group_keys.nunique(dropna=False)
        )

        required_partitions = (
            2
            if self.ratios.validation == 0
            else 3
        )

        minimum_required_groups = (
            required_partitions
            * self.options
            .minimum_groups_per_partition
        )

        if unique_group_count < minimum_required_groups:
            raise ValueError(
                "The dataset contains too few unique groups "
                "for the requested split. "
                f"Required at least {minimum_required_groups}, "
                f"received {unique_group_count}."
            )

    def create_indices(
        self,
        contract: SplitContract,
    ) -> StrategyPartitions:
        """Assign complete groups to partitions."""

        group_keys = self._build_group_keys(
            contract
        )

        group_sizes = group_keys.value_counts(
            sort=False,
            dropna=False,
        )

        group_records = [
            (
                group_key,
                int(group_size),
            )
            for group_key, group_size
            in group_sizes.items()
        ]

        random_generator = np.random.default_rng(
            self.options.random_seed
        )

        if self.options.shuffle_groups:
            random_generator.shuffle(
                group_records
            )

        if self.options.balance_by_row_count:
            group_records.sort(
                key=lambda item: item[1],
                reverse=True,
            )

        group_assignments = self._assign_groups(
            group_records=group_records,
            total_row_count=len(contract.data),
        )

        train_group_keys = set(
            group_assignments[
                PartitionName.TRAIN
            ]
        )

        validation_group_keys = set(
            group_assignments[
                PartitionName.VALIDATION
            ]
        )

        test_group_keys = set(
            group_assignments[
                PartitionName.TEST
            ]
        )

        train_mask = group_keys.isin(
            train_group_keys
        )

        validation_mask = group_keys.isin(
            validation_group_keys
        )

        test_mask = group_keys.isin(
            test_group_keys
        )

        return StrategyPartitions(
            train_indices=contract.data.index[
                train_mask
            ],
            validation_indices=contract.data.index[
                validation_mask
            ],
            test_indices=contract.data.index[
                test_mask
            ],
        )

    def create_findings(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> list[SplitFinding]:
        """Create group-split validation findings."""

        findings: list[SplitFinding] = []

        group_keys = self._build_group_keys(
            contract
        )

        partition_indices = {
            PartitionName.TRAIN: (
                indices.train_indices
            ),
            PartitionName.VALIDATION: (
                indices.validation_indices
            ),
            PartitionName.TEST: (
                indices.test_indices
            ),
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

        partition_names = list(
            partition_groups
        )

        for position, left_partition in enumerate(
            partition_names
        ):
            for right_partition in partition_names[
                position + 1:
            ]:
                overlapping_groups = (
                    partition_groups[
                        left_partition
                    ].intersection(
                        partition_groups[
                            right_partition
                        ]
                    )
                )

                if not overlapping_groups:
                    continue

                findings.append(
                    SplitFinding(
                        code="P2-GROUP-001",
                        category=(
                            SplitFindingCategory
                            .GROUP_OVERLAP
                        ),
                        severity=(
                            SplitFindingSeverity.CRITICAL
                        ),
                        component="group_split",
                        message=(
                            f"Partitions "
                            f"{left_partition.value!r} and "
                            f"{right_partition.value!r} "
                            "contain overlapping groups."
                        ),
                        partitions=[
                            left_partition,
                            right_partition,
                        ],
                        columns=list(
                            contract.columns.group_columns
                        ),
                        recommended_action=(
                            SplitFindingAction.RESPLIT
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=True,
                        details={
                            "overlap_count": len(
                                overlapping_groups
                            ),
                            "examples": [
                                str(value)
                                for value
                                in list(
                                    overlapping_groups
                                )[:10]
                            ],
                        },
                    )
                )

        total_row_count = len(
            contract.data
        )

        requested_ratios = {
            PartitionName.TRAIN: (
                self.ratios.train
            ),
            PartitionName.VALIDATION: (
                self.ratios.validation
            ),
            PartitionName.TEST: (
                self.ratios.test
            ),
        }

        for partition_name, partition_index in (
            partition_indices.items()
        ):
            actual_ratio = (
                len(partition_index)
                / total_row_count
                if total_row_count
                else 0.0
            )

            requested_ratio = requested_ratios[
                partition_name
            ]

            deviation = abs(
                actual_ratio - requested_ratio
            )

            if (
                deviation
                <= self.options
                .maximum_ratio_deviation
            ):
                continue

            findings.append(
                SplitFinding(
                    code="P2-GROUP-002",
                    category=(
                        SplitFindingCategory
                        .SPLIT_SIZE
                    ),
                    severity=(
                        SplitFindingSeverity.WARNING
                    ),
                    component="group_split",
                    message=(
                        f"Partition "
                        f"{partition_name.value!r} "
                        "deviates from its requested ratio "
                        "because complete groups cannot be split."
                    ),
                    partitions=[
                        partition_name
                    ],
                    recommended_action=(
                        SplitFindingAction.WARN
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=False,
                    details={
                        "requested_ratio": (
                            requested_ratio
                        ),
                        "actual_ratio": actual_ratio,
                        "deviation": deviation,
                        "maximum_deviation": (
                            self.options
                            .maximum_ratio_deviation
                        ),
                    },
                )
            )

        for partition_name, groups in (
            partition_groups.items()
        ):
            if (
                len(groups)
                >= self.options
                .minimum_groups_per_partition
            ):
                continue

            findings.append(
                SplitFinding(
                    code="P2-GROUP-003",
                    category=(
                        SplitFindingCategory
                        .GROUP_DISTRIBUTION
                    ),
                    severity=(
                        SplitFindingSeverity.ERROR
                    ),
                    component="group_split",
                    message=(
                        f"Partition "
                        f"{partition_name.value!r} "
                        "contains too few unique groups."
                    ),
                    partitions=[
                        partition_name
                    ],
                    recommended_action=(
                        SplitFindingAction.RESPLIT
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                    details={
                        "group_count": len(groups),
                        "minimum_group_count": (
                            self.options
                            .minimum_groups_per_partition
                        ),
                    },
                )
            )

        return findings

    def create_metadata(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> dict[str, Any]:
        """Create metadata for the group split."""

        metadata = super().create_metadata(
            contract=contract,
            indices=indices,
        )

        group_keys = self._build_group_keys(
            contract
        )

        metadata.update(
            {
                "shuffle": (
                    self.options.shuffle_groups
                ),
                "stratified": False,
                "grouped": True,
                "temporal": False,
                "group_columns": list(
                    contract.columns.group_columns
                ),
                "unique_group_counts": {
                    "train": int(
                        group_keys.loc[
                            indices.train_indices
                        ].nunique(
                            dropna=False
                        )
                    ),
                    "validation": int(
                        group_keys.loc[
                            indices.validation_indices
                        ].nunique(
                            dropna=False
                        )
                    ),
                    "test": int(
                        group_keys.loc[
                            indices.test_indices
                        ].nunique(
                            dropna=False
                        )
                    ),
                },
            }
        )

        return metadata

    def _build_group_keys(
        self,
        contract: SplitContract,
    ) -> pd.Series:
        """Build one stable group key per row."""

        group_columns = (
            contract.columns.group_columns
        )

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
            name="__mlprep_group_key__",
            dtype="object",
        )

    def _assign_groups(
        self,
        *,
        group_records: list[
            tuple[object, int]
        ],
        total_row_count: int,
    ) -> dict[
        PartitionName,
        list[object],
    ]:
        """Assign groups using a greedy row-balance strategy."""

        assignments: dict[
            PartitionName,
            list[object],
        ] = {
            PartitionName.TRAIN: [],
            PartitionName.VALIDATION: [],
            PartitionName.TEST: [],
        }

        assigned_row_counts = {
            PartitionName.TRAIN: 0,
            PartitionName.VALIDATION: 0,
            PartitionName.TEST: 0,
        }

        target_row_counts = {
            PartitionName.TRAIN: (
                total_row_count
                * self.ratios.train
            ),
            PartitionName.VALIDATION: (
                total_row_count
                * self.ratios.validation
            ),
            PartitionName.TEST: (
                total_row_count
                * self.ratios.test
            ),
        }

        active_partitions = [
            PartitionName.TRAIN,
            PartitionName.TEST,
        ]

        if self.ratios.validation > 0:
            active_partitions.insert(
                1,
                PartitionName.VALIDATION,
            )

        for partition_name in active_partitions:
            for _ in range(
                self.options
                .minimum_groups_per_partition
            ):
                if not group_records:
                    raise ValueError(
                        "Not enough groups to satisfy "
                        "minimum partition requirements."
                    )

                group_key, group_size = (
                    group_records.pop(0)
                )

                assignments[
                    partition_name
                ].append(group_key)

                assigned_row_counts[
                    partition_name
                ] += group_size

        for group_key, group_size in group_records:
            partition_name = min(
                active_partitions,
                key=lambda candidate: (
                    assigned_row_counts[
                        candidate
                    ]
                    / max(
                        target_row_counts[
                            candidate
                        ],
                        1.0,
                    )
                ),
            )

            assignments[
                partition_name
            ].append(group_key)

            assigned_row_counts[
                partition_name
            ] += group_size

        return assignments
