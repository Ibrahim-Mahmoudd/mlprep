"""Group-aware stratified splitting for Phase 2."""

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
class GroupStratifiedSplitOptions:
    """Options controlling group-stratified splitting."""

    ratios: SplitRatios
    random_seed: int = 42

    shuffle_groups: bool = True

    minimum_groups_per_partition: int = 1

    row_balance_weight: float = 1.0
    class_balance_weight: float = 2.0

    distribution_tolerance: float = 0.10
    maximum_ratio_deviation: float = 0.10

    def __post_init__(self) -> None:
        """Validate group-stratified options."""

        self.ratios.validate_total()

        if self.minimum_groups_per_partition < 1:
            raise ValueError(
                "minimum_groups_per_partition must be "
                "at least one."
            )

        if self.row_balance_weight < 0:
            raise ValueError(
                "row_balance_weight cannot be negative."
            )

        if self.class_balance_weight < 0:
            raise ValueError(
                "class_balance_weight cannot be negative."
            )

        if (
            self.row_balance_weight == 0
            and self.class_balance_weight == 0
        ):
            raise ValueError(
                "At least one balancing weight must be "
                "greater than zero."
            )

        if not 0.0 <= self.distribution_tolerance <= 1.0:
            raise ValueError(
                "distribution_tolerance must be between "
                "zero and one."
            )

        if not 0.0 <= self.maximum_ratio_deviation <= 1.0:
            raise ValueError(
                "maximum_ratio_deviation must be between "
                "zero and one."
            )


class GroupStratifiedSplitStrategy(
    BaseSplitStrategy
):
    """Preserve groups while approximating target distributions."""

    strategy_name = (
        SplitStrategyName.GROUP_STRATIFIED
    )

    def __init__(
        self,
        options: GroupStratifiedSplitOptions,
    ) -> None:
        """Initialize the strategy."""

        super().__init__(
            ratios=options.ratios,
            random_seed=options.random_seed,
        )

        self.options = options

    def validate_contract(
        self,
        contract: SplitContract,
    ) -> None:
        """Validate group-stratified requirements."""

        super().validate_contract(contract)

        if not contract.columns.group_columns:
            raise ValueError(
                "Group-stratified splitting requires "
                "at least one group column."
            )

        target = contract.target_series()

        if target.isna().any():
            raise ValueError(
                "Group-stratified splitting requires "
                "a nonmissing target."
            )

        if target.nunique(
            dropna=False
        ) < 2:
            raise ValueError(
                "Group-stratified splitting requires "
                "at least two target classes."
            )

        group_keys = self._build_group_keys(
            contract
        )

        if group_keys.isna().any():
            raise ValueError(
                "Group-stratified splitting cannot run "
                "with missing group values."
            )

        required_partitions = (
            2
            if self.ratios.validation == 0
            else 3
        )

        unique_group_count = int(
            group_keys.nunique(
                dropna=False
            )
        )

        minimum_required_groups = (
            required_partitions
            * self.options
            .minimum_groups_per_partition
        )

        if unique_group_count < minimum_required_groups:
            raise ValueError(
                "The dataset contains too few groups "
                "for the requested group-stratified split."
            )

    def create_indices(
        self,
        contract: SplitContract,
    ) -> StrategyPartitions:
        """Assign whole groups using target-distribution balancing."""

        group_keys = self._build_group_keys(
            contract
        )

        target = contract.target_series()

        classes = target.drop_duplicates().tolist()

        group_statistics = (
            self._build_group_statistics(
                group_keys=group_keys,
                target=target,
                classes=classes,
            )
        )

        random_generator = np.random.default_rng(
            self.options.random_seed
        )

        if self.options.shuffle_groups:
            random_generator.shuffle(
                group_statistics
            )

        group_statistics.sort(
            key=lambda record: (
                record["difficulty"],
                record["row_count"],
            ),
            reverse=True,
        )

        assignments = self._assign_groups(
            group_statistics=group_statistics,
            classes=classes,
            global_class_counts={
                class_value: int(
                    (target == class_value).sum()
                )
                for class_value in classes
            },
            total_row_count=len(contract.data),
        )

        train_groups = set(
            assignments[
                PartitionName.TRAIN
            ]
        )

        validation_groups = set(
            assignments[
                PartitionName.VALIDATION
            ]
        )

        test_groups = set(
            assignments[
                PartitionName.TEST
            ]
        )

        return StrategyPartitions(
            train_indices=contract.data.index[
                group_keys.isin(
                    train_groups
                )
            ],
            validation_indices=contract.data.index[
                group_keys.isin(
                    validation_groups
                )
            ],
            test_indices=contract.data.index[
                group_keys.isin(
                    test_groups
                )
            ],
        )

    def create_findings(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> list[object]:
        """Validate group isolation and target balance."""

        findings: list[SplitFinding] = []

        group_keys = self._build_group_keys(
            contract
        )

        target = contract.target_series()

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
                overlap = partition_groups[
                    left_partition
                ].intersection(
                    partition_groups[
                        right_partition
                    ]
                )

                if not overlap:
                    continue

                findings.append(
                    SplitFinding(
                        code="P2-GROUPSTRAT-001",
                        category=(
                            SplitFindingCategory
                            .GROUP_OVERLAP
                        ),
                        severity=(
                            SplitFindingSeverity.CRITICAL
                        ),
                        component=(
                            "group_stratified_split"
                        ),
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
                            contract.columns
                            .group_columns
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
                                overlap
                            ),
                        },
                    )
                )

        global_distribution = (
            target.value_counts(
                normalize=True,
                dropna=False,
            )
        )

        all_classes = set(
            global_distribution.index.tolist()
        )

        for partition_name, partition_index in (
            partition_indices.items()
        ):
            partition_target = target.loc[
                partition_index
            ]

            partition_distribution = (
                partition_target.value_counts(
                    normalize=True,
                    dropna=False,
                )
            )

            missing_classes = all_classes.difference(
                set(
                    partition_distribution
                    .index.tolist()
                )
            )

            if missing_classes:
                findings.append(
                    SplitFinding(
                        code="P2-GROUPSTRAT-002",
                        category=(
                            SplitFindingCategory
                            .CLASS_DISTRIBUTION
                        ),
                        severity=(
                            SplitFindingSeverity.ERROR
                        ),
                        component=(
                            "group_stratified_split"
                        ),
                        message=(
                            f"Partition "
                            f"{partition_name.value!r} "
                            "is missing one or more "
                            "target classes."
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
                            "missing_classes": [
                                str(value)
                                for value
                                in missing_classes
                            ],
                        },
                    )
                )

            distribution_differences: dict[
                str,
                float,
            ] = {}

            for class_value in all_classes:
                difference = abs(
                    float(
                        partition_distribution.get(
                            class_value,
                            0.0,
                        )
                    )
                    - float(
                        global_distribution.get(
                            class_value,
                            0.0,
                        )
                    )
                )

                if (
                    difference
                    > self.options
                    .distribution_tolerance
                ):
                    distribution_differences[
                        str(class_value)
                    ] = difference

            if distribution_differences:
                findings.append(
                    SplitFinding(
                        code="P2-GROUPSTRAT-003",
                        category=(
                            SplitFindingCategory
                            .CLASS_DISTRIBUTION
                        ),
                        severity=(
                            SplitFindingSeverity.WARNING
                        ),
                        component=(
                            "group_stratified_split"
                        ),
                        message=(
                            f"Partition "
                            f"{partition_name.value!r} "
                            "target distribution exceeds "
                            "the configured tolerance."
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
                            "differences": (
                                distribution_differences
                            ),
                            "tolerance": (
                                self.options
                                .distribution_tolerance
                            ),
                        },
                    )
                )

            actual_ratio = (
                len(partition_index)
                / len(contract.data)
            )

            requested_ratio = {
                PartitionName.TRAIN: (
                    self.ratios.train
                ),
                PartitionName.VALIDATION: (
                    self.ratios.validation
                ),
                PartitionName.TEST: (
                    self.ratios.test
                ),
            }[partition_name]

            deviation = abs(
                actual_ratio - requested_ratio
            )

            if (
                deviation
                > self.options
                .maximum_ratio_deviation
            ):
                findings.append(
                    SplitFinding(
                        code="P2-GROUPSTRAT-004",
                        category=(
                            SplitFindingCategory
                            .SPLIT_SIZE
                        ),
                        severity=(
                            SplitFindingSeverity.WARNING
                        ),
                        component=(
                            "group_stratified_split"
                        ),
                        message=(
                            f"Partition "
                            f"{partition_name.value!r} "
                            "deviates from its requested "
                            "row ratio."
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
                            "actual_ratio": (
                                actual_ratio
                            ),
                            "deviation": deviation,
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
        """Create group-stratified split metadata."""

        metadata = super().create_metadata(
            contract=contract,
            indices=indices,
        )

        target = contract.target_series()

        group_keys = self._build_group_keys(
            contract
        )

        partition_indices = {
            "train": indices.train_indices,
            "validation": (
                indices.validation_indices
            ),
            "test": indices.test_indices,
        }

        metadata.update(
            {
                "shuffle": (
                    self.options.shuffle_groups
                ),
                "stratified": True,
                "grouped": True,
                "temporal": False,
                "group_columns": list(
                    contract.columns.group_columns
                ),
                "partition_class_distributions": {
                    partition_name: {
                        str(class_value): float(
                            ratio
                        )
                        for class_value, ratio
                        in target.loc[
                            partition_index
                        ].value_counts(
                            normalize=True,
                            dropna=False,
                        ).items()
                    }
                    for partition_name, partition_index
                    in partition_indices.items()
                },
                "unique_group_counts": {
                    partition_name: int(
                        group_keys.loc[
                            partition_index
                        ].nunique(
                            dropna=False
                        )
                    )
                    for partition_name, partition_index
                    in partition_indices.items()
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

    @staticmethod
    def _build_group_statistics(
        *,
        group_keys: pd.Series,
        target: pd.Series,
        classes: list[object],
    ) -> list[dict[str, Any]]:
        """Build size and class-count statistics per group."""

        statistics: list[dict[str, Any]] = []

        for group_key in group_keys.drop_duplicates():
            group_mask = (
                group_keys == group_key
            )

            group_target = target.loc[
                group_mask
            ]

            class_counts = {
                class_value: int(
                    (
                        group_target
                        == class_value
                    ).sum()
                )
                for class_value in classes
            }

            row_count = int(
                group_mask.sum()
            )

            maximum_class_count = max(
                class_counts.values(),
                default=0,
            )

            class_concentration = (
                maximum_class_count
                / row_count
                if row_count > 0
                else 0.0
            )

            statistics.append(
                {
                    "group_key": group_key,
                    "row_count": row_count,
                    "class_counts": (
                        class_counts
                    ),
                    "difficulty": (
                        row_count
                        * class_concentration
                    ),
                }
            )

        return statistics

    def _assign_groups(
        self,
        *,
        group_statistics: list[
            dict[str, Any]
        ],
        classes: list[object],
        global_class_counts: dict[
            object,
            int,
        ],
        total_row_count: int,
    ) -> dict[
        PartitionName,
        list[object],
    ]:
        """Greedily assign groups by row and class balance."""

        active_partitions = [
            PartitionName.TRAIN,
            PartitionName.TEST,
        ]

        if self.ratios.validation > 0:
            active_partitions.insert(
                1,
                PartitionName.VALIDATION,
            )

        assignments = {
            PartitionName.TRAIN: [],
            PartitionName.VALIDATION: [],
            PartitionName.TEST: [],
        }

        assigned_rows = {
            partition: 0
            for partition in active_partitions
        }

        assigned_class_counts = {
            partition: {
                class_value: 0
                for class_value in classes
            }
            for partition in active_partitions
        }

        ratio_by_partition = {
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

        target_rows = {
            partition: (
                total_row_count
                * ratio_by_partition[
                    partition
                ]
            )
            for partition in active_partitions
        }

        target_class_counts = {
            partition: {
                class_value: (
                    global_class_counts[
                        class_value
                    ]
                    * ratio_by_partition[
                        partition
                    ]
                )
                for class_value in classes
            }
            for partition in active_partitions
        }

        statistics = list(
            group_statistics
        )

        for partition in active_partitions:
            for _ in range(
                self.options
                .minimum_groups_per_partition
            ):
                if not statistics:
                    raise ValueError(
                        "Not enough groups to initialize "
                        "all partitions."
                    )

                group_record = statistics.pop(0)

                self._apply_assignment(
                    partition=partition,
                    group_record=group_record,
                    assignments=assignments,
                    assigned_rows=assigned_rows,
                    assigned_class_counts=(
                        assigned_class_counts
                    ),
                )

        for group_record in statistics:
            best_partition = min(
                active_partitions,
                key=lambda partition: (
                    self._assignment_cost(
                        partition=partition,
                        group_record=group_record,
                        assigned_rows=assigned_rows,
                        assigned_class_counts=(
                            assigned_class_counts
                        ),
                        target_rows=target_rows,
                        target_class_counts=(
                            target_class_counts
                        ),
                        classes=classes,
                    )
                ),
            )

            self._apply_assignment(
                partition=best_partition,
                group_record=group_record,
                assignments=assignments,
                assigned_rows=assigned_rows,
                assigned_class_counts=(
                    assigned_class_counts
                ),
            )

        return assignments

    def _assignment_cost(
        self,
        *,
        partition: PartitionName,
        group_record: dict[str, Any],
        assigned_rows: dict[
            PartitionName,
            int,
        ],
        assigned_class_counts: dict[
            PartitionName,
            dict[object, int],
        ],
        target_rows: dict[
            PartitionName,
            float,
        ],
        target_class_counts: dict[
            PartitionName,
            dict[object, float],
        ],
        classes: list[object],
    ) -> float:
        """Calculate the cost of assigning one group."""

        projected_rows = (
            assigned_rows[partition]
            + int(group_record["row_count"])
        )

        row_cost = abs(
            projected_rows
            - target_rows[partition]
        ) / max(
            target_rows[partition],
            1.0,
        )

        class_cost = 0.0

        group_class_counts: dict[
            object,
            int,
        ] = group_record[
            "class_counts"
        ]

        for class_value in classes:
            projected_class_count = (
                assigned_class_counts[
                    partition
                ][class_value]
                + group_class_counts[
                    class_value
                ]
            )

            target_count = (
                target_class_counts[
                    partition
                ][class_value]
            )

            class_cost += abs(
                projected_class_count
                - target_count
            ) / max(
                target_count,
                1.0,
            )

        class_cost /= max(
            len(classes),
            1,
        )

        return (
            self.options.row_balance_weight
            * row_cost
            + self.options.class_balance_weight
            * class_cost
        )

    @staticmethod
    def _apply_assignment(
        *,
        partition: PartitionName,
        group_record: dict[str, Any],
        assignments: dict[
            PartitionName,
            list[object],
        ],
        assigned_rows: dict[
            PartitionName,
            int,
        ],
        assigned_class_counts: dict[
            PartitionName,
            dict[object, int],
        ],
    ) -> None:
        """Apply one group assignment."""

        assignments[partition].append(
            group_record["group_key"]
        )

        assigned_rows[partition] += int(
            group_record["row_count"]
        )

        group_class_counts: dict[
            object,
            int,
        ] = group_record[
            "class_counts"
        ]

        for class_value, count in (
            group_class_counts.items()
        ):
            assigned_class_counts[
                partition
            ][class_value] += count