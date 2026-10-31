"""Predefined dataset splitting for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
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
class PredefinedSplitOptions:
    """Options controlling predefined partition assignments."""

    split_column: str

    train_values: set[object] = field(
        default_factory=lambda: {
            "train",
        }
    )

    validation_values: set[object] = field(
        default_factory=lambda: {
            "validation",
            "val",
        }
    )

    test_values: set[object] = field(
        default_factory=lambda: {
            "test",
        }
    )

    case_sensitive: bool = False
    require_complete_assignment: bool = True

    def __post_init__(self) -> None:
        """Validate predefined-split options."""

        if not self.split_column.strip():
            raise ValueError(
                "split_column cannot be empty."
            )

        value_sets = {
            "train": self.train_values,
            "validation": self.validation_values,
            "test": self.test_values,
        }

        for name, values in value_sets.items():
            if not values:
                raise ValueError(
                    f"{name}_values cannot be empty."
                )

        normalized_sets = {
            name: {
                self._normalize_value(value)
                for value in values
            }
            for name, values in value_sets.items()
        }

        partition_names = list(
            normalized_sets
        )

        for position, left_name in enumerate(
            partition_names
        ):
            for right_name in partition_names[
                position + 1:
            ]:
                overlap = normalized_sets[
                    left_name
                ].intersection(
                    normalized_sets[
                        right_name
                    ]
                )

                if overlap:
                    raise ValueError(
                        "Predefined partition values overlap "
                        f"between {left_name!r} and "
                        f"{right_name!r}: {overlap}"
                    )

    def _normalize_value(
        self,
        value: object,
    ) -> object:
        """Normalize one configured partition value."""

        if (
            isinstance(value, str)
            and not self.case_sensitive
        ):
            return value.strip().casefold()

        return value


class PredefinedSplitStrategy(
    BaseSplitStrategy
):
    """Use a pre-existing partition-assignment column."""

    strategy_name = SplitStrategyName.PREDEFINED

    def __init__(
        self,
        options: PredefinedSplitOptions,
    ) -> None:
        """Initialize the predefined strategy."""

        super().__init__(
            ratios=SplitRatios(),
            random_seed=None,
        )

        self.options = options

    def validate_contract(
        self,
        contract: SplitContract,
    ) -> None:
        """Validate the predefined split column."""

        super().validate_contract(contract)

        if (
            self.options.split_column
            not in contract.data.columns
        ):
            raise ValueError(
                f"Predefined split column "
                f"{self.options.split_column!r} "
                "does not exist."
            )

    def create_indices(
        self,
        contract: SplitContract,
    ) -> StrategyPartitions:
        """Create indices from predefined partition values."""

        split_values = contract.data[
            self.options.split_column
        ].map(
            self._normalize_value
        )

        train_values = {
            self._normalize_value(value)
            for value in self.options.train_values
        }

        validation_values = {
            self._normalize_value(value)
            for value
            in self.options.validation_values
        }

        test_values = {
            self._normalize_value(value)
            for value in self.options.test_values
        }

        train_mask = split_values.isin(
            train_values
        )

        validation_mask = split_values.isin(
            validation_values
        )

        test_mask = split_values.isin(
            test_values
        )

        assigned_mask = (
            train_mask
            | validation_mask
            | test_mask
        )

        if (
            self.options.require_complete_assignment
            and not bool(
                assigned_mask.all()
            )
        ):
            unassigned_values = (
                split_values.loc[
                    ~assigned_mask
                ]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                "The predefined split contains unassigned "
                f"values: {unassigned_values}"
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
    ) -> list[object]:
        """Validate predefined partition sizes."""

        findings: list[SplitFinding] = []

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

        for partition_name, partition_index in (
            partition_indices.items()
        ):
            if len(partition_index) > 0:
                continue

            findings.append(
                SplitFinding(
                    code="P2-PREDEFINED-001",
                    category=(
                        SplitFindingCategory
                        .SPLIT_SIZE
                    ),
                    severity=(
                        SplitFindingSeverity.ERROR
                    ),
                    component="predefined_split",
                    message=(
                        f"Predefined partition "
                        f"{partition_name.value!r} is empty."
                    ),
                    partitions=[partition_name],
                    recommended_action=(
                        SplitFindingAction.RESPLIT
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        return findings

    def create_metadata(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> dict[str, Any]:
        """Create predefined-split metadata."""

        metadata = super().create_metadata(
            contract=contract,
            indices=indices,
        )

        metadata.update(
            {
                "strategy": (
                    self.strategy_name.value
                ),
                "shuffle": False,
                "stratified": False,
                "grouped": False,
                "temporal": False,
                "predefined": True,
                "split_column": (
                    self.options.split_column
                ),
            }
        )

        return metadata

    def _normalize_value(
        self,
        value: object,
    ) -> object:
        """Normalize one source partition value."""

        if (
            isinstance(value, str)
            and not self.options.case_sensitive
        ):
            return value.strip().casefold()

        return value