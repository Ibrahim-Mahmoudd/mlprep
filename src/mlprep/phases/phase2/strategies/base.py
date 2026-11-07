"""Base interfaces and shared utilities for Phase 2 split strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
)
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionMetadata,
    PartitionName,
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    SplitFinding,
)


@dataclass(slots=True)
class StrategyPartitions:
    """Raw row-index assignments produced by a split strategy."""

    train_indices: pd.Index
    validation_indices: pd.Index
    test_indices: pd.Index

    def __post_init__(self) -> None:
        """Validate index containers."""

        self.train_indices = pd.Index(
            self.train_indices
        )

        self.validation_indices = pd.Index(
            self.validation_indices
        )

        self.test_indices = pd.Index(
            self.test_indices
        )

    def all_indices(self) -> pd.Index:
        """Return all assigned indices."""

        return self.train_indices.append(
            [
                self.validation_indices,
                self.test_indices,
            ]
        )

    def validate_disjoint(self) -> None:
        """Ensure partition indices do not overlap."""

        partition_indices = {
            PartitionName.TRAIN: set(
                self.train_indices.tolist()
            ),
            PartitionName.VALIDATION: set(
                self.validation_indices.tolist()
            ),
            PartitionName.TEST: set(
                self.test_indices.tolist()
            ),
        }

        partition_names = list(
            partition_indices
        )

        for position, left_name in enumerate(
            partition_names
        ):
            left_indices = partition_indices[
                left_name
            ]

            for right_name in partition_names[
                position + 1:
            ]:
                overlap = left_indices.intersection(
                    partition_indices[
                        right_name
                    ]
                )

                if overlap:
                    examples = list(overlap)[:10]

                    raise ValueError(
                        f"Partitions {left_name.value!r} and "
                        f"{right_name.value!r} overlap. "
                        f"Example indices: {examples}"
                    )

    def validate_complete(
        self,
        source_index: pd.Index,
    ) -> None:
        """Ensure every source row is assigned exactly once."""

        self.validate_disjoint()

        source_values = set(
            source_index.tolist()
        )

        assigned_values = set(
            self.all_indices().tolist()
        )

        missing_indices = source_values.difference(
            assigned_values
        )

        unexpected_indices = assigned_values.difference(
            source_values
        )

        if missing_indices:
            examples = list(
                missing_indices
            )[:10]

            raise ValueError(
                "The split did not assign every source row. "
                f"Example missing indices: {examples}"
            )

        if unexpected_indices:
            examples = list(
                unexpected_indices
            )[:10]

            raise ValueError(
                "The split contains indices absent from the "
                f"source dataset. Examples: {examples}"
            )

        if len(self.all_indices()) != len(
            source_index
        ):
            raise ValueError(
                "The number of assigned rows does not match "
                "the source dataset row count."
            )


@dataclass(slots=True)
class StrategyOutput:
    """Complete output produced by one split strategy."""

    strategy: SplitStrategyName

    train: DatasetPartition
    validation: DatasetPartition
    test: DatasetPartition

    indices: StrategyPartitions

    findings: list[SplitFinding]

    metadata: dict[str, Any]

    def partition(
        self,
        name: PartitionName,
    ) -> DatasetPartition:
        """Return one partition by name."""

        if name == PartitionName.TRAIN:
            return self.train

        if name == PartitionName.VALIDATION:
            return self.validation

        if name == PartitionName.TEST:
            return self.test

        raise KeyError(
            f"Partition {name.value!r} is unavailable."
        )


class BaseSplitStrategy(ABC):
    """Abstract interface implemented by every split strategy."""

    strategy_name: SplitStrategyName

    def __init__(
        self,
        *,
        ratios: SplitRatios,
        random_seed: int | None = 42,
    ) -> None:
        """Initialize the split strategy."""

        ratios.validate_total()

        self.ratios = ratios
        self.random_seed = random_seed

    def split(
        self,
        contract: SplitContract,
    ) -> StrategyOutput:
        """Execute the strategy and build dataset partitions."""

        self.validate_contract(contract)

        indices = self.create_indices(
            contract
        )

        indices.validate_complete(
            contract.data.index
        )

        train = self._build_partition(
            name=PartitionName.TRAIN,
            indices=indices.train_indices,
            contract=contract,
        )

        validation = self._build_partition(
            name=PartitionName.VALIDATION,
            indices=indices.validation_indices,
            contract=contract,
        )

        test = self._build_partition(
            name=PartitionName.TEST,
            indices=indices.test_indices,
            contract=contract,
        )

        findings = self.create_findings(
            contract=contract,
            indices=indices,
        )

        return StrategyOutput(
            strategy=self.strategy_name,
            train=train,
            validation=validation,
            test=test,
            indices=indices,
            findings=findings,
            metadata=self.create_metadata(
                contract=contract,
                indices=indices,
            ),
        )

    def validate_contract(
        self,
        contract: SplitContract,
    ) -> None:
        """Validate common split-contract conditions."""

        blocking_findings = [
            finding
            for finding in contract.findings
            if finding.blocks_readiness
        ]

        if blocking_findings:
            codes = [
                finding.code
                for finding in blocking_findings
            ]

            raise ValueError(
                "The split contract contains blocking "
                f"findings: {codes}"
            )

        if contract.data.empty:
            raise ValueError(
                "Cannot split an empty dataset."
            )

        if not contract.feature_columns:
            raise ValueError(
                "Cannot split a dataset without features."
            )

        if (
            contract.target_column
            not in contract.data.columns
        ):
            raise ValueError(
                "The target column does not exist in "
                "the split-contract data."
            )

        if (
            contract.internal_row_id_column
            not in contract.data.columns
        ):
            raise ValueError(
                "The internal row-ID column does not exist."
            )

    @abstractmethod
    def create_indices(
        self,
        contract: SplitContract,
    ) -> StrategyPartitions:
        """Create train, validation, and test indices."""

        raise NotImplementedError

    def create_findings(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> list[SplitFinding]:
        """Create strategy-specific findings."""

        return []

    def create_metadata(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> dict[str, Any]:
        """Create strategy-specific metadata."""

        total_count = len(contract.data)

        return {
            "strategy": self.strategy_name.value,
            "random_seed": self.random_seed,
            "requested_ratios": {
                "train": self.ratios.train,
                "validation": (
                    self.ratios.validation
                ),
                "test": self.ratios.test,
            },
            "actual_row_counts": {
                "train": len(
                    indices.train_indices
                ),
                "validation": len(
                    indices.validation_indices
                ),
                "test": len(
                    indices.test_indices
                ),
            },
            "actual_ratios": {
                "train": self._safe_ratio(
                    len(indices.train_indices),
                    total_count,
                ),
                "validation": self._safe_ratio(
                    len(
                        indices.validation_indices
                    ),
                    total_count,
                ),
                "test": self._safe_ratio(
                    len(indices.test_indices),
                    total_count,
                ),
            },
        }

    def _build_partition(
        self,
        *,
        name: PartitionName,
        indices: pd.Index,
        contract: SplitContract,
    ) -> DatasetPartition:
        """Build one aligned dataset partition."""

        partition_data = contract.data.loc[
            indices
        ].copy()

        x = partition_data.loc[
            :,
            contract.feature_columns,
        ].copy()

        y = partition_data[
            contract.target_column
        ].copy()

        row_ids = partition_data[
            contract.internal_row_id_column
        ].copy()

        metadata = PartitionMetadata(
            name=name,
            row_count=len(partition_data),
            feature_count=len(x.columns),
            row_fraction=self._safe_ratio(
                len(partition_data),
                len(contract.data),
            ),
            target_nonmissing_count=int(
                y.notna().sum()
            ),
            target_missing_count=int(
                y.isna().sum()
            ),
        )

        return DatasetPartition(
            name=name,
            X=x,
            y=y,
            row_ids=row_ids,
            metadata=metadata,
        )

    @staticmethod
    def _safe_ratio(
        numerator: int,
        denominator: int,
    ) -> float:
        """Calculate a ratio safely."""

        if denominator <= 0:
            return 0.0

        return numerator / denominator

    @staticmethod
    def calculate_partition_counts(
        *,
        row_count: int,
        ratios: SplitRatios,
    ) -> tuple[int, int, int]:
        """Calculate deterministic partition row counts."""

        if row_count < 3:
            raise ValueError(
                "At least three rows are required to create "
                "train, validation, and test partitions."
            )

        test_count = max(
            1,
            round(
                row_count * ratios.test
            ),
        )

        validation_count = 0

        if ratios.validation > 0:
            validation_count = max(
                1,
                round(
                    row_count
                    * ratios.validation
                ),
            )

        train_count = (
            row_count
            - validation_count
            - test_count
        )

        if train_count <= 0:
            raise ValueError(
                "Requested split ratios leave no rows for "
                "the training partition."
            )

        return (
            train_count,
            validation_count,
            test_count,
        )