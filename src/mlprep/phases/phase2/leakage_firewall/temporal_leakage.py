"""Temporal leakage detection across Phase 2 partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mlprep.phases.phase2.contract.split_contract import SplitContract
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
)
from mlprep.phases.phase2.output_governance.result import SplitFinding
from mlprep.phases.phase2.strategies.base import StrategyPartitions


@dataclass(slots=True)
class TemporalLeakageOptions:
    """Options controlling temporal leakage detection."""

    enabled: bool = True

    require_train_before_validation: bool = True
    require_validation_before_test: bool = True
    require_train_before_test: bool = True

    allow_equal_boundary_timestamps: bool = False

    block_on_temporal_violation: bool = True

    maximum_examples: int = 10

    def __post_init__(self) -> None:
        """Validate temporal leakage options."""

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )


class TemporalLeakageChecker:
    """Detect invalid temporal ordering between partitions."""

    component_name = "temporal_leakage"

    def __init__(
        self,
        options: TemporalLeakageOptions | None = None,
    ) -> None:
        """Initialize the temporal leakage checker."""

        self.options = options or TemporalLeakageOptions()

    def inspect(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> tuple[
        list[SplitFinding],
        list[dict[str, Any]],
    ]:
        """Inspect chronological ordering across partitions."""

        if not self.options.enabled:
            return [], []

        timestamp_column = contract.columns.timestamp_column

        if timestamp_column is None:
            return [], []

        if timestamp_column not in contract.data.columns:
            finding = SplitFinding(
                code="P2-TEMPORAL-LEAKAGE-001",
                category=(
                    SplitFindingCategory.TEMPORAL_LEAKAGE
                ),
                severity=SplitFindingSeverity.CRITICAL,
                component=self.component_name,
                message=(
                    "Temporal leakage checking requires a valid "
                    "timestamp column."
                ),
                columns=[timestamp_column],
                recommended_action=SplitFindingAction.RESPLIT,
                applied_action=SplitFindingAction.REPORT,
                blocks_readiness=True,
            )

            return [finding], []

        timestamps = pd.to_datetime(
            contract.data[timestamp_column],
            errors="coerce",
        )

        invalid_timestamp_mask = (
            contract.data[timestamp_column].notna()
            & timestamps.isna()
        )

        invalid_count = int(
            invalid_timestamp_mask.sum()
        )

        findings: list[SplitFinding] = []
        violations: list[dict[str, Any]] = []

        if invalid_count > 0:
            findings.append(
                SplitFinding(
                    code="P2-TEMPORAL-LEAKAGE-002",
                    category=(
                        SplitFindingCategory.TEMPORAL_LEAKAGE
                    ),
                    severity=SplitFindingSeverity.ERROR,
                    component=self.component_name,
                    message=(
                        "The temporal leakage check found "
                        "unparseable timestamp values."
                    ),
                    columns=[timestamp_column],
                    affected_row_count=invalid_count,
                    example_row_ids=self._example_row_ids(
                        contract=contract,
                        mask=invalid_timestamp_mask,
                    ),
                    recommended_action=SplitFindingAction.RESPLIT,
                    applied_action=SplitFindingAction.REPORT,
                    blocks_readiness=True,
                )
            )

            return findings, violations

        missing_timestamp_mask = timestamps.isna()

        missing_count = int(
            missing_timestamp_mask.sum()
        )

        if missing_count > 0:
            findings.append(
                SplitFinding(
                    code="P2-TEMPORAL-LEAKAGE-003",
                    category=(
                        SplitFindingCategory.TEMPORAL_LEAKAGE
                    ),
                    severity=SplitFindingSeverity.ERROR,
                    component=self.component_name,
                    message=(
                        "The temporal leakage check found "
                        "missing timestamp values."
                    ),
                    columns=[timestamp_column],
                    affected_row_count=missing_count,
                    example_row_ids=self._example_row_ids(
                        contract=contract,
                        mask=missing_timestamp_mask,
                    ),
                    recommended_action=SplitFindingAction.RESPLIT,
                    applied_action=SplitFindingAction.REPORT,
                    blocks_readiness=True,
                )
            )

            return findings, violations

        partition_indices = {
            PartitionName.TRAIN: indices.train_indices,
            PartitionName.VALIDATION: (
                indices.validation_indices
            ),
            PartitionName.TEST: indices.test_indices,
        }

        if self.options.require_train_before_validation:
            self._check_pair(
                earlier_partition=PartitionName.TRAIN,
                later_partition=PartitionName.VALIDATION,
                partition_indices=partition_indices,
                timestamps=timestamps,
                timestamp_column=timestamp_column,
                contract=contract,
                findings=findings,
                violations=violations,
            )

        if self.options.require_validation_before_test:
            self._check_pair(
                earlier_partition=PartitionName.VALIDATION,
                later_partition=PartitionName.TEST,
                partition_indices=partition_indices,
                timestamps=timestamps,
                timestamp_column=timestamp_column,
                contract=contract,
                findings=findings,
                violations=violations,
            )

        if self.options.require_train_before_test:
            self._check_pair(
                earlier_partition=PartitionName.TRAIN,
                later_partition=PartitionName.TEST,
                partition_indices=partition_indices,
                timestamps=timestamps,
                timestamp_column=timestamp_column,
                contract=contract,
                findings=findings,
                violations=violations,
            )

        return findings, violations

    def _check_pair(
        self,
        *,
        earlier_partition: PartitionName,
        later_partition: PartitionName,
        partition_indices: dict[
            PartitionName,
            pd.Index,
        ],
        timestamps: pd.Series,
        timestamp_column: str,
        contract: SplitContract,
        findings: list[SplitFinding],
        violations: list[dict[str, Any]],
    ) -> None:
        """Check temporal ordering for one partition pair."""

        earlier_indices = partition_indices[
            earlier_partition
        ]

        later_indices = partition_indices[
            later_partition
        ]

        if (
            earlier_indices.empty
            or later_indices.empty
        ):
            return

        earlier_timestamps = timestamps.loc[
            earlier_indices
        ]

        later_timestamps = timestamps.loc[
            later_indices
        ]

        earlier_maximum = earlier_timestamps.max()
        later_minimum = later_timestamps.min()

        if self.options.allow_equal_boundary_timestamps:
            is_valid = (
                earlier_maximum <= later_minimum
            )
        else:
            is_valid = (
                earlier_maximum < later_minimum
            )

        if is_valid:
            return

        if self.options.allow_equal_boundary_timestamps:
            violating_earlier_mask = (
                earlier_timestamps > later_minimum
            )

            violating_later_mask = (
                later_timestamps < earlier_maximum
            )
        else:
            violating_earlier_mask = (
                earlier_timestamps >= later_minimum
            )

            violating_later_mask = (
                later_timestamps <= earlier_maximum
            )

        earlier_row_ids = self._row_ids_from_indices(
            contract=contract,
            indices=earlier_timestamps.index[
                violating_earlier_mask
            ],
        )

        later_row_ids = self._row_ids_from_indices(
            contract=contract,
            indices=later_timestamps.index[
                violating_later_mask
            ],
        )

        violation = {
            "earlier_partition": (
                earlier_partition.value
            ),
            "later_partition": (
                later_partition.value
            ),
            "earlier_maximum_timestamp": (
                earlier_maximum.isoformat()
            ),
            "later_minimum_timestamp": (
                later_minimum.isoformat()
            ),
            "earlier_example_row_ids": (
                earlier_row_ids[
                    : self.options.maximum_examples
                ]
            ),
            "later_example_row_ids": (
                later_row_ids[
                    : self.options.maximum_examples
                ]
            ),
        }

        violations.append(violation)

        findings.append(
            SplitFinding(
                code="P2-TEMPORAL-LEAKAGE-004",
                category=(
                    SplitFindingCategory.TEMPORAL_LEAKAGE
                ),
                severity=SplitFindingSeverity.CRITICAL,
                component=self.component_name,
                message=(
                    f"Partition {earlier_partition.value!r} "
                    f"is not strictly earlier than partition "
                    f"{later_partition.value!r}."
                ),
                partitions=[
                    earlier_partition,
                    later_partition,
                ],
                columns=[timestamp_column],
                affected_row_count=(
                    len(earlier_row_ids)
                    + len(later_row_ids)
                ),
                example_row_ids=[
                    *earlier_row_ids,
                    *later_row_ids,
                ][: self.options.maximum_examples],
                recommended_action=SplitFindingAction.RESPLIT,
                applied_action=SplitFindingAction.REPORT,
                blocks_readiness=(
                    self.options
                    .block_on_temporal_violation
                ),
                details=violation,
            )
        )

    def _example_row_ids(
        self,
        *,
        contract: SplitContract,
        mask: pd.Series,
    ) -> list[str | int]:
        """Return example internal row IDs for a mask."""

        matching_indices = contract.data.index[
            mask
        ]

        return self._row_ids_from_indices(
            contract=contract,
            indices=matching_indices,
        )[: self.options.maximum_examples]

    @staticmethod
    def _row_ids_from_indices(
        *,
        contract: SplitContract,
        indices: pd.Index,
    ) -> list[str | int]:
        """Resolve internal row IDs from DataFrame indices."""

        row_ids = contract.data.loc[
            indices,
            contract.internal_row_id_column,
        ].tolist()

        return [
            value
            if isinstance(value, (str, int))
            else str(value)
            for value in row_ids
        ]
