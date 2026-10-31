"""Purged and embargoed time splitting for Phase 2."""

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
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.result import (
    SplitFinding,
)
from mlprep.phases.phase2.strategies.base import (
    StrategyPartitions,
)
from mlprep.phases.phase2.strategies.time_split import (
    TimeSplitOptions,
    TimeSplitStrategy,
)


@dataclass(slots=True)
class PurgedTimeSplitOptions:
    """Options controlling purged temporal splitting."""

    ratios: SplitRatios

    purge_gap: str | pd.Timedelta = "0 days"
    embargo_gap: str | pd.Timedelta = "0 days"

    keep_equal_timestamps_together: bool = True

    minimum_train_rows: int = 1
    minimum_validation_rows: int = 1
    minimum_test_rows: int = 1

    def __post_init__(self) -> None:
        """Validate purged-time options."""

        self.ratios.validate_total()

        purge_delta = pd.Timedelta(
            self.purge_gap
        )

        embargo_delta = pd.Timedelta(
            self.embargo_gap
        )

        if purge_delta < pd.Timedelta(0):
            raise ValueError(
                "purge_gap cannot be negative."
            )

        if embargo_delta < pd.Timedelta(0):
            raise ValueError(
                "embargo_gap cannot be negative."
            )


class PurgedTimeSplitStrategy(
    TimeSplitStrategy
):
    """Apply chronological splitting with purge and embargo gaps."""

    strategy_name = SplitStrategyName.PURGED_TIME

    def __init__(
        self,
        options: PurgedTimeSplitOptions,
    ) -> None:
        """Initialize the purged-time strategy."""

        self.purged_options = options

        super().__init__(
            options=TimeSplitOptions(
                ratios=options.ratios,
                sort_ascending=True,
                keep_equal_timestamps_together=(
                    options
                    .keep_equal_timestamps_together
                ),
                minimum_train_rows=(
                    options.minimum_train_rows
                ),
                minimum_validation_rows=(
                    options.minimum_validation_rows
                ),
                minimum_test_rows=(
                    options.minimum_test_rows
                ),
                require_strict_temporal_order=True,
            )
        )

    def create_indices(
        self,
        contract: SplitContract,
    ) -> StrategyPartitions:
        """Create temporal indices and apply purge and embargo."""

        base_indices = super().create_indices(
            contract
        )

        timestamp_column = (
            contract.columns.timestamp_column
        )

        if timestamp_column is None:
            raise RuntimeError(
                "Timestamp resolution failed."
            )

        timestamps = pd.to_datetime(
            contract.data[timestamp_column],
            errors="raise",
        )

        purge_gap = pd.Timedelta(
            self.purged_options.purge_gap
        )

        embargo_gap = pd.Timedelta(
            self.purged_options.embargo_gap
        )

        train_indices = base_indices.train_indices
        validation_indices = (
            base_indices.validation_indices
        )

        test_indices = base_indices.test_indices

        if not validation_indices.empty:
            validation_start = timestamps.loc[
                validation_indices
            ].min()

            train_cutoff = (
                validation_start - purge_gap
            )

            train_indices = train_indices[
                timestamps.loc[
                    train_indices
                ].to_numpy()
                < train_cutoff
            ]

            validation_end = timestamps.loc[
                validation_indices
            ].max()

            test_start_cutoff = (
                validation_end + embargo_gap
            )

            test_indices = test_indices[
                timestamps.loc[
                    test_indices
                ].to_numpy()
                > test_start_cutoff
            ]

        else:
            test_start = timestamps.loc[
                test_indices
            ].min()

            train_cutoff = (
                test_start - purge_gap
            )

            train_indices = train_indices[
                timestamps.loc[
                    train_indices
                ].to_numpy()
                < train_cutoff
            ]

        if not test_indices.empty:
            test_start = timestamps.loc[
                test_indices
            ].min()

            validation_cutoff = (
                test_start - purge_gap
            )

            validation_indices = (
                validation_indices[
                    timestamps.loc[
                        validation_indices
                    ].to_numpy()
                    < validation_cutoff
                ]
            )

        return StrategyPartitions(
            train_indices=pd.Index(
                train_indices
            ),
            validation_indices=pd.Index(
                validation_indices
            ),
            test_indices=pd.Index(
                test_indices
            ),
        )

    def create_findings(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> list[object]:
        """Create findings for purged temporal partitions."""

        findings = super().create_findings(
            contract=contract,
            indices=indices,
        )

        minimum_sizes = {
            PartitionName.TRAIN: (
                self.purged_options
                .minimum_train_rows
            ),
            PartitionName.VALIDATION: (
                self.purged_options
                .minimum_validation_rows
            ),
            PartitionName.TEST: (
                self.purged_options
                .minimum_test_rows
            ),
        }

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

        for partition_name, minimum_size in (
            minimum_sizes.items()
        ):
            actual_size = len(
                partition_indices[
                    partition_name
                ]
            )

            if actual_size >= minimum_size:
                continue

            findings.append(
                SplitFinding(
                    code="P2-PURGED-001",
                    category=(
                        SplitFindingCategory
                        .SPLIT_SIZE
                    ),
                    severity=(
                        SplitFindingSeverity.ERROR
                    ),
                    component="purged_time_split",
                    message=(
                        f"Partition "
                        f"{partition_name.value!r} contains "
                        "too few rows after purge or embargo."
                    ),
                    partitions=[partition_name],
                    recommended_action=(
                        SplitFindingAction.RESPLIT
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                    details={
                        "actual_size": actual_size,
                        "minimum_size": minimum_size,
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
        """Create purged-time metadata."""

        metadata = super().create_metadata(
            contract=contract,
            indices=indices,
        )

        metadata.update(
            {
                "purge_gap": str(
                    pd.Timedelta(
                        self.purged_options
                        .purge_gap
                    )
                ),
                "embargo_gap": str(
                    pd.Timedelta(
                        self.purged_options
                        .embargo_gap
                    )
                ),
                "purged": True,
            }
        )

        return metadata