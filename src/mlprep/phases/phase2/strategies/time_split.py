"""Time-ordered dataset splitting for Phase 2."""

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
    BaseSplitStrategy,
    StrategyPartitions,
)


@dataclass(slots=True)
class TimeSplitOptions:
    """Options controlling time-ordered splitting."""

    ratios: SplitRatios

    sort_ascending: bool = True

    keep_equal_timestamps_together: bool = True

    minimum_train_rows: int = 1
    minimum_validation_rows: int = 1
    minimum_test_rows: int = 1

    require_strict_temporal_order: bool = True

    def __post_init__(self) -> None:
        """Validate time-split options."""

        self.ratios.validate_total()

        minimum_values = {
            "minimum_train_rows": self.minimum_train_rows,
            "minimum_validation_rows": (
                self.minimum_validation_rows
            ),
            "minimum_test_rows": self.minimum_test_rows,
        }

        for name, value in minimum_values.items():
            if value < 0:
                raise ValueError(
                    f"{name} cannot be negative."
                )


class TimeSplitStrategy(BaseSplitStrategy):
    """Split data chronologically without random shuffling."""

    strategy_name = SplitStrategyName.TIME

    def __init__(
        self,
        options: TimeSplitOptions,
    ) -> None:
        """Initialize the time split strategy."""

        super().__init__(
            ratios=options.ratios,
            random_seed=None,
        )

        self.options = options

    def validate_contract(
        self,
        contract: SplitContract,
    ) -> None:
        """Validate requirements for chronological splitting."""

        super().validate_contract(contract)

        timestamp_column = (
            contract.columns.timestamp_column
        )

        if timestamp_column is None:
            raise ValueError(
                "Time splitting requires a timestamp column."
            )

        if timestamp_column not in contract.data.columns:
            raise ValueError(
                f"Timestamp column {timestamp_column!r} "
                "does not exist."
            )

        parsed_timestamps = pd.to_datetime(
            contract.data[timestamp_column],
            errors="coerce",
        )

        invalid_mask = (
            contract.data[timestamp_column].notna()
            & parsed_timestamps.isna()
        )

        if bool(invalid_mask.any()):
            raise ValueError(
                "Time splitting cannot run because the "
                "timestamp column contains unparseable values."
            )

        if parsed_timestamps.isna().any():
            raise ValueError(
                "Time splitting cannot run because the "
                "timestamp column contains missing values."
            )

    def create_indices(
        self,
        contract: SplitContract,
    ) -> StrategyPartitions:
        """Create chronological partition indices."""

        timestamp_column = (
            contract.columns.timestamp_column
        )

        if timestamp_column is None:
            raise RuntimeError(
                "Timestamp resolution failed."
            )

        parsed_timestamps = pd.to_datetime(
            contract.data[timestamp_column],
            errors="raise",
        )

        ordered_frame = pd.DataFrame(
            {
                "__index__": contract.data.index,
                "__timestamp__": parsed_timestamps,
            },
            index=contract.data.index,
        )

        ordered_frame = ordered_frame.sort_values(
            by=[
                "__timestamp__",
                "__index__",
            ],
            ascending=[
                self.options.sort_ascending,
                True,
            ],
            kind="mergesort",
        )

        ordered_indices = pd.Index(
            ordered_frame["__index__"].tolist()
        )

        (
            train_count,
            validation_count,
            _,
        ) = self.calculate_partition_counts(
            row_count=len(ordered_indices),
            ratios=self.ratios,
        )

        train_boundary = train_count

        validation_boundary = (
            train_count
            + validation_count
        )

        if (
            self.options.keep_equal_timestamps_together
        ):
            train_boundary = (
                self._move_boundary_forward(
                    ordered_frame=ordered_frame,
                    boundary=train_boundary,
                )
            )

            validation_boundary = (
                self._move_boundary_forward(
                    ordered_frame=ordered_frame,
                    boundary=validation_boundary,
                )
            )

        if validation_boundary >= len(
            ordered_indices
        ):
            raise ValueError(
                "Timestamp grouping leaves no rows for "
                "the test partition."
            )

        if (
            validation_count > 0
            and train_boundary
            >= validation_boundary
        ):
            raise ValueError(
                "Timestamp grouping leaves no rows for "
                "the validation partition."
            )

        train_indices = ordered_indices[
            :train_boundary
        ]

        validation_indices = ordered_indices[
            train_boundary:validation_boundary
        ]

        test_indices = ordered_indices[
            validation_boundary:
        ]

        return StrategyPartitions(
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
        )

    def create_findings(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> list[SplitFinding]:
        """Validate temporal ordering and partition sizes."""

        findings: list[SplitFinding] = []

        timestamp_column = (
            contract.columns.timestamp_column
        )

        if timestamp_column is None:
            return findings

        timestamps = pd.to_datetime(
            contract.data[timestamp_column],
            errors="coerce",
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

        minimum_rows = {
            PartitionName.TRAIN: (
                self.options.minimum_train_rows
            ),
            PartitionName.VALIDATION: (
                self.options.minimum_validation_rows
            ),
            PartitionName.TEST: (
                self.options.minimum_test_rows
            ),
        }

        for partition_name, partition_index in (
            partition_indices.items()
        ):
            minimum_count = minimum_rows[
                partition_name
            ]

            if len(partition_index) >= minimum_count:
                continue

            findings.append(
                SplitFinding(
                    code="P2-TIME-001",
                    category=(
                        SplitFindingCategory.SPLIT_SIZE
                    ),
                    severity=(
                        SplitFindingSeverity.ERROR
                    ),
                    component="time_split",
                    message=(
                        f"Partition "
                        f"{partition_name.value!r} contains "
                        "fewer rows than required."
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
                        "actual_row_count": len(
                            partition_index
                        ),
                        "minimum_row_count": (
                            minimum_count
                        ),
                    },
                )
            )

        train_timestamps = timestamps.loc[
            indices.train_indices
        ]

        validation_timestamps = timestamps.loc[
            indices.validation_indices
        ]

        test_timestamps = timestamps.loc[
            indices.test_indices
        ]

        if not validation_timestamps.empty:
            invalid_train_validation = (
                train_timestamps.max()
                > validation_timestamps.min()
            )

            if (
                self.options
                .require_strict_temporal_order
                and invalid_train_validation
            ):
                findings.append(
                    SplitFinding(
                        code="P2-TIME-002",
                        category=(
                            SplitFindingCategory
                            .TEMPORAL_BOUNDARY
                        ),
                        severity=(
                            SplitFindingSeverity.CRITICAL
                        ),
                        component="time_split",
                        message=(
                            "Training timestamps extend beyond "
                            "the validation boundary."
                        ),
                        partitions=[
                            PartitionName.TRAIN,
                            PartitionName.VALIDATION,
                        ],
                        columns=[timestamp_column],
                        recommended_action=(
                            SplitFindingAction.RESPLIT
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=True,
                    )
                )

        if not test_timestamps.empty:
            preceding_timestamps = (
                validation_timestamps
                if not validation_timestamps.empty
                else train_timestamps
            )

            invalid_test_order = (
                preceding_timestamps.max()
                > test_timestamps.min()
            )

            if (
                self.options
                .require_strict_temporal_order
                and invalid_test_order
            ):
                findings.append(
                    SplitFinding(
                        code="P2-TIME-003",
                        category=(
                            SplitFindingCategory
                            .TEMPORAL_BOUNDARY
                        ),
                        severity=(
                            SplitFindingSeverity.CRITICAL
                        ),
                        component="time_split",
                        message=(
    "Earlier partitions contain timestamps "
                            "after the test boundary."
                        ),
                        partitions=[
                            PartitionName.TEST
                        ],
                        columns=[timestamp_column],
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
        """Create metadata for the time split."""

        metadata = super().create_metadata(
            contract=contract,
            indices=indices,
        )

        timestamp_column = (
            contract.columns.timestamp_column
        )

        if timestamp_column is None:
            return metadata

        timestamps = pd.to_datetime(
            contract.data[timestamp_column],
            errors="coerce",
        )

        def range_for(
            partition_indices: pd.Index,
        ) -> dict[str, str | None]:
            partition_values = timestamps.loc[
                partition_indices
            ]

            if partition_values.empty:
                return {
                    "minimum": None,
                    "maximum": None,
                }

            return {
                "minimum": (
                    partition_values.min().isoformat()
                ),
                "maximum": (
                    partition_values.max().isoformat()
                ),
            }

        metadata.update(
            {
                "shuffle": False,
                "stratified": False,
                "grouped": False,
                "temporal": True,
                "timestamp_column": (
                    timestamp_column
                ),
                "keep_equal_timestamps_together": (
                    self.options
                    .keep_equal_timestamps_together
                ),
                "timestamp_ranges": {
                    "train": range_for(
                        indices.train_indices
                    ),
                    "validation": range_for(
                        indices.validation_indices
                    ),
                    "test": range_for(
                        indices.test_indices
                    ),
                },
            }
        )

        return metadata

    @staticmethod
    def _move_boundary_forward(
        *,
        ordered_frame: pd.DataFrame,
        boundary: int,
    ) -> int:
        """Move a boundary so equal timestamps remain together."""

        if boundary <= 0:
            return boundary

        if boundary >= len(ordered_frame):
            return boundary

        timestamps = ordered_frame[
            "__timestamp__"
        ].reset_index(drop=True)

        previous_timestamp = timestamps.iloc[
            boundary - 1
        ]

        adjusted_boundary = boundary

        while (
            adjusted_boundary < len(timestamps)
            and timestamps.iloc[
                adjusted_boundary
            ]
            == previous_timestamp
        ):
            adjusted_boundary += 1

        return adjusted_boundary