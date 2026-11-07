"""Temporal-boundary validation for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
)
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    SplitBoundaries,
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
class TemporalBoundaryValidationOptions:
    """Options controlling temporal-boundary validation."""

    enabled: bool = True

    allow_equal_boundary_timestamps: bool = False
    require_nonmissing_timestamps: bool = True

    minimum_train_validation_gap: str | None = None
    minimum_validation_test_gap: str | None = None

    block_on_overlap: bool = True


class TemporalBoundaryValidator:
    """Validate chronological partition boundaries."""

    component_name = "temporal_boundaries"

    def __init__(
        self,
        options: (
            TemporalBoundaryValidationOptions | None
        ) = None,
    ) -> None:
        """Initialize the temporal-boundary validator."""

        self.options = (
            options
            or TemporalBoundaryValidationOptions()
        )

    def validate(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> tuple[
        list[SplitFinding],
        SplitBoundaries,
    ]:
        """Validate and return temporal boundaries."""

        boundaries = SplitBoundaries()

        if not self.options.enabled:
            return [], boundaries

        timestamp_column = (
            contract.columns.timestamp_column
        )

        if timestamp_column is None:
            return [], boundaries

        timestamps = pd.to_datetime(
            contract.data[timestamp_column],
            errors="coerce",
        )

        findings: list[SplitFinding] = []

        if (
            self.options.require_nonmissing_timestamps
            and timestamps.isna().any()
        ):
            findings.append(
                SplitFinding(
                    code="P2-TIME-BOUNDARY-001",
                    category=(
                        SplitFindingCategory
                        .TEMPORAL_BOUNDARY
                    ),
                    severity=SplitFindingSeverity.ERROR,
                    component=self.component_name,
                    message=(
                        "Temporal boundary validation found "
                        "missing or invalid timestamps."
                    ),
                    columns=[timestamp_column],
                    affected_row_count=int(
                        timestamps.isna().sum()
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

            return findings, boundaries

        ranges = {
            PartitionName.TRAIN: self._range(
                timestamps.loc[
                    indices.train_indices
                ]
            ),
            PartitionName.VALIDATION: self._range(
                timestamps.loc[
                    indices.validation_indices
                ]
            ),
            PartitionName.TEST: self._range(
                timestamps.loc[
                    indices.test_indices
                ]
            ),
        }

        boundaries.train_start = self._serialize(
            ranges[PartitionName.TRAIN][0]
        )

        boundaries.train_end = self._serialize(
            ranges[PartitionName.TRAIN][1]
        )

        boundaries.validation_start = (
            self._serialize(
                ranges[
                    PartitionName.VALIDATION
                ][0]
            )
        )

        boundaries.validation_end = (
            self._serialize(
                ranges[
                    PartitionName.VALIDATION
                ][1]
            )
        )

        boundaries.test_start = self._serialize(
            ranges[PartitionName.TEST][0]
        )

        boundaries.test_end = self._serialize(
            ranges[PartitionName.TEST][1]
        )

        self._validate_pair(
            earlier=PartitionName.TRAIN,
            later=PartitionName.VALIDATION,
            ranges=ranges,
            minimum_gap=(
                self.options
                .minimum_train_validation_gap
            ),
            timestamp_column=timestamp_column,
            findings=findings,
        )

        self._validate_pair(
            earlier=PartitionName.VALIDATION,
            later=PartitionName.TEST,
            ranges=ranges,
            minimum_gap=(
                self.options
                .minimum_validation_test_gap
            ),
            timestamp_column=timestamp_column,
            findings=findings,
        )

        return findings, boundaries

    def _validate_pair(
        self,
        *,
        earlier: PartitionName,
        later: PartitionName,
        ranges: dict[
            PartitionName,
            tuple[pd.Timestamp | None, pd.Timestamp | None],
        ],
        minimum_gap: str | None,
        timestamp_column: str,
        findings: list[SplitFinding],
    ) -> None:
        """Validate one chronological partition pair."""

        earlier_end = ranges[earlier][1]
        later_start = ranges[later][0]

        if (
            earlier_end is None
            or later_start is None
        ):
            return

        if self.options.allow_equal_boundary_timestamps:
            ordered = earlier_end <= later_start
        else:
            ordered = earlier_end < later_start

        if not ordered:
            findings.append(
                SplitFinding(
                    code="P2-TIME-BOUNDARY-002",
                    category=(
                        SplitFindingCategory
                        .TEMPORAL_BOUNDARY
                    ),
                    severity=(
                        SplitFindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        f"Partition {earlier.value!r} is not "
                        f"earlier than partition {later.value!r}."
                    ),
                    partitions=[
                        earlier,
                        later,
                    ],
                    columns=[timestamp_column],
                    recommended_action=(
                        SplitFindingAction.RESPLIT
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=(
                        self.options.block_on_overlap
                    ),
                    details={
                        "earlier_end": (
                            earlier_end.isoformat()
                        ),
                        "later_start": (
                            later_start.isoformat()
                        ),
                    },
                )
            )

        if minimum_gap is None:
            return

        required_gap = pd.Timedelta(
            minimum_gap
        )

        actual_gap = later_start - earlier_end

        if actual_gap >= required_gap:
            return

        findings.append(
            SplitFinding(
                code="P2-TIME-BOUNDARY-003",
                category=(
                    SplitFindingCategory
                    .TEMPORAL_BOUNDARY
                ),
                severity=SplitFindingSeverity.ERROR,
                component=self.component_name,
                message=(
                    f"The gap between {earlier.value!r} and "
                    f"{later.value!r} is below the required "
                    "minimum."
                ),
                partitions=[
                    earlier,
                    later,
                ],
                columns=[timestamp_column],
                recommended_action=(
                    SplitFindingAction.RESPLIT
                ),
                applied_action=(
                    SplitFindingAction.REPORT
                ),
                blocks_readiness=True,
                details={
                    "required_gap": str(
                        required_gap
                    ),
                    "actual_gap": str(
                        actual_gap
                    ),
                },
            )
        )

    @staticmethod
    def _range(
        timestamps: pd.Series,
    ) -> tuple[
        pd.Timestamp | None,
        pd.Timestamp | None,
    ]:
        """Return minimum and maximum timestamps."""

        if timestamps.empty:
            return None, None

        return (
            timestamps.min(),
            timestamps.max(),
        )

    @staticmethod
    def _serialize(
        value: pd.Timestamp | None,
    ) -> str | None:
        """Serialize a timestamp."""

        if value is None:
            return None

        return value.isoformat()
