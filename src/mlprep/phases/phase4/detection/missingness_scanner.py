"""Missingness scanning for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase4.contract.imputation_contract import (
    ImputationContract,
)
from mlprep.phases.phase4.contract.imputation_plan import (
    ImputationPlan,
)


class ColumnMissingnessScan(BaseModel):
    """Missingness scan for one feature in one partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    partition: PartitionName

    row_count: int = Field(
        default=0,
        ge=0,
    )

    missing_count: int = Field(
        default=0,
        ge=0,
    )

    nonmissing_count: int = Field(
        default=0,
        ge=0,
    )

    missing_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    completely_missing: bool = False

    expected_to_be_imputed: bool = False
    missing_without_plan: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class PartitionMissingnessScan(BaseModel):
    """Missingness scan for one complete partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    partition: PartitionName

    row_count: int = Field(
        default=0,
        ge=0,
    )

    feature_count: int = Field(
        default=0,
        ge=0,
    )

    total_cell_count: int = Field(
        default=0,
        ge=0,
    )

    total_missing_count: int = Field(
        default=0,
        ge=0,
    )

    total_missing_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    rows_with_missing_values: int = Field(
        default=0,
        ge=0,
    )

    rows_with_missing_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    maximum_missing_values_per_row: int = Field(
        default=0,
        ge=0,
    )

    completely_missing_columns: list[str] = Field(
        default_factory=list
    )

    columns_missing_without_plan: list[str] = Field(
        default_factory=list
    )

    columns: dict[
        str,
        ColumnMissingnessScan,
    ] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class MissingnessScanResult(BaseModel):
    """Complete missingness scan across all partitions."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    partitions: dict[
        PartitionName,
        PartitionMissingnessScan,
    ] = Field(default_factory=dict)

    features_with_missing_values: list[str] = Field(
        default_factory=list
    )

    features_missing_without_plan: list[str] = Field(
        default_factory=list
    )

    completely_missing_features: list[str] = Field(
        default_factory=list
    )

    unexpected_increase_features: list[str] = Field(
        default_factory=list
    )

    phase3_missing_percentages: dict[
        str,
        float,
    ] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def partition(
        self,
        name: PartitionName,
    ) -> PartitionMissingnessScan:
        """Return the scan for one partition."""

        if name not in self.partitions:
            raise KeyError(
                f"No missingness scan exists for "
                f"{name.value!r}."
            )

        return self.partitions[name]


@dataclass(slots=True)
class MissingnessScannerOptions:
    """Options controlling Phase 4 missingness scanning."""

    enabled: bool = True

    compare_with_phase3: bool = True

    unexpected_increase_tolerance: float = 0.01

    include_zero_missing_columns: bool = True

    maximum_reported_columns: int = 1_000

    def __post_init__(self) -> None:
        """Validate missingness scanner options."""

        if not (
            0.0
            <= self.unexpected_increase_tolerance
            <= 1.0
        ):
            raise ValueError(
                "unexpected_increase_tolerance must be "
                "between zero and one."
            )

        if self.maximum_reported_columns < 1:
            raise ValueError(
                "maximum_reported_columns must be positive."
            )


class MissingnessScanner:
    """Scan missingness immediately before imputation."""

    component_name = "missingness_scanner"

    def __init__(
        self,
        options: MissingnessScannerOptions | None = None,
    ) -> None:
        """Initialize the missingness scanner."""

        self.options = (
            options or MissingnessScannerOptions()
        )

    def scan(
        self,
        *,
        contract: ImputationContract,
        plan: ImputationPlan,
    ) -> MissingnessScanResult:
        """Scan missingness across all Phase 4 partitions."""

        if not self.options.enabled:
            return MissingnessScanResult(
                metadata={
                    "enabled": False,
                }
            )

        partitions = {
            PartitionName.TRAIN: contract.train,
            PartitionName.VALIDATION: (
                contract.validation
            ),
            PartitionName.TEST: contract.test,
        }

        partition_scans = {
            name: self._scan_partition(
                name=name,
                partition=partition,
                plan=plan,
            )
            for name, partition
            in partitions.items()
        }

        features_with_missing_values = sorted(
            {
                column
                for scan
                in partition_scans.values()
                for column, column_scan
                in scan.columns.items()
                if column_scan.missing_count > 0
            }
        )

        features_missing_without_plan = sorted(
            {
                column
                for scan
                in partition_scans.values()
                for column
                in scan.columns_missing_without_plan
            }
        )

        completely_missing_features = sorted(
            {
                column
                for scan
                in partition_scans.values()
                for column
                in scan.completely_missing_columns
            }
        )

        phase3_percentages = (
            self._phase3_missing_percentages(
                contract
            )
        )

        unexpected_increase_features = (
            self._unexpected_increases(
                training_scan=partition_scans[
                    PartitionName.TRAIN
                ],
                phase3_percentages=(
                    phase3_percentages
                ),
            )
        )

        return MissingnessScanResult(
            partitions=partition_scans,
            features_with_missing_values=(
                features_with_missing_values[
                    : self.options
                    .maximum_reported_columns
                ]
            ),
            features_missing_without_plan=(
                features_missing_without_plan[
                    : self.options
                    .maximum_reported_columns
                ]
            ),
            completely_missing_features=(
                completely_missing_features[
                    : self.options
                    .maximum_reported_columns
                ]
            ),
            unexpected_increase_features=(
                unexpected_increase_features[
                    : self.options
                    .maximum_reported_columns
                ]
            ),
            phase3_missing_percentages=(
                phase3_percentages
            ),
            metadata={
                "enabled": True,
                "partition_count": len(
                    partition_scans
                ),
            },
        )

    def _scan_partition(
        self,
        *,
        name: PartitionName,
        partition: DatasetPartition,
        plan: ImputationPlan,
    ) -> PartitionMissingnessScan:
        """Scan one partition."""

        frame = partition.X

        row_count = len(frame)
        feature_count = len(
            frame.columns
        )

        total_cell_count = (
            row_count * feature_count
        )

        missing_mask = frame.isna()

        total_missing_count = int(
            missing_mask.sum().sum()
        )

        total_missing_percentage = (
            total_missing_count
            / total_cell_count
            * 100.0
            if total_cell_count > 0
            else 0.0
        )

        row_missing_counts = missing_mask.sum(
            axis=1
        )

        rows_with_missing = int(
            (row_missing_counts > 0).sum()
        )

        rows_with_missing_percentage = (
            rows_with_missing
            / row_count
            * 100.0
            if row_count > 0
            else 0.0
        )

        column_scans: dict[
            str,
            ColumnMissingnessScan,
        ] = {}

        completely_missing_columns: list[str] = []
        columns_missing_without_plan: list[str] = []

        for column in frame.columns:
            missing_count = int(
                missing_mask[column].sum()
            )

            if (
                missing_count == 0
                and not self.options
                .include_zero_missing_columns
            ):
                continue

            nonmissing_count = (
                row_count - missing_count
            )

            completely_missing = bool(
                row_count > 0
                and missing_count == row_count
            )

            expected_to_be_imputed = (
                column in plan.entries
            )

            missing_without_plan = bool(
                missing_count > 0
                and not expected_to_be_imputed
            )

            column_scans[str(column)] = (
                ColumnMissingnessScan(
                    column=str(column),
                    partition=name,
                    row_count=row_count,
                    missing_count=missing_count,
                    nonmissing_count=(
                        nonmissing_count
                    ),
                    missing_percentage=(
                        missing_count
                        / row_count
                        * 100.0
                        if row_count > 0
                        else 0.0
                    ),
                    completely_missing=(
                        completely_missing
                    ),
                    expected_to_be_imputed=(
                        expected_to_be_imputed
                    ),
                    missing_without_plan=(
                        missing_without_plan
                    ),
                )
            )

            if completely_missing:
                completely_missing_columns.append(
                    str(column)
                )

            if missing_without_plan:
                columns_missing_without_plan.append(
                    str(column)
                )

        return PartitionMissingnessScan(
            partition=name,
            row_count=row_count,
            feature_count=feature_count,
            total_cell_count=total_cell_count,
            total_missing_count=(
                total_missing_count
            ),
            total_missing_percentage=(
                total_missing_percentage
            ),
            rows_with_missing_values=(
                rows_with_missing
            ),
            rows_with_missing_percentage=(
                rows_with_missing_percentage
            ),
            maximum_missing_values_per_row=(
                int(row_missing_counts.max())
                if not row_missing_counts.empty
                else 0
            ),
            completely_missing_columns=(
                completely_missing_columns
            ),
            columns_missing_without_plan=(
                columns_missing_without_plan
            ),
            columns=column_scans,
        )

    def _phase3_missing_percentages(
        self,
        contract: ImputationContract,
    ) -> dict[str, float]:
        """Return Phase 3 training missingness percentages."""

        if not self.options.compare_with_phase3:
            return {}

        metadata = (
            contract.phase3_state.metadata
        )

        stored_percentages = metadata.get(
            "training_missing_percentages"
        )

        if not isinstance(
            stored_percentages,
            dict,
        ):
            return {}

        percentages: dict[str, float] = {}

        for column, value in (
            stored_percentages.items()
        ):
            try:
                percentages[
                    str(column)
                ] = float(value)

            except (
                TypeError,
                ValueError,
            ):
                continue

        return percentages

    def _unexpected_increases(
        self,
        *,
        training_scan: PartitionMissingnessScan,
        phase3_percentages: dict[str, float],
    ) -> list[object]:
        """Detect missingness increases since Phase 3."""

        if not phase3_percentages:
            return []

        unexpected_columns: list[str] = []

        for column, scan in (
            training_scan.columns.items()
        ):
            old_percentage = (
                phase3_percentages.get(
                    column
                )
            )

            if old_percentage is None:
                continue

            old_ratio = (
                old_percentage / 100.0
                if old_percentage > 1.0
                else old_percentage
            )

            current_ratio = (
                scan.missing_percentage
                / 100.0
            )

            if (
                current_ratio - old_ratio
                > self.options
                .unexpected_increase_tolerance
            ):
                unexpected_columns.append(
                    column
                )

        return unexpected_columns
