"""Partition size validation for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
    SplitRatios,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    SplitFinding,
)


@dataclass(slots=True)
class SplitSizeValidationOptions:
    """Options controlling partition-size validation."""

    enabled: bool = True

    requested_ratios: SplitRatios | None = None

    minimum_train_rows: int = 1
    minimum_validation_rows: int = 1
    minimum_test_rows: int = 1

    maximum_ratio_deviation: float = 0.05

    require_nonempty_validation: bool = True
    require_nonempty_test: bool = True

    def __post_init__(self) -> None:
        """Validate partition-size options."""

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

        if not 0.0 <= self.maximum_ratio_deviation <= 1.0:
            raise ValueError(
                "maximum_ratio_deviation must be between "
                "zero and one."
            )


class SplitSizeValidator:
    """Validate partition sizes and actual split ratios."""

    component_name = "split_sizes"

    def __init__(
        self,
        options: SplitSizeValidationOptions | None = None,
    ) -> None:
        """Initialize the partition-size validator."""

        self.options = (
            options or SplitSizeValidationOptions()
        )

    def validate(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
    ) -> list[SplitFinding]:
        """Validate partition sizes."""

        if not self.options.enabled:
            return []

        findings: list[SplitFinding] = []

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
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

        for name, partition in partitions.items():
            row_count = partition.row_count()
            required_count = minimum_rows[name]

            if row_count >= required_count:
                continue

            findings.append(
                SplitFinding(
                    code="P2-SIZE-001",
                    category=(
                        SplitFindingCategory.SPLIT_SIZE
                    ),
                    severity=SplitFindingSeverity.ERROR,
                    component=self.component_name,
                    message=(
                        f"Partition {name.value!r} contains "
                        f"{row_count} rows, below the required "
                        f"minimum of {required_count}."
                    ),
                    partitions=[name],
                    affected_row_count=row_count,
                    recommended_action=(
                        SplitFindingAction.RESPLIT
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                    details={
                        "actual_row_count": row_count,
                        "minimum_row_count": required_count,
                    },
                )
            )

        if (
            self.options.require_nonempty_validation
            and validation.row_count() == 0
        ):
            findings.append(
                SplitFinding(
                    code="P2-SIZE-002",
                    category=(
                        SplitFindingCategory.SPLIT_SIZE
                    ),
                    severity=SplitFindingSeverity.ERROR,
                    component=self.component_name,
                    message=(
                        "The validation partition is empty."
                    ),
                    partitions=[
                        PartitionName.VALIDATION
                    ],
                    recommended_action=(
                        SplitFindingAction.RESPLIT
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        if (
            self.options.require_nonempty_test
            and test.row_count() == 0
        ):
            findings.append(
                SplitFinding(
                    code="P2-SIZE-003",
                    category=(
                        SplitFindingCategory.SPLIT_SIZE
                    ),
                    severity=SplitFindingSeverity.CRITICAL,
                    component=self.component_name,
                    message="The test partition is empty.",
                    partitions=[PartitionName.TEST],
                    recommended_action=(
                        SplitFindingAction.RESPLIT
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        requested_ratios = (
            self.options.requested_ratios
        )

        if requested_ratios is None:
            return findings

        total_count = sum(
            partition.row_count()
            for partition in partitions.values()
        )

        if total_count == 0:
            return findings

        expected_ratios = {
            PartitionName.TRAIN: (
                requested_ratios.train
            ),
            PartitionName.VALIDATION: (
                requested_ratios.validation
            ),
            PartitionName.TEST: (
                requested_ratios.test
            ),
        }

        for name, partition in partitions.items():
            actual_ratio = (
                partition.row_count()
                / total_count
            )

            expected_ratio = expected_ratios[name]

            deviation = abs(
                actual_ratio - expected_ratio
            )

            if (
                deviation
                <= self.options.maximum_ratio_deviation
            ):
                continue

            findings.append(
                SplitFinding(
                    code="P2-SIZE-004",
                    category=(
                        SplitFindingCategory.SPLIT_SIZE
                    ),
                    severity=SplitFindingSeverity.WARNING,
                    component=self.component_name,
                    message=(
                        f"Partition {name.value!r} deviates "
                        "from its requested ratio."
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
                        "requested_ratio": expected_ratio,
                        "actual_ratio": actual_ratio,
                        "deviation": deviation,
                        "maximum_deviation": (
                            self.options
                            .maximum_ratio_deviation
                        ),
                    },
                )
            )

        return findings
