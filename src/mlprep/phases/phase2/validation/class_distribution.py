"""Classification-target distribution validation for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    SplitFinding,
)


@dataclass(slots=True)
class ClassDistributionValidationOptions:
    """Options controlling class-distribution validation."""

    enabled: bool = True

    require_all_classes_in_train: bool = True
    require_all_classes_in_validation: bool = True
    require_all_classes_in_test: bool = True

    minimum_class_count: int = 1
    maximum_distribution_deviation: float = 0.10

    reference_partition: PartitionName | None = None

    def __post_init__(self) -> None:
        """Validate class-distribution options."""

        if self.minimum_class_count < 1:
            raise ValueError(
                "minimum_class_count must be at least one."
            )

        if not (
            0.0
            <= self.maximum_distribution_deviation
            <= 1.0
        ):
            raise ValueError(
                "maximum_distribution_deviation must be "
                "between zero and one."
            )


class ClassDistributionValidator:
    """Validate classification distributions across partitions."""

    component_name = "class_distribution"

    def __init__(
        self,
        options: (
            ClassDistributionValidationOptions | None
        ) = None,
    ) -> None:
        """Initialize the class-distribution validator."""

        self.options = (
            options
            or ClassDistributionValidationOptions()
        )

    def validate(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
    ) -> list[SplitFinding]:
        """Validate class presence and distribution stability."""

        if not self.options.enabled:
            return []

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        findings: list[SplitFinding] = []

        combined_target = pd.concat(
            [
                train.y,
                validation.y,
                test.y,
            ]
        )

        if combined_target.isna().any():
            findings.append(
                SplitFinding(
                    code="P2-CLASS-001",
                    category=(
                        SplitFindingCategory
                        .CLASS_DISTRIBUTION
                    ),
                    severity=SplitFindingSeverity.ERROR,
                    component=self.component_name,
                    message=(
                        "Classification partitions contain "
                        "missing target values."
                    ),
                    affected_row_count=int(
                        combined_target.isna().sum()
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

        all_classes = set(
            combined_target
            .dropna()
            .unique()
            .tolist()
        )

        reference_distribution = (
            self._reference_distribution(
                partitions=partitions,
                combined_target=combined_target,
            )
        )

        enforcement = {
            PartitionName.TRAIN: (
                self.options.require_all_classes_in_train
            ),
            PartitionName.VALIDATION: (
                self.options
                .require_all_classes_in_validation
            ),
            PartitionName.TEST: (
                self.options.require_all_classes_in_test
            ),
        }

        for name, partition in partitions.items():
            counts = partition.y.value_counts(
                dropna=False
            )

            partition_classes = set(
                partition.y
                .dropna()
                .unique()
                .tolist()
            )

            missing_classes = all_classes.difference(
                partition_classes
            )

            if (
                missing_classes
                and enforcement[name]
            ):
                findings.append(
                    SplitFinding(
                        code="P2-CLASS-002",
                        category=(
                            SplitFindingCategory
                            .CLASS_DISTRIBUTION
                        ),
                        severity=SplitFindingSeverity.ERROR,
                        component=self.component_name,
                        message=(
                            f"Partition {name.value!r} is "
                            "missing one or more target classes."
                        ),
                        partitions=[name],
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
                                for value in sorted(
                                    missing_classes,
                                    key=str,
                                )
                            ],
                        },
                    )
                )

            small_classes = {
                str(class_value): int(count)
                for class_value, count
                in counts.items()
                if (
                    pd.notna(class_value)
                    and count
                    < self.options.minimum_class_count
                )
            }

            if small_classes:
                findings.append(
                    SplitFinding(
                        code="P2-CLASS-003",
                        category=(
                            SplitFindingCategory
                            .CLASS_DISTRIBUTION
                        ),
                        severity=SplitFindingSeverity.ERROR,
                        component=self.component_name,
                        message=(
                            f"Partition {name.value!r} has "
                            "classes below the minimum count."
                        ),
                        partitions=[name],
                        recommended_action=(
                            SplitFindingAction.RESPLIT
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=True,
                        details={
                            "small_classes": small_classes,
                            "minimum_class_count": (
                                self.options
                                .minimum_class_count
                            ),
                        },
                    )
                )

            distribution = partition.y.value_counts(
                normalize=True,
                dropna=False,
            )

            deviations: dict[str, float] = {}

            for class_value in all_classes:
                reference_ratio = float(
                    reference_distribution.get(
                        class_value,
                        0.0,
                    )
                )

                partition_ratio = float(
                    distribution.get(
                        class_value,
                        0.0,
                    )
                )

                deviation = abs(
                    partition_ratio
                    - reference_ratio
                )

                if (
                    deviation
                    > self.options
                    .maximum_distribution_deviation
                ):
                    deviations[
                        str(class_value)
                    ] = deviation

            if deviations:
                findings.append(
                    SplitFinding(
                        code="P2-CLASS-004",
                        category=(
                            SplitFindingCategory
                            .CLASS_DISTRIBUTION
                        ),
                        severity=SplitFindingSeverity.WARNING,
                        component=self.component_name,
                        message=(
                            f"Partition {name.value!r} class "
                            "distribution exceeds tolerance."
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
                            "deviations": deviations,
                            "tolerance": (
                                self.options
                                .maximum_distribution_deviation
                            ),
                        },
                    )
                )

        return findings

    def _reference_distribution(
        self,
        *,
        partitions: dict[
            PartitionName,
            DatasetPartition,
        ],
        combined_target: pd.Series,
    ) -> pd.Series:
        """Return the configured reference distribution."""

        reference_partition = (
            self.options.reference_partition
        )

        if reference_partition is None:
            return combined_target.value_counts(
                normalize=True,
                dropna=False,
            )

        return partitions[
            reference_partition
        ].y.value_counts(
            normalize=True,
            dropna=False,
        )
