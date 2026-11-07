"""Partition-integrity validation for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

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
class PartitionIntegrityValidationOptions:
    """Options controlling partition-integrity validation."""

    enabled: bool = True

    require_identical_feature_order: bool = True
    require_identical_feature_dtypes: bool = True
    require_unique_row_ids: bool = True
    require_disjoint_row_ids: bool = True
    require_aligned_indices: bool = True


class PartitionIntegrityValidator:
    """Validate structural integrity across partitions."""

    component_name = "partition_integrity"

    def __init__(
        self,
        options: (
            PartitionIntegrityValidationOptions | None
        ) = None,
    ) -> None:
        """Initialize the integrity validator."""

        self.options = (
            options
            or PartitionIntegrityValidationOptions()
        )

    def validate(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
    ) -> list[object]:
        """Validate partition alignment and compatibility."""

        if not self.options.enabled:
            return []

        findings: list[SplitFinding] = []

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        for name, partition in partitions.items():
            if (
                self.options.require_aligned_indices
                and (
                    not partition.X.index.equals(
                        partition.y.index
                    )
                    or not partition.X.index.equals(
                        partition.row_ids.index
                    )
                )
            ):
                findings.append(
                    SplitFinding(
                        code="P2-INTEGRITY-001",
                        category=(
                            SplitFindingCategory
                            .PARTITION_INTEGRITY
                        ),
                        severity=(
                            SplitFindingSeverity.CRITICAL
                        ),
                        component=self.component_name,
                        message=(
                            f"Partition {name.value!r} has "
                            "misaligned indices."
                        ),
                        partitions=[name],
                        recommended_action=(
                            SplitFindingAction.RAISE
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=True,
                    )
                )

            if (
                self.options.require_unique_row_ids
                and not partition.row_ids.is_unique
            ):
                findings.append(
                    SplitFinding(
                        code="P2-INTEGRITY-002",
                        category=(
                            SplitFindingCategory
                            .PARTITION_INTEGRITY
                        ),
                        severity=(
                            SplitFindingSeverity.CRITICAL
                        ),
                        component=self.component_name,
                        message=(
                            f"Partition {name.value!r} contains "
                            "duplicate row IDs."
                        ),
                        partitions=[name],
                        recommended_action=(
                            SplitFindingAction.RESPLIT
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=True,
                    )
                )

        reference_features = train.feature_names()

        for name in (
            PartitionName.VALIDATION,
            PartitionName.TEST,
        ):
            partition = partitions[name]

            if (
                self.options
                .require_identical_feature_order
                and partition.feature_names()
                != reference_features
            ):
                findings.append(
                    SplitFinding(
                        code="P2-INTEGRITY-003",
                        category=(
                            SplitFindingCategory
                            .PARTITION_INTEGRITY
                        ),
                        severity=(
                            SplitFindingSeverity.CRITICAL
                        ),
                        component=self.component_name,
                        message=(
                            f"Partition {name.value!r} feature "
                            "order differs from training."
                        ),
                        partitions=[
                            PartitionName.TRAIN,
                            name,
                        ],
                        recommended_action=(
                            SplitFindingAction.RAISE
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=True,
                    )
                )

            if (
                self.options
                .require_identical_feature_dtypes
            ):
                dtype_mismatches = {
                    column: {
                        "train": str(
                            train.X[column].dtype
                        ),
                        name.value: str(
                            partition.X[column].dtype
                        ),
                    }
                    for column in reference_features
                    if (
                        column in partition.X.columns
                        and train.X[column].dtype
                        != partition.X[column].dtype
                    )
                }

                if dtype_mismatches:
                    findings.append(
                        SplitFinding(
                            code="P2-INTEGRITY-004",
                            category=(
                                SplitFindingCategory
                                .PARTITION_INTEGRITY
                            ),
                            severity=(
                                SplitFindingSeverity.ERROR
                            ),
                            component=self.component_name,
                            message=(
                                f"Partition {name.value!r} "
                                "feature dtypes differ from "
                                "training."
                            ),
                            partitions=[
                                PartitionName.TRAIN,
                                name,
                            ],
                            columns=sorted(
                                dtype_mismatches
                            ),
                            recommended_action=(
                                SplitFindingAction.RAISE
                            ),
                            applied_action=(
                                SplitFindingAction.REPORT
                            ),
                            blocks_readiness=True,
                            details={
                                "dtype_mismatches": (
                                    dtype_mismatches
                                ),
                            },
                        )
                    )

        if self.options.require_disjoint_row_ids:
            names = list(partitions)

            for position, left_name in enumerate(
                names
            ):
                left_ids = set(
                    partitions[
                        left_name
                    ].row_ids.tolist()
                )

                for right_name in names[
                    position + 1:
                ]:
                    overlap = left_ids.intersection(
                        partitions[
                            right_name
                        ].row_ids.tolist()
                    )

                    if not overlap:
                        continue

                    findings.append(
                        SplitFinding(
                            code="P2-INTEGRITY-005",
                            category=(
                                SplitFindingCategory
                                .PARTITION_INTEGRITY
                            ),
                            severity=(
                                SplitFindingSeverity.CRITICAL
                            ),
                            component=self.component_name,
                            message=(
                                f"Partitions "
                                f"{left_name.value!r} and "
                                f"{right_name.value!r} share "
                                "internal row IDs."
                            ),
                            partitions=[
                                left_name,
                                right_name,
                            ],
                            affected_row_count=len(
                                overlap
                            ),
                            example_row_ids=[
                                value
                                if isinstance(
                                    value,
                                    (str, int),
                                )
                                else str(value)
                                for value
                                in list(overlap)[:10]
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

        return findings
