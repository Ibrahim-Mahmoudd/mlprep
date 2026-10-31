"""Duplicate feature-row overlap detection for Phase 2."""

from __future__ import annotations

import hashlib
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
class DuplicateOverlapOptions:
    """Options controlling duplicate-overlap checks."""

    enabled: bool = True

    include_target_in_hash: bool = False
    block_exact_duplicates: bool = True

    maximum_examples: int = 10

    def __post_init__(self) -> None:
        """Validate duplicate-overlap options."""

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )


class DuplicateOverlapChecker:
    """Detect identical records appearing in multiple partitions."""

    component_name = "duplicate_overlap"

    def __init__(
        self,
        options: DuplicateOverlapOptions | None = None,
    ) -> None:
        """Initialize the duplicate-overlap checker."""

        self.options = (
            options or DuplicateOverlapOptions()
        )

    def inspect(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
    ) -> list[SplitFinding]:
        """Inspect all partition pairs for identical records."""

        if not self.options.enabled:
            return []

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        hashes = {
            name: self._partition_hashes(partition)
            for name, partition in partitions.items()
        }

        findings: list[SplitFinding] = []
        names = list(partitions)

        for position, left_name in enumerate(names):
            for right_name in names[position + 1:]:
                left_hashes = hashes[left_name]
                right_hashes = hashes[right_name]

                overlap = set(
                    left_hashes.tolist()
                ).intersection(
                    right_hashes.tolist()
                )

                if not overlap:
                    continue

                example_row_ids: list[
                    str | int
                ] = []

                for fingerprint in overlap:
                    matching_indices = (
                        right_hashes[
                            right_hashes
                            == fingerprint
                        ].index
                    )

                    for index in matching_indices:
                        value = partitions[
                            right_name
                        ].row_ids.loc[index]

                        example_row_ids.append(
                            self._safe_identifier(value)
                        )

                        if (
                            len(example_row_ids)
                            >= self.options
                            .maximum_examples
                        ):
                            break

                    if (
                        len(example_row_ids)
                        >= self.options.maximum_examples
                    ):
                        break

                findings.append(
                    SplitFinding(
                        code="P2-DUPLICATE-OVERLAP-001",
                        category=(
                            SplitFindingCategory.DUPLICATE_OVERLAP
                        ),
                        severity=SplitFindingSeverity.ERROR,
                        component=self.component_name,
                        message=(
                            f"Partitions {left_name.value!r} and "
                            f"{right_name.value!r} contain "
                            "identical records."
                        ),
                        partitions=[
                            left_name,
                            right_name,
                        ],
                        affected_row_count=len(overlap),
                        example_row_ids=example_row_ids,
                        recommended_action=(
                            SplitFindingAction.REMOVE_OVERLAP
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=(
                            self.options.block_exact_duplicates
                        ),
                        details={
                            "unique_overlapping_hashes": len(
                                overlap
                            ),
                            "target_included": (
                                self.options
                                .include_target_in_hash
                            ),
                        },
                    )
                )

        return findings

    def _partition_hashes(
        self,
        partition: DatasetPartition,
    ) -> pd.Series:
        """Build one stable hash for each partition record."""

        frame = partition.X.copy()

        if self.options.include_target_in_hash:
            frame["__mlprep_target__"] = (
                partition.y
            )

        pandas_hashes = pd.util.hash_pandas_object(
            frame,
            index=False,
            categorize=True,
        )

        return pandas_hashes.map(
            lambda value: hashlib.sha256(
                str(int(value)).encode("utf-8")
            ).hexdigest()
        )

    @staticmethod
    def _safe_identifier(
        value: object,
    ) -> str | int:
        """Convert a row ID into a supported report value."""

        if isinstance(value, (str, int)):
            return value

        return str(value)
