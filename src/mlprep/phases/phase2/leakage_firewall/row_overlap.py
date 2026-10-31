"""Cross-partition row-overlap detection for Phase 2."""

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
class RowOverlapOptions:
    """Options controlling row-overlap checks."""

    enabled: bool = True
    maximum_examples: int = 10

    def __post_init__(self) -> None:
        """Validate row-overlap options."""

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )


class RowOverlapChecker:
    """Detect internal row IDs shared by multiple partitions."""

    component_name = "row_overlap"

    def __init__(
        self,
        options: RowOverlapOptions | None = None,
    ) -> None:
        """Initialize the row-overlap checker."""

        self.options = options or RowOverlapOptions()

    def inspect(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
    ) -> tuple[
        list[SplitFinding],
        dict[str, list[str | int]],
    ]:
        """Inspect all partition pairs for row-ID overlap."""

        if not self.options.enabled:
            return [], {}

        findings: list[SplitFinding] = []
        overlap_state: dict[
            str,
            list[str | int],
        ] = {}

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        names = list(partitions)

        for position, left_name in enumerate(names):
            left_ids = set(
                partitions[left_name]
                .row_ids
                .tolist()
            )

            for right_name in names[position + 1:]:
                right_ids = set(
                    partitions[right_name]
                    .row_ids
                    .tolist()
                )

                overlap = left_ids.intersection(
                    right_ids
                )

                if not overlap:
                    continue

                pair_name = (
                    f"{left_name.value}__"
                    f"{right_name.value}"
                )

                overlap_values = list(overlap)

                overlap_state[pair_name] = (
                    overlap_values
                )

                findings.append(
                    SplitFinding(
                        code="P2-ROW-OVERLAP-001",
                        category=(
                            SplitFindingCategory.ROW_OVERLAP
                        ),
                        severity=SplitFindingSeverity.CRITICAL,
                        component=self.component_name,
                        message=(
                            f"Partitions {left_name.value!r} and "
                            f"{right_name.value!r} contain "
                            "overlapping internal row IDs."
                        ),
                        partitions=[
                            left_name,
                            right_name,
                        ],
                        affected_row_count=len(
                            overlap_values
                        ),
                        example_row_ids=overlap_values[
                            : self.options.maximum_examples
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

        return findings, overlap_state