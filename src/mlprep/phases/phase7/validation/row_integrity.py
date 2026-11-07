"""Row-integrity validation for Phase 7."""

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


class RowIntegrityIssue(BaseModel):
    """Row-integrity issue for one partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    partition: PartitionName

    source_row_count: int = Field(
        default=0,
        ge=0,
    )

    transformed_row_count: int = Field(
        default=0,
        ge=0,
    )

    row_count_changed: bool = False
    feature_index_changed: bool = False
    target_index_changed: bool = False
    row_ids_changed: bool = False

    duplicate_row_ids: list[
        str | int
    ] = Field(
        default_factory=list,
    )

    missing_row_id_count: int = Field(
        default=0,
        ge=0,
    )

    blocks_readiness: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class RowIntegrityResult(BaseModel):
    """Complete row-integrity validation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    issues: list[
        RowIntegrityIssue
    ] = Field(
        default_factory=list,
    )

    cross_partition_overlaps: dict[
        str,
        list[str | int],
    ] = Field(
        default_factory=dict,
    )

    unchanged_partitions: list[
        PartitionName
    ] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class RowIntegrityOptions:
    """Options controlling row-integrity validation."""

    enabled: bool = True

    require_identical_row_count: bool = True
    require_identical_feature_index: bool = True
    require_identical_target_index: bool = True
    require_identical_row_ids: bool = True

    require_unique_row_ids: bool = True
    require_nonmissing_row_ids: bool = True
    require_disjoint_partitions: bool = True

    maximum_reported_values: int = 20

    def __post_init__(self) -> None:
        """Validate row-integrity options."""

        if self.maximum_reported_values < 1:
            raise ValueError(
                "maximum_reported_values must be positive."
            )


class RowIntegrityValidator:
    """Validate rows after selected-schema application."""

    component_name = "row_integrity"

    def __init__(
        self,
        options: RowIntegrityOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or RowIntegrityOptions()
        )

    def validate(
        self,
        *,
        source_train: DatasetPartition,
        transformed_train: DatasetPartition,
        source_validation: DatasetPartition,
        transformed_validation: DatasetPartition,
        source_test: DatasetPartition,
        transformed_test: DatasetPartition,
    ) -> RowIntegrityResult:
        """Validate row integrity across all partitions."""

        if not self.options.enabled:
            return RowIntegrityResult(
                metadata={
                    "enabled": False,
                },
            )

        pairs = [
            (
                source_train,
                transformed_train,
            ),
            (
                source_validation,
                transformed_validation,
            ),
            (
                source_test,
                transformed_test,
            ),
        ]

        issues: list[
            RowIntegrityIssue
        ] = []

        unchanged: list[
            PartitionName
        ] = []

        transformed_partitions: dict[
            PartitionName,
            DatasetPartition,
        ] = {}

        for source, transformed in pairs:
            transformed_partitions[
                transformed.name
            ] = transformed

            issue = self._compare(
                source=source,
                transformed=transformed,
            )

            if issue is None:
                unchanged.append(
                    source.name
                )

            else:
                issues.append(
                    issue
                )

        overlaps = (
            self._find_overlaps(
                transformed_partitions
            )
            if self.options.require_disjoint_partitions
            else {}
        )

        return RowIntegrityResult(
            valid=bool(
                not any(
                    issue.blocks_readiness
                    for issue in issues
                )
                and not overlaps
            ),
            issues=issues,
            cross_partition_overlaps=overlaps,
            unchanged_partitions=unchanged,
            metadata={
                "enabled": True,
            },
        )

    def _compare(
        self,
        *,
        source: DatasetPartition,
        transformed: DatasetPartition,
    ) -> RowIntegrityIssue | None:
        """Compare rows before and after selection."""

        row_count_changed = bool(
            source.row_count()
            != transformed.row_count()
        )

        feature_index_changed = bool(
            not source.X.index.equals(
                transformed.X.index
            )
        )

        target_index_changed = bool(
            not source.y.index.equals(
                transformed.y.index
            )
        )

        row_ids_changed = bool(
            not source.row_ids.equals(
                transformed.row_ids
            )
        )

        missing_row_id_count = int(
            transformed.row_ids.isna().sum()
        )

        duplicate_mask = (
            transformed.row_ids.duplicated(
                keep=False
            )
        )

        duplicate_values = [
            self._safe_identifier(
                value
            )
            for value
            in transformed.row_ids.loc[
                duplicate_mask
            ].drop_duplicates().tolist()[
                : self.options.maximum_reported_values
            ]
        ]

        blocks_readiness = bool(
            (
                self.options.require_identical_row_count
                and row_count_changed
            )
            or (
                self.options.require_identical_feature_index
                and feature_index_changed
            )
            or (
                self.options.require_identical_target_index
                and target_index_changed
            )
            or (
                self.options.require_identical_row_ids
                and row_ids_changed
            )
            or (
                self.options.require_unique_row_ids
                and duplicate_values
            )
            or (
                self.options.require_nonmissing_row_ids
                and missing_row_id_count > 0
            )
        )

        if not any(
            (
                row_count_changed,
                feature_index_changed,
                target_index_changed,
                row_ids_changed,
                bool(
                    duplicate_values
                ),
                missing_row_id_count > 0,
            )
        ):
            return None

        return RowIntegrityIssue(
            partition=source.name,
            source_row_count=(
                source.row_count()
            ),
            transformed_row_count=(
                transformed.row_count()
            ),
            row_count_changed=(
                row_count_changed
            ),
            feature_index_changed=(
                feature_index_changed
            ),
            target_index_changed=(
                target_index_changed
            ),
            row_ids_changed=(
                row_ids_changed
            ),
            duplicate_row_ids=(
                duplicate_values
            ),
            missing_row_id_count=(
                missing_row_id_count
            ),
            blocks_readiness=(
                blocks_readiness
            ),
        )

    def _find_overlaps(
        self,
        partitions: dict[
            PartitionName,
            DatasetPartition,
        ],
    ) -> dict[
        str,
        list[str | int],
    ]:
        """Find row-ID overlaps across partitions."""

        overlaps: dict[
            str,
            list[str | int],
        ] = {}

        partition_items = list(
            partitions.items()
        )

        for position, (
            left_name,
            left_partition,
        ) in enumerate(
            partition_items
        ):
            left_ids = set(
                left_partition.row_ids.dropna().tolist()
            )

            for (
                right_name,
                right_partition,
            ) in partition_items[
                position + 1:
            ]:
                right_ids = set(
                    right_partition.row_ids.dropna().tolist()
                )

                overlap = left_ids.intersection(
                    right_ids
                )

                if not overlap:
                    continue

                key = (
                    f"{left_name.value}"
                    f"__{right_name.value}"
                )

                overlaps[
                    key
                ] = [
                    self._safe_identifier(
                        value
                    )
                    for value in list(
                        overlap
                    )[
                        : self.options
                        .maximum_reported_values
                    ]
                ]

        return overlaps

    @staticmethod
    def _safe_identifier(
        value: object,
    ) -> str | int:
        """Return a report-safe row identifier."""

        if isinstance(
            value,
            (str, int),
        ):
            return value

        return str(
            value
        )
