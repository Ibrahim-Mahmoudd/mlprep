"""Partition-integrity validation for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase8.lineage import (
    RowLineageState,
)
from mlprep.phases.phase8.orchestration import (
    PartitionGuardResult,
)


class PartitionIntegrityResult(BaseModel):
    """Complete Phase 8 partition-integrity validation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    training_indices_aligned: bool = True
    training_row_ids_unique: bool = True
    training_row_ids_nonmissing: bool = True

    training_lineage_complete: bool = True
    removed_lineage_complete: bool = True

    validation_unchanged: bool = True
    test_unchanged: bool = True

    validation_contains_no_generated_rows: bool = True
    test_contains_no_generated_rows: bool = True

    blocking_reasons: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class PartitionIntegrityOptions:
    """Options controlling partition-integrity validation."""

    enabled: bool = True

    require_training_index_alignment: bool = True
    require_unique_training_row_ids: bool = True
    require_nonmissing_training_row_ids: bool = True

    require_complete_output_lineage: bool = True
    require_complete_removed_lineage: bool = True

    require_unchanged_validation: bool = True
    require_unchanged_test: bool = True

    reject_generated_validation_rows: bool = True
    reject_generated_test_rows: bool = True

    synthetic_prefix: str = "phase8__synthetic"
    duplicate_prefix: str = "phase8__duplicate"


class PartitionIntegrityValidator:
    """Validate training and protected partition integrity."""

    component_name = "partition_integrity"

    def __init__(
        self,
        options: PartitionIntegrityOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or PartitionIntegrityOptions()
        )

    def validate(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        lineage: RowLineageState,
        validation_guard: PartitionGuardResult,
        test_guard: PartitionGuardResult,
    ) -> PartitionIntegrityResult:
        """Validate every Phase 8 partition."""

        training_aligned = bool(
            train.X.index.equals(
                train.y.index
            )
            and train.X.index.equals(
                train.row_ids.index
            )
        )

        training_ids_unique = bool(
            train.row_ids.is_unique
        )

        training_ids_nonmissing = bool(
            not train.row_ids.isna().any()
        )

        output_lineage_complete = bool(
            lineage.complete_output_coverage
            and len(
                lineage.entries
            )
            == train.row_count()
        )

        removed_lineage_complete = bool(
            lineage.complete_removed_row_coverage
        )

        validation_unchanged = bool(
            validation_guard.valid
        )

        test_unchanged = bool(
            test_guard.valid
        )

        validation_generated_ids = (
            self._generated_ids(
                validation
            )
        )

        test_generated_ids = self._generated_ids(
            test
        )

        validation_contains_no_generated = bool(
            not validation_generated_ids
        )

        test_contains_no_generated = bool(
            not test_generated_ids
        )

        if not self.options.enabled:
            return PartitionIntegrityResult(
                training_indices_aligned=(
                    training_aligned
                ),
                training_row_ids_unique=(
                    training_ids_unique
                ),
                training_row_ids_nonmissing=(
                    training_ids_nonmissing
                ),
                training_lineage_complete=(
                    output_lineage_complete
                ),
                removed_lineage_complete=(
                    removed_lineage_complete
                ),
                validation_unchanged=(
                    validation_unchanged
                ),
                test_unchanged=(
                    test_unchanged
                ),
                metadata={
                    "enabled": False,
                },
            )

        blocking_reasons: list[str] = []

        if (
            self.options
            .require_training_index_alignment
            and not training_aligned
        ):
            blocking_reasons.append(
                "Training features, target, and row IDs "
                "have different indices."
            )

        if (
            self.options
            .require_unique_training_row_ids
            and not training_ids_unique
        ):
            blocking_reasons.append(
                "Training row IDs are not unique."
            )

        if (
            self.options
            .require_nonmissing_training_row_ids
            and not training_ids_nonmissing
        ):
            blocking_reasons.append(
                "Training row IDs contain missing values."
            )

        if (
            self.options
            .require_complete_output_lineage
            and not output_lineage_complete
        ):
            blocking_reasons.append(
                "Output row lineage does not cover every "
                "training row."
            )

        if (
            self.options
            .require_complete_removed_lineage
            and not removed_lineage_complete
        ):
            blocking_reasons.append(
                "Removed-row lineage is incomplete."
            )

        if (
            self.options.require_unchanged_validation
            and not validation_unchanged
        ):
            blocking_reasons.append(
                "The validation partition changed."
            )

        if (
            self.options.require_unchanged_test
            and not test_unchanged
        ):
            blocking_reasons.append(
                "The test partition changed."
            )

        if (
            self.options
            .reject_generated_validation_rows
            and not validation_contains_no_generated
        ):
            blocking_reasons.append(
                "Generated Phase 8 row IDs appear in the "
                "validation partition."
            )

        if (
            self.options.reject_generated_test_rows
            and not test_contains_no_generated
        ):
            blocking_reasons.append(
                "Generated Phase 8 row IDs appear in the "
                "test partition."
            )

        return PartitionIntegrityResult(
            valid=not blocking_reasons,
            training_indices_aligned=(
                training_aligned
            ),
            training_row_ids_unique=(
                training_ids_unique
            ),
            training_row_ids_nonmissing=(
                training_ids_nonmissing
            ),
            training_lineage_complete=(
                output_lineage_complete
            ),
            removed_lineage_complete=(
                removed_lineage_complete
            ),
            validation_unchanged=(
                validation_unchanged
            ),
            test_unchanged=(
                test_unchanged
            ),
            validation_contains_no_generated_rows=(
                validation_contains_no_generated
            ),
            test_contains_no_generated_rows=(
                test_contains_no_generated
            ),
            blocking_reasons=(
                blocking_reasons
            ),
            metadata={
                "enabled": True,
                "validation_generated_row_ids": (
                    validation_generated_ids
                ),
                "test_generated_row_ids": (
                    test_generated_ids
                ),
                "training_row_count": (
                    train.row_count()
                ),
                "validation_row_count": (
                    validation.row_count()
                ),
                "test_row_count": (
                    test.row_count()
                ),
            },
        )

    def _generated_ids(
        self,
        partition: DatasetPartition,
    ) -> list[object]:
        """Return generated Phase 8 IDs in one partition."""

        prefixes = (
            self.options.synthetic_prefix,
            self.options.duplicate_prefix,
        )

        return [
            str(
                row_id
            )
            for row_id
            in partition.row_ids.tolist()
            if str(
                row_id
            ).startswith(
                prefixes
            )
        ]
