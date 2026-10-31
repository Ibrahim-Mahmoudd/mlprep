"""Target-integrity validation for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase8.lineage import (
    RowLineageState,
)
from mlprep.phases.phase8.samplers import (
    ResampledRowType,
)


class PartitionTargetIntegrity(BaseModel):
    """Target-integrity status for one partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    partition: PartitionName

    valid: bool = True

    row_count_unchanged: bool = True
    index_unchanged: bool = True
    dtype_unchanged: bool = True
    values_unchanged: bool = True
    missingness_unchanged: bool = True

    violations: list[str] = Field(
        default_factory=list,
    )


class TargetIntegrityResult(BaseModel):
    """Complete Phase 8 target-integrity validation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    training_lineage_targets_valid: bool = True
    training_class_labels_valid: bool = True

    validation: PartitionTargetIntegrity
    test: PartitionTargetIntegrity

    invalid_training_row_ids: list[str] = Field(
        default_factory=list,
    )

    unexpected_training_classes: list[str] = Field(
        default_factory=list,
    )

    blocking_reasons: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class TargetIntegrityOptions:
    """Options controlling Phase 8 target validation."""

    enabled: bool = True

    require_training_lineage_targets: bool = True
    reject_unexpected_training_classes: bool = True

    require_identical_validation_target: bool = True
    require_identical_test_target: bool = True


class TargetIntegrityValidator:
    """Validate training lineage and protected targets."""

    component_name = "target_integrity"

    def __init__(
        self,
        options: TargetIntegrityOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or TargetIntegrityOptions()
        )

    def validate(
        self,
        *,
        source_train: DatasetPartition,
        output_train: DatasetPartition,
        source_validation: DatasetPartition,
        output_validation: DatasetPartition,
        source_test: DatasetPartition,
        output_test: DatasetPartition,
        lineage: RowLineageState,
    ) -> TargetIntegrityResult:
        """Validate training, validation, and test targets."""

        validation_result = (
            self._protected_target(
                source=source_validation,
                output=output_validation,
            )
        )

        test_result = self._protected_target(
            source=source_test,
            output=output_test,
        )

        if not self.options.enabled:
            return TargetIntegrityResult(
                validation=validation_result,
                test=test_result,
                metadata={
                    "enabled": False,
                },
            )

        source_classes = list(
            source_train.y.dropna().unique()
        )

        invalid_training_ids: list[str] = []
        unexpected_classes: set[str] = set()

        for entry in lineage.entries:
            output_target = output_train.y.iloc[
                entry.output_position
            ]

            if not self._values_equal(
                output_target,
                entry.class_value,
            ):
                invalid_training_ids.append(
                    entry.output_row_id
                )

                continue

            if entry.row_type in {
                ResampledRowType.ORIGINAL,
                ResampledRowType.DUPLICATE,
                ResampledRowType
                .RETAINED_AFTER_CLEANING,
            }:
                if not entry.source_positions:
                    invalid_training_ids.append(
                        entry.output_row_id
                    )

                    continue

                source_target = source_train.y.iloc[
                    entry.source_positions[
                        0
                    ]
                ]

                if not self._values_equal(
                    source_target,
                    output_target,
                ):
                    invalid_training_ids.append(
                        entry.output_row_id
                    )

            if not any(
                self._values_equal(
                    output_target,
                    class_value,
                )
                for class_value
                in source_classes
            ):
                unexpected_classes.add(
                    str(
                        output_target
                    )
                )

        training_lineage_valid = bool(
            not invalid_training_ids
        )

        training_classes_valid = bool(
            not unexpected_classes
        )

        blocking_reasons: list[str] = []

        if (
            self.options
            .require_training_lineage_targets
            and not training_lineage_valid
        ):
            blocking_reasons.append(
                "One or more training targets conflict with "
                "their row lineage."
            )

        if (
            self.options
            .reject_unexpected_training_classes
            and not training_classes_valid
        ):
            blocking_reasons.append(
                "The resampled training target contains "
                "unexpected classes."
            )

        if (
            self.options
            .require_identical_validation_target
            and not validation_result.valid
        ):
            blocking_reasons.append(
                "The validation target changed during Phase 8."
            )

        if (
            self.options.require_identical_test_target
            and not test_result.valid
        ):
            blocking_reasons.append(
                "The test target changed during Phase 8."
            )

        return TargetIntegrityResult(
            valid=not blocking_reasons,
            training_lineage_targets_valid=(
                training_lineage_valid
            ),
            training_class_labels_valid=(
                training_classes_valid
            ),
            validation=validation_result,
            test=test_result,
            invalid_training_row_ids=(
                invalid_training_ids
            ),
            unexpected_training_classes=sorted(
                unexpected_classes
            ),
            blocking_reasons=(
                blocking_reasons
            ),
            metadata={
                "enabled": True,
            },
        )

    @staticmethod
    def _protected_target(
        *,
        source: DatasetPartition,
        output: DatasetPartition,
    ) -> PartitionTargetIntegrity:
        """Compare a protected partition target."""

        row_count_unchanged = bool(
            len(
                source.y
            )
            == len(
                output.y
            )
        )

        index_unchanged = bool(
            source.y.index.equals(
                output.y.index
            )
        )

        dtype_unchanged = bool(
            str(
                source.y.dtype
            )
            == str(
                output.y.dtype
            )
        )

        values_unchanged = bool(
            source.y.equals(
                output.y
            )
        )

        missingness_unchanged = bool(
            source.y.isna().equals(
                output.y.isna()
            )
        )

        violations: list[str] = []

        checks = [
            (
                row_count_unchanged,
                "row_count_changed",
            ),
            (
                index_unchanged,
                "index_changed",
            ),
            (
                dtype_unchanged,
                "dtype_changed",
            ),
            (
                values_unchanged,
                "values_changed",
            ),
            (
                missingness_unchanged,
                "missingness_changed",
            ),
        ]

        for passed, violation in checks:
            if not passed:
                violations.append(
                    violation
                )

        return PartitionTargetIntegrity(
            partition=source.name,
            valid=not violations,
            row_count_unchanged=(
                row_count_unchanged
            ),
            index_unchanged=(
                index_unchanged
            ),
            dtype_unchanged=(
                dtype_unchanged
            ),
            values_unchanged=(
                values_unchanged
            ),
            missingness_unchanged=(
                missingness_unchanged
            ),
            violations=violations,
        )

    @staticmethod
    def _values_equal(
        left: object,
        right: object,
    ) -> bool:
        """Safely compare target values."""

        try:
            comparison = (
                left == right
            )

        except Exception:
            return False

        if isinstance(
            comparison,
            (bool, np.bool_),
        ):
            return bool(
                comparison
            )

        return False