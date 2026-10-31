"""Execution-context controls for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase6.numeric.base import (
    NumericExecutionMode,
)


class TransformationOperation(StrEnum):
    """Supported Phase 6 operations."""

    FIT = "fit"
    TRANSFORM = "transform"
    FIT_TRANSFORM = "fit_transform"


@dataclass(frozen=True, slots=True)
class TransformationFitContext:
    """Execution context for one Phase 6 operation."""

    partition: PartitionName

    operation: TransformationOperation

    numeric_execution_mode: NumericExecutionMode

    training_data: bool

    fitting_allowed: bool

    target_access_allowed: bool = False

    test_partition_authorized: bool = False

    @classmethod
    def training_fit(
        cls,
    ) -> TransformationFitContext:
        """Create a training fitting context."""

        return cls(
            partition=PartitionName.TRAIN,
            operation=(
                TransformationOperation.FIT_TRANSFORM
            ),
            numeric_execution_mode=(
                NumericExecutionMode.TRAINING_TRANSFORM
            ),
            training_data=True,
            fitting_allowed=True,
            target_access_allowed=False,
            test_partition_authorized=False,
        )

    @classmethod
    def training_transform(
        cls,
    ) -> TransformationFitContext:
        """Create a training transform-only context."""

        return cls(
            partition=PartitionName.TRAIN,
            operation=(
                TransformationOperation.TRANSFORM
            ),
            numeric_execution_mode=(
                NumericExecutionMode.TRAINING_TRANSFORM
            ),
            training_data=True,
            fitting_allowed=False,
            target_access_allowed=False,
            test_partition_authorized=False,
        )

    @classmethod
    def validation_transform(
        cls,
    ) -> TransformationFitContext:
        """Create a validation transform-only context."""

        return cls(
            partition=PartitionName.VALIDATION,
            operation=(
                TransformationOperation.TRANSFORM
            ),
            numeric_execution_mode=(
                NumericExecutionMode.VALIDATION_TRANSFORM
            ),
            training_data=False,
            fitting_allowed=False,
            target_access_allowed=False,
            test_partition_authorized=False,
        )

    @classmethod
    def test_transform(
        cls,
        *,
        authorized: bool,
    ) -> TransformationFitContext:
        """Create an authorized test transform context."""

        return cls(
            partition=PartitionName.TEST,
            operation=(
                TransformationOperation.TRANSFORM
            ),
            numeric_execution_mode=(
                NumericExecutionMode.TEST_TRANSFORM
            ),
            training_data=False,
            fitting_allowed=False,
            target_access_allowed=False,
            test_partition_authorized=authorized,
        )

    def assert_fitting_allowed(
        self,
    ) -> None:
        """Raise when fitting is prohibited."""

        if not self.fitting_allowed:
            raise PermissionError(
                "Fitting is not allowed in this context."
            )

        if not self.training_data:
            raise PermissionError(
                "Phase 6 fitting requires training data."
            )

        if self.partition != PartitionName.TRAIN:
            raise PermissionError(
                "Phase 6 components may only be fitted on "
                "the training partition."
            )

        if (
            self.operation
            != TransformationOperation.FIT_TRANSFORM
        ):
            raise PermissionError(
                "The fitting context must use the "
                "FIT_TRANSFORM operation."
            )

    def assert_transform_allowed(
        self,
    ) -> None:
        """Validate transformation access."""

        if self.operation not in {
            TransformationOperation.TRANSFORM,
            TransformationOperation.FIT_TRANSFORM,
        }:
            raise PermissionError(
                "Transformation is not allowed in this context."
            )

        if (
            self.partition == PartitionName.TEST
            and not self.test_partition_authorized
        ):
            raise PermissionError(
                "Test transformation requires authorized access."
            )

    def assert_target_access_prohibited(
        self,
    ) -> None:
        """Ensure target values cannot drive transformations."""

        if self.target_access_allowed:
            raise PermissionError(
                "Target-dependent transformation is prohibited."
            )

    def validate(
        self,
    ) -> None:
        """Validate the complete execution context."""

        self.assert_target_access_prohibited()

        if self.fitting_allowed:
            self.assert_fitting_allowed()

        self.assert_transform_allowed()