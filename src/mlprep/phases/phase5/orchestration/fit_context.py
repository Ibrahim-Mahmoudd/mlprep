"""Execution-context controls for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase5.strategies.base import (
    TreatmentExecutionMode,
)


class OutlierOperation(StrEnum):
    """Supported Phase 5 operations."""

    FIT = "fit"
    TRANSFORM = "transform"
    FIT_TRANSFORM = "fit_transform"


@dataclass(frozen=True, slots=True)
class OutlierFitContext:
    """Execution context for outlier detection and treatment."""

    partition: PartitionName

    operation: OutlierOperation

    execution_mode: TreatmentExecutionMode

    training_data: bool

    detector_fitting_allowed: bool
    treatment_fitting_allowed: bool

    target_access_allowed: bool = False

    test_partition_authorized: bool = False

    @classmethod
    def training_fit(
        cls,
    ) -> OutlierFitContext:
        """Create a training fitting context."""

        return cls(
            partition=PartitionName.TRAIN,
            operation=OutlierOperation.FIT_TRANSFORM,
            execution_mode=(
                TreatmentExecutionMode
                .TRAINING_TRANSFORM
            ),
            training_data=True,
            detector_fitting_allowed=True,
            treatment_fitting_allowed=True,
            target_access_allowed=False,
            test_partition_authorized=False,
        )

    @classmethod
    def training_transform(
        cls,
    ) -> OutlierFitContext:
        """Create a training transform-only context."""

        return cls(
            partition=PartitionName.TRAIN,
            operation=OutlierOperation.TRANSFORM,
            execution_mode=(
                TreatmentExecutionMode
                .TRAINING_TRANSFORM
            ),
            training_data=True,
            detector_fitting_allowed=False,
            treatment_fitting_allowed=False,
            target_access_allowed=False,
            test_partition_authorized=False,
        )

    @classmethod
    def validation_transform(
        cls,
    ) -> OutlierFitContext:
        """Create a validation transform-only context."""

        return cls(
            partition=PartitionName.VALIDATION,
            operation=OutlierOperation.TRANSFORM,
            execution_mode=(
                TreatmentExecutionMode
                .VALIDATION_TRANSFORM
            ),
            training_data=False,
            detector_fitting_allowed=False,
            treatment_fitting_allowed=False,
            target_access_allowed=False,
            test_partition_authorized=False,
        )

    @classmethod
    def test_transform(
        cls,
        *,
        authorized: bool,
    ) -> OutlierFitContext:
        """Create an authorized test transform context."""

        return cls(
            partition=PartitionName.TEST,
            operation=OutlierOperation.TRANSFORM,
            execution_mode=(
                TreatmentExecutionMode
                .TEST_TRANSFORM
            ),
            training_data=False,
            detector_fitting_allowed=False,
            treatment_fitting_allowed=False,
            target_access_allowed=False,
            test_partition_authorized=authorized,
        )

    def assert_detector_fitting_allowed(
        self,
    ) -> None:
        """Raise when detector fitting is prohibited."""

        if not self.detector_fitting_allowed:
            raise PermissionError(
                "Detector fitting is not allowed in this context."
            )

        if not self.training_data:
            raise PermissionError(
                "Detector fitting requires training data."
            )

        if self.partition != PartitionName.TRAIN:
            raise PermissionError(
                "Detectors may only be fitted on the "
                "training partition."
            )

    def assert_treatment_fitting_allowed(
        self,
    ) -> None:
        """Raise when treatment fitting is prohibited."""

        if not self.treatment_fitting_allowed:
            raise PermissionError(
                "Treatment fitting is not allowed in this context."
            )

        if not self.training_data:
            raise PermissionError(
                "Treatment fitting requires training data."
            )

        if self.partition != PartitionName.TRAIN:
            raise PermissionError(
                "Treatment strategies may only be fitted on "
                "the training partition."
            )

    def assert_transform_allowed(
        self,
    ) -> None:
        """Validate transformation access."""

        if self.operation not in {
            OutlierOperation.TRANSFORM,
            OutlierOperation.FIT_TRANSFORM,
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
        """Ensure target values cannot drive outlier treatment."""

        if self.target_access_allowed:
            raise PermissionError(
                "Target-dependent outlier treatment is prohibited."
            )

    def validate(
        self,
    ) -> None:
        """Validate the complete execution context."""

        self.assert_target_access_prohibited()

        if self.detector_fitting_allowed:
            self.assert_detector_fitting_allowed()

        if self.treatment_fitting_allowed:
            self.assert_treatment_fitting_allowed()

        if self.operation in {
            OutlierOperation.TRANSFORM,
            OutlierOperation.FIT_TRANSFORM,
        }:
            self.assert_transform_allowed()
