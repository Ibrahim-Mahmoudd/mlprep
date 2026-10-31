"""Fit and transformation context controls for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase4.strategies.base import ImputerExecutionMode


class ImputationOperation(StrEnum):
    """Supported Phase 4 operation types."""

    FIT = "fit"
    TRANSFORM = "transform"
    FIT_TRANSFORM = "fit_transform"


@dataclass(frozen=True, slots=True)
class FitContext:
    """Execution context for one imputation operation."""

    partition: PartitionName
    operation: ImputationOperation
    execution_mode: ImputerExecutionMode
    training_data: bool
    fitting_allowed: bool
    target_access_allowed: bool = False
    test_partition_authorized: bool = False

    @classmethod
    def training_fit(cls) -> "FitContext":
        return cls(
            partition=PartitionName.TRAIN,
            operation=ImputationOperation.FIT_TRANSFORM,
            execution_mode=ImputerExecutionMode.TRAINING_TRANSFORM,
            training_data=True,
            fitting_allowed=True,
        )

    @classmethod
    def training_transform(cls) -> "FitContext":
        return cls(
            partition=PartitionName.TRAIN,
            operation=ImputationOperation.TRANSFORM,
            execution_mode=ImputerExecutionMode.TRAINING_TRANSFORM,
            training_data=True,
            fitting_allowed=False,
        )

    @classmethod
    def validation_transform(cls) -> "FitContext":
        return cls(
            partition=PartitionName.VALIDATION,
            operation=ImputationOperation.TRANSFORM,
            execution_mode=ImputerExecutionMode.VALIDATION_TRANSFORM,
            training_data=False,
            fitting_allowed=False,
        )

    @classmethod
    def test_transform(cls, *, authorized: bool) -> "FitContext":
        return cls(
            partition=PartitionName.TEST,
            operation=ImputationOperation.TRANSFORM,
            execution_mode=ImputerExecutionMode.TEST_TRANSFORM,
            training_data=False,
            fitting_allowed=False,
            test_partition_authorized=authorized,
        )

    def validate(self) -> None:
        """Validate the execution context."""
        if self.target_access_allowed:
            raise PermissionError("Target-dependent imputation is prohibited.")
        if self.fitting_allowed:
            if not self.training_data or self.partition != PartitionName.TRAIN:
                raise PermissionError("Fitting is only permitted on training data.")
        if self.operation in {
            ImputationOperation.TRANSFORM,
            ImputationOperation.FIT_TRANSFORM,
        }:
            if (
                self.partition == PartitionName.TEST
                and not self.test_partition_authorized
            ):
                raise PermissionError(
                    "Test transformation requires authorized access."
                )
