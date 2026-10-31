"""Execution-context controls for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)


class SelectionOperation(StrEnum):
    """Supported Phase 7 operations."""

    FIT_UNSUPERVISED = "fit_unsupervised"
    FIT_SUPERVISED = "fit_supervised"
    FIT_AND_SELECT = "fit_and_select"
    TRANSFORM = "transform"


@dataclass(frozen=True, slots=True)
class SelectionFitContext:
    """Execution context for feature selection."""

    partition: PartitionName

    operation: SelectionOperation

    training_data: bool

    unsupervised_fitting_allowed: bool
    supervised_fitting_allowed: bool

    target_access_allowed: bool = False

    test_partition_authorized: bool = False

    @classmethod
    def training_fit(
        cls,
        *,
        supervised_selection_authorized: bool,
    ) -> SelectionFitContext:
        """Create a training fitting context."""

        return cls(
            partition=PartitionName.TRAIN,
            operation=SelectionOperation.FIT_AND_SELECT,
            training_data=True,
            unsupervised_fitting_allowed=True,
            supervised_fitting_allowed=(
                supervised_selection_authorized
            ),
            target_access_allowed=(
                supervised_selection_authorized
            ),
            test_partition_authorized=False,
        )

    @classmethod
    def training_transform(
        cls,
    ) -> SelectionFitContext:
        """Create a training transform-only context."""

        return cls(
            partition=PartitionName.TRAIN,
            operation=SelectionOperation.TRANSFORM,
            training_data=True,
            unsupervised_fitting_allowed=False,
            supervised_fitting_allowed=False,
            target_access_allowed=False,
            test_partition_authorized=False,
        )

    @classmethod
    def validation_transform(
        cls,
    ) -> SelectionFitContext:
        """Create a validation transform-only context."""

        return cls(
            partition=PartitionName.VALIDATION,
            operation=SelectionOperation.TRANSFORM,
            training_data=False,
            unsupervised_fitting_allowed=False,
            supervised_fitting_allowed=False,
            target_access_allowed=False,
            test_partition_authorized=False,
        )

    @classmethod
    def test_transform(
        cls,
        *,
        authorized: bool,
    ) -> SelectionFitContext:
        """Create an authorized test transform context."""

        return cls(
            partition=PartitionName.TEST,
            operation=SelectionOperation.TRANSFORM,
            training_data=False,
            unsupervised_fitting_allowed=False,
            supervised_fitting_allowed=False,
            target_access_allowed=False,
            test_partition_authorized=authorized,
        )

    def assert_unsupervised_fitting_allowed(
        self,
    ) -> None:
        """Raise when unsupervised fitting is prohibited."""

        if not self.unsupervised_fitting_allowed:
            raise PermissionError(
                "Unsupervised feature-filter fitting is "
                "not allowed in this context."
            )

        if not self.training_data:
            raise PermissionError(
                "Unsupervised feature filters may only "
                "fit on training data."
            )

        if self.partition != PartitionName.TRAIN:
            raise PermissionError(
                "Unsupervised feature filters may only "
                "fit on the training partition."
            )

        if self.operation not in {
            SelectionOperation.FIT_UNSUPERVISED,
            SelectionOperation.FIT_AND_SELECT,
        }:
            raise PermissionError(
                "The operation does not permit "
                "unsupervised fitting."
            )

    def assert_supervised_fitting_allowed(
        self,
    ) -> None:
        """Raise when supervised fitting is prohibited."""

        if not self.supervised_fitting_allowed:
            raise PermissionError(
                "Supervised feature selection is not "
                "authorized in this context."
            )

        if not self.target_access_allowed:
            raise PermissionError(
                "Supervised feature selection requires "
                "authorized training-target access."
            )

        if not self.training_data:
            raise PermissionError(
                "Supervised selectors may only fit on "
                "training data."
            )

        if self.partition != PartitionName.TRAIN:
            raise PermissionError(
                "Supervised selectors may only fit on "
                "the training partition."
            )

        if self.operation not in {
            SelectionOperation.FIT_SUPERVISED,
            SelectionOperation.FIT_AND_SELECT,
        }:
            raise PermissionError(
                "The operation does not permit "
                "supervised fitting."
            )

    def assert_transform_allowed(
        self,
    ) -> None:
        """Validate feature-schema transformation access."""

        if self.operation != SelectionOperation.TRANSFORM:
            raise PermissionError(
                "Partition selection requires a "
                "transform-only context."
            )

        if (
            self.partition == PartitionName.TEST
            and not self.test_partition_authorized
        ):
            raise PermissionError(
                "Test feature selection requires "
                "authorized access."
            )

        if self.target_access_allowed:
            raise PermissionError(
                "Target access is prohibited during "
                "partition transformation."
            )

    def validate_for_fit(
        self,
        *,
        supervised_selection_required: bool,
    ) -> None:
        """Validate the complete fitting context."""

        self.assert_unsupervised_fitting_allowed()

        if supervised_selection_required:
            self.assert_supervised_fitting_allowed()

        elif self.target_access_allowed:
            raise PermissionError(
                "Target access is unnecessary because no "
                "supervised selector is enabled."
            )

    def validate_for_transform(
        self,
    ) -> None:
        """Validate the complete transformation context."""

        self.assert_transform_allowed()

        if self.unsupervised_fitting_allowed:
            raise PermissionError(
                "Unsupervised fitting must be disabled "
                "during partition transformation."
            )

        if self.supervised_fitting_allowed:
            raise PermissionError(
                "Supervised fitting must be disabled "
                "during partition transformation."
            )