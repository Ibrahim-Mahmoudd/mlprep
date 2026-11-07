"""Execution-context controls for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)


class BalancingOperation(StrEnum):
    """Supported Phase 8 operations."""

    ANALYZE = "analyze"
    CALCULATE_WEIGHTS = "calculate_weights"
    RESAMPLE = "resample"
    PROTECT = "protect"


@dataclass(frozen=True, slots=True)
class BalancingFitContext:
    """Execution context for Phase 8 balancing."""

    partition: PartitionName
    operation: BalancingOperation

    training_data: bool

    analysis_allowed: bool = False
    weighting_allowed: bool = False
    resampling_allowed: bool = False

    target_access_allowed: bool = False
    synthetic_generation_allowed: bool = False

    protected_partition: bool = False

    @classmethod
    def training_analysis(
        cls,
        *,
        target_access_authorized: bool,
    ) -> BalancingFitContext:
        """Create a training-analysis context."""

        return cls(
            partition=PartitionName.TRAIN,
            operation=BalancingOperation.ANALYZE,
            training_data=True,
            analysis_allowed=True,
            target_access_allowed=(
                target_access_authorized
            ),
        )

    @classmethod
    def training_weighting(
        cls,
        *,
        target_access_authorized: bool,
    ) -> BalancingFitContext:
        """Create a training-weighting context."""

        return cls(
            partition=PartitionName.TRAIN,
            operation=(
                BalancingOperation.CALCULATE_WEIGHTS
            ),
            training_data=True,
            weighting_allowed=True,
            target_access_allowed=(
                target_access_authorized
            ),
        )

    @classmethod
    def training_resampling(
        cls,
        *,
        resampling_authorized: bool,
        target_access_authorized: bool,
        synthetic_generation_authorized: bool,
    ) -> BalancingFitContext:
        """Create a training-resampling context."""

        return cls(
            partition=PartitionName.TRAIN,
            operation=BalancingOperation.RESAMPLE,
            training_data=True,
            resampling_allowed=(
                resampling_authorized
            ),
            target_access_allowed=(
                target_access_authorized
            ),
            synthetic_generation_allowed=(
                synthetic_generation_authorized
            ),
        )

    @classmethod
    def validation_guard(
        cls,
    ) -> BalancingFitContext:
        """Create a protected validation context."""

        return cls(
            partition=PartitionName.VALIDATION,
            operation=BalancingOperation.PROTECT,
            training_data=False,
            protected_partition=True,
        )

    @classmethod
    def test_guard(
        cls,
    ) -> BalancingFitContext:
        """Create a protected test context."""

        return cls(
            partition=PartitionName.TEST,
            operation=BalancingOperation.PROTECT,
            training_data=False,
            protected_partition=True,
        )

    def assert_analysis_allowed(
        self,
    ) -> None:
        """Validate training-analysis access."""

        if self.operation != BalancingOperation.ANALYZE:
            raise PermissionError(
                "The context does not permit analysis."
            )

        if not self.training_data:
            raise PermissionError(
                "Balancing analysis may only use training data."
            )

        if self.partition != PartitionName.TRAIN:
            raise PermissionError(
                "Balancing analysis may only use the "
                "training partition."
            )

        if not self.analysis_allowed:
            raise PermissionError(
                "Training-distribution analysis is not allowed."
            )

        if not self.target_access_allowed:
            raise PermissionError(
                "Training-distribution analysis requires "
                "authorized target access."
            )

    def assert_weighting_allowed(
        self,
    ) -> None:
        """Validate training-weighting access."""

        if (
            self.operation
            != BalancingOperation.CALCULATE_WEIGHTS
        ):
            raise PermissionError(
                "The context does not permit weight calculation."
            )

        if not self.training_data:
            raise PermissionError(
                "Weights may only be fitted from training data."
            )

        if self.partition != PartitionName.TRAIN:
            raise PermissionError(
                "Weights may only be fitted from the "
                "training partition."
            )

        if not self.weighting_allowed:
            raise PermissionError(
                "Training-weight calculation is not allowed."
            )

        if not self.target_access_allowed:
            raise PermissionError(
                "Training-weight calculation requires "
                "authorized target access."
            )

    def assert_resampling_allowed(
        self,
        *,
        synthetic_required: bool,
    ) -> None:
        """Validate training-resampling access."""

        if self.operation != BalancingOperation.RESAMPLE:
            raise PermissionError(
                "The context does not permit resampling."
            )

        if not self.training_data:
            raise PermissionError(
                "Only training data may be resampled."
            )

        if self.partition != PartitionName.TRAIN:
            raise PermissionError(
                "Only the training partition may be resampled."
            )

        if not self.resampling_allowed:
            raise PermissionError(
                "Training resampling is not authorized."
            )

        if not self.target_access_allowed:
            raise PermissionError(
                "Training resampling requires authorized "
                "target access."
            )

        if (
            synthetic_required
            and not self.synthetic_generation_allowed
        ):
            raise PermissionError(
                "Synthetic-row generation is not authorized."
            )

    def assert_protection_context(
        self,
    ) -> None:
        """Validate a protected partition context."""

        if self.operation != BalancingOperation.PROTECT:
            raise PermissionError(
                "The context is not a partition guard."
            )

        if not self.protected_partition:
            raise PermissionError(
                "The partition is not marked as protected."
            )

        if self.training_data:
            raise PermissionError(
                "A protected validation or test context "
                "cannot represent training data."
            )

        if self.partition not in {
            PartitionName.VALIDATION,
            PartitionName.TEST,
        }:
            raise PermissionError(
                "Only validation and test may use "
                "protected contexts."
            )

        if any(
            (
                self.analysis_allowed,
                self.weighting_allowed,
                self.resampling_allowed,
                self.target_access_allowed,
                self.synthetic_generation_allowed,
            )
        ):
            raise PermissionError(
                "Protected contexts cannot permit fitting, "
                "target access, or synthetic generation."
            )
