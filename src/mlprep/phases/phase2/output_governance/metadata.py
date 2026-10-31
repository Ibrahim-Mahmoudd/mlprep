"""Metadata models and shared types for Phase 2."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Phase2Model(BaseModel):
    """Base model for serializable Phase 2 objects."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=False,
    )


class PartitionName(StrEnum):
    """Supported dataset partition names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    CALIBRATION = "calibration"
    HOLDOUT = "holdout"


class SplitStrategyName(StrEnum):
    """Supported dataset splitting strategies."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    GROUP = "group"
    GROUP_STRATIFIED = "group_stratified"
    TIME = "time"
    PURGED_TIME = "purged_time"
    REGRESSION_STRATIFIED = "regression_stratified"
    PREDEFINED = "predefined"


class Phase2Status(StrEnum):
    """Final status of Phase 2."""

    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class SplitFindingSeverity(StrEnum):
    """Severity of a Phase 2 finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SplitFindingCategory(StrEnum):
    """Category of a Phase 2 finding."""

    INPUT = "input"
    CONTRACT = "contract"
    STRATEGY = "strategy"
    SPLIT_SIZE = "split_size"
    CLASS_DISTRIBUTION = "class_distribution"
    TARGET_DISTRIBUTION = "target_distribution"
    GROUP_DISTRIBUTION = "group_distribution"
    TEMPORAL_BOUNDARY = "temporal_boundary"
    TARGET_LEAKAGE = "target_leakage"
    ROW_OVERLAP = "row_overlap"
    GROUP_OVERLAP = "group_overlap"
    DUPLICATE_OVERLAP = "duplicate_overlap"
    TEMPORAL_LEAKAGE = "temporal_leakage"
    POST_OUTCOME = "post_outcome"
    PREPROCESSING_LEAKAGE = "preprocessing_leakage"
    PARTITION_INTEGRITY = "partition_integrity"
    TEST_LOCK = "test_lock"
    FINGERPRINT = "fingerprint"
    READINESS = "readiness"
    EXECUTION = "execution"


class SplitFindingAction(StrEnum):
    """Recommended or applied action for a Phase 2 finding."""

    NONE = "none"
    REPORT = "report"
    WARN = "warn"
    RAISE = "raise"
    RESPLIT = "resplit"
    DROP_FEATURE = "drop_feature"
    REMOVE_OVERLAP = "remove_overlap"
    LOCK_TEST_SET = "lock_test_set"


class PartitionMetadata(Phase2Model):
    """Metadata describing one dataset partition."""

    name: PartitionName

    row_count: int = Field(
        default=0,
        ge=0,
    )

    feature_count: int = Field(
        default=0,
        ge=0,
    )

    row_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    target_nonmissing_count: int = Field(
        default=0,
        ge=0,
    )

    target_missing_count: int = Field(
        default=0,
        ge=0,
    )

    unique_group_count: int | None = Field(
        default=None,
        ge=0,
    )

    minimum_timestamp: str | None = None
    maximum_timestamp: str | None = None

    dataset_fingerprint: str | None = None
    feature_fingerprint: str | None = None
    target_fingerprint: str | None = None

    is_locked: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class SplitRatios(Phase2Model):
    """Requested train, validation, and test ratios."""

    train: float = Field(
        default=0.70,
        gt=0.0,
        lt=1.0,
    )

    validation: float = Field(
        default=0.15,
        ge=0.0,
        lt=1.0,
    )

    test: float = Field(
        default=0.15,
        gt=0.0,
        lt=1.0,
    )

    tolerance: float = Field(
        default=1e-8,
        ge=0.0,
    )

    def validate_total(self) -> None:
        """Validate that all requested ratios sum to one."""

        total = (
            self.train
            + self.validation
            + self.test
        )

        if abs(total - 1.0) > self.tolerance:
            raise ValueError(
                "Train, validation, and test ratios must "
                f"sum to 1.0. Received {total:.12f}."
            )


class SplitBoundaries(Phase2Model):
    """Temporal or positional boundaries used by a split."""

    train_start: str | None = None
    train_end: str | None = None

    validation_start: str | None = None
    validation_end: str | None = None

    test_start: str | None = None
    test_end: str | None = None

    purge_gap: str | None = None
    embargo_gap: str | None = None


class SplitExecutionMetadata(Phase2Model):
    """Metadata describing one complete split execution."""

    strategy: SplitStrategyName

    random_seed: int | None = None

    shuffle: bool = False
    stratified: bool = False
    grouped: bool = False
    temporal: bool = False

    requested_ratios: SplitRatios
    actual_ratios: SplitRatios | None = None

    boundaries: SplitBoundaries = Field(
        default_factory=SplitBoundaries
    )

    source_dataset_fingerprint: str | None = None
    split_configuration_fingerprint: str | None = None

    partition_metadata: dict[
        PartitionName,
        PartitionMetadata,
    ] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )
