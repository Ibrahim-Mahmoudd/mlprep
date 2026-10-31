"""Shared output-governance metadata for Phase 4."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)


class Phase4Model(BaseModel):
    """Base serializable model for Phase 4."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class Phase4Status(StrEnum):
    """Final execution status of Phase 4."""

    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"
    DISABLED = "disabled"


class Phase4FindingSeverity(StrEnum):
    """Severity of a Phase 4 finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Phase4FindingCategory(StrEnum):
    """Category of a Phase 4 finding."""

    CONTRACT = "contract"
    PLAN = "plan"
    DETECTION = "detection"
    IMPUTATION = "imputation"
    INDICATOR = "indicator"
    VALIDATION = "validation"
    SCHEMA = "schema"
    TARGET_INTEGRITY = "target_integrity"
    ROW_INTEGRITY = "row_integrity"
    DISTRIBUTION_IMPACT = "distribution_impact"
    FINGERPRINT = "fingerprint"
    TEST_GOVERNANCE = "test_governance"
    READINESS = "readiness"
    EXECUTION = "execution"


class Phase4FindingAction(StrEnum):
    """Recommended or applied action for a finding."""

    NONE = "none"
    REPORT = "report"
    WARN = "warn"
    RAISE = "raise"
    REVIEW = "review"
    RECONFIGURE = "reconfigure"
    REIMPUTE = "reimpute"
    RESTORE_INPUT = "restore_input"


class ImputationExecutionStatus(StrEnum):
    """Execution status of one feature imputation."""

    PLANNED = "planned"
    FITTED = "fitted"
    TRANSFORMED = "transformed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


class ImputationMethodMetadata(Phase4Model):
    """Metadata describing one fitted feature imputer."""

    column: str

    feature_family: str
    method: str

    execution_status: ImputationExecutionStatus = (
        ImputationExecutionStatus.PLANNED
    )

    requires_training_fit: bool = True
    fitted_on_training_only: bool = False

    add_missing_indicator: bool = False
    indicator_column: str | None = None

    training_missing_count: int = Field(
        default=0,
        ge=0,
    )

    training_imputed_count: int = Field(
        default=0,
        ge=0,
    )

    validation_imputed_count: int = Field(
        default=0,
        ge=0,
    )

    test_imputed_count: int = Field(
        default=0,
        ge=0,
    )

    input_dtype: str | None = None
    output_dtype: str | None = None

    state_fingerprint: str | None = None

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class PartitionImputationMetadata(Phase4Model):
    """Summary of imputation for one partition."""

    partition: PartitionName

    row_count: int = Field(
        default=0,
        ge=0,
    )

    feature_count_before: int = Field(
        default=0,
        ge=0,
    )

    feature_count_after: int = Field(
        default=0,
        ge=0,
    )

    missing_count_before: int = Field(
        default=0,
        ge=0,
    )

    missing_count_after: int = Field(
        default=0,
        ge=0,
    )

    imputed_value_count: int = Field(
        default=0,
        ge=0,
    )

    invalid_value_count: int = Field(
        default=0,
        ge=0,
    )

    missing_token_count: int = Field(
        default=0,
        ge=0,
    )

    indicator_count: int = Field(
        default=0,
        ge=0,
    )

    indicator_columns: list[str] = Field(
        default_factory=list
    )

    dropped_columns: list[str] = Field(
        default_factory=list
    )

    input_fingerprint: str | None = None
    output_fingerprint: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class Phase4ExecutionMetadata(Phase4Model):
    """Complete execution metadata for Phase 4."""

    execution_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    completed_at: datetime | None = None

    fitted_on_training_only: bool = True
    test_transform_authorized: bool = False
    test_partition_remained_locked: bool = True

    feature_methods: dict[
        str,
        ImputationMethodMetadata,
    ] = Field(default_factory=dict)

    partitions: dict[
        PartitionName,
        PartitionImputationMetadata,
    ] = Field(default_factory=dict)

    configuration_fingerprint: str | None = None
    fitted_state_fingerprint: str | None = None
    output_fingerprint: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def finalize(
        self,
    ) -> None:
        """Mark execution metadata as completed."""

        self.completed_at = datetime.now(UTC)