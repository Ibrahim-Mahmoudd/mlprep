"""Shared output-governance metadata for Phase 6."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)


class Phase6Model(BaseModel):
    """Base serializable model for Phase 6."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class Phase6Status(StrEnum):
    """Final execution status of Phase 6."""

    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"
    DISABLED = "disabled"


class Phase6FindingSeverity(StrEnum):
    """Severity of a Phase 6 finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Phase6FindingCategory(StrEnum):
    """Category of a Phase 6 finding."""

    CONTRACT = "contract"
    PLAN = "plan"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    FEATURE_ENGINEERING = "feature_engineering"
    UNKNOWN_CATEGORY = "unknown_category"
    VALIDATION = "validation"
    SCHEMA = "schema"
    TARGET_INTEGRITY = "target_integrity"
    ROW_INTEGRITY = "row_integrity"
    FINGERPRINT = "fingerprint"
    TEST_GOVERNANCE = "test_governance"
    READINESS = "readiness"
    EXECUTION = "execution"


class Phase6FindingAction(StrEnum):
    """Recommended or applied action for a finding."""

    NONE = "none"
    REPORT = "report"
    WARN = "warn"
    RAISE = "raise"
    REVIEW = "review"
    RECONFIGURE = "reconfigure"
    RETRANSFORM = "retransform"
    RESTORE_INPUT = "restore_input"


class TransformationExecutionStatus(StrEnum):
    """Execution status of one feature transformation."""

    PLANNED = "planned"
    FITTED = "fitted"
    TRANSFORMED = "transformed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


class Phase6Finding(Phase6Model):
    """One finding produced during Phase 6."""

    finding_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    code: str
    category: Phase6FindingCategory
    severity: Phase6FindingSeverity

    component: str
    message: str

    columns: list[str] = Field(
        default_factory=list
    )

    partitions: list[str] = Field(
        default_factory=list
    )

    affected_row_count: int = Field(
        default=0,
        ge=0,
    )

    recommended_action: Phase6FindingAction = (
        Phase6FindingAction.REVIEW
    )

    applied_action: Phase6FindingAction = (
        Phase6FindingAction.REPORT
    )

    blocks_readiness: bool = False

    details: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class FeatureTransformationMetadata(Phase6Model):
    """Execution metadata for one source feature."""

    source_column: str
    feature_family: str

    status: TransformationExecutionStatus = (
        TransformationExecutionStatus.PLANNED
    )

    transformation_steps: list[str] = Field(
        default_factory=list
    )

    output_columns: list[str] = Field(
        default_factory=list
    )

    source_preserved: bool = False

    requires_training_fit: bool = True
    fitted_on_training_only: bool = False

    training_unknown_count: int = Field(
        default=0,
        ge=0,
    )

    validation_unknown_count: int = Field(
        default=0,
        ge=0,
    )

    test_unknown_count: int = Field(
        default=0,
        ge=0,
    )

    state_fingerprint: str | None = None

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class PartitionTransformationSummary(Phase6Model):
    """Transformation summary for one partition."""

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

    generated_feature_count: int = Field(
        default=0,
        ge=0,
    )

    preserved_source_count: int = Field(
        default=0,
        ge=0,
    )

    dropped_source_count: int = Field(
        default=0,
        ge=0,
    )

    missing_count_after: int = Field(
        default=0,
        ge=0,
    )

    nonfinite_count_after: int = Field(
        default=0,
        ge=0,
    )

    unknown_category_count: int = Field(
        default=0,
        ge=0,
    )

    input_fingerprint: str | None = None
    output_fingerprint: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class Phase6ExecutionMetadata(Phase6Model):
    """Complete execution metadata for Phase 6."""

    execution_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    completed_at: datetime | None = None

    fitted_on_training_only: bool = True

    test_transform_authorized: bool = False
    test_partition_remained_protected: bool = True

    source_feature_count: int = Field(
        default=0,
        ge=0,
    )

    output_feature_count: int = Field(
        default=0,
        ge=0,
    )

    features: dict[
        str,
        FeatureTransformationMetadata,
    ] = Field(default_factory=dict)

    partitions: dict[
        PartitionName,
        PartitionTransformationSummary,
    ] = Field(default_factory=dict)

    configuration_fingerprint: str | None = None
    transformation_plan_fingerprint: str | None = None
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
