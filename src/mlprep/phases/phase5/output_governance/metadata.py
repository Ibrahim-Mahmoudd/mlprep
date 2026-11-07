"""Shared output-governance metadata for Phase 5."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)


class Phase5Model(BaseModel):
    """Base serializable model for Phase 5."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class Phase5Status(StrEnum):
    """Final execution status of Phase 5."""

    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"
    DISABLED = "disabled"


class Phase5FindingSeverity(StrEnum):
    """Severity of a Phase 5 finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Phase5FindingCategory(StrEnum):
    """Category of a Phase 5 finding."""

    CONTRACT = "contract"
    PLAN = "plan"
    DETECTION = "detection"
    TREATMENT = "treatment"
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


class Phase5FindingAction(StrEnum):
    """Recommended or applied action for a finding."""

    NONE = "none"
    REPORT = "report"
    WARN = "warn"
    RAISE = "raise"
    REVIEW = "review"
    RECONFIGURE = "reconfigure"
    RETREAT = "retreat"
    RESTORE_INPUT = "restore_input"


class DetectorExecutionStatus(StrEnum):
    """Execution status of one detector."""

    PLANNED = "planned"
    FITTED = "fitted"
    APPLIED = "applied"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


class TreatmentExecutionStatus(StrEnum):
    """Execution status of one treatment strategy."""

    PLANNED = "planned"
    FITTED = "fitted"
    APPLIED = "applied"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


class Phase5Finding(Phase5Model):
    """One finding produced during Phase 5."""

    finding_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    code: str
    category: Phase5FindingCategory
    severity: Phase5FindingSeverity

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

    recommended_action: Phase5FindingAction = (
        Phase5FindingAction.REVIEW
    )

    applied_action: Phase5FindingAction = (
        Phase5FindingAction.REPORT
    )

    blocks_readiness: bool = False

    details: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class FeatureTreatmentMetadata(Phase5Model):
    """Metadata describing one feature treatment."""

    column: str

    detector: str
    treatment: str

    detector_status: DetectorExecutionStatus = (
        DetectorExecutionStatus.PLANNED
    )

    treatment_status: TreatmentExecutionStatus = (
        TreatmentExecutionStatus.PLANNED
    )

    fitted_on_training_only: bool = False

    lower_bound: float | None = None
    upper_bound: float | None = None

    add_indicator: bool = False
    indicator_column: str | None = None

    training_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    validation_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    test_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    training_changed_count: int = Field(
        default=0,
        ge=0,
    )

    validation_changed_count: int = Field(
        default=0,
        ge=0,
    )

    test_changed_count: int = Field(
        default=0,
        ge=0,
    )

    detector_state_fingerprint: str | None = None
    treatment_state_fingerprint: str | None = None

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class PartitionTreatmentSummary(Phase5Model):
    """Treatment summary for one partition."""

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

    detected_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    changed_value_count: int = Field(
        default=0,
        ge=0,
    )

    remaining_outlier_count: int = Field(
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

    input_fingerprint: str | None = None
    output_fingerprint: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class Phase5ExecutionMetadata(Phase5Model):
    """Complete execution metadata for Phase 5."""

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

    features: dict[
        str,
        FeatureTreatmentMetadata,
    ] = Field(default_factory=dict)

    partitions: dict[
        PartitionName,
        PartitionTreatmentSummary,
    ] = Field(default_factory=dict)

    configuration_fingerprint: str | None = None
    treatment_plan_fingerprint: str | None = None
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
