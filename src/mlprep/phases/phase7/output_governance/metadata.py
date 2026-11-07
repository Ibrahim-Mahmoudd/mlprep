"""Shared output-governance metadata for Phase 7."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)


class Phase7Model(BaseModel):
    """Base serializable model for Phase 7."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class Phase7Status(StrEnum):
    """Final execution status of Phase 7."""

    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"
    DISABLED = "disabled"


class Phase7FindingSeverity(StrEnum):
    """Severity of a Phase 7 finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Phase7FindingCategory(StrEnum):
    """Category of a Phase 7 finding."""

    CONTRACT = "contract"
    PLAN = "plan"
    FILTER = "filter"
    SELECTOR = "selector"
    STABILITY = "stability"
    FEATURE_COUNT = "feature_count"
    PROTECTED_FEATURE = "protected_feature"
    SCHEMA = "schema"
    TARGET_INTEGRITY = "target_integrity"
    ROW_INTEGRITY = "row_integrity"
    LINEAGE = "lineage"
    FINGERPRINT = "fingerprint"
    TEST_GOVERNANCE = "test_governance"
    READINESS = "readiness"
    EXECUTION = "execution"


class Phase7FindingAction(StrEnum):
    """Recommended or applied action for a finding."""

    NONE = "none"
    REPORT = "report"
    WARN = "warn"
    RAISE = "raise"
    REVIEW = "review"
    RECONFIGURE = "reconfigure"
    RESELECT = "reselect"
    RESTORE_INPUT = "restore_input"


class FeatureSelectionStatus(StrEnum):
    """Final status of one source feature."""

    SELECTED = "selected"
    REMOVED = "removed"
    REQUIRED = "required"
    PROTECTED = "protected"
    PROHIBITED = "prohibited"
    BLOCKED = "blocked"
    FAILED = "failed"


class Phase7Finding(Phase7Model):
    """One finding produced during Phase 7."""

    finding_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    code: str
    category: Phase7FindingCategory
    severity: Phase7FindingSeverity

    component: str
    message: str

    columns: list[str] = Field(
        default_factory=list,
    )

    partitions: list[str] = Field(
        default_factory=list,
    )

    affected_row_count: int = Field(
        default=0,
        ge=0,
    )

    recommended_action: Phase7FindingAction = (
        Phase7FindingAction.REVIEW
    )

    applied_action: Phase7FindingAction = (
        Phase7FindingAction.REPORT
    )

    blocks_readiness: bool = False

    details: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class FeatureSelectionMetadata(Phase7Model):
    """Execution metadata for one source feature."""

    column: str

    status: FeatureSelectionStatus

    selected: bool = False
    required: bool = False
    protected: bool = False
    prohibited: bool = False

    source_position: int = Field(
        ge=0,
    )

    selected_position: int | None = Field(
        default=None,
        ge=0,
    )

    selection_score: float = 0.0

    selection_reasons: list[str] = Field(
        default_factory=list,
    )

    removal_reasons: list[str] = Field(
        default_factory=list,
    )

    selected_by: list[str] = Field(
        default_factory=list,
    )

    state_fingerprint: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class PartitionSelectionSummary(Phase7Model):
    """Selection summary for one partition."""

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

    selected_feature_count: int = Field(
        default=0,
        ge=0,
    )

    removed_feature_count: int = Field(
        default=0,
        ge=0,
    )

    feature_values_unchanged: bool = True
    target_unchanged: bool = True
    row_ids_unchanged: bool = True
    row_order_unchanged: bool = True

    input_fingerprint: str | None = None
    output_fingerprint: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class Phase7ExecutionMetadata(Phase7Model):
    """Complete execution metadata for Phase 7."""

    execution_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    completed_at: datetime | None = None

    fitted_on_training_only: bool = True

    supervised_target_access_used: bool = False
    supervised_target_access_authorized: bool = False

    validation_used_for_evaluation_only: bool = True

    test_transform_authorized: bool = False
    test_partition_remained_protected: bool = True

    source_feature_count: int = Field(
        default=0,
        ge=0,
    )

    selected_feature_count: int = Field(
        default=0,
        ge=0,
    )

    removed_feature_count: int = Field(
        default=0,
        ge=0,
    )

    features: dict[
        str,
        FeatureSelectionMetadata,
    ] = Field(
        default_factory=dict,
    )

    partitions: dict[
        PartitionName,
        PartitionSelectionSummary,
    ] = Field(
        default_factory=dict,
    )

    configuration_fingerprint: str | None = None
    selection_plan_fingerprint: str | None = None
    fitted_state_fingerprint: str | None = None
    output_fingerprint: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    def finalize(
        self,
    ) -> None:
        """Mark execution metadata as completed."""

        self.completed_at = datetime.now(UTC)
