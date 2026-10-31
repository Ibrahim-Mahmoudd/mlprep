"""Shared result and governance metadata for Phase 8."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import (
    ConfigDict,
    Field,
)

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase8.serialization import (
    Phase8SerializableModel,
)


class Phase8Model(Phase8SerializableModel):
    """Base serializable model for Phase 8."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

class Phase8Status(StrEnum):
    """Final execution status of Phase 8."""

    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"
    DISABLED = "disabled"


class Phase8FindingSeverity(StrEnum):
    """Severity of a Phase 8 finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Phase8FindingCategory(StrEnum):
    """Category of a Phase 8 finding."""

    CONTRACT = "contract"
    PLAN = "plan"
    DISTRIBUTION = "distribution"
    SCHEMA = "schema"
    SYNTHETIC_INTEGRITY = "synthetic_integrity"
    TARGET_INTEGRITY = "target_integrity"
    PARTITION_INTEGRITY = "partition_integrity"
    READINESS = "readiness"
    EXECUTION = "execution"
    FINGERPRINT = "fingerprint"


class Phase8FindingAction(StrEnum):
    """Recommended or applied action for a finding."""

    NONE = "none"
    REPORT = "report"
    REVIEW = "review"
    RECONFIGURE = "reconfigure"
    RESTORE_INPUT = "restore_input"
    RAISE = "raise"


class TrainingRowStatus(StrEnum):
    """Disposition of an output training row."""

    ORIGINAL = "original"
    DUPLICATED = "duplicated"
    SYNTHETIC = "synthetic"
    RETAINED = "retained"


class Phase8Finding(Phase8Model):
    """One finding produced during Phase 8."""

    finding_id: str = Field(default_factory=lambda: str(uuid4()))
    code: str
    category: Phase8FindingCategory
    severity: Phase8FindingSeverity
    component: str
    message: str
    columns: list[str] = Field(default_factory=list)
    classes: list[Any] = Field(default_factory=list)
    partitions: list[str] = Field(default_factory=list)
    affected_row_count: int = Field(default=0, ge=0)
    recommended_action: Phase8FindingAction = Phase8FindingAction.REVIEW
    applied_action: Phase8FindingAction = Phase8FindingAction.REPORT
    blocks_readiness: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TrainingRowMetadata(Phase8Model):
    """Auditable row-level lineage for the output training partition."""

    output_position: int = Field(ge=0)
    output_row_id: str
    status: TrainingRowStatus
    class_value: Any = None
    source_row_ids: list[str] = Field(default_factory=list)
    source_positions: list[int] = Field(default_factory=list)
    synthetic: bool = False
    duplicated: bool = False
    original: bool = True
    sample_weight: float | None = Field(default=None, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PartitionBalancingSummary(Phase8Model):
    """Balancing and protection summary for one partition."""

    partition: PartitionName
    row_count_before: int = Field(default=0, ge=0)
    row_count_after: int = Field(default=0, ge=0)
    feature_count_before: int = Field(default=0, ge=0)
    feature_count_after: int = Field(default=0, ge=0)
    resampled: bool = False
    protected: bool = False
    feature_values_unchanged: bool = True
    target_unchanged: bool = True
    row_ids_unchanged: bool = True
    row_order_unchanged: bool = True
    input_fingerprint: str | None = None
    output_fingerprint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Phase8ExecutionMetadata(Phase8Model):
    """Complete execution metadata for Phase 8."""

    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    fitted_on_training_only: bool = True
    target_access_authorized: bool = False
    resampling_authorized: bool = False
    synthetic_generation_authorized: bool = False
    configured_strategy: str = "none"
    executed_strategy: str = "none"
    source_training_row_count: int = Field(default=0, ge=0)
    output_training_row_count: int = Field(default=0, ge=0)
    original_output_count: int = Field(default=0, ge=0)
    duplicate_output_count: int = Field(default=0, ge=0)
    synthetic_output_count: int = Field(default=0, ge=0)
    removed_original_count: int = Field(default=0, ge=0)
    validation_resampled: bool = False
    test_resampled: bool = False
    validation_remained_protected: bool = True
    test_remained_protected: bool = True
    rows: list[TrainingRowMetadata] = Field(default_factory=list)
    partitions: dict[PartitionName, PartitionBalancingSummary] = Field(
        default_factory=dict
    )
    configuration_fingerprint: str | None = None
    balancing_plan_fingerprint: str | None = None
    fitted_state_fingerprint: str | None = None
    output_fingerprint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def finalize(self) -> None:
        """Mark execution metadata complete."""

        self.completed_at = datetime.now(UTC)
