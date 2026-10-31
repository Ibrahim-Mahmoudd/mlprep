"""Result models produced by Phase 1."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field

from mlprep.types import (
    FindingAction,
    FindingCategory,
    PhaseStatus,
    Severity,
)

if TYPE_CHECKING:
    import pandas as pd


class Phase1ResultModel(BaseModel):
    """Base model for serializable Phase 1 result objects."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=False,
    )


class Finding(Phase1ResultModel):
    """An issue or observation detected during Phase 1."""

    finding_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    code: str
    category: FindingCategory
    severity: Severity

    component: str
    message: str

    columns: list[str] = Field(default_factory=list)

    affected_row_count: int = Field(
        default=0,
        ge=0,
    )

    affected_row_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    example_row_ids: list[str | int] = Field(
        default_factory=list
    )

    example_values: list[Any] = Field(
        default_factory=list
    )

    recommended_action: FindingAction = FindingAction.REPORT
    applied_action: FindingAction = FindingAction.NONE

    blocks_readiness: bool = False

    details: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    def add_example_row_id(
        self,
        row_id: str | int,
        *,
        maximum_examples: int = 5,
    ) -> None:
        """Add one unique example row ID within the limit."""

        if maximum_examples <= 0:
            return

        if row_id in self.example_row_ids:
            return

        if len(self.example_row_ids) >= maximum_examples:
            return

        self.example_row_ids.append(row_id)

    def add_example_value(
        self,
        value: Any,
        *,
        maximum_examples: int = 5,
    ) -> None:
        """Add one example value within the limit."""

        if maximum_examples <= 0:
            return

        if len(self.example_values) >= maximum_examples:
            return

        self.example_values.append(value)


class NormalizationEvent(Phase1ResultModel):
    """A deterministic modification applied during Phase 1."""

    event_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    component: str
    operation: str
    description: str

    column: str | None = None

    affected_row_count: int = Field(
        default=0,
        ge=0,
    )

    before_summary: dict[str, Any] = Field(
        default_factory=dict
    )

    after_summary: dict[str, Any] = Field(
        default_factory=dict
    )

    reversible: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class ColumnInventoryItem(Phase1ResultModel):
    """Summary information describing one dataset column."""

    original_name: str
    normalized_name: str

    roles: list[str] = Field(default_factory=list)

    physical_type: str
    semantic_type: str

    semantic_type_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    required: bool = True
    nullable: bool = True

    use_as_model_feature: bool = True
    available_at_inference: bool = True

    is_sensitive: bool = False
    is_constant: bool = False
    is_near_constant: bool = False
    is_identifier_like: bool = False

    missing_count: int = Field(
        default=0,
        ge=0,
    )

    missing_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    unique_count: int = Field(
        default=0,
        ge=0,
    )

    uniqueness_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    memory_usage_bytes: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class DatasetSummary(Phase1ResultModel):
    """Dataset-level summary generated during Phase 1."""

    row_count_before: int = Field(
        default=0,
        ge=0,
    )

    row_count_after: int = Field(
        default=0,
        ge=0,
    )

    column_count_before: int = Field(
        default=0,
        ge=0,
    )

    column_count_after: int = Field(
        default=0,
        ge=0,
    )

    feature_count: int = Field(
        default=0,
        ge=0,
    )

    identifier_count: int = Field(
        default=0,
        ge=0,
    )

    timestamp_count: int = Field(
        default=0,
        ge=0,
    )

    sensitive_column_count: int = Field(
        default=0,
        ge=0,
    )

    ignored_column_count: int = Field(
        default=0,
        ge=0,
    )

    missing_cell_count: int = Field(
        default=0,
        ge=0,
    )

    duplicate_row_count: int = Field(
        default=0,
        ge=0,
    )

    rejected_row_count: int = Field(
        default=0,
        ge=0,
    )

    memory_usage_before_bytes: int = Field(
        default=0,
        ge=0,
    )

    memory_usage_after_bytes: int = Field(
        default=0,
        ge=0,
    )

    dataset_fingerprint: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @computed_field
    @property
    def removed_row_count(self) -> int:
        """Return the number of rows removed from the dataset."""

        return max(
            0,
            self.row_count_before - self.row_count_after,
        )

    @computed_field
    @property
    def memory_saved_bytes(self) -> int:
        """Return the number of bytes saved."""

        return max(
            0,
            self.memory_usage_before_bytes
            - self.memory_usage_after_bytes,
        )

    @computed_field
    @property
    def memory_reduction_ratio(self) -> float:
        """Return the memory reduction ratio."""

        if self.memory_usage_before_bytes == 0:
            return 0.0

        saved_bytes = (
            self.memory_usage_before_bytes
            - self.memory_usage_after_bytes
        )

        ratio = (
            saved_bytes
            / self.memory_usage_before_bytes
        )

        return max(
            0.0,
            min(1.0, ratio),
        )


class RejectedRecord(Phase1ResultModel):
    """Metadata describing one quarantined record."""

    row_id: str | int

    reason_code: str
    reason: str

    component: str
    severity: Severity

    columns: list[str] = Field(
        default_factory=list
    )

    original_index: str | int | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    rejected_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class ReadinessDecision(Phase1ResultModel):
    """Decision indicating whether Phase 2 may begin."""

    ready: bool
    status: PhaseStatus

    blocking_finding_ids: list[str] = Field(
        default_factory=list
    )

    blocking_finding_codes: list[str] = Field(
        default_factory=list
    )

    reasons: list[str] = Field(
        default_factory=list
    )

    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class Phase1Result:
    """Complete output returned after Phase 1 execution."""

    def __init__(
        self,
        *,
        data: pd.DataFrame,
        internal_row_id_column: str,
    ) -> None:
        """Initialize a new Phase 1 result."""

        self.data = data

        self.internal_row_id_column = (
            internal_row_id_column
        )

        self.findings: list[Finding] = []

        self.normalization_events: list[
            NormalizationEvent
        ] = []

        self.column_inventory: list[
            ColumnInventoryItem
        ] = []

        self.rejected_records: list[
            RejectedRecord
        ] = []

        self.dataset_summary = DatasetSummary(
            row_count_before=len(data),
            row_count_after=len(data),
            column_count_before=len(data.columns),
            column_count_after=len(data.columns),
        )

        self.problem_contract: dict[str, Any] = {}
        self.canonical_schema: dict[str, Any] = {}
        self.phase_state: dict[str, Any] = {}

        self.readiness: ReadinessDecision | None = None

        self.started_at = datetime.now(UTC)
        self.completed_at: datetime | None = None

    def add_finding(
        self,
        finding: Finding,
    ) -> None:
        """Add one finding."""

        self.findings.append(finding)

    def add_findings(
        self,
        findings: list[Finding],
    ) -> None:
        """Add multiple findings."""

        self.findings.extend(findings)

    def add_normalization_event(
        self,
        event: NormalizationEvent,
    ) -> None:
        """Add one normalization event."""

        self.normalization_events.append(event)

    def add_column_inventory_item(
        self,
        item: ColumnInventoryItem,
    ) -> None:
        """Add or replace one column inventory item."""

        for index, existing_item in enumerate(
            self.column_inventory
        ):
            if (
                existing_item.normalized_name
                == item.normalized_name
            ):
                self.column_inventory[index] = item
                return

        self.column_inventory.append(item)

    def reject_record(
        self,
        rejected_record: RejectedRecord,
    ) -> None:
        """Register one quarantined record."""

        self.rejected_records.append(
            rejected_record
        )

    def get_findings_by_severity(
        self,
        severity: Severity,
    ) -> list[object]:
        """Return findings matching one severity level."""

        return [
            finding
            for finding in self.findings
            if finding.severity == severity
        ]

    def get_findings_by_category(
        self,
        category: FindingCategory,
    ) -> list[object]:
        """Return findings matching one category."""

        return [
            finding
            for finding in self.findings
            if finding.category == category
        ]

    def get_blocking_findings(
        self,
    ) -> list[object]:
        """Return findings that block Phase 2."""

        return [
            finding
            for finding in self.findings
            if (
                finding.blocks_readiness
                or finding.severity
                == Severity.CRITICAL
            )
        ]

    def has_errors(self) -> bool:
        """Return whether error or critical findings exist."""

        return any(
            finding.severity
            in {
                Severity.ERROR,
                Severity.CRITICAL,
            }
            for finding in self.findings
        )

    def has_warnings(self) -> bool:
        """Return whether warning findings exist."""

        return any(
            finding.severity == Severity.WARNING
            for finding in self.findings
        )

    def finding_counts_by_severity(
        self,
    ) -> dict[str, int]:
        """Return finding counts grouped by severity."""

        counts = Counter(
            finding.severity.value
            for finding in self.findings
        )

        return {
            severity.value: counts.get(
                severity.value,
                0,
            )
            for severity in Severity
        }

    def finding_counts_by_category(
        self,
    ) -> dict[str, int]:
        """Return finding counts grouped by category."""

        counts = Counter(
            finding.category.value
            for finding in self.findings
        )

        return dict(
            sorted(counts.items())
        )

    def finalize(
        self,
        readiness: ReadinessDecision,
    ) -> None:
        """Finalize the Phase 1 result."""

        self.readiness = readiness
        self.completed_at = datetime.now(UTC)

        self.dataset_summary.row_count_after = len(
            self.data
        )

        self.dataset_summary.column_count_after = len(
            self.data.columns
        )

        self.dataset_summary.rejected_row_count = len(
            self.rejected_records
        )

    def to_report_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the result to a serializable dictionary."""

        readiness_status: str | None = None
        ready_for_phase_2 = False
        readiness_data: dict[str, Any] | None = None

        if self.readiness is not None:
            readiness_status = (
                self.readiness.status.value
            )

            ready_for_phase_2 = (
                self.readiness.ready
            )

            readiness_data = (
                self.readiness.model_dump(
                    mode="json"
                )
            )

        return {
            "phase": "phase_1",
            "status": readiness_status,
            "ready_for_phase_2": ready_for_phase_2,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at is not None
                else None
            ),
            "dataset_summary": (
                self.dataset_summary.model_dump(
                    mode="json"
                )
            ),
            "finding_counts_by_severity": (
                self.finding_counts_by_severity()
            ),
            "finding_counts_by_category": (
                self.finding_counts_by_category()
            ),
            "findings": [
                finding.model_dump(mode="json")
                for finding in self.findings
            ],
            "normalization_events": [
                event.model_dump(mode="json")
                for event
                in self.normalization_events
            ],
            "column_inventory": [
                item.model_dump(mode="json")
                for item in self.column_inventory
            ],
            "rejected_records": [
                record.model_dump(mode="json")
                for record in self.rejected_records
            ],
            "problem_contract": self.problem_contract,
            "canonical_schema": self.canonical_schema,
            "phase_state": self.phase_state,
            "readiness": readiness_data,
        }