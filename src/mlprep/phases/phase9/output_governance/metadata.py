"""Metadata models for the Phase 9 model handoff."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Phase9Model(BaseModel):
    """Base type for serializable Phase 9 metadata."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class Phase9Status(StrEnum):
    """Handoff readiness status."""

    PASSED = "passed"
    FAILED = "failed"


class Phase9FindingSeverity(StrEnum):
    """Severity level for Phase 9 findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Phase9Finding(Phase9Model):
    """A validation finding at the model handoff boundary."""

    code: str
    severity: Phase9FindingSeverity
    message: str
    blocks_readiness: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class Phase9ReadinessDecision(Phase9Model):
    """Decision indicating whether the handoff is ready."""

    ready: bool
    status: Phase9Status
    reasons: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class Phase9HandoffMetadata(Phase9Model):
    """Serializable description of the in-memory handoff payload."""

    handoff_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    target_column: str
    feature_columns: list[str]
    feature_dtypes: dict[str, str]
    partition_row_counts: dict[str, int]
    class_weights: dict[str, float] = Field(default_factory=dict)
    sample_weights_present: bool = False
    phase8_execution_id: str | None = None
    phase8_output_fingerprint: str | None = None
    phase8_fitted_state_fingerprint: str | None = None
    phase8_phase_state: dict[str, Any] = Field(default_factory=dict)
