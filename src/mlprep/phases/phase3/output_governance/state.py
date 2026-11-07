"""Persistent state produced during Phase 3."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)


class Phase3State(BaseModel):
    """Persistent analytical state produced by Phase 3."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    state_version: str = "0.1.0"

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    analyzed_training_only: bool = True
    validation_comparison_enabled: bool = True

    test_partition_access_count_before: int = Field(
        default=0,
        ge=0,
    )

    test_partition_access_count_after: int = Field(
        default=0,
        ge=0,
    )

    test_partition_remained_locked: bool = True

    columns: AnalysisColumnGroups = Field(
        default_factory=AnalysisColumnGroups
    )

    completed_components: list[str] = Field(
        default_factory=list
    )

    failed_components: list[str] = Field(
        default_factory=list
    )

    skipped_components: list[str] = Field(
        default_factory=list
    )

    training_fingerprint_before: str | None = None
    training_fingerprint_after: str | None = None

    validation_fingerprint_before: str | None = None
    validation_fingerprint_after: str | None = None

    phase2_split_fingerprint: str | None = None
    configuration_fingerprint: str | None = None

    analysis_fingerprint: str | None = None

    training_data_unchanged: bool = False
    validation_data_unchanged: bool = False

    recommendation_count: int = Field(
        default=0,
        ge=0,
    )

    generated_plot_count: int = Field(
        default=0,
        ge=0,
    )

    ready_for_phase_4: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def mark_completed(
        self,
        component: str,
    ) -> None:
        """Mark one component as completed."""

        if component not in self.completed_components:
            self.completed_components.append(
                component
            )

    def mark_failed(
        self,
        component: str,
    ) -> None:
        """Mark one component as failed."""

        if component not in self.failed_components:
            self.failed_components.append(
                component
            )

    def mark_skipped(
        self,
        component: str,
    ) -> None:
        """Mark one component as skipped."""

        if component not in self.skipped_components:
            self.skipped_components.append(
                component
            )

    def component_completed(
        self,
        component: str,
    ) -> bool:
        """Return whether a component completed."""

        return component in self.completed_components
