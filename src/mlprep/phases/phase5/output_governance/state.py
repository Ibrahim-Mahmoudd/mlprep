"""Persistent fitted state produced by Phase 5."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase5.output_governance.metadata import (
    Phase5Model,
)
from mlprep.phases.phase5.strategies.base import (
    TreatmentState,
)


class Phase5State(Phase5Model):
    """Persistent fitted outlier-treatment state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    state_version: str = "0.1.0"

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    fitted_on_training_only: bool = True

    test_transform_authorized: bool = False

    test_partition_remained_protected: bool = True

    source_feature_columns: list[str] = Field(
        default_factory=list
    )

    output_feature_columns: list[str] = Field(
        default_factory=list
    )

    indicator_mapping: dict[
        str,
        str,
    ] = Field(default_factory=dict)

    detector_states: dict[
        str,
        dict[str, Any],
    ] = Field(default_factory=dict)

    treatment_states: dict[
        str,
        TreatmentState,
    ] = Field(default_factory=dict)

    completed_components: list[str] = Field(
        default_factory=list
    )

    failed_components: list[str] = Field(
        default_factory=list
    )

    skipped_components: list[str] = Field(
        default_factory=list
    )

    input_partition_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(default_factory=dict)

    output_partition_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(default_factory=dict)

    source_target_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(default_factory=dict)

    output_target_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(default_factory=dict)

    source_row_id_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(default_factory=dict)

    output_row_id_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(default_factory=dict)

    phase4_output_fingerprint: str | None = None

    configuration_fingerprint: str | None = None

    treatment_plan_fingerprint: str | None = None

    fitted_state_fingerprint: str | None = None

    output_fingerprint: str | None = None

    targets_unchanged: bool = False

    row_ids_unchanged: bool = False

    schemas_consistent: bool = False

    total_detected_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    total_changed_value_count: int = Field(
        default=0,
        ge=0,
    )

    total_remaining_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    generated_indicator_count: int = Field(
        default=0,
        ge=0,
    )

    ready_for_phase_6: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def mark_completed(
        self,
        component: str,
    ) -> None:
        """Mark one component as completed."""

        self._validate_component(
            component
        )

        if component not in self.completed_components:
            self.completed_components.append(
                component
            )

        if component in self.failed_components:
            self.failed_components.remove(
                component
            )

        if component in self.skipped_components:
            self.skipped_components.remove(
                component
            )

    def mark_failed(
        self,
        component: str,
    ) -> None:
        """Mark one component as failed."""

        self._validate_component(
            component
        )

        if component not in self.failed_components:
            self.failed_components.append(
                component
            )

        if component in self.completed_components:
            self.completed_components.remove(
                component
            )

    def mark_skipped(
        self,
        component: str,
    ) -> None:
        """Mark one component as skipped."""

        self._validate_component(
            component
        )

        if component not in self.skipped_components:
            self.skipped_components.append(
                component
            )

    def register_treatment_state(
        self,
        *,
        column: str,
        treatment_state: TreatmentState,
    ) -> None:
        """Register fitted treatment state."""

        if column != treatment_state.column:
            raise ValueError(
                "Treatment-state column does not match "
                "the registration key."
            )

        self.treatment_states[column] = (
            treatment_state.model_copy(
                deep=True
            )
        )

    @staticmethod
    def _validate_component(
        component: str,
    ) -> None:
        """Validate a component name."""

        if not component.strip():
            raise ValueError(
                "component cannot be empty."
            )
