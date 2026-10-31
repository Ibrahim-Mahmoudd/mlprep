"""Persistent fitted state produced by Phase 4."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase4.output_governance.metadata import (
    Phase4Model,
)
from mlprep.phases.phase4.strategies.base import (
    ImputerState,
)


class Phase4State(Phase4Model):
    """Persistent fitted imputation state produced by Phase 4."""

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
    test_partition_remained_locked: bool = True

    test_access_count_before: int = Field(
        default=0,
        ge=0,
    )

    test_access_count_after: int = Field(
        default=0,
        ge=0,
    )

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

    imputer_states: dict[
        str,
        ImputerState,
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

    configuration_fingerprint: str | None = None
    imputation_plan_fingerprint: str | None = None
    fitted_state_fingerprint: str | None = None
    output_fingerprint: str | None = None

    training_features_unchanged_before_transform: bool = True
    targets_unchanged: bool = False
    row_ids_unchanged: bool = False
    schemas_consistent: bool = False

    remaining_missing_count: int = Field(
        default=0,
        ge=0,
    )

    imputed_value_count: int = Field(
        default=0,
        ge=0,
    )

    generated_indicator_count: int = Field(
        default=0,
        ge=0,
    )

    ready_for_phase_5: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def mark_completed(
        self,
        component: str,
    ) -> None:
        """Mark one Phase 4 component as completed."""

        if not component.strip():
            raise ValueError(
                "component cannot be empty."
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
        """Mark one Phase 4 component as failed."""

        if not component.strip():
            raise ValueError(
                "component cannot be empty."
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
        """Mark one Phase 4 component as skipped."""

        if not component.strip():
            raise ValueError(
                "component cannot be empty."
            )

        if component not in self.skipped_components:
            self.skipped_components.append(
                component
            )

    def component_completed(
        self,
        component: str,
    ) -> bool:
        """Return whether a component completed successfully."""

        return component in self.completed_components

    def register_imputer_state(
        self,
        *,
        column: str,
        state: ImputerState,
    ) -> None:
        """Store fitted state for one feature."""

        if column != state.column:
            raise ValueError(
                "The imputer-state column does not match "
                "the registration key."
            )

        self.imputer_states[column] = (
            state.model_copy(
                deep=True
            )
        )

    def register_partition_fingerprint(
        self,
        *,
        partition: PartitionName,
        input_fingerprint: str | None = None,
        output_fingerprint: str | None = None,
    ) -> None:
        """Register input or output partition fingerprints."""

        if input_fingerprint is not None:
            self.input_partition_fingerprints[
                partition
            ] = input_fingerprint

        if output_fingerprint is not None:
            self.output_partition_fingerprints[
                partition
            ] = output_fingerprint