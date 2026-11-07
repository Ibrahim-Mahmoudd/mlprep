"""Persistent fitted state produced by Phase 7."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase6.feature_engineering.feature_lineage import (
    FeatureLineageState,
)
from mlprep.phases.phase7.filters.filter_engine import (
    FilterEngineResult,
)
from mlprep.phases.phase7.orchestration.selection_pipeline import (
    FeatureSelectionDecision,
)
from mlprep.phases.phase7.output_governance.metadata import (
    Phase7Model,
)
from mlprep.phases.phase7.selectors.selector_engine import (
    SelectorEngineResult,
)
from mlprep.phases.phase7.stability.stability_engine import (
    StabilityEngineResult,
)


class Phase7State(Phase7Model):
    """Persistent fitted feature-selection state."""

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

    supervised_target_access_used: bool = False
    supervised_target_access_authorized: bool = False

    validation_used_for_evaluation_only: bool = True

    test_transform_authorized: bool = False
    test_partition_remained_protected: bool = True

    source_feature_columns: list[str] = Field(
        default_factory=list,
    )

    selected_feature_columns: list[str] = Field(
        default_factory=list,
    )

    removed_feature_columns: list[str] = Field(
        default_factory=list,
    )

    required_features: list[str] = Field(
        default_factory=list,
    )

    protected_features: list[str] = Field(
        default_factory=list,
    )

    prohibited_features: list[str] = Field(
        default_factory=list,
    )

    decisions: dict[
        str,
        FeatureSelectionDecision,
    ] = Field(
        default_factory=dict,
    )

    filter_result: FilterEngineResult | None = None
    selector_result: SelectorEngineResult | None = None
    stability_result: StabilityEngineResult | None = None

    source_lineage: FeatureLineageState | None = None
    selected_lineage: FeatureLineageState | None = None

    completed_components: list[str] = Field(
        default_factory=list,
    )

    failed_components: list[str] = Field(
        default_factory=list,
    )

    skipped_components: list[str] = Field(
        default_factory=list,
    )

    input_partition_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(
        default_factory=dict,
    )

    output_partition_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(
        default_factory=dict,
    )

    source_target_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(
        default_factory=dict,
    )

    output_target_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(
        default_factory=dict,
    )

    source_row_id_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(
        default_factory=dict,
    )

    output_row_id_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(
        default_factory=dict,
    )

    selected_feature_value_fingerprints: dict[
        PartitionName,
        dict[str, str],
    ] = Field(
        default_factory=dict,
    )

    phase6_output_fingerprint: str | None = None

    configuration_fingerprint: str | None = None
    selection_plan_fingerprint: str | None = None
    fitted_state_fingerprint: str | None = None
    output_fingerprint: str | None = None

    targets_unchanged: bool = False
    row_ids_unchanged: bool = False
    selected_values_unchanged: bool = False

    schemas_consistent: bool = False
    feature_count_valid: bool = False
    protected_features_valid: bool = False
    stability_valid: bool = True

    ready_for_phase_8: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
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

    @staticmethod
    def _validate_component(
        component: str,
    ) -> None:
        """Validate one component name."""

        if not component.strip():
            raise ValueError(
                "component cannot be empty."
            )
