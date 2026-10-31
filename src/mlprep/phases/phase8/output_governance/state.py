"""Persistent fitted state produced by Phase 8."""

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
from mlprep.phases.phase8.analysis import (
    ClassDistributionResult,
    ImbalanceMetricsResult,
    StrategyRecommendationResult,
)
from mlprep.phases.phase8.lineage import (
    RowLineageState,
    SyntheticIdGeneratorState,
)
from mlprep.phases.phase8.output_governance.metadata import (
    Phase8Model,
)
from mlprep.phases.phase8.samplers import (
    SamplerState,
)
from mlprep.phases.phase8.utils.class_identity import (
    ClassIdentityMapping,
)
from mlprep.phases.phase8.weighting import (
    ClassWeightResult,
    SampleWeightResult,
)


class Phase8State(Phase8Model):
    """Persistent fitted balancing state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    state_version: str = "0.1.0"

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            UTC
        ),
    )

    fitted_on_training_only: bool = True

    target_access_authorized: bool = False
    resampling_authorized: bool = False
    synthetic_generation_authorized: bool = False

    configured_strategy: str = "none"
    executed_strategy: str = "none"

    source_training_row_count: int = Field(
        default=0,
        ge=0,
    )

    output_training_row_count: int = Field(
        default=0,
        ge=0,
    )

    source_feature_columns: list[str] = Field(
        default_factory=list,
    )

    source_feature_dtypes: dict[
        str,
        str,
    ] = Field(
        default_factory=dict,
    )

    source_class_distribution: (
        ClassDistributionResult | None
    ) = None

    class_identity_mapping: (
        ClassIdentityMapping | None
    ) = None

    output_class_distribution: (
        ClassDistributionResult | None
    ) = None

    imbalance_metrics_before: (
        ImbalanceMetricsResult | None
    ) = None

    imbalance_metrics_after: (
        ImbalanceMetricsResult | None
    ) = None

    strategy_recommendation: (
        StrategyRecommendationResult | None
    ) = None

    sampler_state: SamplerState | None = None

    class_weights: ClassWeightResult | None = None
    sample_weights: SampleWeightResult | None = None

    row_lineage: RowLineageState | None = None

    synthetic_id_state: (
        SyntheticIdGeneratorState | None
    ) = None

    retained_original_row_ids: list[str] = Field(
        default_factory=list,
    )

    duplicate_row_ids: list[str] = Field(
        default_factory=list,
    )

    synthetic_row_ids: list[str] = Field(
        default_factory=list,
    )

    removed_original_row_ids: list[str] = Field(
        default_factory=list,
    )

    selected_feature_lineage: (
        FeatureLineageState | None
    ) = None

    validation_resampled: bool = False
    test_resampled: bool = False

    validation_remained_protected: bool = True
    test_remained_protected: bool = True

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

    protected_input_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(
        default_factory=dict,
    )

    protected_output_fingerprints: dict[
        PartitionName,
        str,
    ] = Field(
        default_factory=dict,
    )

    phase7_output_fingerprint: str | None = None

    configuration_fingerprint: str | None = None
    balancing_plan_fingerprint: str | None = None
    fitted_state_fingerprint: str | None = None
    output_fingerprint: str | None = None

    distribution_valid: bool = False
    schema_consistent: bool = False
    synthetic_integrity_valid: bool = False
    target_integrity_valid: bool = False
    partition_integrity_valid: bool = False

    class_weights_valid: bool = True
    sample_weights_valid: bool = True
    lineage_complete: bool = False

    ready_for_phase_9: bool = False

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
