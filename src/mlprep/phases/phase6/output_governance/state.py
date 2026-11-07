"""Persistent fitted state for Phase 6."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase6.categorical.categorical_pipeline import (
    CategoricalColumnPipelineState,
)
from mlprep.phases.phase6.feature_engineering.boolean_features import (
    BooleanFeatureState,
)
from mlprep.phases.phase6.feature_engineering.datetime_features import (
    DatetimeFeatureState,
)
from mlprep.phases.phase6.feature_engineering.feature_lineage import (
    FeatureLineageState,
)
from mlprep.phases.phase6.feature_engineering.text_features import (
    TextFeatureState,
)
from mlprep.phases.phase6.numeric.numeric_pipeline import (
    NumericColumnPipelineState,
)
from mlprep.phases.phase6.output_governance.metadata import (
    Phase6Model,
)


class Phase6State(Phase6Model):
    """Serializable state and governance metadata for Phase 6."""

    fitted: bool = False
    fitted_on_training_only: bool = False
    test_transform_authorized: bool = False
    test_partition_remained_protected: bool = True
    phase5_output_fingerprint: str | None = None

    source_feature_columns: list[str] = Field(default_factory=list)
    output_feature_columns: list[str] = Field(default_factory=list)

    numeric_states: dict[str, NumericColumnPipelineState] = Field(
        default_factory=dict
    )
    categorical_states: dict[str, CategoricalColumnPipelineState] = Field(
        default_factory=dict
    )
    boolean_states: dict[str, BooleanFeatureState] = Field(default_factory=dict)
    datetime_states: dict[str, DatetimeFeatureState] = Field(default_factory=dict)
    text_states: dict[str, TextFeatureState] = Field(default_factory=dict)
    timedelta_output_columns: dict[str, str] = Field(default_factory=dict)
    lineage: FeatureLineageState | None = None

    configuration_fingerprint: str = ""
    transformation_plan_fingerprint: str = ""
    fitted_state_fingerprint: str = ""
    output_fingerprint: str = ""

    input_partition_fingerprints: dict[PartitionName, str] = Field(
        default_factory=dict
    )
    output_partition_fingerprints: dict[PartitionName, str] = Field(
        default_factory=dict
    )
    source_target_fingerprints: dict[PartitionName, str] = Field(
        default_factory=dict
    )
    output_target_fingerprints: dict[PartitionName, str] = Field(
        default_factory=dict
    )
    source_row_id_fingerprints: dict[PartitionName, str] = Field(
        default_factory=dict
    )
    output_row_id_fingerprints: dict[PartitionName, str] = Field(
        default_factory=dict
    )

    targets_unchanged: bool = False
    row_ids_unchanged: bool = False
    schemas_consistent: bool = False
    numeric_integrity_valid: bool = False
    unknown_categories_valid: bool = False
    ready_for_phase_7: bool = False
    total_missing_output_count: int = 0
    total_nonfinite_output_count: int = 0
    total_unknown_category_count: int = 0

    completed_components: list[str] = Field(default_factory=list)
    failed_components: list[str] = Field(default_factory=list)
    skipped_components: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def register_numeric_state(
        self,
        *,
        column: str,
        state: NumericColumnPipelineState,
    ) -> None:
        """Store a fitted numeric transformation state."""

        self.numeric_states[column] = state.model_copy(deep=True)

    def register_categorical_state(
        self,
        *,
        column: str,
        state: CategoricalColumnPipelineState,
    ) -> None:
        """Store a fitted categorical transformation state."""

        self.categorical_states[column] = state.model_copy(deep=True)
