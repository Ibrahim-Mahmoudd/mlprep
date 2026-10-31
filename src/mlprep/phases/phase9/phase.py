"""Phase 9: govern and produce a model-ready handoff."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd

from mlprep.phases.phase8.output_governance.fingerprint import (
    Phase8FingerprintManager,
)
from mlprep.phases.phase8.output_governance.result import Phase8Result
from mlprep.phases.phase8.output_governance.state import Phase8State
from mlprep.phases.phase9.config import Phase9Config
from mlprep.phases.phase9.contract import Phase9ContractResolver
from mlprep.phases.phase9.output_governance.metadata import (
    Phase9HandoffMetadata,
    Phase9ReadinessDecision,
    Phase9Status,
)
from mlprep.phases.phase9.output_governance.result import Phase9Result


class Phase9:
    """Create an auditable handoff without training a model."""

    phase_name = "phase_9"

    def __init__(self, config: Phase9Config | None = None) -> None:
        self.config = config or Phase9Config()
        if not isinstance(self.config, Phase9Config):
            raise TypeError("config must be a Phase9Config.")

    def run(
        self,
        *,
        phase8_result: Phase8Result,
        phase8_state: Phase8State | None = None,
    ) -> Phase9Result:
        """Validate Phase 8 and return independent, model-ready partitions."""

        if not isinstance(phase8_result, Phase8Result):
            raise TypeError("phase8_result must be a Phase8Result.")

        resolved_state = phase8_state
        if resolved_state is None:
            if not phase8_result.phase_state:
                raise ValueError(
                    "Phase 8 state is unavailable; pass phase8_state or "
                    "enable state storage in Phase 8."
                )
            resolved_state = Phase8State.model_validate(
                phase8_result.phase_state
            )
        elif not isinstance(resolved_state, Phase8State):
            raise TypeError("phase8_state must be a Phase8State.")

        target_column, feature_columns, feature_dtypes = (
            Phase9ContractResolver.resolve(
                phase8_result=phase8_result,
                phase8_state=resolved_state,
                require_disjoint_rows=self.config.require_disjoint_rows,
            )
        )

        source_partitions = (
            phase8_result.train,
            phase8_result.validation,
            phase8_result.test,
        )
        if self.config.copy_partitions:
            train, validation, test = (
                partition.copy(deep=True)
                for partition in source_partitions
            )
        else:
            train, validation, test = source_partitions

        sample_weights = phase8_result.sample_weights
        if sample_weights is not None:
            if not isinstance(sample_weights, pd.Series):
                raise TypeError(
                    "Phase 8 sample weights must be a pandas Series."
                )
            if len(sample_weights) != train.row_count():
                raise ValueError(
                    "Phase 8 sample weights do not align with training rows."
                )
            if not sample_weights.index.equals(train.X.index):
                raise ValueError(
                    "Phase 8 sample-weight indices do not align with "
                    "training features."
                )
            numeric_weights = sample_weights.to_numpy(dtype=float)
            if not np.isfinite(numeric_weights).all() or (
                numeric_weights <= 0.0
            ).any():
                raise ValueError(
                    "Phase 8 sample weights must be finite and positive."
                )

        for class_key, weight in phase8_result.class_weights.items():
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError(
                    f"Phase 8 class weight for {class_key!r} "
                    "must be finite and positive."
                )

        execution = phase8_result.execution_metadata
        metadata = Phase9HandoffMetadata(
            handoff_id=str(uuid4()),
            target_column=target_column,
            feature_columns=feature_columns,
            feature_dtypes=feature_dtypes,
            partition_row_counts={
                "train": train.row_count(),
                "validation": validation.row_count(),
                "test": test.row_count(),
            },
            class_weights=phase8_result.class_weights,
            sample_weights_present=sample_weights is not None,
            phase8_execution_id=(
                execution.execution_id
                if execution is not None
                else None
            ),
            phase8_output_fingerprint=resolved_state.output_fingerprint,
            phase8_fitted_state_fingerprint=(
                resolved_state.fitted_state_fingerprint
            ),
            phase8_phase_state=(
                Phase8FingerprintManager().serialize_model(
                    resolved_state
                )
            ),
        )
        readiness = Phase9ReadinessDecision(
            ready=True,
            status=Phase9Status.PASSED,
        )
        return Phase9Result(
            train=train,
            validation=validation,
            test=test,
            target_column=target_column,
            feature_columns=feature_columns,
            class_weights=phase8_result.class_weights,
            sample_weights=sample_weights,
            phase8_state=resolved_state,
            metadata=metadata,
            readiness=readiness,
        )
