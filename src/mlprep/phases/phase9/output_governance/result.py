"""Result and handoff payload produced by Phase 9."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mlprep.phases.phase2.output_governance.result import DatasetPartition
from mlprep.phases.phase8.output_governance.state import Phase8State
from mlprep.phases.phase9.output_governance.metadata import (
    Phase9Finding,
    Phase9HandoffMetadata,
    Phase9ReadinessDecision,
)


class Phase9Result:
    """Model-ready partitions and their governed metadata."""

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        target_column: str,
        feature_columns: list[str],
        class_weights: dict[str, float],
        sample_weights: pd.Series | None,
        phase8_state: Phase8State,
        metadata: Phase9HandoffMetadata,
        readiness: Phase9ReadinessDecision,
        findings: list[Phase9Finding] | None = None,
    ) -> None:
        self.train = train
        self.validation = validation
        self.test = test
        self.target_column = target_column
        self.feature_columns = list(feature_columns)
        self.class_weights = dict(class_weights)
        self.sample_weights = (
            sample_weights.copy(deep=True)
            if sample_weights is not None
            else None
        )
        self.phase8_state = phase8_state.model_copy(deep=True)
        self.metadata = metadata
        self.readiness = readiness
        self.findings = list(findings or [])

    def to_report_dict(self) -> dict[str, Any]:
        """Return JSON-compatible handoff metadata, not data payloads."""

        return {
            "phase": "phase_9",
            "ready": self.readiness.ready,
            "status": self.readiness.status.value,
            "target_column": self.target_column,
            "feature_columns": list(self.feature_columns),
            "partition_row_counts": {
                "train": self.train.row_count(),
                "validation": self.validation.row_count(),
                "test": self.test.row_count(),
            },
            "class_weights": dict(self.class_weights),
            "sample_weight_count": (
                len(self.sample_weights)
                if self.sample_weights is not None
                else 0
            ),
            "metadata": self.metadata.model_dump(mode="json"),
            "readiness": self.readiness.model_dump(mode="json"),
            "findings": [
                finding.model_dump(mode="json")
                for finding in self.findings
            ],
        }
