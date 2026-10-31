"""Results and audit metadata for cross-validation folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mlprep.cross_validation.config import (
    CrossValidationConfig,
)
from mlprep.cross_validation.fold_context import (
    FoldContext,
)
from mlprep.phases.phase8.samplers.base import (
    RemovedRowRecord,
    SamplerRowRecord,
    SamplerState,
)
from mlprep.phases.phase8.weighting.class_weights import (
    ClassWeightResult,
)


@dataclass(slots=True)
class FoldResult:
    """Transformed train and validation partitions for one fold."""

    context: FoldContext
    training_features: pd.DataFrame
    training_target: pd.Series
    training_row_ids: pd.Series
    validation_features: pd.DataFrame
    validation_target: pd.Series
    validation_row_ids: pd.Series
    class_weights: ClassWeightResult | None
    row_lineage: list[SamplerRowRecord]
    removed_row_lineage: list[RemovedRowRecord]
    sampler_state: SamplerState | None
    fitted_state_fingerprint: str
    training_output_fingerprint: str
    validation_output_fingerprint: str
    transformer_names: tuple[str, ...]

    def to_report_dict(self) -> dict[str, Any]:
        """Return JSON-friendly fold metadata without data values."""

        return {
            "fold_index": self.context.fold_index,
            "training_row_count": len(self.training_features),
            "validation_row_count": len(self.validation_features),
            "training_input_fingerprint": self.context.training_fingerprint,
            "validation_input_fingerprint": self.context.validation_fingerprint,
            "fitted_state_fingerprint": self.fitted_state_fingerprint,
            "training_output_fingerprint": self.training_output_fingerprint,
            "validation_output_fingerprint": self.validation_output_fingerprint,
            "transformers": list(self.transformer_names),
            "sampler": (
                self.sampler_state.sampler
                if self.sampler_state is not None
                else None
            ),
            "lineage_row_count": len(self.row_lineage),
            "removed_row_count": len(self.removed_row_lineage),
        }


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """Ordered results for all generated folds."""

    config: CrossValidationConfig
    folds: tuple[FoldResult, ...]

    def to_report_dict(self) -> dict[str, Any]:
        """Return JSON-friendly metadata for the run."""

        return {
            "strategy": self.config.strategy.value,
            "n_splits": self.config.n_splits,
            "fold_count": len(self.folds),
            "folds": [fold.to_report_dict() for fold in self.folds],
        }
