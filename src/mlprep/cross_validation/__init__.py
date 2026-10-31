"""Leakage-safe cross-validation primitives."""

from mlprep.cross_validation.config import (
    CrossValidationConfig,
    CrossValidationStrategy,
)
from mlprep.cross_validation.fold_context import (
    FoldContext,
)
from mlprep.cross_validation.fold_pipeline import (
    CrossValidationRunner,
    FoldPipeline,
    FoldStep,
    FoldTransformer,
)
from mlprep.cross_validation.fold_result import (
    CrossValidationResult,
    FoldResult,
)
from mlprep.cross_validation.leakage_guard import (
    LeakageGuard,
)

__all__ = [
    "CrossValidationConfig",
    "CrossValidationResult",
    "CrossValidationRunner",
    "CrossValidationStrategy",
    "FoldContext",
    "FoldPipeline",
    "FoldResult",
    "FoldStep",
    "FoldTransformer",
    "LeakageGuard",
]
