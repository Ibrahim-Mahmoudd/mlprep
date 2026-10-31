"""Categorical transformation components for Phase 6."""

from mlprep.phases.phase6.categorical.categorical_pipeline import (
    CategoricalColumnPipelineState,
    CategoricalPipeline,
    CategoricalPipelineOptions,
    CategoricalPipelineResult,
    UnknownCategorySummary,
)
from mlprep.phases.phase6.categorical.frequency_encoder import (
    FrequencyEncoder,
    FrequencyEncoderOptions,
    FrequencyEncoderState,
)
from mlprep.phases.phase6.categorical.hashing_encoder import (
    HashingEncoder,
    HashingEncoderOptions,
    HashingEncoderState,
)
from mlprep.phases.phase6.categorical.one_hot_encoder import (
    OneHotEncoder,
    OneHotEncoderOptions,
    OneHotEncoderState,
)
from mlprep.phases.phase6.categorical.ordinal_encoder import (
    OrdinalEncoder,
    OrdinalEncoderOptions,
    OrdinalEncoderState,
)
from mlprep.phases.phase6.categorical.rare_category_grouper import (
    RareCategoryGrouper,
    RareCategoryGrouperOptions,
    RareCategoryGrouperState,
)

__all__ = [
    "CategoricalColumnPipelineState",
    "CategoricalPipeline",
    "CategoricalPipelineOptions",
    "CategoricalPipelineResult",
    "FrequencyEncoder",
    "FrequencyEncoderOptions",
    "FrequencyEncoderState",
    "HashingEncoder",
    "HashingEncoderOptions",
    "HashingEncoderState",
    "OneHotEncoder",
    "OneHotEncoderOptions",
    "OneHotEncoderState",
    "OrdinalEncoder",
    "OrdinalEncoderOptions",
    "OrdinalEncoderState",
    "RareCategoryGrouper",
    "RareCategoryGrouperOptions",
    "RareCategoryGrouperState",
    "UnknownCategorySummary",
]