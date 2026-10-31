"""Base training-only samplers for Phase 8."""

from __future__ import annotations

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase8.utils.class_identity import (
    ClassIdentityMapping,
    build_class_identity_mapping,
    normalize_class_value,
)


class SamplerExecutionContext(StrEnum):
    """Supported sampler execution contexts."""

    TRAINING_RESAMPLING = "training_resampling"
    TRAINING_ANALYSIS = "training_analysis"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"


class ResampledRowType(StrEnum):
    """Supported output training-row types."""

    ORIGINAL = "original"
    DUPLICATE = "duplicate"
    SYNTHETIC = "synthetic"
    RETAINED_AFTER_CLEANING = "retained_after_cleaning"


class SamplerRowRecord(BaseModel):
    """Lineage information for one resampled row."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    output_position: int = Field(
        ge=0,
    )

    output_row_id: str

    row_type: ResampledRowType

    class_value: Any

    source_row_ids: list[str] = Field(
        default_factory=list,
    )

    source_positions: list[int] = Field(
        default_factory=list,
    )

    synthetic: bool = False
    duplicated: bool = False
    original: bool = False

    generation_sequence: int | None = Field(
        default=None,
        ge=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class RemovedRowRecord(BaseModel):
    """Lineage information for one removed original row."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    row_id: str
    source_position: int = Field(
        ge=0,
    )

    class_value: Any

    reason: str
    stage: str

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SamplerState(BaseModel):
    """Serializable fitted sampler state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    sampler: str
    strategy: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    resampling_authorized: bool = False
    target_access_authorized: bool = False
    synthetic_generation_authorized: bool = False

    source_feature_columns: list[str] = Field(
        default_factory=list,
    )

    source_feature_dtypes: dict[str, str] = Field(
        default_factory=dict,
    )

    source_row_count: int = Field(
        default=0,
        ge=0,
    )

    output_row_count: int = Field(
        default=0,
        ge=0,
    )

    class_counts_before: dict[str, int] = Field(
        default_factory=dict,
    )

    class_counts_after: dict[str, int] = Field(
        default_factory=dict,
    )

    original_output_count: int = Field(
        default=0,
        ge=0,
    )

    duplicate_output_count: int = Field(
        default=0,
        ge=0,
    )

    synthetic_output_count: int = Field(
        default=0,
        ge=0,
    )

    removed_original_count: int = Field(
        default=0,
        ge=0,
    )

    random_seed: int = 42

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SamplerResult(BaseModel):
    """Complete output from one Phase 8 sampler."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    features: pd.DataFrame
    target: pd.Series
    row_ids: pd.Series

    row_lineage: list[SamplerRowRecord] = Field(
        default_factory=list,
    )

    removed_rows: list[RemovedRowRecord] = Field(
        default_factory=list,
    )

    state: SamplerState

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class BaseTrainingSampler(ABC):
    """Common interface for training-only Phase 8 samplers."""

    sampler_name = "base_sampler"
    strategy_name = "none"

    def __init__(
        self,
        *,
        sampling_strategy: str | float | dict[Any, int] = "auto",
        random_seed: int = 42,
        synthetic_prefix: str = "phase8__synthetic",
        duplicate_prefix: str = "phase8__duplicate",
        include_strategy_in_identifier: bool = True,
        deterministic_row_ids: bool = True,
        preserve_feature_dtypes: bool = True,
        require_finite_values: bool = True,
        allow_missing_values: bool = False,
    ) -> None:
        """Initialize the training sampler."""

        if not synthetic_prefix.strip():
            raise ValueError(
                "synthetic_prefix cannot be empty."
            )

        if not duplicate_prefix.strip():
            raise ValueError(
                "duplicate_prefix cannot be empty."
            )

        if synthetic_prefix == duplicate_prefix:
            raise ValueError(
                "Synthetic and duplicate prefixes must differ."
            )

        self.sampling_strategy = sampling_strategy
        self.random_seed = random_seed

        self.synthetic_prefix = synthetic_prefix
        self.duplicate_prefix = duplicate_prefix

        self.include_strategy_in_identifier = (
            include_strategy_in_identifier
        )
        self.deterministic_row_ids = deterministic_row_ids
        self._id_generator: Any = None

        self.preserve_feature_dtypes = (
            preserve_feature_dtypes
        )

        self.require_finite_values = (
            require_finite_values
        )

        self.allow_missing_values = (
            allow_missing_values
        )

        self.state = SamplerState(
            sampler=self.sampler_name,
            strategy=self.strategy_name,
            random_seed=random_seed,
        )

        self.result: SamplerResult | None = None

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the sampler has executed."""

        return self.state.fitted

    def fit_resample(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
        context: SamplerExecutionContext,
        resampling_authorized: bool,
        target_access_authorized: bool,
        synthetic_generation_authorized: bool = False,
    ) -> SamplerResult:
        """Resample authorized training data."""

        self._validate_inputs(
            features=features,
            target=target,
            row_ids=row_ids,
        )
        identity_mapping = build_class_identity_mapping(target)

        self._validate_context(
            context=context,
            resampling_authorized=resampling_authorized,
            target_access_authorized=(
                target_access_authorized
            ),
            synthetic_generation_authorized=(
                synthetic_generation_authorized
            ),
        )

        if self.is_fitted:
            raise RuntimeError(
                f"Sampler {self.sampler_name!r} "
                "has already executed."
            )

        if len(identity_mapping.class_order) < 2:
            raise ValueError(
                "Training targets must contain at least "
                "two distinct classes."
            )

        original_sampling_strategy = self.sampling_strategy
        if isinstance(self.sampling_strategy, dict):
            self.sampling_strategy = self._resolve_sampling_strategy(
                self.sampling_strategy,
                identity_mapping=identity_mapping,
            )

        source_columns = [
            str(column)
            for column in features.columns
        ]

        source_dtypes = {
            str(column): str(
                features[column].dtype
            )
            for column in features.columns
        }

        class_counts_before = self._class_counts(
            target,
            identity_mapping=identity_mapping,
        )

        encoded_target = pd.Series(
            [
                identity_mapping.key_for(value)
                for value in target
            ],
            index=target.index.copy(),
            name=target.name,
            dtype="object",
        )
        from mlprep.phases.phase8.lineage.synthetic_ids import (
            SyntheticIdGenerator,
            SyntheticIdGeneratorOptions,
        )

        stable_execution_id = self._row_id_execution_id(
            features=features,
            target=encoded_target,
            row_ids=row_ids,
        )
        self._id_generator = SyntheticIdGenerator(
            execution_id=(
                stable_execution_id
                if self.deterministic_row_ids
                else str(uuid.uuid4())
            ),
            random_seed=self.random_seed,
            existing_row_ids=[
                str(value)
                for value in row_ids.tolist()
            ],
            options=SyntheticIdGeneratorOptions(
                synthetic_prefix=self.synthetic_prefix,
                duplicate_prefix=self.duplicate_prefix,
                include_strategy=(
                    self.include_strategy_in_identifier
                ),
                include_execution_token=True,
                reject_prefix_collision=False,
            ),
            deterministic=self.deterministic_row_ids,
        )

        try:
            result = self._resample(
                features=features.copy(
                    deep=True
                ),
                target=encoded_target,
                row_ids=row_ids.copy(
                    deep=True
                ),
            )
        finally:
            self.sampling_strategy = original_sampling_strategy

        decoded_target = [
            identity_mapping.value_for(str(value))
            for value in result.target
        ]
        try:
            result.target = pd.Series(
                decoded_target,
                index=result.target.index.copy(),
                name=result.target.name,
                dtype=target.dtype,
            )
        except (TypeError, ValueError):
            result.target = pd.Series(
                decoded_target,
                index=result.target.index.copy(),
                name=result.target.name,
                dtype="object",
            )

        result.row_lineage = [
            record.model_copy(
                update={
                    "class_value": identity_mapping.value_for(
                        str(record.class_value)
                    )
                }
            )
            for record in result.row_lineage
        ]
        result.removed_rows = [
            record.model_copy(
                update={
                    "class_value": identity_mapping.value_for(
                        str(record.class_value)
                    )
                }
            )
            for record in result.removed_rows
        ]

        self._validate_result(
            result=result,
            source_columns=source_columns,
        )

        if self.preserve_feature_dtypes:
            result.features = self._restore_dtypes(
                frame=result.features,
                source_dtypes=source_dtypes,
            )

        self.state.source_feature_columns = (
            source_columns
        )

        self.state.source_feature_dtypes = (
            source_dtypes
        )

        self.state.source_row_count = len(
            features
        )

        self.state.output_row_count = len(
            result.features
        )

        self.state.class_counts_before = (
            class_counts_before
        )

        self.state.class_counts_after = (
            self._class_counts(
                result.target,
                identity_mapping=identity_mapping,
            )
        )

        self.state.original_output_count = sum(
            record.row_type
            in {
                ResampledRowType.ORIGINAL,
                ResampledRowType.RETAINED_AFTER_CLEANING,
            }
            for record in result.row_lineage
        )

        self.state.duplicate_output_count = sum(
            record.row_type
            == ResampledRowType.DUPLICATE
            for record in result.row_lineage
        )

        self.state.synthetic_output_count = sum(
            record.row_type
            == ResampledRowType.SYNTHETIC
            for record in result.row_lineage
        )

        self.state.removed_original_count = len(
            result.removed_rows
        )
        self.state.metadata["synthetic_id_state"] = (
            self._id_generator.get_state().model_dump(
                mode="json"
            )
        )

        self.state.resampling_authorized = (
            resampling_authorized
        )

        self.state.target_access_authorized = (
            target_access_authorized
        )

        self.state.synthetic_generation_authorized = (
            synthetic_generation_authorized
        )

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        result.state = self.state.model_copy(
            deep=True
        )

        self.result = result

        return result.model_copy(
            deep=True
        )

    def get_result(
        self,
    ) -> SamplerResult:
        """Return the latest resampling output."""

        if self.result is None:
            raise RuntimeError(
                "The sampler has not executed."
            )

        return self.result.model_copy(
            deep=True
        )

    def get_state(
        self,
    ) -> SamplerState:
        """Return fitted sampler state."""

        if not self.is_fitted:
            raise RuntimeError(
                "The sampler has not executed."
            )

        return self.state.model_copy(
            deep=True
        )

    @abstractmethod
    def _resample(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
    ) -> SamplerResult:
        """Execute sampler-specific resampling."""

    def _validate_context(
        self,
        *,
        context: SamplerExecutionContext | str,
        resampling_authorized: bool,
        target_access_authorized: bool,
        synthetic_generation_authorized: bool,
    ) -> None:
        """Validate training-only resampling permissions."""

        if not isinstance(
            context,
            SamplerExecutionContext,
        ):
            context = SamplerExecutionContext(
                context
            )

        if (
            context
            != SamplerExecutionContext.TRAINING_RESAMPLING
        ):
            raise PermissionError(
                "Samplers may only execute in the "
                "training-resampling context."
            )

        if not resampling_authorized:
            raise PermissionError(
                "Training resampling is not authorized."
            )

        if not target_access_authorized:
            raise PermissionError(
                "Training-target access is required "
                "for resampling."
            )

        if (
            self.requires_synthetic_authorization()
            and not synthetic_generation_authorized
        ):
            raise PermissionError(
                "Synthetic-row generation is not authorized."
            )

    def requires_synthetic_authorization(
        self,
    ) -> bool:
        """Return whether this sampler creates synthetic rows."""

        return False

    def _validate_inputs(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
    ) -> None:
        """Validate sampler inputs."""

        if not isinstance(
            features,
            pd.DataFrame,
        ):
            raise TypeError(
                "features must be a pandas DataFrame."
            )

        if not isinstance(
            target,
            pd.Series,
        ):
            raise TypeError(
                "target must be a pandas Series."
            )

        if not isinstance(
            row_ids,
            pd.Series,
        ):
            raise TypeError(
                "row_ids must be a pandas Series."
            )

        if features.empty:
            raise ValueError(
                "Training features cannot be empty."
            )

        if not (
            len(features)
            == len(target)
            == len(row_ids)
        ):
            raise ValueError(
                "Features, target, and row IDs must have "
                "identical row counts."
            )

        if (
            not features.index.equals(
                target.index
            )
            or not features.index.equals(
                row_ids.index
            )
        ):
            raise ValueError(
                "Features, target, and row IDs must have "
                "identical indices."
            )

        if features.columns.duplicated().any():
            raise ValueError(
                "Training feature names must be unique."
            )

        if target.isna().any():
            raise ValueError(
                "Training targets cannot contain missing values."
            )

        if row_ids.isna().any():
            raise ValueError(
                "Training row IDs cannot contain missing values."
            )

        if not row_ids.is_unique:
            raise ValueError(
                "Training row IDs must be unique."
            )

        nonnumeric = [
            str(column)
            for column in features.columns
            if not pd.api.types.is_numeric_dtype(
                features[
                    column
                ].dtype
            )
        ]

        if nonnumeric:
            raise TypeError(
                "Training samplers require numeric features: "
                f"{nonnumeric}"
            )

        if (
            not self.allow_missing_values
            and features.isna().any().any()
        ):
            raise ValueError(
                "Training samplers require complete features."
            )

        if self.require_finite_values:
            invalid_columns = [
                str(column)
                for column in features.columns
                if not np.isfinite(
                    pd.to_numeric(
                        features[
                            column
                        ],
                        errors="coerce",
                    ).astype(float)
                ).all()
            ]

            if invalid_columns:
                raise ValueError(
                    "Training samplers require finite values: "
                    f"{invalid_columns}"
                )

    @staticmethod
    def _validate_result(
        *,
        result: SamplerResult,
        source_columns: list[str],
    ) -> None:
        """Validate one sampler result."""

        if result.features.feature_names if False else False:
            pass

        output_columns = [
            str(column)
            for column in result.features.columns
        ]

        if output_columns != source_columns:
            raise ValueError(
                "Resampling changed the feature schema."
            )

        if not (
            len(result.features)
            == len(result.target)
            == len(result.row_ids)
            == len(result.row_lineage)
        ):
            raise ValueError(
                "Resampled features, targets, row IDs, and "
                "lineage have different row counts."
            )

        if result.row_ids.isna().any():
            raise ValueError(
                "Resampled row IDs contain missing values."
            )

        if not result.row_ids.is_unique:
            raise ValueError(
                "Resampled row IDs are not unique."
            )

        expected_index = pd.RangeIndex(
            len(
                result.features
            )
        )

        if not result.features.index.equals(
            expected_index
        ):
            raise ValueError(
                "Resampled feature index must be a RangeIndex."
            )

        if not result.target.index.equals(
            expected_index
        ):
            raise ValueError(
                "Resampled target index must match features."
            )

        if not result.row_ids.index.equals(
            expected_index
        ):
            raise ValueError(
                "Resampled row-ID index must match features."
            )

    def _new_row_id(
        self,
        *,
        row_type: ResampledRowType,
        sequence: int,
        existing_ids: set[str],
    ) -> str:
        """Generate one reproducible, collision-safe row ID."""

        if self._id_generator is None:
            raise RuntimeError(
                "The row-ID generator is unavailable before "
                "sampler execution."
            )

        generated_state = self._id_generator.get_state()
        expected_sequence = (
            generated_state.synthetic_counter + 1
            if row_type == ResampledRowType.SYNTHETIC
            else generated_state.duplicate_counter + 1
        )
        if sequence != expected_sequence:
            raise ValueError(
                "Sampler row-ID sequence is inconsistent with "
                "the generator state."
            )

        if row_type == ResampledRowType.SYNTHETIC:
            candidate = self._id_generator.next_synthetic_id(
                strategy=self.strategy_name
            )
        else:
            candidate = self._id_generator.next_duplicate_id(
                strategy=self.strategy_name
            )

        if candidate in existing_ids:
            raise RuntimeError(
                "The row-ID generator returned a duplicate ID."
            )

        existing_ids.add(candidate)
        return candidate

    def _row_id_execution_id(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
    ) -> str:
        """Build a stable ID namespace from the sampler inputs and configuration."""

        digest = hashlib.sha256()
        config = {
            "sampler": self.sampler_name,
            "strategy": self.strategy_name,
            "random_seed": self.random_seed,
            "sampling_strategy": self.sampling_strategy,
            "row_ids": [str(value) for value in row_ids.tolist()],
            "target_keys": [str(value) for value in target.tolist()],
            "columns": [str(column) for column in features.columns],
        }
        digest.update(
            json.dumps(
                config,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(
            pd.util.hash_pandas_object(
                features,
                index=True,
            ).values.tobytes()
        )
        return digest.hexdigest()

    @staticmethod
    def _class_counts(
        target: pd.Series,
        *,
        identity_mapping: ClassIdentityMapping,
    ) -> dict[str, int]:
        """Return class counts keyed by canonical typed identity."""

        counts: dict[str, int] = {}
        for value in target:
            normalized = normalize_class_value(value)
            if normalized is None:
                continue
            key = identity_mapping.key_for(normalized)
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _resolve_sampling_strategy(
        strategy: dict[Any, int],
        *,
        identity_mapping: ClassIdentityMapping,
    ) -> dict[str, int]:
        """Normalize configured class labels to canonical identity keys."""

        resolved: dict[str, int] = {}
        for label, count in strategy.items():
            if isinstance(label, str):
                key = identity_mapping.resolve_configured_key(label)
            else:
                key = identity_mapping.key_for(label)
            if key in resolved:
                raise ValueError(
                    "Multiple sampling-strategy labels resolve to "
                    f"class key {key!r}."
                )
            resolved[key] = int(count)
        return resolved

    @staticmethod
    def _restore_dtypes(
        *,
        frame: pd.DataFrame,
        source_dtypes: dict[str, str],
    ) -> pd.DataFrame:
        """Restore source dtypes when conversion is safe."""

        restored = frame.copy(
            deep=True
        )

        for column, dtype in source_dtypes.items():
            if column not in restored.columns:
                continue

            try:
                restored[
                    column
                ] = restored[
                    column
                ].astype(
                    pd.api.types.pandas_dtype(dtype)
                )

            except (
                TypeError,
                ValueError,
                OverflowError,
            ):
                continue

        return restored
