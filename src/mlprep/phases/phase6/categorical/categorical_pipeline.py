"""Fitted categorical transformation pipeline for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase6.categorical.frequency_encoder import (
    FrequencyEncoder,
    FrequencyEncoderOptions,
)
from mlprep.phases.phase6.categorical.hashing_encoder import (
    HashingEncoder,
    HashingEncoderOptions,
)
from mlprep.phases.phase6.categorical.one_hot_encoder import (
    OneHotEncoder,
    OneHotEncoderOptions,
)
from mlprep.phases.phase6.categorical.ordinal_encoder import (
    OrdinalEncoder,
    OrdinalEncoderOptions,
)
from mlprep.phases.phase6.categorical.rare_category_grouper import (
    RareCategoryGrouper,
    RareCategoryGrouperOptions,
)
from mlprep.phases.phase6.contract.transformation_plan import (
    CategoricalEncodingKind,
    TransformationPlan,
    TransformationPlanEntry,
)


class UnknownCategorySummary(BaseModel):
    """Unknown-category summary for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    unknown_count: int = Field(
        default=0,
        ge=0,
    )

    unknown_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    applied_policy: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class CategoricalColumnPipelineState(BaseModel):
    """Fitted categorical state for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    column: str
    encoding: str

    rare_grouping_enabled: bool = False

    rare_grouper_state: dict[
        str,
        Any,
    ] | None = None

    encoder_state: dict[
        str,
        Any,
    ] = Field(default_factory=dict)

    output_columns: list[str] = Field(
        default_factory=list
    )

    fitted: bool = False
    fitted_on_training_only: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class CategoricalPipelineResult(BaseModel):
    """Output from transforming one categorical frame."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    frame: pd.DataFrame

    output_columns: list[str] = Field(
        default_factory=list
    )

    unknown_categories: dict[
        str,
        UnknownCategorySummary,
    ] = Field(default_factory=dict)

    states: dict[
        str,
        CategoricalColumnPipelineState,
    ] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class CategoricalPipelineOptions:
    """Options controlling the categorical pipeline."""

    rare_category: RareCategoryGrouperOptions = field(
        default_factory=RareCategoryGrouperOptions
    )

    one_hot: OneHotEncoderOptions = field(
        default_factory=OneHotEncoderOptions
    )

    ordinal: OrdinalEncoderOptions = field(
        default_factory=OrdinalEncoderOptions
    )

    frequency: FrequencyEncoderOptions = field(
        default_factory=FrequencyEncoderOptions
    )

    hashing: HashingEncoderOptions = field(
        default_factory=HashingEncoderOptions
    )

    require_nonempty_categorical_plan: bool = False

    require_unique_output_columns: bool = True


class CategoricalPipeline:
    """Fit and apply categorical grouping and encoding."""

    component_name = "categorical_pipeline"

    def __init__(
        self,
        options: CategoricalPipelineOptions | None = None,
    ) -> None:
        """Initialize the categorical pipeline."""

        self.options = (
            options
            or CategoricalPipelineOptions()
        )

        self.groupers: dict[
            str,
            RareCategoryGrouper | None,
        ] = {}

        self.encoders: dict[
            str,
            object,
        ] = {}

        self.entries: dict[
            str,
            TransformationPlanEntry,
        ] = {}

        self.states: dict[
            str,
            CategoricalColumnPipelineState,
        ] = {}

        self._fitted = False

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the pipeline is fitted."""

        return self._fitted

    def fit(
        self,
        *,
        frame: pd.DataFrame,
        plan: TransformationPlan,
    ) -> CategoricalPipeline:
        """Fit grouping and encoding on training data."""

        if self._fitted:
            raise RuntimeError(
                "The categorical pipeline is already fitted."
            )

        entries = [
            plan.entries[column]
            for column
            in plan.categorical_columns
        ]

        if (
            not entries
            and self.options
            .require_nonempty_categorical_plan
        ):
            raise ValueError(
                "The categorical transformation plan is empty."
            )

        all_output_columns: list[str] = []

        for entry in entries:
            column = entry.column

            if column not in frame.columns:
                raise KeyError(
                    f"Categorical feature {column!r} "
                    "does not exist."
                )

            source = frame[
                column
            ]

            grouper = None
            grouped = source

            if entry.group_rare_categories:
                grouper = RareCategoryGrouper(
                    column=column,
                    options=self.options.rare_category,
                )

                grouped = grouper.fit_transform(
                    source
                )

            encoder = self._build_encoder(
                entry
            )

            encoded = encoder.fit_transform(
                grouped
            )

            output_columns = [
                str(output_column)
                for output_column
                in encoded.columns
            ]

            collisions = set(
                all_output_columns
            ).intersection(
                output_columns
            )

            if (
                collisions
                and self.options
                .require_unique_output_columns
            ):
                raise ValueError(
                    "Categorical output-column collision: "
                    f"{sorted(collisions)}"
                )

            all_output_columns.extend(
                output_columns
            )

            self.groupers[column] = grouper
            self.encoders[column] = encoder
            self.entries[column] = entry

            grouper_state = (
                grouper.get_state().model_dump(
                    mode="json"
                )
                if grouper is not None
                else None
            )

            encoder_state = (
                encoder.get_state().model_dump(
                    mode="json"
                )
            )

            self.states[
                column
            ] = CategoricalColumnPipelineState(
                column=column,
                encoding=(
                    entry
                    .categorical_encoding
                    .value
                ),
                rare_grouping_enabled=(
                    entry.group_rare_categories
                ),
                rare_grouper_state=(
                    grouper_state
                ),
                encoder_state=(
                    encoder_state
                ),
                output_columns=(
                    output_columns
                ),
                fitted=True,
                fitted_on_training_only=True,
            )

        self._fitted = True

        return self

    def transform(
        self,
        *,
        frame: pd.DataFrame,
    ) -> CategoricalPipelineResult:
        """Transform categorical features using fitted state."""

        if not self._fitted:
            raise RuntimeError(
                "The categorical pipeline must be fitted "
                "before transformation."
            )

        output_parts: list[
            pd.DataFrame
        ] = []

        unknown_summaries: dict[
            str,
            UnknownCategorySummary,
        ] = {}

        for column, _entry in (
            self.entries.items()
        ):
            if column not in frame.columns:
                raise KeyError(
                    f"Categorical feature {column!r} "
                    "does not exist."
                )

            source = frame[
                column
            ]

            grouper = self.groupers[
                column
            ]

            encoder = self.encoders[
                column
            ]

            grouped = (
                grouper.transform(
                    source
                )
                if grouper is not None
                else source
            )

            unknown_mask = encoder.unknown_mask(
                grouped
            )

            unknown_count = int(
                unknown_mask.sum()
            )

            row_count = len(
                grouped
            )

            unknown_summaries[
                column
            ] = UnknownCategorySummary(
                column=column,
                unknown_count=unknown_count,
                unknown_percentage=(
                    unknown_count
                    / row_count
                    * 100.0
                    if row_count > 0
                    else 0.0
                ),
                applied_policy=(
                    self._unknown_policy(
                        encoder
                    )
                ),
            )

            encoded = encoder.transform(
                grouped
            )

            expected_columns = (
                self.states[
                    column
                ].output_columns
            )

            if list(
                encoded.columns
            ) != expected_columns:
                raise ValueError(
                    f"Encoded schema for {column!r} differs "
                    "from the training schema."
                )

            output_parts.append(
                encoded
            )

        if output_parts:
            output = pd.concat(
                output_parts,
                axis=1,
            )

        else:
            output = pd.DataFrame(
                index=frame.index
            )

        if (
            self.options
            .require_unique_output_columns
            and output.columns.duplicated().any()
        ):
            duplicate_columns = [
                str(column)
                for column
                in output.columns[
                    output.columns.duplicated(
                        keep=False
                    )
                ].unique()
            ]

            raise ValueError(
                "Duplicate categorical output columns: "
                f"{duplicate_columns}"
            )

        return CategoricalPipelineResult(
            frame=output,
            output_columns=[
                str(column)
                for column in output.columns
            ],
            unknown_categories=(
                unknown_summaries
            ),
            states={
                column: state.model_copy(
                    deep=True
                )
                for column, state
                in self.states.items()
            },
            metadata={
                "source_feature_count": len(
                    self.entries
                ),
                "output_feature_count": len(
                    output.columns
                ),
            },
        )

    def fit_transform(
        self,
        *,
        frame: pd.DataFrame,
        plan: TransformationPlan,
    ) -> CategoricalPipelineResult:
        """Fit and transform training categories."""

        self.fit(
            frame=frame,
            plan=plan,
        )

        return self.transform(
            frame=frame
        )

    def output_columns(
        self,
    ) -> list[object]:
        """Return fitted categorical output columns."""

        if not self._fitted:
            raise RuntimeError(
                "Output columns are unavailable before fitting."
            )

        return [
            output_column
            for state in self.states.values()
            for output_column
            in state.output_columns
        ]

    def get_states(
        self,
    ) -> dict[
        str,
        CategoricalColumnPipelineState,
    ]:
        """Return copies of fitted categorical states."""

        if not self._fitted:
            raise RuntimeError(
                "Categorical states are unavailable "
                "before fitting."
            )

        return {
            column: state.model_copy(
                deep=True
            )
            for column, state
            in self.states.items()
        }

    def _build_encoder(
        self,
        entry: TransformationPlanEntry,
    ) -> object:
        """Build the configured categorical encoder."""

        encoding = (
            entry.categorical_encoding
        )

        if (
            encoding
            == CategoricalEncodingKind.ONE_HOT
        ):
            return OneHotEncoder(
                column=entry.column,
                output_prefix=(
                    entry.output_prefix
                ),
                options=self.options.one_hot,
            )

        if (
            encoding
            == CategoricalEncodingKind.ORDINAL
        ):
            return OrdinalEncoder(
                column=entry.column,
                category_order=(
                    entry.ordinal_order
                ),
                output_prefix=(
                    entry.output_prefix
                ),
                options=self.options.ordinal,
            )

        if (
            encoding
            == CategoricalEncodingKind.FREQUENCY
        ):
            return FrequencyEncoder(
                column=entry.column,
                output_prefix=(
                    entry.output_prefix
                ),
                options=self.options.frequency,
            )

        if (
            encoding
            == CategoricalEncodingKind.HASHING
        ):
            return HashingEncoder(
                column=entry.column,
                output_prefix=(
                    entry.output_prefix
                ),
                options=self.options.hashing,
            )

        raise ValueError(
            "Unsupported categorical encoding: "
            f"{encoding!r}"
        )

    @staticmethod
    def _unknown_policy(
        encoder: object,
    ) -> str:
        """Return one encoder's unknown-category policy."""

        options = getattr(
            encoder,
            "options",
            None,
        )

        if options is None:
            return "unknown"

        handle_unknown = getattr(
            options,
            "handle_unknown",
            None,
        )

        if handle_unknown is not None:
            return str(
                handle_unknown
            )

        unknown_value = getattr(
            options,
            "unknown_value",
            None,
        )

        if unknown_value is not None:
            return (
                f"use_value:{unknown_value}"
            )

        if isinstance(
            encoder,
            HashingEncoder,
        ):
            return "hash_bucket"

        return "configured_fallback"