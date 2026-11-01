"""Executable transformation-plan construction for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from mlprep.phases.phase6.contract.transformation_contract import (
    TransformationContract,
)


class FeatureFamily(StrEnum):
    """Supported Phase 6 feature families."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    TIMEDELTA = "timedelta"
    UNSUPPORTED = "unsupported"


class NumericTransformationKind(StrEnum):
    """Supported numeric transformation methods."""

    NONE = "none"
    LOG1P = "log1p"
    SIGNED_LOG1P = "signed_log1p"
    BOX_COX = "box_cox"
    YEO_JOHNSON = "yeo_johnson"


class NumericScalingKind(StrEnum):
    """Supported numeric scaling methods."""

    NONE = "none"
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"


class CategoricalEncodingKind(StrEnum):
    """Supported categorical encoders."""

    ONE_HOT = "one_hot"
    ORDINAL = "ordinal"
    FREQUENCY = "frequency"
    HASHING = "hashing"


class FeatureEngineeringKind(StrEnum):
    """Supported deterministic feature engineering."""

    DATETIME_PARTS = "datetime_parts"
    TEXT_STATISTICS = "text_statistics"
    BOOLEAN_BINARY = "boolean_binary"
    TIMEDELTA_SECONDS = "timedelta_seconds"


class SourceColumnPolicy(StrEnum):
    """Policy for source columns after transformation."""

    KEEP = "keep"
    DROP = "drop"


class PlanEntryStatus(StrEnum):
    """Status of one transformation-plan entry."""

    PLANNED = "planned"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class TransformationPlanEntry(BaseModel):
    """Executable transformation plan for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    feature_family: FeatureFamily

    numeric_transformation: NumericTransformationKind | None = None
    numeric_scaling: NumericScalingKind | None = None

    categorical_encoding: CategoricalEncodingKind | None = None
    group_rare_categories: bool = False

    feature_engineering: FeatureEngineeringKind | None = None

    source_column_policy: SourceColumnPolicy = (
        SourceColumnPolicy.DROP
    )

    ordinal_order: list[str] = Field(
        default_factory=list
    )

    output_prefix: str | None = None

    requires_training_fit: bool = True

    status: PlanEntryStatus = (
        PlanEntryStatus.PLANNED
    )

    priority: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )

    reason: str

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_entry(
        self,
    ) -> TransformationPlanEntry:
        """Validate one transformation-plan entry."""

        if not self.column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        if (
            self.feature_family
            == FeatureFamily.NUMERIC
            and self.numeric_transformation is None
        ):
            raise ValueError(
                "Numeric entries require a numeric transformation."
            )

        if (
            self.feature_family
            == FeatureFamily.NUMERIC
            and self.numeric_scaling is None
        ):
            raise ValueError(
                "Numeric entries require a scaling policy."
            )

        if (
            self.feature_family
            == FeatureFamily.CATEGORICAL
            and self.categorical_encoding is None
        ):
            raise ValueError(
                "Categorical entries require an encoder."
            )

        if (
            self.categorical_encoding
            == CategoricalEncodingKind.ORDINAL
            and not self.ordinal_order
        ):
            raise ValueError(
                "Ordinal encoding requires an explicit order."
            )

        if (
            self.feature_family
            in {
                FeatureFamily.BOOLEAN,
                FeatureFamily.DATETIME,
                FeatureFamily.TEXT,
                FeatureFamily.TIMEDELTA,
            }
            and self.feature_engineering is None
        ):
            raise ValueError(
                "The feature family requires a feature-engineering method."
            )

        if (
            self.feature_family
            == FeatureFamily.UNSUPPORTED
            and self.status
            == PlanEntryStatus.PLANNED
        ):
            raise ValueError(
                "Unsupported features cannot have planned status."
            )

        return self


class TransformationPlan(BaseModel):
    """Complete executable Phase 6 transformation plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    entries: dict[
        str,
        TransformationPlanEntry,
    ] = Field(default_factory=dict)

    numeric_columns: list[str] = Field(
        default_factory=list
    )

    categorical_columns: list[str] = Field(
        default_factory=list
    )

    boolean_columns: list[str] = Field(
        default_factory=list
    )

    datetime_columns: list[str] = Field(
        default_factory=list
    )

    text_columns: list[str] = Field(
        default_factory=list
    )

    timedelta_columns: list[str] = Field(
        default_factory=list
    )

    blocked_columns: list[str] = Field(
        default_factory=list
    )

    skipped_columns: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def entry(
        self,
        column: str,
    ) -> TransformationPlanEntry:
        """Return one feature plan entry."""

        if column not in self.entries:
            raise KeyError(
                f"No transformation plan exists for {column!r}."
            )

        return self.entries[column]

    def executable_entries(
        self,
    ) -> list[TransformationPlanEntry]:
        """Return entries eligible for execution."""

        return [
            entry
            for entry in self.entries.values()
            if entry.status == PlanEntryStatus.PLANNED
        ]

    def fitting_entries(
        self,
    ) -> list[object]:
        """Return entries requiring training-only fitting."""

        return [
            entry
            for entry in self.executable_entries()
            if entry.requires_training_fit
        ]


@dataclass(slots=True)
class TransformationPlanOptions:
    """Options controlling plan construction."""

    enabled: bool = True

    default_numeric_transformation: NumericTransformationKind = (
        NumericTransformationKind.NONE
    )

    default_numeric_scaling: NumericScalingKind = (
        NumericScalingKind.ROBUST
    )

    default_categorical_encoding: CategoricalEncodingKind = (
        CategoricalEncodingKind.ONE_HOT
    )

    group_rare_categories_by_default: bool = True

    default_source_column_policy: SourceColumnPolicy = (
        SourceColumnPolicy.DROP
    )

    transform_boolean_features: bool = True
    transform_datetime_features: bool = True
    transform_text_features: bool = True
    transform_timedelta_features: bool = True

    allow_hashing_encoding: bool = False

    excluded_columns: list[str] = field(
        default_factory=list
    )

    numeric_transformation_overrides: dict[
        str,
        NumericTransformationKind,
    ] = field(default_factory=dict)

    numeric_scaling_overrides: dict[
        str,
        NumericScalingKind,
    ] = field(default_factory=dict)

    categorical_encoding_overrides: dict[
        str,
        CategoricalEncodingKind,
    ] = field(default_factory=dict)

    rare_grouping_overrides: dict[
        str,
        bool,
    ] = field(default_factory=dict)

    ordinal_orders: dict[
        str,
        list[str],
    ] = field(default_factory=dict)

    source_column_policy_overrides: dict[
        str,
        SourceColumnPolicy,
    ] = field(default_factory=dict)

    output_prefix_overrides: dict[
        str,
        str,
    ] = field(default_factory=dict)

    priority_overrides: dict[
        str,
        float,
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize plan options."""

        self.numeric_transformation_overrides = {
            str(column): (
                value
                if isinstance(
                    value,
                    NumericTransformationKind,
                )
                else NumericTransformationKind(
                    value
                )
            )
            for column, value
            in self
            .numeric_transformation_overrides
            .items()
        }

        self.numeric_scaling_overrides = {
            str(column): (
                value
                if isinstance(
                    value,
                    NumericScalingKind,
                )
                else NumericScalingKind(
                    value
                )
            )
            for column, value
            in self.numeric_scaling_overrides.items()
        }

        self.categorical_encoding_overrides = {
            str(column): (
                value
                if isinstance(
                    value,
                    CategoricalEncodingKind,
                )
                else CategoricalEncodingKind(
                    value
                )
            )
            for column, value
            in self
            .categorical_encoding_overrides
            .items()
        }

        self.source_column_policy_overrides = {
            str(column): (
                value
                if isinstance(
                    value,
                    SourceColumnPolicy,
                )
                else SourceColumnPolicy(
                    value
                )
            )
            for column, value
            in self
            .source_column_policy_overrides
            .items()
        }

        for column, order in self.ordinal_orders.items():
            if not order:
                raise ValueError(
                    f"Ordinal order for {column!r} cannot be empty."
                )

            if len(order) != len(set(order)):
                raise ValueError(
                    f"Ordinal order for {column!r} contains duplicates."
                )

        for column, prefix in (
            self.output_prefix_overrides.items()
        ):
            if not prefix.strip():
                raise ValueError(
                    f"Output prefix for {column!r} cannot be empty."
                )

        for column, priority in (
            self.priority_overrides.items()
        ):
            if not 0.0 <= priority <= 1.0:
                raise ValueError(
                    f"Priority for {column!r} must be "
                    "between zero and one."
                )


class TransformationPlanBuilder:
    """Build the complete Phase 6 transformation plan."""

    component_name = "transformation_plan"

    def __init__(
        self,
        options: TransformationPlanOptions | None = None,
    ) -> None:
        """Initialize the builder."""

        self.options = (
            options
            or TransformationPlanOptions()
        )

    def build(
        self,
        *,
        contract: TransformationContract,
    ) -> TransformationPlan:
        """Build the complete transformation plan."""

        if not self.options.enabled:
            return TransformationPlan(
                skipped_columns=(
                    contract.feature_columns
                ),
                metadata={
                    "enabled": False,
                },
            )

        excluded = set(
            self.options.excluded_columns
        )

        entries: dict[
            str,
            TransformationPlanEntry,
        ] = {}

        blocked_columns: list[str] = []
        skipped_columns: list[str] = []

        for column in contract.feature_columns:
            if column in excluded:
                skipped_columns.append(
                    column
                )

                continue

            family = FeatureFamily(
                contract.feature_groups.family_for(
                    column
                )
            )

            entry = self._build_entry(
                column=column,
                family=family,
            )

            entries[column] = entry

            if entry.status == PlanEntryStatus.BLOCKED:
                blocked_columns.append(
                    column
                )

            elif entry.status == PlanEntryStatus.SKIPPED:
                skipped_columns.append(
                    column
                )

        return TransformationPlan(
            entries=entries,
            numeric_columns=self._columns_for(
                entries=entries,
                family=FeatureFamily.NUMERIC,
            ),
            categorical_columns=self._columns_for(
                entries=entries,
                family=FeatureFamily.CATEGORICAL,
            ),
            boolean_columns=self._columns_for(
                entries=entries,
                family=FeatureFamily.BOOLEAN,
            ),
            datetime_columns=self._columns_for(
                entries=entries,
                family=FeatureFamily.DATETIME,
            ),
            text_columns=self._columns_for(
                entries=entries,
                family=FeatureFamily.TEXT,
            ),
            timedelta_columns=self._columns_for(
                entries=entries,
                family=FeatureFamily.TIMEDELTA,
            ),
            blocked_columns=blocked_columns,
            skipped_columns=skipped_columns,
            metadata={
                "enabled": True,
                "entry_count": len(entries),
                "executable_entry_count": len(
                    [
                        entry
                        for entry in entries.values()
                        if entry.status
                        == PlanEntryStatus.PLANNED
                    ]
                ),
                "blocked_count": len(
                    blocked_columns
                ),
                "skipped_count": len(
                    skipped_columns
                ),
            },
        )

    def _build_entry(
        self,
        *,
        column: str,
        family: FeatureFamily,
    ) -> TransformationPlanEntry:
        """Build one feature transformation entry."""

        source_policy = (
            self.options
            .source_column_policy_overrides
            .get(
                column,
                self.options
                .default_source_column_policy,
            )
        )

        output_prefix = (
            self.options
            .output_prefix_overrides
            .get(
                column,
                column,
            )
        )

        priority = (
            self.options
            .priority_overrides
            .get(
                column,
                0.50,
            )
        )

        if family == FeatureFamily.NUMERIC:
            transformation = (
                self.options
                .numeric_transformation_overrides
                .get(
                    column,
                    self.options
                    .default_numeric_transformation,
                )
            )

            scaling = (
                self.options
                .numeric_scaling_overrides
                .get(
                    column,
                    self.options
                    .default_numeric_scaling,
                )
            )

            return TransformationPlanEntry(
                column=column,
                feature_family=family,
                numeric_transformation=(
                    transformation
                ),
                numeric_scaling=scaling,
                source_column_policy=(
                    source_policy
                ),
                output_prefix=output_prefix,
                requires_training_fit=(
                    transformation
                    in {
                        NumericTransformationKind.BOX_COX,
                        NumericTransformationKind.YEO_JOHNSON,
                    }
                    or scaling
                    != NumericScalingKind.NONE
                ),
                priority=priority,
                reason=(
                    "Apply the configured numeric "
                    "transformation and scaling policy."
                ),
                metadata={
                    "source": (
                        "override"
                        if (
                            column
                            in self.options
                            .numeric_transformation_overrides
                            or column
                            in self.options
                            .numeric_scaling_overrides
                        )
                        else "default"
                    ),
                },
            )

        if family == FeatureFamily.CATEGORICAL:
            encoding = (
                self.options
                .categorical_encoding_overrides
                .get(
                    column,
                    self.options
                    .default_categorical_encoding,
                )
            )

            group_rare = (
                self.options
                .rare_grouping_overrides
                .get(
                    column,
                    self.options
                    .group_rare_categories_by_default,
                )
            )

            ordinal_order = list(
                self.options
                .ordinal_orders
                .get(
                    column,
                    [],
                )
            )

            status = PlanEntryStatus.PLANNED
            warnings: list[str] = []

            if (
                encoding
                == CategoricalEncodingKind.ORDINAL
                and not ordinal_order
            ):
                status = PlanEntryStatus.BLOCKED

                warnings.append(
                    "Ordinal encoding requires an explicit order."
                )

            if (
                encoding
                == CategoricalEncodingKind.HASHING
                and not self.options
                .allow_hashing_encoding
            ):
                status = PlanEntryStatus.BLOCKED

                warnings.append(
                    "Hashing encoding is disabled."
                )

            return TransformationPlanEntry(
                column=column,
                feature_family=family,
                categorical_encoding=encoding,
                group_rare_categories=(
                    group_rare
                ),
                source_column_policy=(
                    source_policy
                ),
                ordinal_order=ordinal_order,
                output_prefix=output_prefix,
                requires_training_fit=(
                    encoding
                    != CategoricalEncodingKind.HASHING
                    or group_rare
                ),
                status=status,
                priority=priority,
                reason=(
                    "Apply rare-category grouping when enabled "
                    "and encode using the configured method."
                ),
                warnings=warnings,
            )

        if family == FeatureFamily.BOOLEAN:
            return self._engineered_entry(
                column=column,
                family=family,
                enabled=(
                    self.options
                    .transform_boolean_features
                ),
                engineering=(
                    FeatureEngineeringKind
                    .BOOLEAN_BINARY
                ),
                source_policy=source_policy,
                output_prefix=output_prefix,
                priority=priority,
            )

        if family == FeatureFamily.DATETIME:
            return self._engineered_entry(
                column=column,
                family=family,
                enabled=(
                    self.options
                    .transform_datetime_features
                ),
                engineering=(
                    FeatureEngineeringKind
                    .DATETIME_PARTS
                ),
                source_policy=source_policy,
                output_prefix=output_prefix,
                priority=priority,
            )

        if family == FeatureFamily.TEXT:
            return self._engineered_entry(
                column=column,
                family=family,
                enabled=(
                    self.options
                    .transform_text_features
                ),
                engineering=(
                    FeatureEngineeringKind
                    .TEXT_STATISTICS
                ),
                source_policy=source_policy,
                output_prefix=output_prefix,
                priority=priority,
            )

        if family == FeatureFamily.TIMEDELTA:
            return self._engineered_entry(
                column=column,
                family=family,
                enabled=(
                    self.options
                    .transform_timedelta_features
                ),
                engineering=(
                    FeatureEngineeringKind
                    .TIMEDELTA_SECONDS
                ),
                source_policy=source_policy,
                output_prefix=output_prefix,
                priority=priority,
            )

        return TransformationPlanEntry(
            column=column,
            feature_family=FeatureFamily.UNSUPPORTED,
            source_column_policy=(
                SourceColumnPolicy.KEEP
            ),
            requires_training_fit=False,
            status=PlanEntryStatus.SKIPPED,
            priority=priority,
            reason=(
                "The feature dtype is not supported by "
                "the default Phase 6 transformation plan."
            ),
            warnings=[
                "Configure an explicit transformation or "
                "exclude the feature."
            ],
        )

    @staticmethod
    def _engineered_entry(
        *,
        column: str,
        family: FeatureFamily,
        enabled: bool,
        engineering: FeatureEngineeringKind,
        source_policy: SourceColumnPolicy,
        output_prefix: str,
        priority: float,
    ) -> TransformationPlanEntry:
        """Build one deterministic feature-engineering entry."""

        return TransformationPlanEntry(
            column=column,
            feature_family=family,
            feature_engineering=engineering,
            source_column_policy=source_policy,
            output_prefix=output_prefix,
            requires_training_fit=(
                engineering
                == FeatureEngineeringKind.DATETIME_PARTS
            ),
            status=(
                PlanEntryStatus.PLANNED
                if enabled
                else PlanEntryStatus.SKIPPED
            ),
            priority=priority,
            reason=(
                "Generate deterministic numeric features "
                "from the source feature."
            ),
        )

    @staticmethod
    def _columns_for(
        *,
        entries: dict[
            str,
            TransformationPlanEntry,
        ],
        family: FeatureFamily,
    ) -> list[object]:
        """Return planned columns in one feature family."""

        return [
            column
            for column, entry in entries.items()
            if (
                entry.feature_family == family
                and entry.status
                == PlanEntryStatus.PLANNED
            )
        ]
