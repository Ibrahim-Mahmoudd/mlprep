"""Executable imputation-plan construction for Phase 4."""

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

from mlprep.phases.phase3.recommendations.imputation_recommendations import (
    ImputationRecommendation,
)
from mlprep.phases.phase4.contract.imputation_contract import (
    ImputationContract,
)


class ImputationMethodName(StrEnum):
    """Supported Phase 4 imputation methods."""

    NONE = "none"

    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"

    CONSTANT = "constant"
    MISSING_CATEGORY = "missing_category"

    TIME_AWARE = "time_aware"
    GROUP_BASED = "group_based"
    MODEL_BASED = "model_based"

    DROP_FEATURE = "drop_feature"
    REVIEW = "review"


class ImputationFeatureFamily(StrEnum):
    """Supported feature families for imputation."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    TIMEDELTA = "timedelta"
    UNKNOWN = "unknown"


class PlanEntryStatus(StrEnum):
    """Execution status of one imputation plan entry."""

    PLANNED = "planned"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ImputationPlanEntry(BaseModel):
    """Executable imputation plan for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    feature_family: ImputationFeatureFamily

    method: ImputationMethodName

    add_missing_indicator: bool = False

    constant_value: str | int | float | bool | None = None

    group_columns: list[str] = Field(
        default_factory=list
    )

    timestamp_column: str | None = None

    requires_training_fit: bool = True

    status: PlanEntryStatus = (
        PlanEntryStatus.PLANNED
    )

    reason: str

    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_entry(
        self,
    ) -> ImputationPlanEntry:
        """Validate one plan entry."""

        if (
            self.method
            == ImputationMethodName.CONSTANT
            and self.constant_value is None
        ):
            raise ValueError(
                "Constant imputation requires constant_value."
            )

        if (
            self.method
            == ImputationMethodName.GROUP_BASED
            and not self.group_columns
        ):
            raise ValueError(
                "Group-based imputation requires group columns."
            )

        if (
            self.method
            == ImputationMethodName.TIME_AWARE
            and self.timestamp_column is None
        ):
            raise ValueError(
                "Time-aware imputation requires a timestamp column."
            )

        return self


class ImputationPlan(BaseModel):
    """Complete executable Phase 4 imputation plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    entries: dict[
        str,
        ImputationPlanEntry,
    ] = Field(default_factory=dict)

    indicator_columns: list[str] = Field(
        default_factory=list
    )

    drop_feature_columns: list[str] = Field(
        default_factory=list
    )

    review_columns: list[str] = Field(
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
    ) -> ImputationPlanEntry:
        """Return the plan entry for one feature."""

        if column not in self.entries:
            raise KeyError(
                f"No imputation plan exists for {column!r}."
            )

        return self.entries[column]

    def executable_entries(
        self,
    ) -> list[object]:
        """Return entries eligible for execution."""

        return [
            entry
            for entry in self.entries.values()
            if entry.status
            == PlanEntryStatus.PLANNED
            and entry.method
            not in {
                ImputationMethodName.NONE,
                ImputationMethodName.REVIEW,
                ImputationMethodName.DROP_FEATURE,
            }
        ]

    def fitting_entries(
        self,
    ) -> list[object]:
        """Return entries requiring training-only fitting."""

        return [
            entry
            for entry
            in self.executable_entries()
            if entry.requires_training_fit
        ]


@dataclass(slots=True)
class ImputationPlanOptions:
    """Options controlling imputation-plan construction."""

    default_numeric_method: ImputationMethodName = (
        ImputationMethodName.MEDIAN
    )

    default_categorical_method: ImputationMethodName = (
        ImputationMethodName.MISSING_CATEGORY
    )

    default_boolean_method: ImputationMethodName = (
        ImputationMethodName.MOST_FREQUENT
    )

    default_datetime_method: ImputationMethodName = (
        ImputationMethodName.MEDIAN
    )

    default_text_method: ImputationMethodName = (
        ImputationMethodName.CONSTANT
    )

    default_timedelta_method: ImputationMethodName = (
        ImputationMethodName.MEDIAN
    )

    categorical_missing_value: str = "__MISSING__"
    text_missing_value: str = ""

    create_unspecified_missing_plan_entries: bool = True

    allow_drop_feature: bool = False
    allow_review_entries: bool = False

    blocked_columns: list[str] = field(
        default_factory=list
    )

    method_overrides: dict[
        str,
        ImputationMethodName,
    ] = field(default_factory=dict)

    constant_overrides: dict[
        str,
        str | int | float | bool,
    ] = field(default_factory=dict)

    indicator_overrides: dict[
        str,
        bool,
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate imputation-plan options."""

        if not self.categorical_missing_value:
            raise ValueError(
                "categorical_missing_value cannot be empty."
            )


class ImputationPlanBuilder:
    """Convert Phase 3 recommendations into an execution plan."""

    component_name = "imputation_plan"

    def __init__(
        self,
        options: ImputationPlanOptions | None = None,
    ) -> None:
        """Initialize the plan builder."""

        self.options = (
            options or ImputationPlanOptions()
        )

    def build(
        self,
        *,
        contract: ImputationContract,
    ) -> ImputationPlan:
        """Build the complete Phase 4 imputation plan."""

        recommendations = (
            contract
            .phase3_recommendations
            .imputation
            .recommendations
        )

        training = contract.train.X

        protected_columns = {
            contract.target_column,
            contract.internal_row_id_column,
            *self.options.blocked_columns,
        }

        entries: dict[
            str,
            ImputationPlanEntry,
        ] = {}

        skipped_columns: list[str] = []

        for column in contract.feature_columns:
            if column in protected_columns:
                skipped_columns.append(column)
                continue

            missing_count = int(
                training[column].isna().sum()
            )

            recommendation = recommendations.get(
                column
            )

            if recommendation is None:
                if (
                    missing_count == 0
                    or not self.options
                    .create_unspecified_missing_plan_entries
                ):
                    continue

                entry = self._default_entry(
                    contract=contract,
                    column=column,
                    missing_count=missing_count,
                )

            else:
                entry = self._recommended_entry(
                    contract=contract,
                    recommendation=recommendation,
                    missing_count=missing_count,
                )

            entry = self._apply_overrides(
                entry
            )

            entries[column] = entry

        indicator_columns = [
            column
            for column, entry in entries.items()
            if entry.add_missing_indicator
        ]

        drop_columns = [
            column
            for column, entry in entries.items()
            if entry.method
            == ImputationMethodName.DROP_FEATURE
        ]

        review_columns = [
            column
            for column, entry in entries.items()
            if entry.method
            == ImputationMethodName.REVIEW
        ]

        return ImputationPlan(
            entries=entries,
            indicator_columns=indicator_columns,
            drop_feature_columns=drop_columns,
            review_columns=review_columns,
            skipped_columns=skipped_columns,
            metadata={
                "entry_count": len(entries),
                "training_fit_entry_count": len(
                    [
                        entry
                        for entry in entries.values()
                        if entry.requires_training_fit
                    ]
                ),
                "indicator_count": len(
                    indicator_columns
                ),
                "drop_feature_count": len(
                    drop_columns
                ),
                "review_count": len(
                    review_columns
                ),
            },
        )

    def _recommended_entry(
        self,
        *,
        contract: ImputationContract,
        recommendation: ImputationRecommendation,
        missing_count: int,
    ) -> ImputationPlanEntry:
        """Build an entry from a Phase 3 recommendation."""

        column = recommendation.column

        family = self._family(
            contract=contract,
            column=column,
        )

        method = self._map_method(
            recommendation.method
        )

        warnings = list(
            recommendation.warnings
        )

        status = PlanEntryStatus.PLANNED

        if (
            method == ImputationMethodName.DROP_FEATURE
            and not self.options.allow_drop_feature
        ):
            status = PlanEntryStatus.BLOCKED

            warnings.append(
                "Feature dropping is disabled in Phase 4."
            )

        if (
            method == ImputationMethodName.REVIEW
            and not self.options.allow_review_entries
        ):
            status = PlanEntryStatus.BLOCKED

            warnings.append(
                "Review-only entries cannot be executed."
            )

        constant_value = (
            recommendation.constant_value
        )

        if (
            method
            == ImputationMethodName.MISSING_CATEGORY
            and constant_value is None
        ):
            constant_value = (
                self.options
                .categorical_missing_value
            )

        if (
            method == ImputationMethodName.CONSTANT
            and constant_value is None
        ):
            constant_value = self._default_constant(
                family
            )

        requires_training_fit = (
            self._requires_training_fit(
                method
            )
        )

        return ImputationPlanEntry(
            column=column,
            feature_family=family,
            method=method,
            add_missing_indicator=(
                recommendation
                .add_missing_indicator
            ),
            constant_value=constant_value,
            group_columns=(
                list(contract.group_columns)
                if method
                == ImputationMethodName.GROUP_BASED
                else []
            ),
            timestamp_column=(
                contract.timestamp_column
                if method
                == ImputationMethodName.TIME_AWARE
                else None
            ),
            requires_training_fit=(
                requires_training_fit
            ),
            status=status,
            reason=recommendation.reason,
            priority=recommendation.priority,
            warnings=warnings,
            metadata={
                "training_missing_count": (
                    missing_count
                ),
                "phase3_method": (
                    recommendation.method
                ),
                "phase3_metadata": dict(
                    recommendation.metadata
                ),
            },
        )

    def _default_entry(
        self,
        *,
        contract: ImputationContract,
        column: str,
        missing_count: int,
    ) -> ImputationPlanEntry:
        """Build a default plan entry."""

        family = self._family(
            contract=contract,
            column=column,
        )

        method = self._default_method(
            family
        )

        constant_value = None

        if method == ImputationMethodName.CONSTANT:
            constant_value = self._default_constant(
                family
            )

        if (
            method
            == ImputationMethodName.MISSING_CATEGORY
        ):
            constant_value = (
                self.options
                .categorical_missing_value
            )

        return ImputationPlanEntry(
            column=column,
            feature_family=family,
            method=method,
            add_missing_indicator=False,
            constant_value=constant_value,
            timestamp_column=(
                contract.timestamp_column
                if method
                == ImputationMethodName.TIME_AWARE
                else None
            ),
            requires_training_fit=(
                self._requires_training_fit(
                    method
                )
            ),
            reason=(
                "The feature contains missing values but "
                "has no explicit Phase 3 recommendation. "
                "A configured default method was selected."
            ),
            priority=0.50,
            metadata={
                "training_missing_count": (
                    missing_count
                ),
                "source": "default",
            },
        )

    def _apply_overrides(
        self,
        entry: ImputationPlanEntry,
    ) -> ImputationPlanEntry:
        """Apply configured column-specific overrides."""

        update_values: dict[
            str,
            object,
        ] = {}

        if entry.column in self.options.method_overrides:
            method = self.options.method_overrides[
                entry.column
            ]

            update_values["method"] = method

            update_values[
                "requires_training_fit"
            ] = self._requires_training_fit(
                method
            )

        if entry.column in self.options.constant_overrides:
            update_values[
                "constant_value"
            ] = self.options.constant_overrides[
                entry.column
            ]

        if entry.column in self.options.indicator_overrides:
            update_values[
                "add_missing_indicator"
            ] = self.options.indicator_overrides[
                entry.column
            ]

        if not update_values:
            return entry

        updated = entry.model_copy(
            update=update_values,
            deep=True,
        )

        return ImputationPlanEntry.model_validate(
            updated.model_dump()
        )

    def _family(
        self,
        *,
        contract: ImputationContract,
        column: str,
    ) -> ImputationFeatureFamily:
        """Resolve the feature family for one column."""

        groups = contract.phase3_state.columns

        family_mapping = {
            ImputationFeatureFamily.NUMERIC: (
                groups.numeric
            ),
            ImputationFeatureFamily.CATEGORICAL: (
                groups.categorical
            ),
            ImputationFeatureFamily.BOOLEAN: (
                groups.boolean
            ),
            ImputationFeatureFamily.DATETIME: (
                groups.datetime
            ),
            ImputationFeatureFamily.TEXT: (
                groups.text
            ),
            ImputationFeatureFamily.TIMEDELTA: (
                groups.timedelta
            ),
        }

        for family, family_columns in (
            family_mapping.items()
        ):
            if column in family_columns:
                return family

        return ImputationFeatureFamily.UNKNOWN

    def _default_method(
        self,
        family: ImputationFeatureFamily,
    ) -> ImputationMethodName:
        """Return the default method for one feature family."""

        methods = {
            ImputationFeatureFamily.NUMERIC: (
                self.options.default_numeric_method
            ),
            ImputationFeatureFamily.CATEGORICAL: (
                self.options
                .default_categorical_method
            ),
            ImputationFeatureFamily.BOOLEAN: (
                self.options.default_boolean_method
            ),
            ImputationFeatureFamily.DATETIME: (
                self.options.default_datetime_method
            ),
            ImputationFeatureFamily.TEXT: (
                self.options.default_text_method
            ),
            ImputationFeatureFamily.TIMEDELTA: (
                self.options.default_timedelta_method
            ),
            ImputationFeatureFamily.UNKNOWN: (
                ImputationMethodName.REVIEW
            ),
        }

        return methods[family]

    def _default_constant(
        self,
        family: ImputationFeatureFamily,
    ) -> str | int | float | bool:
        """Return the default constant for a feature family."""

        if family == ImputationFeatureFamily.TEXT:
            return self.options.text_missing_value

        if (
            family
            == ImputationFeatureFamily.CATEGORICAL
        ):
            return (
                self.options
                .categorical_missing_value
            )

        if family == ImputationFeatureFamily.BOOLEAN:
            return False

        if family in {
            ImputationFeatureFamily.NUMERIC,
            ImputationFeatureFamily.TIMEDELTA,
        }:
            return 0.0

        return "__MISSING__"

    @staticmethod
    def _map_method(
        method: str,
    ) -> ImputationMethodName:
        """Map a Phase 3 recommendation method."""

        method_mapping = {
            "none": ImputationMethodName.NONE,
            "mean": ImputationMethodName.MEAN,
            "median": ImputationMethodName.MEDIAN,
            "most_frequent": (
                ImputationMethodName.MOST_FREQUENT
            ),
            "constant": (
                ImputationMethodName.CONSTANT
            ),
            "missing_category": (
                ImputationMethodName.MISSING_CATEGORY
            ),
            "time_aware": (
                ImputationMethodName.TIME_AWARE
            ),
            "group_based": (
                ImputationMethodName.GROUP_BASED
            ),
            "model_based": (
                ImputationMethodName.MODEL_BASED
            ),
            "drop_feature": (
                ImputationMethodName.DROP_FEATURE
            ),
            "review": (
                ImputationMethodName.REVIEW
            ),
        }

        return method_mapping.get(
            method,
            ImputationMethodName.REVIEW,
        )

    @staticmethod
    def _requires_training_fit(
        method: ImputationMethodName,
    ) -> bool:
        """Return whether a method learns training state."""

        return method in {
            ImputationMethodName.MEAN,
            ImputationMethodName.MEDIAN,
            ImputationMethodName.MOST_FREQUENT,
            ImputationMethodName.GROUP_BASED,
            ImputationMethodName.TIME_AWARE,
            ImputationMethodName.MODEL_BASED,
        }
