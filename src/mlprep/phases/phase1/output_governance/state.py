"""Persistent state resolved during Phase 1."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from mlprep.types import (
    ColumnRole,
    PhysicalType,
    SemanticType,
    TaskType,
)


class Phase1StateModel(BaseModel):
    """Base model for Phase 1 state objects."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )


class ColumnNameState(Phase1StateModel):
    """Resolved state for one column name."""

    original_name: str
    normalized_name: str

    was_changed: bool = False

    change_reasons: list[str] = Field(
        default_factory=list
    )

    collision_group: str | None = None
    suffix_added: str | None = None


class MissingTokenState(Phase1StateModel):
    """Resolved missing-token state for one column."""

    column: str

    configured_tokens: list[
        str | int | float
    ] = Field(default_factory=list)

    detected_tokens: list[
        str | int | float
    ] = Field(default_factory=list)

    normalized_token_counts: dict[
        str,
        int,
    ] = Field(default_factory=dict)

    total_normalized_count: int = Field(
        default=0,
        ge=0,
    )

    case_sensitive: bool = False
    strip_before_comparison: bool = True


class TypeResolutionState(Phase1StateModel):
    """Resolved physical and semantic type state."""

    column: str

    original_physical_type: PhysicalType
    resolved_physical_type: PhysicalType

    semantic_type: SemanticType

    semantic_type_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    type_was_declared: bool = False
    type_was_inferred: bool = False
    coercion_was_applied: bool = False

    parse_success_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    failed_parse_count: int = Field(
        default=0,
        ge=0,
    )

    detected_python_types: list[str] = Field(
        default_factory=list
    )

    inference_evidence: dict[str, Any] = Field(
        default_factory=dict
    )


class CardinalityState(Phase1StateModel):
    """Cardinality and uniqueness state."""

    column: str

    non_missing_count: int = Field(
        default=0,
        ge=0,
    )

    unique_count: int = Field(
        default=0,
        ge=0,
    )

    uniqueness_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    most_frequent_value_count: int = Field(
        default=0,
        ge=0,
    )

    most_frequent_value_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    cardinality_band: str = "unknown"

    is_constant: bool = False
    is_near_constant: bool = False
    is_identifier_like: bool = False

    count_is_approximate: bool = False


class MemoryChangeState(Phase1StateModel):
    """One physical dtype change."""

    column: str

    dtype_before: str
    dtype_after: str

    memory_before_bytes: int = Field(
        default=0,
        ge=0,
    )

    memory_after_bytes: int = Field(
        default=0,
        ge=0,
    )

    semantic_type_preserved: bool = True
    precision_verified: bool = True
    reversible: bool = True

    @property
    def memory_saved_bytes(self) -> int:
        """Return memory saved by the change."""

        return max(
            0,
            self.memory_before_bytes
            - self.memory_after_bytes,
        )


class ColumnState(Phase1StateModel):
    """Complete resolved state for one column."""

    original_name: str
    normalized_name: str

    roles: set[ColumnRole] = Field(
        default_factory=set
    )

    physical_type: PhysicalType = (
        PhysicalType.UNKNOWN
    )

    semantic_type: SemanticType = (
        SemanticType.UNKNOWN
    )

    semantic_type_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    required: bool = True
    nullable: bool = True

    use_as_model_feature: bool = True
    available_at_inference: bool = True

    is_sensitive: bool = False
    sensitive_type: str | None = None

    missing_count: int = Field(
        default=0,
        ge=0,
    )

    missing_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    unique_count: int = Field(
        default=0,
        ge=0,
    )

    uniqueness_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    is_constant: bool = False
    is_near_constant: bool = False
    is_identifier_like: bool = False

    dtype_before: str | None = None
    dtype_after: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def validate_column_state(
        cls,
        values: Any,
    ) -> Any:
        """Resolve column eligibility from roles."""

        if not isinstance(values, dict):
            return values

        values = values.copy()
        roles = set(values.get("roles", set()))
        non_feature_roles = {
            ColumnRole.TARGET,
            ColumnRole.IDENTIFIER,
            ColumnRole.METADATA,
            ColumnRole.IGNORED,
            ColumnRole.INTERNAL,
            ColumnRole.SAMPLE_WEIGHT,
            ColumnRole.FOLD,
            ColumnRole.OUTCOME_TIMESTAMP,
        }

        if roles.intersection(non_feature_roles):
            values["use_as_model_feature"] = False

        if ColumnRole.IGNORED in roles:
            values["available_at_inference"] = False

        if (
            values.get("semantic_type", SemanticType.UNKNOWN)
            == SemanticType.IDENTIFIER
        ):
            values["is_identifier_like"] = True
            values["use_as_model_feature"] = False

        return values


class ProblemContractState(Phase1StateModel):
    """Resolved machine-learning problem contract."""

    task_type: TaskType
    target_column: str

    positive_class: (
        str | int | bool | None
    ) = None

    unit_of_observation: str

    observation_key: list[str] = Field(
        default_factory=list
    )

    entity_columns: list[str] = Field(
        default_factory=list
    )

    group_columns: list[str] = Field(
        default_factory=list
    )

    prediction_timestamp_column: (
        str | None
    ) = None

    outcome_timestamp_column: (
        str | None
    ) = None

    target_horizon: str | None = None

    feature_columns: list[str] = Field(
        default_factory=list
    )

    inference_columns: list[str] = Field(
        default_factory=list
    )

    resolved: bool = False


class Phase1State(Phase1StateModel):
    """Complete persistent state resolved by Phase 1."""

    state_version: str = "0.1.0"

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    internal_row_id_column: str = (
        "__mlprep_row_id__"
    )

    problem_contract: (
        ProblemContractState | None
    ) = None

    original_column_order: list[str] = Field(
        default_factory=list
    )

    canonical_column_order: list[str] = Field(
        default_factory=list
    )

    column_name_states: dict[
        str,
        ColumnNameState,
    ] = Field(default_factory=dict)

    original_to_normalized_names: dict[
        str,
        str,
    ] = Field(default_factory=dict)

    normalized_to_original_names: dict[
        str,
        str,
    ] = Field(default_factory=dict)

    column_states: dict[
        str,
        ColumnState,
    ] = Field(default_factory=dict)

    missing_token_states: dict[
        str,
        MissingTokenState,
    ] = Field(default_factory=dict)

    type_resolution_states: dict[
        str,
        TypeResolutionState,
    ] = Field(default_factory=dict)

    cardinality_states: dict[
        str,
        CardinalityState,
    ] = Field(default_factory=dict)

    memory_changes: list[
        MemoryChangeState
    ] = Field(default_factory=list)

    duplicate_column_groups: list[
        list[str]
    ] = Field(default_factory=list)

    dataset_fingerprint: str | None = None
    schema_fingerprint: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def register_column_name(
        self,
        state: ColumnNameState,
    ) -> None:
        """Register one column-name mapping."""

        original_name = state.original_name
        normalized_name = state.normalized_name

        existing_normalized_name = (
            self.original_to_normalized_names.get(
                original_name
            )
        )

        if (
            existing_normalized_name is not None
            and existing_normalized_name
            != normalized_name
        ):
            raise ValueError(
                f"Original column {original_name!r} "
                "is already mapped to "
                f"{existing_normalized_name!r}."
            )

        existing_original_name = (
            self.normalized_to_original_names.get(
                normalized_name
            )
        )

        if (
            existing_original_name is not None
            and existing_original_name
            != original_name
        ):
            raise ValueError(
                f"Normalized column {normalized_name!r} "
                "is already mapped from "
                f"{existing_original_name!r}."
            )

        self.column_name_states[
            normalized_name
        ] = state

        self.original_to_normalized_names[
            original_name
        ] = normalized_name

        self.normalized_to_original_names[
            normalized_name
        ] = original_name

    def register_column_state(
        self,
        state: ColumnState,
    ) -> None:
        """Add or replace one column state."""

        self.column_states[
            state.normalized_name
        ] = state

    def register_missing_token_state(
        self,
        state: MissingTokenState,
    ) -> None:
        """Add or replace missing-token state."""

        self.missing_token_states[
            state.column
        ] = state

    def register_type_resolution(
        self,
        state: TypeResolutionState,
    ) -> None:
        """Add or replace type-resolution state."""

        self.type_resolution_states[
            state.column
        ] = state

    def register_cardinality_state(
        self,
        state: CardinalityState,
    ) -> None:
        """Add or replace cardinality state."""

        self.cardinality_states[
            state.column
        ] = state

    def register_memory_change(
        self,
        state: MemoryChangeState,
    ) -> None:
        """Register one memory dtype change."""

        self.memory_changes.append(state)

    def get_column_state(
        self,
        column: str,
    ) -> ColumnState:
        """Return the state for one column."""

        if column not in self.column_states:
            raise KeyError(
                "No Phase 1 state exists for "
                f"column {column!r}."
            )

        return self.column_states[column]

    def get_columns_by_role(
        self,
        role: ColumnRole,
    ) -> list[object]:
        """Return columns containing one role."""

        return [
            column_name
            for column_name, column_state
            in self.column_states.items()
            if role in column_state.roles
        ]

    def get_model_feature_columns(
        self,
    ) -> list[object]:
        """Return model-eligible columns."""

        return [
            column_name
            for column_name, column_state
            in self.column_states.items()
            if column_state.use_as_model_feature
        ]

    def get_inference_columns(
        self,
    ) -> list[object]:
        """Return inference-time input columns."""

        return [
            column_name
            for column_name, column_state
            in self.column_states.items()
            if (
                column_state.available_at_inference
                and ColumnRole.TARGET
                not in column_state.roles
                and ColumnRole.INTERNAL
                not in column_state.roles
            )
        ]

    def get_sensitive_columns(
        self,
    ) -> list[object]:
        """Return sensitive columns."""

        return [
            column_name
            for column_name, column_state
            in self.column_states.items()
            if column_state.is_sensitive
        ]

    def calculate_total_memory_saved(
        self,
    ) -> int:
        """Return total memory saved."""

        return sum(
            change.memory_saved_bytes
            for change in self.memory_changes
        )

    def finalize_column_orders(
        self,
        *,
        canonical_column_order: list[str],
    ) -> None:
        """Validate and save canonical order."""

        seen_columns: set[str] = set()
        duplicated_columns: set[str] = set()

        for column in canonical_column_order:
            if column in seen_columns:
                duplicated_columns.add(column)

            seen_columns.add(column)

        if duplicated_columns:
            duplicates = ", ".join(
                sorted(duplicated_columns)
            )

            raise ValueError(
                "Canonical column order contains "
                f"duplicates: {duplicates}"
            )

        unknown_columns = set(
            canonical_column_order
        ).difference(self.column_states)

        if unknown_columns:
            columns = ", ".join(
                sorted(unknown_columns)
            )

            raise ValueError(
                "Canonical column order contains "
                "columns without registered state: "
                f"{columns}"
            )

        self.canonical_column_order = list(
            canonical_column_order
        )
