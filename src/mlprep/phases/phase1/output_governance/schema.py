"""Canonical schema construction for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    ColumnInventoryItem,
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    ColumnState,
    Phase1State,
)
from mlprep.types import (
    ColumnRole,
    FindingAction,
    FindingCategory,
    PhysicalType,
    SemanticType,
    Severity,
)


@dataclass(slots=True)
class SchemaBuilderOptions:
    """Options controlling canonical schema construction."""

    enabled: bool = True

    require_state_for_every_column: bool = True
    include_internal_columns: bool = True

    required_columns: list[str] = field(
        default_factory=list
    )

    optional_columns: list[str] = field(
        default_factory=list
    )

    descriptions: dict[str, str] = field(
        default_factory=dict
    )

    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate schema-builder options."""

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )

        overlap = set(
            self.required_columns
        ).intersection(self.optional_columns)

        if overlap:
            columns = ", ".join(sorted(overlap))

            raise ValueError(
                "Columns cannot be both required and optional: "
                f"{columns}"
            )


class SchemaBuilder:
    """Build the canonical Phase 1 dataset schema."""

    component_name = "schema"

    def __init__(
        self,
        options: SchemaBuilderOptions | None = None,
    ) -> None:
        """Initialize the schema builder."""

        self.options = options or SchemaBuilderOptions()

    def build(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> dict[str, Any]:
        """Build and register the canonical schema."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Schema construction requires a pandas DataFrame."
            )

        if not self.options.enabled:
            return {}

        self._ensure_column_states(
            data=data,
            result=result,
            state=state,
        )

        required_columns = set(
            self.options.required_columns
        )

        optional_columns = set(
            self.options.optional_columns
        )

        schema_columns: list[dict[str, Any]] = []
        canonical_order: list[str] = []

        result.column_inventory.clear()

        for position, raw_column in enumerate(data.columns):
            column = str(raw_column)

            column_state = state.column_states.get(column)

            if column_state is None:
                continue

            if (
                not self.options.include_internal_columns
                and ColumnRole.INTERNAL
                in column_state.roles
            ):
                continue

            if column in optional_columns:
                column_state.required = False
            elif column in required_columns:
                column_state.required = True

            state.register_column_state(column_state)

            missing_count = int(
                data[column].isna().sum()
            )

            missing_percentage = self._percentage(
                affected_count=missing_count,
                total_count=len(data),
            )

            unique_count = int(
                data[column].nunique(dropna=True)
            )

            non_missing_count = int(
                data[column].notna().sum()
            )

            uniqueness_ratio = (
                unique_count / non_missing_count
                if non_missing_count > 0
                else 0.0
            )

            memory_usage = int(
                data[column].memory_usage(
                    index=False,
                    deep=True,
                )
            )

            column_state.missing_count = missing_count
            column_state.missing_percentage = (
                missing_percentage
            )
            column_state.unique_count = unique_count
            column_state.uniqueness_ratio = (
                uniqueness_ratio
            )

            description = self.options.descriptions.get(
                column
            )

            if description is not None:
                column_state.metadata[
                    "description"
                ] = description

            state.register_column_state(column_state)

            schema_column = self._build_schema_column(
                position=position,
                column_state=column_state,
                pandas_dtype=str(data[column].dtype),
                memory_usage_bytes=memory_usage,
            )

            schema_columns.append(schema_column)
            canonical_order.append(column)

            result.add_column_inventory_item(
                ColumnInventoryItem(
                    original_name=(
                        column_state.original_name
                    ),
                    normalized_name=(
                        column_state.normalized_name
                    ),
                    roles=sorted(
                        role.value
                        for role in column_state.roles
                    ),
                    physical_type=(
                        column_state.physical_type.value
                    ),
                    semantic_type=(
                        column_state.semantic_type.value
                    ),
                    semantic_type_confidence=(
                        column_state
                        .semantic_type_confidence
                    ),
                    required=column_state.required,
                    nullable=column_state.nullable,
                    use_as_model_feature=(
                        column_state
                        .use_as_model_feature
                    ),
                    available_at_inference=(
                        column_state
                        .available_at_inference
                    ),
                    is_sensitive=(
                        column_state.is_sensitive
                    ),
                    is_constant=(
                        column_state.is_constant
                    ),
                    is_near_constant=(
                        column_state.is_near_constant
                    ),
                    is_identifier_like=(
                        column_state.is_identifier_like
                    ),
                    missing_count=missing_count,
                    missing_percentage=(
                        missing_percentage
                    ),
                    unique_count=unique_count,
                    uniqueness_ratio=(
                        uniqueness_ratio
                    ),
                    memory_usage_bytes=memory_usage,
                    metadata=dict(
                        column_state.metadata
                    ),
                )
            )

        state.finalize_column_orders(
            canonical_column_order=canonical_order
        )

        canonical_schema = {
            "schema_version": "0.1.0",
            "row_count": len(data),
            "column_count": len(schema_columns),
            "column_order": canonical_order,
            "columns": schema_columns,
        }

        result.canonical_schema = canonical_schema

        self._update_dataset_summary(
            result=result,
            state=state,
        )

        return canonical_schema

    def _ensure_column_states(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Ensure every DataFrame column has registered state."""

        missing_state_columns = [
            str(column)
            for column in data.columns
            if str(column) not in state.column_states
        ]

        if not missing_state_columns:
            return

        severity = (
            Severity.ERROR
            if self.options.require_state_for_every_column
            else Severity.WARNING
        )

        result.add_finding(
            Finding(
                code="P1-SCHEMA-001",
                category=FindingCategory.SCHEMA,
                severity=severity,
                component=self.component_name,
                message=(
                    "One or more dataset columns do not have "
                    "registered Phase 1 state."
                ),
                columns=missing_state_columns,
                recommended_action=FindingAction.RAISE,
                applied_action=FindingAction.REPORT,
                blocks_readiness=(
                    self.options
                    .require_state_for_every_column
                ),
            )
        )

        if self.options.require_state_for_every_column:
            return

        for column in missing_state_columns:
            column_state = ColumnState(
                original_name=column,
                normalized_name=column,
                roles={ColumnRole.UNKNOWN},
                physical_type=PhysicalType.UNKNOWN,
                semantic_type=SemanticType.UNKNOWN,
                dtype_before=str(data[column].dtype),
                dtype_after=str(data[column].dtype),
            )

            state.register_column_state(column_state)

    @staticmethod
    def _build_schema_column(
        *,
        position: int,
        column_state: ColumnState,
        pandas_dtype: str,
        memory_usage_bytes: int,
    ) -> dict[str, Any]:
        """Build one serializable schema-column entry."""

        return {
            "position": position,
            "original_name": (
                column_state.original_name
            ),
            "name": column_state.normalized_name,
            "roles": sorted(
                role.value
                for role in column_state.roles
            ),
            "pandas_dtype": pandas_dtype,
            "physical_type": (
                column_state.physical_type.value
            ),
            "semantic_type": (
                column_state.semantic_type.value
            ),
            "semantic_type_confidence": (
                column_state.semantic_type_confidence
            ),
            "required": column_state.required,
            "nullable": column_state.nullable,
            "use_as_model_feature": (
                column_state.use_as_model_feature
            ),
            "available_at_inference": (
                column_state.available_at_inference
            ),
            "sensitive": column_state.is_sensitive,
            "sensitive_type": (
                column_state.sensitive_type
            ),
            "constant": column_state.is_constant,
            "near_constant": (
                column_state.is_near_constant
            ),
            "identifier_like": (
                column_state.is_identifier_like
            ),
            "missing_count": (
                column_state.missing_count
            ),
            "missing_percentage": (
                column_state.missing_percentage
            ),
            "unique_count": (
                column_state.unique_count
            ),
            "uniqueness_ratio": (
                column_state.uniqueness_ratio
            ),
            "memory_usage_bytes": (
                memory_usage_bytes
            ),
            "metadata": dict(
                column_state.metadata
            ),
        }

    @staticmethod
    def _update_dataset_summary(
        *,
        result: Phase1Result,
        state: Phase1State,
    ) -> None:
        """Update dataset summary using resolved column states."""

        result.dataset_summary.feature_count = len(
            state.get_model_feature_columns()
        )

        result.dataset_summary.identifier_count = len(
            state.get_columns_by_role(
                ColumnRole.IDENTIFIER
            )
        )

        timestamp_columns = set(
            state.get_columns_by_role(
                ColumnRole.TIMESTAMP
            )
        )

        timestamp_columns.update(
            state.get_columns_by_role(
                ColumnRole.PREDICTION_TIMESTAMP
            )
        )

        timestamp_columns.update(
            state.get_columns_by_role(
                ColumnRole.OUTCOME_TIMESTAMP
            )
        )

        result.dataset_summary.timestamp_count = len(
            timestamp_columns
        )

        result.dataset_summary.sensitive_column_count = len(
            state.get_sensitive_columns()
        )

        result.dataset_summary.ignored_column_count = len(
            state.get_columns_by_role(
                ColumnRole.IGNORED
            )
        )

    @staticmethod
    def _percentage(
        *,
        affected_count: int,
        total_count: int,
    ) -> float:
        """Calculate a percentage safely."""

        if total_count <= 0:
            return 0.0

        return round(
            affected_count / total_count * 100.0,
            6,
        )