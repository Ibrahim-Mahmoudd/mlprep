"""Stable internal row-identity creation for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    NormalizationEvent,
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

ExistingColumnPolicy = Literal[
    "raise",
    "validate",
    "replace",
]


@dataclass(slots=True)
class RowIdentityOptions:
    """Options controlling internal row-identity creation."""

    enabled: bool = True

    column_name: str = "__mlprep_row_id__"

    existing_column_policy: ExistingColumnPolicy = "raise"

    preserve_original_index: bool = True

    require_unique_values: bool = True
    require_nonmissing_values: bool = True

    start_value: int = 0

    def __post_init__(self) -> None:
        """Validate row-identity options."""

        if not self.column_name.strip():
            raise ValueError(
                "column_name cannot be empty."
            )

        if self.start_value < 0:
            raise ValueError(
                "start_value cannot be negative."
            )


class RowIdentityManager:
    """Create and validate an immutable internal row identifier."""

    component_name = "row_identity"

    def __init__(
        self,
        options: RowIdentityOptions | None = None,
    ) -> None:
        """Initialize the row-identity manager."""

        self.options = options or RowIdentityOptions()

    def create(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Create or validate the internal row-ID column."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Row-identity creation requires a pandas DataFrame."
            )

        working_data = data.copy()

        if not self.options.enabled:
            return working_data

        column_name = self.options.column_name

        result.internal_row_id_column = column_name
        state.internal_row_id_column = column_name

        if column_name in working_data.columns:
            return self._handle_existing_column(
                data=working_data,
                result=result,
                state=state,
            )

        row_ids = pd.Series(
            range(
                self.options.start_value,
                self.options.start_value + len(working_data),
            ),
            index=working_data.index,
            dtype="int64",
            name=column_name,
        )

        working_data.insert(
            loc=0,
            column=column_name,
            value=row_ids,
        )

        self._register_state(
            data=working_data,
            state=state,
        )

        result.add_normalization_event(
            NormalizationEvent(
                component=self.component_name,
                operation="create_internal_row_id",
                description=(
                    f"Created internal row-identity column "
                    f"{column_name!r}."
                ),
                column=column_name,
                affected_row_count=len(working_data),
                before_summary={
                    "column_existed": False,
                },
                after_summary={
                    "column_existed": True,
                    "dtype": str(
                        working_data[column_name].dtype
                    ),
                    "unique": True,
                    "nullable": False,
                },
                reversible=True,
                metadata={
                    "start_value": self.options.start_value,
                    "preserved_original_index": (
                        self.options.preserve_original_index
                    ),
                },
            )
        )

        result.add_finding(
            Finding(
                code="P1-ROWID-001",
                category=FindingCategory.SCHEMA,
                severity=Severity.INFO,
                component=self.component_name,
                message=(
                    f"Internal row identity was created in "
                    f"column {column_name!r}."
                ),
                columns=[column_name],
                affected_row_count=len(working_data),
                recommended_action=FindingAction.COERCE,
                applied_action=FindingAction.COERCE,
                blocks_readiness=False,
            )
        )

        if self.options.preserve_original_index:
            state.metadata["original_index"] = [
                self._serialize_index_value(index)
                for index in working_data.index
            ]

            state.metadata["original_index_name"] = (
                str(working_data.index.name)
                if working_data.index.name is not None
                else None
            )

        return working_data

    def validate(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> bool:
        """Validate the internal row-ID column."""

        column_name = self.options.column_name

        if column_name not in data.columns:
            result.add_finding(
                Finding(
                    code="P1-ROWID-002",
                    category=FindingCategory.SCHEMA,
                    severity=Severity.CRITICAL,
                    component=self.component_name,
                    message=(
                        f"Internal row-ID column "
                        f"{column_name!r} does not exist."
                    ),
                    columns=[column_name],
                    recommended_action=FindingAction.RAISE,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=True,
                )
            )

            return False

        is_valid = True

        if self.options.require_nonmissing_values:
            missing_mask = data[column_name].isna()
            missing_count = int(missing_mask.sum())

            if missing_count > 0:
                result.add_finding(
                    Finding(
                        code="P1-ROWID-003",
                        category=FindingCategory.SCHEMA,
                        severity=Severity.CRITICAL,
                        component=self.component_name,
                        message=(
                            "The internal row-ID column contains "
                            "missing values."
                        ),
                        columns=[column_name],
                        affected_row_count=missing_count,
                        affected_row_percentage=self._percentage(
                            affected_count=missing_count,
                            total_count=len(data),
                        ),
                        example_row_ids=[
                            self._serialize_index_value(index)
                            for index in data.index[
                                missing_mask
                            ].tolist()[:5]
                        ],
                        recommended_action=FindingAction.RAISE,
                        applied_action=FindingAction.REPORT,
                        blocks_readiness=True,
                    )
                )

                is_valid = False

        if self.options.require_unique_values:
            duplicate_mask = data[column_name].duplicated(
                keep=False
            )

            duplicate_count = int(
                duplicate_mask.sum()
            )

            if duplicate_count > 0:
                result.add_finding(
                    Finding(
                        code="P1-ROWID-004",
                        category=FindingCategory.SCHEMA,
                        severity=Severity.CRITICAL,
                        component=self.component_name,
                        message=(
                            "The internal row-ID column contains "
                            "duplicate values."
                        ),
                        columns=[column_name],
                        affected_row_count=duplicate_count,
                        affected_row_percentage=self._percentage(
                            affected_count=duplicate_count,
                            total_count=len(data),
                        ),
                        example_values=(
                            data.loc[
                                duplicate_mask,
                                column_name,
                            ]
                            .head(5)
                            .tolist()
                        ),
                        recommended_action=FindingAction.RAISE,
                        applied_action=FindingAction.REPORT,
                        blocks_readiness=True,
                    )
                )

                is_valid = False

        return is_valid

    def _handle_existing_column(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Handle a pre-existing internal row-ID column."""

        policy = self.options.existing_column_policy
        column_name = self.options.column_name

        if policy == "raise":
            result.add_finding(
                Finding(
                    code="P1-ROWID-005",
                    category=FindingCategory.SCHEMA,
                    severity=Severity.CRITICAL,
                    component=self.component_name,
                    message=(
                        f"Input data already contains the reserved "
                        f"internal column {column_name!r}."
                    ),
                    columns=[column_name],
                    recommended_action=FindingAction.RENAME,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=True,
                )
            )

            return data

        if policy == "validate":
            if self.validate(
                data=data,
                result=result,
            ):
                self._register_state(
                    data=data,
                    state=state,
                )

            return data

        replacement_ids = pd.Series(
            range(
                self.options.start_value,
                self.options.start_value + len(data),
            ),
            index=data.index,
            dtype="int64",
        )

        data[column_name] = replacement_ids

        self._register_state(
            data=data,
            state=state,
        )

        result.add_normalization_event(
            NormalizationEvent(
                component=self.component_name,
                operation="replace_internal_row_id",
                description=(
                    f"Replaced existing values in the internal "
                    f"row-ID column {column_name!r}."
                ),
                column=column_name,
                affected_row_count=len(data),
                before_summary={
                    "column_existed": True,
                },
                after_summary={
                    "dtype": str(data[column_name].dtype),
                    "unique": True,
                    "nullable": False,
                },
                reversible=False,
            )
        )

        return data

    def _register_state(
        self,
        *,
        data: pd.DataFrame,
        state: Phase1State,
    ) -> None:
        """Register the internal row-ID column state."""

        column_name = self.options.column_name

        column_state = ColumnState(
            original_name=column_name,
            normalized_name=column_name,
            roles={ColumnRole.INTERNAL},
            physical_type=PhysicalType.INTEGER,
            semantic_type=SemanticType.IDENTIFIER,
            semantic_type_confidence=1.0,
            required=True,
            nullable=False,
            use_as_model_feature=False,
            available_at_inference=False,
            is_sensitive=False,
            is_identifier_like=True,
            dtype_before=str(data[column_name].dtype),
            dtype_after=str(data[column_name].dtype),
            metadata={
                "system_generated": True,
                "immutable": True,
            },
        )

        state.register_column_state(
            column_state
        )

    @staticmethod
    def _serialize_index_value(
        value: object,
    ) -> str | int:
        """Convert an index value into a serializable identifier."""

        if isinstance(value, (str, int)):
            return value

        return str(value)

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