"""Feature-lineage tracking for Phase 6."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FeatureLineageEntry(BaseModel):
    """Lineage information for one output feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    output_column: str

    source_columns: list[str] = Field(
        default_factory=list
    )

    transformation_steps: list[str] = Field(
        default_factory=list
    )

    feature_family: str

    generated: bool = True
    source_preserved: bool = False

    fitted_on_training_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class FeatureLineageState(BaseModel):
    """Complete Phase 6 feature-lineage state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    entries: dict[
        str,
        FeatureLineageEntry,
    ] = Field(default_factory=dict)

    source_to_outputs: dict[
        str,
        list[str],
    ] = Field(default_factory=dict)

    output_to_sources: dict[
        str,
        list[str],
    ] = Field(default_factory=dict)

    source_columns: list[str] = Field(
        default_factory=list
    )

    output_columns: list[str] = Field(
        default_factory=list
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class FeatureLineageTracker:
    """Track deterministic feature derivation."""

    component_name = "feature_lineage"

    def __init__(
        self,
        *,
        source_columns: list[str] | None = None,
    ) -> None:
        """Initialize the lineage tracker."""

        self.state = FeatureLineageState(
            source_columns=list(
                source_columns or []
            )
        )

    def register(
        self,
        *,
        output_column: str,
        source_columns: list[str],
        transformation_steps: list[str],
        feature_family: str,
        generated: bool = True,
        source_preserved: bool = False,
        fitted_on_training_only: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register lineage for one output feature."""

        if not output_column.strip():
            raise ValueError(
                "output_column cannot be empty."
            )

        normalized_sources = [
            str(column)
            for column in source_columns
        ]

        if not normalized_sources:
            raise ValueError(
                "At least one source column is required."
            )

        if output_column in self.state.entries:
            raise ValueError(
                "Feature lineage already exists for "
                f"{output_column!r}."
            )

        normalized_steps = [
            str(step)
            for step in transformation_steps
        ]

        if not normalized_steps:
            raise ValueError(
                "At least one transformation step is required."
            )

        entry = FeatureLineageEntry(
            output_column=output_column,
            source_columns=(
                normalized_sources
            ),
            transformation_steps=(
                normalized_steps
            ),
            feature_family=feature_family,
            generated=generated,
            source_preserved=(
                source_preserved
            ),
            fitted_on_training_only=(
                fitted_on_training_only
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        self.state.entries[
            output_column
        ] = entry

        self.state.output_to_sources[
            output_column
        ] = list(
            normalized_sources
        )

        for source_column in normalized_sources:
            outputs = (
                self.state
                .source_to_outputs
                .setdefault(
                    source_column,
                    [],
                )
            )

            if output_column not in outputs:
                outputs.append(
                    output_column
                )

        if (
            output_column
            not in self.state.output_columns
        ):
            self.state.output_columns.append(
                output_column
            )

    def register_many(
        self,
        *,
        output_columns: list[str],
        source_column: str,
        transformation_steps: list[str],
        feature_family: str,
        source_preserved: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register multiple outputs from one source."""

        for output_column in output_columns:
            self.register(
                output_column=output_column,
                source_columns=[
                    source_column
                ],
                transformation_steps=(
                    transformation_steps
                ),
                feature_family=(
                    feature_family
                ),
                generated=True,
                source_preserved=(
                    source_preserved
                ),
                fitted_on_training_only=True,
                metadata=metadata,
            )

    def register_passthrough(
        self,
        *,
        column: str,
        feature_family: str,
    ) -> None:
        """Register one unchanged passthrough feature."""

        self.register(
            output_column=column,
            source_columns=[
                column
            ],
            transformation_steps=[
                "passthrough"
            ],
            feature_family=feature_family,
            generated=False,
            source_preserved=True,
            fitted_on_training_only=True,
        )

    def outputs_for(
        self,
        source_column: str,
    ) -> list[object]:
        """Return output features generated from one source."""

        return list(
            self.state
            .source_to_outputs
            .get(
                source_column,
                [],
            )
        )

    def sources_for(
        self,
        output_column: str,
    ) -> list[object]:
        """Return source features for one output."""

        return list(
            self.state
            .output_to_sources
            .get(
                output_column,
                [],
            )
        )

    def validate_output_schema(
        self,
        output_columns: list[str],
    ) -> None:
        """Validate lineage against an output schema."""

        normalized_output = [
            str(column)
            for column in output_columns
        ]

        lineage_columns = list(
            self.state.output_columns
        )

        if normalized_output != lineage_columns:
            missing_lineage = [
                column
                for column in normalized_output
                if column not in lineage_columns
            ]

            extra_lineage = [
                column
                for column in lineage_columns
                if column not in normalized_output
            ]

            raise ValueError(
                "Feature lineage differs from the output schema. "
                f"Missing lineage: {missing_lineage}; "
                f"extra lineage: {extra_lineage}."
            )

    def get_state(
        self,
    ) -> FeatureLineageState:
        """Return a deep copy of lineage state."""

        return self.state.model_copy(
            deep=True
        )

    def to_dictionary(
        self,
    ) -> dict[str, Any]:
        """Return JSON-compatible lineage data."""

        return self.state.model_dump(
            mode="json"
        )
