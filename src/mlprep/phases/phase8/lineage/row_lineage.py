"""Training-row lineage tracking for Phase 8."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase8.samplers.base import (
    RemovedRowRecord,
    ResampledRowType,
    SamplerResult,
    SamplerRowRecord,
)


class RowLineageEntry(BaseModel):
    """Auditable lineage for one output training row."""

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

    strategy: str

    original: bool = False
    duplicated: bool = False
    synthetic: bool = False
    retained_after_cleaning: bool = False

    generation_sequence: int | None = Field(
        default=None,
        ge=1,
    )

    feature_values_verified: bool = False
    target_value_verified: bool = False
    row_id_verified: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class RemovedRowLineageEntry(BaseModel):
    """Auditable lineage for one removed original row."""

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
    strategy: str

    source_feature_fingerprint: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class RowLineageState(BaseModel):
    """Complete Phase 8 row-lineage state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    strategy: str

    source_row_count: int = Field(
        default=0,
        ge=0,
    )

    output_row_count: int = Field(
        default=0,
        ge=0,
    )

    entries: list[
        RowLineageEntry
    ] = Field(
        default_factory=list,
    )

    removed_rows: list[
        RemovedRowLineageEntry
    ] = Field(
        default_factory=list,
    )

    output_row_ids: list[str] = Field(
        default_factory=list,
    )

    retained_original_row_ids: list[str] = Field(
        default_factory=list,
    )

    duplicated_row_ids: list[str] = Field(
        default_factory=list,
    )

    synthetic_row_ids: list[str] = Field(
        default_factory=list,
    )

    removed_original_row_ids: list[str] = Field(
        default_factory=list,
    )

    row_type_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict,
    )

    complete_output_coverage: bool = False
    complete_removed_row_coverage: bool = False

    row_ids_unique: bool = False
    row_ids_nonmissing: bool = False

    original_ids_preserved: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class RowLineageTracker:
    """Validate and store Phase 8 training-row lineage."""

    component_name = "row_lineage"

    def __init__(
        self,
        *,
        strategy: str,
    ) -> None:
        """Initialize the row-lineage tracker."""

        normalized_strategy = str(
            strategy
        ).strip()

        if not normalized_strategy:
            raise ValueError(
                "strategy cannot be empty."
            )

        self.strategy = (
            normalized_strategy
        )

        self.state: RowLineageState | None = None

    def build(
        self,
        *,
        source_features: pd.DataFrame,
        source_target: pd.Series,
        source_row_ids: pd.Series,
        sampler_result: SamplerResult,
    ) -> RowLineageState:
        """Build and validate complete resampling lineage."""

        self._validate_source(
            source_features=source_features,
            source_target=source_target,
            source_row_ids=source_row_ids,
        )

        self._validate_sampler_result(
            sampler_result=sampler_result
        )

        source_ids = [
            str(
                value
            )
            for value in source_row_ids.tolist()
        ]

        source_id_to_position = {
            row_id: position
            for position, row_id
            in enumerate(
                source_ids
            )
        }

        output_ids = [
            str(
                value
            )
            for value
            in sampler_result.row_ids.tolist()
        ]

        output_entries = [
            self._build_output_entry(
                record=record,
                sampler_result=sampler_result,
                source_features=source_features,
                source_target=source_target,
                source_row_ids=source_row_ids,
                source_id_to_position=(
                    source_id_to_position
                ),
            )
            for record
            in sampler_result.row_lineage
        ]

        removed_entries = [
            self._build_removed_entry(
                record=record,
                source_features=source_features,
                source_target=source_target,
                source_row_ids=source_row_ids,
            )
            for record
            in sampler_result.removed_rows
        ]

        self._validate_output_entries(
            entries=output_entries,
            output_row_ids=output_ids,
            output_target=(
                sampler_result.target
            ),
        )

        self._validate_removed_entries(
            removed_entries=removed_entries,
            source_row_ids=source_ids,
        )

        row_type_counts = Counter(
            entry.row_type.value
            for entry in output_entries
        )

        retained_original_row_ids = [
            entry.output_row_id
            for entry in output_entries
            if entry.row_type
            in {
                ResampledRowType.ORIGINAL,
                ResampledRowType
                .RETAINED_AFTER_CLEANING,
            }
        ]

        duplicated_row_ids = [
            entry.output_row_id
            for entry in output_entries
            if (
                entry.row_type
                == ResampledRowType.DUPLICATE
            )
        ]

        synthetic_row_ids = [
            entry.output_row_id
            for entry in output_entries
            if (
                entry.row_type
                == ResampledRowType.SYNTHETIC
            )
        ]

        removed_original_row_ids = [
            entry.row_id
            for entry in removed_entries
        ]

        complete_output_coverage = bool(
            len(
                output_entries
            )
            == len(
                sampler_result.features
            )
            and [
                entry.output_position
                for entry in output_entries
            ]
            == list(
                range(
                    len(
                        sampler_result.features
                    )
                )
            )
        )

        expected_removed_ids = {
            row_id
            for row_id in source_ids
            if row_id
            not in set(
                retained_original_row_ids
            )
        }

        complete_removed_coverage = bool(
            expected_removed_ids
            == set(
                removed_original_row_ids
            )
        )

        original_ids_preserved = all(
            entry.output_row_id
            == entry.source_row_ids[
                0
            ]
            for entry in output_entries
            if (
                entry.row_type
                in {
                    ResampledRowType.ORIGINAL,
                    ResampledRowType
                    .RETAINED_AFTER_CLEANING,
                }
                and entry.source_row_ids
            )
        )

        state = RowLineageState(
            strategy=self.strategy,
            source_row_count=len(
                source_features
            ),
            output_row_count=len(
                sampler_result.features
            ),
            entries=output_entries,
            removed_rows=removed_entries,
            output_row_ids=output_ids,
            retained_original_row_ids=(
                retained_original_row_ids
            ),
            duplicated_row_ids=(
                duplicated_row_ids
            ),
            synthetic_row_ids=(
                synthetic_row_ids
            ),
            removed_original_row_ids=(
                removed_original_row_ids
            ),
            row_type_counts={
                row_type: int(
                    count
                )
                for row_type, count
                in row_type_counts.items()
            },
            complete_output_coverage=(
                complete_output_coverage
            ),
            complete_removed_row_coverage=(
                complete_removed_coverage
            ),
            row_ids_unique=bool(
                sampler_result.row_ids.is_unique
            ),
            row_ids_nonmissing=bool(
                not sampler_result
                .row_ids
                .isna()
                .any()
            ),
            original_ids_preserved=(
                original_ids_preserved
            ),
            metadata={
                "source_ids_unique": bool(
                    source_row_ids.is_unique
                ),
                "source_ids_nonmissing": bool(
                    not source_row_ids
                    .isna()
                    .any()
                ),
                "synthetic_count": len(
                    synthetic_row_ids
                ),
                "duplicate_count": len(
                    duplicated_row_ids
                ),
                "removed_original_count": len(
                    removed_original_row_ids
                ),
            },
        )

        if not state.complete_output_coverage:
            raise ValueError(
                "Row lineage does not cover every output "
                "training row."
            )

        if not state.complete_removed_row_coverage:
            raise ValueError(
                "Removed-row lineage does not cover every "
                "removed original training row."
            )

        if not state.row_ids_unique:
            raise ValueError(
                "Output row IDs are not unique."
            )

        if not state.row_ids_nonmissing:
            raise ValueError(
                "Output row IDs contain missing values."
            )

        if not state.original_ids_preserved:
            raise ValueError(
                "One or more retained original rows did not "
                "preserve their original row IDs."
            )

        self.state = state

        return state.model_copy(
            deep=True
        )

    def get_state(
        self,
    ) -> RowLineageState:
        """Return the latest row-lineage state."""

        if self.state is None:
            raise RuntimeError(
                "Row lineage has not been built."
            )

        return self.state.model_copy(
            deep=True
        )

    def output_entry(
        self,
        row_id: object,
    ) -> RowLineageEntry:
        """Return lineage for one output row identifier."""

        state = self.get_state()

        normalized = str(
            row_id
        )

        for entry in state.entries:
            if entry.output_row_id == normalized:
                return entry.model_copy(
                    deep=True
                )

        raise KeyError(
            f"Output row ID {normalized!r} is unavailable."
        )

    def removed_entry(
        self,
        row_id: object,
    ) -> RemovedRowLineageEntry:
        """Return lineage for one removed original row."""

        state = self.get_state()

        normalized = str(
            row_id
        )

        for entry in state.removed_rows:
            if entry.row_id == normalized:
                return entry.model_copy(
                    deep=True
                )

        raise KeyError(
            f"Removed row ID {normalized!r} is unavailable."
        )

    def _build_output_entry(
        self,
        *,
        record: SamplerRowRecord,
        sampler_result: SamplerResult,
        source_features: pd.DataFrame,
        source_target: pd.Series,
        source_row_ids: pd.Series,
        source_id_to_position: dict[str, int],
    ) -> RowLineageEntry:
        """Build one validated output-lineage entry."""

        position = (
            record.output_position
        )

        if position >= len(
            sampler_result.features
        ):
            raise IndexError(
                "Lineage output position exceeds the "
                "resampled row count."
            )

        output_row_id = str(
            sampler_result.row_ids.iloc[
                position
            ]
        )

        row_id_verified = bool(
            output_row_id
            == record.output_row_id
        )

        if not row_id_verified:
            raise ValueError(
                "Sampler lineage row ID differs from the "
                f"output row ID at position {position}."
            )

        output_class = (
            sampler_result.target.iloc[
                position
            ]
        )

        target_verified = (
            self._values_equal(
                output_class,
                record.class_value,
            )
        )

        if not target_verified:
            raise ValueError(
                "Sampler lineage target differs from the "
                f"output target at position {position}."
            )

        source_positions = list(
            record.source_positions
        )

        source_ids = [
            str(
                row_id
            )
            for row_id
            in record.source_row_ids
        ]

        if (
            not source_positions
            and source_ids
        ):
            source_positions = [
                source_id_to_position[
                    source_id
                ]
                for source_id in source_ids
                if source_id
                in source_id_to_position
            ]

        if (
            not source_ids
            and source_positions
        ):
            source_ids = [
                str(
                    source_row_ids.iloc[
                        source_position
                    ]
                )
                for source_position
                in source_positions
            ]

        feature_values_verified = (
            self._verify_feature_values(
                record=record,
                output_position=position,
                source_positions=source_positions,
                source_features=source_features,
                output_features=(
                    sampler_result.features
                ),
            )
        )

        if (
            record.row_type
            in {
                ResampledRowType.ORIGINAL,
                ResampledRowType.DUPLICATE,
                ResampledRowType
                .RETAINED_AFTER_CLEANING,
            }
            and not feature_values_verified
        ):
            raise ValueError(
                "Original or duplicate output values do not "
                "match their recorded source row."
            )

        if (
            record.row_type
            in {
                ResampledRowType.ORIGINAL,
                ResampledRowType.DUPLICATE,
                ResampledRowType
                .RETAINED_AFTER_CLEANING,
            }
            and source_positions
        ):
            source_class = source_target.iloc[
                source_positions[
                    0
                ]
            ]

            if not self._values_equal(
                source_class,
                output_class,
            ):
                raise ValueError(
                    "Original or duplicate row target does not "
                    "match the recorded source target."
                )

        return RowLineageEntry(
            output_position=position,
            output_row_id=output_row_id,
            row_type=record.row_type,
            class_value=output_class,
            source_row_ids=source_ids,
            source_positions=source_positions,
            strategy=self.strategy,
            original=bool(
                record.row_type
                == ResampledRowType.ORIGINAL
            ),
            duplicated=bool(
                record.row_type
                == ResampledRowType.DUPLICATE
            ),
            synthetic=bool(
                record.row_type
                == ResampledRowType.SYNTHETIC
            ),
            retained_after_cleaning=bool(
                record.row_type
                == ResampledRowType
                .RETAINED_AFTER_CLEANING
            ),
            generation_sequence=(
                record.generation_sequence
            ),
            feature_values_verified=(
                feature_values_verified
            ),
            target_value_verified=(
                target_verified
            ),
            row_id_verified=(
                row_id_verified
            ),
            metadata=dict(
                record.metadata
            ),
        )

    def _build_removed_entry(
        self,
        *,
        record: RemovedRowRecord,
        source_features: pd.DataFrame,
        source_target: pd.Series,
        source_row_ids: pd.Series,
    ) -> RemovedRowLineageEntry:
        """Build one validated removed-row entry."""

        position = (
            record.source_position
        )

        if position >= len(
            source_features
        ):
            raise IndexError(
                "Removed-row source position exceeds the "
                "source training row count."
            )

        source_row_id = str(
            source_row_ids.iloc[
                position
            ]
        )

        if source_row_id != record.row_id:
            raise ValueError(
                "Removed-row lineage ID differs from the "
                "source row ID."
            )

        source_class = source_target.iloc[
            position
        ]

        if not self._values_equal(
            source_class,
            record.class_value,
        ):
            raise ValueError(
                "Removed-row lineage target differs from "
                "the source target."
            )

        fingerprint = self._row_fingerprint(
            source_features.iloc[
                position
            ]
        )

        return RemovedRowLineageEntry(
            row_id=source_row_id,
            source_position=position,
            class_value=source_class,
            reason=record.reason,
            stage=record.stage,
            strategy=self.strategy,
            source_feature_fingerprint=(
                fingerprint
            ),
            metadata=dict(
                record.metadata
            ),
        )

    @staticmethod
    def _verify_feature_values(
        *,
        record: SamplerRowRecord,
        output_position: int,
        source_positions: list[int],
        source_features: pd.DataFrame,
        output_features: pd.DataFrame,
    ) -> bool:
        """Verify feature values for traceable rows."""

        if (
            record.row_type
            == ResampledRowType.SYNTHETIC
        ):
            return True

        if not source_positions:
            return False

        source_position = (
            source_positions[
                0
            ]
        )

        if source_position >= len(
            source_features
        ):
            return False

        source_row = source_features.iloc[
            source_position
        ]

        output_row = output_features.iloc[
            output_position
        ]

        return source_row.equals(
            output_row
        )

    @staticmethod
    def _validate_output_entries(
        *,
        entries: list[
            RowLineageEntry
        ],
        output_row_ids: list[str],
        output_target: pd.Series,
    ) -> None:
        """Validate output-lineage sequence and IDs."""

        if len(
            entries
        ) != len(
            output_row_ids
        ):
            raise ValueError(
                "Output lineage and output row IDs have "
                "different lengths."
            )

        positions = [
            entry.output_position
            for entry in entries
        ]

        if positions != list(
            range(
                len(
                    entries
                )
            )
        ):
            raise ValueError(
                "Output lineage positions must form a "
                "complete ordered sequence."
            )

        lineage_ids = [
            entry.output_row_id
            for entry in entries
        ]

        if lineage_ids != output_row_ids:
            raise ValueError(
                "Output lineage IDs differ from output row IDs."
            )

        if len(
            lineage_ids
        ) != len(
            set(
                lineage_ids
            )
        ):
            raise ValueError(
                "Output lineage contains duplicate row IDs."
            )

        if len(
            output_target
        ) != len(
            entries
        ):
            raise ValueError(
                "Output lineage and target have different "
                "row counts."
            )

    @staticmethod
    def _validate_removed_entries(
        *,
        removed_entries: list[
            RemovedRowLineageEntry
        ],
        source_row_ids: list[str],
    ) -> None:
        """Validate removed-row lineage IDs."""

        removed_ids = [
            entry.row_id
            for entry in removed_entries
        ]

        if len(
            removed_ids
        ) != len(
            set(
                removed_ids
            )
        ):
            raise ValueError(
                "Removed-row lineage contains duplicate IDs."
            )

        unknown_ids = sorted(
            set(
                removed_ids
            ).difference(
                source_row_ids
            )
        )

        if unknown_ids:
            raise ValueError(
                "Removed-row lineage contains unknown "
                f"source IDs: {unknown_ids}"
            )

    @staticmethod
    def _validate_source(
        *,
        source_features: pd.DataFrame,
        source_target: pd.Series,
        source_row_ids: pd.Series,
    ) -> None:
        """Validate source training data."""

        if not isinstance(
            source_features,
            pd.DataFrame,
        ):
            raise TypeError(
                "source_features must be a pandas DataFrame."
            )

        if not isinstance(
            source_target,
            pd.Series,
        ):
            raise TypeError(
                "source_target must be a pandas Series."
            )

        if not isinstance(
            source_row_ids,
            pd.Series,
        ):
            raise TypeError(
                "source_row_ids must be a pandas Series."
            )

        if not (
            len(
                source_features
            )
            == len(
                source_target
            )
            == len(
                source_row_ids
            )
        ):
            raise ValueError(
                "Source features, target, and row IDs have "
                "different row counts."
            )

        if (
            not source_features.index.equals(
                source_target.index
            )
            or not source_features.index.equals(
                source_row_ids.index
            )
        ):
            raise ValueError(
                "Source features, target, and row IDs have "
                "different indices."
            )

        if source_row_ids.isna().any():
            raise ValueError(
                "Source row IDs contain missing values."
            )

        if not source_row_ids.is_unique:
            raise ValueError(
                "Source row IDs must be unique."
            )

    @staticmethod
    def _validate_sampler_result(
        *,
        sampler_result: SamplerResult,
    ) -> None:
        """Validate the sampler result shape."""

        if not isinstance(
            sampler_result,
            SamplerResult,
        ):
            raise TypeError(
                "sampler_result must be a SamplerResult."
            )

        output_count = len(
            sampler_result.features
        )

        if not (
            output_count
            == len(
                sampler_result.target
            )
            == len(
                sampler_result.row_ids
            )
            == len(
                sampler_result.row_lineage
            )
        ):
            raise ValueError(
                "Sampler output components have different "
                "row counts."
            )

    @staticmethod
    def _values_equal(
        left: object,
        right: object,
    ) -> bool:
        """Safely compare two target values."""

        try:
            if (
                pd.isna(
                    left
                )
                and pd.isna(
                    right
                )
            ):
                return True

        except (
            TypeError,
            ValueError,
        ):
            pass

        try:
            comparison = (
                left == right
            )

        except Exception:
            return False

        if isinstance(
            comparison,
            bool,
        ):
            return comparison

        try:
            return bool(
                comparison
            )

        except (
            TypeError,
            ValueError,
        ):
            return False

    @staticmethod
    def _row_fingerprint(
        row: pd.Series,
    ) -> str:
        """Generate a deterministic source-row fingerprint."""

        hashes = pd.util.hash_pandas_object(
            row,
            index=True,
            categorize=True,
        )

        return "".join(
            f"{int(value):016x}"
            for value in hashes.to_numpy()
        )