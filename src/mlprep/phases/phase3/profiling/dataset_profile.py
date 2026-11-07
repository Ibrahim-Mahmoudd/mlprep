"""Dataset-level profiling for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)


class DatasetProfile(BaseModel):
    """Dataset-level analytical profile."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    row_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)

    numeric_feature_count: int = Field(ge=0)
    categorical_feature_count: int = Field(ge=0)
    datetime_feature_count: int = Field(ge=0)
    text_feature_count: int = Field(ge=0)
    boolean_feature_count: int = Field(ge=0)
    timedelta_feature_count: int = Field(ge=0)

    identifier_count: int = Field(ge=0)
    sensitive_column_count: int = Field(ge=0)
    ignored_column_count: int = Field(ge=0)
    unknown_column_count: int = Field(ge=0)

    missing_cell_count: int = Field(ge=0)

    missing_cell_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    duplicate_feature_row_count: int = Field(
        ge=0
    )

    duplicate_feature_row_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    completely_missing_columns: list[str] = Field(
        default_factory=list
    )

    constant_columns: list[str] = Field(
        default_factory=list
    )

    near_constant_columns: list[str] = Field(
        default_factory=list
    )

    memory_usage_bytes: int = Field(ge=0)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class DatasetProfilerOptions:
    """Options controlling dataset-level profiling."""

    enabled: bool = True

    detect_duplicate_rows: bool = True
    detect_constant_columns: bool = True
    detect_near_constant_columns: bool = True

    near_constant_threshold: float = 0.995

    include_deep_memory_usage: bool = True

    def __post_init__(self) -> None:
        """Validate dataset-profiler options."""

        if not 0.0 <= self.near_constant_threshold <= 1.0:
            raise ValueError(
                "near_constant_threshold must be between "
                "zero and one."
            )


class DatasetProfiler:
    """Build a dataset-level profile from training data."""

    component_name = "dataset_profile"

    def __init__(
        self,
        options: DatasetProfilerOptions | None = None,
    ) -> None:
        """Initialize the dataset profiler."""

        self.options = (
            options or DatasetProfilerOptions()
        )

    def profile(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> DatasetProfile:
        """Profile the complete training feature matrix."""

        frame = contract.training_features()

        if not self.options.enabled:
            return self._empty_profile()

        row_count = len(frame)
        feature_count = len(frame.columns)

        cell_count = row_count * feature_count

        missing_cell_count = int(
            frame.isna().sum().sum()
        )

        missing_cell_percentage = (
            missing_cell_count / cell_count * 100.0
            if cell_count > 0
            else 0.0
        )

        duplicate_count = 0

        if self.options.detect_duplicate_rows:
            duplicate_count = int(
                frame.duplicated(
                    keep=False
                ).sum()
            )

        duplicate_percentage = (
            duplicate_count / row_count * 100.0
            if row_count > 0
            else 0.0
        )

        completely_missing_columns = [
            str(column)
            for column in frame.columns
            if frame[column].isna().all()
        ]

        constant_columns: list[str] = []
        near_constant_columns: list[str] = []

        for column in frame.columns:
            series = frame[column].dropna()

            if series.empty:
                continue

            unique_count = int(
                series.nunique(dropna=True)
            )

            if (
                self.options.detect_constant_columns
                and unique_count <= 1
            ):
                constant_columns.append(
                    str(column)
                )
                continue

            if not (
                self.options
                .detect_near_constant_columns
            ):
                continue

            frequencies = series.value_counts(
                normalize=True,
                dropna=False,
            )

            if (
                not frequencies.empty
                and float(frequencies.iloc[0])
                >= self.options
                .near_constant_threshold
            ):
                near_constant_columns.append(
                    str(column)
                )

        memory_usage = int(
            frame.memory_usage(
                index=True,
                deep=(
                    self.options
                    .include_deep_memory_usage
                ),
            ).sum()
        )

        return DatasetProfile(
            row_count=row_count,
            feature_count=feature_count,
            numeric_feature_count=len(
                columns.numeric
            ),
            categorical_feature_count=len(
                columns.categorical
            ),
            datetime_feature_count=len(
                columns.datetime
            ),
            text_feature_count=len(
                columns.text
            ),
            boolean_feature_count=len(
                columns.boolean
            ),
            timedelta_feature_count=len(
                columns.timedelta
            ),
            identifier_count=len(
                columns.identifiers
            ),
            sensitive_column_count=len(
                columns.sensitive
            ),
            ignored_column_count=len(
                columns.ignored
            ),
            unknown_column_count=len(
                columns.unknown
            ),
            missing_cell_count=missing_cell_count,
            missing_cell_percentage=round(
                missing_cell_percentage,
                6,
            ),
            duplicate_feature_row_count=(
                duplicate_count
            ),
            duplicate_feature_row_percentage=round(
                duplicate_percentage,
                6,
            ),
            completely_missing_columns=(
                completely_missing_columns
            ),
            constant_columns=constant_columns,
            near_constant_columns=(
                near_constant_columns
            ),
            memory_usage_bytes=memory_usage,
        )

    @staticmethod
    def _empty_profile() -> DatasetProfile:
        """Return an empty dataset profile."""

        return DatasetProfile(
            row_count=0,
            feature_count=0,
            numeric_feature_count=0,
            categorical_feature_count=0,
            datetime_feature_count=0,
            text_feature_count=0,
            boolean_feature_count=0,
            timedelta_feature_count=0,
            identifier_count=0,
            sensitive_column_count=0,
            ignored_column_count=0,
            unknown_column_count=0,
            missing_cell_count=0,
            missing_cell_percentage=0.0,
            duplicate_feature_row_count=0,
            duplicate_feature_row_percentage=0.0,
            memory_usage_bytes=0,
        )
