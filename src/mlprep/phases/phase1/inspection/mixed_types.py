"""Mixed runtime-type detection for Phase 1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import is_object_dtype

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.types import (
    FindingAction,
    FindingCategory,
    Severity,
)


@dataclass(slots=True)
class MixedTypeDetectionOptions:
    """Options controlling mixed runtime-type detection."""

    enabled: bool = True

    include_columns: list[str] | None = None
    exclude_columns: list[str] = field(
        default_factory=list
    )

    inspect_only_object_columns: bool = True

    sample_size: int | None = 50_000

    minimum_secondary_type_count: int = 1
    minimum_secondary_type_ratio: float = 0.0

    compatible_numeric_types: bool = True
    compatible_string_types: bool = True

    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate mixed-type detection options."""

        if (
            self.sample_size is not None
            and self.sample_size <= 0
        ):
            raise ValueError(
                "sample_size must be positive or None."
            )

        if self.minimum_secondary_type_count < 1:
            raise ValueError(
                "minimum_secondary_type_count must be at "
                "least one."
            )

        if not 0.0 <= self.minimum_secondary_type_ratio <= 1.0:
            raise ValueError(
                "minimum_secondary_type_ratio must be between "
                "zero and one."
            )

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )


class MixedTypeDetector:
    """Detect columns containing incompatible Python value types."""

    component_name = "mixed_types"

    def __init__(
        self,
        options: MixedTypeDetectionOptions | None = None,
    ) -> None:
        """Initialize the mixed-type detector."""

        self.options = (
            options or MixedTypeDetectionOptions()
        )

    def inspect(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> dict[str, dict[str, int]]:
        """Inspect selected columns for mixed runtime types."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Mixed-type detection requires a "
                "pandas DataFrame."
            )

        if not self.options.enabled:
            return {}

        findings_by_column: dict[
            str,
            dict[str, int],
        ] = {}

        for column in self._select_columns(data):
            series = data[column].dropna()

            if series.empty:
                continue

            sampled_series = self._sample_series(series)

            type_counts = Counter(
                self._type_group(value)
                for value in sampled_series
            )

            filtered_type_counts = {
                type_name: count
                for type_name, count in type_counts.items()
                if self._meets_threshold(
                    count=count,
                    sample_count=len(sampled_series),
                )
            }

            resolution = (
                state.type_resolution_states.get(column)
            )

            if resolution is not None:
                resolution.detected_python_types = sorted(
                    type_counts
                )

                resolution.inference_evidence[
                    "python_type_counts"
                ] = dict(type_counts)

                state.register_type_resolution(
                    resolution
                )

            if len(filtered_type_counts) <= 1:
                continue

            findings_by_column[column] = dict(
                filtered_type_counts
            )

            example_values = self._extract_examples(
                sampled_series
            )

            result.add_finding(
                Finding(
                    code="P1-MIXED-001",
                    category=FindingCategory.MIXED_TYPE,
                    severity=Severity.WARNING,
                    component=self.component_name,
                    message=(
                        f"Column {column!r} contains multiple "
                        "incompatible runtime value types."
                    ),
                    columns=[column],
                    affected_row_count=len(sampled_series),
                    affected_row_percentage=self._percentage(
                        affected_count=len(sampled_series),
                        total_count=len(data),
                    ),
                    example_values=example_values,
                    recommended_action=FindingAction.COERCE,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=False,
                    details={
                        "type_counts": dict(type_counts),
                        "sampled_row_count": len(
                            sampled_series
                        ),
                    },
                )
            )

        return findings_by_column

    def _select_columns(
        self,
        data: pd.DataFrame,
    ) -> list[str]:
        """Select columns eligible for mixed-type inspection."""

        excluded = set(
            self.options.exclude_columns
        )

        if self.options.include_columns is not None:
            return [
                column
                for column in self.options.include_columns
                if column in data.columns
                and column not in excluded
            ]

        selected_columns: list[str] = []

        for column in data.columns:
            column_name = str(column)

            if column_name in excluded:
                continue

            if (
                self.options.inspect_only_object_columns
                and not is_object_dtype(data[column].dtype)
            ):
                continue

            selected_columns.append(column_name)

        return selected_columns

    def _sample_series(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Select a deterministic inspection sample."""

        if self.options.sample_size is None:
            return series

        if len(series) <= self.options.sample_size:
            return series

        return series.iloc[
            : self.options.sample_size
        ]

    def _type_group(
        self,
        value: object,
    ) -> str:
        """Resolve a Python value into a compatibility group."""

        if (
            self.options.compatible_numeric_types
            and isinstance(value, (int, float, complex))
            and not isinstance(value, bool)
        ):
            return "numeric"

        if (
            self.options.compatible_string_types
            and isinstance(value, str)
        ):
            return "string"

        return type(value).__name__

    def _meets_threshold(
        self,
        *,
        count: int,
        sample_count: int,
    ) -> bool:
        """Return whether a type occurrence meets reporting limits."""

        if count < self.options.minimum_secondary_type_count:
            return False

        if sample_count == 0:
            return False

        ratio = count / sample_count

        return (
            ratio
            >= self.options.minimum_secondary_type_ratio
        )

    def _extract_examples(
        self,
        series: pd.Series,
    ) -> list[object]:
        """Return representative values from different types."""

        if self.options.maximum_examples == 0:
            return []

        examples: list[object] = []
        observed_groups: set[str] = set()

        for value in series:
            type_group = self._type_group(value)

            if type_group in observed_groups:
                continue

            examples.append(value)
            observed_groups.add(type_group)

            if (
                len(examples)
                >= self.options.maximum_examples
            ):
                break

        return examples

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