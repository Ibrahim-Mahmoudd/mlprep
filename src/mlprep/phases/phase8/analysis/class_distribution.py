"""Training target-distribution analysis for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase8.utils.class_identity import (
    build_class_identity_mapping,
    normalize_class_value,
)


class ClassDistributionEntry(BaseModel):
    """Distribution information for one target class."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    class_key: str

    class_value: Any

    count: int = Field(
        default=0,
        ge=0,
    )

    percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    proportion: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    rank_by_count: int = Field(
        ge=1,
    )

    minority_class: bool = False
    majority_class: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ClassDistributionResult(BaseModel):
    """Complete training target-distribution output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    row_count: int = Field(
        default=0,
        ge=0,
    )

    valid_target_count: int = Field(
        default=0,
        ge=0,
    )

    missing_target_count: int = Field(
        default=0,
        ge=0,
    )

    missing_target_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    class_count: int = Field(
        default=0,
        ge=0,
    )

    classes: dict[
        str,
        ClassDistributionEntry,
    ] = Field(
        default_factory=dict,
    )

    class_order: list[str] = Field(
        default_factory=list,
    )

    minority_classes: list[str] = Field(
        default_factory=list,
    )

    majority_classes: list[str] = Field(
        default_factory=list,
    )

    smallest_class_count: int = Field(
        default=0,
        ge=0,
    )

    largest_class_count: int = Field(
        default=0,
        ge=0,
    )

    class_count_difference: int = Field(
        default=0,
        ge=0,
    )

    balanced: bool = False

    analyzed_on_training_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class ClassDistributionOptions:
    """Options controlling target-distribution analysis."""

    enabled: bool = True

    include_missing_as_class: bool = False

    missing_class_key: str = "__MISSING_TARGET__"

    class_key_separator: str = "::"

    balanced_count_tolerance: int = 0

    balanced_percentage_tolerance: float = 0.0

    require_at_least_two_classes: bool = True

    require_nonmissing_target: bool = True

    preserve_first_observed_order: bool = False

    def __post_init__(self) -> None:
        """Validate class-distribution options."""

        if not self.missing_class_key:
            raise ValueError(
                "missing_class_key cannot be empty."
            )

        if not self.class_key_separator:
            raise ValueError(
                "class_key_separator cannot be empty."
            )

        if self.balanced_count_tolerance < 0:
            raise ValueError(
                "balanced_count_tolerance cannot be negative."
            )

        if not 0.0 <= (
            self.balanced_percentage_tolerance
        ) <= 100.0:
            raise ValueError(
                "balanced_percentage_tolerance must be "
                "between zero and one hundred."
            )


class ClassDistributionAnalyzer:
    """Analyze training-target class counts and proportions."""

    component_name = "class_distribution"

    def __init__(
        self,
        options: ClassDistributionOptions | None = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options
            or ClassDistributionOptions()
        )

        self.result: ClassDistributionResult | None = None

    def analyze(
        self,
        *,
        target: pd.Series,
    ) -> ClassDistributionResult:
        """Analyze a training target."""

        self._validate_target(
            target
        )

        if not self.options.enabled:
            result = ClassDistributionResult(
                row_count=len(
                    target
                ),
                valid_target_count=int(
                    target.notna().sum()
                ),
                missing_target_count=int(
                    target.isna().sum()
                ),
                analyzed_on_training_only=True,
                metadata={
                    "enabled": False,
                },
            )

            self.result = result

            return result.model_copy(
                deep=True
            )

        row_count = len(
            target
        )

        missing_target_count = int(
            target.isna().sum()
        )

        valid_target_count = int(
            target.notna().sum()
        )

        if (
            self.options.require_nonmissing_target
            and missing_target_count > 0
        ):
            raise ValueError(
                "Training target contains "
                f"{missing_target_count} missing values."
            )

        identity_mapping = build_class_identity_mapping(
            target,
            reject_missing=False,
        )
        if (
            self.options.include_missing_as_class
            and identity_mapping.contains_value(
                self.options.missing_class_key
            )
        ):
            raise ValueError(
                "missing_class_key conflicts with an existing "
                "training class label."
            )

        counts: dict[str, tuple[object, int, int]] = {}
        for position, raw_value in enumerate(target):
            class_value = normalize_class_value(raw_value)
            if class_value is None:
                if not self.options.include_missing_as_class:
                    continue
                key = self.options.missing_class_key
                value: object = None
            else:
                key = identity_mapping.key_for(class_value)
                value = class_value

            if key not in counts:
                counts[key] = (value, 0, position)
            current_value, count, first_position = counts[key]
            counts[key] = (
                current_value,
                count + 1,
                first_position,
            )

        class_records = [
            (key, value, count, first_position)
            for key, (value, count, first_position) in counts.items()
        ]

        if (
            self.options
            .preserve_first_observed_order
        ):
            ordered_records = list(
                class_records
            )

        else:
            ordered_records = sorted(
                class_records,
                key=lambda item: (
                    -item[
                        2
                    ],
                    item[
                        0
                    ],
                ),
            )

        class_count = len(
            ordered_records
        )

        if (
            self.options.require_at_least_two_classes
            and class_count < 2
        ):
            raise ValueError(
                "Training target must contain at least "
                "two distinct classes."
            )

        if ordered_records:
            smallest_class_count = min(
                record[
                    2
                ]
                for record in ordered_records
            )

            largest_class_count = max(
                record[
                    2
                ]
                for record in ordered_records
            )

        else:
            smallest_class_count = 0
            largest_class_count = 0

        class_count_difference = (
            largest_class_count
            - smallest_class_count
        )

        denominator = (
            row_count
            if self.options
            .include_missing_as_class
            else valid_target_count
        )

        minority_classes = [
            class_key
            for (
                class_key,
                _,
                count,
                _,
            ) in ordered_records
            if (
                count
                == smallest_class_count
            )
        ]

        majority_classes = [
            class_key
            for (
                class_key,
                _,
                count,
                _,
            ) in ordered_records
            if (
                count
                == largest_class_count
            )
        ]

        rank_mapping = self._count_ranks(
            ordered_records
        )

        class_entries: dict[
            str,
            ClassDistributionEntry,
        ] = {}

        for (
            class_key,
            class_value,
            count,
            first_position,
        ) in ordered_records:
            proportion = (
                count
                / denominator
                if denominator > 0
                else 0.0
            )

            class_entries[
                class_key
            ] = ClassDistributionEntry(
                class_key=class_key,
                class_value=class_value,
                count=count,
                percentage=(
                    proportion
                    * 100.0
                ),
                proportion=proportion,
                rank_by_count=(
                    rank_mapping[
                        class_key
                    ]
                ),
                minority_class=(
                    class_key
                    in minority_classes
                ),
                majority_class=(
                    class_key
                    in majority_classes
                ),
                metadata={
                    "first_observed_position": (
                        first_position
                    ),
                },
            )

        balanced_by_count = bool(
            class_count_difference
            <= self.options
            .balanced_count_tolerance
        )

        maximum_percentage_difference = (
            self._percentage_difference(
                entries=class_entries
            )
        )

        balanced_by_percentage = bool(
            maximum_percentage_difference
            <= self.options
            .balanced_percentage_tolerance
        )

        balanced = bool(
            balanced_by_count
            or balanced_by_percentage
        )

        result = ClassDistributionResult(
            row_count=row_count,
            valid_target_count=(
                valid_target_count
            ),
            missing_target_count=(
                missing_target_count
            ),
            missing_target_percentage=(
                missing_target_count
                / row_count
                * 100.0
                if row_count > 0
                else 0.0
            ),
            class_count=class_count,
            classes=class_entries,
            class_order=[
                class_key
                for (
                    class_key,
                    _,
                    _,
                    _,
                ) in ordered_records
            ],
            minority_classes=(
                minority_classes
            ),
            majority_classes=(
                majority_classes
            ),
            smallest_class_count=(
                smallest_class_count
            ),
            largest_class_count=(
                largest_class_count
            ),
            class_count_difference=(
                class_count_difference
            ),
            balanced=balanced,
            analyzed_on_training_only=True,
            metadata={
                "enabled": True,
                "target_dtype": str(
                    target.dtype
                ),
                "balanced_by_count": (
                    balanced_by_count
                ),
                "balanced_by_percentage": (
                    balanced_by_percentage
                ),
                "maximum_percentage_difference": (
                    maximum_percentage_difference
                ),
                "include_missing_as_class": (
                    self.options
                    .include_missing_as_class
                ),
            },
        )

        self.result = result

        return result.model_copy(
            deep=True
        )

    def get_result(
        self,
    ) -> ClassDistributionResult:
        """Return the latest analysis result."""

        if self.result is None:
            raise RuntimeError(
                "The class distribution has not been analyzed."
            )

        return self.result.model_copy(
            deep=True
        )

    @staticmethod
    def _count_ranks(
        records: list[
            tuple[
                str,
                object,
                int,
                int,
            ]
        ],
    ) -> dict[str, int]:
        """Assign deterministic dense ranks by class count."""

        ordered_counts = sorted(
            {
                record[
                    2
                ]
                for record in records
            },
            reverse=True,
        )

        count_to_rank = {
            count: rank
            for rank, count
            in enumerate(
                ordered_counts,
                start=1,
            )
        }

        return {
            class_key: count_to_rank[
                count
            ]
            for (
                class_key,
                _,
                count,
                _,
            ) in records
        }

    @staticmethod
    def _percentage_difference(
        *,
        entries: dict[
            str,
            ClassDistributionEntry,
        ],
    ) -> float:
        """Return maximum class-percentage difference."""

        percentages = [
            entry.percentage
            for entry
            in entries.values()
        ]

        if not percentages:
            return 0.0

        return (
            max(
                percentages
            )
            - min(
                percentages
            )
        )

    @staticmethod
    def _validate_target(
        target: pd.Series,
    ) -> None:
        """Validate the target input."""

        if not isinstance(
            target,
            pd.Series,
        ):
            raise TypeError(
                "target must be a pandas Series."
            )

        if target.empty:
            raise ValueError(
                "Training target cannot be empty."
            )
