"""Unknown-category validation for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase6.categorical.categorical_pipeline import (
    UnknownCategorySummary,
)
from mlprep.phases.phase6.orchestration.transformation_pipeline import (
    TransformationPipelineResult,
)


class UnknownCategoryIssue(BaseModel):
    """Unknown-category issue for one source feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    partition: PartitionName

    unknown_count: int = Field(
        default=0,
        ge=0,
    )

    unknown_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    applied_policy: str

    allowed: bool = True
    blocks_readiness: bool = False

    reason: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class UnknownCategoriesResult(BaseModel):
    """Complete unknown-category validation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    issues: list[
        UnknownCategoryIssue
    ] = Field(default_factory=list)

    unknown_columns: dict[
        PartitionName,
        list[str],
    ] = Field(default_factory=dict)

    blocking_columns: dict[
        PartitionName,
        list[str],
    ] = Field(default_factory=dict)

    total_unknown_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class UnknownCategoryOptions:
    """Options controlling unknown-category validation."""

    enabled: bool = True

    allow_unknown_categories: bool = True

    allowed_columns: list[str] = field(
        default_factory=list
    )

    prohibited_columns: list[str] = field(
        default_factory=list
    )

    maximum_unknown_percentage: float = 10.0

    maximum_unknown_percentage_by_column: dict[
        str,
        float,
    ] = field(default_factory=dict)

    maximum_total_unknown_count: int | None = None

    validate_training: bool = False
    validate_validation: bool = True
    validate_test: bool = True

    def __post_init__(self) -> None:
        """Validate unknown-category options."""

        if not 0.0 <= (
            self.maximum_unknown_percentage
        ) <= 100.0:
            raise ValueError(
                "maximum_unknown_percentage must be "
                "between zero and one hundred."
            )

        if (
            self.maximum_total_unknown_count
            is not None
            and self.maximum_total_unknown_count < 0
        ):
            raise ValueError(
                "maximum_total_unknown_count cannot be negative."
            )

        for column, percentage in (
            self.maximum_unknown_percentage_by_column.items()
        ):
            if not 0.0 <= percentage <= 100.0:
                raise ValueError(
                    "Per-column unknown percentages must be "
                    "between zero and one hundred. "
                    f"Invalid column: {column!r}"
                )


class UnknownCategoryValidator:
    """Validate unknown categorical values after transformation."""

    component_name = "unknown_categories"

    def __init__(
        self,
        options: UnknownCategoryOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or UnknownCategoryOptions()
        )

    def validate(
        self,
        *,
        pipeline_result: TransformationPipelineResult,
    ) -> UnknownCategoriesResult:
        """Validate unknown-category summaries."""

        if not self.options.enabled:
            return UnknownCategoriesResult(
                metadata={
                    "enabled": False,
                }
            )

        validation_policy = {
            PartitionName.TRAIN: (
                self.options.validate_training
            ),
            PartitionName.VALIDATION: (
                self.options.validate_validation
            ),
            PartitionName.TEST: (
                self.options.validate_test
            ),
        }

        allowed_columns = set(
            self.options.allowed_columns
        )

        prohibited_columns = set(
            self.options.prohibited_columns
        )

        issues: list[
            UnknownCategoryIssue
        ] = []

        unknown_columns: dict[
            PartitionName,
            list[str],
        ] = {}

        blocking_columns: dict[
            PartitionName,
            list[str],
        ] = {}

        total_unknown_count = 0

        for partition_name, partition_result in (
            pipeline_result.partitions.items()
        ):
            if not validation_policy.get(
                partition_name,
                False,
            ):
                continue

            partition_unknown: list[str] = []
            partition_blocking: list[str] = []

            summaries = (
                partition_result
                .categorical
                .unknown_categories
            )

            for column, summary in summaries.items():
                if not isinstance(
                    summary,
                    UnknownCategorySummary,
                ):
                    raise TypeError(
                        "Unknown-category metadata must contain "
                        "UnknownCategorySummary values."
                    )

                if summary.unknown_count <= 0:
                    continue

                total_unknown_count += (
                    summary.unknown_count
                )

                partition_unknown.append(
                    column
                )

                (
                    allowed,
                    blocks_readiness,
                    reason,
                ) = self._column_policy(
                    column=column,
                    summary=summary,
                    allowed_columns=(
                        allowed_columns
                    ),
                    prohibited_columns=(
                        prohibited_columns
                    ),
                )

                if blocks_readiness:
                    partition_blocking.append(
                        column
                    )

                issues.append(
                    UnknownCategoryIssue(
                        column=column,
                        partition=partition_name,
                        unknown_count=(
                            summary.unknown_count
                        ),
                        unknown_percentage=(
                            summary
                            .unknown_percentage
                        ),
                        applied_policy=(
                            summary.applied_policy
                        ),
                        allowed=allowed,
                        blocks_readiness=(
                            blocks_readiness
                        ),
                        reason=reason,
                    )
                )

            unknown_columns[
                partition_name
            ] = partition_unknown

            blocking_columns[
                partition_name
            ] = partition_blocking

        total_limit_exceeded = bool(
            self.options.maximum_total_unknown_count
            is not None
            and total_unknown_count
            > self.options.maximum_total_unknown_count
        )

        valid = bool(
            not any(
                issue.blocks_readiness
                for issue in issues
            )
            and not total_limit_exceeded
        )

        return UnknownCategoriesResult(
            valid=valid,
            issues=issues,
            unknown_columns=(
                unknown_columns
            ),
            blocking_columns=(
                blocking_columns
            ),
            total_unknown_count=(
                total_unknown_count
            ),
            metadata={
                "enabled": True,
                "total_limit_exceeded": (
                    total_limit_exceeded
                ),
            },
        )

    def _column_policy(
        self,
        *,
        column: str,
        summary: UnknownCategorySummary,
        allowed_columns: set[str],
        prohibited_columns: set[str],
    ) -> tuple[
        bool,
        bool,
        str,
    ]:
        """Resolve policy for one unknown-category summary."""

        if column in prohibited_columns:
            return (
                False,
                True,
                (
                    "Unknown categories are prohibited for "
                    "this feature."
                ),
            )

        configured_maximum = (
            self.options
            .maximum_unknown_percentage_by_column
            .get(
                column
            )
        )

        if configured_maximum is not None:
            if (
                summary.unknown_percentage
                <= configured_maximum
            ):
                return (
                    True,
                    False,
                    (
                        "Unknown-category percentage is within "
                        "the configured feature limit."
                    ),
                )

            return (
                False,
                True,
                (
                    "Unknown-category percentage exceeds "
                    "the configured feature limit."
                ),
            )

        if column in allowed_columns:
            return (
                True,
                False,
                (
                    "Unknown categories are explicitly "
                    "allowed for this feature."
                ),
            )

        if not self.options.allow_unknown_categories:
            return (
                False,
                True,
                (
                    "Unknown categories are disabled by "
                    "the global policy."
                ),
            )

        if (
            summary.unknown_percentage
            > self.options
            .maximum_unknown_percentage
        ):
            return (
                False,
                True,
                (
                    "Unknown-category percentage exceeds "
                    "the global limit."
                ),
            )

        return (
            True,
            False,
            (
                "Unknown categories were handled by the "
                "configured encoder policy."
            ),
        )