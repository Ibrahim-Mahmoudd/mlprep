"""Selected feature-count validation for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FeatureCountResult(BaseModel):
    """Complete selected feature-count validation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    source_feature_count: int = Field(
        default=0,
        ge=0,
    )

    selected_feature_count: int = Field(
        default=0,
        ge=0,
    )

    removed_feature_count: int = Field(
        default=0,
        ge=0,
    )

    selected_feature_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    reduction_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    minimum_feature_count_satisfied: bool = True
    maximum_feature_count_satisfied: bool = True

    minimum_selected_ratio_satisfied: bool = True
    maximum_selected_ratio_satisfied: bool = True

    reduction_requirement_satisfied: bool = True

    blocking_reasons: list[str] = Field(
        default_factory=list,
    )

    warnings: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class FeatureCountOptions:
    """Options controlling selected feature-count validation."""

    enabled: bool = True

    require_nonempty_selection: bool = True

    minimum_feature_count: int = 1
    maximum_feature_count: int | None = None

    minimum_selected_ratio: float | None = None
    maximum_selected_ratio: float | None = None

    require_feature_reduction: bool = False
    minimum_reduction_ratio: float = 0.0

    warn_when_no_reduction: bool = True

    def __post_init__(self) -> None:
        """Validate feature-count options."""

        if self.minimum_feature_count < 0:
            raise ValueError(
                "minimum_feature_count cannot be negative."
            )

        if (
            self.maximum_feature_count is not None
            and self.maximum_feature_count
            < self.minimum_feature_count
        ):
            raise ValueError(
                "maximum_feature_count cannot be smaller "
                "than minimum_feature_count."
            )

        for name, value in {
            "minimum_selected_ratio": (
                self.minimum_selected_ratio
            ),
            "maximum_selected_ratio": (
                self.maximum_selected_ratio
            ),
        }.items():
            if (
                value is not None
                and not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"{name} must be between zero and one."
                )

        if (
            self.minimum_selected_ratio is not None
            and self.maximum_selected_ratio is not None
            and self.minimum_selected_ratio
            > self.maximum_selected_ratio
        ):
            raise ValueError(
                "minimum_selected_ratio cannot exceed "
                "maximum_selected_ratio."
            )

        if not 0.0 <= self.minimum_reduction_ratio <= 1.0:
            raise ValueError(
                "minimum_reduction_ratio must be between "
                "zero and one."
            )


class FeatureCountValidator:
    """Validate the final number of selected features."""

    component_name = "feature_count"

    def __init__(
        self,
        options: FeatureCountOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or FeatureCountOptions()
        )

    def validate(
        self,
        *,
        source_features: list[str],
        selected_features: list[str],
    ) -> FeatureCountResult:
        """Validate selected feature-count policies."""

        source = self._normalize_features(
            source_features,
            name="source_features",
            allow_empty=False,
        )

        selected = self._normalize_features(
            selected_features,
            name="selected_features",
            allow_empty=True,
        )

        source_set = set(
            source
        )

        unknown_selected = sorted(
            set(
                selected
            ).difference(
                source_set
            )
        )

        if unknown_selected:
            raise ValueError(
                "Selected features are absent from the "
                f"source schema: {unknown_selected}"
            )

        if not self.options.enabled:
            return FeatureCountResult(
                source_feature_count=len(
                    source
                ),
                selected_feature_count=len(
                    selected
                ),
                removed_feature_count=(
                    len(source)
                    - len(selected)
                ),
                metadata={
                    "enabled": False,
                },
            )

        source_count = len(
            source
        )

        selected_count = len(
            selected
        )

        removed_count = (
            source_count
            - selected_count
        )

        selected_ratio = (
            selected_count
            / source_count
            if source_count > 0
            else 0.0
        )

        reduction_ratio = (
            removed_count
            / source_count
            if source_count > 0
            else 0.0
        )

        minimum_count_satisfied = bool(
            selected_count
            >= self.options.minimum_feature_count
        )

        maximum_count_satisfied = bool(
            self.options.maximum_feature_count
            is None
            or selected_count
            <= self.options.maximum_feature_count
        )

        minimum_ratio_satisfied = bool(
            self.options.minimum_selected_ratio
            is None
            or selected_ratio
            >= self.options.minimum_selected_ratio
        )

        maximum_ratio_satisfied = bool(
            self.options.maximum_selected_ratio
            is None
            or selected_ratio
            <= self.options.maximum_selected_ratio
        )

        reduction_satisfied = bool(
            not self.options.require_feature_reduction
            or reduction_ratio
            >= self.options.minimum_reduction_ratio
        )

        blocking_reasons: list[str] = []
        warnings: list[str] = []

        if (
            self.options.require_nonempty_selection
            and selected_count == 0
        ):
            blocking_reasons.append(
                "The selected feature schema is empty."
            )

        if not minimum_count_satisfied:
            blocking_reasons.append(
                "The selected feature count is below the "
                "configured minimum."
            )

        if not maximum_count_satisfied:
            blocking_reasons.append(
                "The selected feature count exceeds the "
                "configured maximum."
            )

        if not minimum_ratio_satisfied:
            blocking_reasons.append(
                "The selected feature ratio is below the "
                "configured minimum."
            )

        if not maximum_ratio_satisfied:
            blocking_reasons.append(
                "The selected feature ratio exceeds the "
                "configured maximum."
            )

        if not reduction_satisfied:
            blocking_reasons.append(
                "The feature reduction ratio is below the "
                "configured requirement."
            )

        if (
            self.options.warn_when_no_reduction
            and removed_count == 0
        ):
            warnings.append(
                "Phase 7 did not reduce the feature count."
            )

        return FeatureCountResult(
            valid=not blocking_reasons,
            source_feature_count=source_count,
            selected_feature_count=(
                selected_count
            ),
            removed_feature_count=removed_count,
            selected_feature_ratio=(
                selected_ratio
            ),
            reduction_ratio=reduction_ratio,
            minimum_feature_count_satisfied=(
                minimum_count_satisfied
            ),
            maximum_feature_count_satisfied=(
                maximum_count_satisfied
            ),
            minimum_selected_ratio_satisfied=(
                minimum_ratio_satisfied
            ),
            maximum_selected_ratio_satisfied=(
                maximum_ratio_satisfied
            ),
            reduction_requirement_satisfied=(
                reduction_satisfied
            ),
            blocking_reasons=(
                blocking_reasons
            ),
            warnings=warnings,
            metadata={
                "enabled": True,
                "minimum_feature_count": (
                    self.options.minimum_feature_count
                ),
                "maximum_feature_count": (
                    self.options.maximum_feature_count
                ),
                "minimum_selected_ratio": (
                    self.options.minimum_selected_ratio
                ),
                "maximum_selected_ratio": (
                    self.options.maximum_selected_ratio
                ),
                "minimum_reduction_ratio": (
                    self.options.minimum_reduction_ratio
                ),
            },
        )

    @staticmethod
    def _normalize_features(
        features: list[str],
        *,
        name: str,
        allow_empty: bool,
    ) -> list[str]:
        """Normalize one feature-name list."""

        normalized = [
            str(feature).strip()
            for feature in features
        ]

        if not allow_empty and not normalized:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if any(
            not feature
            for feature in normalized
        ):
            raise ValueError(
                f"{name} cannot contain empty names."
            )

        if len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(
                f"{name} contains duplicate names."
            )

        return normalized
