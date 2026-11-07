"""Post-resampling distribution validation for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase8.analysis import (
    ClassDistributionResult,
    ImbalanceMetricsResult,
)
from mlprep.phases.phase8.contract import (
    BalancingStrategy,
)


class ClassDistributionValidationEntry(BaseModel):
    """Distribution validation for one target class."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    class_key: str

    source_count: int = Field(
        default=0,
        ge=0,
    )

    output_count: int = Field(
        default=0,
        ge=0,
    )

    count_change: int = 0

    expansion_ratio: float | None = Field(
        default=None,
        ge=0.0,
    )

    contraction_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    class_preserved: bool = True
    minimum_count_satisfied: bool = True

    expansion_policy_satisfied: bool = True
    contraction_policy_satisfied: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class DistributionValidationResult(BaseModel):
    """Complete post-resampling distribution validation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    configured_strategy: BalancingStrategy
    executed_strategy: BalancingStrategy

    source_class_count: int = Field(
        default=0,
        ge=0,
    )

    output_class_count: int = Field(
        default=0,
        ge=0,
    )

    expected_classes: list[str] = Field(
        default_factory=list,
    )

    output_classes: list[str] = Field(
        default_factory=list,
    )

    missing_classes: list[str] = Field(
        default_factory=list,
    )

    unexpected_classes: list[str] = Field(
        default_factory=list,
    )

    classes: dict[
        str,
        ClassDistributionValidationEntry,
    ] = Field(
        default_factory=dict,
    )

    minimum_class_count_satisfied: bool = True
    imbalance_ratio_satisfied: bool = True
    strategy_achievement_satisfied: bool = True

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
class DistributionValidationOptions:
    """Options controlling distribution validation."""

    enabled: bool = True

    require_all_source_classes: bool = True
    reject_unexpected_classes: bool = True

    minimum_output_class_count: int = 1

    maximum_imbalance_ratio: float | None = None

    maximum_class_expansion_ratio: float | None = None
    maximum_class_contraction_ratio: float | None = None

    require_strategy_achievement: bool = True

    class_count_tolerance: int = 0

    warn_when_distribution_unchanged: bool = True

    def __post_init__(self) -> None:
        """Validate distribution-validation options."""

        if self.minimum_output_class_count < 1:
            raise ValueError(
                "minimum_output_class_count must be positive."
            )

        if (
            self.maximum_imbalance_ratio is not None
            and self.maximum_imbalance_ratio < 1.0
        ):
            raise ValueError(
                "maximum_imbalance_ratio must be at least one."
            )

        if (
            self.maximum_class_expansion_ratio is not None
            and self.maximum_class_expansion_ratio < 1.0
        ):
            raise ValueError(
                "maximum_class_expansion_ratio must be "
                "at least one."
            )

        if (
            self.maximum_class_contraction_ratio is not None
            and not 0.0
            <= self.maximum_class_contraction_ratio
            <= 1.0
        ):
            raise ValueError(
                "maximum_class_contraction_ratio must be "
                "between zero and one."
            )

        if self.class_count_tolerance < 0:
            raise ValueError(
                "class_count_tolerance cannot be negative."
            )


class DistributionValidator:
    """Validate the post-resampling class distribution."""

    component_name = "distribution_validation"

    def __init__(
        self,
        options: DistributionValidationOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or DistributionValidationOptions()
        )

    def validate(
        self,
        *,
        source_distribution: ClassDistributionResult,
        output_distribution: ClassDistributionResult,
        output_metrics: ImbalanceMetricsResult,
        configured_strategy: BalancingStrategy,
        executed_strategy: BalancingStrategy,
    ) -> DistributionValidationResult:
        """Validate the final training-target distribution."""

        if not isinstance(
            configured_strategy,
            BalancingStrategy,
        ):
            configured_strategy = BalancingStrategy(
                configured_strategy
            )

        if not isinstance(
            executed_strategy,
            BalancingStrategy,
        ):
            executed_strategy = BalancingStrategy(
                executed_strategy
            )

        if not self.options.enabled:
            return DistributionValidationResult(
                configured_strategy=configured_strategy,
                executed_strategy=executed_strategy,
                metadata={
                    "enabled": False,
                },
            )

        source_classes = list(
            source_distribution.class_order
        )

        output_classes = list(
            output_distribution.class_order
        )

        source_set = set(
            source_classes
        )

        output_set = set(
            output_classes
        )

        missing_classes = [
            class_key
            for class_key in source_classes
            if class_key not in output_set
        ]

        unexpected_classes = [
            class_key
            for class_key in output_classes
            if class_key not in source_set
        ]

        entries: dict[
            str,
            ClassDistributionValidationEntry,
        ] = {}

        minimum_satisfied = True
        expansion_satisfied = True
        contraction_satisfied = True

        for class_key in source_classes:
            source_entry = (
                source_distribution.classes[
                    class_key
                ]
            )

            output_entry = (
                output_distribution.classes.get(
                    class_key
                )
            )

            source_count = source_entry.count

            output_count = (
                output_entry.count
                if output_entry is not None
                else 0
            )

            count_change = (
                output_count
                - source_count
            )

            expansion_ratio = (
                output_count
                / source_count
                if source_count > 0
                else None
            )

            contraction_ratio = (
                (
                    source_count
                    - output_count
                )
                / source_count
                if (
                    source_count > 0
                    and output_count
                    < source_count
                )
                else 0.0
            )

            class_minimum_satisfied = bool(
                output_count
                >= self.options
                .minimum_output_class_count
            )

            class_expansion_satisfied = bool(
                self.options
                .maximum_class_expansion_ratio
                is None
                or expansion_ratio is None
                or expansion_ratio
                <= self.options
                .maximum_class_expansion_ratio
            )

            class_contraction_satisfied = bool(
                self.options
                .maximum_class_contraction_ratio
                is None
                or contraction_ratio
                <= self.options
                .maximum_class_contraction_ratio
            )

            minimum_satisfied = bool(
                minimum_satisfied
                and class_minimum_satisfied
            )

            expansion_satisfied = bool(
                expansion_satisfied
                and class_expansion_satisfied
            )

            contraction_satisfied = bool(
                contraction_satisfied
                and class_contraction_satisfied
            )

            entries[
                class_key
            ] = ClassDistributionValidationEntry(
                class_key=class_key,
                source_count=source_count,
                output_count=output_count,
                count_change=count_change,
                expansion_ratio=expansion_ratio,
                contraction_ratio=(
                    contraction_ratio
                ),
                class_preserved=(
                    output_entry is not None
                ),
                minimum_count_satisfied=(
                    class_minimum_satisfied
                ),
                expansion_policy_satisfied=(
                    class_expansion_satisfied
                ),
                contraction_policy_satisfied=(
                    class_contraction_satisfied
                ),
            )

        imbalance_satisfied = bool(
            self.options.maximum_imbalance_ratio
            is None
            or output_metrics.imbalance_ratio
            <= self.options.maximum_imbalance_ratio
        )

        strategy_satisfied = (
            self._strategy_achievement(
                source_distribution=(
                    source_distribution
                ),
                output_distribution=(
                    output_distribution
                ),
                strategy=executed_strategy,
            )
        )

        blocking_reasons: list[str] = []
        warnings: list[str] = []

        if (
            self.options.require_all_source_classes
            and missing_classes
        ):
            blocking_reasons.append(
                "One or more source target classes are absent "
                "from the resampled training output."
            )

        if (
            self.options.reject_unexpected_classes
            and unexpected_classes
        ):
            blocking_reasons.append(
                "The resampled training output contains "
                "unexpected target classes."
            )

        if not minimum_satisfied:
            blocking_reasons.append(
                "One or more output classes contain fewer "
                "rows than the configured minimum."
            )

        if not expansion_satisfied:
            blocking_reasons.append(
                "One or more classes exceed the configured "
                "maximum expansion ratio."
            )

        if not contraction_satisfied:
            blocking_reasons.append(
                "One or more classes exceed the configured "
                "maximum contraction ratio."
            )

        if not imbalance_satisfied:
            blocking_reasons.append(
                "The output imbalance ratio exceeds the "
                "configured maximum."
            )

        if (
            self.options.require_strategy_achievement
            and not strategy_satisfied
        ):
            blocking_reasons.append(
                "The executed balancing strategy did not "
                "achieve its expected distribution policy."
            )

        distribution_changed = bool(
            any(
                entry.count_change != 0
                for entry in entries.values()
            )
        )

        if (
            self.options.warn_when_distribution_unchanged
            and executed_strategy
            not in {
                BalancingStrategy.NONE,
                BalancingStrategy.CLASS_WEIGHTS_ONLY,
            }
            and not distribution_changed
        ):
            warnings.append(
                "The configured resampling strategy did not "
                "change class counts."
            )

        if (
            configured_strategy
            != executed_strategy
        ):
            warnings.append(
                "The executed strategy differs from the "
                "configured strategy because a fallback "
                "was applied."
            )

        return DistributionValidationResult(
            valid=not blocking_reasons,
            configured_strategy=(
                configured_strategy
            ),
            executed_strategy=(
                executed_strategy
            ),
            source_class_count=(
                source_distribution.class_count
            ),
            output_class_count=(
                output_distribution.class_count
            ),
            expected_classes=source_classes,
            output_classes=output_classes,
            missing_classes=missing_classes,
            unexpected_classes=(
                unexpected_classes
            ),
            classes=entries,
            minimum_class_count_satisfied=(
                minimum_satisfied
            ),
            imbalance_ratio_satisfied=(
                imbalance_satisfied
            ),
            strategy_achievement_satisfied=(
                strategy_satisfied
            ),
            blocking_reasons=(
                blocking_reasons
            ),
            warnings=warnings,
            metadata={
                "enabled": True,
                "source_imbalance_ratio": (
                    source_distribution
                    .largest_class_count
                    / max(
                        source_distribution
                        .smallest_class_count,
                        1,
                    )
                ),
                "output_imbalance_ratio": (
                    output_metrics
                    .imbalance_ratio
                ),
                "distribution_changed": (
                    distribution_changed
                ),
            },
        )

    def _strategy_achievement(
        self,
        *,
        source_distribution: ClassDistributionResult,
        output_distribution: ClassDistributionResult,
        strategy: BalancingStrategy,
    ) -> bool:
        """Evaluate expected class-count direction."""

        tolerance = (
            self.options.class_count_tolerance
        )

        if strategy in {
            BalancingStrategy.NONE,
            BalancingStrategy.CLASS_WEIGHTS_ONLY,
        }:
            return all(
                abs(
                    output_distribution
                    .classes[
                        class_key
                    ]
                    .count
                    - source_distribution
                    .classes[
                        class_key
                    ]
                    .count
                )
                <= tolerance
                for class_key
                in source_distribution.class_order
                if class_key
                in output_distribution.classes
            )

        if strategy in {
            BalancingStrategy.RANDOM_OVERSAMPLING,
            BalancingStrategy.SMOTE,
            BalancingStrategy.BORDERLINE_SMOTE,
            BalancingStrategy.ADASYN,
        }:
            return all(
                output_distribution
                .classes[
                    class_key
                ]
                .count
                + tolerance
                >= source_distribution
                .classes[
                    class_key
                ]
                .count
                for class_key
                in source_distribution.class_order
                if class_key
                in output_distribution.classes
            )

        if (
            strategy
            == BalancingStrategy.RANDOM_UNDERSAMPLING
        ):
            return all(
                output_distribution
                .classes[
                    class_key
                ]
                .count
                <= source_distribution
                .classes[
                    class_key
                ]
                .count
                + tolerance
                for class_key
                in source_distribution.class_order
                if class_key
                in output_distribution.classes
            )

        return True
