"""Training-only class-weight calculation for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase8.contract.balancing_plan import (
    ClassWeightMethod,
    WeightNormalizationKind,
)
from mlprep.phases.phase8.utils.class_identity import (
    ClassIdentityMapping,
    build_class_identity_mapping,
    normalize_class_value,
)


class ClassWeightEntry(BaseModel):
    """Weight and frequency information for one target class."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    class_key: str
    class_value: Any
    count: int = Field(ge=0)
    proportion: float = Field(ge=0.0, le=1.0)
    weight: float = Field(gt=0.0)


class ClassWeightResult(BaseModel):
    """Complete class-weight calculation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    enabled: bool = True
    method: ClassWeightMethod
    normalization: WeightNormalizationKind
    class_weights: dict[str, float] = Field(default_factory=dict)
    class_values: dict[str, Any] = Field(default_factory=dict)
    class_counts: dict[str, int] = Field(default_factory=dict)
    entries: list[ClassWeightEntry] = Field(default_factory=list)
    total_count: int = Field(default=0, ge=0)
    fitted_on_training_only: bool = True
    target_access_authorized: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class ClassWeightOptions:
    """Options controlling class-weight calculation."""

    enabled: bool = True
    method: ClassWeightMethod = ClassWeightMethod.BALANCED
    normalization: WeightNormalizationKind = (
        WeightNormalizationKind.MEAN_ONE
    )
    custom_weights: dict[str, float] | None = None
    effective_number_beta: float = 0.9999

    def __post_init__(self) -> None:
        """Validate class-weight options."""

        if not isinstance(self.method, ClassWeightMethod):
            self.method = ClassWeightMethod(self.method)
        if not isinstance(self.normalization, WeightNormalizationKind):
            self.normalization = WeightNormalizationKind(self.normalization)
        if not 0.0 < self.effective_number_beta < 1.0:
            raise ValueError("effective_number_beta must be between zero and one.")
        if self.custom_weights is not None and any(
            not np.isfinite(weight) or weight <= 0.0
            for weight in self.custom_weights.values()
        ):
            raise ValueError("Custom class weights must be finite and positive.")


class ClassWeightCalculator:
    """Calculate class weights from training target values only."""

    component_name = "class_weights"

    def __init__(
        self,
        options: ClassWeightOptions | None = None,
    ) -> None:
        """Initialize the calculator."""

        self.options = options or ClassWeightOptions()
        self.result: ClassWeightResult | None = None

    def fit(
        self,
        *,
        target: pd.Series,
        target_access_authorized: bool,
        class_identity_mapping: ClassIdentityMapping | None = None,
    ) -> ClassWeightCalculator:
        """Fit weights on an authorized training target."""

        if not isinstance(target, pd.Series):
            raise TypeError("target must be a pandas Series.")
        if not target_access_authorized:
            raise PermissionError(
                "Class-weight calculation requires authorized "
                "training-target access."
            )
        if target.empty:
            raise ValueError("Training target cannot be empty.")
        if target.isna().any():
            raise ValueError("Training target contains missing class labels.")

        identity_mapping = (
            class_identity_mapping
            or build_class_identity_mapping(target)
        )
        counts_by_key: dict[str, int] = {}
        for value in target:
            normalized = normalize_class_value(value)
            if normalized is None:
                raise ValueError(
                    "Training target contains missing class labels."
                )
            key = identity_mapping.key_for(normalized)
            counts_by_key[key] = counts_by_key.get(key, 0) + 1
        values = [
            (
                identity_mapping.value_for(key),
                count,
                key,
            )
            for key, count in sorted(counts_by_key.items())
        ]
        if not self.options.enabled or self.options.method == ClassWeightMethod.NONE:
            self.result = ClassWeightResult(
                enabled=False,
                method=ClassWeightMethod.NONE,
                normalization=self.options.normalization,
                class_weights={
                    key: 1.0 for _, _, key in values
                },
                class_values={
                    key: value for value, _, key in values
                },
                class_counts={
                    key: count for _, count, key in values
                },
                entries=[
                    ClassWeightEntry(
                        class_key=key,
                        class_value=value,
                        count=count,
                        proportion=count / len(target),
                        weight=1.0,
                    )
                    for value, count, key in values
                ],
                total_count=len(target),
                target_access_authorized=True,
            )
            return self

        value_by_key = {
            key: value for value, _, key in values
        }
        raw_weights = self._raw_weights(
            counts_by_key,
            identity_mapping=identity_mapping,
        )
        normalized = self._normalize(raw_weights)
        entries = [
            ClassWeightEntry(
                class_key=key,
                class_value=value_by_key[key],
                count=count,
                proportion=count / len(target),
                weight=normalized[key],
            )
            for key, count in counts_by_key.items()
        ]
        self.result = ClassWeightResult(
            enabled=True,
            method=self.options.method,
            normalization=self.options.normalization,
            class_weights=normalized,
            class_values=value_by_key,
            class_counts=counts_by_key,
            entries=entries,
            total_count=len(target),
            target_access_authorized=True,
            metadata={"class_count": len(counts_by_key)},
        )
        return self

    def get_result(self) -> ClassWeightResult:
        """Return the fitted class-weight result."""

        if self.result is None:
            raise RuntimeError("ClassWeightCalculator.fit() must be called first.")
        return self.result.model_copy(deep=True)

    def _raw_weights(
        self,
        counts: dict[str, int],
        *,
        identity_mapping: ClassIdentityMapping,
    ) -> dict[str, float]:
        total = sum(counts.values())
        class_count = len(counts)
        if self.options.method == ClassWeightMethod.CUSTOM:
            custom = self.options.custom_weights or {}
            normalized_custom: dict[str, float] = {}
            for label, weight in custom.items():
                key = identity_mapping.resolve_configured_key(label)
                if key in normalized_custom:
                    raise ValueError(
                        "Multiple custom weight labels resolve to "
                        f"class key {key!r}."
                    )
                normalized_custom[key] = float(weight)
            resolved: dict[str, float] = {}
            for key in counts:
                if key in normalized_custom:
                    resolved[key] = normalized_custom[key]
                else:
                    raise KeyError(f"No custom weight configured for class {key!r}.")
            return resolved
        if self.options.method in {
            ClassWeightMethod.BALANCED,
            ClassWeightMethod.INVERSE_FREQUENCY,
        }:
            multiplier = total / class_count if (
                self.options.method == ClassWeightMethod.BALANCED
            ) else 1.0
            return {
                key: multiplier / count for key, count in counts.items()
            }
        if self.options.method == ClassWeightMethod.SQRT_INVERSE_FREQUENCY:
            return {key: 1.0 / np.sqrt(count) for key, count in counts.items()}
        if self.options.method == ClassWeightMethod.EFFECTIVE_NUMBER:
            beta = self.options.effective_number_beta
            return {
                key: (1.0 - beta) / (1.0 - beta**count)
                for key, count in counts.items()
            }
        raise ValueError(f"Unsupported class-weight method: {self.options.method}")

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        values = list(weights.values())
        if not values:
            return weights
        if self.options.normalization == WeightNormalizationKind.MEAN_ONE:
            scale = float(np.mean(values))
        elif self.options.normalization == WeightNormalizationKind.SUM_ONE:
            scale = float(np.sum(values))
        elif self.options.normalization == WeightNormalizationKind.MAX_ONE:
            scale = float(np.max(values))
        else:
            return weights
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("Class-weight normalization produced an invalid scale.")
        return {key: value / scale for key, value in weights.items()}
