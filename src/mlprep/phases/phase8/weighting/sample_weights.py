"""Training sample-weight generation for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from mlprep.phases.phase8.samplers.base import (
    ResampledRowType,
    SamplerRowRecord,
)
from mlprep.phases.phase8.serialization import (
    Phase8SerializableModel,
    restore_phase8_series,
)
from mlprep.phases.phase8.utils.class_identity import (
    class_key,
    class_values_equal,
)
from mlprep.phases.phase8.weighting.class_weights import (
    ClassWeightResult,
)


class SampleWeightEntry(Phase8SerializableModel):
    """Sample-weight information for one output training row."""

    output_position: int = Field(
        ge=0,
    )

    output_row_id: str

    class_key: str

    class_value: Any

    weight: float = Field(
        gt=0.0,
    )

    row_type: ResampledRowType

    inherited_from_source: bool = False
    assigned_from_synthetic_class: bool = False

    source_row_ids: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SampleWeightResult(Phase8SerializableModel):
    """Complete output sample-weight mapping."""

    enabled: bool = True

    weights: pd.Series

    entries: list[
        SampleWeightEntry
    ] = Field(
        default_factory=list,
    )

    row_count: int = Field(
        default=0,
        ge=0,
    )

    minimum_weight: float | None = Field(
        default=None,
        gt=0.0,
    )

    maximum_weight: float | None = Field(
        default=None,
        gt=0.0,
    )

    mean_weight: float | None = Field(
        default=None,
        gt=0.0,
    )

    sum_weight: float = Field(
        default=0.0,
        ge=0.0,
    )

    class_weight_method: str

    fitted_on_training_only: bool = True
    target_access_authorized: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("weights", mode="before")
    @classmethod
    def restore_weights_series(cls, value: Any) -> Any:
        """Restore a JSON-serialized weight series."""

        return restore_phase8_series(value)


@dataclass(slots=True)
class SampleWeightOptions:
    """Options controlling sample-weight generation."""

    enabled: bool = True

    require_complete_lineage: bool = True

    require_unique_row_ids: bool = True

    require_matching_target_and_row_ids: bool = True

    require_class_weight_for_every_target: bool = True

    weight_series_name: str = "sample_weight"

    class_key_separator: str = "::"

    def __post_init__(self) -> None:
        """Validate sample-weight options."""

        if not self.weight_series_name.strip():
            raise ValueError(
                "weight_series_name cannot be empty."
            )

        if not self.class_key_separator:
            raise ValueError(
                "class_key_separator cannot be empty."
            )


class SampleWeightGenerator:
    """Map fitted class weights to output training rows."""

    component_name = "sample_weights"

    def __init__(
        self,
        options: SampleWeightOptions | None = None,
    ) -> None:
        """Initialize the sample-weight generator."""

        self.options = (
            options
            or SampleWeightOptions()
        )

        self.result: SampleWeightResult | None = None

    def generate(
        self,
        *,
        target: pd.Series,
        row_ids: pd.Series,
        class_weights: ClassWeightResult,
        row_lineage: list[SamplerRowRecord],
        target_access_authorized: bool,
    ) -> SampleWeightResult:
        """Generate sample weights for resampled training rows."""

        self._validate_inputs(
            target=target,
            row_ids=row_ids,
            class_weights=class_weights,
            row_lineage=row_lineage,
        )

        if not isinstance(
            target_access_authorized,
            bool,
        ):
            raise TypeError(
                "target_access_authorized must be a Boolean."
            )

        if not target_access_authorized:
            raise PermissionError(
                "Sample-weight generation requires authorized "
                "training-target access."
            )

        if not self.options.enabled:
            result = SampleWeightResult(
                enabled=False,
                weights=pd.Series(
                    dtype="float64",
                    name=(
                        self.options
                        .weight_series_name
                    ),
                ),
                row_count=len(
                    target
                ),
                class_weight_method=(
                    class_weights.method.value
                ),
                fitted_on_training_only=True,
                target_access_authorized=True,
                metadata={
                    "enabled": False,
                },
            )

            self.result = result

            return result.model_copy(
                deep=True
            )

        lineage_by_position = {
            record.output_position: record
            for record in row_lineage
        }

        generated_weights: list[float] = []
        entries: list[
            SampleWeightEntry
        ] = []

        missing_class_keys: set[str] = set()

        for output_position in range(
            len(
                target
            )
        ):
            class_value = target.iloc[
                output_position
            ]

            class_key = self._resolve_class_key(
                class_value=class_value,
                class_weights=class_weights,
            )

            if class_key is None:
                missing_class_keys.add(
                    self._class_key(
                        class_value
                    )
                )

                continue

            weight = float(
                class_weights.class_weights[
                    class_key
                ]
            )

            if (
                not np.isfinite(
                    weight
                )
                or weight <= 0.0
            ):
                raise ValueError(
                    "Sample weights must be finite and positive. "
                    f"Invalid class: {class_value!r}"
                )

            lineage_record = lineage_by_position.get(
                output_position
            )

            if lineage_record is None:
                if self.options.require_complete_lineage:
                    raise ValueError(
                        "Output row lineage is missing for "
                        f"position {output_position}."
                    )

                row_type = (
                    ResampledRowType.ORIGINAL
                )

                source_row_ids: list[str] = []

            else:
                row_type = (
                    lineage_record.row_type
                )

                source_row_ids = list(
                    lineage_record
                    .source_row_ids
                )

                if (
                    str(
                        row_ids.iloc[
                            output_position
                        ]
                    )
                    != lineage_record.output_row_id
                ):
                    raise ValueError(
                        "Row lineage and output row IDs differ "
                        f"at position {output_position}."
                    )

                if not self._class_values_equal(
                    class_value,
                    lineage_record.class_value,
                ):
                    raise ValueError(
                        "Row lineage and output target values "
                        f"differ at position {output_position}."
                    )

            inherited_from_source = bool(
                row_type
                in {
                    ResampledRowType.ORIGINAL,
                    ResampledRowType.DUPLICATE,
                    ResampledRowType
                    .RETAINED_AFTER_CLEANING,
                }
            )

            assigned_from_synthetic_class = bool(
                row_type
                == ResampledRowType.SYNTHETIC
            )

            generated_weights.append(
                weight
            )

            entries.append(
                SampleWeightEntry(
                    output_position=(
                        output_position
                    ),
                    output_row_id=str(
                        row_ids.iloc[
                            output_position
                        ]
                    ),
                    class_key=class_key,
                    class_value=class_value,
                    weight=weight,
                    row_type=row_type,
                    inherited_from_source=(
                        inherited_from_source
                    ),
                    assigned_from_synthetic_class=(
                        assigned_from_synthetic_class
                    ),
                    source_row_ids=(
                        source_row_ids
                    ),
                )
            )

        if (
            missing_class_keys
            and self.options
            .require_class_weight_for_every_target
        ):
            raise KeyError(
                "Class weights are missing for output target "
                f"classes: {sorted(missing_class_keys)}"
            )

        if len(
            generated_weights
        ) != len(
            target
        ):
            raise ValueError(
                "Sample-weight generation did not produce "
                "one weight per output training row."
            )

        weights = pd.Series(
            generated_weights,
            index=target.index.copy(),
            name=self.options.weight_series_name,
            dtype="float64",
        )

        minimum_weight = (
            float(
                weights.min()
            )
            if not weights.empty
            else None
        )

        maximum_weight = (
            float(
                weights.max()
            )
            if not weights.empty
            else None
        )

        mean_weight = (
            float(
                weights.mean()
            )
            if not weights.empty
            else None
        )

        result = SampleWeightResult(
            enabled=True,
            weights=weights,
            entries=entries,
            row_count=len(
                weights
            ),
            minimum_weight=(
                minimum_weight
            ),
            maximum_weight=(
                maximum_weight
            ),
            mean_weight=(
                mean_weight
            ),
            sum_weight=float(
                weights.sum()
            ),
            class_weight_method=(
                class_weights.method.value
            ),
            fitted_on_training_only=True,
            target_access_authorized=True,
            metadata={
                "original_weight_count": sum(
                    entry.row_type
                    == ResampledRowType.ORIGINAL
                    for entry in entries
                ),
                "retained_weight_count": sum(
                    entry.row_type
                    == ResampledRowType
                    .RETAINED_AFTER_CLEANING
                    for entry in entries
                ),
                "duplicate_weight_count": sum(
                    entry.row_type
                    == ResampledRowType.DUPLICATE
                    for entry in entries
                ),
                "synthetic_weight_count": sum(
                    entry.row_type
                    == ResampledRowType.SYNTHETIC
                    for entry in entries
                ),
            },
        )

        self.result = result

        return result.model_copy(
            deep=True
        )

    def get_result(
        self,
    ) -> SampleWeightResult:
        """Return the latest sample-weight output."""

        if self.result is None:
            raise RuntimeError(
                "Sample weights have not been generated."
            )

        return self.result.model_copy(
            deep=True
        )

    def _resolve_class_key(
        self,
        *,
        class_value: object,
        class_weights: ClassWeightResult,
    ) -> str | None:
        """Resolve the fitted key for one output class."""

        direct_key = self._class_key(
            class_value
        )

        if (
            direct_key in class_weights.class_weights
            and direct_key in class_weights.class_values
            and self._class_values_equal(
                class_value,
                class_weights.class_values[direct_key],
            )
        ):
            return direct_key

        return None

    def _class_key(
        self,
        value: object,
    ) -> str:
        """Build a stable typed class key."""

        return class_key(value)

    @staticmethod
    def _class_values_equal(
        left: object,
        right: object,
    ) -> bool:
        """Safely compare two class labels."""

        return class_values_equal(left, right)

    def _validate_inputs(
        self,
        *,
        target: pd.Series,
        row_ids: pd.Series,
        class_weights: ClassWeightResult,
        row_lineage: list[SamplerRowRecord],
    ) -> None:
        """Validate sample-weight inputs."""

        if not isinstance(
            target,
            pd.Series,
        ):
            raise TypeError(
                "target must be a pandas Series."
            )

        if not isinstance(
            row_ids,
            pd.Series,
        ):
            raise TypeError(
                "row_ids must be a pandas Series."
            )

        if not isinstance(
            class_weights,
            ClassWeightResult,
        ):
            raise TypeError(
                "class_weights must be a ClassWeightResult."
            )

        if target.empty:
            raise ValueError(
                "Output training target cannot be empty."
            )

        if len(
            target
        ) != len(
            row_ids
        ):
            raise ValueError(
                "Output target and row IDs have different "
                "row counts."
            )

        if (
            self.options
            .require_matching_target_and_row_ids
            and not target.index.equals(
                row_ids.index
            )
        ):
            raise ValueError(
                "Output target and row IDs have different "
                "indices."
            )

        if target.isna().any():
            raise ValueError(
                "Output training target contains missing values."
            )

        if row_ids.isna().any():
            raise ValueError(
                "Output training row IDs contain missing values."
            )

        if (
            self.options.require_unique_row_ids
            and not row_ids.is_unique
        ):
            raise ValueError(
                "Output training row IDs must be unique."
            )

        if (
            self.options.require_complete_lineage
            and len(
                row_lineage
            )
            != len(
                target
            )
        ):
            raise ValueError(
                "Row lineage must contain one entry per "
                "output training row."
            )

        lineage_positions = [
            record.output_position
            for record in row_lineage
        ]

        if (
            self.options.require_complete_lineage
            and lineage_positions
            != list(
                range(
                    len(
                        target
                    )
                )
            )
        ):
            raise ValueError(
                "Row-lineage positions must form a complete "
                "ordered output sequence."
            )

        if (
            not class_weights.enabled
            or not class_weights.class_weights
        ):
            raise ValueError(
                "Enabled fitted class weights are required "
                "to generate sample weights."
            )
