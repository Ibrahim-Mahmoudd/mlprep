"""Missing-value indicator generation for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase4.contract.imputation_plan import (
    ImputationPlan,
)
from mlprep.phases.phase4.indicators.indicator_naming import (
    IndicatorNameResolver,
    IndicatorNamingOptions,
)


class MissingIndicatorMetadata(BaseModel):
    """Metadata for one generated missing indicator."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    source_column: str
    indicator_column: str

    partition: PartitionName

    row_count: int = Field(
        default=0,
        ge=0,
    )

    true_count: int = Field(
        default=0,
        ge=0,
    )

    false_count: int = Field(
        default=0,
        ge=0,
    )

    true_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    source_column_missing: bool = False

    generated: bool = True
    reused_existing_column: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class MissingIndicatorResult(BaseModel):
    """Complete missing-indicator generation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    partition: PartitionName

    source_to_indicator: dict[
        str,
        str,
    ] = Field(default_factory=dict)

    indicators: dict[
        str,
        MissingIndicatorMetadata,
    ] = Field(default_factory=dict)

    generated_columns: list[str] = Field(
        default_factory=list
    )

    skipped_source_columns: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class MissingIndicatorOptions:
    """Options controlling missing-indicator generation."""

    enabled: bool = True

    only_generate_when_missing: bool = False

    indicator_dtype: str = "boolean"

    copy_input_frame: bool = True

    allow_missing_source_columns: bool = False

    require_identical_mapping: bool = True

    validate_existing_indicator_values: bool = True

    naming: IndicatorNamingOptions = field(
        default_factory=IndicatorNamingOptions
    )

    def __post_init__(self) -> None:
        """Validate missing-indicator options."""

        allowed_dtypes = {
            "boolean",
            "bool",
            "int8",
            "uint8",
        }

        if (
            self.indicator_dtype
            not in allowed_dtypes
        ):
            raise ValueError(
                "indicator_dtype must be one of: "
                "'boolean', 'bool', 'int8', or 'uint8'."
            )


class MissingIndicatorGenerator:
    """Generate indicators from original missing-value masks."""

    component_name = "missing_indicator"

    def __init__(
        self,
        options: MissingIndicatorOptions | None = None,
    ) -> None:
        """Initialize the missing-indicator generator."""

        self.options = (
            options or MissingIndicatorOptions()
        )

        self.name_resolver = (
            IndicatorNameResolver(
                options=self.options.naming
            )
        )

        self._source_to_indicator: dict[
            str,
            str,
        ] = {}

        self._fitted = False

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether training names were resolved."""

        return self._fitted

    def fit(
        self,
        *,
        frame: pd.DataFrame,
        plan: ImputationPlan,
    ) -> MissingIndicatorGenerator:
        """Resolve indicator names from the training schema."""

        self._validate_frame(
            frame
        )

        if self._fitted:
            raise RuntimeError(
                "The missing-indicator generator is "
                "already fitted."
            )

        if not self.options.enabled:
            self._source_to_indicator = {}
            self._fitted = True

            return self

        requested_columns = [
            column
            for column in plan.indicator_columns
            if column in frame.columns
        ]

        missing_source_columns = [
            column
            for column in plan.indicator_columns
            if column not in frame.columns
        ]

        if (
            missing_source_columns
            and not self.options
            .allow_missing_source_columns
        ):
            raise ValueError(
                "Indicator source columns do not exist: "
                f"{missing_source_columns}"
            )

        if self.options.only_generate_when_missing:
            requested_columns = [
                column
                for column in requested_columns
                if frame[column].isna().any()
            ]

        self._source_to_indicator = (
            self.name_resolver.resolve_many(
                columns=requested_columns,
                existing_columns=[
                    str(column)
                    for column in frame.columns
                ],
            )
        )

        self._fitted = True

        return self

    def transform(
        self,
        *,
        frame: pd.DataFrame,
        partition: PartitionName,
        original_missing_mask: (
            pd.DataFrame | None
        ) = None,
    ) -> tuple[
        pd.DataFrame,
        MissingIndicatorResult,
    ]:
        """Add fitted missing indicators to one partition."""

        self._validate_frame(
            frame
        )

        if not self._fitted:
            raise RuntimeError(
                "The missing-indicator generator must "
                "be fitted before transformation."
            )

        transformed = (
            frame.copy(deep=True)
            if self.options.copy_input_frame
            else frame
        )

        if not self.options.enabled:
            return (
                transformed,
                MissingIndicatorResult(
                    partition=partition,
                    metadata={
                        "enabled": False,
                    },
                ),
            )

        if original_missing_mask is None:
            missing_mask = frame.isna()
        else:
            missing_mask = (
                original_missing_mask.copy(
                    deep=True
                )
            )

            self._validate_missing_mask(
                frame=frame,
                missing_mask=missing_mask,
            )

        indicators: dict[
            str,
            MissingIndicatorMetadata,
        ] = {}

        generated_columns: list[str] = []
        skipped_columns: list[str] = []

        for source_column, indicator_column in (
            self._source_to_indicator.items()
        ):
            if source_column not in frame.columns:
                if self.options.allow_missing_source_columns:
                    skipped_columns.append(
                        source_column
                    )
                    continue

                raise ValueError(
                    f"Indicator source column "
                    f"{source_column!r} does not exist in "
                    f"partition {partition.value!r}."
                )

            indicator_values = missing_mask[
                source_column
            ].fillna(False).astype(bool)

            reused_existing = (
                indicator_column
                in transformed.columns
            )

            if reused_existing:
                if (
                    self.options
                    .validate_existing_indicator_values
                ):
                    existing_values = (
                        transformed[
                            indicator_column
                        ]
                        .fillna(False)
                        .astype(bool)
                    )

                    if not existing_values.equals(
                        indicator_values
                    ):
                        raise ValueError(
                            "Existing indicator values do not "
                            "match the original missingness mask "
                            f"for {source_column!r}."
                        )

            else:
                transformed[
                    indicator_column
                ] = self._cast_indicator(
                    indicator_values
                )

                generated_columns.append(
                    indicator_column
                )

            true_count = int(
                indicator_values.sum()
            )

            row_count = len(
                indicator_values
            )

            indicators[source_column] = (
                MissingIndicatorMetadata(
                    source_column=source_column,
                    indicator_column=indicator_column,
                    partition=partition,
                    row_count=row_count,
                    true_count=true_count,
                    false_count=(
                        row_count - true_count
                    ),
                    true_percentage=(
                        true_count
                        / row_count
                        * 100.0
                        if row_count > 0
                        else 0.0
                    ),
                    source_column_missing=False,
                    generated=(
                        not reused_existing
                    ),
                    reused_existing_column=(
                        reused_existing
                    ),
                )
            )

        result = MissingIndicatorResult(
            partition=partition,
            source_to_indicator=dict(
                self._source_to_indicator
            ),
            indicators=indicators,
            generated_columns=(
                generated_columns
            ),
            skipped_source_columns=(
                skipped_columns
            ),
            metadata={
                "enabled": True,
                "indicator_dtype": (
                    self.options
                    .indicator_dtype
                ),
                "indicator_count": len(
                    indicators
                ),
            },
        )

        return (
            transformed,
            result,
        )

    def fit_transform(
        self,
        *,
        frame: pd.DataFrame,
        plan: ImputationPlan,
        partition: PartitionName = (
            PartitionName.TRAIN
        ),
        original_missing_mask: (
            pd.DataFrame | None
        ) = None,
    ) -> tuple[
        pd.DataFrame,
        MissingIndicatorResult,
    ]:
        """Fit on training and generate indicators."""

        self.fit(
            frame=frame,
            plan=plan,
        )

        return self.transform(
            frame=frame,
            partition=partition,
            original_missing_mask=(
                original_missing_mask
            ),
        )

    def mapping(
        self,
    ) -> dict[str, str]:
        """Return a copy of the fitted name mapping."""

        if not self._fitted:
            raise RuntimeError(
                "Indicator mapping is unavailable "
                "before fitting."
            )

        return dict(
            self._source_to_indicator
        )

    def expected_output_columns(
        self,
        *,
        input_columns: list[str],
    ) -> list[object]:
        """Return the expected transformed schema."""

        if not self._fitted:
            raise RuntimeError(
                "The generator must be fitted before "
                "resolving output columns."
            )

        output_columns = list(
            input_columns
        )

        for indicator_column in (
            self._source_to_indicator.values()
        ):
            if indicator_column not in output_columns:
                output_columns.append(
                    indicator_column
                )

        return output_columns

    def validate_partition_mapping(
        self,
        result: MissingIndicatorResult,
    ) -> None:
        """Validate one partition's indicator mapping."""

        if not self.options.require_identical_mapping:
            return

        if (
            result.source_to_indicator
            != self._source_to_indicator
        ):
            raise ValueError(
                "Partition indicator mapping differs from "
                "the fitted training mapping."
            )

    def _cast_indicator(
        self,
        values: pd.Series,
    ) -> pd.Series:
        """Cast an indicator to the configured dtype."""

        if self.options.indicator_dtype == "boolean":
            return values.astype(
                "boolean"
            )

        if self.options.indicator_dtype == "bool":
            return values.astype(
                bool
            )

        if self.options.indicator_dtype == "int8":
            return values.astype(
                "int8"
            )

        return values.astype(
            "uint8"
        )

    @staticmethod
    def _validate_frame(
        frame: pd.DataFrame,
    ) -> None:
        """Validate an input feature frame."""

        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                "Missing-indicator input must be a "
                "pandas DataFrame."
            )

        column_names = [
            str(column)
            for column in frame.columns
        ]

        if len(column_names) != len(
            set(column_names)
        ):
            raise ValueError(
                "Input frame contains duplicate column names."
            )

    @staticmethod
    def _validate_missing_mask(
        *,
        frame: pd.DataFrame,
        missing_mask: pd.DataFrame,
    ) -> None:
        """Validate a provided original missingness mask."""

        if not isinstance(
            missing_mask,
            pd.DataFrame,
        ):
            raise TypeError(
                "original_missing_mask must be a "
                "pandas DataFrame."
            )

        if not missing_mask.index.equals(
            frame.index
        ):
            raise ValueError(
                "The original missingness mask has a "
                "different index."
            )

        missing_columns = set(
            frame.columns
        ).difference(
            missing_mask.columns
        )

        if missing_columns:
            raise ValueError(
                "The original missingness mask is missing "
                f"columns: {sorted(missing_columns)}"
            )