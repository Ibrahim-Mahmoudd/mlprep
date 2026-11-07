"""Explicit ordinal encoding for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class OrdinalEncoderState(BaseModel):
    """Serializable ordinal encoder state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    category_order: list[str] = Field(
        default_factory=list
    )

    category_mapping: dict[
        str,
        int,
    ] = Field(default_factory=dict)

    unknown_value: int = -1
    missing_value: int = -2

    output_column: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class OrdinalEncoderOptions:
    """Options controlling ordinal encoding."""

    unknown_value: int = -1
    missing_value: int = -2

    handle_unknown: str = "use_encoded_value"

    missing_token: str = "__MISSING__"

    output_dtype: str = "int64"

    suffix: str = "__ordinal"

    def __post_init__(self) -> None:
        """Validate ordinal options."""

        if self.handle_unknown not in {
            "use_encoded_value",
            "error",
        }:
            raise ValueError(
                "handle_unknown must be "
                "'use_encoded_value' or 'error'."
            )

        if (
            self.unknown_value
            == self.missing_value
        ):
            raise ValueError(
                "unknown_value and missing_value "
                "must be different."
            )

        if not self.suffix:
            raise ValueError(
                "suffix cannot be empty."
            )


class OrdinalEncoder:
    """Encode categories using an explicit semantic order."""

    component_name = "ordinal_encoder"

    def __init__(
        self,
        *,
        column: str,
        category_order: list[str],
        output_prefix: str | None = None,
        options: OrdinalEncoderOptions | None = None,
    ) -> None:
        """Initialize the ordinal encoder."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        if not category_order:
            raise ValueError(
                "category_order cannot be empty."
            )

        normalized_order = [
            str(category)
            for category in category_order
        ]

        if len(normalized_order) != len(
            set(normalized_order)
        ):
            raise ValueError(
                "category_order contains duplicate values."
            )

        self.column = column
        self.category_order = (
            normalized_order
        )

        self.output_prefix = (
            output_prefix or column
        )

        self.options = (
            options
            or OrdinalEncoderOptions()
        )

        self.state = OrdinalEncoderState(
            column=column,
            output_column=(
                f"{self.output_prefix}"
                f"{self.options.suffix}"
            ),
            unknown_value=(
                self.options.unknown_value
            ),
            missing_value=(
                self.options.missing_value
            ),
        )

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the encoder is fitted."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
    ) -> OrdinalEncoder:
        """Validate training categories and store mapping."""

        self._validate_series(
            series
        )

        mapping = {
            category: index
            for index, category
            in enumerate(
                self.category_order
            )
        }

        normalized = series.astype(
            "string"
        )

        known_values = set(
            mapping
        )

        observed_values = {
            str(value)
            for value
            in normalized.dropna().unique()
        }

        unknown_training_values = sorted(
            observed_values.difference(
                known_values
            )
        )

        if (
            unknown_training_values
            and self.options.handle_unknown
            == "error"
        ):
            raise ValueError(
                f"Training data for {self.column!r} "
                "contains values absent from the explicit "
                f"ordinal order: {unknown_training_values}"
            )

        self.state.category_order = list(
            self.category_order
        )

        self.state.category_mapping = (
            mapping
        )

        self.state.metadata[
            "unknown_training_values"
        ] = unknown_training_values

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        return self

    def transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Apply fitted ordinal mapping."""

        self._validate_series(
            series
        )

        if not self.is_fitted:
            raise RuntimeError(
                "The ordinal encoder is not fitted."
            )

        normalized = series.astype(
            "string"
        )

        missing_mask = normalized.isna()

        encoded = normalized.map(
            self.state.category_mapping
        )

        unknown_mask = (
            normalized.notna()
            & encoded.isna()
        )

        if (
            bool(
                unknown_mask.any()
            )
            and self.options.handle_unknown
            == "error"
        ):
            unknown_values = sorted(
                {
                    str(value)
                    for value
                    in normalized.loc[
                        unknown_mask
                    ].unique()
                }
            )

            raise ValueError(
                f"Unknown ordinal categories for "
                f"{self.column!r}: {unknown_values}"
            )

        encoded = encoded.fillna(
            self.options.unknown_value
        )

        encoded.loc[
            missing_mask
        ] = self.options.missing_value

        output = pd.DataFrame(
            {
                self.state.output_column: (
                    encoded.astype(
                        self.options.output_dtype
                    )
                )
            },
            index=series.index,
        )

        return output

    def fit_transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Fit and transform training categories."""

        self.fit(
            series
        )

        return self.transform(
            series
        )

    def unknown_mask(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Return categories outside the explicit order."""

        if not self.is_fitted:
            raise RuntimeError(
                "The ordinal encoder is not fitted."
            )

        normalized = series.astype(
            "string"
        )

        return (
            normalized.notna()
            & ~normalized.isin(
                set(
                    self.state.category_mapping
                )
            )
        )

    def get_state(
        self,
    ) -> OrdinalEncoderState:
        """Return a copy of fitted state."""

        return self.state.model_copy(
            deep=True
        )

    def _validate_series(
        self,
        series: pd.Series,
    ) -> None:
        """Validate one input Series."""

        if not isinstance(
            series,
            pd.Series,
        ):
            raise TypeError(
                "OrdinalEncoder input must be "
                "a pandas Series."
            )

        if (
            series.name is not None
            and str(series.name) != self.column
        ):
            raise ValueError(
                f"Expected column {self.column!r}, "
                f"received {series.name!r}."
            )
