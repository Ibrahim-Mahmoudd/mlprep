"""Boolean feature normalization for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class BooleanFeatureState(BaseModel):
    """Serializable Boolean feature state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    output_column: str

    true_values: list[str] = Field(
        default_factory=list
    )

    false_values: list[str] = Field(
        default_factory=list
    )

    unknown_policy: str

    unknown_value: int

    training_unknown_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class BooleanFeatureOptions:
    """Options controlling Boolean feature normalization."""

    true_values: set[str] = None  # type: ignore[assignment]
    false_values: set[str] = None  # type: ignore[assignment]

    unknown_policy: str = "error"
    unknown_value: int = -1

    missing_policy: str = "error"
    missing_value: int = -1

    output_dtype: str = "int8"

    suffix: str = "__boolean"

    def __post_init__(self) -> None:
        """Validate and initialize Boolean options."""

        if self.true_values is None:
            self.true_values = {
                "true",
                "yes",
                "y",
                "1",
                "t",
            }

        if self.false_values is None:
            self.false_values = {
                "false",
                "no",
                "n",
                "0",
                "f",
            }

        self.true_values = {
            str(value).strip().casefold()
            for value in self.true_values
        }

        self.false_values = {
            str(value).strip().casefold()
            for value in self.false_values
        }

        overlap = self.true_values.intersection(
            self.false_values
        )

        if overlap:
            raise ValueError(
                "Boolean true and false vocabularies overlap: "
                f"{sorted(overlap)}"
            )

        if self.unknown_policy not in {
            "error",
            "use_value",
        }:
            raise ValueError(
                "unknown_policy must be 'error' or 'use_value'."
            )

        if self.missing_policy not in {
            "error",
            "use_value",
        }:
            raise ValueError(
                "missing_policy must be 'error' or 'use_value'."
            )

        if not self.suffix:
            raise ValueError(
                "suffix cannot be empty."
            )


class BooleanFeatureTransformer:
    """Normalize Boolean-like values to numeric output."""

    component_name = "boolean_features"

    def __init__(
        self,
        *,
        column: str,
        output_prefix: str | None = None,
        options: BooleanFeatureOptions | None = None,
    ) -> None:
        """Initialize the Boolean feature transformer."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        self.column = column
        self.output_prefix = (
            output_prefix or column
        )

        self.options = (
            options
            or BooleanFeatureOptions()
        )

        self.state = BooleanFeatureState(
            column=column,
            output_column=(
                f"{self.output_prefix}"
                f"{self.options.suffix}"
            ),
            true_values=sorted(
                self.options.true_values
            ),
            false_values=sorted(
                self.options.false_values
            ),
            unknown_policy=(
                self.options.unknown_policy
            ),
            unknown_value=(
                self.options.unknown_value
            ),
        )

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the transformer is fitted."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
    ) -> BooleanFeatureTransformer:
        """Validate Boolean-like training values."""

        self._validate_series(
            series
        )

        encoded, unknown_mask = self._encode(
            series,
            raise_on_unknown=False,
            raise_on_missing=False,
        )

        unknown_count = int(
            unknown_mask.sum()
        )

        missing_count = int(
            series.isna().sum()
        )

        if (
            unknown_count > 0
            and self.options.unknown_policy
            == "error"
        ):
            unknown_values = sorted(
                {
                    str(value)
                    for value
                    in series.loc[
                        unknown_mask
                    ].dropna().unique()
                }
            )

            raise ValueError(
                f"Boolean feature {self.column!r} contains "
                f"unknown values: {unknown_values}"
            )

        if (
            missing_count > 0
            and self.options.missing_policy
            == "error"
        ):
            raise ValueError(
                f"Boolean feature {self.column!r} contains "
                f"{missing_count} missing values."
            )

        del encoded

        self.state.training_unknown_count = (
            unknown_count
        )

        self.state.metadata[
            "training_missing_count"
        ] = missing_count

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        return self

    def transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Transform Boolean-like values to numeric output."""

        self._validate_series(
            series
        )

        if not self.is_fitted:
            raise RuntimeError(
                "The Boolean feature transformer is not fitted."
            )

        encoded, _ = self._encode(
            series,
            raise_on_unknown=(
                self.options.unknown_policy
                == "error"
            ),
            raise_on_missing=(
                self.options.missing_policy
                == "error"
            ),
        )

        return pd.DataFrame(
            {
                self.state.output_column: (
                    encoded.astype(
                        self.options.output_dtype
                    )
                )
            },
            index=series.index,
        )

    def fit_transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Fit and transform training values."""

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
        """Return values outside the Boolean vocabulary."""

        self._validate_series(
            series
        )

        normalized = (
            series.astype(
                "string"
            )
            .str.strip()
            .str.casefold()
        )

        known = (
            self.options.true_values
            | self.options.false_values
        )

        return (
            normalized.notna()
            & ~normalized.isin(
                known
            )
        )

    def get_state(
        self,
    ) -> BooleanFeatureState:
        """Return a copy of fitted state."""

        return self.state.model_copy(
            deep=True
        )

    def _encode(
        self,
        series: pd.Series,
        *,
        raise_on_unknown: bool,
        raise_on_missing: bool,
    ) -> tuple[
        pd.Series,
        pd.Series,
    ]:
        """Encode Boolean-like values."""

        normalized = (
            series.astype(
                "string"
            )
            .str.strip()
            .str.casefold()
        )

        missing_mask = normalized.isna()

        true_mask = normalized.isin(
            self.options.true_values
        )

        false_mask = normalized.isin(
            self.options.false_values
        )

        unknown_mask = (
            normalized.notna()
            & ~true_mask
            & ~false_mask
        )

        if (
            raise_on_unknown
            and bool(
                unknown_mask.any()
            )
        ):
            unknown_values = sorted(
                {
                    str(value)
                    for value
                    in series.loc[
                        unknown_mask
                    ].unique()
                }
            )

            raise ValueError(
                f"Unknown Boolean-like values for "
                f"{self.column!r}: {unknown_values}"
            )

        if (
            raise_on_missing
            and bool(
                missing_mask.any()
            )
        ):
            raise ValueError(
                f"Missing Boolean values remain in "
                f"{self.column!r}."
            )

        encoded = pd.Series(
            self.options.unknown_value,
            index=series.index,
            dtype="int64",
        )

        encoded.loc[
            true_mask
        ] = 1

        encoded.loc[
            false_mask
        ] = 0

        if self.options.missing_policy == "use_value":
            encoded.loc[
                missing_mask
            ] = self.options.missing_value

        return (
            encoded,
            unknown_mask,
        )

    def _validate_series(
        self,
        series: pd.Series,
    ) -> None:
        """Validate one Boolean Series."""

        if not isinstance(
            series,
            pd.Series,
        ):
            raise TypeError(
                "BooleanFeatureTransformer input must be "
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