"""Datetime feature extraction for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DatetimeFeatureState(BaseModel):
    """Serializable fitted datetime feature state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    column: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    reference_timestamp: str | None = None

    generated_features: list[str] = Field(
        default_factory=list
    )

    output_columns: list[str] = Field(
        default_factory=list
    )

    training_row_count: int = Field(
        default=0,
        ge=0,
    )

    training_missing_count: int = Field(
        default=0,
        ge=0,
    )

    training_invalid_count: int = Field(
        default=0,
        ge=0,
    )

    input_dtype: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class DatetimeFeatureOptions:
    """Options controlling datetime feature extraction.

    Cyclical features encode hour (period 24), weekday (period 7), or
    month (period 12) using sine/cosine pairs. They remain floating-point
    even when ``output_dtype="int64"`` is selected.
    """

    features: list[str] = field(
        default_factory=lambda: [
            "year",
            "quarter",
            "month",
            "week",
            "day",
            "day_of_week",
            "day_of_year",
            "hour",
            "minute",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "elapsed_seconds",
        ]
    )

    day_first: bool = False
    year_first: bool = False

    utc: bool = False

    reference_method: str = "minimum"

    explicit_reference: str | pd.Timestamp | None = None

    allow_missing_values: bool = False
    allow_invalid_values: bool = False

    output_dtype: str = "float64"

    separator: str = "__datetime__"

    def __post_init__(self) -> None:
        """Validate datetime feature options."""

        supported_features = {
            "year",
            "quarter",
            "month",
            "week",
            "day",
            "day_of_week",
            "day_of_year",
            "hour",
            "minute",
            "second",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "elapsed_seconds",
            "elapsed_days",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
        }

        unsupported = set(
            self.features
        ).difference(
            supported_features
        )

        if unsupported:
            raise ValueError(
                "Unsupported datetime features: "
                f"{sorted(unsupported)}"
            )

        if len(self.features) != len(
            set(self.features)
        ):
            raise ValueError(
                "Datetime features contain duplicates."
            )

        if self.reference_method not in {
            "minimum",
            "maximum",
            "median",
            "explicit",
        }:
            raise ValueError(
                "reference_method must be 'minimum', "
                "'maximum', 'median', or 'explicit'."
            )

        if (
            self.reference_method == "explicit"
            and self.explicit_reference is None
        ):
            raise ValueError(
                "explicit_reference is required when "
                "reference_method is 'explicit'."
            )

        if self.output_dtype not in {
            "float32",
            "float64",
            "int64",
        }:
            raise ValueError(
                "Unsupported datetime output dtype."
            )

        if not self.separator:
            raise ValueError(
                "separator cannot be empty."
            )


class DatetimeFeatureExtractor:
    """Fit and extract deterministic datetime features."""

    component_name = "datetime_features"

    def __init__(
        self,
        *,
        column: str,
        output_prefix: str | None = None,
        options: DatetimeFeatureOptions | None = None,
    ) -> None:
        """Initialize the datetime feature extractor."""

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
            or DatetimeFeatureOptions()
        )

        self.reference_timestamp: (
            pd.Timestamp | None
        ) = None

        output_columns = [
            self._output_column(
                feature
            )
            for feature in self.options.features
        ]

        if len(output_columns) != len(
            set(output_columns)
        ):
            raise ValueError(
                "Datetime output-column collision detected."
            )

        self.state = DatetimeFeatureState(
            column=column,
            generated_features=list(
                self.options.features
            ),
            output_columns=output_columns,
        )

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the extractor is fitted."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
    ) -> DatetimeFeatureExtractor:
        """Fit datetime reference state on training data."""

        self._validate_series(
            series
        )

        if self.is_fitted:
            raise RuntimeError(
                f"Datetime extractor for {self.column!r} "
                "is already fitted."
            )

        parsed = self._parse(
            series
        )

        missing_mask = series.isna()

        invalid_mask = (
            series.notna()
            & parsed.isna()
        )

        missing_count = int(
            missing_mask.sum()
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        if (
            missing_count > 0
            and not self.options
            .allow_missing_values
        ):
            raise ValueError(
                f"Datetime feature {self.column!r} contains "
                f"{missing_count} missing values."
            )

        if (
            invalid_count > 0
            and not self.options
            .allow_invalid_values
        ):
            raise ValueError(
                f"Datetime feature {self.column!r} contains "
                f"{invalid_count} invalid values."
            )

        valid = parsed.dropna()

        elapsed_features_requested = bool(
            {
                "elapsed_seconds",
                "elapsed_days",
            }.intersection(
                self.options.features
            )
        )

        if elapsed_features_requested:
            if valid.empty:
                raise ValueError(
                    "Elapsed datetime features require at least "
                    "one valid training timestamp."
                )

            self.reference_timestamp = (
                self._resolve_reference(
                    valid
                )
            )

            self.state.reference_timestamp = (
                self.reference_timestamp
                .isoformat()
            )

        self.state.input_dtype = str(
            series.dtype
        )

        self.state.training_row_count = len(
            series
        )

        self.state.training_missing_count = (
            missing_count
        )

        self.state.training_invalid_count = (
            invalid_count
        )

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        return self

    def transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Extract fitted datetime features."""

        self._validate_series(
            series
        )

        if not self.is_fitted:
            raise RuntimeError(
                "The datetime feature extractor is not fitted."
            )

        parsed = self._parse(
            series
        )

        missing_count = int(
            parsed.isna().sum()
        )

        if (
            missing_count > 0
            and not self.options.allow_missing_values
            and not self.options.allow_invalid_values
        ):
            raise ValueError(
                f"Datetime transformation for {self.column!r} "
                f"contains {missing_count} unresolved values."
            )

        output = pd.DataFrame(
            index=series.index
        )

        for feature in self.options.features:
            output_column = self._output_column(
                feature
            )

            output[
                output_column
            ] = self._extract_feature(
                parsed=parsed,
                feature=feature,
            )

        if list(output.columns) != (
            self.state.output_columns
        ):
            raise ValueError(
                "Datetime output schema differs from "
                "the fitted training schema."
            )

        return output

    def fit_transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Fit and transform training datetime values."""

        self.fit(
            series
        )

        return self.transform(
            series
        )

    def get_state(
        self,
    ) -> DatetimeFeatureState:
        """Return a copy of fitted state."""

        return self.state.model_copy(
            deep=True
        )

    def _extract_feature(
        self,
        *,
        parsed: pd.Series,
        feature: str,
    ) -> pd.Series:
        """Extract one datetime feature."""

        if feature == "year":
            values = parsed.dt.year

        elif feature == "quarter":
            values = parsed.dt.quarter

        elif feature == "month":
            values = parsed.dt.month

        elif feature == "week":
            values = (
                parsed.dt.isocalendar().week
            )

        elif feature == "day":
            values = parsed.dt.day

        elif feature == "day_of_week":
            values = parsed.dt.dayofweek

        elif feature == "day_of_year":
            values = parsed.dt.dayofyear

        elif feature == "hour":
            values = parsed.dt.hour

        elif feature == "minute":
            values = parsed.dt.minute

        elif feature == "second":
            values = parsed.dt.second

        elif feature == "is_weekend":
            values = (
                parsed.dt.dayofweek
                .isin(
                    {
                        5,
                        6,
                    }
                )
                .astype(float)
            )

            values.loc[
                parsed.isna()
            ] = np.nan

        elif feature == "is_month_start":
            values = (
                parsed.dt.is_month_start
                .astype(float)
            )

            values.loc[
                parsed.isna()
            ] = np.nan

        elif feature == "is_month_end":
            values = (
                parsed.dt.is_month_end
                .astype(float)
            )

            values.loc[
                parsed.isna()
            ] = np.nan

        elif feature == "elapsed_seconds":
            reference = self._fitted_reference()

            values = (
                parsed - reference
            ).dt.total_seconds()

        elif feature == "elapsed_days":
            reference = self._fitted_reference()

            values = (
                parsed - reference
            ).dt.total_seconds() / 86400.0

        elif feature in {
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
        }:
            component, function = feature.rsplit(
                "_",
                maxsplit=1,
            )
            component_values, period = {
                "hour": (
                    parsed.dt.hour,
                    24,
                ),
                "day_of_week": (
                    parsed.dt.dayofweek,
                    7,
                ),
                "month": (
                    parsed.dt.month - 1,
                    12,
                ),
            }[component]
            angle = (
                2.0
                * np.pi
                * component_values
                / period
            )
            values = (
                np.sin(angle)
                if function == "sin"
                else np.cos(angle)
            )

            return values.astype(
                self.options.output_dtype
                if self.options.output_dtype != "int64"
                else "float64"
            )

        else:
            raise ValueError(
                f"Unsupported datetime feature: "
                f"{feature!r}"
            )

        if self.options.output_dtype == "int64":
            if values.isna().any():
                return values.astype(
                    "Float64"
                )

            return values.astype(
                "int64"
            )

        return values.astype(
            self.options.output_dtype
        )

    def _resolve_reference(
        self,
        valid: pd.Series,
    ) -> pd.Timestamp:
        """Resolve the fitted training reference timestamp."""

        if self.options.reference_method == "minimum":
            return pd.Timestamp(
                valid.min()
            )

        if self.options.reference_method == "maximum":
            return pd.Timestamp(
                valid.max()
            )

        if self.options.reference_method == "median":
            integer_values = valid.astype(
                "int64"
            )

            return pd.Timestamp(
                int(
                    integer_values.median()
                )
            )

        return pd.Timestamp(
            pd.to_datetime(
                self.options.explicit_reference,
                errors="raise",
                utc=self.options.utc,
            )
        )

    def _fitted_reference(
        self,
    ) -> pd.Timestamp:
        """Return the fitted reference timestamp."""

        if self.reference_timestamp is not None:
            return self.reference_timestamp

        if self.state.reference_timestamp is None:
            raise RuntimeError(
                "The fitted datetime reference is unavailable."
            )

        self.reference_timestamp = pd.Timestamp(
            self.state.reference_timestamp
        )

        return self.reference_timestamp

    def _parse(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Parse input values as datetimes."""

        return pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=self.options.day_first,
            yearfirst=self.options.year_first,
            utc=self.options.utc,
        )

    def _output_column(
        self,
        feature: str,
    ) -> str:
        """Build one datetime output-column name."""

        return (
            f"{self.output_prefix}"
            f"{self.options.separator}"
            f"{feature}"
        )

    def _validate_series(
        self,
        series: pd.Series,
    ) -> None:
        """Validate one datetime Series."""

        if not isinstance(
            series,
            pd.Series,
        ):
            raise TypeError(
                "DatetimeFeatureExtractor input must be "
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
