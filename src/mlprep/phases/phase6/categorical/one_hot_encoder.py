"""One-hot categorical encoding for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class OneHotEncoderState(BaseModel):
    """Serializable one-hot encoder state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    categories: list[str] = Field(
        default_factory=list
    )

    encoded_categories: list[str] = Field(
        default_factory=list
    )

    output_columns: list[str] = Field(
        default_factory=list
    )

    dropped_category: str | None = None

    unknown_token: str = "__UNKNOWN__"
    missing_token: str = "__MISSING__"

    training_row_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class OneHotEncoderOptions:
    """Options controlling one-hot encoding."""

    drop_first: bool = False

    handle_unknown: str = "indicator"

    unknown_token: str = "__UNKNOWN__"
    missing_token: str = "__MISSING__"

    output_dtype: str = "float64"

    separator: str = "__onehot__"

    maximum_categories: int | None = None

    def __post_init__(self) -> None:
        """Validate one-hot options."""

        if self.handle_unknown not in {
            "ignore",
            "indicator",
            "error",
        }:
            raise ValueError(
                "handle_unknown must be 'ignore', "
                "'indicator', or 'error'."
            )

        if self.output_dtype not in {
            "float32",
            "float64",
            "int8",
            "uint8",
        }:
            raise ValueError(
                "Unsupported one-hot output dtype."
            )

        if not self.separator:
            raise ValueError(
                "separator cannot be empty."
            )

        if (
            self.maximum_categories is not None
            and self.maximum_categories < 1
        ):
            raise ValueError(
                "maximum_categories must be positive."
            )


class OneHotEncoder:
    """Fit and apply deterministic one-hot encoding."""

    component_name = "one_hot_encoder"

    def __init__(
        self,
        *,
        column: str,
        output_prefix: str | None = None,
        options: OneHotEncoderOptions | None = None,
    ) -> None:
        """Initialize the encoder."""

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
            or OneHotEncoderOptions()
        )

        self.state = OneHotEncoderState(
            column=column,
            unknown_token=(
                self.options.unknown_token
            ),
            missing_token=(
                self.options.missing_token
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
    ) -> OneHotEncoder:
        """Learn category vocabulary from training data."""

        normalized = self._normalize(
            series
        )

        counts = normalized.value_counts()

        categories = sorted(
            [
                str(category)
                for category in counts.index
            ]
        )

        if (
            self.options.maximum_categories
            is not None
            and len(categories)
            > self.options.maximum_categories
        ):
            ordered = sorted(
                (
                    (
                        str(category),
                        int(count),
                    )
                    for category, count
                    in counts.items()
                ),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )

            categories = [
                category
                for category, _
                in ordered[
                    : self.options
                    .maximum_categories
                ]
            ]

        if (
            self.options.handle_unknown
            == "indicator"
            and self.options.unknown_token
            not in categories
        ):
            categories.append(
                self.options.unknown_token
            )

        dropped_category = None

        encoded_categories = list(
            categories
        )

        if (
            self.options.drop_first
            and encoded_categories
        ):
            dropped_category = (
                encoded_categories.pop(0)
            )

        output_columns = [
            self._column_name(
                category
            )
            for category in encoded_categories
        ]

        if len(output_columns) != len(
            set(output_columns)
        ):
            raise ValueError(
                "One-hot output-column collision detected."
            )

        self.state.categories = (
            categories
        )

        self.state.encoded_categories = (
            encoded_categories
        )

        self.state.output_columns = (
            output_columns
        )

        self.state.dropped_category = (
            dropped_category
        )

        self.state.training_row_count = len(
            series
        )

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        return self

    def transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Apply the fitted one-hot vocabulary."""

        if not self.is_fitted:
            raise RuntimeError(
                "The one-hot encoder is not fitted."
            )

        normalized = self._normalize(
            series
        )

        known_categories = set(
            self.state.categories
        )

        unknown_mask = (
            ~normalized.isin(
                known_categories
            )
        )

        if bool(
            unknown_mask.any()
        ):
            if self.options.handle_unknown == "error":
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
                    f"Unknown categories for {self.column!r}: "
                    f"{unknown_values}"
                )

            if (
                self.options.handle_unknown
                == "indicator"
            ):
                normalized.loc[
                    unknown_mask
                ] = self.options.unknown_token

        output = pd.DataFrame(
            index=series.index
        )

        for category, output_column in zip(
            self.state.encoded_categories,
            self.state.output_columns,
            strict=True,
        ):
            output[
                output_column
            ] = (
                normalized
                .eq(category)
                .astype(
                    self.options.output_dtype
                )
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
        """Return categories unseen during fitting."""

        if not self.is_fitted:
            raise RuntimeError(
                "The one-hot encoder is not fitted."
            )

        normalized = self._normalize(
            series
        )

        return ~normalized.isin(
            set(
                self.state.categories
            )
        )

    def get_state(
        self,
    ) -> OneHotEncoderState:
        """Return a copy of fitted state."""

        return self.state.model_copy(
            deep=True
        )

    def _normalize(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Normalize categorical values."""

        if not isinstance(
            series,
            pd.Series,
        ):
            raise TypeError(
                "OneHotEncoder input must be "
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

        return (
            series.astype(
                "string"
            )
            .fillna(
                self.options.missing_token
            )
        )

    def _column_name(
        self,
        category: str,
    ) -> str:
        """Build one deterministic output-column name."""

        safe_category = (
            category
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        if not safe_category:
            safe_category = "EMPTY"

        return (
            f"{self.output_prefix}"
            f"{self.options.separator}"
            f"{safe_category}"
        )
