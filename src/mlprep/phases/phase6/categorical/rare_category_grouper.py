"""Rare-category grouping for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class RareCategoryGrouperState(BaseModel):
    """Serializable rare-category grouping state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    retained_categories: list[str] = Field(
        default_factory=list
    )

    rare_categories: list[str] = Field(
        default_factory=list
    )

    replacement_token: str = "__RARE__"
    missing_token: str = "__MISSING__"

    category_counts: dict[str, int] = Field(
        default_factory=dict
    )

    training_row_count: int = Field(
        default=0,
        ge=0,
    )

    training_unique_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class RareCategoryGrouperOptions:
    """Options controlling rare-category grouping."""

    minimum_count: int = 2
    minimum_percentage: float = 0.01

    maximum_retained_categories: int | None = None

    replacement_token: str = "__RARE__"
    missing_token: str = "__MISSING__"

    replace_unknown_categories: bool = True
    preserve_string_dtype: bool = True

    def __post_init__(self) -> None:
        """Validate rare-category grouping options."""

        if self.minimum_count < 1:
            raise ValueError(
                "minimum_count must be positive."
            )

        if not 0.0 <= self.minimum_percentage <= 1.0:
            raise ValueError(
                "minimum_percentage must be between "
                "zero and one."
            )

        if (
            self.maximum_retained_categories is not None
            and self.maximum_retained_categories < 1
        ):
            raise ValueError(
                "maximum_retained_categories must be "
                "positive when provided."
            )

        if not self.replacement_token:
            raise ValueError(
                "replacement_token cannot be empty."
            )

        if not self.missing_token:
            raise ValueError(
                "missing_token cannot be empty."
            )

        if (
            self.replacement_token
            == self.missing_token
        ):
            raise ValueError(
                "replacement_token and missing_token "
                "must be different."
            )


class RareCategoryGrouper:
    """Learn and apply rare-category grouping."""

    component_name = "rare_category_grouper"

    def __init__(
        self,
        *,
        column: str,
        options: RareCategoryGrouperOptions | None = None,
    ) -> None:
        """Initialize the rare-category grouper."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        self.column = column

        self.options = (
            options
            or RareCategoryGrouperOptions()
        )

        self.state = RareCategoryGrouperState(
            column=column,
            replacement_token=(
                self.options.replacement_token
            ),
            missing_token=(
                self.options.missing_token
            ),
        )

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the grouper is fitted."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
    ) -> RareCategoryGrouper:
        """Fit retained categories on training data."""

        self._validate_series(
            series
        )

        if self.is_fitted:
            raise RuntimeError(
                f"Rare-category grouper for {self.column!r} "
                "is already fitted."
            )

        normalized = self._normalize(
            series
        )

        counts = normalized.value_counts(
            dropna=False
        )

        row_count = len(
            normalized
        )

        retained: list[str] = []

        ordered_categories = sorted(
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

        for category, count in ordered_categories:
            if category == self.options.missing_token:
                retained.append(
                    category
                )
                continue

            percentage = (
                count / row_count
                if row_count > 0
                else 0.0
            )

            if (
                count >= self.options.minimum_count
                and percentage
                >= self.options.minimum_percentage
            ):
                retained.append(
                    category
                )

        if (
            self.options.maximum_retained_categories
            is not None
        ):
            nonmissing_retained = [
                category
                for category in retained
                if category
                != self.options.missing_token
            ]

            nonmissing_retained = (
                nonmissing_retained[
                    : self.options
                    .maximum_retained_categories
                ]
            )

            retained = nonmissing_retained

            if (
                self.options.missing_token
                in counts.index
            ):
                retained.append(
                    self.options.missing_token
                )

        all_categories = [
            category
            for category, _
            in ordered_categories
        ]

        rare_categories = [
            category
            for category in all_categories
            if category not in retained
        ]

        self.state.retained_categories = (
            retained
        )

        self.state.rare_categories = (
            rare_categories
        )

        self.state.category_counts = {
            str(category): int(count)
            for category, count
            in counts.items()
        }

        self.state.training_row_count = (
            row_count
        )

        self.state.training_unique_count = (
            len(counts)
        )

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        return self

    def transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Apply fitted rare-category grouping."""

        self._validate_series(
            series
        )

        if not self.is_fitted:
            raise RuntimeError(
                "The rare-category grouper is not fitted."
            )

        normalized = self._normalize(
            series
        )

        retained = set(
            self.state.retained_categories
        )

        transformed = normalized.map(
            lambda value: (
                value
                if value in retained
                else self.options.replacement_token
            )
        )

        transformed.name = series.name

        if self.options.preserve_string_dtype:
            return transformed.astype(
                "string"
            )

        return transformed.astype(
            "object"
        )

    def fit_transform(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Fit and transform training categories."""

        self.fit(
            series
        )

        return self.transform(
            series
        )

    def get_state(
        self,
    ) -> RareCategoryGrouperState:
        """Return a copy of fitted state."""

        return self.state.model_copy(
            deep=True
        )

    def _normalize(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Normalize categories and missing values."""

        normalized = series.astype(
            "string"
        )

        normalized = normalized.fillna(
            self.options.missing_token
        )

        return normalized

    def _validate_series(
        self,
        series: pd.Series,
    ) -> None:
        """Validate one categorical Series."""

        if not isinstance(
            series,
            pd.Series,
        ):
            raise TypeError(
                "RareCategoryGrouper input must be "
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
