"""Text-structure feature extraction for Phase 6."""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

TextOutputDType = Literal["float32", "float64", "int64"]


class TextFeatureState(BaseModel):
    """Serializable text feature-extractor state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    generated_features: list[str] = Field(
        default_factory=list
    )

    output_columns: list[str] = Field(
        default_factory=list
    )

    missing_replacement: str = ""

    training_row_count: int = Field(
        default=0,
        ge=0,
    )

    training_missing_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class TextFeatureOptions:
    """Options controlling structural text features."""

    features: list[str] = field(
        default_factory=lambda: [
            "character_count",
            "word_count",
            "digit_count",
            "uppercase_count",
            "lowercase_count",
            "whitespace_count",
            "punctuation_count",
            "unique_word_count",
            "average_word_length",
        ]
    )

    missing_replacement: str = ""

    strip_before_counting: bool = False

    output_dtype: TextOutputDType = "float64"

    separator: str = "__text__"

    def __post_init__(self) -> None:
        """Validate text-feature options."""

        supported_features = {
            "character_count",
            "word_count",
            "digit_count",
            "uppercase_count",
            "lowercase_count",
            "whitespace_count",
            "punctuation_count",
            "unique_word_count",
            "average_word_length",
        }

        unsupported = set(
            self.features
        ).difference(
            supported_features
        )

        if unsupported:
            raise ValueError(
                "Unsupported text features: "
                f"{sorted(unsupported)}"
            )

        if len(self.features) != len(
            set(self.features)
        ):
            raise ValueError(
                "Text features contain duplicates."
            )

        if self.output_dtype not in {
            "float32",
            "float64",
            "int64",
        }:
            raise ValueError(
                "Unsupported text output dtype."
            )

        if not self.separator:
            raise ValueError(
                "separator cannot be empty."
            )


class TextFeatureExtractor:
    """Generate deterministic text-structure features."""

    component_name = "text_features"

    def __init__(
        self,
        *,
        column: str,
        output_prefix: str | None = None,
        options: TextFeatureOptions | None = None,
    ) -> None:
        """Initialize the text feature extractor."""

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
            or TextFeatureOptions()
        )

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
                "Text output-column collision detected."
            )

        self.state = TextFeatureState(
            column=column,
            generated_features=list(
                self.options.features
            ),
            output_columns=output_columns,
            missing_replacement=(
                self.options
                .missing_replacement
            ),
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
    ) -> TextFeatureExtractor:
        """Fit text feature metadata on training data."""

        self._validate_series(
            series
        )

        if self.is_fitted:
            raise RuntimeError(
                f"Text extractor for {self.column!r} "
                "is already fitted."
            )

        normalized = self._normalize(
            series
        )

        self.state.training_row_count = len(
            normalized
        )

        self.state.training_missing_count = int(
            series.isna().sum()
        )

        self.state.metadata.update(
            {
                "maximum_character_count": int(
                    normalized.str.len().max()
                )
                if len(normalized) > 0
                else 0,
                "empty_string_count": int(
                    normalized.eq("").sum()
                ),
            }
        )

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        return self

    def transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Extract deterministic text features."""

        self._validate_series(
            series
        )

        if not self.is_fitted:
            raise RuntimeError(
                "The text feature extractor is not fitted."
            )

        normalized = self._normalize(
            series
        )

        output = pd.DataFrame(
            index=series.index
        )

        for feature in self.options.features:
            output[
                self._output_column(
                    feature
                )
            ] = self._extract_feature(
                normalized=normalized,
                feature=feature,
            )

        if list(output.columns) != (
            self.state.output_columns
        ):
            raise ValueError(
                "Text output schema differs from "
                "the fitted training schema."
            )

        return output

    def fit_transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Fit and transform training text."""

        self.fit(
            series
        )

        return self.transform(
            series
        )

    def get_state(
        self,
    ) -> TextFeatureState:
        """Return a copy of fitted state."""

        return self.state.model_copy(
            deep=True
        )

    def _extract_feature(
        self,
        *,
        normalized: pd.Series,
        feature: str,
    ) -> pd.Series:
        """Extract one structural text feature."""

        if feature == "character_count":
            values = normalized.str.len()

        elif feature == "word_count":
            values = normalized.map(
                lambda value: len(
                    self._words(value)
                )
            )

        elif feature == "digit_count":
            values = normalized.map(
                lambda value: sum(
                    character.isdigit()
                    for character in value
                )
            )

        elif feature == "uppercase_count":
            values = normalized.map(
                lambda value: sum(
                    character.isupper()
                    for character in value
                )
            )

        elif feature == "lowercase_count":
            values = normalized.map(
                lambda value: sum(
                    character.islower()
                    for character in value
                )
            )

        elif feature == "whitespace_count":
            values = normalized.map(
                lambda value: sum(
                    character.isspace()
                    for character in value
                )
            )

        elif feature == "punctuation_count":
            punctuation = set(
                string.punctuation
            )

            values = normalized.map(
                lambda value: sum(
                    character in punctuation
                    for character in value
                )
            )

        elif feature == "unique_word_count":
            values = normalized.map(
                lambda value: len(
                    set(
                        self._words(value)
                    )
                )
            )

        elif feature == "average_word_length":
            values = normalized.map(
                self._average_word_length
            )

        else:
            raise ValueError(
                f"Unsupported text feature: "
                f"{feature!r}"
            )

        return values.astype(
            self.options.output_dtype
        )

    def _normalize(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Normalize source text."""

        normalized = (
            series.astype(
                "string"
            )
            .fillna(
                self.options
                .missing_replacement
            )
        )

        if self.options.strip_before_counting:
            normalized = normalized.str.strip()

        return normalized

    @staticmethod
    def _words(
        value: str,
    ) -> list[str]:
        """Split text into whitespace-delimited words."""

        return [
            word
            for word in value.split()
            if word
        ]

    @classmethod
    def _average_word_length(
        cls,
        value: str,
    ) -> float:
        """Calculate average word length."""

        words = cls._words(
            value
        )

        if not words:
            return 0.0

        return (
            sum(
                len(word)
                for word in words
            )
            / len(words)
        )

    def _output_column(
        self,
        feature: str,
    ) -> str:
        """Build one text output-column name."""

        return (
            f"{self.output_prefix}"
            f"{self.options.separator}"
            f"{feature}"
        )

    def _validate_series(
        self,
        series: pd.Series,
    ) -> None:
        """Validate one text Series."""

        if not isinstance(
            series,
            pd.Series,
        ):
            raise TypeError(
                "TextFeatureExtractor input must be "
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
