"""Deterministic categorical hashing for Phase 6."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class HashingEncoderState(BaseModel):
    """Serializable hashing encoder state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    bucket_count: int = Field(
        ge=2
    )

    seed: str

    output_columns: list[str] = Field(
        default_factory=list
    )

    training_unique_count: int = Field(
        default=0,
        ge=0,
    )

    estimated_collision_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class HashingEncoderOptions:
    """Options controlling deterministic hash encoding."""

    bucket_count: int = 16

    seed: str = "mlprep-phase6"

    missing_token: str = "__MISSING__"

    signed_hashing: bool = False

    output_dtype: str = "float64"

    separator: str = "__hash__"

    def __post_init__(self) -> None:
        """Validate hashing options."""

        if self.bucket_count < 2:
            raise ValueError(
                "bucket_count must be at least two."
            )

        if not self.seed:
            raise ValueError(
                "seed cannot be empty."
            )

        if not self.separator:
            raise ValueError(
                "separator cannot be empty."
            )


class HashingEncoder:
    """Encode categories into deterministic hash buckets."""

    component_name = "hashing_encoder"

    def __init__(
        self,
        *,
        column: str,
        output_prefix: str | None = None,
        options: HashingEncoderOptions | None = None,
    ) -> None:
        """Initialize the hashing encoder."""

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
            or HashingEncoderOptions()
        )

        output_columns = [
            (
                f"{self.output_prefix}"
                f"{self.options.separator}"
                f"{bucket}"
            )
            for bucket in range(
                self.options.bucket_count
            )
        ]

        self.state = HashingEncoderState(
            column=column,
            bucket_count=(
                self.options.bucket_count
            ),
            seed=self.options.seed,
            output_columns=output_columns,
        )

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the encoder was initialized on training."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
    ) -> HashingEncoder:
        """Record training collision metadata."""

        normalized = self._normalize(
            series
        )

        unique_values = sorted(
            {
                str(value)
                for value
                in normalized.unique()
            }
        )

        occupied_buckets = {
            self._bucket(value)
            for value in unique_values
        }

        self.state.training_unique_count = (
            len(unique_values)
        )

        self.state.estimated_collision_count = max(
            0,
            len(unique_values)
            - len(occupied_buckets),
        )

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        return self

    def transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Apply deterministic hashing."""

        if not self.is_fitted:
            raise RuntimeError(
                "The hashing encoder is not fitted."
            )

        normalized = self._normalize(
            series
        )

        output = pd.DataFrame(
            0.0,
            index=series.index,
            columns=self.state.output_columns,
            dtype=self.options.output_dtype,
        )

        for position, value in enumerate(normalized):
            bucket = self._bucket(
                str(value)
            )

            sign = (
                self._sign(
                    str(value)
                )
                if self.options.signed_hashing
                else 1.0
            )

            output.iat[
                position,
                bucket,
            ] += sign

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

    def get_state(
        self,
    ) -> HashingEncoderState:
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
                "HashingEncoder input must be "
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

    def _bucket(
        self,
        value: str,
    ) -> int:
        """Return a deterministic bucket index."""

        payload = (
            f"{self.options.seed}"
            f"|{self.column}"
            f"|{value}"
        ).encode()

        digest = hashlib.blake2b(
            payload,
            digest_size=8,
        ).digest()

        integer = int.from_bytes(
            digest,
            byteorder="big",
            signed=False,
        )

        return (
            integer
            % self.options.bucket_count
        )

    def _sign(
        self,
        value: str,
    ) -> float:
        """Return a deterministic hash sign."""

        payload = (
            f"{self.options.seed}"
            f"|sign"
            f"|{self.column}"
            f"|{value}"
        ).encode()

        digest = hashlib.blake2b(
            payload,
            digest_size=1,
        ).digest()

        return (
            1.0
            if digest[0] % 2 == 0
            else -1.0
        )
