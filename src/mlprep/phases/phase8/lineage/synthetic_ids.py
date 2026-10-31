"""Deterministic output row-ID generation for Phase 8."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GeneratedRowIdKind(StrEnum):
    """Kinds of generated Phase 8 row identifiers."""

    SYNTHETIC = "synthetic"
    DUPLICATE = "duplicate"


class GeneratedRowIdRecord(BaseModel):
    """Metadata for one generated row identifier."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    row_id: str

    kind: GeneratedRowIdKind

    strategy: str

    sequence: int = Field(
        ge=1,
    )

    execution_id: str | None = None

    deterministic: bool = True

    collision_attempts: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SyntheticIdGeneratorState(BaseModel):
    """Serializable state for deterministic ID generation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    synthetic_prefix: str
    duplicate_prefix: str

    execution_id: str

    random_seed: int

    include_strategy: bool = True
    include_execution_token: bool = True
    deterministic: bool = True

    synthetic_counter: int = Field(
        default=0,
        ge=0,
    )

    duplicate_counter: int = Field(
        default=0,
        ge=0,
    )

    collision_count: int = Field(
        default=0,
        ge=0,
    )

    generated_ids: list[str] = Field(
        default_factory=list,
    )

    generated_records: list[
        GeneratedRowIdRecord
    ] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class SyntheticIdGeneratorOptions:
    """Options controlling generated Phase 8 row IDs."""

    synthetic_prefix: str = "phase8__synthetic"
    duplicate_prefix: str = "phase8__duplicate"

    include_strategy: bool = True
    include_execution_token: bool = True

    sequence_width: int = 8

    execution_token_length: int = 12

    maximum_collision_attempts: int = 1000

    normalize_strategy_name: bool = True

    reject_prefix_collision: bool = True

    def __post_init__(self) -> None:
        """Validate row-ID generation options."""

        self.synthetic_prefix = (
            self.synthetic_prefix.strip()
        )

        self.duplicate_prefix = (
            self.duplicate_prefix.strip()
        )

        if not self.synthetic_prefix:
            raise ValueError(
                "synthetic_prefix cannot be empty."
            )

        if not self.duplicate_prefix:
            raise ValueError(
                "duplicate_prefix cannot be empty."
            )

        if (
            self.reject_prefix_collision
            and self.synthetic_prefix
            == self.duplicate_prefix
        ):
            raise ValueError(
                "synthetic_prefix and duplicate_prefix "
                "must be different."
            )

        if self.sequence_width < 1:
            raise ValueError(
                "sequence_width must be positive."
            )

        if self.execution_token_length < 4:
            raise ValueError(
                "execution_token_length must be at least four."
            )

        if self.maximum_collision_attempts < 1:
            raise ValueError(
                "maximum_collision_attempts must be positive."
            )


class SyntheticIdGenerator:
    """Generate collision-safe synthetic and duplicate row IDs."""

    component_name = "synthetic_ids"

    def __init__(
        self,
        *,
        execution_id: str,
        random_seed: int,
        existing_row_ids: list[object] | set[object],
        options: SyntheticIdGeneratorOptions | None = None,
        deterministic: bool = True,
    ) -> None:
        """Initialize the row-ID generator."""

        self.options = (
            options
            or SyntheticIdGeneratorOptions()
        )

        normalized_execution_id = str(
            execution_id
        ).strip()

        if not normalized_execution_id:
            raise ValueError(
                "execution_id cannot be empty."
            )

        self.execution_id = (
            normalized_execution_id
        )

        self.random_seed = random_seed
        self.deterministic = deterministic

        self._existing_ids = {
            str(
                value
            )
            for value in existing_row_ids
        }

        self._generated_ids: set[str] = set()

        self._execution_token = (
            self._build_execution_token()
        )

        self.state = SyntheticIdGeneratorState(
            synthetic_prefix=(
                self.options.synthetic_prefix
            ),
            duplicate_prefix=(
                self.options.duplicate_prefix
            ),
            execution_id=self.execution_id,
            random_seed=self.random_seed,
            include_strategy=(
                self.options.include_strategy
            ),
            include_execution_token=(
                self.options
                .include_execution_token
            ),
            deterministic=self.deterministic,
            metadata={
                "sequence_width": (
                    self.options.sequence_width
                ),
                "execution_token_length": (
                    self.options
                    .execution_token_length
                ),
                "original_id_count": len(
                    self._existing_ids
                ),
                "execution_token": (
                    self._execution_token
                ),
            },
        )

    def next_synthetic_id(
        self,
        *,
        strategy: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate the next synthetic row identifier."""

        return self._next_id(
            kind=GeneratedRowIdKind.SYNTHETIC,
            strategy=strategy,
            metadata=metadata,
        )

    def next_duplicate_id(
        self,
        *,
        strategy: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Generate the next duplicate row identifier."""

        return self._next_id(
            kind=GeneratedRowIdKind.DUPLICATE,
            strategy=strategy,
            metadata=metadata,
        )

    def generate_many(
        self,
        *,
        kind: GeneratedRowIdKind,
        strategy: str,
        count: int,
    ) -> list[str]:
        """Generate multiple row identifiers."""

        if count < 0:
            raise ValueError(
                "count cannot be negative."
            )

        resolved_kind = (
            kind
            if isinstance(
                kind,
                GeneratedRowIdKind,
            )
            else GeneratedRowIdKind(
                kind
            )
        )

        generated: list[str] = []

        for _ in range(
            count
        ):
            if (
                resolved_kind
                == GeneratedRowIdKind.SYNTHETIC
            ):
                generated.append(
                    self.next_synthetic_id(
                        strategy=strategy
                    )
                )

            else:
                generated.append(
                    self.next_duplicate_id(
                        strategy=strategy
                    )
                )

        return generated

    def register_existing_id(
        self,
        row_id: object,
    ) -> None:
        """Register another externally created row identifier."""

        normalized = str(
            row_id
        )

        if normalized in self._generated_ids:
            raise ValueError(
                "The external row ID collides with a "
                "previously generated ID."
            )

        self._existing_ids.add(
            normalized
        )

    def is_available(
        self,
        row_id: object,
    ) -> bool:
        """Return whether a row identifier is unused."""

        normalized = str(
            row_id
        )

        return bool(
            normalized
            not in self._existing_ids
            and normalized
            not in self._generated_ids
        )

    def get_state(
        self,
    ) -> SyntheticIdGeneratorState:
        """Return a copy of the ID-generator state."""

        return self.state.model_copy(
            deep=True
        )

    def generated_ids(
        self,
    ) -> list[str]:
        """Return generated IDs in generation order."""

        return list(
            self.state.generated_ids
        )

    def _next_id(
        self,
        *,
        kind: GeneratedRowIdKind,
        strategy: str,
        metadata: dict[str, Any] | None,
    ) -> str:
        """Generate one collision-safe identifier."""

        normalized_strategy = (
            self._normalize_strategy(
                strategy
            )
        )

        if kind == GeneratedRowIdKind.SYNTHETIC:
            self.state.synthetic_counter += 1

            sequence = (
                self.state.synthetic_counter
            )

            prefix = (
                self.options.synthetic_prefix
            )

        else:
            self.state.duplicate_counter += 1

            sequence = (
                self.state.duplicate_counter
            )

            prefix = (
                self.options.duplicate_prefix
            )

        collision_attempts = 0

        while True:
            candidate = self._format_id(
                prefix=prefix,
                strategy=strategy,
                sequence=sequence,
                collision_attempt=(
                    collision_attempts
                ),
            )

            if self.is_available(
                candidate
            ):
                break

            collision_attempts += 1

            self.state.collision_count += 1

            if (
                collision_attempts
                > self.options
                .maximum_collision_attempts
            ):
                raise RuntimeError(
                    "Unable to generate a collision-free "
                    "Phase 8 row ID."
                )

        self._generated_ids.add(
            candidate
        )

        self.state.generated_ids.append(
            candidate
        )

        self.state.generated_records.append(
            GeneratedRowIdRecord(
                row_id=candidate,
                kind=kind,
                strategy=normalized_strategy,
                sequence=sequence,
                execution_id=self.execution_id,
                deterministic=self.deterministic,
                collision_attempts=(
                    collision_attempts
                ),
                metadata=dict(
                    metadata or {}
                ),
            )
        )

        return candidate

    def _format_id(
        self,
        *,
        prefix: str,
        strategy: str,
        sequence: int,
        collision_attempt: int,
    ) -> str:
        """Format one generated identifier."""

        parts = [
            prefix
        ]

        if self.options.include_strategy:
            parts.append(
                strategy
            )

        if (
            self.options
            .include_execution_token
        ):
            parts.append(
                self._execution_token
            )

        parts.append(
            f"{sequence:0{self.options.sequence_width}d}"
        )

        if collision_attempt > 0:
            parts.append(
                f"retry{collision_attempt}"
            )

        return "__".join(
            parts
        )

    def _build_execution_token(
        self,
    ) -> str:
        """Build a deterministic execution token."""

        digest = hashlib.sha256()

        digest.update(
            (
                f"{self.execution_id}|"
                f"{self.random_seed}"
            ).encode()
        )

        return digest.hexdigest()[
            : self.options
            .execution_token_length
        ]

    def _normalize_strategy(
        self,
        strategy: str,
    ) -> str:
        """Normalize a strategy name for use in an ID."""

        value = str(
            strategy
        ).strip()

        if not value:
            raise ValueError(
                "strategy cannot be empty."
            )

        if not self.options.normalize_strategy_name:
            return value

        normalized = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            value,
        ).strip(
            "_"
        ).casefold()

        if not normalized:
            raise ValueError(
                "strategy does not contain any usable "
                "identifier characters."
            )

        return normalized
