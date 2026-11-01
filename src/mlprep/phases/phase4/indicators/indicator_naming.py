"""Missing-indicator naming and collision handling for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IndicatorCollisionPolicy(StrEnum):
    """Policies for handling indicator-name collisions."""

    RAISE = "raise"
    SUFFIX = "suffix"
    REUSE = "reuse"


@dataclass(slots=True)
class IndicatorNamingOptions:
    """Options controlling missing-indicator names."""

    suffix: str = "__was_missing"

    collision_policy: IndicatorCollisionPolicy = (
        IndicatorCollisionPolicy.RAISE
    )

    collision_suffix_template: str = "__{number}"

    maximum_collision_attempts: int = 1_000

    maximum_name_length: int | None = None

    def __post_init__(self) -> None:
        """Validate indicator-naming options."""

        if not self.suffix:
            raise ValueError(
                "suffix cannot be empty."
            )

        if (
            self.maximum_collision_attempts < 1
        ):
            raise ValueError(
                "maximum_collision_attempts must be positive."
            )

        if (
            self.maximum_name_length is not None
            and self.maximum_name_length < 1
        ):
            raise ValueError(
                "maximum_name_length must be positive "
                "when provided."
            )

        if (
            self.collision_policy
            == IndicatorCollisionPolicy.SUFFIX
            and "{number}"
            not in self.collision_suffix_template
        ):
            raise ValueError(
                "collision_suffix_template must contain "
                "'{number}' when the suffix collision "
                "policy is used."
            )


class IndicatorNameResolver:
    """Build deterministic missing-indicator column names."""

    component_name = "indicator_naming"

    def __init__(
        self,
        options: IndicatorNamingOptions | None = None,
    ) -> None:
        """Initialize the indicator name resolver."""

        self.options = (
            options or IndicatorNamingOptions()
        )

    def base_name(
        self,
        column: str,
    ) -> str:
        """Build the default indicator name."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        candidate = (
            f"{column}{self.options.suffix}"
        )

        return self._truncate(
            candidate
        )

    def resolve(
        self,
        *,
        column: str,
        existing_columns: list[str] | set[str],
        reserved_columns: list[str] | set[str] | None = None,
    ) -> str:
        """Resolve one collision-safe indicator name."""

        existing = {
            str(value)
            for value in existing_columns
        }

        reserved = {
            str(value)
            for value in (
                reserved_columns or []
            )
        }

        unavailable = (
            existing | reserved
        )

        candidate = self.base_name(
            column
        )

        if candidate not in unavailable:
            return candidate

        if (
            self.options.collision_policy
            == IndicatorCollisionPolicy.REUSE
        ):
            return candidate

        if (
            self.options.collision_policy
            == IndicatorCollisionPolicy.RAISE
        ):
            raise ValueError(
                "Missing-indicator name collision for "
                f"feature {column!r}: {candidate!r}"
            )

        for collision_number in range(
            1,
            self.options
            .maximum_collision_attempts
            + 1,
        ):
            collision_suffix = (
                self.options
                .collision_suffix_template
                .format(
                    number=collision_number
                )
            )

            maximum_name_length = (
                self.options.maximum_name_length
            )

            if maximum_name_length is not None:
                collision_suffix = collision_suffix[
                    -maximum_name_length:
                ]

                candidate_prefix = candidate[
                    : maximum_name_length
                    - len(collision_suffix)
                ]

            else:
                candidate_prefix = candidate

            alternative = (
                f"{candidate_prefix}"
                f"{collision_suffix}"
            )

            if alternative not in unavailable:
                return alternative

        raise RuntimeError(
            "Unable to resolve a unique indicator name "
            f"for feature {column!r}."
        )

    def resolve_many(
        self,
        *,
        columns: list[str],
        existing_columns: list[str] | set[str],
    ) -> dict[str, str]:
        """Resolve indicator names for multiple source columns."""

        resolved: dict[str, str] = {}

        reserved: set[str] = set()

        for column in columns:
            indicator_name = self.resolve(
                column=column,
                existing_columns=existing_columns,
                reserved_columns=reserved,
            )

            resolved[column] = indicator_name
            reserved.add(indicator_name)

        return resolved

    def _truncate(
        self,
        value: str,
    ) -> str:
        """Apply the configured maximum name length."""

        maximum_length = (
            self.options.maximum_name_length
        )

        if maximum_length is None:
            return value

        if len(value) <= maximum_length:
            return value

        return value[
            :maximum_length
        ]
