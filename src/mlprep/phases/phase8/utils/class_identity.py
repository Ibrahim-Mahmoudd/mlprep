"""Stable target-class identity utilities for Phase 8."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

CLASS_KEY_VERSION = "v1"
CLASS_KEY_SEPARATOR = "::"


class ClassIdentity(BaseModel):
    """Stable identity information for one target class."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        frozen=True,
    )

    key: str

    type_name: str

    display_value: str

    value: Any

    serialized_value: str

    version: str = CLASS_KEY_VERSION

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ClassIdentityMapping(BaseModel):
    """Bidirectional identity mapping for target classes."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    identities: dict[
        str,
        ClassIdentity,
    ] = Field(
        default_factory=dict,
    )

    class_order: list[str] = Field(
        default_factory=list,
    )

    version: str = CLASS_KEY_VERSION

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @property
    def class_key_to_value(
        self,
    ) -> dict[str, Any]:
        """Return the class-key-to-original-value mapping."""

        return {
            class_key: identity.value
            for class_key, identity
            in self.identities.items()
        }

    @property
    def class_key_to_display_value(
        self,
    ) -> dict[str, str]:
        """Return display values for every class key."""

        return {
            class_key: identity.display_value
            for class_key, identity
            in self.identities.items()
        }

    def key_for(
        self,
        value: object,
    ) -> str:
        """Return the registered key for a class value."""

        candidate_key = class_key(
            value
        )

        identity = self.identities.get(
            candidate_key
        )

        if identity is None:
            raise KeyError(
                f"Class value {value!r} is not registered."
            )

        if not class_values_equal(
            identity.value,
            normalize_class_value(
                value
            ),
        ):
            raise ValueError(
                "The calculated class key matched a different "
                "registered class value."
            )

        return candidate_key

    def value_for(
        self,
        key: str,
    ) -> Any:
        """Return the original class value for a key."""

        identity = self.identities.get(
            str(
                key
            )
        )

        if identity is None:
            raise KeyError(
                f"Unknown Phase 8 class key: {key!r}"
            )

        return identity.value

    def identity_for_key(
        self,
        key: str,
    ) -> ClassIdentity:
        """Return identity metadata for one class key."""

        identity = self.identities.get(
            str(
                key
            )
        )

        if identity is None:
            raise KeyError(
                f"Unknown Phase 8 class key: {key!r}"
            )

        return identity.model_copy(
            deep=True
        )

    def identity_for_value(
        self,
        value: object,
    ) -> ClassIdentity:
        """Return identity metadata for one class value."""

        return self.identity_for_key(
            self.key_for(
                value
            )
        )

    def resolve_configured_key(
        self,
        label: str,
    ) -> str:
        """Resolve a canonical key or an unambiguous legacy class label."""

        if label in self.identities:
            return label

        matches = [
            identity.key
            for identity in self.identities.values()
            if (
                identity.display_value == label
                or (
                    f"{identity.type_name}::"
                    f"{identity.display_value}"
                )
                == label
            )
        ]

        if not matches:
            raise KeyError(
                f"Unknown configured class label: {label!r}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Configured class label {label!r} is ambiguous; "
                "use a canonical typed class key."
            )

        return matches[0]

    def contains_key(
        self,
        key: str,
    ) -> bool:
        """Return whether one key is registered."""

        return str(
            key
        ) in self.identities

    def contains_value(
        self,
        value: object,
    ) -> bool:
        """Return whether one value is registered."""

        try:
            self.key_for(
                value
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return False

        return True

    def resolve_count_mapping(
        self,
        counts: Mapping[str, int],
        *,
        require_all_classes: bool = False,
        reject_unknown_classes: bool = True,
    ) -> dict[Any, int]:
        """Resolve class-key counts to original class values."""

        normalized_counts = {
            str(
                key
            ): int(
                count
            )
            for key, count
            in counts.items()
        }

        invalid_counts = {
            key: count
            for key, count
            in normalized_counts.items()
            if count < 1
        }

        if invalid_counts:
            raise ValueError(
                "Class target counts must be positive: "
                f"{invalid_counts}"
            )

        unknown_keys = sorted(
            set(
                normalized_counts
            ).difference(
                self.identities
            )
        )

        if (
            reject_unknown_classes
            and unknown_keys
        ):
            raise KeyError(
                "Class target counts contain unknown keys: "
                f"{unknown_keys}"
            )

        if require_all_classes:
            missing_keys = [
                class_key_value
                for class_key_value
                in self.class_order
                if class_key_value
                not in normalized_counts
            ]

            if missing_keys:
                raise KeyError(
                    "Class target counts are missing keys: "
                    f"{missing_keys}"
                )

        resolved: dict[
            Any,
            int,
        ] = {}

        for key, count in normalized_counts.items():
            if key not in self.identities:
                continue

            value = self.identities[
                key
            ].value

            try:
                if any(
                    type(existing_value) is not type(value)
                    and existing_value == value
                    for existing_value in resolved
                ):
                    raise ValueError(
                        "Original class labels that compare equal "
                        "cannot be represented as distinct mapping "
                        "keys; use canonical class keys instead."
                    )

                resolved[
                    value
                ] = count

            except TypeError as error:
                raise TypeError(
                    "Target class values must be hashable "
                    "before they can be passed to a sampler."
                ) from error

        return resolved

    def resolve_weight_mapping(
        self,
        weights: Mapping[str, float],
        *,
        require_all_classes: bool = False,
        reject_unknown_classes: bool = True,
    ) -> dict[Any, float]:
        """Resolve class-key weights to original class values."""

        normalized_weights = {
            str(
                key
            ): float(
                weight
            )
            for key, weight
            in weights.items()
        }

        invalid_weights = {
            key: weight
            for key, weight
            in normalized_weights.items()
            if (
                not math.isfinite(
                    weight
                )
                or weight <= 0.0
            )
        }

        if invalid_weights:
            raise ValueError(
                "Class weights must be finite and positive: "
                f"{invalid_weights}"
            )

        unknown_keys = sorted(
            set(
                normalized_weights
            ).difference(
                self.identities
            )
        )

        if (
            reject_unknown_classes
            and unknown_keys
        ):
            raise KeyError(
                "Class weights contain unknown keys: "
                f"{unknown_keys}"
            )

        if require_all_classes:
            missing_keys = [
                class_key_value
                for class_key_value
                in self.class_order
                if class_key_value
                not in normalized_weights
            ]

            if missing_keys:
                raise KeyError(
                    "Class weights are missing keys: "
                    f"{missing_keys}"
                )

        resolved: dict[
            Any,
            float,
        ] = {}

        for key, weight in normalized_weights.items():
            if key not in self.identities:
                continue

            value = self.identities[
                key
            ].value

            try:
                if any(
                    type(existing_value) is not type(value)
                    and existing_value == value
                    for existing_value in resolved
                ):
                    raise ValueError(
                        "Original class labels that compare equal "
                        "cannot be represented as distinct mapping "
                        "keys; use canonical class keys instead."
                    )

                resolved[
                    value
                ] = weight

            except TypeError as error:
                raise TypeError(
                    "Target class values must be hashable "
                    "before they can be used as weight keys."
                ) from error

        return resolved


def normalize_class_value(
    value: object,
) -> Any:
    """Convert scalar library values to stable Python values."""

    if isinstance(
        value,
        np.generic,
    ):
        value = value.item()

    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, Decimal) and value.is_nan():
        return None

    if isinstance(
        value,
        bytearray,
    ):
        return bytes(
            value
        )

    return value


def class_type_name(
    value: object,
) -> str:
    """Return a stable type identifier for a class value."""

    normalized = normalize_class_value(
        value
    )

    if normalized is None:
        return "none"

    if isinstance(
        normalized,
        bool,
    ):
        return "bool"

    if isinstance(
        normalized,
        int,
    ):
        return "int"

    if isinstance(
        normalized,
        float,
    ):
        return "float"

    if isinstance(
        normalized,
        Decimal,
    ):
        return "decimal"

    if isinstance(
        normalized,
        str,
    ):
        return "str"

    if isinstance(
        normalized,
        bytes,
    ):
        return "bytes"

    if isinstance(
        normalized,
        datetime,
    ):
        return "datetime"

    if isinstance(
        normalized,
        date,
    ):
        return "date"

    if isinstance(
        normalized,
        Enum,
    ):
        enum_type = type(
            normalized
        )

        return (
            f"enum:"
            f"{enum_type.__module__}."
            f"{enum_type.__qualname__}"
        )

    value_type = type(
        normalized
    )

    return (
        f"object:"
        f"{value_type.__module__}."
        f"{value_type.__qualname__}"
    )


def serialize_class_value(
    value: object,
) -> str:
    """Serialize one class value deterministically."""

    normalized = normalize_class_value(
        value
    )

    payload: dict[str, Any]

    if normalized is None:
        payload = {
            "kind": "none",
            "value": None,
        }

    elif isinstance(
        normalized,
        bool,
    ):
        payload = {
            "kind": "bool",
            "value": normalized,
        }

    elif isinstance(
        normalized,
        int,
    ):
        payload = {
            "kind": "int",
            "value": str(
                normalized
            ),
        }

    elif isinstance(
        normalized,
        float,
    ):
        if not math.isfinite(
            normalized
        ):
            raise ValueError(
                "Target class labels cannot be NaN or infinite."
            )

        payload = {
            "kind": "float",
            "value": normalized.hex(),
        }

    elif isinstance(
        normalized,
        Decimal,
    ):
        if not normalized.is_finite():
            raise ValueError(
                "Decimal target labels must be finite."
            )

        payload = {
            "kind": "decimal",
            "value": str(
                normalized
            ),
        }

    elif isinstance(
        normalized,
        str,
    ):
        payload = {
            "kind": "str",
            "value": normalized,
        }

    elif isinstance(
        normalized,
        bytes,
    ):
        payload = {
            "kind": "bytes",
            "value": base64.b64encode(
                normalized
            ).decode(
                "ascii"
            ),
        }

    elif isinstance(
        normalized,
        datetime,
    ):
        payload = {
            "kind": "datetime",
            "value": normalized.isoformat(),
        }

    elif isinstance(
        normalized,
        date,
    ):
        payload = {
            "kind": "date",
            "value": normalized.isoformat(),
        }

    elif isinstance(
        normalized,
        Enum,
    ):
        enum_type = type(
            normalized
        )

        payload = {
            "kind": "enum",
            "enum_type": (
                f"{enum_type.__module__}."
                f"{enum_type.__qualname__}"
            ),
            "name": normalized.name,
            "value": serialize_class_value(
                normalized.value
            ),
        }

    else:
        raise TypeError(
            "Unsupported target-class value type: "
            f"{type(normalized).__module__}."
            f"{type(normalized).__qualname__}"
        )

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def class_key(
    value: object,
) -> str:
    """Build a stable typed key for one class value."""

    normalized = normalize_class_value(
        value
    )

    type_name = class_type_name(
        normalized
    )

    serialized_value = serialize_class_value(
        normalized
    )

    encoded_value = base64.urlsafe_b64encode(
        serialized_value.encode(
            "utf-8"
        )
    ).decode(
        "ascii"
    ).rstrip(
        "="
    )

    return CLASS_KEY_SEPARATOR.join(
        [
            CLASS_KEY_VERSION,
            type_name,
            encoded_value,
        ]
    )


def class_display_value(
    value: object,
) -> str:
    """Return a readable representation of a class value."""

    normalized = normalize_class_value(
        value
    )

    if normalized is None:
        return "<missing>"

    if isinstance(
        normalized,
        bytes,
    ):
        return base64.b64encode(
            normalized
        ).decode(
            "ascii"
        )

    if isinstance(
        normalized,
        Enum,
    ):
        enum_type = type(
            normalized
        )

        return (
            f"{enum_type.__qualname__}."
            f"{normalized.name}"
        )

    if isinstance(
        normalized,
        (
            datetime,
            date,
        ),
    ):
        return normalized.isoformat()

    return str(
        normalized
    )


def build_class_identity(
    value: object,
) -> ClassIdentity:
    """Build one complete class identity."""

    normalized = normalize_class_value(
        value
    )

    return ClassIdentity(
        key=class_key(
            normalized
        ),
        type_name=class_type_name(
            normalized
        ),
        display_value=class_display_value(
            normalized
        ),
        value=normalized,
        serialized_value=serialize_class_value(
            normalized
        ),
    )


def build_class_identity_mapping(
    values: Iterable[object] | pd.Series,
    *,
    reject_missing: bool = True,
    require_at_least_two_classes: bool = False,
) -> ClassIdentityMapping:
    """Build an ordered mapping for target class values."""

    normalized_values: list[
        Any
    ] = []

    for value in values:
        normalized = normalize_class_value(
            value
        )

        if normalized is None:
            if reject_missing:
                raise ValueError(
                    "Target class values cannot contain "
                    "missing values."
                )

            continue

        normalized_values.append(
            normalized
        )

    identities: dict[
        str,
        ClassIdentity,
    ] = {}

    class_order: list[str] = []

    for value in normalized_values:
        identity = build_class_identity(
            value
        )

        existing = identities.get(
            identity.key
        )

        if existing is not None:
            if not class_values_equal(
                existing.value,
                identity.value,
            ):
                raise ValueError(
                    "A class-key collision was detected."
                )

            continue

        identities[
            identity.key
        ] = identity

        class_order.append(
            identity.key
        )

    if (
        require_at_least_two_classes
        and len(
            class_order
        )
        < 2
    ):
        raise ValueError(
            "At least two distinct target classes "
            "are required."
        )

    return ClassIdentityMapping(
        identities=identities,
        class_order=class_order,
        metadata={
            "class_count": len(
                class_order
            ),
            "reject_missing": (
                reject_missing
            ),
        },
    )


def resolve_class_count_mapping(
    counts: Mapping[str, int],
    *,
    identity_mapping: ClassIdentityMapping,
    require_all_classes: bool = False,
    reject_unknown_classes: bool = True,
) -> dict[Any, int]:
    """Resolve keyed target counts to original values."""

    return identity_mapping.resolve_count_mapping(
        counts,
        require_all_classes=(
            require_all_classes
        ),
        reject_unknown_classes=(
            reject_unknown_classes
        ),
    )


def resolve_class_weight_mapping(
    weights: Mapping[str, float],
    *,
    identity_mapping: ClassIdentityMapping,
    require_all_classes: bool = False,
    reject_unknown_classes: bool = True,
) -> dict[Any, float]:
    """Resolve keyed class weights to original values."""

    return identity_mapping.resolve_weight_mapping(
        weights,
        require_all_classes=(
            require_all_classes
        ),
        reject_unknown_classes=(
            reject_unknown_classes
        ),
    )


def class_values_equal(
    left: object,
    right: object,
) -> bool:
    """Compare target-class values without type coercion."""

    normalized_left = normalize_class_value(
        left
    )

    normalized_right = normalize_class_value(
        right
    )

    if (
        normalized_left is None
        or normalized_right is None
    ):
        return bool(
            normalized_left is None
            and normalized_right is None
        )

    if type(
        normalized_left
    ) is not type(
        normalized_right
    ):
        return False

    if isinstance(
        normalized_left,
        float,
    ):
        if (
            not math.isfinite(
                normalized_left
            )
            or not math.isfinite(
                normalized_right
            )
        ):
            return False

        return bool(
            normalized_left
            == normalized_right
        )

    try:
        comparison = (
            normalized_left
            == normalized_right
        )

    except Exception:
        return False

    if isinstance(
        comparison,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(
            comparison
        )

    return False


def validate_class_identity_mapping(
    mapping: ClassIdentityMapping,
) -> None:
    """Validate identity-mapping completeness and consistency."""

    if not isinstance(
        mapping,
        ClassIdentityMapping,
    ):
        raise TypeError(
            "mapping must be a ClassIdentityMapping."
        )

    if len(
        mapping.class_order
    ) != len(
        set(
            mapping.class_order
        )
    ):
        raise ValueError(
            "class_order contains duplicate keys."
        )

    missing_identity_keys = [
        key
        for key in mapping.class_order
        if key not in mapping.identities
    ]

    if missing_identity_keys:
        raise ValueError(
            "class_order contains unknown identity keys: "
            f"{missing_identity_keys}"
        )

    extra_identity_keys = sorted(
        set(
            mapping.identities
        ).difference(
            mapping.class_order
        )
    )

    if extra_identity_keys:
        raise ValueError(
            "Identity entries are absent from class_order: "
            f"{extra_identity_keys}"
        )

    for key, identity in mapping.identities.items():
        expected_key = class_key(
            identity.value
        )

        if key != expected_key:
            raise ValueError(
                "A stored class identity has an invalid key: "
                f"{key!r}"
            )

        if identity.key != key:
            raise ValueError(
                "The dictionary key and identity key differ."
            )
