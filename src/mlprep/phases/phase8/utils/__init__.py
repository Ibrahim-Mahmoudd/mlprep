"""Shared utility interfaces for Phase 8."""

from mlprep.phases.phase8.utils.class_identity import (
    CLASS_KEY_SEPARATOR,
    CLASS_KEY_VERSION,
    ClassIdentity,
    ClassIdentityMapping,
    build_class_identity,
    build_class_identity_mapping,
    class_display_value,
    class_key,
    class_type_name,
    class_values_equal,
    normalize_class_value,
    resolve_class_count_mapping,
    resolve_class_weight_mapping,
    serialize_class_value,
    validate_class_identity_mapping,
)

__all__ = [
    "CLASS_KEY_SEPARATOR",
    "CLASS_KEY_VERSION",
    "ClassIdentity",
    "ClassIdentityMapping",
    "build_class_identity",
    "build_class_identity_mapping",
    "class_display_value",
    "class_key",
    "class_type_name",
    "class_values_equal",
    "normalize_class_value",
    "resolve_class_count_mapping",
    "resolve_class_weight_mapping",
    "serialize_class_value",
    "validate_class_identity_mapping",
]