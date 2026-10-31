"""Shared enumerations and type definitions for MLPrep."""

from enum import StrEnum


class TaskType(StrEnum):
    """Supported machine learning task types."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class Severity(StrEnum):
    """Severity level assigned to a Phase 1 finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PhaseStatus(StrEnum):
    """Final execution status of a pipeline phase."""

    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class ColumnRole(StrEnum):
    """Functional role assigned to a dataset column."""

    FEATURE = "feature"
    TARGET = "target"
    IDENTIFIER = "identifier"
    GROUP = "group"
    TIMESTAMP = "timestamp"
    PREDICTION_TIMESTAMP = "prediction_timestamp"
    OUTCOME_TIMESTAMP = "outcome_timestamp"
    SENSITIVE = "sensitive"
    METADATA = "metadata"
    SAMPLE_WEIGHT = "sample_weight"
    FOLD = "fold"
    IGNORED = "ignored"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class PhysicalType(StrEnum):
    """Physical storage type detected for a column."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TIMEDELTA = "timedelta"
    CATEGORY = "category"
    OBJECT = "object"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


class SemanticType(StrEnum):
    """Real-world meaning inferred or declared for a column."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    BINARY = "binary"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    DURATION = "duration"
    IDENTIFIER = "identifier"
    FREE_TEXT = "free_text"
    SHORT_TEXT = "short_text"
    EMAIL = "email"
    PHONE_NUMBER = "phone_number"
    IP_ADDRESS = "ip_address"
    URL = "url"
    POSTAL_CODE = "postal_code"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    GEOGRAPHIC = "geographic"
    STRUCTURED_CODE = "structured_code"
    UNKNOWN = "unknown"


class FindingCategory(StrEnum):
    """Category identifying the source of a Phase 1 finding."""

    INPUT = "input"
    PROBLEM_CONTRACT = "problem_contract"
    COLUMN_NAME = "column_name"
    COLUMN_ROLE = "column_role"
    PHYSICAL_TYPE = "physical_type"
    SEMANTIC_TYPE = "semantic_type"
    MIXED_TYPE = "mixed_type"
    TEMPORAL = "temporal"
    TEXT_NORMALIZATION = "text_normalization"
    MISSING_TOKEN = "missing_token"
    DUPLICATE_COLUMN = "duplicate_column"
    DUPLICATE_ROW = "duplicate_row"
    CARDINALITY = "cardinality"
    CONSTANCY = "constancy"
    TARGET = "target"
    CONSTRAINT = "constraint"
    SENSITIVE_DATA = "sensitive_data"
    LEAKAGE_RISK = "leakage_risk"
    MEMORY = "memory"
    SCHEMA = "schema"
    FINGERPRINT = "fingerprint"
    READINESS = "readiness"


class FindingAction(StrEnum):
    """Action taken or recommended for a detected finding."""

    NONE = "none"
    REPORT = "report"
    WARN = "warn"
    COERCE = "coerce"
    SET_MISSING = "set_missing"
    QUARANTINE = "quarantine"
    DROP_COLUMN = "drop_column"
    DROP_ROW = "drop_row"
    RENAME = "rename"
    RAISE = "raise"


class ErrorPolicy(StrEnum):
    """Behavior used when a component encounters an error."""

    RAISE = "raise"
    WARN = "warn"
    SKIP_COMPONENT = "skip_component"
    USE_FALLBACK = "use_fallback"


class CollisionPolicy(StrEnum):
    """Policy for column names that collide after normalization."""

    RAISE = "raise"
    ADD_SUFFIX = "add_suffix"
    USE_EXPLICIT_MAPPING = "use_explicit_mapping"


class DuplicateRowPolicy(StrEnum):
    """Policy for detected duplicate rows."""

    REPORT = "report"
    RAISE = "raise"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    REMOVE_EXACT = "remove_exact"
    QUARANTINE = "quarantine"


class InvalidValuePolicy(StrEnum):
    """Policy for values that violate a schema or constraint."""

    REPORT = "report"
    RAISE = "raise"
    SET_MISSING = "set_missing"
    QUARANTINE_ROW = "quarantine_row"


class TypeCoercionPolicy(StrEnum):
    """Policy controlling automatic physical type conversion."""

    DISABLED = "disabled"
    SAFE = "safe"
    STRICT = "strict"


class MemoryOptimizationLevel(StrEnum):
    """Safety level for DataFrame memory optimization."""

    DISABLED = "disabled"
    SAFE = "safe"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"