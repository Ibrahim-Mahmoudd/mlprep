"""Custom exception hierarchy for MLPrep."""


class MLPrepError(Exception):
    """Base exception for all MLPrep errors."""


class ConfigurationError(MLPrepError):
    """Raised when an MLPrep configuration is invalid."""


class PhaseExecutionError(MLPrepError):
    """Raised when a pipeline phase fails during execution."""


class Phase1Error(PhaseExecutionError):
    """Base exception for Phase 1 execution failures."""


class ProblemContractError(Phase1Error):
    """Raised when the prediction problem contract is invalid."""


class SchemaValidationError(Phase1Error):
    """Raised when dataset schema validation fails."""


class ColumnNameCollisionError(Phase1Error):
    """Raised when normalized column names collide."""


class ColumnRoleError(Phase1Error):
    """Raised when column roles are invalid or incompatible."""


class TargetValidationError(Phase1Error):
    """Raised when the supervised target is invalid."""


class ConstraintViolationError(Phase1Error):
    """Raised when a blocking data constraint is violated."""


class LeakageRiskError(Phase1Error):
    """Raised when a blocking leakage risk is detected."""


class ReadinessError(Phase1Error):
    """Raised when a dataset is not ready for the next phase."""


class ArtifactError(MLPrepError):
    """Raised when artifact creation or loading fails."""


class DataSourceError(MLPrepError):
    """Base exception for data-source failures."""


class DataSourceLoadError(DataSourceError):
    """Raised when a source cannot be loaded safely."""


class DataSourceSchemaError(DataSourceError):
    """Raised when loaded data violates the expected schema."""
