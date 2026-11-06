from mlprep.exceptions import (
    ArtifactError,
    ColumnNameCollisionError,
    ColumnRoleError,
    ConfigurationError,
    ConstraintViolationError,
    LeakageRiskError,
    MLPrepError,
    Phase1Error,
    PhaseExecutionError,
    ProblemContractError,
    ReadinessError,
    SchemaValidationError,
    TargetValidationError,
)


def test_exception_hierarchy_preserves_phase_and_common_bases() -> None:
    phase_one_errors = (
        ProblemContractError,
        SchemaValidationError,
        ColumnNameCollisionError,
        ColumnRoleError,
        TargetValidationError,
        ConstraintViolationError,
        LeakageRiskError,
        ReadinessError,
    )

    assert issubclass(ConfigurationError, MLPrepError)
    assert issubclass(ArtifactError, MLPrepError)
    assert issubclass(PhaseExecutionError, MLPrepError)
    assert issubclass(Phase1Error, PhaseExecutionError)
    assert all(issubclass(error, Phase1Error) for error in phase_one_errors)
