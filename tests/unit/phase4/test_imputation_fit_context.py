import pytest

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase4.orchestration.fit_context import (
    FitContext,
    ImputationOperation,
)
from mlprep.phases.phase4.strategies.base import ImputerExecutionMode


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (
            FitContext(
                partition=PartitionName.TRAIN,
                operation=ImputationOperation.FIT,
                execution_mode=ImputerExecutionMode.TRAINING_TRANSFORM,
                training_data=True,
                fitting_allowed=False,
                target_access_allowed=True,
            ),
            "Target-dependent",
        ),
        (
            FitContext(
                partition=PartitionName.VALIDATION,
                operation=ImputationOperation.FIT,
                execution_mode=ImputerExecutionMode.VALIDATION_TRANSFORM,
                training_data=False,
                fitting_allowed=True,
            ),
            "training data",
        ),
        (
            FitContext(
                partition=PartitionName.TEST,
                operation=ImputationOperation.TRANSFORM,
                execution_mode=ImputerExecutionMode.TEST_TRANSFORM,
                training_data=False,
                fitting_allowed=False,
            ),
            "authorized access",
        ),
    ],
)
def test_invalid_fit_contexts_are_rejected(
    context: FitContext,
    message: str,
) -> None:
    with pytest.raises(PermissionError, match=message):
        context.validate()


@pytest.mark.parametrize(
    "context",
    [
        FitContext.training_fit(),
        FitContext.training_transform(),
        FitContext.validation_transform(),
        FitContext.test_transform(authorized=True),
    ],
)
def test_valid_fit_contexts_are_accepted(context: FitContext) -> None:
    context.validate()
