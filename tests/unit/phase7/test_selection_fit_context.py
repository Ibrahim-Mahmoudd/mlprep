import pytest

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase7.orchestration.fit_context import (
    SelectionFitContext,
    SelectionOperation,
)


def _context(**overrides: object) -> SelectionFitContext:
    values: dict[str, object] = {
        "partition": PartitionName.TRAIN,
        "operation": SelectionOperation.FIT_AND_SELECT,
        "training_data": True,
        "unsupervised_fitting_allowed": True,
        "supervised_fitting_allowed": True,
        "target_access_allowed": True,
    }
    values.update(overrides)
    return SelectionFitContext(**values)


def test_selection_context_factories_cover_authorization_modes() -> None:
    SelectionFitContext.training_fit(
        supervised_selection_authorized=True
    ).validate_for_fit(supervised_selection_required=True)
    SelectionFitContext.training_fit(
        supervised_selection_authorized=False
    ).validate_for_fit(supervised_selection_required=False)
    SelectionFitContext.training_transform().validate_for_transform()
    SelectionFitContext.validation_transform().validate_for_transform()
    SelectionFitContext.test_transform(authorized=True).validate_for_transform()


@pytest.mark.parametrize(
    ("overrides", "method", "message"),
    [
        (
            {"unsupervised_fitting_allowed": False},
            "assert_unsupervised_fitting_allowed",
            "not allowed",
        ),
        (
            {"training_data": False},
            "assert_unsupervised_fitting_allowed",
            "training data",
        ),
        (
            {"partition": PartitionName.VALIDATION},
            "assert_unsupervised_fitting_allowed",
            "training partition",
        ),
        (
            {"operation": SelectionOperation.TRANSFORM},
            "assert_unsupervised_fitting_allowed",
            "operation",
        ),
        (
            {"supervised_fitting_allowed": False},
            "assert_supervised_fitting_allowed",
            "not authorized",
        ),
        (
            {"target_access_allowed": False},
            "assert_supervised_fitting_allowed",
            "target",
        ),
        (
            {"training_data": False},
            "assert_supervised_fitting_allowed",
            "training data",
        ),
        (
            {"partition": PartitionName.VALIDATION},
            "assert_supervised_fitting_allowed",
            "training partition",
        ),
        (
            {"operation": SelectionOperation.TRANSFORM},
            "assert_supervised_fitting_allowed",
            "operation",
        ),
    ],
)
def test_selection_fit_rejects_unsafe_contexts(
    overrides: dict[str, object],
    method: str,
    message: str,
) -> None:
    with pytest.raises(PermissionError, match=message):
        getattr(_context(**overrides), method)()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"operation": SelectionOperation.FIT_AND_SELECT},
            "transform-only",
        ),
        (
            {
                "partition": PartitionName.TEST,
                "operation": SelectionOperation.TRANSFORM,
                "test_partition_authorized": False,
                "unsupervised_fitting_allowed": False,
                "supervised_fitting_allowed": False,
                "target_access_allowed": False,
            },
            "authorized access",
        ),
        (
            {
                "operation": SelectionOperation.TRANSFORM,
                "unsupervised_fitting_allowed": False,
                "supervised_fitting_allowed": False,
                "target_access_allowed": True,
            },
            "Target access",
        ),
    ],
)
def test_selection_transform_rejects_unsafe_contexts(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(PermissionError, match=message):
        _context(**overrides).assert_transform_allowed()


def test_selection_validation_rejects_unneeded_target_and_transform_fitting() -> None:
    with pytest.raises(PermissionError, match="unnecessary"):
        _context(target_access_allowed=True).validate_for_fit(
            supervised_selection_required=False
        )

    with pytest.raises(PermissionError, match="Unsupervised fitting"):
        _context(
            operation=SelectionOperation.TRANSFORM,
            unsupervised_fitting_allowed=True,
            supervised_fitting_allowed=False,
            target_access_allowed=False,
        ).validate_for_transform()

    with pytest.raises(PermissionError, match="Supervised fitting"):
        _context(
            operation=SelectionOperation.TRANSFORM,
            unsupervised_fitting_allowed=False,
            supervised_fitting_allowed=True,
            target_access_allowed=False,
        ).validate_for_transform()
