import pytest

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase8.orchestration.fit_context import (
    BalancingFitContext,
    BalancingOperation,
)


def _context(**overrides: object) -> BalancingFitContext:
    values: dict[str, object] = {
        "partition": PartitionName.TRAIN,
        "operation": BalancingOperation.ANALYZE,
        "training_data": True,
        "analysis_allowed": True,
        "weighting_allowed": True,
        "resampling_allowed": True,
        "target_access_allowed": True,
        "synthetic_generation_allowed": True,
    }
    values.update(overrides)
    return BalancingFitContext(**values)


def test_balancing_context_factories_accept_authorized_operations() -> None:
    BalancingFitContext.training_analysis(
        target_access_authorized=True
    ).assert_analysis_allowed()
    BalancingFitContext.training_weighting(
        target_access_authorized=True
    ).assert_weighting_allowed()
    BalancingFitContext.training_resampling(
        resampling_authorized=True,
        target_access_authorized=True,
        synthetic_generation_authorized=True,
    ).assert_resampling_allowed(synthetic_required=True)
    BalancingFitContext.validation_guard().assert_protection_context()
    BalancingFitContext.test_guard().assert_protection_context()


@pytest.mark.parametrize(
    ("method", "overrides", "message"),
    [
        (
            "assert_analysis_allowed",
            {"operation": BalancingOperation.RESAMPLE},
            "permit analysis",
        ),
        ("assert_analysis_allowed", {"training_data": False}, "training data"),
        (
            "assert_analysis_allowed",
            {"partition": PartitionName.TEST},
            "training partition",
        ),
        ("assert_analysis_allowed", {"analysis_allowed": False}, "not allowed"),
        (
            "assert_analysis_allowed",
            {"target_access_allowed": False},
            "target access",
        ),
        (
            "assert_weighting_allowed",
            {"operation": BalancingOperation.ANALYZE},
            "weight calculation",
        ),
        (
            "assert_weighting_allowed",
            {
                "operation": BalancingOperation.CALCULATE_WEIGHTS,
                "training_data": False,
            },
            "training data",
        ),
        (
            "assert_weighting_allowed",
            {
                "operation": BalancingOperation.CALCULATE_WEIGHTS,
                "partition": PartitionName.VALIDATION,
            },
            "training partition",
        ),
        (
            "assert_weighting_allowed",
            {
                "operation": BalancingOperation.CALCULATE_WEIGHTS,
                "weighting_allowed": False,
            },
            "not allowed",
        ),
        (
            "assert_weighting_allowed",
            {
                "operation": BalancingOperation.CALCULATE_WEIGHTS,
                "target_access_allowed": False,
            },
            "target access",
        ),
        (
            "assert_resampling_allowed",
            {"operation": BalancingOperation.ANALYZE},
            "resampling",
        ),
        (
            "assert_resampling_allowed",
            {
                "operation": BalancingOperation.RESAMPLE,
                "training_data": False,
            },
            "training data",
        ),
        (
            "assert_resampling_allowed",
            {
                "operation": BalancingOperation.RESAMPLE,
                "partition": PartitionName.TEST,
            },
            "training partition",
        ),
        (
            "assert_resampling_allowed",
            {
                "operation": BalancingOperation.RESAMPLE,
                "resampling_allowed": False,
            },
            "not authorized",
        ),
        (
            "assert_resampling_allowed",
            {
                "operation": BalancingOperation.RESAMPLE,
                "target_access_allowed": False,
            },
            "target access",
        ),
    ],
)
def test_balancing_training_guards_reject_unsafe_contexts(
    method: str,
    overrides: dict[str, object],
    message: str,
) -> None:
    context = _context(**overrides)
    with pytest.raises(PermissionError, match=message):
        if method == "assert_resampling_allowed":
            context.assert_resampling_allowed(synthetic_required=False)
        elif method == "assert_weighting_allowed":
            context.assert_weighting_allowed()
        else:
            getattr(context, method)()


def test_balancing_resampling_requires_synthetic_authorization() -> None:
    with pytest.raises(PermissionError, match="Synthetic-row"):
        _context(
            operation=BalancingOperation.RESAMPLE,
            synthetic_generation_allowed=False,
        ).assert_resampling_allowed(synthetic_required=True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"operation": BalancingOperation.ANALYZE}, "not a partition guard"),
        ({"protected_partition": False}, "not marked as protected"),
        ({"training_data": True}, "training data"),
        ({"partition": PartitionName.TRAIN}, "validation and test"),
        ({"target_access_allowed": True}, "cannot permit"),
    ],
)
def test_balancing_protection_rejects_unsafe_contexts(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "partition": PartitionName.VALIDATION,
        "operation": BalancingOperation.PROTECT,
        "training_data": False,
        "protected_partition": True,
    }
    values.update(overrides)
    with pytest.raises(PermissionError, match=message):
        BalancingFitContext(**values).assert_protection_context()
