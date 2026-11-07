import pandas as pd
import pytest

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase2.output_governance.result import DatasetPartition
from mlprep.phases.phase8.orchestration.fit_context import (
    BalancingFitContext,
)
from mlprep.phases.phase8.orchestration.partition_guard import (
    PartitionGuard,
    PartitionGuardOptions,
)


def _partition(
    *,
    name: PartitionName = PartitionName.VALIDATION,
) -> DatasetPartition:
    index = pd.Index([10, 11, 12])
    return DatasetPartition(
        name=name,
        X=pd.DataFrame(
            {"feature_a": [1.0, 2.0, 3.0], "feature_b": [4, 5, 6]},
            index=index,
        ),
        y=pd.Series([0, 1, 0], index=index, name="target"),
        row_ids=pd.Series(["row-0", "row-1", "row-2"], index=index, name="row_id"),
    )


def test_capture_and_protected_copy_are_deep_and_fingerprint_partition() -> None:
    partition = _partition()
    guard = PartitionGuard()

    snapshot = guard.capture(
        partition=partition,
        context=BalancingFitContext.validation_guard(),
    )
    copied = guard.protected_copy(PartitionName.VALIDATION)
    copied.X.loc[10, "feature_a"] = 99.0

    assert snapshot.row_count == 3
    assert snapshot.feature_columns == ["feature_a", "feature_b"]
    assert guard.protected_copy(PartitionName.VALIDATION).X.loc[10, "feature_a"] == 1.0
    assert guard.verify(
        partition=partition,
        context=BalancingFitContext.validation_guard(),
    ).valid


def test_capture_rejects_invalid_context_partition_and_training_partition() -> None:
    guard = PartitionGuard()
    validation = _partition()

    with pytest.raises(PermissionError, match="not a partition guard"):
        guard.capture(
            partition=validation,
            context=BalancingFitContext.training_analysis(
                target_access_authorized=True
            ),
        )
    with pytest.raises(ValueError, match="does not match"):
        guard.capture(
            partition=validation,
            context=BalancingFitContext.test_guard(),
        )
    with pytest.raises(PermissionError, match="validation and test"):
        guard.capture(
            partition=_partition(name=PartitionName.TRAIN),
            context=BalancingFitContext(
                partition=PartitionName.TRAIN,
                operation="protect",
                training_data=False,
                protected_partition=True,
            ),
        )


def test_protected_copy_and_verify_require_capture_and_matching_context() -> None:
    guard = PartitionGuard()
    with pytest.raises(KeyError, match="has not been captured"):
        guard.protected_copy(PartitionName.TEST)
    with pytest.raises(RuntimeError, match="was not captured"):
        guard.verify(
            partition=_partition(),
            context=BalancingFitContext.validation_guard(),
        )
    guard.capture(
        partition=_partition(),
        context=BalancingFitContext.validation_guard(),
    )
    with pytest.raises(ValueError, match="does not match"):
        guard.verify(
            partition=_partition(),
            context=BalancingFitContext.test_guard(),
        )


@pytest.mark.parametrize(
    ("change", "violation", "attribute"),
    [
        ("rows", "row_count_changed", "row_count_unchanged"),
        ("feature_count", "feature_count_changed", "feature_count_unchanged"),
        ("schema", "feature_schema_changed", "feature_schema_unchanged"),
        ("dtype", "feature_dtypes_changed", "feature_dtypes_unchanged"),
        ("features", "feature_values_changed", "feature_values_unchanged"),
        ("target", "target_changed", "target_unchanged"),
        ("row_ids", "row_ids_changed", "row_ids_unchanged"),
        ("order", "row_order_changed", "row_order_unchanged"),
    ],
)
def test_verify_reports_each_partition_mutation(
    change: str,
    violation: str,
    attribute: str,
) -> None:
    guard = PartitionGuard()
    original = _partition()
    guard.capture(
        partition=original,
        context=BalancingFitContext.validation_guard(),
    )
    mutated = _partition()

    if change == "rows":
        mutated.X = mutated.X.iloc[:2]
        mutated.y = mutated.y.iloc[:2]
        mutated.row_ids = mutated.row_ids.iloc[:2]
    elif change == "feature_count":
        mutated.X["feature_c"] = [7.0, 8.0, 9.0]
    elif change == "schema":
        mutated.X = mutated.X.rename(columns={"feature_a": "renamed"})
    elif change == "dtype":
        mutated.X["feature_a"] = mutated.X["feature_a"].astype("string")
    elif change == "features":
        mutated.X.loc[10, "feature_a"] = 99.0
    elif change == "target":
        mutated.y.loc[10] = 1
    elif change == "row_ids":
        mutated.row_ids.loc[10] = "changed"
    elif change == "order":
        mutated.X = mutated.X.iloc[::-1]
        mutated.y = mutated.y.iloc[::-1]
        mutated.row_ids = mutated.row_ids.iloc[::-1]

    result = guard.verify(
        partition=mutated,
        context=BalancingFitContext.validation_guard(),
    )

    assert not result.valid
    assert not getattr(result, attribute)
    assert violation in result.violations


def test_verify_can_disable_checks_but_still_reports_observed_changes() -> None:
    options = PartitionGuardOptions(
        require_identical_feature_values=False,
        require_identical_target=False,
        require_identical_row_ids=False,
        require_identical_row_order=False,
    )
    guard = PartitionGuard(options)
    original = _partition()
    guard.capture(
        partition=original,
        context=BalancingFitContext.validation_guard(),
    )
    mutated = _partition()
    mutated.X.loc[10, "feature_a"] = 99.0
    mutated.y.loc[10] = 1
    mutated.row_ids.loc[10] = "changed"
    mutated.X = mutated.X.iloc[::-1]
    mutated.y = mutated.y.iloc[::-1]
    mutated.row_ids = mutated.row_ids.iloc[::-1]

    result = guard.verify(
        partition=mutated,
        context=BalancingFitContext.validation_guard(),
    )

    assert result.valid
    assert not result.feature_values_unchanged
    assert not result.target_unchanged
    assert not result.row_ids_unchanged
    assert not result.row_order_unchanged
    assert result.violations == []


def test_capture_can_disable_deep_copy() -> None:
    partition = _partition()
    guard = PartitionGuard(PartitionGuardOptions(copy_protected_partition=False))

    guard.capture(
        partition=partition,
        context=BalancingFitContext.validation_guard(),
    )
    partition.X.loc[10, "feature_a"] = 42.0

    result = guard.verify(
        partition=partition,
        context=BalancingFitContext.validation_guard(),
    )

    assert not result.valid
    assert not result.feature_values_unchanged
