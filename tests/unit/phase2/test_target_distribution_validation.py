from __future__ import annotations

import warnings

import pandas as pd
import pytest

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase2.validation.target_distribution import (
    TargetDistributionValidationOptions,
    TargetDistributionValidator,
)


def _partition(
    name: PartitionName,
    target: list[object],
) -> DatasetPartition:
    index = pd.RangeIndex(len(target))
    return DatasetPartition(
        name=name,
        X=pd.DataFrame(
            {"feature": range(len(target))},
            index=index,
        ),
        y=pd.Series(target, index=index, name="target"),
        row_ids=pd.Series(
            [f"{name.value}-{row}" for row in index],
            index=index,
            name="row_id",
        ),
    )


def _validate(
    train: list[object],
    validation: list[object],
    test: list[object],
    *,
    options: TargetDistributionValidationOptions | None = None,
) -> list[object]:
    return TargetDistributionValidator(options).validate(
        train=_partition(PartitionName.TRAIN, train),
        validation=_partition(PartitionName.VALIDATION, validation),
        test=_partition(PartitionName.TEST, test),
    )


@pytest.mark.parametrize(
    "values",
    [
        {"mean_relative_tolerance": -0.1},
        {"median_relative_tolerance": -0.1},
        {"standard_deviation_relative_tolerance": -0.1},
        {"quantile_relative_tolerance": -0.1},
        {"quantiles": (-0.1,)},
        {"quantiles": (1.1,)},
    ],
)
def test_target_distribution_options_reject_invalid_tolerances(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        TargetDistributionValidationOptions(**values)


def test_target_distribution_can_be_disabled_or_reject_non_numeric_targets() -> None:
    disabled_findings = _validate(
        [1.0, 2.0],
        [100.0],
        [200.0],
        options=TargetDistributionValidationOptions(enabled=False),
    )
    non_numeric_findings = _validate(
        ["low", "high"],
        ["low"],
        ["high"],
    )

    assert disabled_findings == []
    assert [finding.code for finding in non_numeric_findings] == [
        "P2-TARGET-DIST-001"
    ]
    assert non_numeric_findings[0].blocks_readiness


def test_target_distribution_reports_missing_and_configurable_nonfinite_values(
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        blocked = _validate(
            [1.0, 2.0],
            [float("nan"), float("inf")],
            [1.0, 2.0],
        )
        allowed = _validate(
            [1.0, 2.0],
            [float("inf"), 3.0],
            [1.0, 2.0],
            options=TargetDistributionValidationOptions(
                block_nonfinite_values=False,
            ),
        )

    assert [finding.code for finding in blocked] == [
        "P2-TARGET-DIST-002",
        "P2-TARGET-DIST-003",
    ]
    assert blocked[0].affected_row_count == 1
    assert blocked[1].affected_row_count == 1
    assert blocked[0].blocks_readiness and blocked[1].blocks_readiness
    assert all(finding.code != "P2-TARGET-DIST-003" for finding in allowed)


def test_target_distribution_reports_statistical_deviations_without_mutation() -> None:
    train = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    validation = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    original_train = train.copy()
    original_validation = validation.copy()
    findings = _validate(
        train,
        validation,
        train,
    )

    deviations = next(
        finding.details["deviations"]
        for finding in findings
        if finding.code == "P2-TARGET-DIST-004"
        and PartitionName.VALIDATION in finding.partitions
    )

    assert {"mean", "median", "quantile_0.05", "quantile_0.95"} <= set(
        deviations
    )
    assert all(
        finding.severity.value == "warning"
        and not finding.blocks_readiness
        for finding in findings
    )
    assert train == original_train
    assert validation == original_validation


def test_target_distribution_handles_empty_training_and_candidate_targets() -> None:
    empty_reference = _validate(
        [float("nan")],
        [10.0],
        [],
    )
    empty_candidates = _validate(
        [0.0, 0.0],
        [float("nan")],
        [0.0],
    )

    assert [finding.code for finding in empty_reference] == [
        "P2-TARGET-DIST-002"
    ]
    assert [finding.code for finding in empty_candidates] == [
        "P2-TARGET-DIST-002"
    ]
    assert (
        TargetDistributionValidator._relative_difference(0.0, 0.0)
        == 0.0
    )
    assert (
        TargetDistributionValidator._relative_difference(0.0, 1.0)
        == 1.0
    )
