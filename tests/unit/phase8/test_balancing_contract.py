import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase2.output_governance.result import DatasetPartition
from mlprep.phases.phase6.feature_engineering.feature_lineage import (
    FeatureLineageState,
)
from mlprep.phases.phase7.output_governance.metadata import Phase7Status
from mlprep.phases.phase7.output_governance.result import (
    Phase7ReadinessDecision,
    Phase7Result,
)
from mlprep.phases.phase7.output_governance.state import Phase7State
from mlprep.phases.phase8.contract.balancing_contract import (
    BalancingContractIssue,
    BalancingContractOptions,
    BalancingContractResolver,
)


def _partition(
    name: PartitionName,
    *,
    values: list[float] | None = None,
    target: list[int] | None = None,
    row_ids: list[str] | None = None,
) -> DatasetPartition:
    index = pd.RangeIndex(4)
    return DatasetPartition(
        name=name,
        X=pd.DataFrame(
            {
                "feature_a": values or [1.0, 2.0, 3.0, 4.0],
                "feature_b": [4.0, 5.0, 6.0, 7.0],
            },
            index=index,
        ),
        y=pd.Series(target or [0, 1, 0, 1], index=index, name="target"),
        row_ids=pd.Series(
            row_ids or [f"{name.value}-{n}" for n in range(4)],
            index=index,
            name="row_id",
        ),
    )


def _result() -> Phase7Result:
    result = Phase7Result(
        train=_partition(PartitionName.TRAIN),
        validation=_partition(PartitionName.VALIDATION),
        test=_partition(PartitionName.TEST),
    )
    result.readiness = Phase7ReadinessDecision(
        ready=True,
        status=Phase7Status.PASSED,
    )
    return result


def _state() -> Phase7State:
    return Phase7State(
        selected_lineage=FeatureLineageState(
            source_columns=["feature_a", "feature_b"],
            output_columns=["feature_a", "feature_b"],
        ),
        output_fingerprint="phase7-output",
        fitted_on_training_only=True,
        targets_unchanged=True,
        row_ids_unchanged=True,
        selected_values_unchanged=True,
        schemas_consistent=True,
        test_partition_remained_protected=True,
    )


def _resolve(
    result: Phase7Result | None = None,
    *,
    state: Phase7State | None = None,
    options: BalancingContractOptions | None = None,
):
    return BalancingContractResolver(options).resolve(
        phase7_result=result or _result(),
        phase7_state=state or _state(),
    )


def test_valid_contract_copies_partitions_and_exposes_class_metadata() -> None:
    result = _result()
    contract = _resolve(result)

    assert contract.feature_columns == ["feature_a", "feature_b"]
    assert len(contract.target_classes) == 2
    assert contract.class_count == 2
    assert contract.smallest_class_count == 2
    assert contract.largest_class_count == 2
    assert not contract.has_blocking_issues
    contract.assert_ready()
    assert contract.partition(PartitionName.TRAIN) is contract.train
    assert contract.train is not result.train


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("require_phase7_state", "yes", TypeError),
        ("minimum_training_class_count", 0, ValueError),
        ("maximum_reported_columns", 0, ValueError),
        ("maximum_reported_row_ids", 0, ValueError),
        ("maximum_reported_classes", 0, ValueError),
    ],
)
def test_contract_options_reject_invalid_limits(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        BalancingContractOptions(**{field: value})


def test_resolver_reads_state_from_phase7_result_and_rejects_missing_state() -> None:
    result = _result()
    result.state = _state()
    contract = BalancingContractResolver().resolve(phase7_result=result)

    assert contract.phase7_state.output_fingerprint == "phase7-output"

    result.state = None
    result.phase_state = {}
    with pytest.raises(ValueError, match="state is unavailable"):
        BalancingContractResolver().resolve(phase7_result=result)


def test_resolver_rejects_invalid_result_and_unauthorized_test_partition() -> None:
    resolver = BalancingContractResolver()
    with pytest.raises(TypeError, match="phase7_result"):
        resolver.resolve(phase7_result=object())  # type: ignore[arg-type]

    with pytest.raises(PermissionError, match="authorized selected"):
        resolver.resolve(
            phase7_result=_result(),
            phase7_state=_state(),
            authorized_test_partition=object(),  # type: ignore[arg-type]
        )


def test_empty_training_schema_is_reported() -> None:
    result = _result()
    result.train.X = result.train.X.iloc[:, :0]

    contract = _resolve(result)

    assert any(issue.code == "P8-CONTRACT-019" for issue in contract.issues)


def test_contract_readiness_reports_blocking_issues_and_empty_class_counts() -> None:
    contract = _resolve()
    contract.training_class_counts.clear()
    assert contract.smallest_class_count == 0
    assert contract.largest_class_count == 0

    contract.issues.append(
        BalancingContractIssue(
            code="P8-TEST",
            severity="critical",
            message="blocked",
        )
    )
    assert contract.has_blocking_issues
    with pytest.raises(ValueError, match="blocking issues"):
        contract.assert_ready()


@pytest.mark.parametrize(
    ("readiness", "code"),
    [
        (None, "P8-CONTRACT-001"),
        (
            Phase7ReadinessDecision(
                ready=False,
                status=Phase7Status.FAILED,
                reasons=["blocked"],
            ),
            "P8-CONTRACT-002",
        ),
    ],
)
def test_phase7_readiness_findings_are_recorded(
    readiness: Phase7ReadinessDecision | None,
    code: str,
) -> None:
    result = _result()
    result.readiness = readiness

    contract = _resolve(result)

    finding = next(issue for issue in contract.issues if issue.code == code)
    assert finding.blocks_execution
    assert finding.severity == "critical"


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("output_fingerprint", "P8-CONTRACT-003"),
        ("fitted_on_training_only", "P8-CONTRACT-004"),
        ("selected_lineage", "P8-CONTRACT-005"),
        ("targets_unchanged", "P8-CONTRACT-006"),
        ("row_ids_unchanged", "P8-CONTRACT-007"),
        ("selected_values_unchanged", "P8-CONTRACT-008"),
        ("schemas_consistent", "P8-CONTRACT-009"),
        ("test_partition_remained_protected", "P8-CONTRACT-010"),
    ],
)
def test_phase7_state_governance_findings_are_recorded(
    field: str,
    code: str,
) -> None:
    state = _state()
    invalid_value = (
        None
        if field in {"output_fingerprint", "selected_lineage"}
        else False
    )
    setattr(state, field, invalid_value)

    contract = _resolve(state=state)

    finding = next(issue for issue in contract.issues if issue.code == code)
    assert finding.blocks_execution


def test_phase7_state_requirement_can_be_disabled() -> None:
    state = Phase7State()
    options = BalancingContractOptions(
        require_phase7_state=False,
        require_phase7_readiness=False,
        require_phase7_output_fingerprint=False,
        require_training_only_phase7_fit=False,
        require_phase7_selected_lineage=False,
        require_unchanged_phase7_targets=False,
        require_unchanged_phase7_row_ids=False,
        require_unchanged_phase7_feature_values=False,
        require_consistent_phase7_schemas=False,
        require_protected_test_partition=False,
    )

    contract = _resolve(state=state, options=options)

    assert contract.issues == []


@pytest.mark.parametrize(
    ("partition", "code"),
    [
        ("train", "P8-CONTRACT-011"),
        ("validation", "P8-CONTRACT-011"),
        ("test", "P8-CONTRACT-011"),
    ],
)
def test_partition_names_are_checked(
    partition: str,
    code: str,
) -> None:
    result = _result()
    replacement = (
        PartitionName.TEST
        if partition != "test"
        else PartitionName.TRAIN
    )
    getattr(result, partition).name = replacement

    contract = _resolve(result)

    assert any(issue.code == code for issue in contract.issues)


@pytest.mark.parametrize(
    ("partition", "code"),
    [
        ("train", "P8-CONTRACT-012"),
        ("validation", "P8-CONTRACT-013"),
        ("test", "P8-CONTRACT-014"),
    ],
)
def test_empty_partitions_are_rejected(
    partition: str,
    code: str,
) -> None:
    result = _result()
    empty = DatasetPartition(
        name=getattr(result, partition).name,
        X=pd.DataFrame(
            {"feature_a": pd.Series(dtype="float64")},
        ),
        y=pd.Series(dtype="int64", name="target"),
        row_ids=pd.Series(dtype="string", name="row_id"),
    )
    setattr(result, partition, empty)

    contract = _resolve(result)

    assert any(issue.code == code for issue in contract.issues)


def test_contract_options_can_disable_schema_and_identity_requirements() -> None:
    result = _result()
    result.validation.X = result.validation.X.drop(columns="feature_b")
    result.validation.row_ids.iloc[0] = result.train.row_ids.iloc[0]
    options = BalancingContractOptions(
        require_identical_feature_schema=False,
        require_identical_feature_order=False,
        require_compatible_feature_dtypes=False,
        require_disjoint_row_ids=False,
    )

    contract = _resolve(result, options=options)

    assert "P8-CONTRACT-021" not in {issue.code for issue in contract.issues}
    assert "P8-CONTRACT-018" not in {issue.code for issue in contract.issues}


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("schema", "P8-CONTRACT-021"),
        ("order", "P8-CONTRACT-022"),
        ("dtype", "P8-CONTRACT-023"),
        ("nonnumeric", "P8-CONTRACT-024"),
        ("missing", "P8-CONTRACT-025"),
        ("nonfinite", "P8-CONTRACT-026"),
    ],
)
def test_feature_schema_and_value_contracts_reject_invalid_partitions(
    mutation: str,
    code: str,
) -> None:
    result = _result()
    if mutation == "schema":
        result.validation.X = result.validation.X.drop(columns="feature_b")
    elif mutation == "order":
        result.validation.X = result.validation.X[["feature_b", "feature_a"]]
    elif mutation == "dtype":
        result.validation.X["feature_a"] = result.validation.X["feature_a"].astype(
            "string"
        )
    elif mutation == "nonnumeric":
        result.validation.X["feature_a"] = ["a", "b", "c", "d"]
    elif mutation == "missing":
        result.validation.X.loc[0, "feature_a"] = np.nan
    else:
        result.validation.X.loc[0, "feature_a"] = np.inf

    contract = _resolve(result)

    assert any(issue.code == code for issue in contract.issues)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("overlap", "P8-CONTRACT-018"),
        ("target_missing", "P8-CONTRACT-027"),
        ("one_class", "P8-CONTRACT-028"),
        ("undersized", "P8-CONTRACT-029"),
    ],
)
def test_identity_and_target_contracts_reject_invalid_partitions(
    mutation: str,
    code: str,
) -> None:
    result = _result()
    if mutation == "overlap":
        result.validation.row_ids.iloc[0] = result.train.row_ids.iloc[0]
    elif mutation == "target_missing":
        result.train.y.iloc[0] = np.nan
    elif mutation == "one_class":
        result.train.y[:] = 0
    else:
        result.train.y[:] = [0, 0, 0, 1]

    contract = _resolve(result)

    assert any(issue.code == code for issue in contract.issues)
