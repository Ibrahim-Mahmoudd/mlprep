"""Unit tests for the Phase 1 problem contract."""

from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase1.contract.problem_contract import (
    ProblemContractOptions,
    ProblemContractResolver,
)
from mlprep.phases.phase1.output_governance.result import (
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.types import (
    FindingCategory,
    Severity,
    TaskType,
)


def build_result(
    data: pd.DataFrame,
) -> Phase1Result:
    """Create a Phase 1 result for testing."""

    return Phase1Result(
        data=data,
        internal_row_id_column="__mlprep_row_id__",
    )


def build_state() -> Phase1State:
    """Create an empty Phase 1 state for testing."""

    return Phase1State(
        internal_row_id_column="__mlprep_row_id__",
    )


def test_valid_binary_problem_contract() -> None:
    """Resolve a valid binary-classification contract."""

    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "snapshot_date": [
                "2026-01-01",
                "2026-01-01",
                "2026-02-01",
                "2026-02-01",
            ],
            "age": [
                24,
                31,
                45,
                52,
            ],
            "churn": [
                0,
                1,
                0,
                1,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        positive_class=1,
        unit_of_observation=(
            "One customer at one monthly snapshot"
        ),
        observation_key=[
            "customer_id",
            "snapshot_date",
        ],
        entity_columns=[
            "customer_id",
        ],
        group_columns=[],
        prediction_timestamp_column=(
            "snapshot_date"
        ),
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None
    assert contract.resolved is True

    assert (
        contract.task_type
        == TaskType.BINARY_CLASSIFICATION
    )

    assert contract.target_column == "churn"
    assert contract.positive_class == 1

    assert contract.observation_key == [
        "customer_id",
        "snapshot_date",
    ]

    assert contract.entity_columns == [
        "customer_id",
    ]

    assert (
        contract.prediction_timestamp_column
        == "snapshot_date"
    )

    assert state.problem_contract == contract

    assert result.problem_contract[
        "target_column"
    ] == "churn"

    assert not result.get_blocking_findings()


def test_valid_regression_problem_contract() -> None:
    """Resolve a valid regression problem contract."""

    data = pd.DataFrame(
        {
            "property_id": [
                "P001",
                "P002",
                "P003",
            ],
            "area": [
                80.0,
                120.0,
                150.0,
            ],
            "price": [
                1_500_000.0,
                2_100_000.0,
                2_900_000.0,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.REGRESSION,
        target_column="price",
        unit_of_observation="One property listing",
        observation_key=[
            "property_id",
        ],
        entity_columns=[
            "property_id",
        ],
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None
    assert contract.resolved is True

    assert (
        contract.task_type
        == TaskType.REGRESSION
    )

    assert contract.target_column == "price"
    assert contract.positive_class is None

    assert contract.feature_columns == [
        "property_id",
        "area",
    ]


def test_empty_target_name_is_rejected() -> None:
    """Reject an empty target-column name."""

    with pytest.raises(
        ValueError,
        match="target_column cannot be empty",
    ):
        ProblemContractOptions(
            task_type=TaskType.REGRESSION,
            target_column=" ",
            unit_of_observation="One record",
        )


def test_empty_unit_of_observation_is_rejected() -> None:
    """Reject an empty unit-of-observation definition."""

    with pytest.raises(
        ValueError,
        match="unit_of_observation cannot be empty",
    ):
        ProblemContractOptions(
            task_type=TaskType.REGRESSION,
            target_column="target",
            unit_of_observation=" ",
        )


def test_binary_classification_requires_positive_class() -> None:
    """Reject binary classification without a positive class."""

    with pytest.raises(
        ValueError,
        match="positive_class is required",
    ):
        ProblemContractOptions(
            task_type=TaskType.BINARY_CLASSIFICATION,
            target_column="churn",
            unit_of_observation="One customer",
        )


def test_regression_rejects_positive_class() -> None:
    """Reject a positive class in a regression contract."""

    with pytest.raises(
        ValueError,
        match=(
            "positive_class is supported only "
            "for binary classification"
        ),
    ):
        ProblemContractOptions(
            task_type=TaskType.REGRESSION,
            target_column="price",
            positive_class=1,
            unit_of_observation="One property",
        )


def test_target_cannot_be_in_observation_key() -> None:
    """Reject the target as part of the observation key."""

    with pytest.raises(
        ValueError,
        match=(
            "target column cannot be part "
            "of the observation key"
        ),
    ):
        ProblemContractOptions(
            task_type=TaskType.REGRESSION,
            target_column="price",
            unit_of_observation="One property",
            observation_key=[
                "property_id",
                "price",
            ],
        )


def test_target_cannot_be_entity_column() -> None:
    """Reject the target as an entity column."""

    with pytest.raises(
        ValueError,
        match=(
            "target column cannot be an "
            "entity column"
        ),
    ):
        ProblemContractOptions(
            task_type=TaskType.REGRESSION,
            target_column="price",
            unit_of_observation="One property",
            entity_columns=[
                "price",
            ],
        )


def test_target_cannot_be_group_column() -> None:
    """Reject the target as a group column."""

    with pytest.raises(
        ValueError,
        match=(
            "target column cannot be a group column"
        ),
    ):
        ProblemContractOptions(
            task_type=TaskType.REGRESSION,
            target_column="price",
            unit_of_observation="One property",
            group_columns=[
                "price",
            ],
        )


def test_prediction_and_outcome_timestamps_must_differ() -> None:
    """Reject matching prediction and outcome timestamps."""

    with pytest.raises(
        ValueError,
        match=(
            "Prediction and outcome timestamp "
            "columns must be different"
        ),
    ):
        ProblemContractOptions(
            task_type=TaskType.REGRESSION,
            target_column="target",
            unit_of_observation="One record",
            prediction_timestamp_column="event_date",
            outcome_timestamp_column="event_date",
        )


def test_duplicate_contract_declarations_are_rejected() -> None:
    """Reject repeated columns within one contract declaration."""

    with pytest.raises(
        ValueError,
        match=(
            "column declarations contain duplicates"
        ),
    ):
        ProblemContractOptions(
            task_type=TaskType.REGRESSION,
            target_column="target",
            unit_of_observation="One customer",
            observation_key=[
                "customer_id",
                "customer_id",
            ],
        )


def test_missing_target_column_creates_critical_finding() -> None:
    """Report a critical finding when the target is absent."""

    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
            ],
            "age": [
                24,
                31,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        positive_class=1,
        unit_of_observation="One customer",
        observation_key=[
            "customer_id",
        ],
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is None
    assert state.problem_contract is None

    critical_findings = (
        result.get_findings_by_severity(
            Severity.CRITICAL
        )
    )

    assert len(critical_findings) >= 1

    assert any(
        finding.code == "P1-CONTRACT-004"
        for finding in critical_findings
    )

    assert any(
        finding.blocks_readiness
        for finding in critical_findings
    )


def test_empty_dataset_rows_create_critical_finding() -> None:
    """Report a critical finding for a dataset with no rows."""

    data = pd.DataFrame(
        columns=[
            "customer_id",
            "churn",
        ]
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        positive_class=1,
        unit_of_observation="One customer",
        observation_key=[
            "customer_id",
        ],
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is None

    assert any(
        finding.code == "P1-CONTRACT-003"
        for finding in result.findings
    )


def test_missing_required_column_creates_error() -> None:
    """Report missing non-target contract columns."""

    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
            ],
            "churn": [
                0,
                1,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        positive_class=1,
        unit_of_observation="One customer snapshot",
        observation_key=[
            "customer_id",
            "snapshot_date",
        ],
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None
    assert contract.resolved is False

    findings = (
        result.get_findings_by_category(
            FindingCategory.PROBLEM_CONTRACT
        )
    )

    assert any(
        finding.code == "P1-CONTRACT-005"
        and finding.blocks_readiness
        for finding in findings
    )


def test_missing_observation_key_warn_policy() -> None:
    """Allow a missing key while creating a warning."""

    data = pd.DataFrame(
        {
            "age": [
                24,
                31,
            ],
            "churn": [
                0,
                1,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churn",
        positive_class=1,
        unit_of_observation="One customer",
        missing_observation_key_policy="warn",
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None
    assert contract.resolved is True

    warnings = (
        result.get_findings_by_severity(
            Severity.WARNING
        )
    )

    assert any(
        finding.code == "P1-CONTRACT-008"
        for finding in warnings
    )


def test_missing_observation_key_allow_policy() -> None:
    """Allow a missing key without creating a warning."""

    data = pd.DataFrame(
        {
            "age": [
                24,
                31,
            ],
            "target": [
                0,
                1,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=1,
        unit_of_observation="One record",
        missing_observation_key_policy="allow",
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None
    assert contract.resolved is True

    assert not any(
        finding.code == "P1-CONTRACT-008"
        for finding in result.findings
    )

    assert not any(
        finding.code == "P1-CONTRACT-009"
        for finding in result.findings
    )


def test_missing_observation_key_error_policy() -> None:
    """Block resolution when a key is required."""

    data = pd.DataFrame(
        {
            "age": [
                24,
                31,
            ],
            "target": [
                0,
                1,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=1,
        unit_of_observation="One record",
        missing_observation_key_policy="error",
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None
    assert contract.resolved is False

    assert any(
        finding.code == "P1-CONTRACT-009"
        and finding.blocks_readiness
        for finding in result.findings
    )


def test_observation_key_missing_values_are_detected() -> None:
    """Detect missing values inside an observation key."""

    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                None,
                "C003",
            ],
            "target": [
                0,
                1,
                0,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=1,
        unit_of_observation="One customer",
        observation_key=[
            "customer_id",
        ],
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None
    assert contract.resolved is False

    finding = next(
        finding
        for finding in result.findings
        if finding.code == "P1-CONTRACT-006"
    )

    assert finding.affected_row_count == 1
    assert finding.blocks_readiness is True


def test_duplicate_observation_key_is_detected() -> None:
    """Detect duplicated business observation keys."""

    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C001",
                "C002",
            ],
            "snapshot_date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
            ],
            "target": [
                0,
                0,
                1,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=1,
        unit_of_observation=(
            "One customer at one snapshot"
        ),
        observation_key=[
            "customer_id",
            "snapshot_date",
        ],
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None
    assert contract.resolved is False

    finding = next(
        finding
        for finding in result.findings
        if finding.code == "P1-CONTRACT-007"
    )

    assert finding.affected_row_count == 2
    assert finding.blocks_readiness is True


def test_duplicate_key_can_be_allowed() -> None:
    """Allow nonunique keys when uniqueness is disabled."""

    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C001",
                "C002",
            ],
            "target": [
                0,
                0,
                1,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=1,
        unit_of_observation="One event",
        observation_key=[
            "customer_id",
        ],
        require_unique_observation_key=False,
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None
    assert contract.resolved is True

    assert not any(
        finding.code == "P1-CONTRACT-007"
        for finding in result.findings
    )


def test_invalid_timestamp_order_is_detected() -> None:
    """Detect outcomes occurring before prediction time."""

    data = pd.DataFrame(
        {
            "record_id": [
                "R001",
                "R002",
            ],
            "prediction_date": [
                "2026-03-10",
                "2026-03-20",
            ],
            "outcome_date": [
                "2026-03-15",
                "2026-03-01",
            ],
            "target": [
                0,
                1,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=1,
        unit_of_observation="One record",
        observation_key=[
            "record_id",
        ],
        prediction_timestamp_column=(
            "prediction_date"
        ),
        outcome_timestamp_column=(
            "outcome_date"
        ),
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None
    assert contract.resolved is False

    finding = next(
        finding
        for finding in result.findings
        if finding.code == "P1-CONTRACT-012"
    )

    assert finding.affected_row_count == 1
    assert finding.blocks_readiness is True


def test_unparseable_prediction_timestamp_is_detected() -> None:
    """Detect invalid prediction timestamp values."""

    data = pd.DataFrame(
        {
            "record_id": [
                "R001",
                "R002",
            ],
            "prediction_date": [
                "2026-03-10",
                "invalid-date",
            ],
            "outcome_date": [
                "2026-03-15",
                "2026-03-25",
            ],
            "target": [
                0,
                1,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=1,
        unit_of_observation="One record",
        observation_key=[
            "record_id",
        ],
        prediction_timestamp_column=(
            "prediction_date"
        ),
        outcome_timestamp_column=(
            "outcome_date"
        ),
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None
    assert contract.resolved is False

    finding = next(
        finding
        for finding in result.findings
        if finding.code == "P1-CONTRACT-010"
    )

    assert finding.affected_row_count == 1
    assert finding.blocks_readiness is True


def test_contract_builds_candidate_feature_columns() -> None:
    """Build candidate features by excluding the target."""

    data = pd.DataFrame(
        {
            "record_id": [
                "R001",
                "R002",
            ],
            "age": [
                25,
                30,
            ],
            "income": [
                5000.0,
                7000.0,
            ],
            "target": [
                0,
                1,
            ],
        }
    )

    options = ProblemContractOptions(
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        positive_class=1,
        unit_of_observation="One record",
        observation_key=[
            "record_id",
        ],
    )

    resolver = ProblemContractResolver(
        options=options
    )

    result = build_result(data)
    state = build_state()

    contract = resolver.resolve(
        data=data,
        result=result,
        state=state,
    )

    assert contract is not None

    assert contract.feature_columns == [
        "record_id",
        "age",
        "income",
    ]

    assert contract.inference_columns == [
        "record_id",
        "age",
        "income",
    ]
