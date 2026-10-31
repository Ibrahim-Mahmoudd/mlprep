"""Unit tests for Phase 1 column-name normalization."""

from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase1.deterministic_normalization.column_names import (
    ColumnNameNormalizer,
    ColumnNameOptions,
)
from mlprep.phases.phase1.output_governance.result import (
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.types import (
    CollisionPolicy,
    FindingCategory,
    Severity,
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


def test_normalize_column_names_to_snake_case() -> None:
    """Normalize common column names to snake case."""

    data = pd.DataFrame(
        {
            "Customer ID": [
                "C001",
                "C002",
            ],
            "Monthly Income": [
                5000.0,
                7000.0,
            ],
            "JoinDate": [
                "2026-01-01",
                "2026-02-01",
            ],
        }
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "customer_id",
        "monthly_income",
        "join_date",
    ]


def test_original_dataframe_is_not_modified() -> None:
    """Keep the original DataFrame unchanged."""

    data = pd.DataFrame(
        {
            "Customer ID": [
                "C001",
                "C002",
            ],
            "Monthly Income": [
                5000,
                7000,
            ],
        }
    )

    original_columns = list(data.columns)

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(data.columns) == original_columns

    assert list(normalized_data.columns) == [
        "customer_id",
        "monthly_income",
    ]

    assert normalized_data is not data


def test_strip_leading_and_trailing_whitespace() -> None:
    """Remove whitespace around column names."""

    data = pd.DataFrame(
        {
            " Customer ID ": [
                "C001",
                "C002",
            ],
            " Age ": [
                20,
                30,
            ],
        }
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "customer_id",
        "age",
    ]


def test_replace_hyphens_and_spaces() -> None:
    """Replace spaces and hyphens with underscores."""

    data = pd.DataFrame(
        {
            "Customer-ID": [
                "C001",
                "C002",
            ],
            "Monthly Income": [
                5000,
                7000,
            ],
        }
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "customer_id",
        "monthly_income",
    ]


def test_remove_unsupported_special_characters() -> None:
    """Remove unsupported characters from names."""

    data = pd.DataFrame(
        {
            "Customer@ID": [
                "C001",
                "C002",
            ],
            "Monthly$Income": [
                5000,
                7000,
            ],
            "City.Name": [
                "Cairo",
                "Giza",
            ],
        }
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "customerid",
        "monthlyincome",
        "cityname",
    ]


def test_collapse_repeated_separators() -> None:
    """Collapse repeated underscores into one separator."""

    data = pd.DataFrame(
        {
            "Customer___ID": [
                "C001",
                "C002",
            ],
            "Monthly   Income": [
                5000,
                7000,
            ],
        }
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "customer_id",
        "monthly_income",
    ]


def test_remove_leading_and_trailing_separators() -> None:
    """Remove separators from the beginning and end."""

    data = pd.DataFrame(
        {
            "__Customer ID__": [
                "C001",
                "C002",
            ],
            "_Age_": [
                20,
                30,
            ],
        }
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "customer_id",
        "age",
    ]


def test_lower_case_style() -> None:
    """Normalize names using lowercase style."""

    data = pd.DataFrame(
        {
            "Customer ID": [
                "C001",
                "C002",
            ],
            "AGE": [
                20,
                30,
            ],
        }
    )

    options = ColumnNameOptions(
        case_style="lower",
    )

    normalizer = ColumnNameNormalizer(
        options=options
    )

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "customer_id",
        "age",
    ]


def test_upper_case_style() -> None:
    """Normalize names using uppercase style."""

    data = pd.DataFrame(
        {
            "Customer ID": [
                "C001",
                "C002",
            ],
            "Age": [
                20,
                30,
            ],
        }
    )

    options = ColumnNameOptions(
        case_style="upper",
    )

    normalizer = ColumnNameNormalizer(
        options=options
    )

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "CUSTOMER_ID",
        "AGE",
    ]


def test_preserve_case_style() -> None:
    """Preserve the original letter case."""

    data = pd.DataFrame(
        {
            "Customer ID": [
                "C001",
                "C002",
            ],
            "AGE": [
                20,
                30,
            ],
        }
    )

    options = ColumnNameOptions(
        case_style="preserve",
    )

    normalizer = ColumnNameNormalizer(
        options=options
    )

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "Customer_ID",
        "AGE",
    ]


def test_apply_explicit_mapping() -> None:
    """Apply explicit column-name mappings."""

    data = pd.DataFrame(
        {
            "CustNo": [
                "C001",
                "C002",
            ],
            "TargetValue": [
                0,
                1,
            ],
        }
    )

    options = ColumnNameOptions(
        explicit_mapping={
            "CustNo": "customer_id",
            "TargetValue": "churn",
        }
    )

    normalizer = ColumnNameNormalizer(
        options=options
    )

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "customer_id",
        "churn",
    ]

    assert (
        state.original_to_normalized_names[
            "CustNo"
        ]
        == "customer_id"
    )

    assert (
        state.original_to_normalized_names[
            "TargetValue"
        ]
        == "churn"
    )


def test_name_mapping_is_registered_in_state() -> None:
    """Register original and normalized name mappings."""

    data = pd.DataFrame(
        {
            "Customer ID": [
                "C001",
                "C002",
            ],
            "Monthly Income": [
                5000,
                7000,
            ],
        }
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert (
        state.original_to_normalized_names[
            "Customer ID"
        ]
        == "customer_id"
    )

    assert (
        state.normalized_to_original_names[
            "customer_id"
        ]
        == "Customer ID"
    )

    assert (
        state.column_name_states[
            "customer_id"
        ].was_changed
        is True
    )

    assert (
        state.column_name_states[
            "customer_id"
        ].original_name
        == "Customer ID"
    )


def test_normalization_events_are_created() -> None:
    """Create events for renamed columns."""

    data = pd.DataFrame(
        {
            "Customer ID": [
                "C001",
                "C002",
            ],
            "age": [
                20,
                30,
            ],
        }
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert len(
        result.normalization_events
    ) == 1

    event = result.normalization_events[0]

    assert event.operation == "rename_column"
    assert event.column == "customer_id"

    assert (
        event.before_summary["column_name"]
        == "Customer ID"
    )

    assert (
        event.after_summary["column_name"]
        == "customer_id"
    )


def test_summary_finding_is_created() -> None:
    """Create one summary finding for normalized names."""

    data = pd.DataFrame(
        {
            "Customer ID": [
                "C001",
                "C002",
            ],
            "Monthly Income": [
                5000,
                7000,
            ],
        }
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    findings = (
        result.get_findings_by_category(
            FindingCategory.COLUMN_NAME
        )
    )

    summary_finding = next(
        finding
        for finding in findings
        if finding.code == "P1-COLNAME-001"
    )

    assert summary_finding.severity == Severity.INFO

    assert set(
        summary_finding.columns
    ) == {
        "customer_id",
        "monthly_income",
    }


def test_collision_raise_policy_creates_critical_finding() -> None:
    """Report a critical finding for normalized-name collisions."""

    data = pd.DataFrame(
        [
            [
                "C001",
                "C001",
            ],
            [
                "C002",
                "C002",
            ],
        ],
        columns=[
            "Customer ID",
            "customer-id",
        ],
    )

    options = ColumnNameOptions(
        collision_policy=CollisionPolicy.RAISE,
    )

    normalizer = ColumnNameNormalizer(
        options=options
    )

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "customer_id",
        "customer_id",
    ]

    critical_findings = (
        result.get_findings_by_severity(
            Severity.CRITICAL
        )
    )

    assert any(
        finding.code == "P1-COLNAME-002"
        and finding.blocks_readiness
        for finding in critical_findings
    )


def test_collision_add_suffix_policy() -> None:
    """Resolve normalized-name collisions with suffixes."""

    data = pd.DataFrame(
        [
            [
                "C001",
                "A001",
                "X001",
            ],
            [
                "C002",
                "A002",
                "X002",
            ],
        ],
        columns=[
            "Customer ID",
            "customer-id",
            "customer_id",
        ],
    )

    options = ColumnNameOptions(
        collision_policy=(
            CollisionPolicy.ADD_SUFFIX
        ),
    )

    normalizer = ColumnNameNormalizer(
        options=options
    )

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "customer_id",
        "customer_id_2",
        "customer_id_3",
    ]

    warning_findings = (
        result.get_findings_by_severity(
            Severity.WARNING
        )
    )

    assert any(
        finding.code == "P1-COLNAME-003"
        for finding in warning_findings
    )

    assert (
        state.column_name_states[
            "customer_id_2"
        ].suffix_added
        == "_2"
    )

    assert (
        state.column_name_states[
            "customer_id_3"
        ].suffix_added
        == "_3"
    )


def test_reserved_name_creates_critical_finding() -> None:
    """Reject an input column using a reserved internal name."""

    data = pd.DataFrame(
        {
            "__mlprep_row_id__": [
                1,
                2,
            ],
            "target": [
                0,
                1,
            ],
        }
    )

    options = ColumnNameOptions(
        reserved_names={
            "__mlprep_row_id__",
        }
    )

    normalizer = ColumnNameNormalizer(
        options=options
    )

    result = build_result(data)
    state = build_state()

    normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert any(
        finding.code == "P1-COLNAME-005"
        and finding.blocks_readiness
        for finding in result.findings
    )


def test_empty_normalized_name_creates_critical_finding() -> None:
    """Detect a name that becomes empty after cleaning."""

    data = pd.DataFrame(
        [
            [
                10,
                0,
            ],
            [
                20,
                1,
            ],
        ],
        columns=[
            "@@@",
            "target",
        ],
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert normalized_data.columns[0] == ""

    assert any(
        finding.code == "P1-COLNAME-004"
        and finding.blocks_readiness
        for finding in result.findings
    )


def test_maximum_name_length() -> None:
    """Truncate names longer than the configured limit."""

    data = pd.DataFrame(
        {
            "This Is A Very Long Column Name": [
                1,
                2,
            ],
            "target": [
                0,
                1,
            ],
        }
    )

    options = ColumnNameOptions(
        maximum_length=10,
    )

    normalizer = ColumnNameNormalizer(
        options=options
    )

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert len(
        normalized_data.columns[0]
    ) == 10

    assert (
        normalized_data.columns[0]
        == "this_is_a_"
    )


def test_disabled_normalization_keeps_names() -> None:
    """Keep all names unchanged when disabled."""

    data = pd.DataFrame(
        {
            "Customer ID": [
                "C001",
                "C002",
            ],
            "Monthly Income": [
                5000,
                7000,
            ],
        }
    )

    options = ColumnNameOptions(
        enabled=False,
    )

    normalizer = ColumnNameNormalizer(
        options=options
    )

    result = build_result(data)
    state = build_state()

    normalized_data = normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    assert list(normalized_data.columns) == [
        "Customer ID",
        "Monthly Income",
    ]

    assert result.normalization_events == []

    assert (
        state.original_to_normalized_names[
            "Customer ID"
        ]
        == "Customer ID"
    )


def test_normalize_reference() -> None:
    """Normalize one configured column reference."""

    data = pd.DataFrame(
        {
            "Customer ID": [
                "C001",
                "C002",
            ],
            "target": [
                0,
                1,
            ],
        }
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    normalized_reference = (
        normalizer.normalize_reference(
            column="Customer ID",
            state=state,
        )
    )

    assert normalized_reference == "customer_id"


def test_normalize_multiple_references() -> None:
    """Normalize multiple configured references."""

    data = pd.DataFrame(
        {
            "Customer ID": [
                "C001",
                "C002",
            ],
            "Snapshot Date": [
                "2026-01-01",
                "2026-02-01",
            ],
            "target": [
                0,
                1,
            ],
        }
    )

    normalizer = ColumnNameNormalizer()

    result = build_result(data)
    state = build_state()

    normalizer.normalize(
        data=data,
        result=result,
        state=state,
    )

    normalized_references = (
        normalizer.normalize_references(
            columns=[
                "Customer ID",
                "Snapshot Date",
            ],
            state=state,
        )
    )

    assert normalized_references == [
        "customer_id",
        "snapshot_date",
    ]


def test_invalid_maximum_length_is_rejected() -> None:
    """Reject a nonpositive maximum name length."""

    with pytest.raises(
        ValueError,
        match=(
            "maximum_length must be greater than zero"
        ),
    ):
        ColumnNameOptions(
            maximum_length=0,
        )


def test_explicit_mapping_policy_requires_mapping() -> None:
    """Require mappings for the explicit-mapping policy."""

    with pytest.raises(
        ValueError,
        match="explicit_mapping is required",
    ):
        ColumnNameOptions(
            collision_policy=(
                CollisionPolicy.USE_EXPLICIT_MAPPING
            ),
            explicit_mapping={},
        )