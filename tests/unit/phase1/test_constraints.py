from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase1.output_governance.result import Phase1Result
from mlprep.phases.phase1.output_governance.state import Phase1State
from mlprep.phases.phase1.validation_and_risk.constraints import (
    AllowedValuesConstraint,
    ConstraintOptions,
    ConstraintValidator,
    CrossColumnConstraint,
    RangeConstraint,
    RegexConstraint,
)
from mlprep.types import Severity


def _validate(
    frame: pd.DataFrame,
    options: ConstraintOptions,
) -> tuple[bool, Phase1Result]:
    result = Phase1Result(
        data=frame,
        internal_row_id_column="__row_id__",
    )
    is_valid = ConstraintValidator(options).validate(
        frame,
        result,
        Phase1State(),
    )
    return is_valid, result


def test_range_constraint_reports_inclusive_bounds_and_missing_values() -> None:
    frame = pd.DataFrame(
        {"age": [0, 1, 10, 11, None]},
        index=[10, 11, 12, 13, 14],
    )
    valid, result = _validate(
        frame,
        ConstraintOptions(
            maximum_examples=2,
            ranges=[
                RangeConstraint(
                    name="valid_age",
                    column="age",
                    minimum=0,
                    maximum=10,
                    inclusive_minimum=False,
                    inclusive_maximum=True,
                    allow_missing=False,
                    severity="warning",
                    blocks_readiness=True,
                )
            ],
        ),
    )

    finding = result.findings[0]
    assert not valid
    assert finding.affected_row_count == 3
    assert finding.affected_row_percentage == 60.0
    assert finding.example_row_ids == [10, 13]
    assert finding.severity == Severity.WARNING
    assert finding.blocks_readiness


def test_allowed_values_respects_case_and_missing_policies() -> None:
    frame = pd.DataFrame(
        {"status": ["ACTIVE", "closed", "unknown", None]},
        index=[1, 2, 3, 4],
    )
    valid, result = _validate(
        frame,
        ConstraintOptions(
            allowed_values=[
                AllowedValuesConstraint(
                    name="known_status",
                    column="status",
                    allowed_values={"active", "closed"},
                    case_sensitive=False,
                    allow_missing=False,
                )
            ],
        ),
    )

    finding = result.findings[0]
    assert not valid
    assert finding.affected_row_count == 2
    assert finding.example_row_ids == [3, 4]
    assert finding.columns == ["status"]


def test_regex_full_match_and_search_have_distinct_behavior() -> None:
    frame = pd.DataFrame(
        {"code": ["AB12", "prefix-AB12", None]},
        index=[20, 21, 22],
    )
    full_valid, full_result = _validate(
        frame,
        ConstraintOptions(
            regex_patterns=[
                RegexConstraint(
                    name="full_code",
                    column="code",
                    pattern=r"[A-Z]{2}\d{2}",
                    allow_missing=False,
                )
            ],
        ),
    )
    search_valid, search_result = _validate(
        frame,
        ConstraintOptions(
            regex_patterns=[
                RegexConstraint(
                    name="embedded_code",
                    column="code",
                    pattern=r"[A-Z]{2}\d{2}",
                    full_match=False,
                )
            ],
        ),
    )

    assert not full_valid
    assert full_result.findings[0].example_row_ids == [21, 22]
    assert search_valid
    assert search_result.findings == []


def test_cross_column_rule_checks_missing_values_and_index() -> None:
    frame = pd.DataFrame(
        {
            "start": [1, 3, 4],
            "end": [2, 2, 5],
        },
        index=[30, 31, 32],
    )

    def rule(data: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [True, False, pd.NA],
            index=data.index,
            dtype="boolean",
        )

    valid, result = _validate(
        frame,
        ConstraintOptions(
            cross_column_rules=[
                CrossColumnConstraint(
                    name="ordered_dates",
                    columns=["start", "end"],
                    rule=rule,
                )
            ],
        ),
    )

    assert not valid
    assert result.findings[0].affected_row_count == 2
    assert result.findings[0].example_row_ids == [31, 32]
    assert result.findings[0].columns == ["start", "end"]


def test_cross_column_rule_rejects_nonseries_and_misaligned_results() -> None:
    frame = pd.DataFrame({"value": [1, 2]}, index=[4, 5])
    for rule, exception, message in [
        (lambda data: [True, False], TypeError, "must return a pandas Series"),
        (
            lambda data: pd.Series([True, False], index=[0, 1]),
            ValueError,
            "returned an invalid index",
        ),
    ]:
        with pytest.raises(exception, match=message):
            ConstraintValidator(
                ConstraintOptions(
                    cross_column_rules=[
                        CrossColumnConstraint(
                            name="custom",
                            columns=["value"],
                            rule=rule,
                        )
                    ],
                )
            ).validate(
                frame,
                Phase1Result(
                    data=frame,
                    internal_row_id_column="__row_id__",
                ),
                Phase1State(),
            )


def test_missing_columns_and_disabled_validation_are_explicit() -> None:
    frame = pd.DataFrame({"present": [1, 2]})
    valid, result = _validate(
        frame,
        ConstraintOptions(
            ranges=[
                RangeConstraint(name="range", column="missing")
            ],
            allowed_values=[
                AllowedValuesConstraint(
                    name="allowed",
                    column="also_missing",
                    allowed_values={1},
                )
            ],
            regex_patterns=[
                RegexConstraint(
                    name="regex",
                    column="another_missing",
                    pattern="x",
                )
            ],
            cross_column_rules=[
                CrossColumnConstraint(
                    name="cross",
                    columns=["missing_a", "missing_b"],
                    rule=lambda data: pd.Series(True, index=data.index),
                )
            ],
        ),
    )
    disabled, disabled_result = _validate(
        frame,
        ConstraintOptions(
            enabled=False,
            ranges=[
                RangeConstraint(name="range", column="missing")
            ],
        ),
    )

    assert not valid
    assert len(result.findings) == 4
    assert all(finding.blocks_readiness for finding in result.findings)
    assert result.findings[-1].columns == ["missing_a", "missing_b"]
    assert disabled
    assert disabled_result.findings == []


def test_nonblocking_info_finding_does_not_make_validation_fail() -> None:
    frame = pd.DataFrame({"score": [-1, 1]})
    valid, result = _validate(
        frame,
        ConstraintOptions(
            ranges=[
                RangeConstraint(
                    name="informational_range",
                    column="score",
                    minimum=0,
                    severity="info",
                )
            ],
        ),
    )

    assert valid
    assert len(result.findings) == 1
    assert result.findings[0].severity == Severity.INFO
    assert not result.findings[0].blocks_readiness
