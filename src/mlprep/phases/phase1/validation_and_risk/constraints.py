"""Declarative data constraints for Phase 1."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import Phase1State
from mlprep.types import (
    FindingAction,
    FindingCategory,
    Severity,
)

ConstraintSeverity = Literal[
    "info",
    "warning",
    "error",
    "critical",
]


@dataclass(slots=True)
class RangeConstraint:
    """Minimum and maximum constraint for one column."""

    name: str
    column: str

    minimum: float | int | pd.Timestamp | None = None
    maximum: float | int | pd.Timestamp | None = None

    inclusive_minimum: bool = True
    inclusive_maximum: bool = True

    allow_missing: bool = True
    severity: ConstraintSeverity = "error"
    blocks_readiness: bool = False


@dataclass(slots=True)
class AllowedValuesConstraint:
    """Allowed-value constraint for one column."""

    name: str
    column: str
    allowed_values: set[object]

    allow_missing: bool = True
    case_sensitive: bool = True

    severity: ConstraintSeverity = "error"
    blocks_readiness: bool = False


@dataclass(slots=True)
class RegexConstraint:
    """Regular-expression constraint for one text column."""

    name: str
    column: str
    pattern: str

    allow_missing: bool = True
    full_match: bool = True

    severity: ConstraintSeverity = "error"
    blocks_readiness: bool = False


@dataclass(slots=True)
class CrossColumnConstraint:
    """Registered cross-column validation rule."""

    name: str
    columns: list[str]
    rule: Callable[[pd.DataFrame], pd.Series]

    severity: ConstraintSeverity = "error"
    blocks_readiness: bool = False


@dataclass(slots=True)
class ConstraintOptions:
    """Collection of Phase 1 constraints."""

    enabled: bool = True

    ranges: list[RangeConstraint] = field(
        default_factory=list
    )

    allowed_values: list[
        AllowedValuesConstraint
    ] = field(default_factory=list)

    regex_patterns: list[
        RegexConstraint
    ] = field(default_factory=list)

    cross_column_rules: list[
        CrossColumnConstraint
    ] = field(default_factory=list)

    maximum_examples: int = 5


class ConstraintValidator:
    """Validate declarative column and cross-column constraints."""

    component_name = "constraints"

    def __init__(
        self,
        options: ConstraintOptions | None = None,
    ) -> None:
        """Initialize the constraint validator."""

        self.options = options or ConstraintOptions()

    def validate(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> bool:
        """Validate all configured constraints."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Constraint validation requires a pandas DataFrame."
            )

        if not self.options.enabled:
            return True

        is_valid = True

        for constraint in self.options.ranges:
            is_valid = (
                self._validate_range(
                    data=data,
                    constraint=constraint,
                    result=result,
                )
                and is_valid
            )

        for constraint in self.options.allowed_values:
            is_valid = (
                self._validate_allowed_values(
                    data=data,
                    constraint=constraint,
                    result=result,
                )
                and is_valid
            )

        for constraint in self.options.regex_patterns:
            is_valid = (
                self._validate_regex(
                    data=data,
                    constraint=constraint,
                    result=result,
                )
                and is_valid
            )

        for constraint in self.options.cross_column_rules:
            is_valid = (
                self._validate_cross_column(
                    data=data,
                    constraint=constraint,
                    result=result,
                )
                and is_valid
            )

        return is_valid

    def _validate_range(
        self,
        *,
        data: pd.DataFrame,
        constraint: RangeConstraint,
        result: Phase1Result,
    ) -> bool:
        """Validate one range constraint."""

        if constraint.column not in data.columns:
            return self._report_missing_columns(
                columns=[constraint.column],
                constraint_name=constraint.name,
                result=result,
            )

        series = data[constraint.column]

        invalid_mask = pd.Series(
            False,
            index=data.index,
            dtype=bool,
        )

        if not constraint.allow_missing:
            invalid_mask |= series.isna()

        valid_values = series.notna()

        if constraint.minimum is not None:
            if constraint.inclusive_minimum:
                invalid_mask |= (
                    valid_values
                    & (series < constraint.minimum)
                )
            else:
                invalid_mask |= (
                    valid_values
                    & (series <= constraint.minimum)
                )

        if constraint.maximum is not None:
            if constraint.inclusive_maximum:
                invalid_mask |= (
                    valid_values
                    & (series > constraint.maximum)
                )
            else:
                invalid_mask |= (
                    valid_values
                    & (series >= constraint.maximum)
                )

        return self._report_violations(
            data=data,
            invalid_mask=invalid_mask,
            constraint_name=constraint.name,
            columns=[constraint.column],
            severity=constraint.severity,
            blocks_readiness=constraint.blocks_readiness,
            result=result,
        )

    def _validate_allowed_values(
        self,
        *,
        data: pd.DataFrame,
        constraint: AllowedValuesConstraint,
        result: Phase1Result,
    ) -> bool:
        """Validate one allowed-values constraint."""

        if constraint.column not in data.columns:
            return self._report_missing_columns(
                columns=[constraint.column],
                constraint_name=constraint.name,
                result=result,
            )

        series = data[constraint.column]

        if constraint.case_sensitive:
            allowed_values = constraint.allowed_values
            checked_series = series
        else:
            allowed_values = {
                str(value).casefold()
                for value in constraint.allowed_values
            }

            checked_series = series.map(
                lambda value: (
                    str(value).casefold()
                    if pd.notna(value)
                    else value
                )
            )

        invalid_mask = (
            series.notna()
            & ~checked_series.isin(allowed_values)
        )

        if not constraint.allow_missing:
            invalid_mask |= series.isna()

        return self._report_violations(
            data=data,
            invalid_mask=invalid_mask,
            constraint_name=constraint.name,
            columns=[constraint.column],
            severity=constraint.severity,
            blocks_readiness=constraint.blocks_readiness,
            result=result,
        )

    def _validate_regex(
        self,
        *,
        data: pd.DataFrame,
        constraint: RegexConstraint,
        result: Phase1Result,
    ) -> bool:
        """Validate one regex constraint."""

        if constraint.column not in data.columns:
            return self._report_missing_columns(
                columns=[constraint.column],
                constraint_name=constraint.name,
                result=result,
            )

        compiled_pattern = re.compile(
            constraint.pattern
        )

        series = data[constraint.column]

        def value_is_valid(value: object) -> bool:
            if pd.isna(value):
                return constraint.allow_missing

            text = str(value)

            if constraint.full_match:
                return (
                    compiled_pattern.fullmatch(text)
                    is not None
                )

            return (
                compiled_pattern.search(text)
                is not None
            )

        valid_mask = series.map(
            value_is_valid
        )

        return self._report_violations(
            data=data,
            invalid_mask=~valid_mask,
            constraint_name=constraint.name,
            columns=[constraint.column],
            severity=constraint.severity,
            blocks_readiness=constraint.blocks_readiness,
            result=result,
        )

    def _validate_cross_column(
        self,
        *,
        data: pd.DataFrame,
        constraint: CrossColumnConstraint,
        result: Phase1Result,
    ) -> bool:
        """Validate one registered cross-column rule."""

        missing_columns = [
            column
            for column in constraint.columns
            if column not in data.columns
        ]

        if missing_columns:
            return self._report_missing_columns(
                columns=missing_columns,
                constraint_name=constraint.name,
                result=result,
            )

        valid_mask = constraint.rule(data)

        if not isinstance(valid_mask, pd.Series):
            raise TypeError(
                f"Constraint {constraint.name!r} must return a pandas Series."
            )

        if not valid_mask.index.equals(data.index):
            raise ValueError(
                f"Constraint {constraint.name!r} returned an invalid index."
            )

        return self._report_violations(
            data=data,
            invalid_mask=~valid_mask.fillna(False),
            constraint_name=constraint.name,
            columns=constraint.columns,
            severity=constraint.severity,
            blocks_readiness=constraint.blocks_readiness,
            result=result,
        )

    def _report_violations(
        self,
        *,
        data: pd.DataFrame,
        invalid_mask: pd.Series,
        constraint_name: str,
        columns: list[str],
        severity: ConstraintSeverity,
        blocks_readiness: bool,
        result: Phase1Result,
    ) -> bool:
        """Report rows violating one constraint."""

        invalid_count = int(
            invalid_mask.sum()
        )

        if invalid_count == 0:
            return True

        finding_severity = Severity(
            severity
        )

        result.add_finding(
            Finding(
                code="P1-CONSTRAINT-001",
                category=FindingCategory.CONSTRAINT,
                severity=finding_severity,
                component=self.component_name,
                message=(
                    f"Constraint {constraint_name!r} was violated."
                ),
                columns=columns,
                affected_row_count=invalid_count,
                affected_row_percentage=round(
                    invalid_count / len(data) * 100.0,
                    6,
                )
                if len(data) > 0
                else 0.0,
                example_row_ids=[
                    self._safe_row_id(index)
                    for index in data.index[
                        invalid_mask
                    ].tolist()[
                        : self.options.maximum_examples
                    ]
                ],
                recommended_action=FindingAction.REPORT,
                applied_action=FindingAction.REPORT,
                blocks_readiness=(
                    blocks_readiness
                    or finding_severity
                    == Severity.CRITICAL
                ),
                details={
                    "constraint_name": constraint_name,
                },
            )
        )

        return not (
            blocks_readiness
            or finding_severity
            in {
                Severity.ERROR,
                Severity.CRITICAL,
            }
        )

    def _report_missing_columns(
        self,
        *,
        columns: list[str],
        constraint_name: str,
        result: Phase1Result,
    ) -> bool:
        """Report missing columns required by a constraint."""

        result.add_finding(
            Finding(
                code="P1-CONSTRAINT-002",
                category=FindingCategory.CONSTRAINT,
                severity=Severity.ERROR,
                component=self.component_name,
                message=(
                    f"Constraint {constraint_name!r} references "
                    "missing columns."
                ),
                columns=columns,
                recommended_action=FindingAction.RAISE,
                applied_action=FindingAction.REPORT,
                blocks_readiness=True,
            )
        )

        return False

    @staticmethod
    def _safe_row_id(
        value: object,
    ) -> str | int:
        """Convert an index into a supported identifier."""

        if isinstance(value, (str, int)):
            return value

        return str(value)