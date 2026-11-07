"""Column selection and analysis-family resolution for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Number

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_timedelta64_dtype,
)

from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)
from mlprep.types import (
    ColumnRole,
    SemanticType,
)


@dataclass(slots=True)
class ColumnSelectionOptions:
    """Options controlling Phase 3 analysis-column selection."""

    include_columns: list[str] | None = None

    exclude_columns: list[str] = field(
        default_factory=list
    )

    explicit_numeric_columns: list[str] = field(
        default_factory=list
    )

    explicit_categorical_columns: list[str] = field(
        default_factory=list
    )

    explicit_datetime_columns: list[str] = field(
        default_factory=list
    )

    explicit_text_columns: list[str] = field(
        default_factory=list
    )

    explicit_boolean_columns: list[str] = field(
        default_factory=list
    )

    include_sensitive_columns: bool = False
    include_identifier_columns: bool = False
    include_internal_columns: bool = False

    use_phase1_semantic_types: bool = True

    free_text_unique_ratio_threshold: float = 0.50
    free_text_average_length_threshold: float = 40.0

    categorical_unique_limit: int = 100

    def __post_init__(self) -> None:
        """Validate column-selection options."""

        if not (
            0.0
            <= self.free_text_unique_ratio_threshold
            <= 1.0
        ):
            raise ValueError(
                "free_text_unique_ratio_threshold must be "
                "between zero and one."
            )

        if (
            self.free_text_average_length_threshold
            < 0.0
        ):
            raise ValueError(
                "free_text_average_length_threshold cannot "
                "be negative."
            )

        if self.categorical_unique_limit < 2:
            raise ValueError(
                "categorical_unique_limit must be at least two."
            )

        explicit_groups = {
            "numeric": self.explicit_numeric_columns,
            "categorical": (
                self.explicit_categorical_columns
            ),
            "datetime": (
                self.explicit_datetime_columns
            ),
            "text": self.explicit_text_columns,
            "boolean": (
                self.explicit_boolean_columns
            ),
        }

        column_to_groups: dict[
            str,
            list[str],
        ] = {}

        for group_name, columns in (
            explicit_groups.items()
        ):
            for column in columns:
                column_to_groups.setdefault(
                    column,
                    [],
                ).append(group_name)

        conflicting_columns = {
            column: groups
            for column, groups
            in column_to_groups.items()
            if len(groups) > 1
        }

        if conflicting_columns:
            raise ValueError(
                "Columns cannot be explicitly assigned to "
                "multiple analysis families: "
                f"{conflicting_columns}"
            )


@dataclass(slots=True)
class AnalysisColumnGroups:
    """Resolved Phase 3 analysis-column groups."""

    all_features: list[str] = field(
        default_factory=list
    )

    numeric: list[str] = field(
        default_factory=list
    )

    categorical: list[str] = field(
        default_factory=list
    )

    datetime: list[str] = field(
        default_factory=list
    )

    text: list[str] = field(
        default_factory=list
    )

    boolean: list[str] = field(
        default_factory=list
    )

    timedelta: list[str] = field(
        default_factory=list
    )

    identifiers: list[str] = field(
        default_factory=list
    )

    sensitive: list[str] = field(
        default_factory=list
    )

    ignored: list[str] = field(
        default_factory=list
    )

    unknown: list[str] = field(
        default_factory=list
    )

    target_column: str | None = None

    def analyzable_columns(self) -> list[str]:
        """Return columns assigned to analysis families."""

        return list(
            dict.fromkeys(
                [
                    *self.numeric,
                    *self.categorical,
                    *self.datetime,
                    *self.text,
                    *self.boolean,
                    *self.timedelta,
                ]
            )
        )

    def family_for(
        self,
        column: str,
    ) -> str:
        """Return the resolved analysis family."""

        families = {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "datetime": self.datetime,
            "text": self.text,
            "boolean": self.boolean,
            "timedelta": self.timedelta,
            "identifier": self.identifiers,
            "sensitive": self.sensitive,
            "ignored": self.ignored,
            "unknown": self.unknown,
        }

        matches = [
            family
            for family, columns
            in families.items()
            if column in columns
        ]

        if not matches:
            raise KeyError(
                f"Column {column!r} is not registered."
            )

        return matches[0]


class ColumnSelector:
    """Assign training columns to Phase 3 analysis families."""

    component_name = "column_selection"

    def __init__(
        self,
        options: ColumnSelectionOptions | None = None,
    ) -> None:
        """Initialize the column selector."""

        self.options = (
            options or ColumnSelectionOptions()
        )

    def resolve(
        self,
        *,
        contract: EDAContract,
    ) -> AnalysisColumnGroups:
        """Resolve all analysis-column groups."""

        frame = contract.train.X

        phase1_state = contract.phase1_state

        candidate_columns = self._candidate_columns(
            frame=frame
        )

        groups = AnalysisColumnGroups(
            all_features=list(
                candidate_columns
            ),
            target_column=(
                contract.target_column
            ),
        )

        explicit_assignments = (
            self._explicit_assignments()
        )

        for column in candidate_columns:
            if column in explicit_assignments:
                self._assign_family(
                    groups=groups,
                    column=column,
                    family=explicit_assignments[
                        column
                    ],
                )

                continue

            column_state = (
                phase1_state
                .column_states
                .get(column)
            )

            if column_state is not None:
                if column_state.is_sensitive:
                    groups.sensitive.append(column)

                    if not (
                        self.options
                        .include_sensitive_columns
                    ):
                        groups.ignored.append(column)
                        continue

                if (
                    ColumnRole.INTERNAL
                    in column_state.roles
                    and not self.options.include_internal_columns
                ):
                    groups.ignored.append(column)
                    continue

                if (
                    ColumnRole.IDENTIFIER
                    in column_state.roles
                    or column_state
                    .is_identifier_like
                ):
                    groups.identifiers.append(
                        column
                    )

                    if not (
                        self.options
                        .include_identifier_columns
                    ):
                        groups.ignored.append(
                            column
                        )
                        continue

                if (
                    ColumnRole.IGNORED
                    in column_state.roles
                ):
                    groups.ignored.append(column)
                    continue

            semantic_type = (
                self._semantic_type(
                    column=column,
                    phase1_state=phase1_state,
                )
            )

            family = self._family_from_semantic_type(
                semantic_type
            )

            if family is None:
                family = self._family_from_dtype(
                    series=frame[column]
                )

            if family is None:
                family = self._infer_object_family(
                    series=frame[column]
                )

            if family is None:
                groups.unknown.append(column)
                continue

            self._assign_family(
                groups=groups,
                column=column,
                family=family,
            )

        self._remove_duplicate_memberships(
            groups
        )

        return groups

    def _candidate_columns(
        self,
        *,
        frame: pd.DataFrame,
    ) -> list[object]:
        """Select columns eligible for analysis."""

        excluded = set(
            self.options.exclude_columns
        )

        if self.options.include_columns is not None:
            return [
                column
                for column
                in self.options.include_columns
                if (
                    column in frame.columns
                    and column not in excluded
                )
            ]

        return [
            str(column)
            for column in frame.columns
            if str(column) not in excluded
        ]

    def _explicit_assignments(
        self,
    ) -> dict[str, str]:
        """Build explicit column-family assignments."""

        assignments: dict[str, str] = {}

        explicit_groups = {
            "numeric": (
                self.options
                .explicit_numeric_columns
            ),
            "categorical": (
                self.options
                .explicit_categorical_columns
            ),
            "datetime": (
                self.options
                .explicit_datetime_columns
            ),
            "text": (
                self.options
                .explicit_text_columns
            ),
            "boolean": (
                self.options
                .explicit_boolean_columns
            ),
        }

        for family, columns in (
            explicit_groups.items()
        ):
            for column in columns:
                assignments[column] = family

        return assignments

    def _semantic_type(
        self,
        *,
        column: str,
        phase1_state: Phase1State,
    ) -> SemanticType:
        """Return a Phase 1 semantic type."""

        if not self.options.use_phase1_semantic_types:
            return SemanticType.UNKNOWN

        column_state = (
            phase1_state
            .column_states
            .get(column)
        )

        if column_state is None:
            return SemanticType.UNKNOWN

        return column_state.semantic_type

    @staticmethod
    def _family_from_semantic_type(
        semantic_type: SemanticType,
    ) -> str | None:
        """Map a semantic type to an analysis family."""

        numeric_types = {
            SemanticType.CONTINUOUS,
            SemanticType.DISCRETE,
            SemanticType.CURRENCY,
            SemanticType.PERCENTAGE,
        }

        categorical_types = {
            SemanticType.NOMINAL,
            SemanticType.ORDINAL,
        }

        text_types = {
            SemanticType.SHORT_TEXT,
            SemanticType.FREE_TEXT,
        }

        if semantic_type in numeric_types:
            return "numeric"

        if semantic_type in categorical_types:
            return "categorical"

        if semantic_type in text_types:
            return "text"

        if semantic_type == SemanticType.BINARY:
            return "boolean"

        if semantic_type == SemanticType.DATETIME:
            return "datetime"

        if semantic_type == SemanticType.DURATION:
            return "timedelta"

        if semantic_type == SemanticType.IDENTIFIER:
            return "identifier"

        return None

    @staticmethod
    def _family_from_dtype(
        *,
        series: pd.Series,
    ) -> str | None:
        """Resolve an analysis family from pandas dtype."""

        dtype = series.dtype

        if is_bool_dtype(dtype):
            return "boolean"

        if is_datetime64_any_dtype(dtype):
            return "datetime"

        if is_timedelta64_dtype(dtype):
            return "timedelta"

        if is_numeric_dtype(dtype):
            return "numeric"

        if isinstance(dtype, pd.CategoricalDtype):
            return "categorical"

        if is_object_dtype(dtype):
            return None

        return None

    def _infer_object_family(
        self,
        *,
        series: pd.Series,
    ) -> str | None:
        """Resolve an object column using value statistics."""

        nonmissing = series.dropna()

        if nonmissing.empty:
            return "unknown"

        if nonmissing.map(
            lambda value: isinstance(
                value,
                bool,
            )
        ).all():
            return "boolean"

        if nonmissing.map(
            lambda value: isinstance(
                value,
                Number,
            )
            and not isinstance(value, bool)
        ).all():
            return "numeric"

        string_values = nonmissing.astype(
            "string"
        )

        unique_count = int(
            string_values.nunique(
                dropna=True
            )
        )

        unique_ratio = (
            unique_count / len(string_values)
            if len(string_values) > 0
            else 0.0
        )

        average_length = float(
            string_values.str.len().mean()
        )

        if (
            average_length
            >= self.options
            .free_text_average_length_threshold
            and unique_ratio
            >= self.options
            .free_text_unique_ratio_threshold
        ):
            return "text"

        if (
            unique_count
            <= self.options
            .categorical_unique_limit
        ):
            return "categorical"

        return "text"

    @staticmethod
    def _assign_family(
        *,
        groups: AnalysisColumnGroups,
        column: str,
        family: str,
    ) -> None:
        """Assign one column to one analysis family."""

        family_lists = {
            "numeric": groups.numeric,
            "categorical": groups.categorical,
            "datetime": groups.datetime,
            "text": groups.text,
            "boolean": groups.boolean,
            "timedelta": groups.timedelta,
            "identifier": groups.identifiers,
            "sensitive": groups.sensitive,
            "ignored": groups.ignored,
            "unknown": groups.unknown,
        }

        if family not in family_lists:
            raise ValueError(
                f"Unsupported analysis family: {family!r}"
            )

        if column not in family_lists[family]:
            family_lists[family].append(column)

    @staticmethod
    def _remove_duplicate_memberships(
        groups: AnalysisColumnGroups,
    ) -> None:
        """Ensure primary analysis families are mutually exclusive."""

        primary_families = [
            groups.numeric,
            groups.categorical,
            groups.datetime,
            groups.text,
            groups.boolean,
            groups.timedelta,
            groups.unknown,
        ]

        seen: set[str] = set()

        for columns in primary_families:
            unique_columns: list[str] = []

            for column in columns:
                if column in seen:
                    continue

                seen.add(column)
                unique_columns.append(column)

            columns[:] = unique_columns
