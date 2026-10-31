"""Deterministic column-name normalization for Phase 1."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    NormalizationEvent,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    ColumnNameState,
    Phase1State,
)
from mlprep.types import (
    CollisionPolicy,
    FindingAction,
    FindingCategory,
    Severity,
)

ColumnCaseStyle = Literal[
    "preserve",
    "lower",
    "upper",
    "snake_case",
]


@dataclass(slots=True)
class ColumnNameOptions:
    """Options controlling deterministic column-name normalization."""

    enabled: bool = True

    strip_whitespace: bool = True
    normalize_unicode: bool = True

    case_style: ColumnCaseStyle = "snake_case"

    whitespace_replacement: str = "_"

    remove_special_characters: bool = True
    allowed_special_characters: str = "_"

    collapse_repeated_separators: bool = True
    remove_leading_separators: bool = True
    remove_trailing_separators: bool = True

    maximum_length: int = 128

    collision_policy: CollisionPolicy = (
        CollisionPolicy.RAISE
    )

    explicit_mapping: dict[str, str] = field(
        default_factory=dict
    )

    reserved_names: set[str] = field(
        default_factory=lambda: {
            "__mlprep_row_id__",
            "__mlprep_split__",
            "__mlprep_status__",
        }
    )

    def __post_init__(self) -> None:
        """Validate column-name options."""

        if self.maximum_length <= 0:
            raise ValueError(
                "maximum_length must be greater than zero."
            )

        if len(self.whitespace_replacement) > 5:
            raise ValueError(
                "whitespace_replacement must contain at most "
                "five characters."
            )

        if (
            self.collision_policy
            == CollisionPolicy.USE_EXPLICIT_MAPPING
            and not self.explicit_mapping
        ):
            raise ValueError(
                "explicit_mapping is required when the collision "
                "policy is use_explicit_mapping."
            )


class ColumnNameNormalizer:
    """Normalize DataFrame column names deterministically."""

    component_name = "column_names"

    def __init__(
        self,
        options: ColumnNameOptions | None = None,
    ) -> None:
        """Initialize the column-name normalizer."""

        self.options = options or ColumnNameOptions()

    def normalize(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> pd.DataFrame:
        """Normalize column names and register name mappings."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Column-name normalization requires a "
                "pandas DataFrame."
            )

        working_data = data.copy()

        original_columns = [
            str(column)
            for column in working_data.columns
        ]

        state.original_column_order = list(
            original_columns
        )

        if not self.options.enabled:
            self._register_unchanged_names(
                columns=original_columns,
                state=state,
            )

            state.canonical_column_order = list(
                original_columns
            )

            return working_data

        candidates: list[str] = []
        reasons_by_position: list[list[str]] = []

        for original_name in original_columns:
            candidate, reasons = self._normalize_name(
                original_name
            )

            candidates.append(candidate)
            reasons_by_position.append(reasons)

        resolved_names = self._resolve_collisions(
            original_columns=original_columns,
            candidates=candidates,
            result=result,
        )

        self._validate_empty_names(
            original_columns=original_columns,
            normalized_columns=resolved_names,
            result=result,
        )

        self._validate_reserved_names(
            original_columns=original_columns,
            normalized_columns=resolved_names,
            result=result,
        )

        working_data.columns = resolved_names

        for position, original_name in enumerate(
            original_columns
        ):
            candidate = candidates[position]
            normalized_name = resolved_names[position]

            reasons = list(
                reasons_by_position[position]
            )

            suffix_added = self._extract_suffix(
                candidate=candidate,
                final_name=normalized_name,
            )

            if suffix_added is not None:
                reasons.append(
                    "A numeric suffix was added to resolve "
                    "a naming collision."
                )

            existing_original = (
                state.normalized_to_original_names.get(
                    normalized_name
                )
            )
            if (
                existing_original is None
                or existing_original == original_name
            ):
                state.register_column_name(
                    ColumnNameState(
                        original_name=original_name,
                        normalized_name=normalized_name,
                        was_changed=(
                            original_name != normalized_name
                        ),
                        change_reasons=reasons,
                        collision_group=(
                            candidate
                            if candidates.count(candidate) > 1
                            else None
                        ),
                        suffix_added=suffix_added,
                    )
                )

            if original_name == normalized_name:
                continue

            result.add_normalization_event(
                NormalizationEvent(
                    component=self.component_name,
                    operation="rename_column",
                    description=(
                        f"Renamed column {original_name!r} "
                        f"to {normalized_name!r}."
                    ),
                    column=normalized_name,
                    affected_row_count=len(working_data),
                    before_summary={
                        "column_name": original_name,
                    },
                    after_summary={
                        "column_name": normalized_name,
                    },
                    reversible=True,
                    metadata={
                        "change_reasons": reasons,
                    },
                )
            )

        state.canonical_column_order = list(
            resolved_names
        )

        renamed_count = sum(
            original != normalized
            for original, normalized in zip(
                original_columns,
                resolved_names,
                strict=True,
            )
        )

        if renamed_count > 0:
            result.add_finding(
                Finding(
                    code="P1-COLNAME-001",
                    category=FindingCategory.COLUMN_NAME,
                    severity=Severity.INFO,
                    component=self.component_name,
                    message=(
                        f"{renamed_count} column name or names "
                        "were normalized."
                    ),
                    columns=[
                        normalized
                        for original, normalized in zip(
                            original_columns,
                            resolved_names,
                            strict=True,
                        )
                        if original != normalized
                    ],
                    recommended_action=FindingAction.RENAME,
                    applied_action=FindingAction.RENAME,
                    blocks_readiness=False,
                )
            )

        return working_data

    def normalize_reference(
        self,
        column: str,
        state: Phase1State,
    ) -> str:
        """Normalize one configured column reference."""

        return state.original_to_normalized_names.get(
            column,
            column,
        )

    def normalize_references(
        self,
        columns: list[str],
        state: Phase1State,
    ) -> list[object]:
        """Normalize multiple configured column references."""

        return [
            self.normalize_reference(
                column=column,
                state=state,
            )
            for column in columns
        ]

    def _normalize_name(
        self,
        original_name: str,
    ) -> tuple[str, list[str]]:
        """Normalize one name and return change reasons."""

        value = self.options.explicit_mapping.get(
            original_name,
            original_name,
        )

        reasons: list[str] = []

        if value != original_name:
            reasons.append(
                "An explicit name mapping was applied."
            )

        if self.options.normalize_unicode:
            normalized_unicode = unicodedata.normalize(
                "NFKC",
                value,
            )

            if normalized_unicode != value:
                reasons.append(
                    "Unicode characters were normalized."
                )

            value = normalized_unicode

        if self.options.strip_whitespace:
            stripped_value = value.strip()

            if stripped_value != value:
                reasons.append(
                    "Leading or trailing whitespace was removed."
                )

            value = stripped_value

        converted_case = self._apply_case_style(value)

        if converted_case != value:
            reasons.append(
                f"The name was converted using "
                f"{self.options.case_style} style."
            )

        value = converted_case

        whitespace_replaced = re.sub(
            r"\s+",
            self.options.whitespace_replacement,
            value,
        )

        if whitespace_replaced != value:
            reasons.append(
                "Whitespace characters were replaced."
            )

        value = whitespace_replaced

        if self.options.remove_special_characters:
            cleaned_value = self._remove_special_characters(
                value
            )

            if cleaned_value != value:
                reasons.append(
                    "Unsupported special characters were removed."
                )

            value = cleaned_value

        separator = self.options.whitespace_replacement

        if (
            self.options.collapse_repeated_separators
            and separator
        ):
            collapsed_value = re.sub(
                rf"(?:{re.escape(separator)})+",
                separator,
                value,
            )

            if collapsed_value != value:
                reasons.append(
                    "Repeated separators were collapsed."
                )

            value = collapsed_value

        if (
            self.options.remove_leading_separators
            and separator
        ):
            stripped_leading = value.lstrip(separator)

            if stripped_leading != value:
                reasons.append(
                    "Leading separators were removed."
                )

            value = stripped_leading

        if (
            self.options.remove_trailing_separators
            and separator
        ):
            stripped_trailing = value.rstrip(separator)

            if stripped_trailing != value:
                reasons.append(
                    "Trailing separators were removed."
                )

            value = stripped_trailing

        if len(value) > self.options.maximum_length:
            value = value[: self.options.maximum_length]

            reasons.append(
                "The name was truncated to the maximum length."
            )

        return value, reasons

    def _apply_case_style(
        self,
        value: str,
    ) -> str:
        """Apply the configured case style."""

        if self.options.case_style == "preserve":
            return value

        if self.options.case_style == "lower":
            return value.lower()

        if self.options.case_style == "upper":
            return value.upper()

        return self._to_snake_case(value)

    @staticmethod
    def _to_snake_case(
        value: str,
    ) -> str:
        """Convert a name to snake_case."""

        value = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1_\2",
            value,
        )

        value = re.sub(
            r"([A-Z]+)([A-Z][a-z])",
            r"\1_\2",
            value,
        )

        value = re.sub(
            r"[\s\-]+",
            "_",
            value,
        )

        return value.lower()

    def _remove_special_characters(
        self,
        value: str,
    ) -> str:
        """Remove unsupported characters from a name."""

        allowed = re.escape(
            self.options.allowed_special_characters
        )

        return re.sub(
            rf"[^0-9A-Za-z{allowed}]",
            "",
            value,
        )

    def _resolve_collisions(
        self,
        *,
        original_columns: list[str],
        candidates: list[str],
        result: Phase1Result,
    ) -> list[str]:
        """Resolve duplicate names created by normalization."""

        counts = Counter(candidates)

        collisions = {
            candidate
            for candidate, count in counts.items()
            if count > 1
        }

        if not collisions:
            return list(candidates)

        details = {
            candidate: [
                original_columns[position]
                for position, name in enumerate(candidates)
                if name == candidate
            ]
            for candidate in sorted(collisions)
        }

        if (
            self.options.collision_policy
            != CollisionPolicy.ADD_SUFFIX
        ):
            result.add_finding(
                Finding(
                    code="P1-COLNAME-002",
                    category=FindingCategory.COLUMN_NAME,
                    severity=Severity.CRITICAL,
                    component=self.component_name,
                    message=(
                        "Column-name normalization produced "
                        "duplicate names."
                    ),
                    columns=sorted(collisions),
                    recommended_action=FindingAction.RAISE,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=True,
                    details={
                        "collisions": details,
                    },
                )
            )

            return list(candidates)

        resolved: list[str] = []
        used_names: set[str] = set()
        next_suffix: dict[str, int] = {}

        for candidate in candidates:
            if candidate not in used_names:
                resolved.append(candidate)
                used_names.add(candidate)
                next_suffix[candidate] = 2
                continue

            suffix = next_suffix.get(candidate, 2)
            resolved_candidate = (
                f"{candidate}_{suffix}"
            )

            while resolved_candidate in used_names:
                suffix += 1
                resolved_candidate = (
                    f"{candidate}_{suffix}"
                )

            resolved.append(resolved_candidate)
            used_names.add(resolved_candidate)
            next_suffix[candidate] = suffix + 1

        result.add_finding(
            Finding(
                code="P1-COLNAME-003",
                category=FindingCategory.COLUMN_NAME,
                severity=Severity.WARNING,
                component=self.component_name,
                message=(
                    "Column-name collisions were resolved using "
                    "numeric suffixes."
                ),
                columns=sorted(collisions),
                recommended_action=FindingAction.RENAME,
                applied_action=FindingAction.RENAME,
                blocks_readiness=False,
                details={
                    "collisions": details,
                    "resolved_names": resolved,
                },
            )
        )

        return resolved

    def _validate_empty_names(
        self,
        *,
        original_columns: list[str],
        normalized_columns: list[str],
        result: Phase1Result,
    ) -> None:
        """Detect names that became empty after normalization."""

        empty_positions = [
            position
            for position, name in enumerate(
                normalized_columns
            )
            if not name
        ]

        if not empty_positions:
            return

        result.add_finding(
            Finding(
                code="P1-COLNAME-004",
                category=FindingCategory.COLUMN_NAME,
                severity=Severity.CRITICAL,
                component=self.component_name,
                message=(
                    "One or more column names became empty "
                    "after normalization."
                ),
                columns=[
                    original_columns[position]
                    for position in empty_positions
                ],
                recommended_action=FindingAction.RENAME,
                applied_action=FindingAction.REPORT,
                blocks_readiness=True,
                details={
                    "positions": empty_positions,
                },
            )
        )

    def _validate_reserved_names(
        self,
        *,
        original_columns: list[str],
        normalized_columns: list[str],
        result: Phase1Result,
    ) -> None:
        """Detect conflicts with reserved internal names."""

        conflicting_positions = [
            position
            for position, name in enumerate(
                normalized_columns
            )
            if (
                name in self.options.reserved_names
                or original_columns[position]
                in self.options.reserved_names
            )
        ]

        if not conflicting_positions:
            return

        result.add_finding(
            Finding(
                code="P1-COLNAME-005",
                category=FindingCategory.COLUMN_NAME,
                severity=Severity.CRITICAL,
                component=self.component_name,
                message=(
                    "Input columns conflict with reserved "
                    "MLPrep names."
                ),
                columns=[
                    normalized_columns[position]
                    for position in conflicting_positions
                ],
                recommended_action=FindingAction.RENAME,
                applied_action=FindingAction.REPORT,
                blocks_readiness=True,
                details={
                    "original_names": [
                        original_columns[position]
                        for position in conflicting_positions
                    ],
                },
            )
        )

    @staticmethod
    def _register_unchanged_names(
        *,
        columns: list[str],
        state: Phase1State,
    ) -> None:
        """Register unchanged column-name mappings."""

        for column in columns:
            state.register_column_name(
                ColumnNameState(
                    original_name=column,
                    normalized_name=column,
                    was_changed=False,
                )
            )

    @staticmethod
    def _extract_suffix(
        *,
        candidate: str,
        final_name: str,
    ) -> str | None:
        """Extract an automatically added collision suffix."""

        if candidate == final_name:
            return None

        expected_prefix = f"{candidate}_"

        if not final_name.startswith(expected_prefix):
            return None

        return final_name[len(candidate):]
