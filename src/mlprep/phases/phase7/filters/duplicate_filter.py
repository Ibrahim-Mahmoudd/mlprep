"""Duplicate-feature filtering for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DuplicateGroupResult(BaseModel):
    """One group of identical training features."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    retained_feature: str
    duplicate_features: list[str] = Field(default_factory=list)
    all_features: list[str] = Field(default_factory=list)
    retention_reason: str
    required_features: list[str] = Field(default_factory=list)
    protected_features: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DuplicateFilterResult(BaseModel):
    """Complete duplicate-feature filter output."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    groups: list[DuplicateGroupResult] = Field(default_factory=list)
    removal_candidates: list[str] = Field(default_factory=list)
    retained_features: list[str] = Field(default_factory=list)
    overridden_features: list[str] = Field(default_factory=list)
    fitted_on_training_only: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class DuplicateFilterOptions:
    """Options controlling duplicate-feature filtering."""

    enabled: bool = True
    verify_dtypes: bool = False
    verify_exact_values: bool = True


class DuplicateFilter:
    """Detect identical columns in training data."""

    component_name = "duplicate_filter"

    def __init__(
        self,
        options: DuplicateFilterOptions | None = None,
    ) -> None:
        """Initialize the filter."""

        self.options = options or DuplicateFilterOptions()
        self.result: DuplicateFilterResult | None = None

    def fit(
        self,
        *,
        frame: pd.DataFrame,
        candidate_features: list[str] | None = None,
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
        source_feature_order: list[str] | None = None,
    ) -> DuplicateFilter:
        """Find duplicate columns without changing the source frame."""

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame.")
        candidates = list(
            candidate_features
            if candidate_features is not None
            else frame.columns.map(str)
        )
        missing = [column for column in candidates if column not in frame.columns]
        if missing:
            raise KeyError(f"Candidate columns are missing: {missing}")

        required = set(required_features or [])
        protected = set(protected_features or [])
        order = source_feature_order or list(frame.columns.map(str))
        order_index = {column: index for index, column in enumerate(order)}
        candidates.sort(key=lambda column: order_index.get(column, len(order)))

        if not self.options.enabled or not self.options.verify_exact_values:
            self.result = DuplicateFilterResult(
                retained_features=candidates,
                metadata={"enabled": self.options.enabled},
            )
            return self

        groups: list[list[str]] = []
        for column in candidates:
            for group in groups:
                first = group[0]
                same_dtype = (
                    not self.options.verify_dtypes
                    or frame[first].dtype == frame[column].dtype
                )
                if same_dtype and frame[first].equals(frame[column]):
                    group.append(column)
                    break
            else:
                groups.append([column])

        group_results: list[DuplicateGroupResult] = []
        removal_candidates: list[str] = []
        overridden: list[str] = []
        for group in groups:
            if len(group) < 2:
                continue
            mandatory = [column for column in group if column in required]
            protected_group = [column for column in group if column in protected]
            priority = mandatory or protected_group or group
            retained = priority[0]
            removals = [
                column
                for column in group
                if column != retained
                and column not in required
                and column not in protected
            ]
            overridden.extend(
                column
                for column in group
                if column != retained and column in (required | protected)
            )
            removal_candidates.extend(removals)
            group_results.append(
                DuplicateGroupResult(
                    retained_feature=retained,
                    duplicate_features=[
                        column for column in group if column != retained
                    ],
                    all_features=group,
                    retention_reason=(
                        "required"
                        if retained in required
                        else "protected"
                        if retained in protected
                        else "source_order"
                    ),
                    required_features=[
                        column for column in group if column in required
                    ],
                    protected_features=[
                        column for column in group if column in protected
                    ],
                )
            )

        removed = set(removal_candidates)
        self.result = DuplicateFilterResult(
            groups=group_results,
            removal_candidates=removal_candidates,
            retained_features=[
                column for column in candidates if column not in removed
            ],
            overridden_features=list(dict.fromkeys(overridden)),
            fitted_on_training_only=True,
        )
        return self

    def get_result(self) -> DuplicateFilterResult:
        """Return the fitted duplicate-filter result."""

        if self.result is None:
            raise RuntimeError("DuplicateFilter.fit() must be called first.")
        return self.result.model_copy(deep=True)
