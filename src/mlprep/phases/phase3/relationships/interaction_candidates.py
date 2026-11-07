"""Feature-interaction candidate discovery for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.relationships.mixed_associations import (
    MixedAssociationResult,
)
from mlprep.phases.phase3.relationships.numeric_associations import (
    NumericAssociationResult,
)
from mlprep.phases.phase3.relationships.target_associations import (
    TargetAssociationResult,
)

InteractionKind = Literal[
    "numeric_product",
    "numeric_ratio",
    "numeric_difference",
    "numeric_sum",
    "categorical_cross",
    "numeric_by_category",
]


class InteractionCandidate(BaseModel):
    """One recommended feature interaction."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str

    kind: InteractionKind

    source_columns: list[str] = Field(
        min_length=2
    )

    priority: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str

    leakage_safe: bool = True

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


class InteractionCandidateResult(BaseModel):
    """Complete interaction-candidate output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    candidates: list[InteractionCandidate] = Field(
        default_factory=list
    )

    skipped_columns: list[str] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class InteractionCandidateOptions:
    """Options controlling interaction discovery."""

    enabled: bool = True

    maximum_candidates: int = 100

    minimum_target_association: float = 0.05

    minimum_mixed_eta_squared: float = 0.05

    maximum_numeric_pair_correlation: float = 0.95

    suggest_products: bool = True
    suggest_ratios: bool = True
    suggest_differences: bool = True
    suggest_categorical_crosses: bool = True
    suggest_numeric_by_category: bool = True

    maximum_source_columns: int = 100

    def __post_init__(self) -> None:
        """Validate interaction-candidate options."""

        if self.maximum_candidates < 1:
            raise ValueError(
                "maximum_candidates must be positive."
            )

        bounded_values = {
            "minimum_target_association": (
                self.minimum_target_association
            ),
            "minimum_mixed_eta_squared": (
                self.minimum_mixed_eta_squared
            ),
            "maximum_numeric_pair_correlation": (
                self.maximum_numeric_pair_correlation
            ),
        }

        for name, value in bounded_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between zero and one."
                )


class InteractionCandidateAnalyzer:
    """Generate safe feature-interaction recommendations."""

    component_name = "interaction_candidates"

    def __init__(
        self,
        options: (
            InteractionCandidateOptions | None
        ) = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options
            or InteractionCandidateOptions()
        )

    def analyze(
        self,
        *,
        columns: AnalysisColumnGroups,
        numeric_associations: NumericAssociationResult,
        mixed_associations: MixedAssociationResult,
        target_associations: TargetAssociationResult,
    ) -> InteractionCandidateResult:
        """Generate candidate interactions from prior analyses."""

        if not self.options.enabled:
            return InteractionCandidateResult()

        candidates: list[
            InteractionCandidate
        ] = []

        target_scores = {
            association.feature: (
                association.association_score
                or 0.0
            )
            for association
            in target_associations.associations
        }

        eligible_numeric = [
            column
            for column in columns.numeric
            if (
                target_scores.get(
                    column,
                    0.0,
                )
                >= self.options
                .minimum_target_association
            )
        ][
            : self.options.maximum_source_columns
        ]

        correlation_lookup = {
            frozenset(
                [
                    pair.left_column,
                    pair.right_column,
                ]
            ): (
                pair.absolute_primary_correlation
                or 0.0
            )
            for pair
            in numeric_associations.pairs
        }

        for left_position, left_column in enumerate(
            eligible_numeric
        ):
            for right_column in eligible_numeric[
                left_position + 1:
            ]:
                correlation = (
                    correlation_lookup.get(
                        frozenset(
                            [
                                left_column,
                                right_column,
                            ]
                        ),
                        0.0,
                    )
                )

                if (
                    correlation
                    > self.options
                    .maximum_numeric_pair_correlation
                ):
                    continue

                priority = min(
                    1.0,
                    (
                        target_scores.get(
                            left_column,
                            0.0,
                        )
                        + target_scores.get(
                            right_column,
                            0.0,
                        )
                    )
                    / 2.0,
                )

                if self.options.suggest_products:
                    candidates.append(
                        self._candidate(
                            kind="numeric_product",
                            left=left_column,
                            right=right_column,
                            priority=priority,
                            reason=(
                                "Both numeric features have "
                                "measurable target association "
                                "and are not excessively correlated."
                            ),
                        )
                    )

                if self.options.suggest_differences:
                    candidates.append(
                        self._candidate(
                            kind="numeric_difference",
                            left=left_column,
                            right=right_column,
                            priority=priority,
                            reason=(
                                "A difference may capture relative "
                                "position between two informative "
                                "numeric features."
                            ),
                        )
                    )

                if self.options.suggest_ratios:
                    candidates.append(
                        self._candidate(
                            kind="numeric_ratio",
                            left=left_column,
                            right=right_column,
                            priority=priority,
                            reason=(
                                "A ratio may capture scale-relative "
                                "behavior. Phase 6 must apply "
                                "division-by-zero protection."
                            ),
                        )
                    )

        if self.options.suggest_numeric_by_category:
            for pair in mixed_associations.pairs:
                eta_squared = (
                    pair.eta_squared or 0.0
                )

                if (
                    eta_squared
                    < self.options
                    .minimum_mixed_eta_squared
                ):
                    continue

                candidates.append(
                    self._candidate(
                        kind="numeric_by_category",
                        left=pair.numeric_column,
                        right=pair.categorical_column,
                        priority=min(
                            1.0,
                            eta_squared,
                        ),
                        reason=(
                            "The numeric feature distribution "
                            "differs across categories."
                        ),
                    )
                )

        if self.options.suggest_categorical_crosses:
            categorical_columns = [
                column
                for column
                in columns.categorical
                if (
                    target_scores.get(
                        column,
                        0.0,
                    )
                    >= self.options
                    .minimum_target_association
                )
            ][
                : self.options.maximum_source_columns
            ]

            for left_position, left_column in enumerate(
                categorical_columns
            ):
                for right_column in categorical_columns[
                    left_position + 1:
                ]:
                    priority = min(
                        1.0,
                        (
                            target_scores.get(
                                left_column,
                                0.0,
                            )
                            + target_scores.get(
                                right_column,
                                0.0,
                            )
                        )
                        / 2.0,
                    )

                    candidates.append(
                        self._candidate(
                            kind="categorical_cross",
                            left=left_column,
                            right=right_column,
                            priority=priority,
                            reason=(
                                "Both categorical features show "
                                "target association and may contain "
                                "useful joint categories."
                            ),
                        )
                    )

        candidates = self._deduplicate(
            candidates
        )

        candidates.sort(
            key=lambda candidate: (
                candidate.priority
            ),
            reverse=True,
        )

        return InteractionCandidateResult(
            candidates=candidates[
                : self.options.maximum_candidates
            ],
        )

    @staticmethod
    def _candidate(
        *,
        kind: InteractionKind,
        left: str,
        right: str,
        priority: float,
        reason: str,
    ) -> InteractionCandidate:
        """Create one interaction candidate."""

        operation_names = {
            "numeric_product": "x",
            "numeric_ratio": "div",
            "numeric_difference": "minus",
            "numeric_sum": "plus",
            "categorical_cross": "cross",
            "numeric_by_category": "by",
        }

        return InteractionCandidate(
            name=(
                f"{left}__"
                f"{operation_names[kind]}__"
                f"{right}"
            ),
            kind=kind,
            source_columns=[
                left,
                right,
            ],
            priority=priority,
            reason=reason,
            leakage_safe=True,
        )

    @staticmethod
    def _deduplicate(
        candidates: list[InteractionCandidate],
    ) -> list[object]:
        """Deduplicate candidates using name and kind."""

        seen: set[
            tuple[str, str]
        ] = set()

        unique: list[
            InteractionCandidate
        ] = []

        for candidate in candidates:
            key = (
                candidate.name,
                candidate.kind,
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(candidate)

        return unique
