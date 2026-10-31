"""Pre-split preprocessing leakage detection for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field

from mlprep.phases.phase2.contract.split_contract import SplitContract
from mlprep.phases.phase2.output_governance.metadata import (
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
)
from mlprep.phases.phase2.output_governance.result import SplitFinding


@dataclass(slots=True)
class PreprocessingLeakageOptions:
    """Options controlling pre-split preprocessing checks."""

    enabled: bool = True

    allowed_phase1_operations: set[str] = field(
        default_factory=lambda: {
            "rename_column",
            "normalize_text_values",
            "normalize_missing_tokens",
            "create_internal_row_id",
            "replace_internal_row_id",
            "downcast_dtype",
        }
    )

    forbidden_operation_terms: set[str] = field(
        default_factory=lambda: {
            "impute",
            "imputation",
            "scale",
            "scaling",
            "standardize",
            "standardization",
            "normalize_numeric",
            "winsorize",
            "clip_outlier",
            "target_encode",
            "target_encoding",
            "one_hot",
            "onehot",
            "pca",
            "feature_select",
            "feature_selection",
            "variance_threshold",
            "mutual_information",
        }
    )

    block_forbidden_operations: bool = True
    report_unknown_operations: bool = True

    suspicious_state_keys: set[str] = field(
        default_factory=lambda: {
            "imputation_values",
            "scaler_statistics",
            "feature_selection_scores",
            "target_encoding_maps",
            "outlier_thresholds",
            "pca_components",
            "learned_means",
            "learned_medians",
        }
    )


class PreprocessingLeakageChecker:
    """Detect fitted preprocessing performed before Phase 2."""

    component_name = "preprocessing_leakage"

    def __init__(
        self,
        options: PreprocessingLeakageOptions | None = None,
    ) -> None:
        """Initialize the preprocessing leakage checker."""

        self.options = (
            options or PreprocessingLeakageOptions()
        )

    def inspect(
        self,
        contract: SplitContract,
    ) -> list[object]:
        """Inspect Phase 1 logs and state for fitted preprocessing."""

        if not self.options.enabled:
            return []

        findings: list[SplitFinding] = []

        for event in (
            contract.phase1_result
            .normalization_events
        ):
            normalized_operation = (
                event.operation
                .strip()
                .casefold()
            )

            matched_forbidden_terms = [
                term
                for term
                in self.options
                .forbidden_operation_terms
                if term.casefold()
                in normalized_operation
            ]

            if matched_forbidden_terms:
                findings.append(
                    SplitFinding(
                        code="P2-PREPROCESSING-001",
                        category=(
                            SplitFindingCategory
                            .PREPROCESSING_LEAKAGE
                        ),
                        severity=SplitFindingSeverity.CRITICAL,
                        component=self.component_name,
                        message=(
                            f"Phase 1 operation "
                            f"{event.operation!r} appears to "
                            "contain fitted preprocessing that "
                            "should occur after splitting."
                        ),
                        columns=(
                            [event.column]
                            if event.column is not None
                            else []
                        ),
                        affected_row_count=(
                            event.affected_row_count
                        ),
                        recommended_action=(
                            SplitFindingAction.RAISE
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=(
                            self.options
                            .block_forbidden_operations
                        ),
                        details={
                            "operation": event.operation,
                            "matched_terms": (
                                matched_forbidden_terms
                            ),
                        },
                    )
                )

                continue

            if (
                self.options.report_unknown_operations
                and event.operation
                not in self.options
                .allowed_phase1_operations
            ):
                findings.append(
                    SplitFinding(
                        code="P2-PREPROCESSING-002",
                        category=(
                            SplitFindingCategory
                            .PREPROCESSING_LEAKAGE
                        ),
                        severity=SplitFindingSeverity.WARNING,
                        component=self.component_name,
                        message=(
                            f"Phase 1 operation "
                            f"{event.operation!r} is not in the "
                            "approved deterministic-operation list."
                        ),
                        columns=(
                            [event.column]
                            if event.column is not None
                            else []
                        ),
                        recommended_action=(
                            SplitFindingAction.WARN
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=False,
                        details={
                            "operation": event.operation,
                        },
                    )
                )

        serialized_state = (
            contract.phase1_state.model_dump(
                mode="json"
            )
        )

        matched_state_paths = (
            self._find_suspicious_state_paths(
                value=serialized_state,
                path="phase1_state",
            )
        )

        if matched_state_paths:
            findings.append(
                SplitFinding(
                    code="P2-PREPROCESSING-003",
                    category=(
                        SplitFindingCategory
                        .PREPROCESSING_LEAKAGE
                    ),
                    severity=SplitFindingSeverity.CRITICAL,
                    component=self.component_name,
                    message=(
                        "Phase 1 state contains keys associated "
                        "with fitted statistical preprocessing."
                    ),
                    recommended_action=(
                        SplitFindingAction.RAISE
                    ),
                    applied_action=SplitFindingAction.REPORT,
                    blocks_readiness=(
                        self.options
                        .block_forbidden_operations
                    ),
                    details={
                        "matched_state_paths": (
                            matched_state_paths
                        ),
                    },
                )
            )

        return findings

    def _find_suspicious_state_paths(
        self,
        *,
        value: object,
        path: str,
    ) -> list[object]:
        """Recursively find suspicious keys in serialized state."""

        matches: list[str] = []

        if isinstance(value, dict):
            for key, child_value in value.items():
                child_path = f"{path}.{key}"

                normalized_key = (
                    str(key)
                    .strip()
                    .casefold()
                )

                if normalized_key in {
                    key_name.casefold()
                    for key_name
                    in self.options.suspicious_state_keys
                }:
                    matches.append(child_path)

                matches.extend(
                    self._find_suspicious_state_paths(
                        value=child_value,
                        path=child_path,
                    )
                )

        elif isinstance(value, list):
            for index, child_value in enumerate(
                value
            ):
                matches.extend(
                    self._find_suspicious_state_paths(
                        value=child_value,
                        path=f"{path}[{index}]",
                    )
                )

        return matches