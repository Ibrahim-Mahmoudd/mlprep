"""JSON and text reporting for Phase 7."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlprep.phases.phase7.output_governance.metadata import (
    Phase7FindingSeverity,
)
from mlprep.phases.phase7.output_governance.result import (
    Phase7Result,
)
from mlprep.phases.phase7.output_governance.state import (
    Phase7State,
)


@dataclass(slots=True)
class Phase7ReportOptions:
    """Options controlling Phase 7 reports."""

    include_info_findings: bool = True
    include_phase_state: bool = False

    include_validation_details: bool = True
    include_execution_metadata: bool = True

    include_feature_decisions: bool = True
    include_filter_details: bool = True
    include_selector_details: bool = True
    include_stability_details: bool = True

    include_feature_lineage: bool = True
    include_fingerprints: bool = True

    expose_selector_scores: bool = True
    expose_bootstrap_selections: bool = False

    indent: int = 2

    def __post_init__(self) -> None:
        """Validate report options."""

        if self.indent < 0:
            raise ValueError(
                "indent cannot be negative."
            )


class Phase7ReportBuilder:
    """Build JSON and text reports for Phase 7."""

    def __init__(
        self,
        options: Phase7ReportOptions | None = None,
    ) -> None:
        """Initialize the report builder."""

        self.options = (
            options
            or Phase7ReportOptions()
        )

    def build_dictionary(
        self,
        *,
        result: Phase7Result,
        state: Phase7State,
    ) -> dict[str, Any]:
        """Build the complete report dictionary."""

        report = result.to_report_dict()

        if not self.options.include_info_findings:
            report["findings"] = [
                finding
                for finding in report["findings"]
                if finding["severity"]
                != Phase7FindingSeverity.INFO.value
            ]

        if not self.options.include_validation_details:
            report.pop(
                "validation",
                None,
            )

        if not self.options.include_execution_metadata:
            report.pop(
                "execution_metadata",
                None,
            )

        report["governance"] = {
            "fitted_on_training_only": (
                state.fitted_on_training_only
            ),
            "supervised_target_access_used": (
                state.supervised_target_access_used
            ),
            "supervised_target_access_authorized": (
                state
                .supervised_target_access_authorized
            ),
            "validation_used_for_evaluation_only": (
                state
                .validation_used_for_evaluation_only
            ),
            "test_transform_authorized": (
                state.test_transform_authorized
            ),
            "test_partition_remained_protected": (
                state
                .test_partition_remained_protected
            ),
            "targets_unchanged": (
                state.targets_unchanged
            ),
            "row_ids_unchanged": (
                state.row_ids_unchanged
            ),
            "selected_values_unchanged": (
                state.selected_values_unchanged
            ),
            "schemas_consistent": (
                state.schemas_consistent
            ),
            "feature_count_valid": (
                state.feature_count_valid
            ),
            "protected_features_valid": (
                state.protected_features_valid
            ),
            "stability_valid": (
                state.stability_valid
            ),
        }

        report["selection_summary"] = {
            "source_feature_count": len(
                state.source_feature_columns
            ),
            "selected_feature_count": len(
                state.selected_feature_columns
            ),
            "removed_feature_count": len(
                state.removed_feature_columns
            ),
            "required_feature_count": len(
                state.required_features
            ),
            "protected_feature_count": len(
                state.protected_features
            ),
            "prohibited_feature_count": len(
                state.prohibited_features
            ),
            "selected_features": list(
                state.selected_feature_columns
            ),
            "removed_features": list(
                state.removed_feature_columns
            ),
        }

        if self.options.include_feature_decisions:
            report["feature_decisions"] = {
                column: decision.model_dump(
                    mode="json"
                )
                for column, decision
                in state.decisions.items()
            }

        if (
            self.options.include_filter_details
            and state.filter_result is not None
        ):
            report["filter_result"] = (
                state.filter_result.model_dump(
                    mode="json"
                )
            )

        if (
            self.options.include_selector_details
            and state.selector_result is not None
        ):
            selector_result = (
                state.selector_result.model_dump(
                    mode="json"
                )
            )

            if not self.options.expose_selector_scores:
                for selector_state in (
                    selector_result
                    .get(
                        "selector_states",
                        {},
                    )
                    .values()
                ):
                    selector_state[
                        "feature_scores"
                    ] = "<redacted>"

                selector_result[
                    "aggregated_scores"
                ] = "<redacted>"

            report["selector_result"] = (
                selector_result
            )

        if (
            self.options.include_stability_details
            and state.stability_result is not None
        ):
            stability_result = (
                state.stability_result.model_dump(
                    mode="json"
                )
            )

            if not self.options.expose_bootstrap_selections:
                selection = stability_result.get(
                    "selection"
                )

                if selection is not None:
                    selection[
                        "bootstrap_selected_features"
                    ] = "<redacted>"

            report["stability_result"] = (
                stability_result
            )

        if self.options.include_feature_lineage:
            report["selected_lineage"] = (
                state.selected_lineage.model_dump(
                    mode="json"
                )
                if state.selected_lineage is not None
                else None
            )

        if self.options.include_fingerprints:
            report["fingerprints"] = {
                "phase6_output": (
                    state.phase6_output_fingerprint
                ),
                "inputs": {
                    partition.value: fingerprint
                    for partition, fingerprint
                    in state
                    .input_partition_fingerprints
                    .items()
                },
                "outputs": {
                    partition.value: fingerprint
                    for partition, fingerprint
                    in state
                    .output_partition_fingerprints
                    .items()
                },
                "configuration": (
                    state.configuration_fingerprint
                ),
                "selection_plan": (
                    state.selection_plan_fingerprint
                ),
                "fitted_state": (
                    state.fitted_state_fingerprint
                ),
                "combined_output": (
                    state.output_fingerprint
                ),
            }

        if self.options.include_phase_state:
            report["phase_state"] = (
                state.model_dump(
                    mode="json"
                )
            )

        return report

    def build_json(
        self,
        *,
        result: Phase7Result,
        state: Phase7State,
    ) -> str:
        """Build a JSON report."""

        return json.dumps(
            self.build_dictionary(
                result=result,
                state=state,
            ),
            ensure_ascii=False,
            indent=self.options.indent,
            default=str,
        )

    def build_text(
        self,
        *,
        result: Phase7Result,
        state: Phase7State,
    ) -> str:
        """Build a readable text report."""

        readiness = result.readiness

        status = (
            readiness.status.value
            if readiness is not None
            else "not_evaluated"
        )

        ready = (
            readiness.ready
            if readiness is not None
            else False
        )

        lines = [
            "Phase 7 Report",
            "==============",
            f"Status: {status}",
            f"Ready for Phase 8: {ready}",
            "",
            "Feature selection",
            "-----------------",
            (
                "Source features: "
                f"{len(state.source_feature_columns)}"
            ),
            (
                "Selected features: "
                f"{len(state.selected_feature_columns)}"
            ),
            (
                "Removed features: "
                f"{len(state.removed_feature_columns)}"
            ),
            (
                "Required features: "
                f"{len(state.required_features)}"
            ),
            (
                "Protected features: "
                f"{len(state.protected_features)}"
            ),
            (
                "Prohibited features: "
                f"{len(state.prohibited_features)}"
            ),
            "",
            "Governance",
            "----------",
            (
                "Fitted on training only: "
                f"{state.fitted_on_training_only}"
            ),
            (
                "Supervised target access used: "
                f"{state.supervised_target_access_used}"
            ),
            (
                "Supervised target access authorized: "
                f"{state.supervised_target_access_authorized}"
            ),
            (
                "Validation used for evaluation only: "
                f"{state.validation_used_for_evaluation_only}"
            ),
            (
                "Selected values unchanged: "
                f"{state.selected_values_unchanged}"
            ),
            (
                "Targets unchanged: "
                f"{state.targets_unchanged}"
            ),
            (
                "Row IDs unchanged: "
                f"{state.row_ids_unchanged}"
            ),
            (
                "Schemas consistent: "
                f"{state.schemas_consistent}"
            ),
            "",
            "Findings",
            "--------",
        ]

        for severity, count in (
            result.finding_counts().items()
        ):
            lines.append(
                f"{severity}: {count}"
            )

        blocking = result.blocking_findings()

        if blocking:
            lines.extend(
                [
                    "",
                    "Blocking findings",
                    "-----------------",
                ]
            )

            for finding in blocking:
                lines.append(
                    f"- [{finding.code}] "
                    f"{finding.message}"
                )

        lines.extend(
            [
                "",
                "Selected features",
                "-----------------",
            ]
        )

        lines.extend(
            f"- {feature}"
            for feature in state.selected_feature_columns
        )

        return "\n".join(
            lines
        )

    def save_json(
        self,
        *,
        path: str | Path,
        result: Phase7Result,
        state: Phase7State,
    ) -> Path:
        """Save the JSON report."""

        output_path = Path(
            path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.build_json(
                result=result,
                state=state,
            ),
            encoding="utf-8",
        )

        return output_path

    def save_text(
        self,
        *,
        path: str | Path,
        result: Phase7Result,
        state: Phase7State,
    ) -> Path:
        """Save the text report."""

        output_path = Path(
            path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.build_text(
                result=result,
                state=state,
            ),
            encoding="utf-8",
        )

        return output_path