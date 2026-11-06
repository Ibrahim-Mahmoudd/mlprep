"""JSON and text reporting for Phase 3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlprep.phases.phase3.output_governance.result import (
    Phase3FindingSeverity,
    Phase3Result,
)
from mlprep.phases.phase3.output_governance.state import (
    Phase3State,
)


@dataclass(slots=True)
class Phase3ReportOptions:
    """Options controlling Phase 3 reports."""

    include_info_findings: bool = True
    include_state: bool = False

    include_column_profiles: bool = True
    include_relationship_details: bool = True
    include_diagnostic_details: bool = True
    include_plot_index: bool = True

    indent: int = 2

    def __post_init__(self) -> None:
        """Validate report options."""

        if self.indent < 0:
            raise ValueError(
                "indent cannot be negative."
            )


class Phase3ReportBuilder:
    """Build JSON and text Phase 3 reports."""

    def __init__(
        self,
        options: Phase3ReportOptions | None = None,
    ) -> None:
        """Initialize the report builder."""

        self.options = (
            options or Phase3ReportOptions()
        )

    def build_dictionary(
        self,
        *,
        result: Phase3Result,
        state: Phase3State,
    ) -> dict[str, Any]:
        """Build the complete report dictionary."""

        report = result.to_report_dict()

        if not self.options.include_info_findings:
            report["findings"] = [
                finding
                for finding in report["findings"]
                if finding["severity"]
                != Phase3FindingSeverity.INFO.value
            ]

        if not self.options.include_column_profiles:
            report.pop(
                "numeric_profiles",
                None,
            )

            report.pop(
                "categorical_profiles",
                None,
            )

            report.pop(
                "datetime_profiles",
                None,
            )

            report.pop(
                "text_profiles",
                None,
            )

        if not self.options.include_relationship_details:
            report.pop(
                "relationships",
                None,
            )

        if not self.options.include_diagnostic_details:
            report.pop(
                "diagnostics",
                None,
            )

        if not self.options.include_plot_index:
            report.pop(
                "plots",
                None,
            )

        report["fingerprints"] = {
            "training_before": (
                state.training_fingerprint_before
            ),
            "training_after": (
                state.training_fingerprint_after
            ),
            "validation_before": (
                state.validation_fingerprint_before
            ),
            "validation_after": (
                state.validation_fingerprint_after
            ),
            "phase2_split": (
                state.phase2_split_fingerprint
            ),
            "configuration": (
                state.configuration_fingerprint
            ),
            "analysis": (
                state.analysis_fingerprint
            ),
        }

        report["governance"] = {
            "training_data_unchanged": (
                state.training_data_unchanged
            ),
            "validation_data_unchanged": (
                state.validation_data_unchanged
            ),
            "test_partition_remained_locked": (
                state.test_partition_remained_locked
            ),
            "test_access_count_before": (
                state.test_partition_access_count_before
            ),
            "test_access_count_after": (
                state.test_partition_access_count_after
            ),
            "completed_components": list(
                state.completed_components
            ),
            "failed_components": list(
                state.failed_components
            ),
        }

        if self.options.include_state:
            report["phase_state"] = state.model_dump(
                mode="json"
            )

        return report

    def build_json(
        self,
        *,
        result: Phase3Result,
        state: Phase3State,
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
        result: Phase3Result,
        state: Phase3State,
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

        dataset_profile = result.dataset_profile

        lines = [
            "Phase 3 Report",
            "==============",
            f"Status: {status}",
            f"Ready for Phase 4: {ready}",
            "",
            "Dataset profile",
            "---------------",
        ]

        if dataset_profile is not None:
            lines.extend(
                [
                    (
                        "Training rows: "
                        f"{dataset_profile.row_count}"
                    ),
                    (
                        "Features: "
                        f"{dataset_profile.feature_count}"
                    ),
                    (
                        "Numeric features: "
                        f"{dataset_profile.numeric_feature_count}"
                    ),
                    (
                        "Categorical features: "
                        f"{dataset_profile.categorical_feature_count}"
                    ),
                    (
                        "Missing cells: "
                        f"{dataset_profile.missing_cell_count}"
                    ),
                ]
            )

        lines.extend(
            [
                "",
                "Governance",
                "----------",
                (
                    "Training unchanged: "
                    f"{state.training_data_unchanged}"
                ),
                (
                    "Validation unchanged: "
                    f"{state.validation_data_unchanged}"
                ),
                (
                    "Test remained locked: "
                    f"{state.test_partition_remained_locked}"
                ),
                "",
                "Findings",
                "--------",
            ]
        )

        for severity, count in (
            result.finding_counts_by_severity().items()
        ):
            lines.append(
                f"{severity}: {count}"
            )

        transformation_count = 0
        imputation_count = 0
        outlier_count = 0
        encoding_count = 0
        interaction_count = 0

        if result.recommendations is not None:
            recommendation_metadata = (
                result.recommendations.metadata
            )

            transformation_count = (
                recommendation_metadata.get(
                    "transformation_recommendation_count",
                    0,
                )
            )

            imputation_count = (
                recommendation_metadata.get(
                    "imputation_recommendation_count",
                    0,
                )
            )

            outlier_count = (
                recommendation_metadata.get(
                    "outlier_recommendation_count",
                    0,
                )
            )

            encoding_count = (
                recommendation_metadata.get(
                    "encoding_recommendation_count",
                    0,
                )
            )

            interaction_count = (
                recommendation_metadata.get(
                    "interaction_recommendation_count",
                    0,
                )
            )

        lines.extend(
            [
                "",
                "Recommendations",
                "---------------",
                (
                    "Transformations: "
                    f"{transformation_count}"
                ),
                (
                    "Imputation: "
                    f"{imputation_count}"
                ),
                (
                    "Outliers: "
                    f"{outlier_count}"
                ),
                (
                    "Encoding: "
                    f"{encoding_count}"
                ),
                (
                    "Interactions: "
                    f"{interaction_count}"
                ),
            ]
        )

        blocking = result.get_blocking_findings()

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

        return "\n".join(lines)

    def save_json(
        self,
        *,
        path: str | Path,
        result: Phase3Result,
        state: Phase3State,
    ) -> Path:
        """Save the JSON report."""

        output_path = Path(path)

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
        result: Phase3Result,
        state: Phase3State,
    ) -> Path:
        """Save the text report."""

        output_path = Path(path)

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
