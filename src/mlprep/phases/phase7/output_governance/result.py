"""Result models produced by Phase 7."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from pydantic import Field

from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase7.output_governance.metadata import (
    Phase7ExecutionMetadata,
    Phase7Finding,
    Phase7FindingSeverity,
    Phase7Model,
    Phase7Status,
)
from mlprep.phases.phase7.validation.feature_count import (
    FeatureCountResult,
)
from mlprep.phases.phase7.validation.protected_features import (
    ProtectedFeaturesResult,
)
from mlprep.phases.phase7.validation.row_integrity import (
    RowIntegrityResult,
)
from mlprep.phases.phase7.validation.schema_consistency import (
    SchemaConsistencyResult,
)
from mlprep.phases.phase7.validation.target_integrity import (
    TargetIntegrityResult,
)


class Phase7ReadinessDecision(Phase7Model):
    """Decision indicating whether Phase 8 may begin."""

    ready: bool
    status: Phase7Status

    blocking_finding_ids: list[str] = Field(
        default_factory=list,
    )

    blocking_finding_codes: list[str] = Field(
        default_factory=list,
    )

    reasons: list[str] = Field(
        default_factory=list,
    )

    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class Phase7ValidationResults(Phase7Model):
    """Combined Phase 7 validation results."""

    schema_consistency: (
        SchemaConsistencyResult | None
    ) = None

    feature_count: FeatureCountResult | None = None

    protected_features: (
        ProtectedFeaturesResult | None
    ) = None

    target_integrity: (
        TargetIntegrityResult | None
    ) = None

    row_integrity: RowIntegrityResult | None = None


class Phase7Result:
    """Complete output produced by Phase 7."""

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
    ) -> None:
        """Initialize a Phase 7 result."""

        self.train = train
        self.validation = validation
        self.test = test

        self.findings: list[
            Phase7Finding
        ] = []

        self.validation_results = (
            Phase7ValidationResults()
        )

        self.execution_metadata: (
            Phase7ExecutionMetadata | None
        ) = None

        self.readiness: (
            Phase7ReadinessDecision | None
        ) = None

        self.phase_state: dict[
            str,
            Any,
        ] = {}

        self.started_at = datetime.now(UTC)
        self.completed_at: datetime | None = None

    def add_finding(
        self,
        finding: Phase7Finding,
    ) -> None:
        """Add one finding."""

        self.findings.append(
            finding
        )

    def add_findings(
        self,
        findings: list[Phase7Finding],
    ) -> None:
        """Add multiple findings."""

        self.findings.extend(
            findings
        )

    def blocking_findings(
        self,
    ) -> list[Phase7Finding]:
        """Return findings that block Phase 8."""

        return [
            finding
            for finding in self.findings
            if (
                finding.blocks_readiness
                or finding.severity
                == Phase7FindingSeverity.CRITICAL
            )
        ]

    def findings_by_severity(
        self,
        severity: Phase7FindingSeverity,
    ) -> list[Phase7Finding]:
        """Return findings matching one severity."""

        return [
            finding
            for finding in self.findings
            if finding.severity == severity
        ]

    def finding_counts(
        self,
    ) -> dict[str, int]:
        """Return finding counts by severity."""

        counts = Counter(
            finding.severity.value
            for finding in self.findings
        )

        return {
            severity.value: counts.get(
                severity.value,
                0,
            )
            for severity
            in Phase7FindingSeverity
        }

    def finalize(
        self,
        decision: Phase7ReadinessDecision,
    ) -> None:
        """Finalize the Phase 7 result."""

        self.readiness = decision
        self.completed_at = datetime.now(UTC)

        if self.execution_metadata is not None:
            self.execution_metadata.finalize()

    def to_report_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the result into a report dictionary."""

        return {
            "phase": "phase_7",
            "status": (
                self.readiness.status.value
                if self.readiness is not None
                else None
            ),
            "ready_for_phase_8": (
                self.readiness.ready
                if self.readiness is not None
                else False
            ),
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at is not None
                else None
            ),
            "partition_rows": {
                "train": self.train.row_count(),
                "validation": (
                    self.validation.row_count()
                ),
                "test": self.test.row_count(),
            },
            "partition_features": {
                "train": self.train.feature_count(),
                "validation": (
                    self.validation.feature_count()
                ),
                "test": self.test.feature_count(),
            },
            "finding_counts": self.finding_counts(),
            "findings": [
                finding.model_dump(
                    mode="json"
                )
                for finding in self.findings
            ],
            "validation": (
                self.validation_results.model_dump(
                    mode="json"
                )
            ),
            "execution_metadata": (
                self.execution_metadata.model_dump(
                    mode="json"
                )
                if self.execution_metadata is not None
                else None
            ),
            "readiness": (
                self.readiness.model_dump(
                    mode="json"
                )
                if self.readiness is not None
                else None
            ),
        }
