"""Result models produced by Phase 4."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import Field

from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    LockedTestPartition,
)
from mlprep.phases.phase4.output_governance.metadata import (
    Phase4ExecutionMetadata,
    Phase4FindingAction,
    Phase4FindingCategory,
    Phase4FindingSeverity,
    Phase4Model,
    Phase4Status,
)
from mlprep.phases.phase4.validation.distribution_impact import (
    DistributionImpactResult,
)
from mlprep.phases.phase4.validation.remaining_missingness import (
    RemainingMissingnessResult,
)
from mlprep.phases.phase4.validation.row_integrity import (
    RowIntegrityResult,
)
from mlprep.phases.phase4.validation.schema_consistency import (
    SchemaConsistencyResult,
)
from mlprep.phases.phase4.validation.target_integrity import (
    TargetIntegrityResult,
)


class Phase4Finding(Phase4Model):
    """One finding produced during Phase 4."""

    finding_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    code: str
    category: Phase4FindingCategory
    severity: Phase4FindingSeverity

    component: str
    message: str

    columns: list[str] = Field(
        default_factory=list
    )

    partitions: list[str] = Field(
        default_factory=list
    )

    affected_row_count: int = Field(
        default=0,
        ge=0,
    )

    recommended_action: Phase4FindingAction = (
        Phase4FindingAction.REVIEW
    )

    applied_action: Phase4FindingAction = (
        Phase4FindingAction.REPORT
    )

    blocks_readiness: bool = False

    details: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class Phase4ReadinessDecision(Phase4Model):
    """Decision indicating whether Phase 5 may begin."""

    ready: bool
    status: Phase4Status

    blocking_finding_ids: list[str] = Field(
        default_factory=list
    )

    blocking_finding_codes: list[str] = Field(
        default_factory=list
    )

    reasons: list[str] = Field(
        default_factory=list
    )

    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class Phase4ValidationResults(Phase4Model):
    """Combined Phase 4 validation results."""

    remaining_missingness: (
        RemainingMissingnessResult | None
    ) = None

    schema_consistency: (
        SchemaConsistencyResult | None
    ) = None

    target_integrity: (
        TargetIntegrityResult | None
    ) = None

    row_integrity: RowIntegrityResult | None = None

    distribution_impact: (
        DistributionImpactResult | None
    ) = None


class Phase4Result:
    """Complete output produced by Phase 4."""

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition | LockedTestPartition,
    ) -> None:
        """Initialize a Phase 4 result."""

        self.train = train
        self.validation = validation
        self.test = test

        self.findings: list[Phase4Finding] = []

        self.validation_results = (
            Phase4ValidationResults()
        )

        self.execution_metadata: (
            Phase4ExecutionMetadata | None
        ) = None

        self.readiness: (
            Phase4ReadinessDecision | None
        ) = None

        self.phase_state: dict[str, Any] = {}

        self.started_at = datetime.now(UTC)
        self.completed_at: datetime | None = None

    def add_finding(
        self,
        finding: Phase4Finding,
    ) -> None:
        """Add one finding."""

        self.findings.append(finding)

    def add_findings(
        self,
        findings: list[Phase4Finding],
    ) -> None:
        """Add multiple findings."""

        self.findings.extend(findings)

    def blocking_findings(
        self,
    ) -> list[object]:
        """Return findings that block Phase 5."""

        return [
            finding
            for finding in self.findings
            if (
                finding.blocks_readiness
                or finding.severity
                == Phase4FindingSeverity.CRITICAL
            )
        ]

    def findings_by_severity(
        self,
        severity: Phase4FindingSeverity,
    ) -> list[object]:
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
            in Phase4FindingSeverity
        }

    def finalize(
        self,
        decision: Phase4ReadinessDecision,
    ) -> None:
        """Finalize the Phase 4 result."""

        self.readiness = decision
        self.completed_at = datetime.now(UTC)

        if self.execution_metadata is not None:
            self.execution_metadata.finalize()

    def to_report_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the result to a report dictionary."""

        return {
            "phase": "phase_4",
            "status": (
                self.readiness.status.value
                if self.readiness is not None
                else None
            ),
            "ready_for_phase_5": (
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
                "validation": self.validation.row_count(),
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
