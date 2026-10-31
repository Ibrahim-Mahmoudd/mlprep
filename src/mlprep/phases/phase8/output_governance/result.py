"""Result models produced by Phase 8."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from pydantic import Field

from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase8.output_governance.metadata import (
    Phase8ExecutionMetadata,
    Phase8Finding,
    Phase8FindingSeverity,
    Phase8Model,
    Phase8Status,
)
from mlprep.phases.phase8.validation import (
    DistributionValidationResult,
    PartitionIntegrityResult,
    SchemaConsistencyResult,
    SyntheticIntegrityResult,
    TargetIntegrityResult,
)


class Phase8ReadinessDecision(Phase8Model):
    """Decision indicating whether Phase 9 may begin."""

    ready: bool
    status: Phase8Status

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
        default_factory=lambda: datetime.now(
            UTC
        ),
    )


class Phase8ValidationResults(Phase8Model):
    """Combined Phase 8 validation results."""

    distribution: (
        DistributionValidationResult | None
    ) = None

    schema_consistency: (
        SchemaConsistencyResult | None
    ) = None

    synthetic_integrity: (
        SyntheticIntegrityResult | None
    ) = None

    target_integrity: (
        TargetIntegrityResult | None
    ) = None

    partition_integrity: (
        PartitionIntegrityResult | None
    ) = None


class Phase8Result:
    """Complete output produced by Phase 8."""

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
    ) -> None:
        """Initialize a Phase 8 result."""

        self.train = train
        self.validation = validation
        self.test = test

        self.findings: list[
            Phase8Finding
        ] = []

        self.validation_results = (
            Phase8ValidationResults()
        )

        self.execution_metadata: (
            Phase8ExecutionMetadata | None
        ) = None

        self.readiness: (
            Phase8ReadinessDecision | None
        ) = None

        self.class_weights: dict[
            str,
            float,
        ] = {}

        self.sample_weights: Any | None = None

        self.phase_state: dict[
            str,
            Any,
        ] = {}

        self.started_at = datetime.now(
            UTC
        )

        self.completed_at: datetime | None = None

    def add_finding(
        self,
        finding: Phase8Finding,
    ) -> None:
        """Add one finding."""

        self.findings.append(
            finding
        )

    def add_findings(
        self,
        findings: list[Phase8Finding],
    ) -> None:
        """Add multiple findings."""

        self.findings.extend(
            findings
        )

    def blocking_findings(
        self,
    ) -> list[Phase8Finding]:
        """Return findings that block Phase 9."""

        return [
            finding
            for finding in self.findings
            if (
                finding.blocks_readiness
                or finding.severity
                == Phase8FindingSeverity.CRITICAL
            )
        ]

    def findings_by_severity(
        self,
        severity: Phase8FindingSeverity,
    ) -> list[Phase8Finding]:
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
            in Phase8FindingSeverity
        }

    def finalize(
        self,
        decision: Phase8ReadinessDecision,
    ) -> None:
        """Finalize the Phase 8 result."""

        self.readiness = decision
        self.completed_at = datetime.now(
            UTC
        )

        if self.execution_metadata is not None:
            self.execution_metadata.finalize()

    def to_report_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the result into a report dictionary."""

        from mlprep.phases.phase8.output_governance.fingerprint import (
            Phase8FingerprintManager,
        )

        serializer = Phase8FingerprintManager()

        return {
            "phase": "phase_8",
            "status": (
                self.readiness.status.value
                if self.readiness is not None
                else None
            ),
            "ready_for_phase_9": (
                self.readiness.ready
                if self.readiness is not None
                else False
            ),
            "started_at": (
                self.started_at.isoformat()
            ),
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
            "class_weights": dict(
                self.class_weights
            ),
            "sample_weight_count": (
                len(
                    self.sample_weights
                )
                if self.sample_weights is not None
                else 0
            ),
            "finding_counts": (
                self.finding_counts()
            ),
            "findings": [
                serializer.serialize_model(finding)
                for finding in self.findings
            ],
            "validation": (
                serializer.serialize_model(
                    self.validation_results
                )
            ),
            "execution_metadata": (
                serializer.serialize_model(
                    self.execution_metadata
                )
                if self.execution_metadata is not None
                else None
            ),
            "readiness": (
                serializer.serialize_model(self.readiness)
                if self.readiness is not None
                else None
            ),
        }
