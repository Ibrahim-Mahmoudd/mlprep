"""Result models produced by Phase 3."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.diagnostics.data_quality import (
    DataQualityDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.distribution_shift import (
    DistributionShiftResult,
)
from mlprep.phases.phase3.diagnostics.high_cardinality import (
    HighCardinalityDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.rare_categories import (
    RareCategoryDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.skewness import (
    SkewnessDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.temporal_patterns import (
    TemporalPatternResult,
)
from mlprep.phases.phase3.profiling.categorical_profile import (
    CategoricalColumnProfile,
)
from mlprep.phases.phase3.profiling.dataset_profile import (
    DatasetProfile,
)
from mlprep.phases.phase3.profiling.datetime_profile import (
    DatetimeColumnProfile,
)
from mlprep.phases.phase3.profiling.missingness_profile import (
    MissingnessProfile,
)
from mlprep.phases.phase3.profiling.numeric_profile import (
    NumericColumnProfile,
)
from mlprep.phases.phase3.profiling.target_profile import (
    TargetProfile,
)
from mlprep.phases.phase3.profiling.text_profile import (
    TextColumnProfile,
)
from mlprep.phases.phase3.recommendations.recommendation_engine import (
    Phase3Recommendations,
)
from mlprep.phases.phase3.relationships.categorical_associations import (
    CategoricalAssociationResult,
)
from mlprep.phases.phase3.relationships.interaction_candidates import (
    InteractionCandidateResult,
)
from mlprep.phases.phase3.relationships.missingness_associations import (
    MissingnessAssociationResult,
)
from mlprep.phases.phase3.relationships.mixed_associations import (
    MixedAssociationResult,
)
from mlprep.phases.phase3.relationships.multicollinearity import (
    MulticollinearityResult,
)
from mlprep.phases.phase3.relationships.numeric_associations import (
    NumericAssociationResult,
)
from mlprep.phases.phase3.relationships.target_associations import (
    TargetAssociationResult,
)
from mlprep.phases.phase3.visualization.plot_spec import (
    PlotCollection,
)


class Phase3Model(BaseModel):
    """Base serializable model for Phase 3."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class Phase3Status(StrEnum):
    """Final status of Phase 3."""

    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class Phase3FindingSeverity(StrEnum):
    """Severity of a Phase 3 finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Phase3FindingCategory(StrEnum):
    """Category of a Phase 3 finding."""

    CONTRACT = "contract"
    PROFILING = "profiling"
    RELATIONSHIP = "relationship"
    DIAGNOSTIC = "diagnostic"
    VISUALIZATION = "visualization"
    RECOMMENDATION = "recommendation"
    PRIVACY = "privacy"
    FINGERPRINT = "fingerprint"
    READINESS = "readiness"
    EXECUTION = "execution"


class Phase3Finding(Phase3Model):
    """One finding produced during Phase 3."""

    finding_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    code: str
    category: Phase3FindingCategory
    severity: Phase3FindingSeverity

    component: str
    message: str

    columns: list[str] = Field(
        default_factory=list
    )

    blocks_readiness: bool = False

    details: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class Phase3ReadinessDecision(Phase3Model):
    """Decision indicating whether Phase 4 may begin."""

    ready: bool
    status: Phase3Status

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


class Phase3Result:
    """Complete analytical output produced by Phase 3."""

    def __init__(self) -> None:
        """Initialize an empty Phase 3 result."""

        self.dataset_profile: DatasetProfile | None = None

        self.numeric_profiles: dict[
            str,
            NumericColumnProfile,
        ] = {}

        self.categorical_profiles: dict[
            str,
            CategoricalColumnProfile,
        ] = {}

        self.datetime_profiles: dict[
            str,
            DatetimeColumnProfile,
        ] = {}

        self.text_profiles: dict[
            str,
            TextColumnProfile,
        ] = {}

        self.missingness_profile: (
            MissingnessProfile | None
        ) = None

        self.target_profile: TargetProfile | None = None

        self.numeric_associations: (
            NumericAssociationResult | None
        ) = None

        self.categorical_associations: (
            CategoricalAssociationResult | None
        ) = None

        self.mixed_associations: (
            MixedAssociationResult | None
        ) = None

        self.target_associations: (
            TargetAssociationResult | None
        ) = None

        self.missingness_associations: (
            MissingnessAssociationResult | None
        ) = None

        self.multicollinearity: (
            MulticollinearityResult | None
        ) = None

        self.interaction_candidates: (
            InteractionCandidateResult | None
        ) = None

        self.skewness_diagnostics: (
            SkewnessDiagnosticResult | None
        ) = None

        self.rare_category_diagnostics: (
            RareCategoryDiagnosticResult | None
        ) = None

        self.cardinality_diagnostics: (
            HighCardinalityDiagnosticResult | None
        ) = None

        self.distribution_shift: (
            DistributionShiftResult | None
        ) = None

        self.temporal_patterns: (
            TemporalPatternResult | None
        ) = None

        self.data_quality: (
            DataQualityDiagnosticResult | None
        ) = None

        self.recommendations: (
            Phase3Recommendations | None
        ) = None

        self.plots: PlotCollection = PlotCollection()

        self.findings: list[Phase3Finding] = []

        self.readiness: (
            Phase3ReadinessDecision | None
        ) = None

        self.phase_state: dict[str, Any] = {}

        self.started_at = datetime.now(UTC)
        self.completed_at: datetime | None = None

    def add_finding(
        self,
        finding: Phase3Finding,
    ) -> None:
        """Add one Phase 3 finding."""

        self.findings.append(finding)

    def add_findings(
        self,
        findings: list[Phase3Finding],
    ) -> None:
        """Add multiple Phase 3 findings."""

        self.findings.extend(findings)

    def get_findings_by_severity(
        self,
        severity: Phase3FindingSeverity,
    ) -> list[Phase3Finding]:
        """Return findings matching one severity."""

        return [
            finding
            for finding in self.findings
            if finding.severity == severity
        ]

    def get_findings_by_category(
        self,
        category: Phase3FindingCategory,
    ) -> list[Phase3Finding]:
        """Return findings matching one category."""

        return [
            finding
            for finding in self.findings
            if finding.category == category
        ]

    def get_blocking_findings(
        self,
    ) -> list[Phase3Finding]:
        """Return findings that block Phase 4."""

        return [
            finding
            for finding in self.findings
            if (
                finding.blocks_readiness
                or finding.severity
                == Phase3FindingSeverity.CRITICAL
            )
        ]

    def has_warnings(
        self,
    ) -> bool:
        """Return whether warning findings exist."""

        return any(
            finding.severity
            == Phase3FindingSeverity.WARNING
            for finding in self.findings
        )

    def finding_counts_by_severity(
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
            in Phase3FindingSeverity
        }

    def finalize(
        self,
        decision: Phase3ReadinessDecision,
    ) -> None:
        """Finalize the Phase 3 result."""

        self.readiness = decision
        self.completed_at = datetime.now(UTC)

    def to_report_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the result into a report dictionary."""

        return {
            "phase": "phase_3",
            "status": (
                self.readiness.status.value
                if self.readiness is not None
                else None
            ),
            "ready_for_phase_4": (
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
            "dataset_profile": self._dump(
                self.dataset_profile
            ),
            "numeric_profiles": self._dump_mapping(
                self.numeric_profiles
            ),
            "categorical_profiles": self._dump_mapping(
                self.categorical_profiles
            ),
            "datetime_profiles": self._dump_mapping(
                self.datetime_profiles
            ),
            "text_profiles": self._dump_mapping(
                self.text_profiles
            ),
            "missingness_profile": self._dump(
                self.missingness_profile
            ),
            "target_profile": self._dump(
                self.target_profile
            ),
            "relationships": {
                "numeric": self._dump(
                    self.numeric_associations
                ),
                "categorical": self._dump(
                    self.categorical_associations
                ),
                "mixed": self._dump(
                    self.mixed_associations
                ),
                "target": self._dump(
                    self.target_associations
                ),
                "missingness": self._dump(
                    self.missingness_associations
                ),
                "multicollinearity": self._dump(
                    self.multicollinearity
                ),
                "interaction_candidates": self._dump(
                    self.interaction_candidates
                ),
            },
            "diagnostics": {
                "skewness": self._dump(
                    self.skewness_diagnostics
                ),
                "rare_categories": self._dump(
                    self.rare_category_diagnostics
                ),
                "cardinality": self._dump(
                    self.cardinality_diagnostics
                ),
                "distribution_shift": self._dump(
                    self.distribution_shift
                ),
                "temporal_patterns": self._dump(
                    self.temporal_patterns
                ),
                "data_quality": self._dump(
                    self.data_quality
                ),
            },
            "recommendations": self._dump(
                self.recommendations
            ),
            "plots": self.plots.model_dump(
                mode="json"
            ),
            "finding_counts": (
                self.finding_counts_by_severity()
            ),
            "findings": [
                finding.model_dump(
                    mode="json"
                )
                for finding in self.findings
            ],
            "readiness": self._dump(
                self.readiness
            ),
        }

    @staticmethod
    def _dump(
        value: BaseModel | None,
    ) -> dict[str, Any] | None:
        """Serialize one Pydantic model."""

        if value is None:
            return None

        return value.model_dump(
            mode="json"
        )

    @staticmethod
    def _dump_mapping(
        values: dict[str, BaseModel],
    ) -> dict[str, dict[str, Any]]:
        """Serialize a mapping of Pydantic models."""

        return {
            key: value.model_dump(
                mode="json"
            )
            for key, value in values.items()
        }