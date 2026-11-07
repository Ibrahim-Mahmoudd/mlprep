"""Result objects produced by Phase 2."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionMetadata,
    PartitionName,
    Phase2Model,
    Phase2Status,
    SplitExecutionMetadata,
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
)

if TYPE_CHECKING:
    import pandas as pd

    from mlprep.phases.phase2.output_governance.state import (
        Phase2State,
    )


class SplitFinding(Phase2Model):
    """One finding detected during Phase 2."""

    finding_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    code: str

    category: SplitFindingCategory
    severity: SplitFindingSeverity

    component: str
    message: str

    partitions: list[PartitionName] = Field(
        default_factory=list
    )

    columns: list[str] = Field(
        default_factory=list
    )

    affected_row_count: int = Field(
        default=0,
        ge=0,
    )

    affected_row_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    example_row_ids: list[str | int] = Field(
        default_factory=list
    )

    recommended_action: SplitFindingAction = (
        SplitFindingAction.REPORT
    )

    applied_action: SplitFindingAction = (
        SplitFindingAction.NONE
    )

    blocks_readiness: bool = False

    details: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class Phase2ReadinessDecision(Phase2Model):
    """Decision indicating whether Phase 3 may begin."""

    ready: bool
    status: Phase2Status

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


class DatasetPartition:
    """One train, validation, or test partition."""

    def __init__(
        self,
        *,
        name: PartitionName,
        X: pd.DataFrame,  # noqa: N803 - public partition API uses X/y notation
        y: pd.Series,
        row_ids: pd.Series,
        metadata: PartitionMetadata | None = None,
    ) -> None:
        """Initialize one dataset partition."""

        self.name = name
        self.X = X
        self.y = y
        self.row_ids = row_ids

        self.metadata = (
            metadata
            or PartitionMetadata(
                name=name,
                row_count=len(X),
                feature_count=len(X.columns),
            )
        )

        self._validate()

    def _validate(
        self,
    ) -> None:
        """Validate row alignment and identity."""

        if len(self.X) != len(self.y):
            raise ValueError(
                f"Partition {self.name.value!r} "
                "contains different feature and target "
                "row counts."
            )

        if len(self.X) != len(self.row_ids):
            raise ValueError(
                f"Partition {self.name.value!r} "
                "contains different feature and row-ID "
                "counts."
            )

        if not self.X.index.equals(
            self.y.index
        ):
            raise ValueError(
                f"Feature and target indices are not "
                f"aligned in {self.name.value!r}."
            )

        if not self.X.index.equals(
            self.row_ids.index
        ):
            raise ValueError(
                f"Feature and row-ID indices are not "
                f"aligned in {self.name.value!r}."
            )

        if self.row_ids.isna().any():
            raise ValueError(
                f"Partition {self.name.value!r} "
                "contains missing row IDs."
            )

        if not self.row_ids.is_unique:
            raise ValueError(
                f"Partition {self.name.value!r} "
                "contains duplicate row IDs."
            )

        if len(self.X.columns) != len(
            set(self.X.columns)
        ):
            raise ValueError(
                f"Partition {self.name.value!r} "
                "contains duplicate feature names."
            )

    def copy(
        self,
        *,
        deep: bool = True,
    ) -> DatasetPartition:
        """Return a partition copy."""

        return DatasetPartition(
            name=self.name,
            X=self.X.copy(deep=deep),
            y=self.y.copy(deep=deep),
            row_ids=self.row_ids.copy(
                deep=deep
            ),
            metadata=self.metadata.model_copy(
                deep=deep
            ),
        )

    def row_count(
        self,
    ) -> int:
        """Return the row count."""

        return len(self.X)

    def feature_count(
        self,
    ) -> int:
        """Return the feature count."""

        return len(self.X.columns)

    def feature_names(
        self,
    ) -> list[str]:
        """Return ordered feature names."""

        return [
            str(column)
            for column in self.X.columns
        ]

    def to_frame(
        self,
        *,
        target_column: str,
        row_id_column: str,
    ) -> pd.DataFrame:
        """Combine features, target, and row IDs."""

        frame = self.X.copy()

        frame[target_column] = self.y
        frame[row_id_column] = self.row_ids

        return frame


class LockedTestPartition:
    """Restricted wrapper around the test partition."""

    def __init__(
        self,
        partition: DatasetPartition,
    ) -> None:
        """Initialize a locked test partition."""

        if partition.name != PartitionName.TEST:
            raise ValueError(
                "LockedTestPartition requires "
                "the test partition."
            )

        self._partition = partition
        self._locked = True
        self._access_log: list[str] = []

    @property
    def is_locked(
        self,
    ) -> bool:
        """Return whether the partition is locked."""

        return self._locked

    @property
    def metadata(
        self,
    ) -> PartitionMetadata:
        """Return test metadata without exposing values."""

        return self._partition.metadata

    @property
    def row_count(
        self,
    ) -> int:
        """Return the test row count."""

        return self._partition.row_count()

    @property
    def feature_count(
        self,
    ) -> int:
        """Return the test feature count."""

        return self._partition.feature_count()

    @property
    def feature_names(
        self,
    ) -> list[str]:
        """Return ordered test feature names."""

        return self._partition.feature_names()

    def access(
        self,
        *,
        reason: str,
        allow: bool = False,
    ) -> DatasetPartition:
        """Return an authorized copy of the test partition."""

        if not reason.strip():
            raise ValueError(
                "A test-access reason cannot be empty."
            )

        if not allow:
            raise PermissionError(
                "The test partition is locked."
            )

        self._access_log.append(reason)

        return self._partition.copy(
            deep=True
        )

    def access_log(
        self,
    ) -> list[object]:
        """Return a copy of the access log."""

        return list(self._access_log)


class Phase2Result:
    """Complete Phase 2 output."""

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        state: Phase2State,
        lock_test_set: bool = True,
    ) -> None:
        """Initialize the Phase 2 result."""

        self.train = train
        self.validation = validation

        if lock_test_set:
            self.test: (
                LockedTestPartition
                | DatasetPartition
            ) = LockedTestPartition(test)
        else:
            self.test = test

        self.state = state

        self.findings: list[SplitFinding] = []

        self.execution_metadata: (
            SplitExecutionMetadata | None
        ) = None

        self.readiness: (
            Phase2ReadinessDecision | None
        ) = None

        self.started_at = datetime.now(UTC)
        self.completed_at: datetime | None = None

    def add_finding(
        self,
        finding: SplitFinding,
    ) -> None:
        """Add one finding."""

        self.findings.append(finding)

    def add_findings(
        self,
        findings: list[SplitFinding],
    ) -> None:
        """Add multiple findings."""

        self.findings.extend(findings)

    def get_findings_by_severity(
        self,
        severity: SplitFindingSeverity,
    ) -> list[object]:
        """Return findings matching one severity."""

        return [
            finding
            for finding in self.findings
            if finding.severity == severity
        ]

    def get_findings_by_category(
        self,
        category: SplitFindingCategory,
    ) -> list[object]:
        """Return findings matching one category."""

        return [
            finding
            for finding in self.findings
            if finding.category == category
        ]

    def get_blocking_findings(
        self,
    ) -> list[object]:
        """Return findings that block Phase 3."""

        return [
            finding
            for finding in self.findings
            if (
                finding.blocks_readiness
                or finding.severity
                == SplitFindingSeverity.CRITICAL
            )
        ]

    def has_errors(
        self,
    ) -> bool:
        """Return whether errors exist."""

        return any(
            finding.severity
            in {
                SplitFindingSeverity.ERROR,
                SplitFindingSeverity.CRITICAL,
            }
            for finding in self.findings
        )

    def has_warnings(
        self,
    ) -> bool:
        """Return whether warnings exist."""

        return any(
            finding.severity
            == SplitFindingSeverity.WARNING
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
            in SplitFindingSeverity
        }

    def finding_counts_by_category(
        self,
    ) -> dict[str, int]:
        """Return finding counts by category."""

        counts = Counter(
            finding.category.value
            for finding in self.findings
        )

        return dict(
            sorted(counts.items())
        )

    def get_partition(
        self,
        partition: PartitionName,
        *,
        test_access_reason: str | None = None,
        authorize_test_access: bool = False,
    ) -> DatasetPartition:
        """Return one partition with lock enforcement."""

        if partition == PartitionName.TRAIN:
            return self.train

        if partition == PartitionName.VALIDATION:
            return self.validation

        if partition == PartitionName.TEST:
            if isinstance(
                self.test,
                LockedTestPartition,
            ):
                return self.test.access(
                    reason=(
                        test_access_reason
                        or "No reason provided"
                    ),
                    allow=authorize_test_access,
                )

            return self.test

        raise KeyError(
            f"Partition {partition.value!r} "
            "is unavailable."
        )

    def finalize(
        self,
        readiness: Phase2ReadinessDecision,
    ) -> None:
        """Finalize the Phase 2 result."""

        self.readiness = readiness
        self.completed_at = datetime.now(UTC)

        self.state.ready_for_phase_3 = (
            readiness.ready
        )

    def to_report_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the result to a report dictionary."""

        status: str | None = None
        ready_for_phase_3 = False
        readiness_data: dict[str, Any] | None = None

        if self.readiness is not None:
            status = self.readiness.status.value
            ready_for_phase_3 = (
                self.readiness.ready
            )

            readiness_data = (
                self.readiness.model_dump(
                    mode="json"
                )
            )

        return {
            "phase": "phase_2",
            "status": status,
            "ready_for_phase_3": (
                ready_for_phase_3
            ),
            "started_at": (
                self.started_at.isoformat()
            ),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at is not None
                else None
            ),
            "strategy": self.state.strategy.value,
            "partitions": {
                "train": (
                    self.train.metadata.model_dump(
                        mode="json"
                    )
                ),
                "validation": (
                    self.validation.metadata.model_dump(
                        mode="json"
                    )
                ),
                "test": (
                    self.test.metadata.model_dump(
                        mode="json"
                    )
                ),
            },
            "finding_counts_by_severity": (
                self.finding_counts_by_severity()
            ),
            "finding_counts_by_category": (
                self.finding_counts_by_category()
            ),
            "findings": [
                finding.model_dump(
                    mode="json"
                )
                for finding in self.findings
            ],
            "execution_metadata": (
                self.execution_metadata.model_dump(
                    mode="json"
                )
                if self.execution_metadata
                is not None
                else None
            ),
            "readiness": readiness_data,
        }
