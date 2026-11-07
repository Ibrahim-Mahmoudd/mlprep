"""Protected partition guards for Phase 8."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase8.orchestration.fit_context import (
    BalancingFitContext,
)


class PartitionGuardSnapshot(BaseModel):
    """Immutable snapshot of one protected partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    partition: PartitionName

    row_count: int = Field(
        ge=0,
    )

    feature_count: int = Field(
        ge=0,
    )

    feature_columns: list[str] = Field(
        default_factory=list,
    )

    feature_dtypes: dict[str, str] = Field(
        default_factory=dict,
    )

    feature_fingerprint: str
    target_fingerprint: str
    row_id_fingerprint: str
    partition_fingerprint: str

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class PartitionGuardResult(BaseModel):
    """Verification result for one protected partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    partition: PartitionName

    valid: bool = True

    row_count_unchanged: bool = True
    feature_count_unchanged: bool = True
    feature_schema_unchanged: bool = True
    feature_dtypes_unchanged: bool = True
    feature_values_unchanged: bool = True

    target_unchanged: bool = True
    row_ids_unchanged: bool = True
    row_order_unchanged: bool = True
    partition_name_unchanged: bool = True

    source_partition_fingerprint: str
    output_partition_fingerprint: str

    violations: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class PartitionGuardOptions:
    """Options controlling protected-partition verification."""

    enabled: bool = True

    require_identical_row_count: bool = True
    require_identical_feature_count: bool = True
    require_identical_feature_schema: bool = True
    require_identical_feature_dtypes: bool = True
    require_identical_feature_values: bool = True

    require_identical_target: bool = True
    require_identical_row_ids: bool = True
    require_identical_row_order: bool = True
    require_identical_partition_name: bool = True

    copy_protected_partition: bool = True


class PartitionGuard:
    """Capture and verify validation or test partitions."""

    component_name = "partition_guard"

    def __init__(
        self,
        options: PartitionGuardOptions | None = None,
    ) -> None:
        """Initialize the partition guard."""

        self.options = (
            options
            or PartitionGuardOptions()
        )

        self._snapshots: dict[
            PartitionName,
            PartitionGuardSnapshot,
        ] = {}

        self._partitions: dict[
            PartitionName,
            DatasetPartition,
        ] = {}

    def capture(
        self,
        *,
        partition: DatasetPartition,
        context: BalancingFitContext,
    ) -> PartitionGuardSnapshot:
        """Capture a protected partition before Phase 8."""

        context.assert_protection_context()

        if partition.name != context.partition:
            raise ValueError(
                "The partition does not match the "
                "protection context."
            )

        if partition.name not in {
            PartitionName.VALIDATION,
            PartitionName.TEST,
        }:
            raise PermissionError(
                "Only validation and test may be protected."
            )

        protected = partition.copy(
            deep=self.options.copy_protected_partition
        )

        snapshot = PartitionGuardSnapshot(
            partition=protected.name,
            row_count=protected.row_count(),
            feature_count=protected.feature_count(),
            feature_columns=protected.feature_names(),
            feature_dtypes={
                str(column): str(
                    protected.X[
                        column
                    ].dtype
                )
                for column
                in protected.X.columns
            },
            feature_fingerprint=(
                self._frame_fingerprint(
                    protected.X
                )
            ),
            target_fingerprint=(
                self._series_fingerprint(
                    protected.y
                )
            ),
            row_id_fingerprint=(
                self._series_fingerprint(
                    protected.row_ids
                )
            ),
            partition_fingerprint=(
                self._partition_fingerprint(
                    protected
                )
            ),
        )

        self._snapshots[
            partition.name
        ] = snapshot

        self._partitions[
            partition.name
        ] = protected

        return snapshot.model_copy(
            deep=True
        )

    def protected_copy(
        self,
        partition: PartitionName,
    ) -> DatasetPartition:
        """Return a deep copy of a captured partition."""

        if partition not in self._partitions:
            raise KeyError(
                f"Partition {partition.value!r} "
                "has not been captured."
            )

        return self._partitions[
            partition
        ].copy(
            deep=True
        )

    def verify(
        self,
        *,
        partition: DatasetPartition,
        context: BalancingFitContext,
    ) -> PartitionGuardResult:
        """Verify that a protected partition is unchanged."""

        context.assert_protection_context()

        if partition.name != context.partition:
            raise ValueError(
                "The partition does not match the "
                "protection context."
            )

        snapshot = self._snapshots.get(
            partition.name
        )

        source = self._partitions.get(
            partition.name
        )

        if snapshot is None or source is None:
            raise RuntimeError(
                f"Partition {partition.name.value!r} "
                "was not captured before verification."
            )

        row_count_unchanged = bool(
            partition.row_count()
            == snapshot.row_count
        )

        feature_count_unchanged = bool(
            partition.feature_count()
            == snapshot.feature_count
        )

        feature_schema_unchanged = bool(
            partition.feature_names()
            == snapshot.feature_columns
        )

        output_dtypes = {
            str(column): str(
                partition.X[
                    column
                ].dtype
            )
            for column
            in partition.X.columns
        }

        feature_dtypes_unchanged = bool(
            output_dtypes
            == snapshot.feature_dtypes
        )

        feature_values_unchanged = bool(
            self._frame_fingerprint(
                partition.X
            )
            == snapshot.feature_fingerprint
            and partition.X.equals(
                source.X
            )
        )

        target_unchanged = bool(
            self._series_fingerprint(
                partition.y
            )
            == snapshot.target_fingerprint
            and partition.y.equals(
                source.y
            )
        )

        row_ids_unchanged = bool(
            self._series_fingerprint(
                partition.row_ids
            )
            == snapshot.row_id_fingerprint
            and partition.row_ids.equals(
                source.row_ids
            )
        )

        row_order_unchanged = bool(
            partition.X.index.equals(
                source.X.index
            )
            and partition.y.index.equals(
                source.y.index
            )
            and partition.row_ids.index.equals(
                source.row_ids.index
            )
        )

        partition_name_unchanged = bool(
            partition.name
            == snapshot.partition
        )

        violations: list[str] = []

        checks = [
            (
                self.options.require_identical_row_count,
                row_count_unchanged,
                "row_count_changed",
            ),
            (
                self.options.require_identical_feature_count,
                feature_count_unchanged,
                "feature_count_changed",
            ),
            (
                self.options.require_identical_feature_schema,
                feature_schema_unchanged,
                "feature_schema_changed",
            ),
            (
                self.options.require_identical_feature_dtypes,
                feature_dtypes_unchanged,
                "feature_dtypes_changed",
            ),
            (
                self.options.require_identical_feature_values,
                feature_values_unchanged,
                "feature_values_changed",
            ),
            (
                self.options.require_identical_target,
                target_unchanged,
                "target_changed",
            ),
            (
                self.options.require_identical_row_ids,
                row_ids_unchanged,
                "row_ids_changed",
            ),
            (
                self.options.require_identical_row_order,
                row_order_unchanged,
                "row_order_changed",
            ),
            (
                self.options.require_identical_partition_name,
                partition_name_unchanged,
                "partition_name_changed",
            ),
        ]

        for required, passed, violation in checks:
            if required and not passed:
                violations.append(
                    violation
                )

        output_fingerprint = (
            self._partition_fingerprint(
                partition
            )
        )

        return PartitionGuardResult(
            partition=partition.name,
            valid=not violations,
            row_count_unchanged=(
                row_count_unchanged
            ),
            feature_count_unchanged=(
                feature_count_unchanged
            ),
            feature_schema_unchanged=(
                feature_schema_unchanged
            ),
            feature_dtypes_unchanged=(
                feature_dtypes_unchanged
            ),
            feature_values_unchanged=(
                feature_values_unchanged
            ),
            target_unchanged=target_unchanged,
            row_ids_unchanged=row_ids_unchanged,
            row_order_unchanged=(
                row_order_unchanged
            ),
            partition_name_unchanged=(
                partition_name_unchanged
            ),
            source_partition_fingerprint=(
                snapshot.partition_fingerprint
            ),
            output_partition_fingerprint=(
                output_fingerprint
            ),
            violations=violations,
            metadata={
                "fingerprints_identical": (
                    snapshot.partition_fingerprint
                    == output_fingerprint
                ),
            },
        )

    @classmethod
    def _partition_fingerprint(
        cls,
        partition: DatasetPartition,
    ) -> str:
        """Generate a complete partition fingerprint."""

        digest = hashlib.sha256()

        digest.update(
            partition.name.value.encode(
                "utf-8"
            )
        )

        digest.update(
            cls._frame_fingerprint(
                partition.X
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            cls._series_fingerprint(
                partition.y
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            cls._series_fingerprint(
                partition.row_ids
            ).encode(
                "utf-8"
            )
        )

        return digest.hexdigest()

    @staticmethod
    def _frame_fingerprint(
        frame: pd.DataFrame,
    ) -> str:
        """Generate a deterministic DataFrame fingerprint."""

        digest = hashlib.sha256()

        schema = "|".join(
            f"{column}:{frame[column].dtype}"
            for column in frame.columns
        )

        digest.update(
            schema.encode(
                "utf-8"
            )
        )

        hashes = pd.util.hash_pandas_object(
            frame,
            index=True,
            categorize=True,
        )

        digest.update(
            hashes.to_numpy().tobytes()
        )

        return digest.hexdigest()

    @staticmethod
    def _series_fingerprint(
        series: pd.Series,
    ) -> str:
        """Generate a deterministic Series fingerprint."""

        digest = hashlib.sha256()

        digest.update(
            (
                f"{series.name}|"
                f"{series.dtype}|"
                f"{len(series)}"
            ).encode()
        )

        hashes = pd.util.hash_pandas_object(
            series,
            index=True,
            categorize=True,
        )

        digest.update(
            hashes.to_numpy().tobytes()
        )

        return digest.hexdigest()
