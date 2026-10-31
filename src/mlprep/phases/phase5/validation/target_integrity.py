"""Target-integrity validation for Phase 5."""

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


class TargetIntegrityIssue(BaseModel):
    """Target-integrity issue for one partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    partition: PartitionName

    source_row_count: int = Field(
        default=0,
        ge=0,
    )

    transformed_row_count: int = Field(
        default=0,
        ge=0,
    )

    row_count_changed: bool = False
    index_changed: bool = False
    name_changed: bool = False
    dtype_changed: bool = False
    values_changed: bool = False
    missingness_changed: bool = False

    source_fingerprint: str
    transformed_fingerprint: str

    blocks_readiness: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class TargetIntegrityResult(BaseModel):
    """Complete target-integrity validation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    issues: list[
        TargetIntegrityIssue
    ] = Field(default_factory=list)

    unchanged_partitions: list[
        PartitionName
    ] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class TargetIntegrityOptions:
    """Options controlling target-integrity validation."""

    enabled: bool = True

    require_identical_row_count: bool = True
    require_identical_index: bool = True
    require_identical_name: bool = True
    require_identical_dtype: bool = True
    require_identical_values: bool = True
    require_identical_missingness: bool = True


class TargetIntegrityValidator:
    """Ensure Phase 5 did not modify target values."""

    component_name = "target_integrity"

    def __init__(
        self,
        options: TargetIntegrityOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or TargetIntegrityOptions()
        )

    def validate(
        self,
        *,
        source_train: DatasetPartition,
        transformed_train: DatasetPartition,
        source_validation: DatasetPartition,
        transformed_validation: DatasetPartition,
        source_test: DatasetPartition,
        transformed_test: DatasetPartition,
    ) -> TargetIntegrityResult:
        """Validate targets across every partition."""

        if not self.options.enabled:
            return TargetIntegrityResult(
                metadata={
                    "enabled": False,
                }
            )

        pairs = [
            (
                source_train,
                transformed_train,
            ),
            (
                source_validation,
                transformed_validation,
            ),
            (
                source_test,
                transformed_test,
            ),
        ]

        issues: list[
            TargetIntegrityIssue
        ] = []

        unchanged: list[
            PartitionName
        ] = []

        for source, transformed in pairs:
            issue = self._compare(
                source=source,
                transformed=transformed,
            )

            if issue is None:
                unchanged.append(
                    source.name
                )

            else:
                issues.append(
                    issue
                )

        return TargetIntegrityResult(
            valid=not any(
                issue.blocks_readiness
                for issue in issues
            ),
            issues=issues,
            unchanged_partitions=unchanged,
            metadata={
                "enabled": True,
            },
        )

    def _compare(
        self,
        *,
        source: DatasetPartition,
        transformed: DatasetPartition,
    ) -> TargetIntegrityIssue | None:
        """Compare one target before and after treatment."""

        source_target = source.y
        transformed_target = transformed.y

        source_fingerprint = (
            self._fingerprint(
                source_target
            )
        )

        transformed_fingerprint = (
            self._fingerprint(
                transformed_target
            )
        )

        row_count_changed = bool(
            len(source_target)
            != len(transformed_target)
        )

        index_changed = bool(
            not source_target.index.equals(
                transformed_target.index
            )
        )

        name_changed = bool(
            source_target.name
            != transformed_target.name
        )

        dtype_changed = bool(
            str(source_target.dtype)
            != str(
                transformed_target.dtype
            )
        )

        values_changed = bool(
            source_fingerprint
            != transformed_fingerprint
        )

        missingness_changed = bool(
            not source_target.isna().equals(
                transformed_target.isna()
            )
        )

        blocks_readiness = bool(
            (
                self.options
                .require_identical_row_count
                and row_count_changed
            )
            or (
                self.options
                .require_identical_index
                and index_changed
            )
            or (
                self.options
                .require_identical_name
                and name_changed
            )
            or (
                self.options
                .require_identical_dtype
                and dtype_changed
            )
            or (
                self.options
                .require_identical_values
                and values_changed
            )
            or (
                self.options
                .require_identical_missingness
                and missingness_changed
            )
        )

        has_issue = bool(
            row_count_changed
            or index_changed
            or name_changed
            or dtype_changed
            or values_changed
            or missingness_changed
        )

        if not has_issue:
            return None

        return TargetIntegrityIssue(
            partition=source.name,
            source_row_count=(
                len(source_target)
            ),
            transformed_row_count=(
                len(transformed_target)
            ),
            row_count_changed=(
                row_count_changed
            ),
            index_changed=index_changed,
            name_changed=name_changed,
            dtype_changed=dtype_changed,
            values_changed=values_changed,
            missingness_changed=(
                missingness_changed
            ),
            source_fingerprint=(
                source_fingerprint
            ),
            transformed_fingerprint=(
                transformed_fingerprint
            ),
            blocks_readiness=(
                blocks_readiness
            ),
        )

    @staticmethod
    def _fingerprint(
        target: pd.Series,
    ) -> str:
        """Generate a deterministic target fingerprint."""

        digest = hashlib.sha256()

        digest.update(
            (
                f"{target.name}|"
                f"{target.dtype}|"
                f"{len(target)}"
            ).encode("utf-8")
        )

        hashes = pd.util.hash_pandas_object(
            target,
            index=True,
            categorize=True,
        )

        digest.update(
            hashes.to_numpy().tobytes()
        )

        return digest.hexdigest()