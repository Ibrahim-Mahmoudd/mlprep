"""Fingerprint verification for Phase 3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel

from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)
from mlprep.phases.phase3.output_governance.result import (
    Phase3Finding,
    Phase3FindingCategory,
    Phase3FindingSeverity,
)
from mlprep.phases.phase3.output_governance.state import (
    Phase3State,
)


class Phase3FingerprintManager:
    """Generate and verify Phase 3 fingerprints."""

    component_name = "fingerprint"

    def capture_before(
        self,
        *,
        contract: EDAContract,
        state: Phase3State,
    ) -> None:
        """Capture partition fingerprints before analysis."""

        state.training_fingerprint_before = (
            self.partition_fingerprint(
                X=contract.train.X,
                y=contract.train.y,
                row_ids=contract.train.row_ids,
            )
        )

        if contract.validation is not None:
            state.validation_fingerprint_before = (
                self.partition_fingerprint(
                    X=contract.validation.X,
                    y=contract.validation.y,
                    row_ids=(
                        contract.validation.row_ids
                    ),
                )
            )

        state.phase2_split_fingerprint = (
            contract.phase2_state.split_fingerprint
        )

    def verify_after(
        self,
        *,
        contract: EDAContract,
        state: Phase3State,
    ) -> list[Phase3Finding]:
        """Verify that Phase 3 did not modify source partitions."""

        findings: list[Phase3Finding] = []

        state.training_fingerprint_after = (
            self.partition_fingerprint(
                X=contract.train.X,
                y=contract.train.y,
                row_ids=contract.train.row_ids,
            )
        )

        state.training_data_unchanged = (
            state.training_fingerprint_before
            == state.training_fingerprint_after
        )

        if not state.training_data_unchanged:
            findings.append(
                Phase3Finding(
                    code="P3-FINGERPRINT-001",
                    category=(
                        Phase3FindingCategory.FINGERPRINT
                    ),
                    severity=(
                        Phase3FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "The training partition changed during "
                        "Phase 3 analysis."
                    ),
                    blocks_readiness=True,
                )
            )

        if contract.validation is not None:
            state.validation_fingerprint_after = (
                self.partition_fingerprint(
                    X=contract.validation.X,
                    y=contract.validation.y,
                    row_ids=(
                        contract.validation.row_ids
                    ),
                )
            )

            state.validation_data_unchanged = (
                state.validation_fingerprint_before
                == state.validation_fingerprint_after
            )

            if not state.validation_data_unchanged:
                findings.append(
                    Phase3Finding(
                        code="P3-FINGERPRINT-002",
                        category=(
                            Phase3FindingCategory.FINGERPRINT
                        ),
                        severity=(
                            Phase3FindingSeverity.CRITICAL
                        ),
                        component=self.component_name,
                        message=(
                            "The validation partition changed "
                            "during Phase 3 analysis."
                        ),
                        blocks_readiness=True,
                    )
                )

        else:
            state.validation_data_unchanged = True

        return findings

    def configuration_fingerprint(
        self,
        configuration: object,
    ) -> str:
        """Generate a configuration fingerprint."""

        serializable = self._serialize(
            configuration
        )

        return hashlib.sha256(
            json.dumps(
                serializable,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def analysis_fingerprint(
        self,
        *,
        state: Phase3State,
    ) -> str:
        """Generate a fingerprint for Phase 3 analysis state."""

        payload = {
            "training": (
                state.training_fingerprint_after
            ),
            "validation": (
                state.validation_fingerprint_after
            ),
            "configuration": (
                state.configuration_fingerprint
            ),
            "components": sorted(
                state.completed_components
            ),
            "columns": asdict(state.columns),
        }

        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def partition_fingerprint(
        *,
        X: pd.DataFrame,
        y: pd.Series,
        row_ids: pd.Series,
    ) -> str:
        """Generate one partition fingerprint."""

        digest = hashlib.sha256()

        metadata = {
            "columns": [
                str(column)
                for column in X.columns
            ],
            "dtypes": {
                str(column): str(X[column].dtype)
                for column in X.columns
            },
            "target_dtype": str(y.dtype),
            "row_id_dtype": str(row_ids.dtype),
            "row_count": len(X),
        }

        digest.update(
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

        for value in (
            X,
            y,
            row_ids,
        ):
            hashes = pd.util.hash_pandas_object(
                value,
                index=True,
                categorize=True,
            )

            digest.update(
                hashes.to_numpy().tobytes()
            )

        return digest.hexdigest()

    @classmethod
    def _serialize(
        cls,
        value: Any,
    ) -> Any:
        """Recursively serialize a configuration object."""

        if isinstance(value, BaseModel):
            return value.model_dump(
                mode="json"
            )

        if is_dataclass(value) and not isinstance(value, type):
            return {
                key: cls._serialize(item)
                for key, item
                in asdict(value).items()
            }

        if isinstance(value, dict):
            return {
                str(key): cls._serialize(item)
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                cls._serialize(item)
                for item in value
            ]

        if hasattr(value, "value"):
            return value.value

        return value
