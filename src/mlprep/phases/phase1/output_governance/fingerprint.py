"""Deterministic dataset and schema fingerprinting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.types import (
    FindingAction,
    FindingCategory,
    Severity,
)

FingerprintAlgorithm = Literal[
    "sha256",
    "sha512",
]


@dataclass(slots=True)
class FingerprintOptions:
    """Options controlling dataset fingerprint creation."""

    enabled: bool = True

    algorithm: FingerprintAlgorithm = "sha256"

    include_data_values: bool = True
    include_index: bool = True

    sample_size: int | None = None

    def __post_init__(self) -> None:
        """Validate fingerprint options."""

        if (
            self.sample_size is not None
            and self.sample_size <= 0
        ):
            raise ValueError(
                "sample_size must be positive or None."
            )


class FingerprintGenerator:
    """Generate deterministic dataset and schema fingerprints."""

    component_name = "fingerprint"

    def __init__(
        self,
        options: FingerprintOptions | None = None,
    ) -> None:
        """Initialize the fingerprint generator."""

        self.options = options or FingerprintOptions()

    def generate(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> tuple[str | None, str | None]:
        """Generate dataset and schema fingerprints."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Fingerprint generation requires a "
                "pandas DataFrame."
            )

        if not self.options.enabled:
            return None, None

        schema_fingerprint = self._schema_fingerprint(
            data=data,
            schema=result.canonical_schema,
        )

        dataset_fingerprint = self._dataset_fingerprint(
            data
        )

        state.schema_fingerprint = schema_fingerprint
        state.dataset_fingerprint = dataset_fingerprint

        result.dataset_summary.dataset_fingerprint = (
            dataset_fingerprint
        )

        result.add_finding(
            Finding(
                code="P1-FINGERPRINT-001",
                category=FindingCategory.FINGERPRINT,
                severity=Severity.INFO,
                component=self.component_name,
                message=(
                    "Dataset and schema fingerprints were "
                    "generated successfully."
                ),
                recommended_action=FindingAction.REPORT,
                applied_action=FindingAction.REPORT,
                blocks_readiness=False,
                details={
                    "algorithm": self.options.algorithm,
                    "dataset_fingerprint": (
                        dataset_fingerprint
                    ),
                    "schema_fingerprint": (
                        schema_fingerprint
                    ),
                },
            )
        )

        return (
            dataset_fingerprint,
            schema_fingerprint,
        )

    def _dataset_fingerprint(
        self,
        data: pd.DataFrame,
    ) -> str:
        """Generate a deterministic dataset fingerprint."""

        metadata = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "column_order": [
                str(column)
                for column in data.columns
            ],
            "dtypes": {
                str(column): str(data[column].dtype)
                for column in data.columns
            },
        }

        digest = self._new_digest()

        digest.update(
            self._stable_json(metadata)
        )

        if self.options.include_data_values:
            sampled_data = self._sample_data(data)

            hashed_values = pd.util.hash_pandas_object(
                sampled_data,
                index=self.options.include_index,
                categorize=True,
            )

            digest.update(
                hashed_values.to_numpy().tobytes()
            )

        return digest.hexdigest()

    def _schema_fingerprint(
        self,
        *,
        data: pd.DataFrame,
        schema: dict[str, Any],
    ) -> str:
        """Generate a deterministic schema fingerprint."""

        schema_payload = schema or {
            "column_order": [
                str(column)
                for column in data.columns
            ],
            "dtypes": {
                str(column): str(data[column].dtype)
                for column in data.columns
            },
        }

        digest = self._new_digest()

        digest.update(
            self._stable_json(schema_payload)
        )

        return digest.hexdigest()

    def _sample_data(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a deterministic fingerprint sample."""

        if self.options.sample_size is None:
            return data

        if len(data) <= self.options.sample_size:
            return data

        return data.iloc[
            : self.options.sample_size
        ]

    def _new_digest(
        self,
    ) -> Any:
        """Create the configured hash object."""

        return hashlib.new(
            self.options.algorithm
        )

    @staticmethod
    def _stable_json(
        value: object,
    ) -> bytes:
        """Serialize a value deterministically."""

        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
