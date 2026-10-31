"""Deterministic fingerprints and execution manifest records."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class PhaseExecutionRecord(BaseModel):
    """Non-sensitive summary of one phase execution."""

    model_config = ConfigDict(extra="forbid")

    phase: str
    status: str
    readiness_passed: bool
    output_fingerprint: str
    row_counts: dict[str, int] = Field(default_factory=dict)


class PipelineExecutionManifest(BaseModel):
    """Stable identifiers and outcomes for a pipeline execution."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    schema_version: str = "1"
    execution_id: str
    source_fingerprint: str
    configuration_fingerprint: str
    status: str = "running"
    dry_run: bool = False
    phases: list[PhaseExecutionRecord] = Field(default_factory=list)

    @staticmethod
    def fingerprint_configuration(config: object) -> str:
        """Fingerprint a resolved configuration without exposing its values."""

        serialized = json.dumps(
            _json_compatible(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def dataframe_fingerprint(data: pd.DataFrame) -> str:
        """Fingerprint DataFrame schema, values, index, and object identities."""

        digest = hashlib.sha256()
        schema = {
            "columns": [_typed_token(column) for column in data.columns],
            "dtypes": [str(dtype) for dtype in data.dtypes],
            "index_type": type(data.index).__qualname__,
            "index_dtype": str(data.index.dtype),
            "index_names": [_typed_token(name) for name in data.index.names],
        }
        digest.update(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        hashed_values = pd.util.hash_pandas_object(
            data,
            index=True,
            categorize=True,
        )
        digest.update(hashed_values.to_numpy(copy=False).tobytes())

        for column in data.columns:
            if data[column].dtype == object:
                _update_typed_values(digest, data[column].array)
        if data.index.dtype == object:
            _update_typed_values(digest, data.index)
        return digest.hexdigest()

    @classmethod
    def execution_id_for(
        cls,
        data: pd.DataFrame,
        config: object,
    ) -> tuple[str, str, str]:
        """Return deterministic execution, source, and configuration fingerprints."""

        source_fingerprint = cls.dataframe_fingerprint(data)
        config_fingerprint = cls.fingerprint_configuration(config)
        execution_id = hashlib.sha256(
            (
                "mlprep-execution-v1:"
                f"{source_fingerprint}:{config_fingerprint}"
            ).encode()
        ).hexdigest()
        return execution_id, source_fingerprint, config_fingerprint


def _update_typed_values(
    digest: Any,
    values: Iterable[Any],
) -> None:
    for value in values:
        digest.update(_typed_token(value).encode("utf-8"))
        digest.update(b"\0")


def _typed_token(value: Any) -> str:
    value_type = type(value)
    return (
        f"{value_type.__module__}.{value_type.__qualname__}:"
        f"{value!r}"
    )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_compatible(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if isinstance(value, (Path, date, datetime)):
        return value.isoformat() if isinstance(value, (date, datetime)) else str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return _json_compatible(value.item())
    raise TypeError(
        f"Configuration value of type {type(value).__name__} "
        "cannot be fingerprinted."
    )
