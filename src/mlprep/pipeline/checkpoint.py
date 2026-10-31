"""Atomic, JSON-only pipeline checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.pipeline.execution_manifest import PipelineExecutionManifest


class PipelineCheckpointData(BaseModel):
    """Checkpoint contents safe to parse without object deserialization."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    source_fingerprint: str
    configuration_fingerprint: str
    phase_fingerprints: dict[str, str] = Field(default_factory=dict)
    pipeline_state: dict[str, Any]


class PipelineCheckpoint:
    """Read and atomically write one pipeline checkpoint file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> PipelineCheckpointData | None:
        """Load the checkpoint envelope for safe, fingerprint-checked replay."""

        if not self.path.exists():
            return None
        snapshot = PipelineCheckpointData.model_validate_json(
            self.path.read_text(encoding="utf-8")
        )
        if snapshot.schema_version != "1":
            raise ValueError(
                "Unsupported pipeline checkpoint schema version: "
                f"{snapshot.schema_version!r}."
            )
        return snapshot

    def save(self, snapshot: PipelineCheckpointData) -> None:
        """Atomically persist a JSON checkpoint."""

        _atomic_write_json(self.path, snapshot.model_dump(mode="json"))


def write_execution_report(
    path: str | Path,
    manifest: PipelineExecutionManifest,
) -> None:
    """Atomically persist the non-sensitive execution manifest."""

    _atomic_write_json(path, manifest.model_dump(mode="json"))


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                payload,
                temporary_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
