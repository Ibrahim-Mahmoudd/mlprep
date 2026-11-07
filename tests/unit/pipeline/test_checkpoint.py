import json

import pytest

from mlprep.pipeline.checkpoint import (
    PipelineCheckpoint,
    PipelineCheckpointData,
    write_execution_report,
)
from mlprep.pipeline.execution_manifest import PipelineExecutionManifest


def _snapshot() -> PipelineCheckpointData:
    return PipelineCheckpointData(
        source_fingerprint="source",
        configuration_fingerprint="configuration",
        phase_fingerprints={"phase1": "fingerprint"},
        pipeline_state={"rows": 2},
    )


def test_checkpoint_round_trip_and_missing_file(tmp_path) -> None:
    checkpoint = PipelineCheckpoint(tmp_path / "nested" / "checkpoint.json")

    assert checkpoint.load() is None
    checkpoint.save(_snapshot())

    restored = checkpoint.load()
    assert restored is not None
    assert restored.pipeline_state == {"rows": 2}
    assert restored.phase_fingerprints == {"phase1": "fingerprint"}


def test_checkpoint_rejects_unsupported_schema_and_extra_fields(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps(
            {
                **_snapshot().model_dump(mode="json"),
                "schema_version": "2",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported pipeline checkpoint"):
        PipelineCheckpoint(path).load()

    path.write_text(
        json.dumps({**_snapshot().model_dump(mode="json"), "unexpected": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        PipelineCheckpoint(path).load()


def test_execution_report_is_written_atomically(tmp_path) -> None:
    path = tmp_path / "reports" / "execution.json"
    manifest = PipelineExecutionManifest(
        execution_id="execution",
        source_fingerprint="source",
        configuration_fingerprint="configuration",
    )

    write_execution_report(path, manifest)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["execution_id"] == "execution"
    assert not list(path.parent.glob(".*.tmp"))
