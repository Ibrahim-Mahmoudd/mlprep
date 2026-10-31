"""Pipeline execution results and explicit readiness failures."""

from __future__ import annotations

from dataclasses import dataclass

from mlprep.phases.phase9.output_governance.result import Phase9Result
from mlprep.pipeline.execution_manifest import PipelineExecutionManifest
from mlprep.pipeline.state import MLPrepPipelineState


@dataclass(slots=True)
class PipelineExecutionResult:
    """Detailed execution data retained by an MLPrepPipeline instance."""

    handoff: Phase9Result
    state: MLPrepPipelineState
    manifest: PipelineExecutionManifest


class PipelineReadinessError(RuntimeError):
    """Raised when a phase blocks safe continuation of the pipeline."""

    def __init__(self, phase: str, reasons: list[str]) -> None:
        self.phase = phase
        self.reasons = tuple(reasons)
        reason_text = "; ".join(reasons) or "readiness decision was not ready"
        super().__init__(
            f"{phase} blocked the pipeline: {reason_text}"
        )


class PipelineResumeError(ValueError):
    """Raised when a checkpoint cannot safely be replayed."""
