"""Orchestration for the complete nine-phase MLPrep workflow."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pandas as pd

from mlprep.phases import (
    Phase1,
    Phase2,
    Phase3,
    Phase3Config,
    Phase4,
    Phase5,
    Phase6,
    Phase7,
    Phase8,
    Phase9,
    Phase9Result,
)
from mlprep.pipeline.checkpoint import (
    PipelineCheckpoint,
    PipelineCheckpointData,
    write_execution_report,
)
from mlprep.pipeline.config import MLPrepPipelineConfig
from mlprep.pipeline.execution_manifest import (
    PhaseExecutionRecord,
    PipelineExecutionManifest,
)
from mlprep.pipeline.result import (
    PipelineExecutionResult,
    PipelineReadinessError,
    PipelineResumeError,
)
from mlprep.pipeline.state import MLPrepPipelineState


class MLPrepPipeline:
    """Run Phases 1 through 9 and return the model-ready Phase 9 handoff."""

    def __init__(self, config: MLPrepPipelineConfig) -> None:
        if not isinstance(config, MLPrepPipelineConfig):
            raise TypeError(
                "config must be an instance of MLPrepPipelineConfig."
            )
        self.config = config
        self.state = MLPrepPipelineState()
        self.manifest: PipelineExecutionManifest | None = None
        self.result: PipelineExecutionResult | None = None

    def run(
        self,
        *,
        data: pd.DataFrame,
        dry_run: bool = False,
        resume: bool = False,
    ) -> Phase9Result:
        """Execute all phases; dry runs suppress every persistent output."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(dry_run, bool) or not isinstance(resume, bool):
            raise TypeError("dry_run and resume must be booleans.")

        execution_id, source_fingerprint, config_fingerprint = (
            PipelineExecutionManifest.execution_id_for(data, self.config)
        )
        run_config = copy.deepcopy(self.config)
        self.state = MLPrepPipelineState()
        self.result = None
        self.manifest = PipelineExecutionManifest(
            execution_id=execution_id,
            source_fingerprint=source_fingerprint,
            configuration_fingerprint=config_fingerprint,
            dry_run=dry_run,
        )

        checkpoint = self._checkpoint()
        prior_checkpoint: PipelineCheckpointData | None = None
        if resume:
            if dry_run:
                raise ValueError("dry_run and resume cannot be combined.")
            if checkpoint is None:
                raise PipelineResumeError(
                    "Resume requires an explicit output_directory and "
                    "save_intermediate_states=True."
                )
            prior_checkpoint = checkpoint.load()
            if prior_checkpoint is None:
                raise PipelineResumeError(
                    f"No checkpoint exists at {checkpoint.path}."
                )
            if prior_checkpoint.source_fingerprint != source_fingerprint:
                raise PipelineResumeError(
                    "Source data fingerprint differs from the checkpoint."
                )
            if (
                prior_checkpoint.configuration_fingerprint
                != config_fingerprint
            ):
                raise PipelineResumeError(
                    "Configuration fingerprint differs from the checkpoint."
                )

        phase_fingerprints: dict[str, str] = {}

        phase1 = Phase1(run_config.phase1)
        phase1_result = phase1.run(data)
        self._advance(
            phase="phase1",
            result=phase1_result,
            state=phase1.state,
            output_fingerprint=self._phase_fingerprint(
                "phase1", phase1_result, phase1.state
            ),
            prior_checkpoint=prior_checkpoint,
            phase_fingerprints=phase_fingerprints,
        )

        phase2 = Phase2(run_config.phase2)
        phase2_result = phase2.run(
            phase1_result=phase1_result,
            phase1_state=phase1.state,
        )
        self._advance(
            phase="phase2",
            result=phase2_result,
            state=phase2.state,
            output_fingerprint=self._phase_fingerprint(
                "phase2", phase2_result, phase2.state
            ),
            prior_checkpoint=prior_checkpoint,
            phase_fingerprints=phase_fingerprints,
        )

        phase3_config = self._phase3_config(
            run_config,
            dry_run=dry_run,
        )
        phase3 = Phase3(phase3_config)
        phase3_result = phase3.run(
            phase2_result=phase2_result,
            phase1_state=phase1.state,
            phase2_state=phase2.state,
        )
        self._advance(
            phase="phase3",
            result=phase3_result,
            state=phase3.state,
            output_fingerprint=self._phase_fingerprint(
                "phase3", phase3_result, phase3.state
            ),
            prior_checkpoint=prior_checkpoint,
            phase_fingerprints=phase_fingerprints,
        )

        authorized_test = phase2_result.test.access(
            reason="MLPrep pipeline Phase 4 transform",
            allow=True,
        )
        phase4 = Phase4(run_config.phase4)
        phase4_result = phase4.run(
            phase2_result=phase2_result,
            phase3_result=phase3_result,
            phase1_state=phase1.state,
            phase2_state=phase2.state,
            phase3_state=phase3.state,
            authorized_test_partition=authorized_test,
        )
        self._advance(
            phase="phase4",
            result=phase4_result,
            state=phase4.state,
            output_fingerprint=self._phase_fingerprint(
                "phase4", phase4_result, phase4.state
            ),
            prior_checkpoint=prior_checkpoint,
            phase_fingerprints=phase_fingerprints,
        )

        phase5 = Phase5(run_config.phase5)
        phase5_result = phase5.run(
            phase4_result=phase4_result,
            phase4_state=phase4.state,
            authorized_test_partition=phase4_result.test,
        )
        self._advance(
            phase="phase5",
            result=phase5_result,
            state=phase5.state,
            output_fingerprint=self._phase_fingerprint(
                "phase5", phase5_result, phase5.state
            ),
            prior_checkpoint=prior_checkpoint,
            phase_fingerprints=phase_fingerprints,
        )

        phase6 = Phase6(run_config.phase6)
        phase6_result = phase6.run(
            phase5_result=phase5_result,
            phase5_state=phase5.state,
            authorized_test_partition=phase5_result.test,
        )
        self._advance(
            phase="phase6",
            result=phase6_result,
            state=phase6.state,
            output_fingerprint=self._phase_fingerprint(
                "phase6", phase6_result, phase6.state
            ),
            prior_checkpoint=prior_checkpoint,
            phase_fingerprints=phase_fingerprints,
        )

        phase7 = Phase7(run_config.phase7)
        phase7_result = phase7.run(
            phase6_result=phase6_result,
            phase6_state=phase6.state,
            authorized_test_partition=phase6_result.test,
        )
        self._advance(
            phase="phase7",
            result=phase7_result,
            state=phase7.state,
            output_fingerprint=self._phase_fingerprint(
                "phase7", phase7_result, phase7.state
            ),
            prior_checkpoint=prior_checkpoint,
            phase_fingerprints=phase_fingerprints,
        )

        phase8 = Phase8(run_config.phase8)
        phase8_result = phase8.run(
            phase7_result=phase7_result,
            phase7_state=phase7.state,
            authorized_test_partition=phase7_result.test,
        )
        self._advance(
            phase="phase8",
            result=phase8_result,
            state=phase8.state,
            output_fingerprint=self._phase_fingerprint(
                "phase8", phase8_result, phase8.state
            ),
            prior_checkpoint=prior_checkpoint,
            phase_fingerprints=phase_fingerprints,
        )

        phase9 = Phase9(run_config.phase9)
        handoff = phase9.run(
            phase8_result=phase8_result,
            phase8_state=phase8.state,
        )
        self._advance(
            phase="phase9",
            result=handoff,
            state=phase8.state,
            output_fingerprint=self._phase_fingerprint(
                "phase9", handoff, phase8.state
            ),
            prior_checkpoint=prior_checkpoint,
            phase_fingerprints=phase_fingerprints,
        )

        self.manifest.status = "completed"
        self.result = PipelineExecutionResult(
            handoff=handoff,
            state=self.state,
            manifest=self.manifest,
        )
        self._persist_report(dry_run=dry_run)
        return handoff

    def _phase3_config(
        self,
        config: MLPrepPipelineConfig,
        *,
        dry_run: bool,
    ) -> Phase3Config:
        phase3_config = config.phase3
        output_directory = config.output_directory
        if dry_run or output_directory is None:
            phase3_config.visualizations.enabled = False
        else:
            phase3_config.visualizations.output_directory = str(
                output_directory / "phase3" / "plots"
            )
        return phase3_config

    def _checkpoint(self) -> PipelineCheckpoint | None:
        if (
            self.config.output_directory is None
            or not self.config.save_intermediate_states
        ):
            return None
        return PipelineCheckpoint(
            self.config.output_directory / "checkpoint.json"
        )

    def _advance(
        self,
        *,
        phase: str,
        result: Any,
        state: Any,
        output_fingerprint: str,
        prior_checkpoint: PipelineCheckpointData | None,
        phase_fingerprints: dict[str, str],
    ) -> None:
        readiness = getattr(result, "readiness", None)
        readiness_passed = bool(
            readiness is not None and readiness.ready
        )
        row_counts = _row_counts(result)
        record = PhaseExecutionRecord(
            phase=phase,
            status="completed" if readiness_passed else "blocked",
            readiness_passed=readiness_passed,
            output_fingerprint=output_fingerprint,
            row_counts=row_counts,
        )
        assert self.manifest is not None
        self.manifest.phases.append(record)

        if phase != "phase9" and state is not None:
            setattr(
                self.state,
                f"{phase}_state",
                state,
            )
        if not readiness_passed:
            self.manifest.status = "blocked"
            self._persist_report(dry_run=self.manifest.dry_run)
            reasons = (
                list(readiness.reasons)
                if readiness is not None
                else ["phase did not return a readiness decision"]
            )
            raise PipelineReadinessError(phase, reasons)

        if phase != "phase9":
            self.state.last_completed_phase = phase
        phase_fingerprints[phase] = output_fingerprint

        if prior_checkpoint is not None:
            expected_fingerprint = prior_checkpoint.phase_fingerprints.get(
                phase
            )
            if (
                expected_fingerprint is not None
                and expected_fingerprint != output_fingerprint
            ):
                raise PipelineResumeError(
                    f"Replayed {phase} output fingerprint differs "
                    "from the checkpoint."
                )

        checkpoint = self._checkpoint()
        if checkpoint is not None and not self.manifest.dry_run:
            checkpoint.save(
                PipelineCheckpointData(
                    source_fingerprint=self.manifest.source_fingerprint,
                    configuration_fingerprint=(
                        self.manifest.configuration_fingerprint
                    ),
                    phase_fingerprints=dict(phase_fingerprints),
                    pipeline_state=self.state.model_dump(mode="json"),
                )
            )

    def _persist_report(self, *, dry_run: bool) -> None:
        if self.config.output_directory is None or dry_run:
            return
        if self.manifest is None:
            return
        write_execution_report(
            self.config.output_directory / "execution-report.json",
            self.manifest,
        )

    @staticmethod
    def _phase_fingerprint(
        phase: str,
        result: Any,
        state: Any,
    ) -> str:
        for attribute in (
            "output_fingerprint",
            "analysis_fingerprint",
            "dataset_fingerprint",
            "split_fingerprint",
        ):
            fingerprint = getattr(state, attribute, None)
            if isinstance(fingerprint, str) and fingerprint:
                return fingerprint

        partition_fingerprints = _partition_fingerprints(result)
        if partition_fingerprints:
            payload = json.dumps(
                partition_fingerprints,
                sort_keys=True,
                separators=(",", ":"),
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        data = getattr(result, "data", None)
        if isinstance(data, pd.DataFrame):
            return PipelineExecutionManifest.dataframe_fingerprint(data)

        stable_state = {}
        for attribute in (
            "training_fingerprint_after",
            "validation_fingerprint_after",
            "configuration_fingerprint",
            "fitted_state_fingerprint",
        ):
            value = getattr(state, attribute, None)
            if isinstance(value, str) and value:
                stable_state[attribute] = value
        if stable_state:
            payload = json.dumps(stable_state, sort_keys=True)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        raise RuntimeError(
            f"Cannot determine a stable output fingerprint for {phase}."
        )


def _partition_fingerprints(result: Any) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for name in ("train", "validation", "test"):
        partition = getattr(result, name, None)
        if partition is None or not all(
            hasattr(partition, attribute)
            for attribute in ("X", "y", "row_ids")
        ):
            continue
        fingerprints[name] = hashlib.sha256(
            (
                PipelineExecutionManifest.dataframe_fingerprint(
                    partition.X
                )
                + PipelineExecutionManifest.dataframe_fingerprint(
                    partition.y.to_frame()
                )
                + PipelineExecutionManifest.dataframe_fingerprint(
                    partition.row_ids.to_frame()
                )
            ).encode("utf-8")
        ).hexdigest()
    return fingerprints


def _row_counts(result: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in ("train", "validation", "test"):
        partition = getattr(result, name, None)
        row_count = getattr(partition, "row_count", None)
        if callable(row_count):
            counts[name] = int(row_count())
        elif isinstance(row_count, int):
            counts[name] = row_count
    data = getattr(result, "data", None)
    if isinstance(data, pd.DataFrame):
        counts["rows"] = len(data)
    return counts
