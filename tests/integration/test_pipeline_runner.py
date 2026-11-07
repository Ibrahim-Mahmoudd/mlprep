"""End-to-end tests for the unified pipeline runner."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from mlprep.data_sources import DataFrameSource
from mlprep.phases import (
    Phase1Config,
    Phase2Config,
    Phase3Config,
    Phase4Config,
    Phase5Config,
    Phase6Config,
    Phase7Config,
    Phase8Config,
    Phase9Config,
)
from mlprep.phases.phase2.output_governance.metadata import SplitStrategyName
from mlprep.pipeline import (
    MLPrepPipeline,
    MLPrepPipelineConfig,
    PipelineConfigLoader,
    PipelineReadinessError,
    PipelineResumeError,
)
from mlprep.types import TaskType


def _pipeline_config(
    *,
    output_directory=None,
    plots: bool = False,
) -> MLPrepPipelineConfig:
    return MLPrepPipelineConfig(
        phase1=Phase1Config.create(
            task_type=TaskType.BINARY_CLASSIFICATION,
            target_column="target",
            positive_class=1,
            unit_of_observation="One row",
            feature_columns=["feature_a", "feature_b"],
        ),
        phase2=Phase2Config.create(
            strategy=SplitStrategyName.STRATIFIED,
            target_column="target",
            feature_columns=["feature_a", "feature_b"],
            random_seed=42,
        ),
        phase3=Phase3Config.create(
            generate_plots=plots,
            plot_output_directory="should-not-be-used",
        ),
        phase4=Phase4Config(),
        phase5=Phase5Config(),
        phase6=Phase6Config(),
        phase7=Phase7Config(),
        phase8=Phase8Config(),
        phase9=Phase9Config(),
        output_directory=output_directory,
    )


def _source_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    target = np.tile([0, 1], 60)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(loc=target * 0.5, size=len(target)),
            "feature_b": rng.normal(size=len(target)),
            "target": target,
        }
    )


def test_pipeline_runs_all_phases_and_persists_atomic_outputs(
    tmp_path,
) -> None:
    config = _pipeline_config(
        output_directory=tmp_path / "run",
        plots=True,
    )
    source = _source_data()
    source_copy = source.copy(deep=True)
    original_plot_directory = (
        config.phase3.visualizations.output_directory
    )
    pipeline = MLPrepPipeline(config)

    handoff = pipeline.run(data=source)

    assert handoff.readiness.ready
    assert pipeline.manifest is not None
    assert pipeline.manifest.status == "completed"
    assert [record.phase for record in pipeline.manifest.phases] == [
        f"phase{number}" for number in range(1, 10)
    ]
    assert all(record.readiness_passed for record in pipeline.manifest.phases)
    assert pipeline.state.last_completed_phase == "phase8"
    assert pipeline.result is not None
    assert pipeline.result.handoff is handoff
    assert source.equals(source_copy)
    assert (tmp_path / "run" / "checkpoint.json").is_file()
    assert (tmp_path / "run" / "execution-report.json").is_file()
    assert (
        config.phase3.visualizations.output_directory
        == original_plot_directory
    )
    phase3_config = pipeline._phase3_config(
        copy.deepcopy(config),
        dry_run=False,
    )
    assert phase3_config.visualizations.enabled
    assert phase3_config.visualizations.output_directory == str(
        tmp_path / "run" / "phase3" / "plots"
    )
    assert not (tmp_path / "artifacts").exists()
    json.loads(
        (tmp_path / "run" / "execution-report.json").read_text(
            encoding="utf-8"
        )
    )

    original_execution_id = pipeline.manifest.execution_id
    resumed_handoff = pipeline.run(data=source, resume=True)
    assert resumed_handoff.readiness.ready
    assert pipeline.manifest is not None
    assert pipeline.manifest.execution_id == original_execution_id

    changed_source = source.copy(deep=True)
    changed_source.loc[0, "feature_a"] += 1
    with pytest.raises(PipelineResumeError, match="Source data fingerprint"):
        pipeline.run(data=changed_source, resume=True)


def test_pipeline_resume_rejects_configuration_mismatch(tmp_path) -> None:
    output_directory = tmp_path / "run"
    source = _source_data()
    MLPrepPipeline(
        _pipeline_config(output_directory=output_directory)
    ).run(data=source)

    changed_config = _pipeline_config(output_directory=output_directory)
    changed_config.phase3.execution.enabled = False
    with pytest.raises(
        PipelineResumeError,
        match="Configuration fingerprint",
    ):
        MLPrepPipeline(changed_config).run(data=source, resume=True)


def test_pipeline_resume_rejects_corrupt_checkpoint(tmp_path) -> None:
    output_directory = tmp_path / "run"
    source = _source_data()
    MLPrepPipeline(
        _pipeline_config(output_directory=output_directory)
    ).run(data=source)
    (output_directory / "checkpoint.json").write_text(
        "{not-json",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        MLPrepPipeline(
            _pipeline_config(output_directory=output_directory)
        ).run(data=source, resume=True)


def test_pipeline_resume_requires_checkpoint_and_rejects_dry_run() -> None:
    source = _source_data()
    with pytest.raises(PipelineResumeError, match="explicit output_directory"):
        MLPrepPipeline(_pipeline_config()).run(data=source, resume=True)

    with pytest.raises(ValueError, match="cannot be combined"):
        MLPrepPipeline(_pipeline_config()).run(
            data=source,
            dry_run=True,
            resume=True,
        )


def test_pipeline_accepts_unified_dataframe_source_contract() -> None:
    config = _pipeline_config()
    source = DataFrameSource(_source_data())

    handoff = MLPrepPipeline(config).run(data=source)

    assert handoff.readiness.ready


def test_pipeline_stops_when_phase_one_is_not_ready() -> None:
    config = _pipeline_config()
    config.phase1.execution.enabled = False
    pipeline = MLPrepPipeline(config)

    with pytest.raises(PipelineReadinessError, match="phase1 blocked"):
        pipeline.run(data=_source_data())

    assert pipeline.manifest is not None
    assert pipeline.manifest.status == "blocked"
    assert [record.phase for record in pipeline.manifest.phases] == ["phase1"]


def test_dry_run_does_not_create_outputs(tmp_path) -> None:
    pipeline = MLPrepPipeline(
        _pipeline_config(output_directory=tmp_path / "dry-run", plots=True)
    )

    handoff = pipeline.run(data=_source_data(), dry_run=True)

    assert handoff.readiness.ready
    assert pipeline.manifest is not None
    assert pipeline.manifest.dry_run
    assert not (tmp_path / "dry-run").exists()
    assert not (tmp_path / "artifacts").exists()


def test_execution_id_is_deterministic_for_same_data_and_config() -> None:
    config = _pipeline_config()
    source = _source_data()

    first = MLPrepPipeline(config)
    second = MLPrepPipeline(config)
    first.run(data=source, dry_run=True)
    second.run(data=source.copy(deep=True), dry_run=True)

    assert first.manifest is not None
    assert second.manifest is not None
    assert first.manifest.execution_id == second.manifest.execution_id


def test_pipeline_config_loader_rejects_unknown_root_keys() -> None:
    with pytest.raises(ValueError, match="Unknown pipeline configuration keys"):
        PipelineConfigLoader.from_dict(
            {
                "phase1": {},
                "phase2": {},
                "unknown": True,
            }
        )


def test_pipeline_config_loader_builds_phase_configs() -> None:
    config = _pipeline_config()

    loaded = PipelineConfigLoader.from_dict(
        {
            "phase1": config.phase1.to_dict(),
            "phase2": config.phase2.to_dict(),
        }
    )

    assert isinstance(loaded, MLPrepPipelineConfig)
    assert loaded.phase1.problem_contract.target_column == "target"
    assert loaded.phase2.split_roles.target_column == "target"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_pipeline_config_loader_reads_supported_file_formats(
    tmp_path,
    suffix: str,
) -> None:
    config = _pipeline_config()
    values = {
        "phase1": config.phase1.to_dict(),
        "phase2": config.phase2.to_dict(),
    }
    content = (
        json.dumps(values)
        if suffix == ".json"
        else yaml.safe_dump(values, sort_keys=False)
    )
    path = tmp_path / f"pipeline{suffix}"
    path.write_text(content, encoding="utf-8")

    loaded = PipelineConfigLoader.load(path)

    assert loaded.phase1.problem_contract.target_column == "target"
    assert loaded.phase2.split_roles.target_column == "target"
