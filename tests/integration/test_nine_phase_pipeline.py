"""End-to-end tests for the preprocessing and model-handoff pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mlprep.phases import (
    Phase1,
    Phase1Config,
    Phase2,
    Phase2Config,
    Phase3,
    Phase3Config,
    Phase4,
    Phase5,
    Phase6,
    Phase7,
    Phase8,
    Phase9,
)
from mlprep.phases.phase2.output_governance.metadata import SplitStrategyName
from mlprep.types import TaskType


def test_full_pipeline_preserves_protected_partitions_and_handoff(
    tmp_path,
) -> None:
    rng = np.random.default_rng(42)
    row_count = 120
    target = np.tile([0, 1], row_count // 2)
    source = pd.DataFrame(
        {
            "feature_a": rng.normal(
                loc=target * 0.5,
                size=row_count,
            ),
            "feature_b": rng.normal(size=row_count),
            "target": target,
        }
    )
    source_copy = source.copy(deep=True)

    phase1 = Phase1(
        Phase1Config.create(
            task_type=TaskType.BINARY_CLASSIFICATION,
            target_column="target",
            positive_class=1,
            unit_of_observation="One row",
            feature_columns=["feature_a", "feature_b"],
        )
    )
    phase1_result = phase1.run(source)
    phase2 = Phase2(
        Phase2Config.create(
            strategy=SplitStrategyName.STRATIFIED,
            target_column="target",
            feature_columns=["feature_a", "feature_b"],
        )
    )
    phase2_result = phase2.run(
        phase1_result=phase1_result,
        phase1_state=phase1.state,
    )
    assert phase1_result.readiness.ready
    assert phase2_result.readiness is not None
    assert phase2_result.readiness.ready
    phase3 = Phase3(
        Phase3Config.create(
            plot_output_directory=str(tmp_path / "phase3-plots")
        )
    )
    phase3_result = phase3.run(
        phase2_result=phase2_result,
        phase1_state=phase1.state,
        phase2_state=phase2.state,
    )
    assert phase3_result.readiness is not None
    assert phase3_result.readiness.ready
    authorized_test = phase2_result.test.access(
        reason="Authorized synthetic integration test",
        allow=True,
    )
    phase4_result = Phase4().run(
        phase2_result=phase2_result,
        phase3_result=phase3_result,
        phase1_state=phase1.state,
        phase2_state=phase2.state,
        phase3_state=phase3.state,
        authorized_test_partition=authorized_test,
    )
    assert phase4_result.readiness is not None
    assert phase4_result.readiness.ready
    phase5_result = Phase5().run(
        phase4_result=phase4_result,
        authorized_test_partition=phase4_result.test,
    )
    assert phase5_result.readiness is not None
    assert phase5_result.readiness.ready
    phase6_result = Phase6().run(
        phase5_result=phase5_result,
        authorized_test_partition=phase5_result.test,
    )
    assert phase6_result.readiness is not None
    assert phase6_result.readiness.ready
    phase7 = Phase7()
    phase7_result = phase7.run(
        phase6_result=phase6_result,
        authorized_test_partition=phase6_result.test,
    )
    assert phase7_result.readiness is not None
    assert phase7_result.readiness.ready
    phase8 = Phase8()
    phase8_result = phase8.run(
        phase7_result=phase7_result,
        phase7_state=phase7.state,
        authorized_test_partition=phase7_result.test,
    )
    assert phase8_result.readiness is not None
    assert phase8_result.readiness.ready
    phase8_result.to_report_dict()

    handoff = Phase9().run(phase8_result=phase8_result)

    assert source.equals(source_copy)
    for partition_name in ("validation", "test"):
        source_partition = getattr(phase7_result, partition_name)
        balanced_partition = getattr(phase8_result, partition_name)
        assert balanced_partition.X.equals(source_partition.X)
        assert balanced_partition.y.equals(source_partition.y)
        assert balanced_partition.row_ids.equals(source_partition.row_ids)

    assert handoff.readiness.ready
    assert handoff.target_column == "target"
    assert handoff.feature_columns == list(handoff.train.X.columns)
    assert handoff.phase8_state.ready_for_phase_9
    json.dumps(handoff.to_report_dict())
