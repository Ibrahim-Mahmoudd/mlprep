from __future__ import annotations

import json

import pandas as pd

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase2.output_governance.result import DatasetPartition
from mlprep.phases.phase4.output_governance.report import (
    Phase4ReportBuilder,
    Phase4ReportOptions,
)
from mlprep.phases.phase4.output_governance.result import Phase4Result
from mlprep.phases.phase4.output_governance.state import Phase4State
from mlprep.phases.phase5.output_governance.report import (
    Phase5ReportBuilder,
    Phase5ReportOptions,
)
from mlprep.phases.phase5.output_governance.result import Phase5Result
from mlprep.phases.phase5.output_governance.state import Phase5State
from mlprep.phases.phase6.output_governance.report import (
    Phase6ReportBuilder,
    Phase6ReportOptions,
)
from mlprep.phases.phase6.output_governance.result import Phase6Result
from mlprep.phases.phase6.output_governance.state import Phase6State
from mlprep.phases.phase7.output_governance.report import (
    Phase7ReportBuilder,
    Phase7ReportOptions,
)
from mlprep.phases.phase7.output_governance.result import Phase7Result
from mlprep.phases.phase7.output_governance.state import Phase7State
from mlprep.phases.phase8.output_governance.report import (
    Phase8ReportBuilder,
    Phase8ReportOptions,
)
from mlprep.phases.phase8.output_governance.result import Phase8Result
from mlprep.phases.phase8.output_governance.state import Phase8State


def _partition(name: PartitionName) -> DatasetPartition:
    index = pd.Index([10, 11], name="row")
    return DatasetPartition(
        name=name,
        X=pd.DataFrame({"feature": [1.0, 2.0]}, index=index),
        y=pd.Series([0, 1], index=index, name="target"),
        row_ids=pd.Series(
            [f"{name.value}-a", f"{name.value}-b"],
            index=index,
            name="row_id",
        ),
    )


def _partitions() -> tuple[DatasetPartition, DatasetPartition, DatasetPartition]:
    return (
        _partition(PartitionName.TRAIN),
        _partition(PartitionName.VALIDATION),
        _partition(PartitionName.TEST),
    )


def test_phase4_report_exports_governance_and_omits_disabled_details(
    tmp_path,
) -> None:
    train, validation, test = _partitions()
    result = Phase4Result(train=train, validation=validation, test=test)
    state = Phase4State(
        imputed_value_count=2,
        indicator_mapping={"income": "income_missing"},
    )
    builder = Phase4ReportBuilder(
        Phase4ReportOptions(
            include_phase_state=True,
            include_validation_details=False,
            include_execution_metadata=False,
            include_fingerprints=False,
            expose_learned_values=False,
        )
    )

    report = builder.build_dictionary(result=result, state=state)

    assert report["phase"] == "phase_4"
    assert report["partition_rows"] == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    assert report["governance"]["fitted_on_training_only"] is True
    assert report["indicator_mapping"] == {"income": "income_missing"}
    assert report["imputer_states"] == {}
    assert "validation" not in report
    assert "execution_metadata" not in report
    assert "fingerprints" not in report
    assert report["phase_state"]["imputed_value_count"] == 2
    assert "Imputed values: 2" in builder.build_text(
        result=result,
        state=state,
    )

    json_path = builder.save_json(
        path=tmp_path / "phase4" / "report.json",
        result=result,
        state=state,
    )
    text_path = builder.save_text(
        path=tmp_path / "phase4" / "report.txt",
        result=result,
        state=state,
    )
    assert json.loads(json_path.read_text(encoding="utf-8"))["phase"] == "phase_4"
    assert "Phase 4 Report" in text_path.read_text(encoding="utf-8")


def test_phase5_report_exports_governance_and_can_redact_detector_bounds(
    tmp_path,
) -> None:
    train, validation, test = _partitions()
    result = Phase5Result(train=train, validation=validation, test=test)
    state = Phase5State(
        total_detected_outlier_count=3,
        indicator_mapping={"income": "income_outlier"},
        detector_states={
            "income": {
                "lower_bound": -1.0,
                "upper_bound": 10.0,
                "median": 4.0,
                "fitted": True,
            }
        },
    )
    builder = Phase5ReportBuilder(
        Phase5ReportOptions(
            include_phase_state=True,
            include_validation_details=False,
            include_execution_metadata=False,
            include_feature_treatments=False,
            expose_fitted_bounds=False,
        )
    )
    report = builder.build_dictionary(result=result, state=state)

    assert report["phase"] == "phase_5"
    assert report["governance"]["fitted_on_training_only"] is True
    assert report["treatment_summary"]["detected_outliers"] == 3
    assert report["detector_states"]["income"] == {
        "fitted": True,
    }
    assert "validation" not in report
    assert "fingerprints" in report
    assert "Detected outliers: 3" in builder.build_text(
        result=result,
        state=state,
    )
    output = builder.save_json(
        path=tmp_path / "phase5" / "report.json",
        result=result,
        state=state,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["phase"] == "phase_5"


def test_phase6_report_summarizes_features_and_controls_sensitive_details(
    tmp_path,
) -> None:
    train, validation, test = _partitions()
    result = Phase6Result(train=train, validation=validation, test=test)
    state = Phase6State(
        source_feature_columns=["income"],
        output_feature_columns=["income_scaled", "region_a"],
        timedelta_output_columns={"duration": "duration_days"},
        total_unknown_category_count=1,
    )
    builder = Phase6ReportBuilder(
        Phase6ReportOptions(
            include_phase_state=True,
            include_validation_details=False,
            include_execution_metadata=False,
            include_feature_states=False,
            include_feature_lineage=False,
            expose_category_vocabularies=False,
            expose_numeric_parameters=False,
        )
    )
    report = builder.build_dictionary(result=result, state=state)

    assert report["phase"] == "phase_6"
    assert report["feature_summary"]["source_feature_count"] == 1
    assert report["feature_summary"]["output_feature_count"] == 2
    assert report["feature_summary"]["unknown_category_count"] == 1
    assert report["output_columns"] == ["income_scaled", "region_a"]
    assert "numeric_states" not in report
    assert "feature_lineage" not in report
    assert report["phase_state"]["output_feature_columns"] == [
        "income_scaled",
        "region_a",
    ]
    assert "Unknown categories: 1" in builder.build_text(
        result=result,
        state=state,
    )
    output = builder.save_json(
        path=tmp_path / "phase6" / "report.json",
        result=result,
        state=state,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["phase"] == "phase_6"


def test_phase7_report_summarizes_feature_decisions_and_redacts_scores(
    tmp_path,
) -> None:
    train, validation, test = _partitions()
    result = Phase7Result(train=train, validation=validation, test=test)
    state = Phase7State(
        source_feature_columns=["income", "region"],
        selected_feature_columns=["income"],
        removed_feature_columns=["region"],
        protected_features=["income"],
    )
    builder = Phase7ReportBuilder(
        Phase7ReportOptions(
            include_phase_state=True,
            include_validation_details=False,
            include_execution_metadata=False,
            include_feature_decisions=False,
            include_filter_details=False,
            include_selector_details=False,
            include_stability_details=False,
            include_feature_lineage=False,
            expose_selector_scores=False,
        )
    )
    report = builder.build_dictionary(result=result, state=state)

    assert report["phase"] == "phase_7"
    assert report["selection_summary"]["selected_features"] == ["income"]
    assert report["selection_summary"]["removed_features"] == ["region"]
    assert report["governance"]["fitted_on_training_only"] is True
    assert "feature_decisions" not in report
    assert "selected_lineage" not in report
    assert report["phase_state"]["protected_features"] == ["income"]
    assert "Selected features: 1" in builder.build_text(
        result=result,
        state=state,
    )
    output = builder.save_json(
        path=tmp_path / "phase7" / "report.json",
        result=result,
        state=state,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["phase"] == "phase_7"


def test_phase8_report_limits_lineage_and_includes_model_weights(
    tmp_path,
) -> None:
    train, validation, test = _partitions()
    result = Phase8Result(train=train, validation=validation, test=test)
    result.class_weights = {"class-key": 2.0}
    result.sample_weights = pd.Series([1.0, 2.0], name="sample_weight")
    state = Phase8State(
        configured_strategy="random_oversampling",
        executed_strategy="random_oversampling",
        source_training_row_count=2,
        output_training_row_count=3,
        synthetic_row_ids=["synthetic-1"],
    )
    builder = Phase8ReportBuilder(
        Phase8ReportOptions(
            include_phase_state=True,
            include_validation_details=False,
            include_execution_metadata=False,
            include_distribution_details=False,
            include_sampler_state=False,
            include_sample_weights=True,
            include_row_lineage=False,
            include_feature_lineage=False,
            maximum_reported_rows=1,
        )
    )
    report = builder.build_dictionary(result=result, state=state)

    assert report["phase"] == "phase_8"
    assert report["balancing_summary"]["synthetic_rows"] == 1
    assert report["class_weight_state"] is None
    assert report["sample_weight_state"] is None
    assert report["sample_weight_count"] == 2
    assert report["governance"]["validation_remained_protected"] is True
    assert "distribution" not in report
    assert "sampler_state" not in report
    assert "row_lineage" not in report
    assert report["phase_state"]["synthetic_row_ids"] == ["synthetic-1"]
    assert "Synthetic rows: 1" in builder.build_text(
        result=result,
        state=state,
    )
    output = builder.save_json(
        path=tmp_path / "phase8" / "report.json",
        result=result,
        state=state,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["phase"] == "phase_8"
