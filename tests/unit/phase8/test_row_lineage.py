import pandas as pd
import pytest

from mlprep.phases.phase8.lineage.row_lineage import (
    RowLineageTracker,
)
from mlprep.phases.phase8.samplers.base import (
    RemovedRowRecord,
    ResampledRowType,
    SamplerResult,
    SamplerRowRecord,
    SamplerState,
)


def _result(
    *,
    row_lineage: list[SamplerRowRecord] | None = None,
    removed_rows: list[RemovedRowRecord] | None = None,
) -> SamplerResult:
    index = pd.RangeIndex(2)
    return SamplerResult(
        features=pd.DataFrame({"feature": [1.0, 2.0]}, index=index),
        target=pd.Series([0, 1], index=index, name="target"),
        row_ids=pd.Series(["row-0", "row-1"], index=index, name="row_id"),
        row_lineage=row_lineage
        or [
            SamplerRowRecord(
                output_position=position,
                output_row_id=f"row-{position}",
                row_type=ResampledRowType.ORIGINAL,
                class_value=class_value,
                source_row_ids=[f"row-{position}"],
                source_positions=[position],
                original=True,
            )
            for position, class_value in enumerate([0, 1])
        ],
        removed_rows=removed_rows or [],
        state=SamplerState(
            sampler="test",
            strategy="identity",
            fitted=True,
            fitted_on_training_only=True,
        ),
    )


def _tracker() -> RowLineageTracker:
    return RowLineageTracker(strategy="identity")


def test_row_lineage_builds_complete_state_and_supports_lookup() -> None:
    tracker = _tracker()
    state = tracker.build(
        source_features=pd.DataFrame({"feature": [1.0, 2.0]}),
        source_target=pd.Series([0, 1], name="target"),
        source_row_ids=pd.Series(["row-0", "row-1"], name="row_id"),
        sampler_result=_result(),
    )

    assert state.complete_output_coverage
    assert state.complete_removed_row_coverage
    assert state.original_ids_preserved
    assert state.retained_original_row_ids == ["row-0", "row-1"]
    assert tracker.get_state() == state
    assert tracker.output_entry("row-0").original
    with pytest.raises(KeyError, match="unavailable"):
        tracker.output_entry("missing")
    with pytest.raises(RuntimeError, match="has not been built"):
        RowLineageTracker(strategy="other").get_state()


def test_removed_row_lineage_is_fingerprinted_and_lookupable() -> None:
    removed = RemovedRowRecord(
        row_id="row-1",
        source_position=1,
        class_value=1,
        reason="undersampled",
        stage="sampling",
    )
    result = _result(
        row_lineage=[
            SamplerRowRecord(
                output_position=0,
                output_row_id="row-0",
                row_type=ResampledRowType.ORIGINAL,
                class_value=0,
                source_row_ids=["row-0"],
                source_positions=[0],
                original=True,
            )
        ],
        removed_rows=[removed],
    )
    result.features = result.features.iloc[:1]
    result.target = result.target.iloc[:1]
    result.row_ids = result.row_ids.iloc[:1]
    tracker = _tracker()
    state = tracker.build(
        source_features=pd.DataFrame({"feature": [1.0, 2.0]}),
        source_target=pd.Series([0, 1]),
        source_row_ids=pd.Series(["row-0", "row-1"]),
        sampler_result=result,
    )

    assert state.removed_original_row_ids == ["row-1"]
    assert tracker.removed_entry("row-1").source_feature_fingerprint
    with pytest.raises(KeyError, match="unavailable"):
        tracker.removed_entry("missing")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"features": "bad"}, "source_features"),
        ({"target": "bad"}, "source_target"),
        ({"row_ids": "bad"}, "source_row_ids"),
    ],
)
def test_row_lineage_rejects_invalid_source_types(
    source: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "source_features": pd.DataFrame({"feature": [1.0, 2.0]}),
        "source_target": pd.Series([0, 1]),
        "source_row_ids": pd.Series(["row-0", "row-1"]),
    }
    values.update(
        {
            f"source_{key}": value
            for key, value in source.items()
        }
    )
    with pytest.raises(TypeError, match=message):
        _tracker().build(sampler_result=_result(), **values)


def test_row_lineage_rejects_invalid_source_shape_and_sampler_result() -> None:
    with pytest.raises(ValueError, match="different row counts"):
        _tracker().build(
            source_features=pd.DataFrame({"feature": [1.0]}),
            source_target=pd.Series([0, 1]),
            source_row_ids=pd.Series(["row-0", "row-1"]),
            sampler_result=_result(),
        )
    with pytest.raises(TypeError, match="sampler_result"):
        _tracker().build(
            source_features=pd.DataFrame({"feature": [1.0, 2.0]}),
            source_target=pd.Series([0, 1]),
            source_row_ids=pd.Series(["row-0", "row-1"]),
            sampler_result=object(),  # type: ignore[arg-type]
        )
