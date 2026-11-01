from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from mlprep.phases.phase5.contract.treatment_plan import (
    DetectorKind,
    TreatmentKind,
    TreatmentPlanBuilder,
    TreatmentPlanEntry,
    TreatmentPlanOptions,
    TreatmentPlanStatus,
)


def _contract(frame: pd.DataFrame) -> SimpleNamespace:
    columns = list(frame.columns)
    return SimpleNamespace(
        train=SimpleNamespace(X=frame),
        numeric_columns=columns,
        feature_columns=columns,
    )


def test_builder_creates_ordered_entries_applies_overrides_and_indicators() -> None:
    frame = pd.DataFrame(
        {
            "iqr": [1.0, 2.0, 3.0, 4.0],
            "domain": [0.0, 1.0, 2.0, 3.0],
            "set_missing": [1.0, 2.0, 3.0, 4.0],
            "report": [1.0, 2.0, 3.0, 4.0],
            "low_unique": [1.0, 1.0, 1.0, 1.0],
            "excluded": [1.0, 2.0, 3.0, 4.0],
        }
    )
    plan = TreatmentPlanBuilder(
        TreatmentPlanOptions(
            detector_overrides={"domain": DetectorKind.DOMAIN},
            treatment_overrides={
                "set_missing": TreatmentKind.SET_MISSING,
                "report": TreatmentKind.REPORT_ONLY,
            },
            indicator_overrides={"report": False},
            domain_bounds={"domain": (0.0, 10.0)},
            set_missing_fallbacks={"set_missing": 0.0},
            priority_overrides={"domain": 0.9},
            allow_set_missing=True,
            allow_report_only=True,
            excluded_columns=["excluded"],
        )
    ).build(contract=_contract(frame))

    assert list(plan.entries) == ["iqr", "domain", "set_missing", "report"]
    assert plan.entry("iqr").detector == DetectorKind.IQR
    assert plan.entry("domain").domain_minimum == 0.0
    assert plan.entry("domain").domain_maximum == 10.0
    assert plan.entry("domain").priority == 0.9
    assert plan.entry("domain").requires_training_fit is False
    assert plan.entry("set_missing").set_missing_fallback == 0.0
    assert plan.entry("set_missing").status == TreatmentPlanStatus.PLANNED
    assert plan.entry("report").treatment == TreatmentKind.REPORT_ONLY
    assert plan.entry("report").status == TreatmentPlanStatus.PLANNED
    assert plan.indicator_columns == {
        "iqr": "iqr__was_outlier",
        "domain": "domain__was_outlier",
        "set_missing": "set_missing__was_outlier",
    }
    assert plan.report_only_columns == ["report"]
    assert plan.skipped_columns == ["low_unique", "excluded"]
    assert plan.blocked_columns == []
    assert plan.metadata == {
        "enabled": True,
        "entry_count": 4,
        "fitting_entry_count": 3,
        "indicator_count": 3,
        "report_only_count": 1,
        "blocked_count": 0,
    }


def test_unsafe_treatments_and_unconfigured_domain_are_blocked() -> None:
    frame = pd.DataFrame(
        {
            "missing_fallback": [1.0, 2.0, 3.0],
            "disabled_missing": [1.0, 2.0, 3.0],
            "report": [1.0, 2.0, 3.0],
            "domain": [1.0, 2.0, 3.0],
        }
    )
    plan = TreatmentPlanBuilder(
        TreatmentPlanOptions(
            treatment_overrides={
                "missing_fallback": TreatmentKind.SET_MISSING,
                "disabled_missing": TreatmentKind.SET_MISSING,
                "report": TreatmentKind.REPORT_ONLY,
            },
            detector_overrides={"domain": DetectorKind.DOMAIN},
            allow_report_only=False,
        )
    ).build(contract=_contract(frame))

    assert plan.blocked_columns == [
        "missing_fallback",
        "disabled_missing",
        "report",
        "domain",
    ]
    assert all(
        entry.status == TreatmentPlanStatus.BLOCKED
        for entry in plan.entries.values()
    )
    assert plan.entry("missing_fallback").warnings == [
        "set_missing treatment is disabled.",
        "set_missing requires an explicit fallback.",
    ]
    assert plan.entry("disabled_missing").warnings == [
        "set_missing treatment is disabled.",
        "set_missing requires an explicit fallback.",
    ]
    assert plan.entry("report").warnings == [
        "report_only treatment is disabled."
    ]
    assert plan.entry("domain").warnings == [
        "Domain detection requires explicitly configured feature bounds."
    ]
    assert plan.executable_entries() == []
    assert plan.fitting_entries() == []


def test_indicator_name_collisions_fail_instead_of_overwriting_features() -> None:
    frame = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0],
            "amount__was_outlier": [0.0, 0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="Outlier-indicator name collision"):
        TreatmentPlanBuilder().build(contract=_contract(frame))


def test_disabled_builder_and_training_fit_queries() -> None:
    frame = pd.DataFrame(
        {
            "fitted": [1.0, 2.0, 3.0],
            "domain": [1.0, 2.0, 3.0],
        }
    )
    disabled = TreatmentPlanBuilder(
        TreatmentPlanOptions(enabled=False)
    ).build(contract=_contract(frame))
    plan = TreatmentPlanBuilder(
        TreatmentPlanOptions(
            detector_overrides={"domain": DetectorKind.DOMAIN},
            domain_bounds={"domain": (None, 5.0)},
        )
    ).build(contract=_contract(frame))

    assert disabled.entries == {}
    assert disabled.metadata == {"enabled": False}
    assert [entry.column for entry in plan.executable_entries()] == [
        "fitted",
        "domain",
    ]
    assert [entry.column for entry in plan.fitting_entries()] == ["fitted"]
    with pytest.raises(KeyError, match="missing"):
        plan.entry("missing")


def test_treatment_entries_validate_domain_and_set_missing_requirements() -> None:
    with pytest.raises(ValueError, match="column cannot be empty"):
        TreatmentPlanEntry(
            column=" ",
            detector=DetectorKind.IQR,
            treatment=TreatmentKind.CLIP,
            reason="invalid column",
        )
    with pytest.raises(ValueError, match="indicator_suffix"):
        TreatmentPlanEntry(
            column="feature",
            detector=DetectorKind.IQR,
            treatment=TreatmentKind.CLIP,
            indicator_suffix="",
            reason="invalid suffix",
        )
    with pytest.raises(ValueError, match="at least one configured bound"):
        TreatmentPlanEntry(
            column="feature",
            detector=DetectorKind.DOMAIN,
            treatment=TreatmentKind.CLIP,
            reason="missing domain bounds",
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        TreatmentPlanEntry(
            column="feature",
            detector=DetectorKind.DOMAIN,
            treatment=TreatmentKind.CLIP,
            domain_minimum=10.0,
            domain_maximum=0.0,
            reason="reversed domain bounds",
        )
    with pytest.raises(ValueError, match="explicit fallback"):
        TreatmentPlanEntry(
            column="feature",
            detector=DetectorKind.IQR,
            treatment=TreatmentKind.SET_MISSING,
            reason="missing fallback",
        )


def test_options_normalize_enum_values_and_reject_invalid_settings() -> None:
    options = TreatmentPlanOptions(
        detector_overrides={"value": "domain"},
        treatment_overrides={"value": "winsorize"},
    )

    assert options.detector_overrides["value"] == DetectorKind.DOMAIN
    assert options.treatment_overrides["value"] == TreatmentKind.WINSORIZE

    with pytest.raises(ValueError, match="minimum_unique_count"):
        TreatmentPlanOptions(minimum_unique_count=1)
    with pytest.raises(ValueError, match="indicator_suffix"):
        TreatmentPlanOptions(indicator_suffix="")
    with pytest.raises(ValueError, match="between zero and one"):
        TreatmentPlanOptions(priority_overrides={"value": 1.1})
    with pytest.raises(ValueError, match="contain two values"):
        TreatmentPlanOptions(domain_bounds={"value": (0.0,)})
    with pytest.raises(ValueError, match="Invalid domain bounds"):
        TreatmentPlanOptions(domain_bounds={"value": (10.0, 0.0)})
