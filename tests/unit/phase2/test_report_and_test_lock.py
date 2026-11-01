from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionMetadata,
    PartitionName,
    Phase2Status,
    SplitFindingCategory,
    SplitFindingSeverity,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.report import (
    Phase2ReportBuilder,
    Phase2ReportOptions,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    Phase2ReadinessDecision,
    Phase2Result,
    SplitFinding,
)
from mlprep.phases.phase2.output_governance.state import (
    LeakageFirewallState,
    PartitionMembership,
    Phase2State,
    SplitColumnState,
)
from mlprep.phases.phase2.output_governance.test_lock import (
    TestLockOptions as LockOptions,
)
from mlprep.phases.phase2.output_governance.test_lock import (
    TestPartitionLocker as PartitionLocker,
)


def _partition(
    name: PartitionName,
    row_ids: list[str],
) -> DatasetPartition:
    index = pd.Index(range(len(row_ids)), name="row")
    return DatasetPartition(
        name=name,
        X=pd.DataFrame(
            {"feature": range(len(row_ids))},
            index=index,
        ),
        y=pd.Series(
            [row % 2 for row in range(len(row_ids))],
            index=index,
            name="target",
        ),
        row_ids=pd.Series(
            row_ids,
            index=index,
            name="row_id",
        ),
        metadata=PartitionMetadata(
            name=name,
            row_count=len(row_ids),
            feature_count=1,
        ),
    )


def _phase2(
    *,
    lock_test: bool = True,
) -> tuple[Phase2Result, Phase2State]:
    state = Phase2State(
        strategy=SplitStrategyName.RANDOM,
        columns=SplitColumnState(
            target_column="target",
            feature_columns=["feature"],
        ),
        leakage_firewall=LeakageFirewallState(
            passed=True,
            row_overlap_checked=True,
            group_overlap_checked=True,
            temporal_order_checked=True,
        ),
        source_dataset_fingerprint="source",
        train_fingerprint="train",
        validation_fingerprint="validation",
        test_fingerprint="test",
        split_fingerprint="split",
        configuration_fingerprint="config",
    )
    train = _partition(PartitionName.TRAIN, ["train-1", "train-2"])
    validation = _partition(PartitionName.VALIDATION, ["validation-1"])
    test = _partition(PartitionName.TEST, ["test-1"])
    if lock_test:
        PartitionLocker().lock(
            test=test,
            state=state,
        )
    result = Phase2Result(
        train=train,
        validation=validation,
        test=test,
        state=state,
        lock_test_set=lock_test,
    )
    result.add_finding(
        SplitFinding(
            code="INFO-001",
            category=SplitFindingCategory.INPUT,
            severity=SplitFindingSeverity.INFO,
            component="split",
            message="Informational split detail.",
        )
    )
    result.add_finding(
        SplitFinding(
            code="WARN-001",
            category=SplitFindingCategory.ROW_OVERLAP,
            severity=SplitFindingSeverity.WARNING,
            component="row_overlap",
            message="No row overlap was found.",
        )
    )
    result.finalize(
        Phase2ReadinessDecision(
            ready=True,
            status=Phase2Status.PASSED,
        )
    )
    return result, state


def test_test_locker_requires_reason_and_valid_token_and_records_access() -> None:
    partition = _partition(PartitionName.TEST, ["test-1"])
    state = Phase2State(
        strategy=SplitStrategyName.RANDOM,
        columns=SplitColumnState(target_column="target"),
    )
    locker = PartitionLocker(
        LockOptions(lock_token_length=24)
    )
    locked, token, findings = locker.lock(
        test=partition,
        state=state,
    )

    assert token is not None
    assert len(token) >= 24
    assert state.test_lock.locked
    assert state.test_lock.lock_token_hash == hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()
    assert token not in str(state.model_dump(mode="json"))
    assert partition.metadata.is_locked
    assert findings[0].code == "P2-TEST-LOCK-001"
    assert findings[0].details == {"token_generated": True}

    with pytest.raises(ValueError, match="reason is required"):
        locker.authorize_access(
            locked_test=locked,
            state=state,
            reason=" ",
            token=token,
        )
    with pytest.raises(PermissionError, match="token is required"):
        locker.authorize_access(
            locked_test=locked,
            state=state,
            reason="final evaluation",
        )
    with pytest.raises(PermissionError, match="token is invalid"):
        locker.authorize_access(
            locked_test=locked,
            state=state,
            reason="final evaluation",
            token="incorrect",
        )

    access = locker.authorize_access(
        locked_test=locked,
        state=state,
        reason="final evaluation",
        token=token,
    )
    access.X.loc[0, "feature"] = 100

    assert access.X.loc[0, "feature"] == 100
    assert partition.X.loc[0, "feature"] == 0
    assert state.test_lock.access_count == 1
    assert state.test_lock.access_reasons == ["final evaluation"]
    assert locked.access_log() == ["final evaluation"]
    assert locked.is_locked


def test_test_locker_disabled_and_tokenless_authorization() -> None:
    partition = _partition(PartitionName.TEST, ["test-1"])
    state = Phase2State(
        strategy=SplitStrategyName.RANDOM,
        columns=SplitColumnState(target_column="target"),
    )
    passthrough, token, findings = PartitionLocker(
        LockOptions(enabled=False)
    ).lock(test=partition, state=state)

    assert passthrough is partition
    assert token is None and findings == []
    assert not state.test_lock.locked
    assert not partition.metadata.is_locked

    locker = PartitionLocker(
        LockOptions(generate_lock_token=False)
    )
    wrapped, token, _ = locker.lock(test=partition, state=state)
    access = locker.authorize_access(
        locked_test=wrapped,
        state=state,
        reason="audited evaluation",
    )

    assert token is None
    assert access is not partition
    assert state.test_lock.lock_token_hash is None
    assert state.test_lock.access_count == 1


def test_phase2_report_filters_findings_fingerprints_and_sensitive_membership(
    tmp_path,
) -> None:
    result, state = _phase2()
    state.register_membership(
        PartitionMembership(
            partition=PartitionName.TRAIN,
            row_ids=["train-1", "train-2"],
            row_count=2,
            fingerprint="train-membership",
        )
    )
    builder = Phase2ReportBuilder(
        Phase2ReportOptions(
            include_info_findings=False,
            include_state=True,
            include_membership_row_ids=False,
            indent=4,
        )
    )

    report = builder.build_dictionary(result=result, state=state)
    full_report = Phase2ReportBuilder(
        Phase2ReportOptions(
            include_state=True,
            include_membership_row_ids=True,
        )
    ).build_dictionary(result=result, state=state)
    json_path = builder.save_json(
        path=tmp_path / "nested" / "phase2.json",
        result=result,
        state=state,
    )
    text_path = builder.save_text(
        path=tmp_path / "nested" / "phase2.txt",
        result=result,
        state=state,
    )
    saved_report = json.loads(json_path.read_text(encoding="utf-8"))
    text = text_path.read_text(encoding="utf-8")

    assert [finding["code"] for finding in report["findings"]] == [
        "WARN-001"
    ]
    assert report["fingerprints"] == {
        "source": "source",
        "train": "train",
        "validation": "validation",
        "test": "test",
        "split": "split",
        "configuration": "config",
    }
    assert report["leakage_firewall"]["passed"]
    assert report["test_lock"]["locked"]
    assert "row_ids" not in report["phase_state"]["memberships"]["train"]
    assert full_report["phase_state"]["memberships"]["train"]["row_ids"] == [
        "train-1",
        "train-2",
    ]
    assert state.get_row_ids(PartitionName.TRAIN) == [
        "train-1",
        "train-2",
    ]
    assert saved_report == report
    assert "Ready for Phase 3: True" in text
    assert "Test rows: 1" in text
    assert json_path.is_file() and json_path.stat().st_size > 0
    assert text_path.is_file() and text_path.stat().st_size > 0


def test_phase2_report_formats_unlocked_test_partition_and_no_readiness() -> None:
    result, state = _phase2(lock_test=False)
    result.readiness = None

    text = Phase2ReportBuilder().build_text(
        result=result,
        state=state,
    )
    report = Phase2ReportBuilder().build_dictionary(
        result=result,
        state=state,
    )

    assert "Ready for Phase 3: False" in text
    assert "Status: not_evaluated" in text
    assert "Test rows: 1" in text
    assert report["readiness"] is None


def test_phase2_report_options_validate_nonnegative_indent() -> None:
    with pytest.raises(ValueError, match="indent cannot be negative"):
        Phase2ReportOptions(indent=-1)
