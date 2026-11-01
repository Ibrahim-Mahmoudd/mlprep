from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlprep.phases.phase2.leakage_firewall.post_outcome import (
    PostOutcomeLeakageChecker,
    PostOutcomeLeakageOptions,
)


def _contract(
    feature_columns: list[str],
    outcome_timestamp_column: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        feature_columns=feature_columns,
        phase1_state=SimpleNamespace(
            problem_contract=SimpleNamespace(
                outcome_timestamp_column=outcome_timestamp_column,
            )
        ),
    )


def test_declared_unavailable_and_suspicious_features_emit_distinct_findings() -> None:
    contract = _contract(
        [
            "actual_outcome",
            "inference_only_missing",
            "claim_paid_refund_completed",
            "ordinary_signal",
        ]
    )
    checker = PostOutcomeLeakageChecker(
        PostOutcomeLeakageOptions(
            declared_post_outcome_columns=["actual_outcome"],
            declared_inference_unavailable_columns=[
                "inference_only_missing"
            ],
            block_declared_post_outcome_features=False,
            block_inference_unavailable_features=False,
        )
    )

    findings, suspicious = checker.inspect(contract)
    findings_by_code = {
        code: [
            finding
            for finding in findings
            if finding.code == code
        ]
        for code in {
            "P2-POST-OUTCOME-001",
            "P2-POST-OUTCOME-002",
            "P2-POST-OUTCOME-003",
        }
    }

    assert suspicious == [
        "actual_outcome",
        "claim_paid_refund_completed",
        "inference_only_missing",
    ]
    assert len(findings_by_code["P2-POST-OUTCOME-001"]) == 1
    assert not findings_by_code["P2-POST-OUTCOME-001"][0].blocks_readiness
    assert len(findings_by_code["P2-POST-OUTCOME-002"]) == 1
    assert not findings_by_code["P2-POST-OUTCOME-002"][0].blocks_readiness
    term_findings = findings_by_code["P2-POST-OUTCOME-003"]
    matched_terms = {
        finding.columns[0]: finding.details["matched_terms"]
        for finding in term_findings
    }
    assert matched_terms == {
        "actual_outcome": ["outcome"],
        "claim_paid_refund_completed": [
            "completed",
            "claim_paid",
            "refund_completed",
        ],
    }
    assert all(not finding.blocks_readiness for finding in term_findings)


def test_outcome_timestamp_is_blocked_and_similar_names_are_flagged() -> None:
    outcome_timestamp = "outcome_timestamp"
    similar_name = "outcome_timestam"
    findings, suspicious = PostOutcomeLeakageChecker().inspect(
        _contract(
            [
                outcome_timestamp,
                similar_name,
                "ordinary_signal",
            ],
            outcome_timestamp_column=outcome_timestamp,
        )
    )

    timestamp_finding = next(
        finding
        for finding in findings
        if finding.code == "P2-POST-OUTCOME-004"
    )
    similarity_finding = next(
        finding
        for finding in findings
        if finding.code == "P2-POST-OUTCOME-005"
    )
    assert timestamp_finding.columns == [outcome_timestamp]
    assert timestamp_finding.blocks_readiness
    assert similarity_finding.columns == [
        similar_name,
        outcome_timestamp,
    ]
    assert similarity_finding.details["similarity"] >= 0.9
    assert not similarity_finding.blocks_readiness
    assert suspicious == [similar_name, outcome_timestamp]


def test_disabled_checker_and_invalid_similarity_threshold() -> None:
    assert PostOutcomeLeakageChecker(
        PostOutcomeLeakageOptions(enabled=False)
    ).inspect(
        _contract(["completed_at"], "outcome_at")
    ) == ([], [])

    with pytest.raises(ValueError, match="between zero and one"):
        PostOutcomeLeakageOptions(
            outcome_timestamp_name_similarity_threshold=1.1
        )
