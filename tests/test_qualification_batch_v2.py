from __future__ import annotations

import copy

import pytest

from leadpoet_canonical.qualification_batch_v2 import (
    QualificationBatchV2Error,
    build_qualification_batch_output_v2,
    validate_qualification_batch_output_v2,
)


SALT = "ab" * 32


def _leads():
    return [
        {
            "lead_id": "lead-approve",
            "miner_hotkey": "miner-a",
            "lead_blob": {"email": "a@example.com"},
        },
        {
            "lead_id": "lead-deny",
            "miner_hotkey": "miner-b",
            "lead_blob": {"email": "b@example.com"},
        },
        {
            "lead_id": "lead-skip",
            "miner_hotkey": "miner-c",
            "lead_blob": {"email": "c@example.com"},
        },
    ]


def _results():
    return [
        (
            True,
            {
                "rep_score": {"total_score": 40},
                "is_icp_multiplier": 20.0,
                "company_linkedin_data": {"cache": "not-hashed"},
            },
        ),
        (
            False,
            {
                "rep_score": {"total_score": 48},
                "is_icp_multiplier": 5.0,
                "rejection_reason": {"message": "denied"},
            },
        ),
        (None, {"is_icp_multiplier": 0.0, "skipped": True}),
    ]


def test_batch_output_matches_legacy_decision_and_score_semantics_exactly():
    output = build_qualification_batch_output_v2(
        epoch_id=100,
        container_id=3,
        sequence_start=20,
        leads=_leads(),
        batch_results=_results(),
        salt_hex=SALT,
    )
    validate_qualification_batch_output_v2(output)
    assert [row["decision"] for row in output["validation_results"]] == [
        "approve",
        "deny",
        "deny",
    ]
    assert [row["rep_score"] for row in output["validation_results"]] == [40, 0, 0]
    assert output["validation_results"][0]["decision_hash"] == (
        "e7dffa5bf737c67d15bb7aa3e83114cf8797fcac73b5e5ce5ad82e379dcc55a4"
    )
    assert output["validation_results"][0]["rep_score_hash"] == (
        "dcca4e631cc05c248587b5154d44c3564a9b080526356df7cd3df02f0501e23f"
    )
    assert [row["effective_rep_score"] for row in output["sourcing_decisions"]] == [
        60,
        0,
        0,
    ]
    assert output["sourcing_decisions"][0]["sequence"] == 20
    assert output["sourcing_decisions"][-1]["sequence"] == 22
    assert output["validation_results"][2]["evidence_blob"]["skipped"] is True


def test_batch_output_rejects_tampering_and_cardinality_drift():
    output = build_qualification_batch_output_v2(
        epoch_id=100,
        container_id=0,
        sequence_start=0,
        leads=_leads(),
        batch_results=_results(),
        salt_hex=SALT,
    )
    tampered = copy.deepcopy(output)
    tampered["sourcing_decisions"][0]["effective_rep_score"] = 999
    with pytest.raises(QualificationBatchV2Error):
        validate_qualification_batch_output_v2(tampered)
    with pytest.raises(QualificationBatchV2Error, match="count"):
        build_qualification_batch_output_v2(
            epoch_id=100,
            container_id=0,
            sequence_start=0,
            leads=_leads(),
            batch_results=_results()[:-1],
            salt_hex=SALT,
        )
