from __future__ import annotations

import copy

import pytest

from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.qualification_admission_v2 import (
    QUALIFICATION_ADMISSION_INPUT_SCHEMA_VERSION,
    CoordinatorQualificationAdmissionV2,
    QualificationAdmissionV2Error,
)


SALT = "12" * 32
LEAD_A = "11111111-1111-4111-8111-111111111111"
LEAD_B = "22222222-2222-4222-8222-222222222222"
LEAD_C = "33333333-3333-4333-8333-333333333333"


def _row(lead_id, email, hotkey):
    return {
        "lead_id": lead_id,
        "lead_blob": {"email": "stale@example.com", "private_flag": True},
        "lead_blob_hash": "hash-" + lead_id[:8],
        "miner_hotkey": hotkey,
        "email": email,
        "company_name": "Example Co",
    }


def _lead(row):
    blob = dict(row["lead_blob"])
    blob["email"] = row["email"]
    blob["business"] = row["company_name"]
    return {
        "lead_id": row["lead_id"],
        "lead_blob": blob,
        "lead_blob_hash": row["lead_blob_hash"],
        "miner_hotkey": row["miner_hotkey"],
    }


class FakeReader:
    def __init__(self, rows):
        self.rows = {row["lead_id"]: dict(row) for row in rows}
        self.calls = []

    def read(self, *, policy_id, parameters, **_kwargs):
        self.calls.append((policy_id, dict(parameters)))
        if policy_id == "qualification_epoch_assignment":
            return [
                {
                    "payload": {
                        "epoch_id": 100,
                        "assignment": {
                            "assigned_lead_ids": [LEAD_A, LEAD_B, LEAD_C]
                        },
                    }
                }
            ]
        if policy_id == "qualification_leads_by_ids":
            return [dict(self.rows[lead_id]) for lead_id in parameters["lead_ids"]]
        raise AssertionError(policy_id)


def _payload(leads):
    return {
        "schema_version": QUALIFICATION_ADMISSION_INPUT_SCHEMA_VERSION,
        "epoch_id": 100,
        "container_id": 0,
        "container_count": 2,
        "sequence_start": 0,
        "leads": leads,
        "salt_hex": SALT,
    }


def _context():
    return ExecutionContextV2(
        job_id="qualification-admission:100:0",
        purpose="research_lab.admission.v2",
        epoch_id=100,
    )


def test_admission_reconstructs_exact_worker_slice_from_authenticated_rows():
    rows = [
        _row(LEAD_A, "a@example.com", "miner-a"),
        _row(LEAD_B, "b@example.com", "miner-b"),
        _row(LEAD_C, "c@example.com", "miner-c"),
    ]
    reader = FakeReader(rows)
    proposed = [_lead(rows[1]), _lead(rows[0])]

    result = CoordinatorQualificationAdmissionV2(reader).resolve(
        payload=_payload(proposed),
        context=_context(),
    )

    assert result == {
        "epoch_id": 100,
        "container_id": 0,
        "sequence_start": 0,
        "leads": proposed,
        "salt_hex": SALT,
    }
    assert reader.calls == [
        ("qualification_epoch_assignment", {"epoch_id": 100}),
        ("qualification_leads_by_ids", {"lead_ids": [LEAD_A, LEAD_B]}),
    ]


def test_admission_rejects_selective_omission_and_modified_lead_content():
    rows = [
        _row(LEAD_A, "a@example.com", "miner-a"),
        _row(LEAD_B, "b@example.com", "miner-b"),
        _row(LEAD_C, "c@example.com", "miner-c"),
    ]
    source = CoordinatorQualificationAdmissionV2(FakeReader(rows))
    with pytest.raises(QualificationAdmissionV2Error, match="lead set"):
        source.resolve(
            payload=_payload([_lead(rows[0])]),
            context=_context(),
        )

    altered = [_lead(rows[0]), _lead(rows[1])]
    altered = copy.deepcopy(altered)
    altered[1]["lead_blob"]["email"] = "attacker@example.com"
    with pytest.raises(QualificationAdmissionV2Error, match="authenticated source"):
        source.resolve(payload=_payload(altered), context=_context())


def test_admission_rejects_sequence_or_assignment_ambiguity():
    rows = [
        _row(LEAD_A, "a@example.com", "miner-a"),
        _row(LEAD_B, "b@example.com", "miner-b"),
        _row(LEAD_C, "c@example.com", "miner-c"),
    ]
    payload = _payload([_lead(rows[0]), _lead(rows[1])])
    payload["sequence_start"] = 1
    with pytest.raises(QualificationAdmissionV2Error, match="sequence"):
        CoordinatorQualificationAdmissionV2(FakeReader(rows)).resolve(
            payload=payload,
            context=_context(),
        )

    class AmbiguousReader(FakeReader):
        def read(self, *, policy_id, parameters, **kwargs):
            values = super().read(
                policy_id=policy_id,
                parameters=parameters,
                **kwargs,
            )
            return values + values if policy_id == "qualification_epoch_assignment" else values

    with pytest.raises(QualificationAdmissionV2Error, match="ambiguous"):
        CoordinatorQualificationAdmissionV2(AmbiguousReader(rows)).resolve(
            payload=_payload([_lead(rows[0]), _lead(rows[1])]),
            context=_context(),
        )
