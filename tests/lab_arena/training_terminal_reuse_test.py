"""Bound terminal privacy work without changing complete exported evidence."""

import base64
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lab_arena import contracts, training_export
from tests.lab_arena.training_export_test import add_call, fixture


def values_for(case):
    # The whole-export hashes bind the original policy as well as gzip metadata.
    # A new production model must not silently change this historical fixture.
    policy = json.loads((Path(__file__).parents[1] / "fixtures" /
                         "training_terminal_policy_6470d6bd8.json").read_text())
    with patch("time.time", return_value=1_760_000_000), patch(
            "lab_arena.scoring.build_scorer_policy", return_value=policy):
        values = fixture()
    add_call(values)
    rows = values["ledger_rows"]
    if case == "credential":
        rows[-1]["terminal_response"]["body_b64"] = base64.b64encode(json.dumps({
            "authorization": "Bearer offline-private-value", "public": "retained",
        }).encode()).decode()
    if case in {"same_prior", "different_prior", "latest_invalid"}:
        earlier = copy.deepcopy(rows[-1])
        earlier["entry_kind"] = "uncertain"
        rows[-1]["entry_id"] = 4
        if case == "different_prior":
            earlier["terminal_response"]["body_b64"] = base64.b64encode(b'{"public":"earlier"}').decode()
        rows.append(earlier)
    if case == "malformed_json":
        rows[-1]["terminal_response"]["body_b64"] = base64.b64encode(b'{bad').decode()
    if case in {"invalid_base64", "latest_invalid"}:
        latest = max(rows, key=lambda row: row["entry_id"])
        latest["terminal_response"]["body_b64"] = "!!!"
    return values


def test_single_terminal_privacy_validation_is_bounded_to_one_pass(monkeypatch):
    values = values_for("normal")
    original = training_export._response_value
    seen = []

    def counted(terminal, provider):
        seen.append(terminal)
        return original(terminal, provider)

    monkeypatch.setattr(training_export, "_response_value", counted)
    result = training_export.assemble_example(**values)
    assert len(seen) == 1
    assert seen[0] is values["ledger_rows"][-1]["terminal_response"]
    call = result["provider_calls"][0]
    receipt = call["response_receipts"][0]
    raw = base64.b64decode(values["ledger_rows"][-1]["terminal_response"]["body_b64"])
    assert receipt["wire_body_sha256"] == contracts.hash_bytes(raw)
    assert receipt["wire_body_bytes"] == len(raw)
    assert receipt["response_sha256"] == contracts.document_hash(call["response"])
    assert receipt["response_ref"] == {"field": "response", "entry_id": 3}
    assert "response" not in receipt


# Canonical whole-export hashes captured on pre-change 6470d6bd8. These bind
# every field, receipt, privacy flag and completeness reason, not only output.
BEFORE_EXPORT_HASHES = {
    "normal": "sha256:e3f20afb7c38c7996b5923edf7a65938e3a1f150c19baf2dcc78bd3d52a978f3",
    "credential": "sha256:8f4b08d9b3fe4aaf1256735f7d90d53e2e98e6797735407ba11b9af7be48f323",
    "same_prior": "sha256:8db6024f33790aff8d35b627390f420ac855884514a518f30145655603bd31f4",
    "different_prior": "sha256:a4920655bec9154330911e61a6f3adeffa1453a68adda836f4854267e51a8974",
    "malformed_json": "sha256:8a79f5e8812f2cf78a5af17105aa955bfe72555dda31c218ff83fa63e3c093d2",
    "invalid_base64": "sha256:8a79f5e8812f2cf78a5af17105aa955bfe72555dda31c218ff83fa63e3c093d2",
    "latest_invalid": "sha256:8705c87b76441ec741427c76b7c75370e1a43d6d32f0b476038a83fe5ba0cb43",
}


@pytest.mark.parametrize("case", ["normal", "credential", "same_prior", "different_prior",
                                 "malformed_json", "invalid_base64", "latest_invalid"])
def test_complete_export_matches_prechange_evidence(case):
    values = values_for(case)
    before_rows = copy.deepcopy(values["ledger_rows"])
    exported = training_export.assemble_example(**values)
    assert contracts.document_hash(exported) == BEFORE_EXPORT_HASHES[case]
    assert values["ledger_rows"] == before_rows
    call = exported["provider_calls"][0]
    if case == "credential":
        assert "offline-private-value" not in json.dumps(exported)
        assert call["response"]["authorization"] == "[REDACTED]"
        assert "credential" in call["response_receipts"][0]["privacy_redactions"]
        assert not exported["privacy"]["exact_replay"]
    if case == "latest_invalid":
        assert call["response_receipts"][-1]["response_unavailable"]
        assert call["response_receipts"][0]["response"]["choices"][0]["message"]["content"] == "result"
    if case == "same_prior":
        assert all(row["response_ref"] == {"field": "response", "entry_id": 4}
                   for row in call["response_receipts"])
