"""Source attribution is scoped to the judged execution, never miner text."""
from copy import deepcopy
import base64
import json

import pytest

from lab_arena import contact_evidence, contact_policy, operations, scoring, verify


def contact():
    return {"full_name": "Jane Doe", "role": "CEO", "linkedin_url": "https://www.linkedin.com/in/jane-doe/",
            "location": {"country": "US"}, "email": "jane@example.com",
            "email_source": {"provider": "harvestapi", "tool": "harvestapi_get_profile", "broker_call_id": "sha256:abc"}}


class Store:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def list_ledger(self, **kwargs):
        self.calls.append(kwargs)
        return self.rows


def ledger():
    body = {"status": "completed", "result": {"data": {"element": {"id": "profile-1", "email": "jane@example.com"}}}}
    scope = {"run_id": "execution-1", "call_identity": "sha256:abc", "provider": "deepline", "operation_id": "deepline.execute"}
    return [
        {**scope, "entry_kind": "reservation", "entry_doc": {"tool": "harvestapi_get_profile"}},
        {**scope, "entry_kind": "settlement", "terminal_response": {
            "status": 200, "body_b64": base64.b64encode(json.dumps(body).encode()).decode()}, "created_at": "2026-09-11T12:00:00Z"},
    ]


def test_scoped_broker_evidence_is_gateway_constructed():
    store = Store(ledger())
    row = {"contact": contact(), "source_evidence": {"response": "forged"}}
    result = contact_evidence.resolve_sources(store, {"run_id": "execution-1"}, [row])
    assert store.calls == [{"run_id": "execution-1", "call_identity": "sha256:abc"}]
    assert result["sha256:abc"]["response"]["result"]["data"]["element"]["id"] == "profile-1"
    assert "forged" not in json.dumps(result)


@pytest.mark.parametrize("mutation", ["other_run", "other_tool", "no_settlement", "http_error", "bad_body"])
def test_invalid_broker_reference_cannot_fall_back_to_a_fresh_lookup(mutation):
    rows = ledger()
    if mutation == "other_run":
        rows[0]["run_id"] = "other-execution"
    elif mutation == "other_tool":
        rows[0]["entry_doc"]["tool"] = "exa_search"
    elif mutation == "no_settlement":
        rows.pop()
    elif mutation == "http_error":
        rows[1]["terminal_response"]["status"] = 500
    else:
        rows[1]["terminal_response"]["body_b64"] = "%%%"
    result = contact_evidence.resolve_sources(Store(rows), {"run_id": "execution-1"}, [{"contact": contact()}])
    assert result["sha256:abc"] == contact_evidence.INVALID_REFERENCE


def test_record_reference_requests_a_trusted_profile_refetch():
    claim = contact()
    claim["email_source"].pop("broker_call_id")
    claim["email_source"]["record_id"] = "profile-1"
    store = Store([])
    result = contact_evidence.resolve_sources(store, {"run_id": "execution-1"}, [{"contact": claim}])
    assert store.calls == []
    assert result["profile-1"] == {"provider": "harvestapi", "tool": "harvestapi_get_profile", "input": {"url": claim["linkedin_url"], "findEmail": "true"}}


@pytest.mark.parametrize("tool,payload", [
    ("harvestapi_get_profile", {"url": "https://www.linkedin.com/in/jane-doe/", "findEmail": "true"}),
    ("zerobounce_validate", {"email": "jane@example.com"}),
    ("bounceban_verify_single", {"email": "jane@example.com"}),
    ("bounceban_get_single_status", {"id": "request-1"}),
])
def test_actual_contact_tool_request_shape(tool, payload):
    outgoing = operations.build_outbound_request("deepline.execute", {"tool": tool, "payload": payload})
    assert outgoing.url == f"https://code.deepline.com/api/v2/integrations/{tool}/execute"
    assert json.loads(outgoing.body) == {"provider": tool.split("_", 1)[0], "operation": tool, "payload": payload}


def test_email_verification_does_not_allow_user_webhooks():
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("deepline.execute", {"tool": "bounceban_verify_single", "payload": {"email": "jane@example.com", "url": "https://example.org/hook"}})


def test_failed_contact_cannot_create_score_or_qualification_credit():
    row = {"contact_qualified": False, "contact_identity_key": "person:1", "email_status": "unknown",
           "contact_verification": {"decision": "unverified"}, "final_score": 0.0, "company_qualified": False}
    contact_policy.validate_contact_breakdown(row)
    for field, value in (("final_score", 60.0), ("company_qualified", True)):
        with pytest.raises(ValueError):
            contact_policy.validate_contact_breakdown({**row, field: value})


def test_public_contact_verdict_drops_raw_provider_data():
    row = {"final_score": 0.0, "contact_qualified": False, "email_status": "invalid", "contact_identity_key": "person:1",
           "contact_verification": {"decision": "mismatch", "reason": "contact_email_invalid", "response": "private-raw-data", "subchecks": {
               "email_verification": {"status": "fail", "reason": "invalid", "response": "private-raw-data"}, "extra": {"status": "private-raw-data"}}}}
    redacted = verify.redact_breakdown(row)
    assert "private-raw-data" not in json.dumps(redacted)
    assert redacted["contact_verification"]["subchecks"]["email_verification"] == {"status": "fail", "reason": "invalid"}
    assert verify.redact_breakdown(redacted) == redacted


def test_verified_contact_cannot_rescue_an_unqualified_company():
    checks = {key: {"status": "pass"} for key in (
        "claim", "identity", "source", "company", "role", "location",
        "email_attribution", "email_verification",
    )}
    row = {"contact_qualified": True, "contact_identity_key": "person:1", "email_status": "valid",
           "contact_verification": {"decision": "verified", "subchecks": checks},
           "final_score": 100.0, "company_qualified": True}
    contact_policy.validate_contact_breakdown(row)
    with pytest.raises(ValueError, match="company and contact qualification"):
        contact_policy.validate_contact_breakdown({**row, "company_qualified": False})
