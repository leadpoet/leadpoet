import copy
import json
import time

import pytest

from lab_arena import contracts
from lab_arena.collection_research_capture import (
    REQUIRED_APPROVALS,
    ResearchGrant,
    ResearchObservations,
    ResearchWebEgress,
)
from lab_arena.service import LocalObjectStore
from lab_arena.web_egress import WebEgressError


def licensed(tmp_path):
    job = {
        "job_id": "test-job",
        "batch_id": "grant-test",
        "registration_id": "00000000-0000-4000-8000-000000000021",
        "miner_hotkey": "5" + "A" * 47,
        "owner_coldkey": "5" + "B" * 47,
        "terms_sha256": "sha256:" + "1" * 64,
        "scenario_id": "research-test",
        "model": "fixture/open",
        "task_scope": "licensed_research",
        "deepline_key_id": "delegated",
    }
    case = {"icp": {"company": "Synthetic example"}}
    doc = {
        **{
            k: v
            for k, v in job.items()
            if k not in {"job_id", "task_scope", "deepline_key_id"}
        },
        "schema": "collection.research-grant.v1",
        "icp_sha256": contracts.document_hash(case["icp"]),
        "approvals": dict.fromkeys(REQUIRED_APPROVALS, "sha256:" + "a" * 64),
        "hosts": ["example.com"],
        "expires_at": int(time.time()) + 60,
        "deepline_requests": [],
    }
    grant = ResearchGrant(doc, contracts.document_hash(doc))
    return grant, job, case, LocalObjectStore(tmp_path)


def test_rights_and_signed_funding_are_checked_before_private_io(tmp_path):
    grant, job, case, obj = licensed(tmp_path)
    for k, v in (
        ("task_scope", "synthetic_corrections"),
        ("registration_id", "other"),
        ("terms_sha256", "new"),
    ):
        with pytest.raises(ValueError, match="grant does not cover"):
            grant.bind(dict(job, **{k: v}), case, obj)
    with pytest.raises(ValueError, match="grant does not cover"):
        grant.bind(job, {"icp": {}}, obj)
    with pytest.raises(ValueError, match="unapproved"):
        grant.provider_request(
            "deepline.execute", {"tool": "send_email", "payload": {}}
        )
    with pytest.raises(ValueError, match="hosted search"):
        grant.provider_request(
            "openrouter.responses", {"tools": [{"type": "openrouter:web_search"}]}
        )
    assert not list(tmp_path.rglob("*.json"))
    grant.document["expires_at"] = 0
    with pytest.raises(ValueError, match="expired"):
        grant.bind(job, case, obj)


def test_full_observed_page_persisted_before_provider_or_worker_failure(tmp_path):
    grant, job, case, obj = licensed(tmp_path)
    archive = grant.bind(job, case, obj)
    text = "Late section with evidence.\n" * 1000
    digest = contracts.hash_bytes(text.encode())
    event = {
        "id": 0,
        "kind": "web_fetch",
        "observed_at": "2026-11-24T00:01:00Z",
        "after_action_sequence": 7,
        "depends_on": [],
        "data": {
            "url": "https://example.com/a",
            "final_url": "https://example.com/a",
            "retrieved_at": "2026-11-24T00:00:59Z",
            "status": 200,
            "content_type": "text/plain",
            "body": text,
            "body_sha256": digest,
            "body_bytes": len(text.encode()),
            "body_ref": None,
            "stored_body_bytes": None,
            "stored_body_sha256": None,
        },
        "privacy": {"redacted": False, "redactions": []},
    }
    buffer = ResearchObservations(archive.observation)
    assert buffer.append_agent_observation(event) == 0
    assert buffer.append_agent_observation({**event, "id": 1}) == 1
    # The generic buffer deduplicates equal payloads. Both durable documents
    # still prove exactly which body this request observed, even after a crash.
    refs = sorted(tmp_path.rglob("*.json"))
    assert len(refs) == 2
    for idx, f in enumerate(refs):
        d = json.loads(f.read_text())
        assert d["data"]["data"]["body"] == text and d["data"]["id"] == idx
        assert d["grant_sha256"] == grant.sha256 and d["job_id"] == job["job_id"]
        assert d["authority"] == "sandbox_http" and d["sequence"] == 7
    bad = copy.deepcopy(event)
    bad["id"] = 2
    bad["data"]["final_url"] = "https://elsewhere.com/a"
    with pytest.raises(ValueError, match="unlicensed"):
        buffer.append_agent_observation(bad)
    assert len(list(tmp_path.rglob("*.json"))) == 2
    assert "capture_incomplete" in buffer.failures


def test_research_egress_rejects_unlicensed_destinations_before_dns(tmp_path):
    calls = []
    guard = ResearchWebEgress(
        "/tmp/collection-unit-test.sock",
        allowed_hosts=["example.com"],
        expires_at=time.time() + 50,
        resolver=lambda *a, **kw: calls.append(a),
    )
    for host, port in (("evil.example.com", 443), ("example.com", 80)):
        with pytest.raises(WebEgressError, match="licensed"):
            guard._resolved_ip(host, port)
    assert calls == []


def test_authorized_deepline_result_keeps_exact_search_data_on_partial_run(tmp_path):
    from types import SimpleNamespace
    from lab_arena import collection_research as research
    from unittest.mock import patch

    grant, job, case, objects = licensed(tmp_path)
    params = {"tool": "exa_search", "payload": {"query": "Fictional account"}}
    grant_doc = dict(
        grant.document, deepline_requests=[contracts.document_hash(params)]
    )
    grant = ResearchGrant(grant_doc, contracts.document_hash(grant_doc))
    journal = grant.bind(job, case, objects)
    result = {
        "status": "completed",
        "result": {
            "hits": [
                {"url": "https://example.com/proof", "text": "long excerpt " * 3000}
            ]
        },
    }
    calls = []

    def execute(identity, request):
        calls.append(identity)
        assert request == params
        return result

    broker = SimpleNamespace(job=job, catalog=object(), deepline=execute)
    frame = {
        "action_sequence": 7,
        "operation_id": "deepline.execute",
        "parameters": params,
        "timeout_ms": 5000,
    }
    # Catalog validation is exercised separately. This test covers the actual
    # gateway relay after it obtains validated request parameters.
    with patch.object(
        research.operations,
        "validate_operation_request",
        side_effect=lambda op, p, **kwargs: p,
    ):
        research._public_call(frame, broker, {}, journal=journal)
        with pytest.raises(ValueError, match="unapproved"):
            research._public_call(
                dict(
                    frame,
                    action_sequence=8,
                    parameters={**params, "payload": {"query": "changed"}},
                ),
                broker,
                {},
                journal=journal,
            )
    assert len(calls) == 1
    frames = [json.loads(f.read_text()) for f in sorted(tmp_path.rglob("*.json"))]
    assert [f["authority"] for f in frames] == ["gateway_request", "gateway_result"]
    assert frames[1]["data"]["response"] == result and frames[1]["sequence"] == 7
    assert frames[1]["data"]["request_id"] == "generate-research-7"
