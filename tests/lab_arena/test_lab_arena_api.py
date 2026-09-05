"""HTTP surface of /arena/v1 (labarena.md section 14): routing, headers,
body limits, and generic error shapes, with the service stubbed."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts
from lab_arena.api import MAX_JSON_BODY_BYTES, create_app
from lab_arena.service import ServiceError


class StubStore:
    def get_submission(self, submission_id):
        return {
            "status": "accepted",
            "rejection_rule": None,
            "source_ref": "arena/arena-2026-09-02/sources/sub-1.tar.gz",
            "source_size_bytes": 100,
        } if submission_id == "sub-1" else None


class StubService:
    def __init__(self) -> None:
        self.calls: Dict[str, Any] = {}
        self.store = StubStore()

    def public_current(self):
        return {"mode": "shadow", "round": None, "king": None}

    def signing_key_document(self):
        return {"schema_version": contracts.SIGNING_KEY_DOCUMENT_SCHEMA_VERSION, "public_key_hash": contracts.document_hash("k")}

    def public_reward_basis(self, epoch):
        return {"round_id": "arena-2026-09-02", "effective_reward_epoch": epoch} if epoch == 24801 else None

    def public_round(self, round_id):
        if round_id != "arena-2026-09-02":
            raise ServiceError("round_missing", 404)
        return {"round_id": round_id, "status": "open"}

    def public_benchmark(self, round_id):
        raise ServiceError("benchmark_not_public", 403)

    def public_results(self, round_id, submission_id):
        return {"round_id": round_id, "submission_id": submission_id}

    def handle_submission_presign(self, envelope):
        self.calls["submission_presign"] = envelope
        return {"status": "upload_ready", "submission_id": "sub-1"}

    def handle_submission_finalize(self, submission_id, envelope):
        self.calls["submission_finalize"] = (submission_id, envelope)
        return {"status": "accepted", "submission_id": submission_id}

    def handle_claim(self, envelope):
        self.calls["claim"] = envelope
        return {"status": "no_pending"}

    def handle_provider(self, run_id, lease_token, frame):
        self.calls["provider"] = (run_id, lease_token, frame)
        try:
            asyncio.get_running_loop()
            self.calls["provider_has_event_loop"] = True
        except RuntimeError:
            self.calls["provider_has_event_loop"] = False
        return {"status": 200, "headers": {}, "body_b64": "e30=", "call": {}}

    def handle_source(self, run_id, lease_token):
        self.calls["source"] = (run_id, lease_token)
        return b"source archive"

    def handle_complete(self, envelope):
        if envelope.get("bad"):
            raise ServiceError("run_result_invalid", 400)
        return {"status": "accepted"}


@pytest.fixture()
def client():
    service = StubService()
    app = create_app(service)
    return TestClient(app), service


def test_public_routes(client):
    http, _service = client
    assert http.get("/arena/v1/current").json()["mode"] == "shadow"
    assert http.get("/arena/v1/signing-key").json()["schema_version"] == contracts.SIGNING_KEY_DOCUMENT_SCHEMA_VERSION
    assert http.get("/arena/v1/recipient").status_code == 404
    assert http.get("/arena/v1/reward-basis", params={"epoch": 24801}).json()["effective_reward_epoch"] == 24801
    assert http.get("/arena/v1/reward-basis", params={"epoch": 1}).status_code == 404
    assert http.get("/arena/v1/rounds/arena-2026-09-02").json()["status"] == "open"
    missing = http.get("/arena/v1/rounds/arena-2026-01-01")
    assert missing.status_code == 404 and missing.json() == {"status": "rejected", "code": "round_missing"}
    assert http.get("/arena/v1/rounds/arena-2026-09-02/benchmark").status_code == 403
    assert http.get("/arena/v1/rounds/arena-2026-09-02/results/sub-1").json()["submission_id"] == "sub-1"
    assert http.get("/arena/v1/submissions/sub-1").json() == {
        "submission_id": "sub-1", "status": "accepted", "rejection_rule": None,
    }
    assert http.get("/arena/v1/submissions/nope").status_code == 404
    assert http.get("/docs").status_code == 404 and http.get("/openapi.json").status_code == 404


def test_runner_routes_require_lease_header_and_bounded_bodies(client):
    http, service = client
    claim = http.post("/arena/v1/runs/claim", content=json.dumps({"scope": contracts.SCOPE_CLAIM}))
    assert claim.status_code == 200 and service.calls["claim"] == {"scope": contracts.SCOPE_CLAIM}
    frame = {"operation_id": "deepline.execute", "parameters": {"tool": "exa_search", "payload": {"query": "x"}}, "timeout_ms": 1000, "action_sequence": 0}
    no_lease = http.post("/arena/v1/runs/r1/provider", content=json.dumps(frame))
    assert no_lease.status_code == 401
    bad_lease = http.post("/arena/v1/runs/r1/provider", content=json.dumps(frame), headers={"x-lab-arena-lease": "short"})
    assert bad_lease.status_code == 401
    token = "a" * 64
    ok = http.post("/arena/v1/runs/r1/provider", content=json.dumps(frame), headers={"x-lab-arena-lease": token})
    assert ok.status_code == 200 and service.calls["provider"] == ("r1", token, frame)
    assert service.calls["provider_has_event_loop"] is False
    assert http.get("/arena/v1/runs/r1/source").status_code == 401
    source = http.get(
        "/arena/v1/runs/r1/source", headers={"x-lab-arena-lease": token}
    )
    assert source.status_code == 200 and source.content == b"source archive"
    assert source.headers["content-type"] == "application/gzip"
    assert service.calls["source"] == ("r1", token)
    assert http.post("/arena/v1/runs/r1/events", content=json.dumps({"events": []}), headers={"x-lab-arena-lease": token}).status_code == 404
    mismatch = http.post("/arena/v1/runs/r1/complete", content=json.dumps({"body": {"run_id": "r2"}}))
    assert mismatch.status_code == 400
    complete = http.post("/arena/v1/runs/r1/complete", content=json.dumps({"bad": True, "body": {"run_id": "r1"}}))
    assert complete.status_code == 400 and complete.json()["code"] == "run_result_invalid"
    too_big = http.post("/arena/v1/runs/claim", content=b'{"a":"' + b"x" * (MAX_JSON_BODY_BYTES + 10) + b'"}')
    assert too_big.status_code == 413
    streamed = http.post(
        "/arena/v1/runs/claim",
        content=(chunk for chunk in (b'{"a":"', b"x" * MAX_JSON_BODY_BYTES, b'"}')),
    )
    assert streamed.status_code == 413
    not_json = http.post("/arena/v1/runs/claim", content=b"{not json")
    assert not_json.status_code == 400
    nan = http.post("/arena/v1/runs/claim", content=b'{"a": NaN}')
    assert nan.status_code == 400


def test_complete_route_accepts_the_shared_completion_size(client):
    http, _service = client
    envelope = {
        "body": {
            "run_id": "r1",
            "output": {"chunks": ["x" * 65_000 for _ in range(18)]},
        }
    }
    raw = json.dumps(envelope).encode("utf-8")
    assert MAX_JSON_BODY_BYTES < len(raw) < contracts.COMPLETION_REQUEST_LIMITS.max_total_bytes
    assert http.post("/arena/v1/runs/r1/complete", content=raw).status_code == 200

    oversized = http.post(
        "/arena/v1/runs/r1/complete",
        content=b"{}",
        headers={
            "content-length": str(
                contracts.COMPLETION_REQUEST_LIMITS.max_total_bytes + 1
            )
        },
    )
    assert oversized.status_code == 413


def test_submission_routes_presign_and_finalize_one_source_upload(client):
    http, service = client
    envelope = {
        "scope": contracts.SCOPE_SUBMISSION_PRESIGN,
        "body": {
            "source_size_bytes": 100,
            "consent": {"public_rerun": True},
        },
    }
    response = http.post("/arena/v1/submissions/presign", content=json.dumps(envelope))
    assert response.status_code == 200 and service.calls["submission_presign"] == envelope
    finalize = {
        "scope": contracts.SCOPE_SUBMISSION_FINALIZE,
        "body": {
            "submission_id": "sub-1",
            "source_ref": "arena/arena-2026-09-02/sources/sub-1.tar.gz",
            "source_size_bytes": 100,
        },
    }
    response = http.post("/arena/v1/submissions/sub-1/finalize", content=json.dumps(finalize))
    assert response.status_code == 200
    assert service.calls["submission_finalize"] == ("sub-1", finalize)
    mismatch = http.post("/arena/v1/submissions/sub-2/finalize", content=json.dumps(finalize))
    assert mismatch.status_code == 400
    assert http.post("/arena/v1/submissions/presign", content=b"x").status_code == 400
    assert http.post("/arena/v1/submissions/presign", content=b"{bad").status_code == 400
    assert http.post("/arena/v1/funding/confirm", content=json.dumps({"scope": "gone"})).status_code == 404
    for provider in contracts.PROVIDERS:
        assert http.post("/arena/v1/credentials/%s" % provider, content=b"{}").status_code == 404


def test_provider_frames_carry_the_judges_long_prompts(client):
    """The judge sends page content of tens of thousands of characters in one chat message."""

    http, service = client
    content = "x" * 32_000
    frame = {"operation_id": "openrouter.chat", "parameters": {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": content}]}, "timeout_ms": 120000, "action_sequence": 3}
    ok = http.post("/arena/v1/runs/r1/provider", content=json.dumps(frame), headers={"x-lab-arena-lease": "a" * 64})
    assert ok.status_code == 200 and service.calls["provider"][2]["parameters"]["messages"][0]["content"] == content
    from lab_arena import contracts as c

    c.check_strict_document(frame, c.PROVIDER_FRAME_LIMITS)  # the service applies these to the frame



def test_an_oversized_declared_body_is_refused_before_it_is_read(client):
    """A content-length above the limit gets 413 without the body being buffered."""

    http, service = client
    refused = http.post("/arena/v1/runs/claim", content=b"{}", headers={"content-length": str(MAX_JSON_BODY_BYTES + 1)})
    assert refused.status_code == 413
    submission = http.post("/arena/v1/submissions/presign", content=b"{}", headers={"content-length": str(MAX_JSON_BODY_BYTES + 1)})
    assert submission.status_code == 413
    assert "submission_presign" not in service.calls
