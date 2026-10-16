"""Runtime input evidence is complete, private, immutable and observational."""

import copy
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, runner, scoring, training_input
from lab_arena.api import create_app
from lab_arena.service import ArenaService, LocalObjectStore, ServiceError
from lab_arena.store import hash_lease_token
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime, FakeApi, lease, make_config, scoring_lease, valid_company,
)

NOW = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
TOKEN = "a" * 64
CLAIM_HASH = "sha256:" + "c" * 64


def service_for(tmp_path, **changes):
    run = {
        "run_id": "r1", "round_id": "arena-test", "kind": "execute",
        "status": "leased", "lease_token_hash": hash_lease_token(TOKEN),
        "lease_generation": 1, "claim_request_hash": CLAIM_HASH,
        "lease_expires_at": (NOW + timedelta(minutes=2)).isoformat(),
        **changes,
    }
    objects = LocalObjectStore(tmp_path / "objects")
    service = ArenaService.__new__(ArenaService)
    service._store = SimpleNamespace(get_run=lambda run_id: run if run_id == "r1" else None)
    service._config = SimpleNamespace(training_object_store=objects)
    service._clock = lambda: NOW
    service._require_round_ownership = lambda _round: None
    return service, objects


def test_long_input_roundtrip_and_immutable_duplicate(tmp_path):
    service, objects = service_for(tmp_path)
    document = {"schema_version": "test", "icp": {"prompt": "a long input\n" * 100_000}, "unicode": "é"}
    receipt = service.handle_training_input("r1", TOKEN, document)
    assert receipt["status"] == "stored"
    assert service.handle_training_input("r1", TOKEN, document) == receipt
    saved = training_input.read_input(objects, "r1", "execute", 1, CLAIM_HASH)
    assert saved["input"] == document
    assert saved["input_sha256"] == contracts.document_hash(document)
    assert saved["privacy"] == {"redacted": False, "redactions": []}
    with pytest.raises(ServiceError) as rejected:
        service.handle_training_input("r1", TOKEN, {"different": True})
    assert rejected.value.status == 409
    assert training_input.read_input(objects, "r1", "execute", 1, CLAIM_HASH) == saved


@pytest.mark.parametrize("changes, token, code", [
    ({}, "b" * 64, "lease_invalid"),
    ({"status": "completed"}, TOKEN, "lease_inactive"),
    ({"lease_expires_at": NOW.isoformat()}, TOKEN, "lease_expired"),
    ({"lease_expires_at": "invalid"}, TOKEN, "lease_invalid"),
    ({"lease_token_hash": ""}, TOKEN, "lease_invalid"),
])
def test_lease_rejection_precedes_storage(tmp_path, changes, token, code):
    service, objects = service_for(tmp_path, **changes)
    with pytest.raises(ServiceError) as rejected:
        service.handle_training_input("r1", token, {"input": "private"})
    assert rejected.value.code == code
    assert not list((tmp_path / "objects").rglob("*.gz"))


def test_round_ownership_rejection_precedes_storage(tmp_path):
    service, objects = service_for(tmp_path)
    def reject(_round):
        raise ServiceError("round_not_owned", 403)
    service._require_round_ownership = reject
    with pytest.raises(ServiceError, match="round_not_owned"):
        service.handle_training_input("r1", TOKEN, {})
    assert not list((tmp_path / "objects").rglob("*.gz"))


def test_input_redacts_credentials_and_hidden_reasoning(tmp_path):
    service, objects = service_for(tmp_path)
    document = {"prompt": "hello " + TOKEN, "api_key": "credential", "reasoning_content": "private reasoning", "icp": {"prompt": "complete"}}
    service.handle_training_input("r1", TOKEN, document)
    saved = training_input.read_input(objects, "r1", "execute", 1, CLAIM_HASH)
    assert TOKEN not in json.dumps(saved)
    assert "private reasoning" not in json.dumps(saved)
    assert saved["privacy"] == {"redacted": True, "redactions": ["credential", "hidden_reasoning"]}
    assert saved["input"]["icp"] == document["icp"]


@pytest.mark.parametrize("mutation", ["hash", "run", "kind", "schema", "privacy", "inflate", "generation", "claim"])
def test_reader_refuses_corrupt_or_unbounded_input(tmp_path, mutation):
    objects = LocalObjectStore(tmp_path)
    snapshot, data = training_input.build_snapshot("r1", "execute", {"icp": "input"}, lease_generation=1, claim_request_hash=CLAIM_HASH)
    if mutation == "hash":
        snapshot["input"]["icp"] = "changed"
    elif mutation == "run":
        snapshot["run_id"] = "other"
    elif mutation == "kind":
        snapshot["kind"] = "score"
    elif mutation == "generation":
        snapshot["lease_generation"] = 2
    elif mutation == "claim":
        snapshot["claim_request_hash"] = "sha256:" + "d" * 64
    elif mutation == "schema":
        snapshot["schema_version"] = "unknown"
    elif mutation == "privacy":
        snapshot["privacy"]["redacted"] = True
    data = gzip.compress(b"x" * (training_input.MAX_OBJECT_BYTES + 1) if mutation == "inflate" else contracts.canonical_json(snapshot).encode(), mtime=0)
    objects.put(training_input.object_ref("r1", 1), data)
    with pytest.raises(training_input.InputCaptureError):
        training_input.read_input(objects, "r1", "execute", 1, CLAIM_HASH)


def test_api_requires_lease_bounds_body_and_marks_response_private(tmp_path):
    service, objects = service_for(tmp_path)
    http = TestClient(create_app(service))
    path = "/arena/v1/runs/r1/training-input"
    assert http.post(path, json={}).status_code == 401
    response = http.post(path, json={"prompt": "x" * 1_200_000}, headers={"x-lab-arena-lease": TOKEN})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["status"] == "stored"
    assert http.post(path, content=b"{}", headers={"x-lab-arena-lease": TOKEN, "content-length": str(training_input.MAX_INPUT_BYTES + 1)}).status_code == 413


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("capture_fails", [False, True])
def test_capture_matches_final_runtime_boundary_without_affecting_execution(tmp_path, kind, capture_fails):
    active = scoring_lease() if kind == "score" else lease()
    active["deepline_catalog"] = {"source": "complete catalog", "tools": []}
    snapshots = []
    class CapturingApi(FakeApi):
        def training_input(self, run_id, token, document):
            snapshots.append(copy.deepcopy(document))
            if capture_fails:
                raise RuntimeError("private upload failure")
            return {"status": "stored"}
    api = CapturingApi([active])
    output = scoring.build_scoring_output("r1", [{"final_score": 71.0, "failure_reason": ""}, {"final_score": 44.5, "failure_reason": ""}]) if kind == "score" else {"companies": [valid_company(1)]}
    sandbox = BridgingRuntime(output=output, calls=0)
    (tmp_path / "work").mkdir()
    assert runner.Runner(make_config(tmp_path, api, sandbox)).run_once() == 1
    assert len(snapshots) == (2 if capture_fails else 1)
    assert all(saved == sandbox.input_documents[0] for saved in snapshots)
    assert snapshots[0]["deepline_catalog"] == active["deepline_catalog"]
    assert api.completions[0]["body"]["result"]["terminal_status"] == "accepted"


def test_http_capture_uses_bounded_timeout_and_full_document():
    observed = {}
    def respond(request):
        observed.update(document=json.loads(request.content), timeout=request.extensions["timeout"], lease=request.headers["x-lab-arena-lease"])
        return httpx.Response(200, json={"status": "stored"})
    client = runner.HttpArenaApiClient("http://127.0.0.1", client=httpx.Client(transport=httpx.MockTransport(respond)))
    document = {"input": "whole input" * 100_000}
    assert client.training_input("r1", TOKEN, document)["status"] == "stored"
    assert observed["document"] == document
    assert set(observed["timeout"].values()) == {2.0}
    assert observed["lease"] == TOKEN


@pytest.mark.parametrize("context", ["company_judgment_cache", "prior_terminal_breakdowns"])
def test_judge_capture_retains_validated_recovery_context(tmp_path, context):
    if context == "company_judgment_cache":
        from tests.lab_arena.company_judgments_test import _input
        from tests.lab_arena.partial_company_judgments_test import _lease
        document = _input()
        extra = _lease(document)
    else:
        from tests.lab_arena.legacy_partial_runtime_test import _input, _row
        document = _input()
        extra = [_row()]
    active = scoring_lease()
    active.update(
        icp=document["icp"], scorer_policy=document["scorer_policy"],
        scored_output={"companies": document["companies"]},
    )
    active[context] = extra
    snapshots = []
    class CapturingApi(FakeApi):
        def training_input(self, run_id, token, value):
            snapshots.append(copy.deepcopy(value))
            return {"status": "stored"}
    api = CapturingApi([active])
    sandbox = BridgingRuntime(output=scoring.build_scoring_failure("r1", "judge_error"), calls=0)
    (tmp_path / "work").mkdir()
    assert runner.Runner(make_config(tmp_path, api, sandbox)).run_once() == 1
    assert snapshots[0] == sandbox.input_documents[0]
    assert snapshots[0][context] == extra


def test_unconfigured_capture_does_not_use_main_object_client(tmp_path):
    service, _objects = service_for(tmp_path)
    service._config.training_object_store = None
    with pytest.raises(ServiceError) as rejected:
        service.handle_training_input("r1", TOKEN, {})
    assert rejected.value.code == "training_input_unavailable"


def test_oversize_input_fails_without_truncation_or_storage(tmp_path):
    service, _objects = service_for(tmp_path)
    with pytest.raises(ServiceError) as rejected:
        service.handle_training_input("r1", TOKEN, {"prompt": "x" * training_input.MAX_INPUT_BYTES})
    assert rejected.value.code == "training_input_invalid"
    assert not list((tmp_path / "objects").rglob("*.gz"))


def test_local_capture_is_immutable_under_concurrent_uploads(tmp_path):
    service, objects = service_for(tmp_path)
    def upload(index):
        try:
            service.handle_training_input("r1", TOKEN, {"input": str(index) * 10_000})
            return index
        except ServiceError as exc:
            assert exc.code == "training_input_conflict"
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = [value for value in pool.map(upload, range(8)) if value is not None]
    assert len(accepted) == 1
    assert training_input.read_input(objects, "r1", "execute", 1, CLAIM_HASH)["input"] == {"input": str(accepted[0]) * 10_000}
    assert not list(tmp_path.rglob(".arena-*"))


def test_reclaimed_run_keeps_both_inputs_and_reads_only_final_lease_authority(tmp_path):
    service, objects = service_for(tmp_path)
    old_document = {"icp": {"prompt": "first"}, "company_judgment_cache": {"hits": []}}
    first = service.handle_training_input("r1", TOKEN, old_document)
    run = service._store.get_run("r1")
    next_token = "b" * 64
    next_claim = "sha256:" + "d" * 64
    run.update(lease_generation=2, claim_request_hash=next_claim,
               lease_token_hash=hash_lease_token(next_token))
    final_document = {"icp": {"prompt": "first"}, "company_judgment_cache": {"hits": [{"verified": True}]}}
    # A client cannot set the authority fields. They remain ordinary input data.
    final_document["lease_generation"] = 99
    final_document["claim_request_hash"] = "client supplied"
    with pytest.raises(ServiceError) as stale:
        service.handle_training_input("r1", TOKEN, final_document)
    assert stale.value.code == "lease_invalid"
    final = service.handle_training_input("r1", next_token, final_document)
    assert first["ref"] != final["ref"]
    assert final["lease_generation"] == 2 and final["claim_request_hash"] == next_claim
    assert training_input.read_input(objects, "r1", "execute", 1, CLAIM_HASH)["input"] == old_document
    saved = training_input.read_input(objects, "r1", "execute", 2, next_claim)
    assert saved["input"] == final_document
    with pytest.raises(training_input.InputCaptureError):
        training_input.read_input(objects, "r1", "execute", 2, CLAIM_HASH)


@pytest.mark.parametrize("authority", [
    {"lease_generation": None}, {"lease_generation": 0}, {"lease_generation": True},
    {"claim_request_hash": None}, {"claim_request_hash": "invalid"},
])
def test_missing_claim_authority_cannot_create_input_evidence(tmp_path, authority):
    service, _objects = service_for(tmp_path, **authority)
    with pytest.raises(ServiceError) as error:
        service.handle_training_input("r1", TOKEN, {})
    assert error.value.code == "training_input_invalid"
    assert not list((tmp_path / "objects").rglob("*.gz"))
