"""Signed failed completions persist only leased terminal company evidence."""
from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import threading

import pytest

from lab_arena import contracts, scoring
from lab_arena.service import ArenaService, ServiceConfig, S3ObjectStore, ServiceError
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.company_judgments_postgres_test import _open, _raw, _complete
from tests.lab_arena.lab_arena_pg_harness import POSTGREST_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_migration_postgres import claim, hotkey
from tests.lab_arena.test_lab_arena_service_round import keypair, wallet_verify, FakeChain
from tests.lab_arena.test_lab_arena_object_store import _ConditionalS3

MIGRATION = "465-lab-arena-partial-company-judgments.sql"


@pytest.fixture(scope="module")
def database():
    # The exact numbered migration must also be safe on a second application.
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS + (MIGRATION, MIGRATION))


@pytest.fixture()
def store(database):
    psycopg2, dsn = database
    return ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def _setup(store, marker):
    cases = ["judge_error", "judge_timeout", "credential_error", "foreign_runner", "lease", "generation", "slot", "input", "wrong_run", "provider_failure", "expired", "conflict", "concurrent"]
    round_id = "arena-2026-09-11-p%02d" % (50 + cases.index(marker))
    _, items = _open(store, round_id=round_id, markers_by_submission=[["positive", "negative", "unresolved"]])
    key = keypair("partial-validator-" + marker)
    leased, token, _, _ = claim(store, round_id, key.ss58_address, excluded=[key.ss58_address])
    assert leased["status"] == "leased"
    objects = S3ObjectStore("test-arena", client=_ConditionalS3())
    service = ArenaService(ServiceConfig(
        mode="live", store=store, object_store=objects, signer=None,
        chain=FakeChain([key.ss58_address]), verify_signature=wallet_verify,
        daily_icp_source=lambda **_: {}, banned_hotkeys_source=lambda: (),
        broker_factory=lambda *_: None,
    ))
    return service, round_id, key, leased, token, items[0]["company_judgment_refs"]


def _request(round_id, key, leased, token, *, terminal="judge_error"):
    misses = sorted(leased["company_judgment_cache"]["misses"], key=lambda row: row["company_index"])
    rows = [{**miss, "raw_judgment": _raw("negative" if index == 1 else "positive")}
            for index, miss in enumerate(misses[:2])]
    result = {
        "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
        "resource_summary": {"wall_seconds": 1.0, "cpu_seconds": 0.0, "max_rss_bytes": 1024,
                             "stdout_bytes": 0, "stderr_bytes": 0, "provider_call_count": 1},
        "started_at": "2026-09-11T00:00:00Z", "finished_at": "2026-09-11T00:00:01Z",
        "terminal_status": terminal,
        # A caller's fabricated completion binding is overwritten by the RPC.
        "company_judgment_completion_request_hash": "sha256:" + "1" * 64,
    }
    output = scoring.build_scoring_failure(leased["scored_run_id"], "judge_timeout" if terminal == "judge_timeout" else "judge_error", company_judgments=rows)
    return _sign(round_id, key, {"run_id": leased["run_id"], "lease_token": token, "result": result, "output": output})


def _sign(round_id, key, body):
    return contracts.build_signed_request(scope=contracts.SCOPE_COMPLETE, round_id=round_id,
        hotkey=key.ss58_address, body=body, timestamp=int(datetime.now(timezone.utc).timestamp()),
        sign_message=lambda message: key.sign(message.encode()).hex())


@pytest.mark.parametrize("terminal", ["judge_error", "judge_timeout", "credential_error"])
def test_failed_signed_completion_keeps_company_evidence_and_never_accepts_icp(store, terminal):
    service, round_id, key, leased, token, refs = _setup(store, terminal)
    envelope = _request(round_id, key, leased, token, terminal=terminal)
    result = service.handle_complete(envelope)
    assert result["status"] == "failed"
    assert result["company_judgments_stored"] == 2
    failed = store.get_run(leased["run_id"])
    assert failed["status"] == "failed" and failed["per_icp_score"] is None
    assert failed["result_doc"]["terminal_status"] == terminal
    assert failed["result_doc"]["company_judgment_completion_request_hash"] == contracts.request_bytes_hash(envelope)
    for ref in refs[:2]:
        evidence = service._company_judgment_evidence(cache_key=ref["cache_key"], company_input_hash=ref["company_input_hash"], authority_slot=0)
        assert evidence["source_score_run_id"] == leased["run_id"]
    assert store.get_company_judgment(refs[2]["cache_key"], 0) is None
    if terminal != "credential_error":
        another = hotkey("partial-retry-" + terminal)
        retry, retry_token, _, _ = claim(store, round_id, another, excluded=[another])
        assert retry["status"] == "leased"
        assert sorted(row["company_index"] for row in retry["company_judgment_cache"]["hits"]) == [0, 1]
        assert [row["company_index"] for row in retry["company_judgment_cache"]["misses"]] == [2]
        assert _complete(store, retry, retry_token, refs=refs)["status"] == "accepted"
        assert store.get_run(leased["run_id"])["status"] == "failed"
    # A replay remains idempotent and cannot restamp the immutable source.
    assert service.handle_complete(envelope)["idempotent"] is True
    assert store.get_run(leased["run_id"])["result_doc"] == failed["result_doc"]


@pytest.mark.parametrize("change", ["foreign_runner", "lease", "generation", "slot", "input", "wrong_run", "provider_failure", "expired"])
def test_failed_partial_completion_rejects_unbound_authority(store, database, change):
    service, round_id, key, leased, token, refs = _setup(store, change)
    request = _request(round_id, key, leased, token)
    body = copy.deepcopy(request["body"])
    row = body["output"]["company_judgments"][0]
    if change == "foreign_runner":
        key = keypair("foreign-partial-validator")
        service.config.chain.runners.append(key.ss58_address)
    elif change == "lease":
        body["lease_token"] = "wrong-token"
    elif change == "generation":
        psycopg2, dsn = database
        with psycopg2.connect(**dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("UPDATE public.lab_arena_company_judgment_reservations SET lease_generation = lease_generation + 1 WHERE run_id = %s", (leased["run_id"],))
    elif change == "slot":
        row["authority_slot"] += 1
    elif change == "input":
        row["company_input_hash"] = "sha256:" + "b" * 64
    elif change == "wrong_run":
        body["output"]["scored_run_id"] = "foreign-execution"
    elif change == "provider_failure":
        row["raw_judgment"]["failure_reason"] = "provider timeout"
        row["raw_judgment"]["final_score"] = 0
    else:
        psycopg2, dsn = database
        with psycopg2.connect(**dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("UPDATE public.lab_arena_runs SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE run_id = %s", (leased["run_id"],))
    request = _sign(round_id, key, body)
    try:
        result = service.handle_complete(request)
    except (ServiceError, ValueError, RuntimeError):
        pass
    else:
        assert result["status"] in ("stale", "accounting_open")
    assert store.get_run(leased["run_id"])["status"] == "leased"
    assert all(store.get_company_judgment(ref["cache_key"], 0) is None for ref in refs)


def test_failed_source_and_its_bytes_survive_terminal_conflicting_completion(store):
    service, round_id, key, leased, token, refs = _setup(store, "conflict")
    request = _request(round_id, key, leased, token)
    assert service.handle_complete(request)["status"] == "failed"
    before = store.get_run(leased["run_id"])
    blob = service._objects.get(before["output_ref"])
    changed_body = copy.deepcopy(request["body"])
    changed_body["output"]["company_judgments"][0]["raw_judgment"]["final_score"] = 19
    with pytest.raises(contracts.ArenaContractError, match="different bytes"):
        service.handle_complete(_sign(round_id, key, changed_body))
    assert service._objects.get(before["output_ref"]) == blob
    assert store.get_run(leased["run_id"])["result_doc"] == before["result_doc"]
    service._company_judgment_evidence(cache_key=refs[0]["cache_key"], company_input_hash=refs[0]["company_input_hash"], authority_slot=0)


def test_concurrent_conflicting_completions_keep_one_bound_source(store):
    service, round_id, key, leased, token, refs = _setup(store, "concurrent")
    first = _request(round_id, key, leased, token)
    body = copy.deepcopy(first["body"])
    body["output"]["company_judgments"][0]["raw_judgment"]["final_score"] = 17
    second = _sign(round_id, key, body)
    client = service._objects._client
    lock = threading.Lock()
    original_put = client.put_object

    def conditional_put(**request):
        with lock:
            return original_put(**request)

    client.put_object = conditional_put
    barrier = threading.Barrier(2)
    original_get = store.get_run

    def synchronized_get(run_id):
        run = original_get(run_id)
        if run_id == leased["run_id"] and run["status"] == "leased":
            barrier.wait(timeout=5)
        return run

    store.get_run = synchronized_get
    def complete(request):
        try:
            return service.handle_complete(request)
        except contracts.ArenaContractError:
            return {"status": "conflicting_bytes"}
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(complete, (first, second)))
    finally:
        store.get_run = original_get
    assert sorted(result["status"] for result in outcomes) == ["conflicting_bytes", "failed"]
    source = store.get_run(leased["run_id"])
    output = json.loads(service._objects.get(source["output_ref"]))
    evidence = service._company_judgment_evidence(cache_key=refs[0]["cache_key"], company_input_hash=refs[0]["company_input_hash"], authority_slot=0)
    assert evidence["source_output_hash"] == contracts.document_hash(output)
