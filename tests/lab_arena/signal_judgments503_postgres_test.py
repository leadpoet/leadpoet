"""Database authority for terminal signal evidence and live reservations."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import tempfile

import pytest

from lab_arena import contact_policy, contracts, runner, shim, signal_cache_control
from lab_arena.service import ArenaService, LocalObjectStore, ServiceError
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, hash_lease_token
from tests.lab_arena import legacy_company_cache501_postgres_test as prior
from tests.lab_arena import company_judgments_postgres_test as company
from tests.test_arena_score_integrity import _icp, _public_company


SCRIPTS = Path(__file__).parents[2] / "scripts"


@pytest.fixture(scope="module")
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute((SCRIPTS / "502-lab-arena-company-judgment-finality.sql").read_text())
            cur.execute("CREATE ROLE signal503_migrator NOLOGIN NOSUPERUSER INHERIT")
            cur.execute("GRANT lab_arena_owner TO signal503_migrator")
            cur.execute("GRANT USAGE,CREATE ON SCHEMA public TO signal503_migrator WITH GRANT OPTION")
            cur.execute("ALTER DEFAULT PRIVILEGES FOR ROLE signal503_migrator IN SCHEMA public GRANT ALL ON TABLES TO anon,authenticated,service_role,lab_arena_service")
            cur.execute("ALTER DEFAULT PRIVILEGES FOR ROLE signal503_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role,lab_arena_service")
            cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
            assert cur.fetchone() == (False,)
            cur.execute("SET ROLE signal503_migrator")
            cur.execute((SCRIPTS / "503-lab-arena-signal-judgment-cache.sql").read_text())
            cur.execute("RESET ROLE")
            cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
            assert cur.fetchone() == (False,)
        yield db
    finally:
        gen.close()


@pytest.fixture()
def store(database):
    result = ArenaStore(PsycopgTransport(lambda: database[0].connect(**database[1])))
    result._test_database = database
    try:
        yield result
    finally:
        result.close()


def _scope(round_id: str, signal: str) -> dict:
    scope = {
        "schema_version": "leadpoet.lab_arena.signal_judgment_scope.v1",
        "round_id": round_id, "network_name": "finney", "netuid": 71,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": "registry/scorer@sha256:" + "a" * 64,
        "evaluation_date": "2099-10-20",
        "signal_judgment_policy": "signal_judgment_v1",
        "signal_input_hash": contracts.document_hash({"signal": signal}),
        "expected_urls": ["https://example.com/" + signal],
        "expected_index": 0, "expected_evidence_type": "HIRING",
    }
    return {**scope, "cache_key": contracts.document_hash(scope)}


def _enable_round(database, round_id: str) -> dict:
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                    "jsonb_set(configuration_doc,'{scorer_policy,signal_judgment_policy}',"
                    "'\"signal_judgment_v1\"'::jsonb) WHERE round_id=%s", (round_id,))
        cur.execute("SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s", (round_id,))
        config = cur.fetchone()[0]
    return config


def _signal_service(store, leased, tmp_path):
    """Use the real handler, frozen round, database store, and output reader."""
    submitted = _public_company("Acme", domain="acme.example", signal_date="2099-10-19")
    score_run = store.get_run(leased["run_id"])
    configuration = store.get_round(score_run["round_id"])["configuration_doc"]
    output_schema = contact_policy.output_schema(configuration)
    if output_schema == "leadpoet.lab_arena.output.v6":
        submitted.pop("fit_summary")
        submitted.pop("fit_evidence_urls")
        submitted["intent_details"] = "Acme's product launch creates a timely buying trigger."
        submitted["intent_signals"][0].pop("why_now")
        submitted["intent_signals"][0].pop("snippet")
    objects = LocalObjectStore(tmp_path)
    executed = store.get_run(leased["scored_run_id"])
    objects.put(executed["output_ref"], json.dumps({
        "schema_version": output_schema,
        "companies": [submitted],
    }).encode())
    service = ArenaService.__new__(ArenaService)
    service._store = store
    service._objects = objects
    service._require_round_ownership = lambda _: None
    service._hot_round = lambda round_id: store.get_round(round_id)
    service.evaluation_icps = lambda _: [_icp()]
    request = {
        "company_index": 0, "signal_index": 0,
        "verified_company_identity": {
            "decision": "match", "observed_domain": "acme.example",
        },
        "verified_funding_stage_context": [],
    }
    signal = submitted["intent_signals"][0]
    url = signal["url"]
    quote = signal.get("snippet") or "Acme launched the product."
    judgment = {
        "raw": 54.0, "after_decay": 54.0, "decay": 1.0,
        "confidence": 90, "date_status": "in_window",
        "date_verdict": "in_window", "claim_support_verdict": "supported",
        "matched_icp_signal": 0, "evidence_type": "UNSPECIFIED",
        "evidence_urls": [url],
        "judge_verdict": {
            "decision": "verified", "pipeline_decision": "approve",
            "claim_support_verdict": "supported", "client_ready": True,
            "verification_trace": {
                "evidence_url": url, "evidence_urls": [url],
                "evidence_source": "company_news",
                "extracted_signal_date": signal["date"],
                "verified_source_context": [{"url": url, "text": quote}],
                "intent_verdict": {"signal_evaluations": [{
                    "signal_status": "supported", "same_entity_check": "pass",
                    "verification_mode": "source_grounded",
                    "supporting_quotes": [quote], "evidence_urls_used": [url],
                }]},
                "final_disposition": "approve",
            },
        },
    }
    return service, request, judgment


def _make_baseline(database, rid, participants):
    """Freeze the fixture's first participant as the broker's real king."""
    baseline_hotkey = participants[0]["miner_hotkey"]
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_submissions SET is_king=true WHERE submission_id=%s",
                    (participants[0]["submission_id"],))
        cur.execute("UPDATE public.lab_arena_rounds SET participants=jsonb_set(participants,'{0,is_king}','true'::jsonb),configuration_doc=jsonb_set(configuration_doc,'{baseline_hotkey}',%s::jsonb) WHERE round_id=%s",
                    (json.dumps(baseline_hotkey), rid))
    return baseline_hotkey


def _store_terminal(service, leased, token, request, judgment):
    miss = service.handle_signal_judgment(leased["run_id"], token, {**request, "operation": "claim"})
    assert miss["status"] == "miss"
    assert service.handle_signal_judgment(leased["run_id"], token, {
        **request, "operation": "complete", "reference": miss["reference"],
        "judgment": judgment,
    })["status"] == "stored"
    return miss


def test_full_thinking_capture_worker_claim_reaches_real_signal_service(store, database, tmp_path, monkeypatch):
    from scripts.verify_arena_full_thinking_e2e import FullThinkingCaptureApi
    rid = "arena-2026-09-11-capture503"
    participants, _ = prior.open_new(store, round_id=rid, markers_by_submission=[["one"]])
    _enable_round(database, rid)
    baseline_hotkey = _make_baseline(database, rid, participants)
    leased, token, _, _ = company.claim(store, rid, baseline_hotkey, excluded=[baseline_hotkey])
    service, request, _judgment = _signal_service(store, leased, tmp_path)
    state = runner.RunState(lease={"run_id": leased["run_id"], "kind": "score",
                                   "scorer_policy": {"signal_judgment_policy": signal_cache_control.POLICY}},
                            lease_token=token)
    api = FullThinkingCaptureApi(service)
    with tempfile.TemporaryDirectory(prefix="signal-", dir="/tmp") as directory:
        path = Path(directory) / "worker.sock"
        worker = runner.WorkerSocketServer(path, api, state)
        worker.start()
        monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(path))
        monkeypatch.setenv(shim.TRUSTED_SCORER_ENV, "1")
        try:
            claim = asyncio.run(signal_cache_control.exchange({**request, "operation": "claim"}))
            complete = asyncio.run(signal_cache_control.exchange({
                **request, "operation": "complete", "reference": claim["reference"],
                "judgment": None,
            }))
        finally:
            worker.stop()
    assert claim["status"] == "miss"
    assert claim["reference"]["schema_version"] == "leadpoet.lab_arena.signal_judgment_reference.v1"
    assert complete == {"status": "released", "reference": claim["reference"]}
    assert state.action_sequence == 0 and state.calls == []
    assert api.quota_usage(leased["run_id"], token) == service.handle_quota_snapshot(
        leased["run_id"], token)
    assert api.quota_usage(leased["run_id"], token,
                           include_sourcing_cost=True) == {
                               "status": "rejected", "code": "quota_unavailable"}


def test_same_score_lease_reuses_its_terminal_signal(store, database, tmp_path):
    rid = "arena-2026-09-11-own503"
    participants, _ = prior.open_new(store, round_id=rid, markers_by_submission=[["one"]])
    _enable_round(database, rid)
    baseline_hotkey = _make_baseline(database, rid, participants)
    leased, token, _, _ = company.claim(store, rid, baseline_hotkey, excluded=[baseline_hotkey])
    assert leased["status"] == "leased" and leased["miner_hotkey"] == baseline_hotkey
    service, request, judgment = _signal_service(store, leased, tmp_path)
    miss = _store_terminal(service, leased, token, request, judgment)
    source = store.get_signal_judgment(miss["reference"]["cache_key"], miss["reference"]["authority_slot"])
    assert baseline_hotkey in source["runner_authority_exclusions"]
    hit = service.handle_signal_judgment(leased["run_id"], token, {**request, "operation": "claim"})
    assert hit == {"status": "hit", "reference": miss["reference"], "judgment": judgment}


@pytest.mark.parametrize("tamper", ["claim_hash", "generation", "source_run"])
def test_same_run_hit_rejects_changed_authority(store, database, tmp_path, tamper):
    rid = "arena-2026-09-11-bound503"
    participants, _ = prior.open_new(store, round_id=rid, markers_by_submission=[["one"]])
    _enable_round(database, rid)
    runner = _make_baseline(database, rid, participants)
    leased, token, _, _ = company.claim(store, rid, runner, excluded=[runner])
    service, request, judgment = _signal_service(store, leased, tmp_path)
    miss = _store_terminal(service, leased, token, request, judgment)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        if tamper == "claim_hash":
            cur.execute("UPDATE public.lab_arena_runs SET claim_request_hash=%s WHERE run_id=%s",
                        ("sha256:" + "e" * 64, leased["run_id"]))
        elif tamper == "generation":
            cur.execute("UPDATE public.lab_arena_runs SET lease_generation=lease_generation+1 WHERE run_id=%s",
                        (leased["run_id"],))
        else:
            cur.execute("UPDATE public.lab_arena_signal_judgments SET source_scored_run_id=%s WHERE cache_key=%s AND authority_slot=%s",
                        ("wrong-execute-run", miss["reference"]["cache_key"], miss["reference"]["authority_slot"]))
    with pytest.raises(ServiceError, match="signal_judgment_cache_invalid"):
        service.handle_signal_judgment(leased["run_id"], token, {**request, "operation": "claim"})
    with pytest.raises(ServiceError, match="signal_judgment_lease_invalid"):
        service.handle_signal_judgment(leased["run_id"], "wrong-token", {**request, "operation": "claim"})


def test_later_score_run_cannot_reuse_own_prior_signal(store, database, tmp_path):
    rid = "arena-2026-09-11-retry503"
    participants, _ = prior.open_new(store, round_id=rid, markers_by_submission=[["one"]])
    _enable_round(database, rid)
    runner = _make_baseline(database, rid, participants)
    first, first_token, _, _ = company.claim(store, rid, runner, excluded=[runner])
    service, request, judgment = _signal_service(store, first, tmp_path)
    miss = _store_terminal(service, first, first_token, request, judgment)
    assert store.complete_attempt(
        run_id=first["run_id"], lease_token_hash=hash_lease_token(first_token),
        result={"terminal_status": "judge_error"}, terminal_cause="judge_error",
        output_ref="", company_judgment_evidence=[], output_hash="",
        completion_request_hash="sha256:" + "e" * 64,
    )["status"] == "failed"
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND kind='score' AND status='pending'",
                    (rid,))
    second, second_token, _, _ = company.claim(store, rid, runner, excluded=[runner])
    assert second["status"] == "leased" and second["run_id"] != first["run_id"]
    second_service, _, _ = _signal_service(store, second, tmp_path)
    second_claim = second_service.handle_signal_judgment(
        second["run_id"], second_token, {**request, "operation": "claim"}
    )
    assert second_claim["status"] == "miss"
    assert second_claim["reference"]["authority_slot"] != miss["reference"]["authority_slot"]


def test_other_miner_hits_terminal_signal_while_source_is_leased(store, database, tmp_path):
    rid = "arena-2026-09-11-cross503"
    participants, _ = prior.open_new(store, round_id=rid, markers_by_submission=[["one"], ["two"]])
    _enable_round(database, rid)
    runner = _make_baseline(database, rid, participants)
    first, first_token, _, _ = company.claim(store, rid, runner, parallelism=2, excluded=[runner])
    assert first["miner_hotkey"] == runner
    service, request, judgment = _signal_service(store, first, tmp_path)
    miss = _store_terminal(service, first, first_token, request, judgment)
    second, second_token, _, _ = company.claim(store, rid, runner, parallelism=2, excluded=[runner])
    assert second["status"] == "leased" and second["miner_hotkey"] != runner
    second_service, _, _ = _signal_service(store, second, tmp_path)
    assert second_service.handle_signal_judgment(
        second["run_id"], second_token, {**request, "operation": "claim"}
    ) == {"status": "hit", "reference": miss["reference"], "judgment": judgment}
    assert store.get_run(first["run_id"])["status"] == "leased"


def test_signal_hit_and_stale_lease_are_bound_to_source(store, database):
    rid = "arena-2026-09-11-s503"
    _, items = prior.open_new(store, round_id=rid, markers_by_submission=[["one"], ["one"], ["one"]])
    config = _enable_round(database, rid)
    scope = _scope(rid, "A")
    scope["scorer_image_digest"] = config["scorer_image_digest"]
    scope["scorer_image_reference"] = config["scorer_image_reference"]
    scope.pop("cache_key")
    scope["cache_key"] = contracts.document_hash(scope)
    runner = company.hotkey("signal503-runner")
    inherited_exclusion = company.hotkey("signal503-original-authority")
    first, token, _, _ = company.claim(
        store, rid, runner, excluded=[runner, inherited_exclusion]
    )
    assert first["status"] == "leased"
    run_id = first["run_id"]
    miss = store.claim_signal_judgment(
        run_id=run_id, lease_token_hash=hash_lease_token(token),
        company_index=0, signal_index=0, scope=scope,
    )
    assert miss["status"] == "miss"
    with pytest.raises(ArenaStoreError, match="signal_completion_invalid"):
        store.complete_signal_judgment(
            run_id=run_id, lease_token_hash="wrong", company_index=0,
            signal_index=0, scope=scope, reference=miss["reference"],
            judgment={"raw": 1}, judgment_hash=contracts.document_hash({"raw": 1}),
        )
    judgment = {"raw": 1, "decision": "terminal"}
    completed = store.complete_signal_judgment(
        run_id=run_id, lease_token_hash=hash_lease_token(token),
        company_index=0, signal_index=0, scope=scope,
        reference=miss["reference"], judgment=judgment,
        judgment_hash=contracts.document_hash(judgment),
    )
    assert completed["status"] == "stored"
    assert store.complete_signal_judgment(
        run_id=run_id, lease_token_hash=hash_lease_token(token),
        company_index=0, signal_index=0, scope=scope,
        reference=miss["reference"], judgment=judgment,
        judgment_hash=contracts.document_hash(judgment),
    )["status"] == "existing"
    # The next assignment gets an immutable hit from the first score run.
    assert store.complete_attempt(
        run_id=run_id, lease_token_hash=hash_lease_token(token),
        result={"terminal_status": "judge_error"}, terminal_cause="judge_error",
        output_ref="", company_judgment_evidence=[], output_hash="",
        completion_request_hash="sha256:" + "e" * 64,
    )["status"] == "failed"
    second, token2, _, _ = company.claim(store, rid, runner, excluded=[runner])
    assert second["status"] == "leased"
    hit = store.claim_signal_judgment(
        run_id=second["run_id"], lease_token_hash=hash_lease_token(token2),
        company_index=0, signal_index=0, scope=scope,
    )
    assert hit["status"] == "hit" and hit["judgment"] == judgment
    assert store.get_signal_judgment_uses(second["run_id"])[0]["source_score_run_id"] == run_id
    assert inherited_exclusion in store.signal_authority_exclusions(second["run_id"])
    refs = next(item["company_judgment_refs"] for item in items
                if item["submission_id"] == second["submission_id"])
    def inherit_source_exclusions(rows):
        for row in rows:
            row["evidence_doc"]["runner_authority_exclusions"] = store.signal_authority_exclusions(second["run_id"])
            row["evidence_hash"] = contracts.document_hash(row["evidence_doc"])
    assert prior.complete_subset(store, second, token2, refs,
                                 tamper=inherit_source_exclusions)["company_judgments_stored"] == 1
    third, _, _, _ = company.claim(store, rid, runner, excluded=[runner])
    assert third["status"] == "leased"
    company_hit = third["company_judgment_cache"]["hits"][0]
    assert inherited_exclusion in company_hit["evidence_doc"]["runner_authority_exclusions"]


def test_terminal_signal_is_reused_while_original_score_is_still_leased(store, database):
    rid = "arena-2026-09-11-l503"
    prior.open_new(store, round_id=rid, markers_by_submission=[["one"], ["two"]])
    config = _enable_round(database, rid)
    scope = _scope(rid, "inflight")
    scope["scorer_image_digest"] = config["scorer_image_digest"]
    scope["scorer_image_reference"] = config["scorer_image_reference"]
    scope.pop("cache_key")
    scope["cache_key"] = contracts.document_hash(scope)
    runner = company.hotkey("signal503-inflight")
    source, source_token, _, _ = company.claim(
        store, rid, runner, parallelism=2, excluded=[runner]
    )
    recipient, recipient_token, _, _ = company.claim(
        store, rid, runner, parallelism=2, excluded=[runner]
    )
    assert source["status"] == recipient["status"] == "leased"
    miss = store.claim_signal_judgment(
        run_id=source["run_id"], lease_token_hash=hash_lease_token(source_token),
        company_index=0, signal_index=0, scope=scope,
    )
    assert miss["status"] == "miss"
    judgment = {"raw": 1, "decision": "terminal"}
    assert store.complete_signal_judgment(
        run_id=source["run_id"], lease_token_hash=hash_lease_token(source_token),
        company_index=0, signal_index=0, scope=scope,
        reference=miss["reference"], judgment=judgment,
        judgment_hash=contracts.document_hash(judgment),
    )["status"] == "stored"
    assert store.get_run(source["run_id"])["status"] == "leased"
    hit = store.claim_signal_judgment(
        run_id=recipient["run_id"], lease_token_hash=hash_lease_token(recipient_token),
        company_index=0, signal_index=0, scope=scope,
    )
    assert hit["status"] == "hit" and hit["judgment"] == judgment
    assert store.get_signal_judgment_uses(recipient["run_id"])[0]["source_score_run_id"] == source["run_id"]
    assert store.get_run(source["run_id"])["status"] == "leased"


def test_competing_claim_waits_then_release_and_service_cannot_write_table(store, database):
    rid = "arena-2026-09-11-t503"
    prior.open_new(store, round_id=rid, markers_by_submission=[["one"], ["two"]])
    config = _enable_round(database, rid)
    scope = _scope(rid, "shared")
    scope["scorer_image_digest"] = config["scorer_image_digest"]
    scope["scorer_image_reference"] = config["scorer_image_reference"]
    scope.pop("cache_key")
    scope["cache_key"] = contracts.document_hash(scope)
    runner = company.hotkey("signal503-concurrent")
    first, token1, _, _ = company.claim(store, rid, runner, parallelism=2, excluded=[runner])
    second, token2, _, _ = company.claim(store, rid, runner, parallelism=2, excluded=[runner])
    assert first["status"] == second["status"] == "leased"
    args = {"company_index": 0, "signal_index": 0, "scope": scope}
    barrier = threading.Barrier(2)
    def race(run_id, token):
        barrier.wait(timeout=5)
        return store.claim_signal_judgment(
            run_id=run_id, lease_token_hash=hash_lease_token(token), **args
        )
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(race, first["run_id"], token1)
        b = pool.submit(race, second["run_id"], token2)
        results = [a.result(), b.result()]
    assert sorted(result["status"] for result in results) == ["busy", "miss"]
    first_claim = results[0] if results[0]["status"] == "miss" else results[1]
    owner, owner_token = (first, token1) if results[0]["status"] == "miss" else (second, token2)
    waiter, waiter_token = (second, token2) if results[0]["status"] == "miss" else (first, token1)
    assert store.complete_signal_judgment(
        run_id=owner["run_id"], lease_token_hash=hash_lease_token(owner_token),
        reference=first_claim["reference"], judgment=None, judgment_hash="", **args
    )["status"] == "released"
    assert store.claim_signal_judgment(
        run_id=waiter["run_id"], lease_token_hash=hash_lease_token(waiter_token), **args
    )["status"] == "miss"
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute("SET ROLE lab_arena_service")
        with pytest.raises(database[0].Error, match="permission denied"):
            cur.execute("INSERT INTO public.lab_arena_signal_judgments(cache_key) VALUES ('x')")


def test_excluded_recipient_gets_distinct_authority_slot(store, database):
    rid = "arena-2026-09-11-u503"
    participants, _ = prior.open_new(
        store, round_id=rid, markers_by_submission=[["one"], ["two"]]
    )
    config = _enable_round(database, rid)
    scope = _scope(rid, "same")
    scope["scorer_image_digest"] = config["scorer_image_digest"]
    scope["scorer_image_reference"] = config["scorer_image_reference"]
    scope.pop("cache_key")
    scope["cache_key"] = contracts.document_hash(scope)
    runner = company.hotkey("signal503-exclusion-runner")
    # Excluding participant two from the first claim also records its hotkey
    # in the original authority family.
    second_miner = participants[1]["miner_hotkey"]
    first, token1, _, _ = company.claim(store, rid, runner,
                                       excluded=[runner, second_miner])
    assert first["status"] == "leased"
    miss = store.claim_signal_judgment(
        run_id=first["run_id"], lease_token_hash=hash_lease_token(token1),
        company_index=0, signal_index=0, scope=scope,
    )
    judgment = {"raw": 1}
    assert store.complete_signal_judgment(
        run_id=first["run_id"], lease_token_hash=hash_lease_token(token1),
        company_index=0, signal_index=0, scope=scope,
        reference=miss["reference"], judgment=judgment,
        judgment_hash=contracts.document_hash(judgment),
    )["status"] == "stored"
    source = store.get_signal_judgment(scope["cache_key"], 0)
    assert second_miner in source["runner_authority_exclusions"]
    assert store.complete_attempt(
        run_id=first["run_id"], lease_token_hash=hash_lease_token(token1),
        result={"terminal_status": "judge_error"}, terminal_cause="judge_error",
        output_ref="", company_judgment_evidence=[], output_hash="",
        completion_request_hash="sha256:" + "e" * 64,
    )["status"] == "failed"
    second, token2, _, _ = company.claim(store, rid, runner, excluded=[runner])
    assert second["status"] == "leased" and second["miner_hotkey"] == second_miner
    fresh = store.claim_signal_judgment(
        run_id=second["run_id"], lease_token_hash=hash_lease_token(token2),
        company_index=0, signal_index=0, scope=scope,
    )
    assert fresh["status"] == "miss" and fresh["reference"]["authority_slot"] == 1


def test_live_renewed_owner_keeps_reservation_even_if_snapshot_is_stale(store, database):
    rid = "arena-2026-09-11-v503"
    prior.open_new(store, round_id=rid, markers_by_submission=[["one"], ["two"]])
    config = _enable_round(database, rid)
    scope = _scope(rid, "renewed")
    scope["scorer_image_digest"] = config["scorer_image_digest"]
    scope["scorer_image_reference"] = config["scorer_image_reference"]
    scope.pop("cache_key")
    scope["cache_key"] = contracts.document_hash(scope)
    runner = company.hotkey("signal503-renewed")
    owner, owner_token, _, _ = company.claim(store, rid, runner, parallelism=2, excluded=[runner])
    waiter, waiter_token, _, _ = company.claim(store, rid, runner, parallelism=2, excluded=[runner])
    args = {"company_index": 0, "signal_index": 0, "scope": scope}
    miss = store.claim_signal_judgment(
        run_id=owner["run_id"], lease_token_hash=hash_lease_token(owner_token), **args
    )
    assert miss["status"] == "miss"
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        # A stale snapshot must not override the current live score lease.
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()+interval '1 hour' WHERE run_id=%s", (owner["run_id"],))
        cur.execute("UPDATE public.lab_arena_signal_judgment_reservations SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s", (owner["run_id"],))
    assert store.claim_signal_judgment(
        run_id=waiter["run_id"], lease_token_hash=hash_lease_token(waiter_token), **args
    )["status"] == "busy"
    judgment = {"raw": "finished"}
    assert store.complete_signal_judgment(
        run_id=owner["run_id"], lease_token_hash=hash_lease_token(owner_token),
        reference=miss["reference"], judgment=judgment,
        judgment_hash=contracts.document_hash(judgment), **args
    )["status"] == "stored"


def test_default_grants_are_revoked_and_repeat_stops_before_writes(database):
    tables = (
        "public.lab_arena_signal_judgments",
        "public.lab_arena_signal_judgment_reservations",
        "public.lab_arena_signal_judgment_uses",
    )
    functions = (
        "public.lab_arena_signal_authority_exclusions_v1(text)",
        "public.lab_arena__signal_lease_valid_v1(public.lab_arena_runs,text)",
        "public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)",
        "public.lab_arena_complete_signal_judgment_v1(text,text,integer,integer,jsonb,jsonb,jsonb,text)",
        "public.lab_arena_signal_judgment_lease_sync_v1()",
    )
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for role in ("anon", "authenticated", "service_role"):
            for table in tables:
                cur.execute("SELECT has_table_privilege(%s,%s,'SELECT'),has_table_privilege(%s,%s,'INSERT')",
                            (role, table, role, table))
                assert cur.fetchone() == (False, False)
            for function in functions:
                cur.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, function))
                assert cur.fetchone() == (False,)
        for table in tables:
            cur.execute("SELECT has_table_privilege('lab_arena_service',%s,'SELECT'),has_table_privilege('lab_arena_service',%s,'INSERT')", (table, table))
            assert cur.fetchone() == (table != tables[1], False)
        for function in functions:
            cur.execute("SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE')", (function,))
            assert cur.fetchone() == (function not in (functions[1], functions[4]),)
        cur.execute("SELECT pg_get_functiondef('public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)'::regprocedure)")
        before = cur.fetchone()[0]
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        with pytest.raises(database[0].Error, match="already or partly installed"):
            cur.execute((SCRIPTS / "503-lab-arena-signal-judgment-cache.sql").read_text())
        conn.rollback()
        cur.execute("SELECT pg_get_functiondef('public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)'::regprocedure)")
        assert cur.fetchone()[0] == before


def test_historical_company_completion_keeps_original_exclusion_order(store, database):
    rid = "arena-2026-09-11-h503"
    _, items = prior.open_new(store, round_id=rid, markers_by_submission=[["historical"]])
    runner = company.hotkey("historical-runner")
    leased, token, _, _ = company.claim(store, rid, runner, excluded=[runner])
    assert leased["status"] == "leased"
    original = sorted([
        runner, company.hotkey("historical-other-a"),
        company.hotkey("historical-other-z"),
    ], reverse=True)
    assert original != sorted(original)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_runs SET claim_response=jsonb_set(claim_response,'{runner_authority_exclusions}',%s::jsonb) WHERE run_id=%s",
                    (json.dumps(original), leased["run_id"]))
    leased["runner_authority_exclusions"] = original
    assert store.signal_authority_exclusions(leased["run_id"]) == original
    refs = items[0]["company_judgment_refs"]
    assert prior.complete_subset(store, leased, token, refs)["company_judgments_stored"] == 1
    cached = store.get_company_judgment(refs[0]["cache_key"], 0)
    assert cached["evidence_doc"]["runner_authority_exclusions"] == original
