"""New judge claims inherit verified fallback without rewriting old payers."""

import json
from pathlib import Path

import pytest

from lab_arena.store import hash_lease_token
from lab_arena.store import ArenaStoreError
from lab_arena import broker as br, contracts
from lab_arena.submission_runtime import SubmissionProviderKeys
from tests.lab_arena import concurrent_baseline_judging572_postgres_test as prior
from tests.lab_arena import company_judgment_validation_projection548_postgres_test as completed
from tests.lab_arena.firecrawl_diagnostic_adapter_test import URL, _envelope
from tests.lab_arena.test_lab_arena_broker import DL_KEY, FakeTransport, price_table
from tests.lab_arena.deepline_execution_recovery_test import exact_bill


ROOT = Path(__file__).parents[2]
SQL = ROOT / "scripts/576-lab-arena-champion-score-funding.sql"
SIGNATURES = (
    "public.lab_arena_provider_funding(text,text)",
    "public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)",
    "public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)",
)
HOST_DL_KEY = "dl_secret_" + "h" * 30


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    from scripts.verify_arena_full_thinking_e2e import full_thinking_database
    generator = full_thinking_database()
    db = next(generator)
    fixture = completed.completed.__wrapped__(db, tmp_path_factory)
    try:
        h, row = next(fixture)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute((ROOT / "scripts/575-lab-arena-score-host-continuation.sql").read_text())
            saved = {}
            for table in ("lab_arena_rounds", "lab_arena_runs", "lab_arena_ledger",
                          "lab_arena_judgment_cache"):
                cur.execute(f"SELECT to_jsonb(t) FROM public.{table} t")
                saved[table] = [item[0] for item in cur.fetchall()]
        yield db, h, row, saved
    finally:
        fixture.close()
        generator.close()


@pytest.fixture(autouse=True)
def restore(scenario):
    db = scenario[0]
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        originals = [scalar(cur, "pg_get_functiondef(%s::regprocedure)", (s,)) for s in SIGNATURES]
        cur.execute("TRUNCATE public.lab_arena_trajectory_events CASCADE")
        cur.execute("TRUNCATE public.lab_arena_company_judgment_reservations CASCADE")
    yield
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        for source in originals:
            cur.execute(source)


scalar = prior.scalar


def prepare(scenario, flagged=("deepline",)):
    db, h, _, _ = scenario
    rid, baseline, pending = prior.prepare(scenario)
    # The completed round supplies real immutable execution and judge evidence.
    # Set only its fixture's promoted owner and already-verified provider flags.
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        owner = scalar(cur, "submission_id FROM public.lab_arena_submissions "
                       "WHERE round_id=%s AND NOT is_king ORDER BY submission_id LIMIT 1", (rid,))
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_rounds SET champion_funding_frozen=true,"
                    "champion_submission_id=%s,champion_hotkey=(SELECT miner_hotkey FROM "
                    "public.lab_arena_submissions WHERE submission_id=%s),"
                    "champion_fallback_providers=%s WHERE round_id=%s",
                    (owner, owner, list(flagged), rid))
        cur.execute("SET LOCAL session_replication_role=origin")
    h.service._invalidate_hot_round()
    return rid, baseline, pending


def funding(cur, run_id, provider="deepline"):
    return scalar(cur, "public.lab_arena_provider_funding(%s,%s)", (run_id, provider))


def rows(cur, rid):
    return {table: scalar(cur, "coalesce(jsonb_agg(to_jsonb(t) ORDER BY "
                          + ("entry_id" if table == "lab_arena_ledger" else "run_id" if table == "lab_arena_runs" else "round_id")
                          + "),'[]'::jsonb) FROM public." + table + " t WHERE round_id=%s", (rid,))
            for table in ("lab_arena_rounds", "lab_arena_runs", "lab_arena_ledger")}


@pytest.mark.parametrize("flagged", [("deepline",), ("scrapingdog",), ("openrouter",)])
def test_new_claim_snapshots_only_verified_provider_and_preserves_old_runs(scenario, flagged):
    db, h, _, _ = scenario
    rid, _, pending = prepare(scenario, flagged)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        # Reproduce: the original payer remains miner-funded despite the flag.
        assert funding(cur, pending["run_id"], flagged[0])["funding_source"] == "miner_key"
        before = rows(cur, rid)
        metadata = [scalar(cur, "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s::regprocedure", (s,)) for s in SIGNATURES]
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        assert rows(cur, rid) == before
        assert [scalar(cur, "to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=%s::regprocedure", (s,)) for s in SIGNATURES] == metadata
        # An existing NULL score snapshot retains its original bill provenance.
        assert funding(cur, pending["run_id"], flagged[0])["funding_source"] == "miner_key"
    runner = h.runner(0)
    try:
        lease = runner.claim_one()
        assert lease["run_id"] == pending["run_id"]
        stored = h.service.store.get_run(lease["run_id"])
        expected = {provider: "host" if provider in flagged else "miner_key"
                    for provider in ("openrouter", "deepline", "scrapingdog")}
        assert stored["champion_funding_sources"] == expected
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            for provider, source in expected.items():
                selected = funding(cur, lease["run_id"], provider)
                assert selected["funding_source"] == source
                assert selected["restart_required"] is False
                assert (selected["credential_submission_id"] is None) == (source == "host")
            assert scalar(cur, "public.lab_arena__champion_reward_factor(%s,champion_hotkey) "
                          "FROM public.lab_arena_rounds WHERE round_id=%s", (rid, rid)) == 500000
            # Claiming does not replace execution, output references or judgments.
            after = rows(cur, rid)
            assert [r for r in after["lab_arena_runs"] if r["kind"] == "execute"] == [r for r in before["lab_arena_runs"] if r["kind"] == "execute"]
            assert after["lab_arena_ledger"] == before["lab_arena_ledger"]
    finally:
        runner.close()


def test_new_migration_rejects_drift_and_preserves_atomicity(scenario):
    db = scenario[0]
    for signature in SIGNATURES:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            sources = [scalar(cur, "pg_get_functiondef(%s::regprocedure)", (s,)) for s in SIGNATURES]
            for drift in ("source", "acl", "security"):
                cur.execute("SAVEPOINT drift576")
                if drift == "source":
                    source = scalar(cur, "pg_get_functiondef(%s::regprocedure)", (signature,))
                    cur.execute(source.replace("BEGIN", "BEGIN\n  NULL;", 1))
                elif drift == "acl":
                    cur.execute("GRANT EXECUTE ON FUNCTION " + signature + " TO anon")
                else:
                    cur.execute("ALTER FUNCTION " + signature + " SECURITY INVOKER")
                with pytest.raises(db[0].Error):
                    cur.execute(SQL.read_text())
                conn.rollback()
                assert [scalar(cur, "pg_get_functiondef(%s::regprocedure)", (s,)) for s in SIGNATURES] == sources


def score_broker(h, lease):
    round_row = h.service.store.get_round(h.round_id)

    class Credentials:
        def runtime_key(self, row, provider):
            assert provider == "deepline"
            assert row["submission_id"] == round_row["champion_submission_id"]
            assert row["miner_hotkey"] == round_row["champion_hotkey"]
            return DL_KEY

    payer = SubmissionProviderKeys(store=h.service.store, credentials=Credentials(),
                                   organizer_keys={"deepline": HOST_DL_KEY})
    transport = FakeTransport([(200, _envelope("<html>Verified source.</html>"))])
    broker = br.Broker(
        store=h.service.store, key_for=lambda _: DL_KEY,
        credential_for=payer.credential_for,
        provider_funding_source_for=payer.provider_funding_source_for,
        champion_score_for=payer.champion_score_for,
        transport=transport, price_table=price_table(),
    )
    stored = h.service.store.get_run(lease["run_id"])
    context = br.RunContext(
        run_id=lease["run_id"], assignment_id=stored["assignment_id"],
        icp_position=stored["icp_position"], attempt=stored["attempt"],
        lease_token_hash=hash_lease_token(lease["lease_token"]),
        miner_hotkey=stored["miner_hotkey"], submission_id=stored["submission_id"],
        stage=stored["stage"], kind="score", round_id=stored["round_id"],
    )
    return broker, context, transport


@pytest.mark.parametrize("migrated", [False, True], ids=["python-before-sql", "new-host-claim"])
def test_real_score_broker_uses_frozen_provider_payer_and_replays_once(scenario, migrated):
    db, h, _, _ = scenario
    rid, _, pending = prepare(scenario)
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        # The saved accepted fixture reuses the first score identifier. Remove
        # its old test bills to make this a genuine fresh score attempt.
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("DELETE FROM public.lab_arena_ledger WHERE run_id=%s", (pending["run_id"],))
        cur.execute("SET LOCAL session_replication_role=origin")
        if migrated:
            cur.execute(SQL.read_text())
    runner = h.runner(0)
    try:
        lease = runner.claim_one()
        broker, context, transport = score_broker(h, lease)
        result = broker.execute(context, operation_id="scrapingdog.scrape",
                                parameters={"url": URL}, action_sequence=721, timeout_ms=30_000)
        assert result.status == 200, result
        expected = "host" if migrated else "miner_key"
        assert result.call["provider"] == "deepline"
        assert result.call["funding_source"] == expected
        assert result.call["actual_microusd"] == 2_000
        assert result.call["adapter"] == "firecrawl_raw_html"
        assert transport.sent[0]["headers"]["authorization"] == "Bearer " + (
            HOST_DL_KEY if migrated else DL_KEY)
        ledger = h.service.store.list_ledger(run_id=lease["run_id"])
        assert len([entry for entry in ledger if entry["entry_kind"] == "reservation"]) == 1
        assert {entry["funding_source"] for entry in ledger} == {expected}
        replay = broker.execute(context, operation_id="scrapingdog.scrape",
                                parameters={"url": URL}, action_sequence=721, timeout_ms=30_000)
        assert replay.status == 200 and replay.call["idempotent"] is True
        assert len(transport.sent) == 1
        assert h.service.store.list_ledger(run_id=lease["run_id"]) == ledger
    finally:
        runner.close()


@pytest.mark.parametrize("new_snapshot", [False, True])
def test_paid_score_reconciles_with_original_key_after_later_flag(scenario, new_snapshot):
    db, h, _, _ = scenario
    rid, _, pending = prepare(scenario, flagged=())
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("DELETE FROM public.lab_arena_ledger WHERE run_id=%s", (pending["run_id"],))
        cur.execute("SET LOCAL session_replication_role=origin")
        if new_snapshot:
            cur.execute(SQL.read_text())
    runner = h.runner(0)
    try:
        lease = runner.claim_one()
        snapshot = h.service.store.get_run(lease["run_id"])["champion_funding_sources"]
        assert snapshot == ({provider: "miner_key" for provider in ("openrouter", "deepline", "scrapingdog")}
                            if new_snapshot else None)
        broker, context, _ = score_broker(h, lease)
        store = h.service.store
        identity = contracts.document_hash({"lease": lease["run_id"], "recovery": new_snapshot})
        request_id = "ctx-tool-" + identity[7:39]
        native = "public-execute:11111111-2222-4333-8444-555555555555"
        fingerprint = br._credential_fingerprint(DL_KEY)
        doc = {"request_hash": contracts.document_hash({"url": URL}),
               "tool": "firecrawl_scrape", "deepline_request_id": request_id,
               "deepline_execution_key": "arena:" + identity[7:],
               "deepline_billing_provider": "firecrawl", "credential_fingerprint": fingerprint}
        assert store.reserve_call(run_id=context.run_id, lease_token_hash=context.lease_token_hash,
            call_identity=identity, operation_id="scrapingdog.scrape", provider="deepline",
            funding_source="miner_key", amount_microusd=0, call_doc=doc)["status"] == "reserved"
        assert store.mark_dispatched(run_id=context.run_id, lease_token_hash=context.lease_token_hash,
                                     call_identity=identity)["status"] == "dispatched"
        assert store.mark_uncertain(run_id=context.run_id, lease_token_hash=context.lease_token_hash,
            call_identity=identity, call_doc={"reason": "transport_failure", "call_succeeded": False,
                "deepline_job_id": native, "deepline_request_id": request_id,
                "deepline_operation": "firecrawl_scrape", "credential_fingerprint": fingerprint})["status"] == "uncertain"
        candidates = store.list_deepline_cost_reconciliations(rid, run_id=context.run_id)
        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate["funding_source"] == "miner_key"
        assert candidate["credential_fingerprint"] == fingerprint
        ledger_before = store.list_ledger(call_identity=identity)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute(SQL.read_text())
            before = h.service.store.get_run(lease["run_id"])
            cur.execute("UPDATE public.lab_arena_rounds SET champion_fallback_providers="
                        "ARRAY['deepline']::text[] WHERE round_id=%s", (rid,))
            assert funding(cur, lease["run_id"])["funding_source"] == "miner_key"
            assert funding(cur, lease["run_id"])["restart_required"] is False
        assert h.service.store.get_run(lease["run_id"]) == before
        class ReceiptTransport:
            def __init__(self):
                self.requests = []

            def send(self, **request):
                self.requests.append(request)
                assert request["method"] == "GET"
                assert request["headers"]["authorization"] == "Bearer " + DL_KEY
                assert request["url"] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(native, safe="")
                return br.ProviderResponse(200, {}, json.dumps(exact_bill(
                    request_id=native, credits="0.02", delta="-0.02")).encode())

        receipt_transport = ReceiptTransport()
        broker._transport = receipt_transport
        settled = broker.reconcile_deepline_cost(candidate)
        assert settled["status"] == "settled" and settled["actual_microusd"] == 2000
        assert broker.reconcile_deepline_cost(candidate)["idempotent"] is True
        assert all(request["method"] == "GET" for request in receipt_transport.requests)
        after = store.list_ledger(call_identity=identity)
        assert after[:-1] == ledger_before
        assert after[-1]["entry_kind"] == "settlement"
        assert after[-1]["funding_source"] == "miner_key"
        assert store.get_run(context.run_id) == before
        # A later flag does not restart or reject another call from this
        # already-claimed immutable score snapshot.
        next_identity = contracts.document_hash({"lease": lease["run_id"], "next": new_snapshot})
        assert store.reserve_call(run_id=context.run_id, lease_token_hash=context.lease_token_hash,
            call_identity=next_identity, operation_id="deepline.execute", provider="deepline",
            funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
        with pytest.raises(ArenaStoreError):
            store.mark_champion_provider_fallback(context.run_id, context.lease_token_hash,
                "openrouter", {"error_class": "account_credential_failure", "provider_status": 401,
                               "provider_attempts": 4, "action_sequence": 0})
        assert h.service.store.get_round(rid)["champion_fallback_providers"] == ["deepline"]
    finally:
        runner.close()
