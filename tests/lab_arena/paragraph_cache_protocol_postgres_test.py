"""Real Broker/SQL cache and exact financial replay for paragraph protocol v3."""
from __future__ import annotations

import json
import copy

import pytest

from lab_arena import broker as br
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.test_lab_arena_broker import FakeTransport, price_table
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, round_config
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.verifier_provider_cache_postgres_test import (
    DATE, POLICY, ROUND, RUNS, _call, _sha,
)
from tests.lab_arena.paragraph_judgment_cache_protocol_test import answer, request, verdict, NEW_NAME, OLD_NAME


@pytest.fixture(scope="module")
def database():
    yield from current_service_database()


@pytest.fixture()
def seeded(database):
    """The existing provider-cache fixture identities on the current schema."""
    pg, dsn = database
    cfg = round_config(ROUND, [hotkey("cache-runner")])
    cfg.update(scorer_policy=POLICY, baseline_hotkey=hotkey("cache-baseline"))
    with pg.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute("TRUNCATE public.lab_arena_ledger, public.lab_arena_runs, "
                       "public.lab_arena_submissions, public.lab_arena_rounds RESTART IDENTITY CASCADE")
        cursor.execute("INSERT INTO public.lab_arena_rounds "
                       "(round_id,status,stage_generation,configuration_doc,evaluation_date) "
                       "VALUES (%s,'stage1_scoring',1,%s::jsonb,%s)", (ROUND, json.dumps(cfg), DATE))
        for label in RUNS:
            baseline = label == "baseline"
            submission = "baseline-" + ROUND.removeprefix("arena-") if baseline else "cache-submission-" + label
            miner = hotkey("cache-baseline" if baseline else "cache-" + label)
            cursor.execute("INSERT INTO public.lab_arena_submissions "
                           "(submission_id,round_id,miner_hotkey,status,is_king) "
                           "VALUES (%s,%s,%s,'frozen',%s)", (submission, ROUND, miner, baseline))
            cursor.execute("INSERT INTO public.lab_arena_runs "
                "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,"
                "attempt,kind,status,runner_hotkey,lease_token_hash,lease_generation,stage_generation,"
                "lease_expires_at,scored_run_id,scoring_generation_floor) "
                "VALUES (%s,%s,%s,%s,%s,1,0,1,%s,'leased',%s,%s,1,1,"
                "clock_timestamp()+interval '1 hour',%s,0)",
                ("cache-run-" + label, "cache-assignment-" + label, ROUND, submission, miner,
                 "execute" if label == "execute" else "score", hotkey("cache-runner"),
                 _sha("lease-" + label), "executed-" + label))
        cursor.execute("SET session_replication_role=origin")
    return database


def _broker(seeded, replies):
    pg, dsn = seeded
    sql = PsycopgTransport(lambda: pg.connect(**dsn))
    provider = FakeTransport(replies)
    instance = br.Broker(store=ArenaStore(sql), key_for=lambda _: "synthetic-host-credential",
        credential_for=lambda c, _: ("synthetic-host-credential" if c.run_id.endswith("baseline")
                                     else "synthetic-miner-credential"),
        funding_source_for=lambda c: "host" if c.run_id.endswith("baseline") else "miner_key",
        provider_funding_source_for=lambda c, _: "host" if c.run_id.endswith("baseline") else "miner_key",
        price_table=price_table(), judge_models=["openai/gpt-4o-mini"], transport=provider)
    frozen = {"round_id": ROUND, "evaluation_date": DATE,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": "registry.example/lab/scorer@sha256:" + "a" * 64,
        "scorer_policy": POLICY}

    def call(label, action, operation, *, name=None):
        context = br.RunContext(run_id="cache-run-" + label, assignment_id="cache-assignment-" + label,
            icp_position=0, lease_token_hash=_sha("lease-" + label),
            miner_hotkey=hotkey("cache-baseline" if label == "baseline" else "cache-" + label),
            submission_id=("baseline-" + ROUND.removeprefix("arena-") if label == "baseline"
                           else "cache-submission-" + label), stage=1, kind="score", attempt=1,
            round_id=ROUND, judgment_cache_scope=frozen)
        parameters = request(operation) if name is None else request(operation, name=name)
        return instance.execute(context, operation_id=operation, parameters=parameters,
                                action_sequence=action, timeout_ms=5000)
    return sql, provider, call


def _rows(seeded, identity):
    pg, dsn = seeded
    with pg.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT row_to_json(x) FROM public.lab_arena_ledger x "
                           "WHERE call_identity=%s ORDER BY entry_id", (identity,))
            return [row[0] for row in cursor.fetchall()]


@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
@pytest.mark.parametrize("source_label,destination", [("baseline", "miner_a"), ("miner_a", "baseline")])
@pytest.mark.parametrize("valid_aggregate,valid_status", [(True, "VERIFIED"), (False, "UNPROVEN")])
def test_invalid_paid_response_does_not_block_fresh_call_and_valid_reply_reuses_across_payers(
    seeded, operation, source_label, destination, valid_aggregate, valid_status,
):
    invalid = answer(verdict(), operation)
    valid = answer(verdict(aggregate=valid_aggregate, status=valid_status), operation)
    sql, provider, call = _broker(seeded, [(200, invalid), (200, valid)])
    try:
        first = call(source_label, 1, operation)
        assert first.status == 200 and first.call["outcome"] == "settled"
        assert first.call["actual_microusd"] == 120 and not first.call.get("cached")
        original = _rows(seeded, first.call["call_identity"])
        assert [row["entry_kind"] for row in original] == ["reservation", "dispatch", "settlement"]
        assert {row["funding_source"] for row in original} == {
            "host" if source_label == "baseline" else "miner_key"}
        assert original[-1]["terminal_response"].get("judgment_cache_eligible") is not True
        own = call(source_label, 1, operation)
        assert own.status == 200 and own.body == first.body and own.call["idempotent"] is True
        assert len(provider.sent) == 1 and _rows(seeded, first.call["call_identity"]) == original
        fresh = call(source_label, 2, operation)
        assert fresh.status == 200 and not fresh.call.get("cached") and len(provider.sent) == 2
        assert fresh.call["actual_microusd"] == 120 and fresh.body != first.body
        hit = call(destination, 1, operation)
        assert hit.status == 200 and hit.call["cached"] is True and hit.body == fresh.body
        assert hit.call["actual_microusd"] == 0 and len(provider.sent) == 2
        assert hit.call["source_call_identity"] == fresh.call["call_identity"]
        assert call(destination, 1, operation).body == hit.body and len(provider.sent) == 2
        # The original paid invalid answer stays the exact-call authority even
        # after a newer valid answer becomes available for distinct calls.
        assert call(source_label, 1, operation).body == first.body
        assert len(provider.sent) == 2 and _rows(seeded, first.call["call_identity"]) == original
        assert sum(row["amount_microusd"] for result in [first, fresh, hit]
                   for row in _rows(seeded, result.call["call_identity"])
                   if row["entry_kind"] == "settlement") == 240
    finally:
        sql.close()


@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
@pytest.mark.parametrize("label", ["baseline", "miner_a"])
@pytest.mark.parametrize("name", [NEW_NAME, OLD_NAME])
def test_invalid_pending_reply_keeps_own_body_and_exact_financial_recovery(seeded, monkeypatch, operation, label, name):
    pending = answer(verdict(), operation, cost=None)
    generation = "gen-paragraph-pending-proof"
    pending["id"] = generation
    # The exact billing GET is pending; never infer a free call or resend POST.
    monkeypatch.setattr(br, "_openrouter_generation_readback", lambda **_: None)
    sql, provider, call = _broker(seeded, [(200, pending)])
    try:
        first = call(label, 1, operation, name=name)
        assert first.status == 200 and first.call["outcome"] == "uncertain"
        assert "actual_microusd" not in first.call
        saved = _rows(seeded, first.call["call_identity"])
        assert [r["entry_kind"] for r in saved] == ["reservation", "dispatch", "uncertain"]
        uncertainty = saved[-1]
        doc = uncertainty["entry_doc"]["call"]
        assert doc["openrouter_generation_id"] == generation
        assert (doc["judgment_cache_response"].get("judgment_cache_eligible") is True) == (name == OLD_NAME)
        own = call(label, 1, operation, name=name)
        assert own.status == 200 and own.body == first.body and own.call["idempotent"] is True
        assert own.call["outcome"] == "uncertain" and "actual_microusd" not in own.call
        assert len(provider.sent) == 1 and _rows(seeded, first.call["call_identity"]) == saved
        repeated_uncertainty = ArenaStore(sql).mark_uncertain(run_id="cache-run-" + label,
            lease_token_hash=_sha("lease-" + label), call_identity=first.call["call_identity"],
            call_doc=doc, lease_ttl_seconds=120)
        assert repeated_uncertainty["status"] == "uncertain" and repeated_uncertainty["idempotent"] is True
        assert _rows(seeded, first.call["call_identity"]) == saved
        # Controlled exact read-only billing result, through the canonical SQL
        # reconciliation protocol; no provider operation is replayed.
        settled = _call(seeded, "lab_arena_reconcile_openrouter_cost_v1", ROUND,
            "cache-run-" + label, first.call["call_identity"], uncertainty["entry_id"],
            generation, doc["credential_fingerprint"], 140, "0.000140")
        assert settled["status"] == "settled"
        reconciled = call(label, 1, operation, name=name)
        assert reconciled.status == 200 and reconciled.body == first.body
        assert reconciled.call["outcome"] == "settled" and reconciled.call["actual_microusd"] == 140
        assert len(provider.sent) == 1
        history = _rows(seeded, first.call["call_identity"])
        assert [r["entry_kind"] for r in history] == ["reservation", "dispatch", "uncertain", "settlement"]
        assert history[:3] == saved and history[-1]["amount_microusd"] == 140
        if name == NEW_NAME:
            assert history[-1]["terminal_response"]["status"] == 502
            assert history[-1]["terminal_response"]["provider_cost"]["basis"] == "openrouter_generation_cost"
            # Success is an accounting classification, not evidence that the
            # 502 billing placeholder replaced the body delivered earlier.
            native_reserve = ArenaStore.reserve_judgment_call
            def successful_cost_readback(store, **arguments):
                state = copy.deepcopy(native_reserve(store, **arguments))
                assert state["status"] == "settled"
                state["terminal_response"]["call_succeeded"] = True
                return state
            with monkeypatch.context() as projected_success:
                projected_success.setattr(ArenaStore, "reserve_judgment_call", successful_cost_readback)
                success_replay = call(label, 1, operation, name=name)
            assert success_replay.status == 200 and success_replay.body == first.body
            assert success_replay.call["actual_microusd"] == 140 and len(provider.sent) == 1
            assert _rows(seeded, first.call["call_identity"]) == history
    finally:
        sql.close()


@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
def test_old_bad_namespace_remains_replayable_but_cannot_supply_new_protocol(seeded, operation):
    invalid, valid = answer(verdict(), operation), answer(verdict(aggregate=True), operation)
    sql, provider, call = _broker(seeded, [(200, invalid), (200, valid)])
    try:
        old = call("baseline", 1, operation, name=OLD_NAME)
        old_rows = _rows(seeded, old.call["call_identity"])
        legacy_hit = call("miner_a", 1, operation, name=OLD_NAME)
        assert legacy_hit.call["cached"] is True and legacy_hit.body == old.body
        fresh = call("miner_b", 1, operation)
        assert fresh.status == 200 and not fresh.call.get("cached") and len(provider.sent) == 2
        assert fresh.body != old.body
        assert call("baseline", 1, operation, name=OLD_NAME).body == old.body
        assert _rows(seeded, old.call["call_identity"]) == old_rows
    finally:
        sql.close()


@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
def test_bad_shared_source_cannot_bypass_new_protocol_readback(seeded, monkeypatch, operation):
    invalid = answer(verdict(), operation)
    sql, provider, call = _broker(seeded, [(200, invalid)])
    try:
        # Create an append-only bad source using the exact old eligibility
        # behavior; do not rewrite any ledger or invent a paid settlement.
        with monkeypatch.context() as historical:
            historical.setattr(br, "_complete_judgment_response", lambda *a, **kw: True)
            old = call("baseline", 1, operation)
        original = _rows(seeded, old.call["call_identity"])
        blocked = call("miner_a", 1, operation)
        assert blocked.status != 200 and blocked.call["error_code"] == "broker_unavailable"
        assert len(provider.sent) == 1
        assert call("baseline", 1, operation).body == old.body and len(provider.sent) == 1
        assert _rows(seeded, old.call["call_identity"]) == original
    finally:
        sql.close()
