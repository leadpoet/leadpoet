"""Exact score-provider request reuse in disposable PostgreSQL."""

from __future__ import annotations

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lab_arena import broker as br, contracts, scoring
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, round_config
from tests.lab_arena.test_lab_arena_broker import FakeTransport, price_table


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = "391-lab-arena-verifier-provider-request-cache.sql"
LEASE_MIGRATION = "392-lab-arena-judgment-cache-extended-lease.sql"
SINGLE_TOOL_MIGRATION = "518-lab-arena-single-tool-response-cache.sql"
OWN_CALL_MIGRATION = "550-lab-arena-tool-reply-own-call-cache.sql"
ROUND = "arena-2099-10-02"
DATE = "2099-10-02"
POLICY = scoring.build_scorer_policy(
    judge_models={"test_verdict": "openai/gpt-4o-mini"}
)
RUNS = ("baseline", "miner_a", "miner_b", "miner_c", "execute")


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        tuple(
            migration for migration in CURRENT_SERVICE_MIGRATIONS
            if migration not in (MIGRATION, LEASE_MIGRATION)
        ) + (
            "264-lab-arena-codex-cost-reconciliation.sql",
            "289-lab-arena-per-icp-cost-policy.sql",
            "311-lab-arena-per-icp-closed-billing-reconciliation.sql",
            "312-lab-arena-temporary-hold-admission.sql",
            "314-lab-arena-openrouter-web-search-reservation.sql",
            "319-lab-arena-quota-sourcing-cost.sql",
            "321-lab-arena-confirmed-cost-admission.sql",
            "329-lab-arena-explicit-90m-lease.sql",
            "354-lab-arena-60m-lease.sql",
            MIGRATION,
            LEASE_MIGRATION,
            SINGLE_TOOL_MIGRATION,
            OWN_CALL_MIGRATION,
        )
    )


@pytest.fixture()
def seeded(database):
    psycopg2, dsn = database
    config = round_config(ROUND, [hotkey("cache-runner")])
    config["scorer_policy"] = POLICY
    config["baseline_hotkey"] = hotkey("cache-baseline")
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "TRUNCATE public.lab_arena_ledger, public.lab_arena_runs, "
                "public.lab_arena_submissions, public.lab_arena_rounds "
                "RESTART IDENTITY CASCADE"
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id,status,stage_generation,configuration_doc,"
                "evaluation_date) VALUES (%s,'stage1_scoring',1,%s::jsonb,%s)",
                (ROUND, json.dumps(config), DATE),
            )
            for label in RUNS:
                baseline = label == "baseline"
                kind = "execute" if label == "execute" else "score"
                submission = (
                    "baseline-" + ROUND.removeprefix("arena-")
                    if baseline else "cache-submission-" + label
                )
                miner = (
                    hotkey("cache-baseline") if baseline
                    else hotkey("cache-" + label)
                )
                cursor.execute(
                    "INSERT INTO public.lab_arena_submissions "
                    "(submission_id,round_id,miner_hotkey,status,is_king) "
                    "VALUES (%s,%s,%s,'frozen',%s)",
                    (submission, ROUND, miner, baseline),
                )
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs "
                    "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                    "stage,icp_position,attempt,kind,status,runner_hotkey,"
                    "lease_token_hash,lease_generation,stage_generation,"
                    "lease_expires_at,scored_run_id) "
                    "VALUES (%s,%s,%s,%s,%s,1,0,1,%s,'leased',%s,%s,1,1,"
                    "clock_timestamp()+interval '1 hour',%s)",
                    (
                        "cache-run-" + label, "cache-assignment-" + label,
                        ROUND, submission, miner, kind, hotkey("cache-runner"),
                        _sha("lease-" + label), "executed-" + label,
                    ),
                )
            cursor.execute("SET session_replication_role=origin")
    return database


def _scope(request: str, operation: str = "openrouter.chat") -> dict:
    return {
        "schema_version": "leadpoet.lab_arena.verifier_provider_request.v1",
        "round_id": ROUND,
        "evaluation_date": DATE,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": (
            "registry.example/lab/scorer@sha256:" + "a" * 64
        ),
        "scorer_policy": POLICY,
        "requested_operation_id": operation,
        "effective_operation_id": operation,
        "request_hash": _sha("request-" + request),
        "outbound_body_hash": _sha("outbound-" + request),
    }


def _call_doc(request: str, operation: str = "openrouter.chat") -> dict:
    scope = _scope(request, operation)
    canonical = contracts.canonical_json(scope)
    return {
        "request_hash": scope["request_hash"],
        "judgment_cache_scope": scope,
        "judgment_cache_canonical": canonical,
        "judgment_cache_key": _sha(canonical),
    }


def _call(database, function: str, *args):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET ROLE lab_arena_service")
            cursor.execute(
                "SELECT public.%s(%s)" % (
                    function, ",".join("%s" for _ in args)
                ), args,
            )
            return cursor.fetchone()[0]


def _reserve(database, label: str, request: str, action: str,
             *, call_doc=None, funding=None, amount=500, ttl=120):
    document = _call_doc(request) if call_doc is None else call_doc
    return _call(
        database, "lab_arena_reserve_judgment_call",
        "cache-run-" + label, _sha("lease-" + label), _sha(action),
        document["judgment_cache_scope"]["requested_operation_id"], "openrouter",
        funding or ("host" if label == "baseline" else "miner_key"),
        amount, json.dumps(document), ttl,
    )


def _terminal(*, eligible=True, status=200, body=None):
    if body is None:
        body = {"choices": [{"message": {"content": "verified"}}]}
    document = {
        "status": status,
        "headers": {"content-type": "application/json"},
        "body_b64": base64.b64encode(json.dumps(body).encode()).decode(),
        "call_succeeded": status == 200,
    }
    if eligible:
        document["judgment_cache_eligible"] = True
    return document


def _dispatch(database, label: str, action: str):
    return _call(
        database, "lab_arena_mark_dispatched", "cache-run-" + label,
        _sha("lease-" + label), _sha(action),
    )


def _settle(database, label: str, action: str, terminal: dict, amount=120):
    return _call(
        database, "lab_arena_settle_call", "cache-run-" + label,
        _sha("lease-" + label), _sha(action), amount,
        json.dumps(terminal), 120,
    )


def _ledger(database, action: str):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT entry_kind,amount_microusd,entry_doc,terminal_response "
                "FROM public.lab_arena_ledger WHERE call_identity=%s "
                "ORDER BY entry_id", (_sha(action),),
            )
            return cursor.fetchall()


def test_cross_payer_hit_is_run_bound_and_charges_source_once(seeded):
    first = _reserve(seeded, "baseline", "same", "paid")
    assert first["status"] == "reserved"
    assert _dispatch(seeded, "baseline", "paid")["status"] == "dispatched"
    source_terminal = _terminal()
    source_terminal["provider_cost"] = {
        "basis": "native", "units": "0.000120", "unit_name": "usd",
        "operation": "openrouter.chat",
    }
    assert _settle(seeded, "baseline", "paid", source_terminal)["status"] == "settled"

    hit = _reserve(seeded, "miner_a", "same", "hit-a")
    assert hit["status"] == "cache_hit"
    assert hit["amount_microusd"] == hit["actual_microusd"] == 0
    assert hit["source_call_identity"] == _sha("paid")
    assert hit["terminal_response"]["body_b64"] == source_terminal["body_b64"]
    assert "provider_cost" not in hit["terminal_response"]
    assert hit["terminal_response"]["judgment_cache_source_run_id"] == (
        "cache-run-baseline"
    )
    assert _reserve(seeded, "miner_a", "same", "hit-a")["status"] == "settled"
    assert _reserve(seeded, "miner_a", "same", "same-run-next")["status"] == "cache_hit"
    assert _reserve(seeded, "miner_b", "same", "hit-b")["status"] == "cache_hit"
    assert _reserve(seeded, "miner_c", "different", "different")["status"] == "reserved"
    assert [row[1] for row in _ledger(seeded, "paid")] == [500, 500, 120]
    assert [row[1] for row in _ledger(seeded, "hit-a")] == [0, 0]
    assert _ledger(seeded, "hit-a")[0][2]["judgment_cache_source_call_identity"] == _sha("paid")

    wrong = _call_doc("different")
    with pytest.raises(Exception, match="lab_arena_judgment_cache_identity_conflict"):
        _reserve(seeded, "miner_a", "different", "hit-a", call_doc=wrong)
    assert _reserve(seeded, "execute", "same", "execute-poison")["status"] == "stale"
    assert not _ledger(seeded, "execute-poison")


def test_two_connections_serialize_same_key_but_not_different_key(seeded):
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(_reserve, seeded, "miner_a", "race", "race-a")
        second = workers.submit(_reserve, seeded, "miner_b", "race", "race-b")
        states = [first.result(), second.result()]
    assert sorted(state["status"] for state in states) == ["cache_busy", "reserved"]
    source_label, source_action = (
        ("miner_a", "race-a") if states[0]["status"] == "reserved"
        else ("miner_b", "race-b")
    )
    follower_label, follower_action = (
        ("miner_b", "race-b") if source_label == "miner_a"
        else ("miner_a", "race-a")
    )
    assert _dispatch(seeded, source_label, source_action)["status"] == "dispatched"
    assert _settle(seeded, source_label, source_action, _terminal())["status"] == "settled"
    assert _reserve(seeded, follower_label, "race", follower_action)["status"] == "cache_hit"
    with ThreadPoolExecutor(max_workers=2) as workers:
        a = workers.submit(_reserve, seeded, "miner_a", "other-a", "other-a")
        b = workers.submit(_reserve, seeded, "miner_b", "other-b", "other-b")
        assert a.result()["status"] == b.result()["status"] == "reserved"


def test_promoted_baseline_uses_miner_payer_and_replays_to_other_miner(seeded):
    psycopg2, dsn = seeded
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET champion_funding_frozen=TRUE,"
                "champion_submission_id=%s,champion_hotkey=%s WHERE round_id=%s",
                ("cache-submission-miner_a", hotkey("cache-miner_a"), ROUND),
            )
            cursor.execute("SET session_replication_role=origin")
    source = _reserve(
        seeded, "baseline", "champion", "champion-paid", funding="miner_key"
    )
    assert source["status"] == "reserved"
    _dispatch(seeded, "baseline", "champion-paid")
    _settle(seeded, "baseline", "champion-paid", _terminal())
    hit = _reserve(seeded, "miner_b", "champion", "champion-hit")
    assert hit["status"] == "cache_hit"
    assert hit["source_run_id"] == "cache-run-baseline"
    assert _ledger(seeded, "champion-paid")[0][2]["judgment_cache_key"] == (
        _ledger(seeded, "champion-hit")[0][2]["judgment_cache_key"]
    )


def test_complete_reply_survives_pending_and_reconciled_cost(seeded):
    assert _reserve(seeded, "miner_a", "pending", "pending-source")["status"] == "reserved"
    assert _dispatch(seeded, "miner_a", "pending-source")["status"] == "dispatched"
    pending = _terminal()
    uncertain = _call(
        seeded, "lab_arena_mark_uncertain", "cache-run-miner_a",
        _sha("lease-miner_a"), _sha("pending-source"),
        json.dumps({
            "reason": "missing_provider_cost",
            "openrouter_generation_id": "gen-cache-pending",
            "credential_fingerprint": _sha("credential"),
            "judgment_cache_response": pending,
        }), 120,
    )
    assert uncertain["status"] == "uncertain"
    own_pending = _reserve(seeded, "miner_a", "pending", "pending-source")
    assert own_pending["status"] == "uncertain"
    assert own_pending["judgment_cache_pending_reply"] is True
    assert own_pending["terminal_response"]["body_b64"] == pending["body_b64"]
    hit = _reserve(seeded, "miner_b", "pending", "pending-hit")
    assert hit["status"] == "cache_hit"
    assert hit["terminal_response"]["body_b64"] == pending["body_b64"]

    psycopg2, dsn = seeded
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT entry_id FROM public.lab_arena_ledger "
                "WHERE call_identity=%s AND entry_kind='uncertain'",
                (_sha("pending-source"),),
            )
            uncertain_id = cursor.fetchone()[0]
    reconciled = _call(
        seeded, "lab_arena_reconcile_openrouter_cost_v1", ROUND,
        "cache-run-miner_a", _sha("pending-source"), uncertain_id,
        "gen-cache-pending", _sha("credential"), 140, "0.000140",
    )
    assert reconciled["status"] == "settled"
    own_reconciled = _reserve(seeded, "miner_a", "pending", "pending-source")
    assert own_reconciled["status"] == "settled"
    assert own_reconciled["amount_microusd"] == 140
    assert own_reconciled["terminal_response"]["body_b64"] == pending["body_b64"]
    later = _reserve(seeded, "miner_c", "pending", "pending-later")
    assert later["status"] == "cache_hit"
    assert later["terminal_response"]["body_b64"] == pending["body_b64"]
    assert [row[0] for row in _ledger(seeded, "pending-source")] == [
        "reservation", "dispatch", "uncertain", "settlement"
    ]
    assert _ledger(seeded, "pending-source")[-1][1] == 140
    assert _ledger(seeded, "pending-hit")[-1][1] == 0
    assert _ledger(seeded, "pending-later")[-1][1] == 0


def test_cached_hosted_search_does_not_hold_remaining_budget(seeded):
    document = _call_doc("hosted-search", "openrouter.responses")
    document["reserve_remaining_budget"] = True
    assert _reserve(
        seeded, "miner_a", "hosted-search", "search-source",
        call_doc=document,
    )["status"] == "reserved"
    _dispatch(seeded, "miner_a", "search-source")
    _settle(seeded, "miner_a", "search-source", _terminal(), amount=120)
    hit = _reserve(
        seeded, "miner_b", "hosted-search", "search-hit",
        call_doc=document,
    )
    assert hit["status"] == "cache_hit"
    reservation, settlement = _ledger(seeded, "search-hit")
    assert reservation[1] == settlement[1] == 0
    assert "reserve_remaining_budget" not in reservation[2]


def test_failures_malformed_replies_and_expired_claims_do_not_poison_key(seeded):
    assert _reserve(seeded, "miner_a", "failed", "failed-source")["status"] == "reserved"
    _dispatch(seeded, "miner_a", "failed-source")
    _settle(seeded, "miner_a", "failed-source", _terminal(eligible=False, status=503))
    assert _reserve(seeded, "miner_b", "failed", "failed-retry")["status"] == "reserved"

    assert _reserve(seeded, "miner_a", "malformed", "malformed-source")["status"] == "reserved"
    _dispatch(seeded, "miner_a", "malformed-source")
    bad = _terminal()
    bad["body_b64"] = "not base64"
    _settle(seeded, "miner_a", "malformed-source", bad)
    assert _reserve(seeded, "miner_b", "malformed", "malformed-retry")["status"] == "reserved"

    assert _reserve(seeded, "miner_a", "expiry", "expired-source")["status"] == "reserved"
    assert _reserve(seeded, "miner_b", "expiry", "expired-follower")["status"] == "cache_busy"
    psycopg2, dsn = seeded
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET lease_expires_at="
                "clock_timestamp()-interval '1 second' WHERE run_id=%s",
                ("cache-run-miner_a",),
            )
            cursor.execute("SET session_replication_role=origin")
    assert _reserve(seeded, "miner_b", "expiry", "expired-follower")["status"] == "reserved"


def test_acl_and_extended_lease_migration_replay(seeded):
    psycopg2, dsn = seeded
    signature = (
        "public.lab_arena_reserve_judgment_call(text,text,text,text,text,"
        "text,bigint,jsonb,integer)"
    )
    sql = (ROOT / "scripts" / LEASE_MIGRATION).read_text(encoding="utf-8")
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE'),"
                "has_function_privilege('authenticated',%s,'EXECUTE'),"
                "has_table_privilege('authenticated','public.lab_arena_ledger','SELECT')",
                (signature, signature),
            )
            assert cursor.fetchone() == (True, False, False)
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            before = cursor.fetchone()[0]
            cursor.execute(sql)
            cursor.execute(sql)
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            assert cursor.fetchone()[0] == before


def test_extended_lease_rejects_other_values(seeded):
    for ttl in (59, 3601, 4499, 4501, 6299, 6301):
        with pytest.raises(Exception, match="lab_arena_judgment_cache_input_invalid"):
            _reserve(seeded, "baseline", "invalid-ttl", "invalid-ttl", ttl=ttl)
    assert not _ledger(seeded, "invalid-ttl")


@pytest.mark.parametrize("lease_ttl_seconds", (120, 4500, 6300))
def test_broker_replays_exact_score_request_through_real_ledger(
    seeded, lease_ttl_seconds,
):
    psycopg2, dsn = seeded
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    store = ArenaStore(transport)
    provider = FakeTransport([(
        200,
        {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"verdict":"verified"}'},
            }],
            "usage": {"cost": "0.000021"},
        },
    )])
    frozen_scope = {
        "round_id": ROUND,
        "evaluation_date": DATE,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": (
            "registry.example/lab/scorer@sha256:" + "a" * 64
        ),
        "scorer_policy": POLICY,
    }
    broker = br.Broker(
        store=store,
        key_for=lambda _provider: "host-test-credential",
        credential_for=lambda context, _provider: (
            "host-test-credential" if context.submission_id.startswith("baseline-")
            else "miner-test-credential"
        ),
        funding_source_for=lambda context: (
            "host" if context.submission_id.startswith("baseline-")
            else "miner_key"
        ),
        provider_funding_source_for=lambda context, _provider: (
            "host" if context.submission_id.startswith("baseline-")
            else "miner_key"
        ),
        price_table=price_table(),
        judge_models=["openai/gpt-4o-mini"],
        transport=provider,
        lease_ttl_seconds=lease_ttl_seconds,
    )
    parameters = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "same exact claim"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "verdict", "strict": True,
                "schema": {
                    "type": "object", "additionalProperties": False,
                    "required": ["verdict"],
                    "properties": {"verdict": {
                        "type": "string", "enum": ["verified", "contradicted"],
                    }},
                },
            },
        },
        "max_tokens": 200,
    }

    def context(label: str) -> br.RunContext:
        return br.RunContext(
            run_id="cache-run-" + label,
            assignment_id="cache-assignment-" + label,
            icp_position=0,
            lease_token_hash=_sha("lease-" + label),
            miner_hotkey=(
                hotkey("cache-baseline") if label == "baseline"
                else hotkey("cache-" + label)
            ),
            submission_id=(
                "baseline-" + ROUND.removeprefix("arena-")
                if label == "baseline" else "cache-submission-" + label
            ),
            stage=1, kind="score", attempt=1, round_id=ROUND,
            judgment_cache_scope=frozen_scope,
        )

    try:
        first = broker.execute(
            context("baseline"), operation_id="openrouter.chat",
            parameters=parameters, action_sequence=1, timeout_ms=5000,
        )
        second = broker.execute(
            context("miner_a"), operation_id="openrouter.chat",
            parameters=parameters, action_sequence=1, timeout_ms=5000,
        )
        assert first.status == second.status == 200
        assert first.body == second.body
        assert first.call["actual_microusd"] == 21
        assert second.call["actual_microusd"] == 0
        assert second.call["cached"] is True
        assert len(provider.sent) == 1
        source_call = first.call["call_identity"]
        hit_call = second.call["call_identity"]
        assert source_call != hit_call
        with psycopg2.connect(**dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT amount_microusd,entry_doc,terminal_response "
                    "FROM public.lab_arena_ledger WHERE call_identity=%s "
                    "AND entry_kind='settlement'",
                    (hit_call,),
                )
                amount, entry_doc, terminal = cursor.fetchone()
        assert amount == 0
        assert entry_doc["judgment_cache_source_call_identity"] == source_call
        assert terminal["judgment_cache_source_run_id"] == "cache-run-baseline"
    finally:
        transport.close()


@pytest.mark.parametrize("single_required,count", [(True, 3), (True, 2), (True, 1), (False, 3)])
def test_historical_multi_source_respects_explicit_single_tool_and_preserves_paid_identity(
    seeded, single_required, count,
):
    request, action = "historical-tools", "historical-paid"
    assert _reserve(seeded, "baseline", request, action)["status"] == "reserved"
    assert _dispatch(seeded, "baseline", action)["status"] == "dispatched"
    body = {"choices": [{"finish_reason": "tool_calls", "message": {
        "tool_calls": [{"id": str(i), "type": "function", "function": {
            "name": "fetch_page", "arguments": '{"url":"https://acme.example"}',
        }} for i in range(count)],
    }}]}
    assert _settle(seeded, "baseline", action, _terminal(body=body))["status"] == "settled"
    original = _ledger(seeded, action)
    doc = _call_doc(request)
    if single_required:
        doc["judgment_cache_single_tool_required"] = True
    destination = _reserve(seeded, "miner_a", request, "destination", call_doc=doc)
    skip = single_required and count > 1
    assert destination["status"] == ("reserved" if skip else "cache_hit")
    if skip:
        # Exactly this new paid identity dispatches; the rejected paid source is retained.
        assert _dispatch(seeded, "miner_a", "destination")["status"] == "dispatched"
        body["choices"][0]["message"]["tool_calls"] = body["choices"][0]["message"]["tool_calls"][:1]
        _settle(seeded, "miner_a", "destination", _terminal(body=body))
        hit = _reserve(seeded, "miner_b", request, "valid-hit", call_doc=doc)
        assert hit["status"] == "cache_hit"
        assert hit["source_call_identity"] == _sha("destination")
        assert sum(row[0] == "dispatch" for row in _ledger(seeded, "destination")) == 1
    # Same paid identity never becomes another dispatch, even under the new source gate.
    replay = _reserve(seeded, "baseline", request, action, call_doc=doc)
    assert replay["status"] == "settled"
    assert _ledger(seeded, action) == original


def test_own_call_migration_exact_replay_and_tamper_fail_closed(seeded):
    psycopg2, dsn = seeded
    signature = "public.lab_arena_reserve_judgment_call(text,text,text,text,text,text,bigint,jsonb,integer)"
    sql = (ROOT / "scripts" / OWN_CALL_MIGRATION).read_text()
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            before = cursor.fetchone()[0]
            cursor.execute(sql)
            cursor.execute(sql)
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            assert cursor.fetchone()[0] == before
            cursor.execute("BEGIN")
            cursor.execute(before.replace("-- A native success is still paid", "-- Tampered native success is still paid"))
            with pytest.raises(Exception, match="tool-reply cache source differs"):
                cursor.execute(sql)
            connection.rollback()


def test_real_broker_skips_legacy_multi_cache_and_dispatches_once(seeded, monkeypatch):
    from tests.lab_arena.test_verifier_provider_cache import (
        _research_tool_answer, _research_tool_request,
    )
    psycopg2, dsn = seeded
    sql_transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    provider = FakeTransport([
        (200, _research_tool_answer(("fetch_page", "fetch_page", "search_web"))),
        (200, _research_tool_answer(("search_web",))),
        (200, _research_tool_answer(("search_web",))),
    ])
    instance = br.Broker(
        store=ArenaStore(sql_transport), key_for=lambda _: "test-credential",
        credential_for=lambda _context, _provider: "test-credential",
        funding_source_for=lambda context: "host" if context.run_id.endswith("baseline") else "miner_key",
        provider_funding_source_for=lambda context, _: "host" if context.run_id.endswith("baseline") else "miner_key",
        price_table=price_table(), judge_models=["openai/gpt-4o-mini"], transport=provider,
    )
    frozen = {"round_id": ROUND, "evaluation_date": DATE,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": "registry.example/lab/scorer@sha256:" + "a" * 64,
        "scorer_policy": POLICY}
    def context(label):
        return br.RunContext(run_id="cache-run-" + label,
            assignment_id="cache-assignment-" + label, icp_position=0,
            lease_token_hash=_sha("lease-" + label), miner_hotkey=hotkey("cache-" + label),
            submission_id=("baseline-" + ROUND.removeprefix("arena-") if label == "baseline"
                           else "cache-submission-" + label), stage=1, kind="score",
            attempt=1, round_id=ROUND, judgment_cache_scope=frozen)
    def call(label):
        return instance.execute(context(label), operation_id="openrouter.chat",
            parameters=_research_tool_request(False), action_sequence=1, timeout_ms=5000)
    try:
        # Produce historical authority through the real append-only ledger,
        # with exactly the pre-fix response eligibility behavior.
        complete = br._complete_declared_tool_calls
        with monkeypatch.context() as legacy:
            legacy.setattr(br, "_judgment_requires_own_call", lambda _: False)
            legacy.setattr(br, "_complete_declared_tool_calls", lambda message, request:
                complete(message, {k: v for k, v in request.items()
                                   if k != "parallel_tool_calls"}))
            old = call("baseline")
        assert old.status == 200
        identity = old.call["call_identity"]
        with psycopg2.connect(**dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT row_to_json(x) FROM public.lab_arena_ledger x "
                    "WHERE call_identity=%s ORDER BY entry_id", (identity,))
                before = cursor.fetchall()
        fresh, hit, own = call("miner_a"), call("miner_b"), call("baseline")
        assert fresh.status == hit.status == own.status == 200
        assert fresh.call.get("cached") is None and hit.call.get("cached") is None
        assert len(provider.sent) == 3  # each new tool conversation has its own paid reply
        with psycopg2.connect(**dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT row_to_json(x) FROM public.lab_arena_ledger x "
                    "WHERE call_identity=%s ORDER BY entry_id", (identity,))
                assert cursor.fetchall() == before
                cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE "
                    "call_identity=%s AND entry_kind='dispatch'", (fresh.call["call_identity"],))
                assert cursor.fetchone()[0] == 1
    finally:
        sql_transport.close()


@pytest.mark.parametrize("state", ["reserved", "dispatched", "uncertain", "settled", "cached"])
def test_tool_policy_keeps_historic_own_head_and_never_reads_other_calls(seeded, state):
    request, action = "tool-policy", "old-paid"
    assert _reserve(seeded, "baseline", request, action)["status"] == "reserved"
    if state != "reserved":
        _dispatch(seeded, "baseline", action)
    if state == "uncertain":
        pending = _terminal()
        _call(seeded, "lab_arena_mark_uncertain", "cache-run-baseline",
              _sha("lease-baseline"), _sha(action), json.dumps({
                  "reason": "missing_provider_cost",
                  "openrouter_generation_id": "gen-own-tool-policy",
                  "credential_fingerprint": _sha("credential"),
                  "judgment_cache_response": pending,
              }), 120)
    elif state in {"settled", "cached"}:
        _settle(seeded, "baseline", action, _terminal())
    label = "baseline"
    if state == "cached":
        label, action = "miner_a", "old-cached"
        assert _reserve(seeded, label, request, action)["status"] == "cache_hit"
    before = _ledger(seeded, action)
    doc = {**_call_doc(request), "judgment_cache_own_call_only": True}
    own = _reserve(seeded, label, request, action, call_doc=doc)
    assert own["status"] == ("settled" if state == "cached" else state)
    assert own["idempotent"] is True
    assert _ledger(seeded, action) == before
    if state == "uncertain":
        assert own["judgment_cache_pending_reply"] is True
        assert own["terminal_response"] == pending
    if state == "cached":
        assert own["amount_microusd"] == 0
    fresh = _reserve(seeded, "miner_b", request, "new-tool-call", call_doc=doc)
    assert fresh["status"] == "reserved"  # neither cache hit nor cache busy
    assert _ledger(seeded, "new-tool-call")[0][2]["judgment_cache_own_call_only"] is True
    assert not any(row[0] == "dispatch" for row in _ledger(seeded, "new-tool-call"))


def test_tool_only_source_cannot_feed_an_older_gateway_shared_lookup(seeded):
    doc = {**_call_doc("mixed-gateway"), "judgment_cache_own_call_only": True}
    assert _reserve(seeded, "baseline", "mixed-gateway", "only-source", call_doc=doc)["status"] == "reserved"
    _dispatch(seeded, "baseline", "only-source")
    _settle(seeded, "baseline", "only-source", _terminal())
    before = _ledger(seeded, "only-source")
    assert _reserve(seeded, "miner_a", "mixed-gateway", "old-reader")["status"] == "reserved"
    assert _ledger(seeded, "only-source") == before


@pytest.mark.parametrize("invalid", [None, "true", 0, 1, [], {}])
def test_invalid_tool_cache_policy_fails_before_reservation(seeded, invalid):
    doc = {**_call_doc("invalid-policy"), "judgment_cache_own_call_only": invalid}
    with pytest.raises(Exception, match="lab_arena_judgment_cache_input_invalid"):
        _reserve(seeded, "baseline", "invalid-policy", "invalid-policy", call_doc=doc)
    assert not _ledger(seeded, "invalid-policy")
