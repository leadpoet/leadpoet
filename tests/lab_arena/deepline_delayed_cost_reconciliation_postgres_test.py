"""Exact-ID Deepline billing reconciliation and retry deferral."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import (
    ArenaStore,
    ArenaStoreError,
    PsycopgTransport,
    hash_lease_token,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    _commit_plan,
    _execute_everything,
    _scoring_items,
    claim,
    complete,
    open_round,
    sha,
)


MIGRATIONS = CURRENT_SERVICE_MIGRATIONS
MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/243-lab-arena-deepline-delayed-cost-reconciliation.sql"
)
DEFERRAL_BYPASS_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/245-lab-arena-deepline-credential-deferral-bypass.sql"
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


@pytest.fixture()
def resources(database):
    psycopg2, dsn = database
    connections = []

    def connect():
        connection = psycopg2.connect(**dsn)
        connections.append(connection)
        return connection

    store = ArenaStore(PsycopgTransport(connect), lease_ttl_seconds=120)
    yield store, connect
    store.close()
    for connection in connections:
        if not connection.closed:
            connection.close()


def _store(database):
    psycopg2, dsn = database
    return ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def _open_scoring(
    store: ArenaStore, round_id: str, *, participants: int, runners: int
):
    runner_keys, participant_rows = open_round(
        store,
        round_id,
        participants=participants,
        runners=runners,
        prefix=round_id[-8:],
        max_attempts=2,
    )
    executed = _execute_everything(store, round_id, runner_keys[0])
    assert store.close_stage(round_id, 1)["status"] == "closed"
    _commit_plan(store, round_id, 1)
    assert store.open_scoring(
        round_id, 1, _scoring_items(executed)
    )["status"] == "ok"
    return runner_keys, participant_rows


def _open_run(store: ArenaStore, label: str):
    round_id = "arena-2026-09-14-%s" % label
    runners, participants = open_round(
        store,
        round_id,
        participants=1,
        runners=1,
        prefix=label,
        execution_cap_microusd=10_000_000,
    )
    run, token, _, _ = claim(store, round_id, runners[0])
    return round_id, participants[0]["submission_id"], run, token


def _uncertain_call(
    store: ArenaStore,
    run,
    token: str,
    *,
    label: str,
    amount: int = 2_000,
    request_id: str | None = None,
    operation: str = "firecrawl_scrape",
    fingerprint: str | None = None,
    reason: str = "transport_failure",
    call_succeeded: bool = False,
    credential_failure_status: int | None = None,
    evidence_overrides: dict | None = None,
):
    fingerprint = fingerprint or "sha256:" + "a" * 64
    identity = contracts.provider_call_identity(
        attempt=run["attempt"],
        assignment_id=run["assignment_id"],
        icp_position=run["icp_position"],
        action_sequence=0,
        operation_id="scrapingdog.scrape",
        request_hash=sha(label),
    )
    request_id = request_id or "ctx-tool-" + identity.removeprefix("sha256:")[:32]
    account_failure_evidence = None
    if credential_failure_status is not None:
        account_failure_evidence = {
            "error_class": "account_credential_failure",
            "provider_status": credential_failure_status,
            "base_call_identity": identity,
            "provider_attempt": 1,
            "action_sequence": 0,
            **(evidence_overrides or {}),
        }
    token_hash = hash_lease_token(token)
    assert store.reserve_call(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
        operation_id="scrapingdog.scrape",
        provider="deepline",
        funding_source="miner_key",
        amount_microusd=amount,
        call_doc={
            "request_hash": sha(label),
            "base_call_identity": identity,
            "provider_attempt": 1,
            "action_sequence": 0,
            "tool": operation,
            "deepline_request_id": request_id,
            "credential_fingerprint": fingerprint,
        },
    )["status"] == "reserved"
    assert store.mark_dispatched(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
    )["status"] == "dispatched"
    assert store.mark_uncertain(
        run_id=run["run_id"],
        lease_token_hash=token_hash,
        call_identity=identity,
        call_doc={
            "reason": reason,
            "call_succeeded": call_succeeded,
            **(
                {"provider_status": credential_failure_status}
                if credential_failure_status is not None
                else {}
            ),
            "deepline_request_id": request_id,
            "deepline_operation": operation,
            "credential_fingerprint": fingerprint,
            **(
                {"account_failure_evidence": account_failure_evidence}
                if account_failure_evidence is not None
                else {}
            ),
        },
    )["status"] == "uncertain"
    return identity, request_id, operation, fingerprint


def _settlements(store: ArenaStore, identity: str):
    return [
        row
        for row in store.list_ledger(call_identity=identity)
        if row["entry_kind"] == "settlement"
    ]


def test_functions_are_service_only_and_owned_by_arena_owner(resources):
    _store_resource, connect = resources
    signatures = (
        "public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)",
        "public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)",
    )
    with connect() as connection, connection.cursor() as cursor:
        for signature in signatures:
            cursor.execute(
                "SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE')",
                (signature,),
            )
            assert cursor.fetchone() == (True,)
            for role in ("anon", "authenticated", "service_role"):
                cursor.execute(
                    "SELECT has_function_privilege(%s,%s,'EXECUTE')",
                    (role, signature),
                )
                assert cursor.fetchone() == (False,)
        cursor.execute(
            "SELECT DISTINCT owner.rolname FROM pg_proc procedure "
            "JOIN pg_namespace namespace ON namespace.oid=procedure.pronamespace "
            "JOIN pg_roles owner ON owner.oid=procedure.proowner "
            "WHERE namespace.nspname='public' AND procedure.proname IN "
            "('lab_arena_list_deepline_cost_reconciliations_v1',"
            "'lab_arena_reconcile_deepline_cost_v1')"
        )
        assert cursor.fetchall() == [("lab_arena_owner",)]

        cursor.execute(
            "SELECT pg_get_functiondef("
            "'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure)"
        )
        definition = cursor.fetchone()[0]
        assert definition.count(
            "lab_arena_deepline_reconciliation_retry_deferral"
        ) == 1
        assert definition.count(
            "lab_arena_deepline_credential_deferral_bypass"
        ) == 1
    with connect() as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                DEFERRAL_BYPASS_MIGRATION.read_text(encoding="utf-8")
            )
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure)"
            )
            assert cursor.fetchone()[0] == definition


def test_cancelled_round_exact_cost_settles_append_only_and_replays(resources):
    store, _connect = resources
    round_id, _submission, run, token = _open_run(store, "cancelled")
    identity, request_id, operation, fingerprint = _uncertain_call(
        store, run, token, label="cancelled"
    )
    candidate = store.list_deepline_cost_reconciliations(round_id)[0]
    assert candidate["request_id"] == request_id
    assert datetime.fromisoformat(candidate["reservation_at"].replace("Z", "+00:00"))
    assert datetime.fromisoformat(candidate["uncertain_at"].replace("Z", "+00:00"))
    assert candidate["operation"] == operation
    assert candidate["credential_fingerprint"] == fingerprint
    assert datetime.fromisoformat(candidate["uncertain_at"].replace("Z", "+00:00"))
    assert store.cancel_round(round_id, "test_cancel")["status"] == "cancelled"

    arguments = dict(
        round_id=round_id,
        run_id=run["run_id"],
        call_identity=identity,
        uncertain_entry_id=candidate["uncertain_entry_id"],
        request_id=request_id,
        operation=operation,
        credential_fingerprint=fingerprint,
        actual_microusd=3_000,
        cost_units="0.03",
    )
    first = store.reconcile_deepline_cost(**arguments)
    second = store.reconcile_deepline_cost(**arguments)

    assert first == {
        "status": "settled",
        "idempotent": False,
        "actual_microusd": 3_000,
        "released_microusd": -1_000,
        "variance_microusd": 1_000,
    }
    assert second == dict(first, idempotent=True)
    rows = store.list_ledger(call_identity=identity)
    assert [row["entry_kind"] for row in rows] == [
        "reservation", "dispatch", "uncertain", "settlement"
    ]
    assert rows[-1]["terminal_response"]["provider_cost"] == {
        "basis": "deepline_billing_ledger_charge_credits_x_0.10_usd",
        "units": "0.03",
        "unit_name": "credits",
        "operation": operation,
        "request_id": request_id,
    }
    assert base64.b64decode(rows[-1]["terminal_response"]["body_b64"]) == (
        b'{"error":{"code":"provider_unavailable"}}'
    )
    assert rows[-1]["terminal_response"]["headers"]["content-length"] == "41"
    conflict = dict(arguments, cost_units="0.04", actual_microusd=4_000)
    assert store.reconcile_deepline_cost(**conflict)["status"] == "conflict"
    assert len(_settlements(store, identity)) == 1


def test_missing_cost_settlement_preserves_successful_provider_call(resources):
    store, _connect = resources
    round_id, _submission, run, token = _open_run(store, "successful")
    identity, request_id, operation, fingerprint = _uncertain_call(
        store,
        run,
        token,
        label="successful",
        reason="missing_provider_cost",
        call_succeeded=True,
    )
    candidate = store.list_deepline_cost_reconciliations(round_id)[0]

    assert store.reconcile_deepline_cost(
        round_id=round_id,
        run_id=run["run_id"],
        call_identity=identity,
        uncertain_entry_id=candidate["uncertain_entry_id"],
        request_id=request_id,
        operation=operation,
        credential_fingerprint=fingerprint,
        actual_microusd=2_000,
        cost_units="0.02",
    )["status"] == "settled"
    settlement = _settlements(store, identity)[0]
    assert settlement["terminal_response"]["call_succeeded"] is True


@pytest.mark.parametrize("malformed", [None, "true"], ids=["absent", "string"])
def test_missing_or_malformed_success_flag_cannot_reconcile(resources, malformed):
    store, connect = resources
    label = "flagabs" if malformed is None else "flagstr"
    round_id, _submission, run, token = _open_run(store, label)
    identity, request_id, operation, fingerprint = _uncertain_call(
        store,
        run,
        token,
        label=label,
        reason="missing_provider_cost",
        call_succeeded=True,
    )
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        if malformed is None:
            cursor.execute(
                "UPDATE public.lab_arena_ledger SET entry_doc="
                "entry_doc #- '{call,call_succeeded}' "
                "WHERE call_identity=%s AND entry_kind='uncertain'",
                (identity,),
            )
        else:
            cursor.execute(
                "UPDATE public.lab_arena_ledger SET entry_doc="
                "jsonb_set(entry_doc,'{call,call_succeeded}',to_jsonb(%s::text)) "
                "WHERE call_identity=%s AND entry_kind='uncertain'",
                (malformed, identity),
            )
    uncertainty = store.list_ledger(call_identity=identity)[-1]
    assert uncertainty["entry_kind"] == "uncertain"
    assert store.list_deepline_cost_reconciliations(round_id) == []
    assert store.reconcile_deepline_cost(
        round_id=round_id,
        run_id=run["run_id"],
        call_identity=identity,
        uncertain_entry_id=uncertainty["entry_id"],
        request_id=request_id,
        operation=operation,
        credential_fingerprint=fingerprint,
        actual_microusd=2_000,
        cost_units="0.02",
    )["status"] == "stale"
    assert _settlements(store, identity) == []


def test_transport_failure_cannot_claim_the_provider_call_succeeded(resources):
    store, _connect = resources
    round_id, _submission, run, token = _open_run(store, "badtransport")
    identity, request_id, operation, fingerprint = _uncertain_call(
        store,
        run,
        token,
        label="badtransport",
        reason="transport_failure",
        call_succeeded=True,
    )
    uncertainty = store.list_ledger(call_identity=identity)[-1]

    assert store.list_deepline_cost_reconciliations(round_id) == []
    assert store.reconcile_deepline_cost(
        round_id=round_id,
        run_id=run["run_id"],
        call_identity=identity,
        uncertain_entry_id=uncertainty["entry_id"],
        request_id=request_id,
        operation=operation,
        credential_fingerprint=fingerprint,
        actual_microusd=2_000,
        cost_units="0.02",
    )["status"] == "stale"
    assert _settlements(store, identity) == []


@pytest.mark.parametrize("field", ["request_id", "operation", "credential_fingerprint"])
def test_reconciliation_rejects_unbound_identity_fields(resources, field):
    store, _connect = resources
    label = {
        "request_id": "boundr",
        "operation": "boundo",
        "credential_fingerprint": "boundf",
    }[field]
    round_id, _submission, run, token = _open_run(store, label)
    identity, request_id, operation, fingerprint = _uncertain_call(
        store, run, token, label=label
    )
    candidate = store.list_deepline_cost_reconciliations(round_id)[0]
    arguments = dict(
        round_id=round_id,
        run_id=run["run_id"],
        call_identity=identity,
        uncertain_entry_id=candidate["uncertain_entry_id"],
        request_id=request_id,
        operation=operation,
        credential_fingerprint=fingerprint,
        actual_microusd=2_000,
        cost_units="0.02",
    )
    arguments[field] = {
        "request_id": "ctx-tool-" + "b" * 32,
        "operation": "exa_search",
        "credential_fingerprint": "sha256:" + "c" * 64,
    }[field]
    if field == "request_id":
        with pytest.raises(
            ArenaStoreError,
            match="lab_arena_deepline_reconciliation_input_invalid",
        ):
            store.reconcile_deepline_cost(**arguments)
        return
    assert store.reconcile_deepline_cost(**arguments)["status"] == "stale"
    assert _settlements(store, identity) == []


def test_candidate_list_rejects_valid_but_nondeterministic_request_id(resources):
    store, _connect = resources
    round_id, _submission, run, token = _open_run(store, "nondeterministic")
    identity, _request_id, _operation, _fingerprint = _uncertain_call(
        store,
        run,
        token,
        label="nondeterministic",
        request_id="ctx-tool-" + "c" * 32,
    )

    assert "ctx-tool-" + identity.removeprefix("sha256:")[:32] != (
        "ctx-tool-" + "c" * 32
    )
    assert store.list_deepline_cost_reconciliations(round_id) == []
    assert _settlements(store, identity) == []


def test_conversion_is_exact_and_concurrent_settlement_has_one_winner(
    resources,
):
    store, _connect = resources
    round_id, _submission, run, token = _open_run(store, "race")
    identity, request_id, operation, fingerprint = _uncertain_call(
        store, run, token, label="race"
    )
    candidate = store.list_deepline_cost_reconciliations(round_id)[0]
    arguments = dict(
        round_id=round_id,
        run_id=run["run_id"],
        call_identity=identity,
        uncertain_entry_id=candidate["uncertain_entry_id"],
        request_id=request_id,
        operation=operation,
        credential_fingerprint=fingerprint,
        actual_microusd=1,
        cost_units="0.000001",
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _unused: store.reconcile_deepline_cost(**arguments), range(2)))
    assert sorted(result["idempotent"] for result in results) == [False, True]
    assert len(_settlements(store, identity)) == 1

    other_round, _submission, other_run, other_token = _open_run(store, "rounding")
    other_identity, other_request, other_operation, other_fp = _uncertain_call(
        store, other_run, other_token, label="rounding"
    )
    other = store.list_deepline_cost_reconciliations(other_round)[0]
    with pytest.raises(Exception, match="deepline_reconciliation_input_invalid"):
        store.reconcile_deepline_cost(
            round_id=other_round,
            run_id=other_run["run_id"],
            call_identity=other_identity,
            uncertain_entry_id=other["uncertain_entry_id"],
            request_id=other_request,
            operation=other_operation,
            credential_fingerprint=other_fp,
            actual_microusd=1,
            cost_units="0.000011",
        )
    assert _settlements(store, other_identity) == []


def test_fresh_uncertainty_defers_same_submission_then_settlement_unblocks(database):
    store = _store(database)
    psycopg2, dsn = database
    round_id = "arena-2026-09-14-deferral"
    try:
        runners, _participants = _open_scoring(
            store, round_id, participants=2, runners=3
        )
        first, token, _, _ = claim(
            store, round_id, runners[0], parallelism=8, ceiling=8,
            excluded=[runners[0]],
        )
        identity, request_id, operation, fingerprint = _uncertain_call(
            store, first, token, label="deferral"
        )
        assert complete(
            store, first["run_id"], hash_lease_token(token), "judge_error"
        )["status"] == "failed"
        retry_id = first["assignment_id"] + ":2"

        # Retain the retry, one new ICP at attempt 1 for the same submission,
        # and pending work from the other submission. The fresh uncertainty
        # must hold both same-submission rows without blocking unrelated work.
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='score' AND status='pending' AND submission_id=%s "
                "AND attempt=1 ORDER BY run_id LIMIT 1",
                (round_id, first["submission_id"]),
            )
            same_submission_new_id = cursor.fetchone()[0]
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',"
                "terminal_cause='stage_closed' WHERE round_id=%s AND kind='score' "
                "AND status='pending' AND run_id<>ALL(%s) AND submission_id=%s",
                (
                    round_id,
                    [retry_id, same_submission_new_id],
                    first["submission_id"],
                ),
            )
        unrelated, unrelated_token, _, _ = claim(
            store, round_id, runners[1], parallelism=8, ceiling=8,
            excluded=[runners[1]],
        )
        assert unrelated["status"] == "leased"
        assert unrelated["submission_id"] != first["submission_id"]
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT run_id,status,attempt FROM public.lab_arena_runs "
                "WHERE run_id=ANY(%s) ORDER BY run_id",
                ([retry_id, same_submission_new_id],),
            )
            held = cursor.fetchall()
        assert {row[0] for row in held} == {retry_id, same_submission_new_id}
        assert all(row[1] == "pending" for row in held)
        assert {row[2] for row in held} == {1, 2}
        assert complete(
            store, unrelated["run_id"], hash_lease_token(unrelated_token),
            "accepted", output_ref="arena/test/unrelated.json",
        )["status"] == "accepted"

        candidate = store.list_deepline_cost_reconciliations(round_id)[0]
        assert store.reconcile_deepline_cost(
            round_id=round_id,
            run_id=first["run_id"],
            call_identity=identity,
            uncertain_entry_id=candidate["uncertain_entry_id"],
            request_id=request_id,
            operation=operation,
            credential_fingerprint=fingerprint,
            actual_microusd=2_000,
            cost_units="0.02",
        )["status"] == "settled"
        released, _retry_token, _, _ = claim(
            store, round_id, runners[2], parallelism=8, ceiling=8,
            excluded=[runners[2]],
        )
        assert released["status"] == "leased"
        assert released["run_id"] in {retry_id, same_submission_new_id}
    finally:
        store.close()


@pytest.mark.parametrize("provider_status", [401, 402, 403])
def test_bound_credential_failure_bypasses_only_claim_deferral(
    database, provider_status
):
    store = _store(database)
    round_id = "arena-2026-09-14-auth%s" % provider_status
    try:
        runners, _participants = _open_scoring(
            store, round_id, participants=1, runners=2
        )
        first, token, _, _ = claim(
            store, round_id, runners[0], parallelism=8, ceiling=8,
            excluded=[runners[0]],
        )
        identity, _request_id, _operation, _fingerprint = _uncertain_call(
            store,
            first,
            token,
            label="auth%s" % provider_status,
            reason="missing_provider_cost",
            credential_failure_status=provider_status,
        )
        assert complete(
            store, first["run_id"], hash_lease_token(token), "credential_error"
        )["status"] == "failed"

        released, _released_token, _, _ = claim(
            store, round_id, runners[1], parallelism=8, ceiling=8,
            excluded=[runners[1]],
        )
        assert released["status"] == "leased"
        assert released["submission_id"] == first["submission_id"]
        ledger = store.list_ledger(call_identity=identity)
        assert ledger[-1]["entry_kind"] == "uncertain"
        assert ledger[-1]["amount_microusd"] == 2_000
        assert _settlements(store, identity) == []
    finally:
        store.close()


@pytest.mark.parametrize(
    "label,status,overrides",
    [
        ("missing", None, None),
        ("rate", 429, None),
        (
            "identity",
            401,
            {"base_call_identity": "sha256:" + "f" * 64},
        ),
        ("attempt", 401, {"provider_attempt": 2}),
        ("sequence", 401, {"action_sequence": 1}),
    ],
)
def test_untrusted_credential_failure_evidence_cannot_release_deferral(
    database, label, status, overrides
):
    store = _store(database)
    round_id = "arena-2026-09-14-forged%s" % label
    try:
        runners, _participants = _open_scoring(
            store, round_id, participants=1, runners=2
        )
        first, token, _, _ = claim(
            store, round_id, runners[0], parallelism=8, ceiling=8,
            excluded=[runners[0]],
        )
        identity, _request_id, _operation, _fingerprint = _uncertain_call(
            store,
            first,
            token,
            label="forged%s" % label,
            reason="missing_provider_cost",
            credential_failure_status=status,
            evidence_overrides=overrides,
        )
        assert complete(
            store, first["run_id"], hash_lease_token(token), "credential_error"
        )["status"] == "failed"

        blocked, _blocked_token, _, _ = claim(
            store, round_id, runners[1], parallelism=8, ceiling=8,
            excluded=[runners[1]],
        )
        assert blocked["status"] == "no_pending"
        assert store.list_ledger(call_identity=identity)[-1]["entry_kind"] == (
            "uncertain"
        )
        assert _settlements(store, identity) == []
    finally:
        store.close()


def test_retry_deferral_does_not_expire_and_settlement_releases(
    database,
):
    store = _store(database)
    psycopg2, dsn = database
    round_id = "arena-2026-09-14-expiry"
    try:
        runners, _participants = _open_scoring(
            store, round_id, participants=1, runners=2
        )
        first, token, _, _ = claim(
            store, round_id, runners[0], parallelism=8, ceiling=8,
            excluded=[runners[0]],
        )
        identity, request_id, operation, fingerprint = _uncertain_call(
            store, first, token, label="expiry"
        )
        assert complete(
            store, first["run_id"], hash_lease_token(token), "judge_error"
        )["status"] == "failed"
        retry_id = first["assignment_id"] + ":2"
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',"
                "terminal_cause='stage_closed' WHERE round_id=%s AND kind='score' "
                "AND status='pending' AND run_id<>%s",
                (round_id, retry_id),
            )
        blocked, _, _, _ = claim(
            store, round_id, runners[1], parallelism=8, ceiling=8,
            excluded=[runners[1]],
        )
        assert blocked["status"] == "no_pending"
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_ledger SET created_at="
                "clock_timestamp()-interval '121 seconds' "
                "WHERE call_identity=%s AND entry_kind='uncertain'",
                (identity,),
            )
        still_blocked, _, _, _ = claim(
            store, round_id, runners[1], parallelism=8, ceiling=8,
            excluded=[runners[1]],
        )
        assert still_blocked["status"] == "no_pending"
        assert _settlements(store, identity) == []
        assert store.list_ledger(call_identity=identity)[-1]["amount_microusd"] == 2_000
        candidate = store.list_deepline_cost_reconciliations(round_id)[0]
        assert store.reconcile_deepline_cost(
            round_id=round_id,
            run_id=first["run_id"],
            call_identity=identity,
            uncertain_entry_id=candidate["uncertain_entry_id"],
            request_id=request_id,
            operation=operation,
            credential_fingerprint=fingerprint,
            actual_microusd=2_000,
            cost_units="0.02",
        )["status"] == "settled"
        retry, _retry_token, _, _ = claim(
            store, round_id, runners[1], parallelism=8, ceiling=8,
            excluded=[runners[1]],
        )
        assert retry["status"] == "leased"
        assert retry["run_id"] == retry_id
    finally:
        store.close()
