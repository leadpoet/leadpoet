"""Old and new settlement-failure heads use only exact Deepline billing recovery."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena import broker as broker_module
from lab_arena.store import hash_lease_token
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import _store
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import (
    ReconciliationStore, _broker, _candidate,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim, commit_round, frozen_participants, hotkey, round_config, sha,
    stage_positions,
)


MIGRATION_NAME = "261-lab-arena-settlement-failure-deepline-cost-reconciliation.sql"
MIGRATION = Path(__file__).resolve().parents[2] / "scripts" / MIGRATION_NAME
RESERVE_MICROUSD = 74_796_189


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + (MIGRATION_NAME,)
    )


@pytest.fixture(scope="module")
def database_before():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _uncertain(store, label, call_doc, *, dispatch=True):
    round_id = "arena-2026-09-14-sf" + label
    prefix = "sf" + label
    runner_keys = [hotkey(prefix + "-runner-0")]
    config = round_config(
        round_id, runner_keys, execution_cap_microusd=100_000_000,
    )
    # Migration258 defers open-round code review until the final hour of
    # intake. Keep this disposable fixture within that existing window.
    config["schedule"]["submission_cutoff"] = (
        datetime.now(timezone.utc) + timedelta(minutes=30)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert store.create_round(round_id, config)["status"] == "created"
    parts = frozen_participants(store, round_id, 1, prefix=prefix)
    commit_round(store, round_id, parts)
    assert store.open_stage(round_id, 1, parts, stage_positions(1))[
        "status"
    ] == "ok"
    run, token, _, _ = claim(store, round_id, runner_keys[0])
    identity = contracts.provider_call_identity(
        attempt=run["attempt"], assignment_id=run["assignment_id"],
        icp_position=run["icp_position"], action_sequence=0,
        operation_id="deepline.execute", request_hash=sha(label),
    )
    request_id = "ctx-tool-" + identity.removeprefix("sha256:")[:32]
    fingerprint = "sha256:" + "a" * 64
    token_hash = hash_lease_token(token)
    assert store.reserve_call(
        run_id=run["run_id"], lease_token_hash=token_hash,
        call_identity=identity, operation_id="deepline.execute",
        provider="deepline", funding_source="miner_key",
        amount_microusd=RESERVE_MICROUSD,
        call_doc={
            "request_hash": sha(label), "tool": "harvestapi_get_company",
            "deepline_request_id": request_id,
            "credential_fingerprint": fingerprint,
        },
    )["status"] == "reserved"
    if dispatch:
        assert store.mark_dispatched(
            run_id=run["run_id"], lease_token_hash=token_hash,
            call_identity=identity,
        )["status"] == "dispatched"
    assert store.mark_uncertain(
        run_id=run["run_id"], lease_token_hash=token_hash,
        call_identity=identity, call_doc=call_doc,
    )["status"] == "uncertain"
    return round_id, run, identity, request_id, fingerprint


def _old_call(**changes):
    return {
        "reason": "settle_failure", "failure_stage": "settlement",
        "error_class": "ArenaStoreError", "call_succeeded": False,
        **changes,
    }


def test_migration_replays_without_replacing_normal_rpc_or_permissions(database_before):
    psycopg2, dsn = database_before
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            signatures = (
                "public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)",
                "public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)",
            )
            before = []
            for signature in signatures:
                cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
                before.append(cursor.fetchone()[0])
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            for signature, expected in zip(signatures, before):
                cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
                assert cursor.fetchone()[0] == expected
                for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
                    cursor.execute(
                        "SELECT has_function_privilege(%s,%s,'EXECUTE')",
                        (role, signature),
                    )
                    assert cursor.fetchone() == (role == "lab_arena_service",)
            cursor.execute("SELECT has_function_privilege('lab_arena_service',"
                           "'public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)',"
                           "'EXECUTE')")
            assert cursor.fetchone() == (False,)
    finally:
        connection.close()


def test_exact_shadow14_shape_is_only_a_candidate_and_preserves_reserve(database):
    store = _store(database)
    try:
        round_id, run, identity, request_id, fingerprint = _uncertain(
            store, "old", _old_call()
        )
        entries = store.list_ledger(call_identity=identity)
        assert [entry["entry_kind"] for entry in entries] == [
            "reservation", "dispatch", "uncertain"
        ]
        assert entries[-1]["entry_doc"] == {
            "reason": "worker_reported", "call": _old_call()
        }
        assert entries[-1]["amount_microusd"] == RESERVE_MICROUSD
        assert "deepline_request_id" not in entries[-1]["entry_doc"]["call"]
        candidates = store.list_deepline_cost_reconciliations(round_id)
        assert len(candidates) == 1
        assert candidates[0]["request_id"] == request_id
        assert candidates[0]["operation"] == "harvestapi_get_company"
        assert candidates[0]["credential_fingerprint"] == fingerprint
        assert candidates[0]["run_id"] == run["run_id"]
        assert store.list_ledger(call_identity=identity) == entries
        costs = store.submission_costs(run["submission_id"])
        assert sum(
            row["reserved_or_uncertain_microusd"] for row in costs["providers"]
        ) == RESERVE_MICROUSD
        # Listing cannot settle a zero. The normal broker must first observe
        # one exact terminal provider billing row before invoking the RPC.
    finally:
        store.close()


@pytest.mark.parametrize("label,call", [
    ("stage", _old_call(failure_stage="terminal_response")),
    ("class", _old_call(error_class="ValueError")),
    ("reason", _old_call(reason="provider_failure")),
    ("bool", {"reason": "settle_failure", "failure_stage": "settlement",
              "error_class": "ArenaStoreError"}),
    ("partial", _old_call(deepline_request_id="ctx-tool-" + "f" * 32)),
    ("nullid", _old_call(deepline_request_id=None)),
    ("nulljob", _old_call(deepline_job_id=None)),
    ("native", _old_call(deepline_job_id="iad1::job-1789361973860-8634524a6f3a")),
])
def test_untrusted_old_head_never_becomes_candidate(database, label, call):
    store = _store(database)
    try:
        round_id, _, identity, _, _ = _uncertain(store, label, call)
        assert store.list_deepline_cost_reconciliations(round_id) == []
        assert store.list_ledger(call_identity=identity)[-1]["entry_kind"] == "uncertain"
    finally:
        store.close()


def test_full_binding_accepts_known_success_but_rejects_credential_drift(database):
    store = _store(database)
    psycopg2, dsn = database
    try:
        # Construct the matching future metadata after deriving the call ID.
        round_id, _, identity, request_id, fingerprint = _uncertain(
            store, "full", _old_call()
        )
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set("
                "entry_doc,'{call}',%s::jsonb) WHERE call_identity=%s "
                "AND entry_kind='uncertain'",
                (json.dumps(_old_call(
                    call_succeeded=True, deepline_request_id=request_id,
                    deepline_operation="harvestapi_get_company",
                    credential_fingerprint=fingerprint,
                )), identity),
            )
        assert len(store.list_deepline_cost_reconciliations(round_id)) == 1
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set("
                "entry_doc,'{credential_fingerprint}',to_jsonb(%s::text)) "
                "WHERE call_identity=%s AND entry_kind='reservation'",
                ("sha256:" + "b" * 64, identity),
            )
        assert store.list_deepline_cost_reconciliations(round_id) == []
    finally:
        store.close()


@pytest.mark.parametrize("field", [
    "deepline_request_id", "deepline_operation", "credential_fingerprint",
])
def test_partial_or_unbound_future_metadata_is_rejected(database, field):
    store = _store(database)
    psycopg2, dsn = database
    try:
        round_id, _, identity, request_id, fingerprint = _uncertain(
            store, {"deepline_request_id": "id", "deepline_operation": "op",
                    "credential_fingerprint": "fp"}[field], _old_call()
        )
        full = _old_call(
            call_succeeded=True, deepline_request_id=request_id,
            deepline_operation="harvestapi_get_company",
            credential_fingerprint=fingerprint,
        )
        full[field] = {
            "deepline_request_id": "ctx-tool-" + "f" * 32,
            "deepline_operation": "other_tool",
            "credential_fingerprint": "sha256:" + "b" * 64,
        }[field]
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set("
                "entry_doc,'{call}',%s::jsonb) WHERE call_identity=%s "
                "AND entry_kind='uncertain'", (json.dumps(full), identity),
            )
        assert store.list_deepline_cost_reconciliations(round_id) == []
    finally:
        store.close()


@pytest.mark.parametrize("reason,succeeded", [
    ("transport_failure", False), ("missing_provider_cost", True),
])
def test_existing_worker_uncertainty_reasons_keep_their_binding(
    database, reason, succeeded
):
    store = _store(database)
    psycopg2, dsn = database
    try:
        label = "tr" if reason == "transport_failure" else "mc"
        round_id, _, identity, request_id, fingerprint = _uncertain(
            store, label, _old_call()
        )
        call = {
            "reason": reason, "call_succeeded": succeeded,
            "deepline_request_id": request_id,
            "deepline_operation": "harvestapi_get_company",
            "credential_fingerprint": fingerprint,
        }
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set("
                "entry_doc,'{call}',%s::jsonb) WHERE call_identity=%s "
                "AND entry_kind='uncertain'", (json.dumps(call), identity),
            )
        candidates = store.list_deepline_cost_reconciliations(round_id)
        assert len(candidates) == 1
        assert candidates[0]["request_id"] == request_id
    finally:
        store.close()


@pytest.mark.parametrize("case", ["missing_dispatch", "round_open"])
def test_existing_dispatch_and_round_guards_remain_fail_closed(database, case):
    store = _store(database)
    psycopg2, dsn = database
    try:
        round_id, _, identity, _, _ = _uncertain(
            store, {"missing_dispatch": "md", "round_open": "ro"}[case],
            _old_call(),
        )
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            if case == "missing_dispatch":
                cursor.execute(
                    "DELETE FROM public.lab_arena_ledger WHERE call_identity=%s "
                    "AND entry_kind='dispatch'", (identity,),
                )
            else:
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET status='open' "
                    "WHERE round_id=%s", (round_id,),
                )
        assert store.list_deepline_cost_reconciliations(round_id) == []
    finally:
        store.close()


def test_no_exact_terminal_billing_row_cannot_become_a_zero_settlement():
    class PendingLedgerTransport:
        def __init__(self):
            self.methods = []

        def send(self, **kwargs):
            self.methods.append(kwargs["method"])
            return broker_module.ProviderResponse(
                200, {"content-type": "application/json"},
                b'{"entries":[],"has_more":false}',
            )

    store = ReconciliationStore()
    transport = PendingLedgerTransport()
    result = _broker(store, transport).reconcile_deepline_cost(
        _candidate(), timeout_seconds=0.1,
    )
    assert result == {"status": "pending"}
    assert store.calls == []
    assert transport.methods and set(transport.methods) == {"GET"}


def test_later_head_is_not_relisted_and_cross_run_settlement_is_stale(database):
    store = _store(database)
    try:
        round_id, run, identity, _, _ = _uncertain(store, "later", _old_call())
        candidate = store.list_deepline_cost_reconciliations(round_id)[0]
        wrong = store.reconcile_deepline_cost(
            round_id=round_id, run_id="other-run", call_identity=identity,
            uncertain_entry_id=candidate["uncertain_entry_id"],
            request_id=candidate["request_id"],
            operation=candidate["operation"],
            credential_fingerprint=candidate["credential_fingerprint"],
            actual_microusd=2_000, cost_units="0.02",
        )
        assert wrong["status"] == "stale"
        assert store.list_ledger(call_identity=identity)[-1]["entry_kind"] == "uncertain"
        settled = store.reconcile_deepline_cost(
            round_id=round_id, run_id=run["run_id"],
            call_identity=identity,
            uncertain_entry_id=candidate["uncertain_entry_id"],
            request_id=candidate["request_id"],
            operation=candidate["operation"],
            credential_fingerprint=candidate["credential_fingerprint"],
            actual_microusd=2_000, cost_units="0.02",
        )
        assert settled["status"] == "settled"
        assert store.list_deepline_cost_reconciliations(round_id) == []
        assert store.list_ledger(call_identity=identity)[-1]["entry_doc"][
            "reconciled_uncertainty_reason"
        ] == "worker_reported"
    finally:
        store.close()
