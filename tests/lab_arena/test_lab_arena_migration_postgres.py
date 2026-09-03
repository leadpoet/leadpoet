"""Disposable-PostgreSQL behavior of scripts/178-lab-arena-v1.sql through
``lab_arena.store`` (labarena.md sections 18.1, 18.2, 18.3).

Every write goes through the SECURITY DEFINER functions as ``lab_arena_service``
via ``PsycopgTransport``; superuser access is used only to simulate time
(moving ``lease_expires_at`` into the past) and to prove trigger guards.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import pytest

from bittensor_wallet import Keypair

from lab_arena import contracts
from lab_arena.store import (
    ArenaRoleError,
    ArenaStore,
    ArenaStoreError,
    PsycopgTransport,
    hash_lease_token,
    new_lease_token,
)
from tests.lab_arena.lab_arena_pg_harness import LAB_ARENA_MIGRATION, database_with_lab_arena_migration
from tests.test_source_add_end_to_end_postgres import SCRIPTS

_KEYS: Dict[str, str] = {}


def hotkey(label: str) -> str:
    if label not in _KEYS:
        _KEYS[label] = Keypair.create_from_uri("//" + label).ss58_address
    return _KEYS[label]


def sha(seed: str) -> str:
    return contracts.document_hash({"seed": seed})


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration((LAB_ARENA_MIGRATION,))


@pytest.fixture(scope="module")
def connect(database):
    psycopg2, dsn = database

    def _connect():
        return psycopg2.connect(**dsn)

    return _connect


@pytest.fixture()
def store(connect):
    transport = PsycopgTransport(connect)
    yield ArenaStore(transport, lease_ttl_seconds=120)
    transport.close()


@pytest.fixture()
def superuser(connect):
    connection = connect()
    connection.autocommit = True
    yield connection
    connection.close()


def round_config(round_id: str, runners: List[str], *, quotas=None, stage_1_icps=20, stage_2_icps=30, max_attempts=2) -> Dict[str, Any]:
    """The configuration fields the SQL reads: runner allowlist and the call-quota inputs."""

    body = {
        "round_id": round_id,
        "runner_allowlist": runners,
        "call_quotas": dict(quotas or contracts.CALL_QUOTAS_PER_ICP),
        "stage_1_icp_count": stage_1_icps,
        "stage_2_icp_count": stage_2_icps,
        "max_attempts_per_assignment": max_attempts,
    }
    return contracts.hashed_document(body, "configuration_hash")


def register_keys(store: ArenaStore, miner: str, label: str, *, preflight: str = "ok", providers=contracts.MINER_KEY_PROVIDERS) -> None:
    """Miners bring one key per provider; the account aggregate is ok only when every provider passed."""

    for provider in providers:
        key_hash = sha("key" + label + provider)[7:]
        record = {"preflight_status": preflight, "key_hash": key_hash, "provider": provider, "limit_microusd": 20_000_000 if provider == "openrouter" else None,
                  "limit_remaining_microusd": 10_000_000 if provider == "openrouter" else None, "usage_microusd": 0 if provider == "openrouter" else None, "observed_at": "2026-09-02T00:00:00Z", "probe": {}}
        store.upsert_account_credential(miner, provider, "ciphertext-%s-%s" % (label, provider), key_hash, record)


def frozen_participants(store: ArenaStore, round_id: str, count: int, *, prefix: str, king_index=None) -> List[Dict[str, Any]]:
    participants = []
    for index in range(count):
        miner = hotkey("%s-miner-%d" % (prefix, index))
        submission_id = "%s-sub-%d" % (prefix, index)
        result = store.register_submission(round_id, submission_id, miner, {"package_hash": sha(submission_id + "pkg"), "consent": {"source_publication": True, "public_rerun": True}})
        assert result["status"] == "registered"
        result = store.update_submission(round_id, submission_id, "uploaded", "accepted", {"source_tree_hash": sha(submission_id + "tree"), "image_digest": "sha256:" + sha(submission_id + "img")[7:]})
        assert result["status"] == "ok", result
        result = store.update_submission(round_id, submission_id, "accepted", "frozen", {"is_king": king_index == index})
        assert result["status"] == "ok"
        register_keys(store, miner, submission_id)
        participants.append({"submission_id": submission_id, "miner_hotkey": miner, "preflight_failed": False})
    return participants


def commit_round(store: ArenaStore, round_id: str, participants) -> None:
    commitment = contracts.hashed_document({"round_id": round_id, "configuration_hash": store.get_round(round_id)["configuration_hash"], "roots": sha(round_id + "root")}, "commitment_hash")
    result = store.transition_round(round_id, "open", "committed", {
        "commitment_hash": commitment["commitment_hash"],
        "commitment_doc": commitment,
        "participant_set_hash": sha(round_id + "participants"),
        "participants": participants,
        "benchmark_ref": "arena/%s/benchmark.json" % round_id,
        "evaluation_date": "2026-09-02",
    })
    assert result["status"] == "ok", result


def stage_positions(stage: int):
    return list(range(0, 20)) if stage == 1 else list(range(20, 50))


def open_round(store: ArenaStore, round_id: str, *, participants=3, runners=1, prefix: str, quotas=None, stage_1_icps=20, max_attempts=2, king_index=None):
    runner_keys = [hotkey("%s-runner-%d" % (prefix, i)) for i in range(runners)]
    config = round_config(round_id, runner_keys, quotas=quotas, stage_1_icps=stage_1_icps, max_attempts=max_attempts)
    assert store.create_round(round_id, config)["status"] == "created"
    parts = frozen_participants(store, round_id, participants, prefix=prefix, king_index=king_index)
    commit_round(store, round_id, parts)
    positions = stage_positions(1)
    result = store.open_stage(round_id, 1, parts, positions, [sha("%s-icp-%d" % (round_id, p)) for p in positions])
    assert result["status"] == "ok", result
    return runner_keys, parts


def claim(store: ArenaStore, round_id: str, runner: str, *, parallelism=1, ceiling=8, excluded=(), request_id=None, token=None):
    token = token or new_lease_token()
    request_id = request_id or contracts.new_request_id()
    request_hash = sha(request_id + "bytes")
    response = store.claim_assignment(
        round_id=round_id, runner_hotkey=runner, declared_parallelism=parallelism, slot_ceiling=ceiling,
        excluded_miner_hotkeys=list(excluded), request_id=request_id, request_hash=request_hash,
        lease_token_hash=hash_lease_token(token),
    )
    return response, token, request_id, request_hash


def make_event(run: Dict[str, Any], cursor: int, head: str, event_type="stdout", payload=None):
    body = {"event_type": event_type, "sequence": cursor, "prev_hash": head, "timestamp": "2026-09-02T01:00:00Z", "payload": payload or {"line": "x"}}
    body["event_hash"] = contracts.chain_hash(head or None, {k: v for k, v in body.items() if k != "prev_hash"})
    return body


def expire_now(superuser, run_id: str) -> None:
    with superuser.cursor() as cursor:
        cursor.execute("UPDATE public.lab_arena_runs SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE run_id = %s", (run_id,))


# ---------------------------------------------------------------------------
# 18.1 schema and role
# ---------------------------------------------------------------------------


def test_migration_applies_twice_and_roles_have_exact_attributes(superuser):
    with superuser.cursor() as cursor:
        cursor.execute((SCRIPTS / LAB_ARENA_MIGRATION).read_text(encoding="utf-8"))
        cursor.execute("SELECT rolname, rolsuper, rolbypassrls, rolcanlogin, rolinherit, rolcreaterole, rolcreatedb, rolreplication FROM pg_roles WHERE rolname IN ('lab_arena_owner', 'lab_arena_service') ORDER BY rolname")
        rows = cursor.fetchall()
        assert rows == [
            ("lab_arena_owner", False, False, False, False, False, False, False),
            ("lab_arena_service", False, False, False, False, False, False, False),
        ]
        cursor.execute("SELECT granted.rolname FROM pg_auth_members m JOIN pg_roles granted ON granted.oid = m.roleid JOIN pg_roles r ON r.oid = m.member WHERE r.rolname = 'lab_arena_service'")
        assert cursor.fetchall() == []  # authenticator absent in the harness: no memberships at all
        cursor.execute("SELECT count(*) FROM pg_default_acl d JOIN pg_roles r ON r.oid = d.defaclrole WHERE r.rolname IN ('lab_arena_owner', 'lab_arena_service')")
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT relname, relrowsecurity FROM pg_class WHERE relname LIKE 'lab_arena_%' AND relkind = 'r' ORDER BY relname")
        assert all(row[1] for row in cursor.fetchall())
        cursor.execute("SELECT tablename, policyname FROM pg_policies WHERE tablename LIKE 'lab_arena_%' ORDER BY 1")
        policies = cursor.fetchall()
        assert len(policies) == 6 and all(name.endswith("_service_read") for _, name in policies)


def test_service_role_function_grants_and_non_arena_denial(superuser, store):
    with superuser.cursor() as cursor:
        cursor.execute("CREATE TABLE IF NOT EXISTS public.lab_arena_test_other_table (id INT)")
        cursor.execute("CREATE OR REPLACE FUNCTION public.lab_arena_test_other_function() RETURNS INT LANGUAGE sql AS $$ SELECT 1 $$")
        cursor.execute("REVOKE ALL ON TABLE public.lab_arena_test_other_table FROM PUBLIC")
        cursor.execute("REVOKE ALL ON FUNCTION public.lab_arena_test_other_function() FROM PUBLIC")
        for role in ("lab_arena_service", "anon", "authenticated"):
            cursor.execute("SELECT has_table_privilege(%s, 'public.lab_arena_test_other_table', 'SELECT')", (role,))
            assert cursor.fetchone()[0] is False
            cursor.execute("SELECT has_function_privilege(%s, 'public.lab_arena_test_other_function()', 'EXECUTE')", (role,))
            assert cursor.fetchone()[0] is False
        for role in ("anon", "authenticated", "service_role"):
            for table in ("lab_arena_rounds", "lab_arena_submissions", "lab_arena_runs", "lab_arena_events", "lab_arena_accounts", "lab_arena_ledger"):
                cursor.execute("SELECT has_table_privilege(%s, %s, 'SELECT')", (role, "public." + table))
                assert cursor.fetchone()[0] is False, (role, table)
            cursor.execute("SELECT has_function_privilege(%s, 'public.lab_arena_create_round(text, text, jsonb)', 'EXECUTE')", (role,))
            assert cursor.fetchone()[0] is False
        cursor.execute("SELECT has_function_privilege('lab_arena_service', 'public.lab_arena_create_round(text, text, jsonb)', 'EXECUTE')")
        assert cursor.fetchone()[0] is True
        cursor.execute("SELECT has_function_privilege('lab_arena_service', 'public.lab_arena__terminate_open_calls(text, text)', 'EXECUTE')")
        assert cursor.fetchone()[0] is False
        for table in ("lab_arena_events", "lab_arena_ledger", "lab_arena_rounds"):
            for privilege in ("INSERT", "UPDATE", "DELETE", "TRUNCATE"):
                cursor.execute("SELECT has_table_privilege('lab_arena_service', %s, %s)", ("public." + table, privilege))
                assert cursor.fetchone()[0] is False, (table, privilege)
    identity = store.require_service_role()
    assert identity["current_user"] == "lab_arena_service" and identity["rolsuper"] is False


def test_service_refuses_to_start_as_superuser(connect):
    transport = PsycopgTransport(connect, role=None)
    try:
        with pytest.raises(ArenaRoleError):
            ArenaStore(transport).require_service_role()
    finally:
        transport.close()


def test_append_only_tables_and_write_once_rounds_resist_owner_level_mutation(store, superuser):
    round_id = "arena-2026-09-02-ao"
    runners, parts = open_round(store, round_id, participants=1, prefix="ao")
    response, token, _, _ = claim(store, round_id, runners[0])
    run_id = response["run_id"]
    lease_hash = hash_lease_token(token)
    identity = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=0, action_sequence=0, operation_id="deepline.execute", request_hash=sha("q"))
    assert store.reserve_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={"q": 1})["status"] == "reserved"
    event = make_event(response, 0, "")
    assert store.append_events(run_id=run_id, lease_token_hash=lease_hash, events=[event])["status"] == "appended"
    with superuser.cursor() as cursor:
        for statement in (
            "UPDATE public.lab_arena_events SET event_type = 'x'",
            "DELETE FROM public.lab_arena_events",
            "UPDATE public.lab_arena_ledger SET amount_microusd = 0",
            "DELETE FROM public.lab_arena_ledger",
            "UPDATE public.lab_arena_rounds SET configuration_hash = 'sha256:' || repeat('0', 64) WHERE round_id = %s" % ("'" + round_id + "'"),
            "DELETE FROM public.lab_arena_rounds WHERE round_id = '%s'" % round_id,
            "DELETE FROM public.lab_arena_runs WHERE run_id = '%s'" % run_id,
        ):
            with pytest.raises(Exception):
                cursor.execute(statement)
            superuser.rollback() if not superuser.autocommit else None
    with superuser.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM public.lab_arena_events WHERE run_id = %s", (run_id,))
        assert cursor.fetchone()[0] == 1


def test_journal_chain_is_enforced_and_idempotent(store):
    round_id = "arena-2026-09-02-jn"
    assert store.create_round(round_id, round_config(round_id, [hotkey("jn-runner")]))["status"] == "created"
    entries = []
    prev = ""
    for sequence in range(3):
        entry = contracts.finalize_journal_entry({
            "schema_version": contracts.GENERATION_JOURNAL_SCHEMA_VERSION, "sequence": sequence, "kind": "request",
            "batch_id": "b1", "attempt": 1, "slots": [0], "industries": ["Software"], "request_hash": sha("r%d" % sequence),
            "timestamp": "2026-09-02T00:00:00Z", "prev_hash": prev,
        })
        entries.append(entry)
        result = store.append_journal_entry(round_id, entry)
        assert result["status"] == "appended" and result["journal_length"] == sequence + 1
        prev = entry["entry_hash"]
    assert store.append_journal_entry(round_id, entries[-1])["status"] == "existing"
    with pytest.raises(ArenaStoreError):
        store.append_journal_entry(round_id, dict(entries[1], sequence=3))
    stored = store.get_round(round_id)["journal"]
    assert contracts.verify_journal_chain(stored) == prev


def test_concurrent_publications_produce_one_result(store, superuser, connect):
    round_id = "arena-2026-09-02-pub"
    assert store.create_round(round_id, round_config(round_id, [hotkey("pub-runner")]))["status"] == "created"
    with superuser.cursor() as cursor:
        cursor.execute("UPDATE public.lab_arena_rounds SET status = 'scored', stage1_scoring_plan_hash = %s, stage2_scoring_plan_hash = %s WHERE round_id = %s", (sha("p1"), sha("p2"), round_id))
    basis = {"reward_basis_hash": sha("basis"), "result_bundle_hash": sha("bundle"), "king_outcome": "crowned", "effective_reward_epoch": 100}
    patch = {
        "result_bundle_hash": sha("bundle"), "publication_doc": {"result_bundle_hash": sha("bundle")},
        "king_outcome": "crowned", "king_hotkey": hotkey("king"), "king_start_epoch": 100,
        "effective_reward_epoch": 100, "reward_basis_hash": sha("basis"), "reward_basis_doc": basis,
    }
    results = []

    def publish():
        transport = PsycopgTransport(connect)
        try:
            results.append(ArenaStore(transport).transition_round(round_id, "scored", "published", patch)["status"])
        finally:
            transport.close()

    threads = [threading.Thread(target=publish) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert sorted(results) == ["ok", "stale"]
    with superuser.cursor() as cursor:
        for statement in (
            "UPDATE public.lab_arena_rounds SET cancel_reason = 'x' WHERE round_id = %s",
            "UPDATE public.lab_arena_rounds SET status = 'scored' WHERE round_id = %s",
            "UPDATE public.lab_arena_rounds SET reward_basis_doc = '{}'::jsonb WHERE round_id = %s",
        ):
            with pytest.raises(Exception, match="immutable"):
                cursor.execute(statement, (round_id,))
    # A second published round cannot reuse the effective reward epoch.
    other = "arena-2026-09-03-pub"
    assert store.create_round(other, round_config(other, [hotkey("pub-runner")]))["status"] == "created"
    with superuser.cursor() as cursor:
        cursor.execute("UPDATE public.lab_arena_rounds SET status = 'scored', stage1_scoring_plan_hash = %s, stage2_scoring_plan_hash = %s WHERE round_id = %s", (sha("p3"), sha("p4"), other))
    other_patch = dict(patch, result_bundle_hash=sha("bundle2"), publication_doc={"result_bundle_hash": sha("bundle2")}, reward_basis_hash=sha("basis2"), reward_basis_doc=dict(basis, reward_basis_hash=sha("basis2"), result_bundle_hash=sha("bundle2")))
    with pytest.raises(ArenaStoreError):
        store.transition_round(other, "scored", "published", other_patch)


# ---------------------------------------------------------------------------
# 18.2 assignments, leases, and stage close
# ---------------------------------------------------------------------------


def test_hundred_concurrent_claims_through_two_instances_never_duplicate(connect):
    round_id = "arena-2026-09-02-cc"
    setup = ArenaStore(PsycopgTransport(connect))
    runner_keys, parts = open_round(setup, round_id, participants=8, runners=100, prefix="cc")
    instances = [ArenaStore(PsycopgTransport(connect)), ArenaStore(PsycopgTransport(connect))]
    responses: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def worker(index: int):
        response, _, _, _ = claim(instances[index % 2], round_id, runner_keys[index], parallelism=1)
        with lock:
            responses.append(response)

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(worker, range(100)))
    leased = [r for r in responses if r["status"] == "leased"]
    assert len(leased) == 100
    assert len({r["run_id"] for r in leased}) == 100
    assert len({r["assignment_id"] for r in leased}) == 100
    runs = setup.list_runs(round_id, stage=1, status="leased")
    assert len(runs) == 100
    # ICP-major order: the 100 leased assignments are exactly positions 0..11 for every
    # participant plus the first four of position 12 (8 participants * 12 = 96).
    positions = sorted(r["icp_position"] for r in leased)
    assert positions[:96] == sorted([p for p in range(12) for _ in range(8)])
    assert positions[96:] == [12, 12, 12, 12]
    for instance in instances:
        instance.close()
    setup.close()


def test_claim_replay_reuse_ceiling_and_self_exclusion(store, connect):
    round_id = "arena-2026-09-02-cl"
    runners, parts = open_round(store, round_id, participants=2, runners=2, prefix="cl")
    first, token, request_id, request_hash = claim(store, round_id, runners[0], parallelism=2)
    assert first["status"] == "leased" and first["icp_position"] == 0
    # Replay with the same bytes returns the stored response, also through a fresh instance.
    replay = store.claim_assignment(round_id=round_id, runner_hotkey=runners[0], declared_parallelism=2, slot_ceiling=8, excluded_miner_hotkeys=[], request_id=request_id, request_hash=request_hash, lease_token_hash=hash_lease_token(new_lease_token()))
    assert replay == first
    fresh = ArenaStore(PsycopgTransport(connect))
    assert fresh.claim_assignment(round_id=round_id, runner_hotkey=runners[0], declared_parallelism=2, slot_ceiling=8, excluded_miner_hotkeys=[], request_id=request_id, request_hash=request_hash, lease_token_hash=hash_lease_token(new_lease_token())) == first
    fresh.close()
    reused = store.claim_assignment(round_id=round_id, runner_hotkey=runners[0], declared_parallelism=2, slot_ceiling=8, excluded_miner_hotkeys=[], request_id=request_id, request_hash=sha("different"), lease_token_hash=hash_lease_token(new_lease_token()))
    assert reused["status"] == "request_id_reused"
    # Slot ceiling: declared 5 but ceiling 2 -> second lease ok, third refused.
    second, *_ = claim(store, round_id, runners[0], parallelism=5, ceiling=2)
    assert second["status"] == "leased"
    third, *_ = claim(store, round_id, runners[0], parallelism=5, ceiling=2)
    assert third["status"] == "no_free_slot" and third["slot_limit"] == 2
    # Declared parallelism 1 with a large ceiling also refuses.
    assert claim(store, round_id, runners[1], parallelism=1, ceiling=8)[0]["status"] == "leased"
    assert claim(store, round_id, runners[1], parallelism=1, ceiling=8)[0]["status"] == "no_free_slot"
    # Self-execution exclusion: excluding both miners leaves nothing.
    excluded, *_ = claim(store, round_id, hotkey("cl-runner-0"), parallelism=8, ceiling=8, excluded=[p["miner_hotkey"] for p in parts])
    assert excluded["status"] == "no_pending"
    only_one, *_ = claim(store, round_id, hotkey("cl-runner-0"), parallelism=8, ceiling=8, excluded=[parts[0]["miner_hotkey"]])
    assert only_one["status"] == "leased" and only_one["miner_hotkey"] == parts[1]["miner_hotkey"]
    # Not allowlisted runner.
    assert claim(store, round_id, hotkey("stranger"))[0]["status"] == "not_allowlisted"


def test_stale_lease_fails_provider_event_and_completion_and_expiry_retries_once(store, superuser):
    round_id = "arena-2026-09-02-lx"
    # Two Deepline calls per ICP attempt and a single attempt in the stage quota: the retry after
    # expiry inherits the lost attempt's uncertain call and is refused by the stage quota once
    # the per-ICP quota still has room.
    runners, parts = open_round(store, round_id, participants=1, prefix="lx", quotas={"deepline": 2, "scrapingdog": 30, "openrouter": 60}, stage_1_icps=1, max_attempts=1)
    response, token, _, _ = claim(store, round_id, runners[0], parallelism=8)
    run_id, lease_hash = response["run_id"], hash_lease_token(token)
    identity = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=0, action_sequence=0, operation_id="deepline.execute", request_hash=sha("q1"))
    dispatched_identity = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=0, action_sequence=1, operation_id="deepline.execute", request_hash=sha("q2"))
    assert store.reserve_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
    assert store.reserve_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=dispatched_identity, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
    assert store.mark_dispatched(run_id=run_id, lease_token_hash=lease_hash, call_identity=dispatched_identity)["status"] == "dispatched"
    # Wrong token is stale everywhere.
    wrong = hash_lease_token("other")
    assert store.mark_dispatched(run_id=run_id, lease_token_hash=wrong, call_identity=identity)["status"] == "stale"
    assert store.append_events(run_id=run_id, lease_token_hash=wrong, events=[make_event(response, 0, "")])["status"] == "stale"
    # Expiry: recover undispatched once, uncertain for dispatched, second attempt with fresh cap.
    expire_now(superuser, run_id)
    assert store.expire_leases(round_id) == {"status": "ok", "expired": 1, "retried": 1}
    assert store.expire_leases(round_id) == {"status": "ok", "expired": 0, "retried": 0}
    heads = {row["call_identity"]: row["entry_kind"] for row in store.list_ledger(run_id=run_id)}
    assert heads[identity] == "recovery" and heads[dispatched_identity] == "uncertain"
    for call in (store.reserve_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=sha("x"), operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={}),
                 store.settle_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=dispatched_identity, actual_microusd=0, terminal_response={}, event=None),
                 store.append_events(run_id=run_id, lease_token_hash=lease_hash, events=[make_event(response, 0, "")]),
                 store.complete_attempt(run_id=run_id, lease_token_hash=lease_hash, receipt={"receipt_hash": sha("r")}, receipt_hash=sha("r"), terminal_cause="accepted", output_hash=sha("o"), output_ref="ref", provider_call_root=sha("a"), private_event_root=sha("b"), cost_root=sha("c"))):
        assert call["status"] == "stale"
    old = store.get_run(run_id)
    assert old["status"] == "failed" and old["terminal_cause"] == "lease_expired"
    second, token2, _, _ = claim(store, round_id, runners[0], parallelism=8)
    assert second["assignment_id"] == response["assignment_id"] and second["attempt"] == 2
    assert "per_icp_cap_microusd" not in second and second["lease_generation"] == 2
    # The stage quota still counts the lost attempt's uncertain call (the recovered one does not).
    lease2 = hash_lease_token(token2)
    heavy = contracts.provider_call_identity(assignment_id=second["assignment_id"], icp_position=0, action_sequence=0, operation_id="deepline.execute", request_hash=sha("heavy"))
    first_ok = contracts.provider_call_identity(assignment_id=second["assignment_id"], icp_position=0, action_sequence=1, operation_id="deepline.execute", request_hash=sha("first-ok"))
    assert store.reserve_call(run_id=second["run_id"], lease_token_hash=lease2, call_identity=first_ok, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
    refused = store.reserve_call(run_id=second["run_id"], lease_token_hash=lease2, call_identity=heavy, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})
    assert refused["status"] == "refused" and refused["reason"] == "stage_quota"
    # Another provider's quota is independent, and the refusal is recorded and replayed.
    other = contracts.provider_call_identity(assignment_id=second["assignment_id"], icp_position=0, action_sequence=2, operation_id="scrapingdog.google", request_hash=sha("other"))
    assert store.reserve_call(run_id=second["run_id"], lease_token_hash=lease2, call_identity=other, operation_id="scrapingdog.google", provider="scrapingdog", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
    assert store.reserve_call(run_id=second["run_id"], lease_token_hash=lease2, call_identity=heavy, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "refused"
    # No third attempt after the second expires.
    expire_now(superuser, second["run_id"])
    assert store.expire_leases(round_id) == {"status": "ok", "expired": 1, "retried": 0}
    assert claim(store, round_id, runners[0], parallelism=8)[0]["status"] == "leased"  # other positions remain
    attempts = [r for r in store.list_runs(round_id, stage=1) if r["assignment_id"] == response["assignment_id"]]
    assert sorted(r["attempt"] for r in attempts) == [1, 2]


def test_model_caused_failure_gets_no_second_attempt_and_completion_requires_closed_accounting(store):
    round_id = "arena-2026-09-02-mc"
    runners, parts = open_round(store, round_id, participants=1, prefix="mc")
    response, token, _, _ = claim(store, round_id, runners[0], parallelism=8)
    run_id, lease_hash = response["run_id"], hash_lease_token(token)
    identity = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=0, action_sequence=0, operation_id="openrouter.chat", request_hash=sha("q"))
    assert store.reserve_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, operation_id="openrouter.chat", provider="openrouter", funding_source="miner_key", amount_microusd=9000, call_doc={})["status"] == "reserved"
    receipt = {"receipt_hash": sha("receipt")}
    blocked = store.complete_attempt(run_id=run_id, lease_token_hash=lease_hash, receipt=receipt, receipt_hash=sha("receipt"), terminal_cause="model_timeout", output_hash="", output_ref="", provider_call_root=sha("a"), private_event_root=sha("b"), cost_root=sha("c"))
    assert blocked == {"status": "accounting_open", "open_calls": 1}
    assert store.mark_dispatched(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity)["status"] == "dispatched"
    event = make_event(response, 0, "", event_type="provider_call")
    settled = store.settle_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, actual_microusd=4000, terminal_response={"status": 200}, event=event)
    assert settled["status"] == "settled" and settled["released_microusd"] == 5000 and settled["event_cursor"] == 1
    account = store.get_account(parts[0]["miner_hotkey"])
    assert account["outstanding_openrouter_reservation_microusd"] == 0 and account["settled_since_preflight_microusd"] == 4000
    again = store.settle_call(run_id=run_id, lease_token_hash=lease_hash, call_identity=identity, actual_microusd=1, terminal_response={}, event=None)
    assert again["status"] == "settled" and again["idempotent"] is True and again["terminal_response"] == {"status": 200}
    assert store.get_account(parts[0]["miner_hotkey"])["settled_since_preflight_microusd"] == 4000  # released once
    done = store.complete_attempt(run_id=run_id, lease_token_hash=lease_hash, receipt=receipt, receipt_hash=sha("receipt"), terminal_cause="model_timeout", output_hash="", output_ref="", provider_call_root=sha("a"), private_event_root=sha("b"), cost_root=sha("c"))
    assert done["status"] == "failed"
    replay = store.complete_attempt(run_id=run_id, lease_token_hash=lease_hash, receipt=receipt, receipt_hash=sha("receipt"), terminal_cause="model_timeout", output_hash="", output_ref="", provider_call_root=sha("a"), private_event_root=sha("b"), cost_root=sha("c"))
    assert replay["status"] == "failed" and replay["idempotent"] is True
    assert store.expire_leases(round_id)["expired"] == 0
    # No second attempt exists for a model-caused failure; the other 19 positions remain claimable.
    attempts = [r for r in store.list_runs(round_id, stage=1) if r["assignment_id"] == response["assignment_id"]]
    assert [r["attempt"] for r in attempts] == [1]
    nxt, *_ = claim(store, round_id, runners[0], parallelism=8)
    assert nxt["status"] == "leased" and nxt["icp_position"] == 1


def test_close_stage_races_hundred_operations_without_deadlock(connect):
    round_id = "arena-2026-09-02-cs"
    setup = ArenaStore(PsycopgTransport(connect))
    runners, parts = open_round(setup, round_id, participants=5, runners=1, prefix="cs")
    leases = []
    for _ in range(100):
        response, token, _, _ = claim(setup, round_id, runners[0], parallelism=100, ceiling=100)
        assert response["status"] == "leased"
        leases.append((response, hash_lease_token(token)))
    # Prepare one reserved and one dispatched call on every lease before the race.
    for index, (response, lease_hash) in enumerate(leases):
        reserved = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=response["icp_position"], action_sequence=0, operation_id="deepline.execute", request_hash=sha("r%d" % index))
        dispatched = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=response["icp_position"], action_sequence=1, operation_id="deepline.execute", request_hash=sha("d%d" % index))
        assert setup.reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=reserved, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
        assert setup.reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=dispatched, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
        assert setup.mark_dispatched(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=dispatched)["status"] == "dispatched"
    workers = [ArenaStore(PsycopgTransport(connect)) for _ in range(2)]
    closer = ArenaStore(PsycopgTransport(connect))
    outcomes: List[str] = []
    lock = threading.Lock()
    start = threading.Barrier(9)

    def late_work(index: int):
        response, lease_hash = leases[index]
        worker = workers[index % 2]
        kind = index % 6
        start.wait(timeout=30)
        identity_r = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=response["icp_position"], action_sequence=0, operation_id="deepline.execute", request_hash=sha("r%d" % index))
        identity_d = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=response["icp_position"], action_sequence=1, operation_id="deepline.execute", request_hash=sha("d%d" % index))
        if kind == 0:
            result = claim(worker, round_id, runners[0], parallelism=100, ceiling=100)[0]
        elif kind == 1:
            result = worker.reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=sha("new%d" % index), operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})
        elif kind == 2:
            result = worker.mark_dispatched(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity_r)
        elif kind == 3:
            result = worker.settle_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity_d, actual_microusd=100, terminal_response={"ok": True}, event=None)
        elif kind == 4:
            result = worker.append_events(run_id=response["run_id"], lease_token_hash=lease_hash, events=[make_event(response, 0, "")])
        else:
            result = worker.complete_attempt(run_id=response["run_id"], lease_token_hash=lease_hash, receipt={"receipt_hash": sha("rc%d" % index)}, receipt_hash=sha("rc%d" % index), terminal_cause="accepted", output_hash=sha("o"), output_ref="ref", provider_call_root=sha("a"), private_event_root=sha("b"), cost_root=sha("c"))
        with lock:
            outcomes.append("%d:%s" % (kind, result["status"]))

    def close():
        start.wait(timeout=30)
        result = closer.close_stage(round_id, 1)
        with lock:
            outcomes.append("close:%s" % result["status"])

    threads = [threading.Thread(target=late_work, args=(i,)) for i in range(8)]
    threads.append(threading.Thread(target=close))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads), "deadlock or hang"
    # Remaining 92 leases are handled after the close: every late op is stale.
    for index in range(8, 100):
        response, lease_hash = leases[index]
        assert workers[index % 2].append_events(run_id=response["run_id"], lease_token_hash=lease_hash, events=[make_event(response, 0, "")])["status"] == "stale"
    assert "close:cancelled" in outcomes or "close:closed" in outcomes
    final = setup.get_round(round_id)
    assert final["status"] == "cancelled"  # 100 in-flight leases lacked accepted results for an infrastructure reason
    heads: Dict[str, str] = {}
    for row in setup.list_ledger():
        if row["round_id"] == round_id and row["call_identity"]:
            heads[row["call_identity"]] = row["entry_kind"]
    assert not any(kind in ("reservation", "dispatch") for kind in heads.values()), "an open call survived the close"
    assert all(r["status"] == "failed" for r in setup.list_runs(round_id, stage=1) if r["attempt"] >= 1)
    for instance in workers + [closer, setup]:
        instance.close()


def test_close_stage_with_only_model_failures_freezes_results(store):
    round_id = "arena-2026-09-02-cf"
    runners, parts = open_round(store, round_id, participants=1, prefix="cf")
    for position in range(20):
        response, token, _, _ = claim(store, round_id, runners[0], parallelism=8)
        assert response["icp_position"] == position
        cause = "accepted" if position % 2 == 0 else "invalid_output"
        result = store.complete_attempt(run_id=response["run_id"], lease_token_hash=hash_lease_token(token), receipt={"receipt_hash": sha("rc%d" % position)}, receipt_hash=sha("rc%d" % position), terminal_cause=cause, output_hash=sha("o%d" % position) if cause == "accepted" else "", output_ref="ref", provider_call_root=sha("a"), private_event_root=sha("b"), cost_root=sha("c"))
        assert result["status"] == ("accepted" if cause == "accepted" else "failed")
    closed = store.close_stage(round_id, 1)
    assert closed["status"] == "closed" and closed["incomplete_assignments"] == 0
    assert store.close_stage(round_id, 1)["status"] == "existing"
    assert store.get_round(round_id)["status"] == "stage1_closed"
    # Scores are write-once per attempt.
    runs = store.list_runs(round_id, stage=1)
    scores = [{"run_id": r["run_id"], "per_icp_score": 12.5 if r["status"] == "accepted" else 0.0, "score_ref": "s"} for r in runs]
    assert store.record_run_scores(round_id, 1, scores)["recorded"] == 20
    assert store.record_run_scores(round_id, 1, scores)["existing"] == 20
    with pytest.raises(ArenaStoreError):
        store.record_run_scores(round_id, 1, [dict(scores[0], per_icp_score=1.0)])


def test_preflight_failed_king_records_and_stage_two_positions(store):
    round_id = "arena-2026-09-02-kg"
    runner_keys = [hotkey("kg-runner")]
    assert store.create_round(round_id, round_config(round_id, runner_keys))["status"] == "created"
    parts = frozen_participants(store, round_id, 2, prefix="kg", king_index=1)
    parts[1]["preflight_failed"] = True
    commit_round(store, round_id, parts)
    positions = stage_positions(1)
    assert store.open_stage(round_id, 1, parts, positions, [sha("i%d" % p) for p in positions])["status"] == "ok"
    king_runs = store.list_runs(round_id, stage=1, submission_id=parts[1]["submission_id"])
    assert len(king_runs) == 20 and all(r["attempt"] == 0 and r["terminal_cause"] == "preflight_failed" and r["status"] == "failed" for r in king_runs)
    assert claim(store, round_id, runner_keys[0], parallelism=8)[0]["miner_hotkey"] == parts[0]["miner_hotkey"]
    # Stage 2 requires stage1_scored and positions 20..49.
    assert store.open_stage(round_id, 2, parts, stage_positions(2), [sha("i%d" % p) for p in stage_positions(2)])["status"] == "stale"
    assert store.cancel_round(round_id, "test")["status"] == "cancelled"
    assert store.cancel_round(round_id, "test")["status"] == "existing"
    assert claim(store, round_id, runner_keys[0], parallelism=8)[0]["status"] == "stage_closed"


# ---------------------------------------------------------------------------
# 18.3 accounting and dispatch
# ---------------------------------------------------------------------------


def test_concurrent_reservations_never_exceed_the_call_quota(connect):
    round_id = "arena-2026-09-02-ov"
    setup = ArenaStore(PsycopgTransport(connect))
    runners, parts = open_round(setup, round_id, participants=1, prefix="ov", quotas={"deepline": 12, "scrapingdog": 30, "openrouter": 60})
    miner = parts[0]["miner_hotkey"]
    response, token, _, _ = claim(setup, round_id, runners[0], parallelism=8)
    lease_hash = hash_lease_token(token)
    instances = [ArenaStore(PsycopgTransport(connect)), ArenaStore(PsycopgTransport(connect))]
    results = []
    lock = threading.Lock()

    def reserve(index: int):
        identity = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=0, action_sequence=index, operation_id="deepline.execute", request_hash=sha("c%d" % index))
        result = instances[index % 2].reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity, operation_id="deepline.execute", provider="deepline", funding_source="miner_key", amount_microusd=0, call_doc={})
        with lock:
            results.append((result["status"], result.get("reason")))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(reserve, range(50)))
    statuses = [status for status, _ in results]
    assert statuses.count("reserved") == 12  # the per-ICP Deepline quota, under sixteen concurrent workers
    assert statuses.count("refused") == 38 and {reason for status, reason in results if status == "refused"} == {"per_icp_quota"}
    # Quotas are per provider: Scrapingdog calls on the same attempt are untouched by the Deepline quota.
    other = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=0, action_sequence=99, operation_id="scrapingdog.google", request_hash=sha("g"))
    assert setup.reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=other, operation_id="scrapingdog.google", provider="scrapingdog", funding_source="miner_key", amount_microusd=0, call_doc={})["status"] == "reserved"
    # A miner missing one provider key is refused for every provider, even the ones it holds.
    account = setup.get_account(miner)
    assert account["preflight_status"] == "ok" and "balance_microusd" not in account
    setup.record_preflight(miner, "deepline", {"preflight_status": "failed", "provider": "deepline", "key_hash": account["credentials"]["deepline"]["key_hash"]})
    blocked = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=0, action_sequence=100, operation_id="scrapingdog.google", request_hash=sha("blocked"))
    refused = setup.reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=blocked, operation_id="scrapingdog.google", provider="scrapingdog", funding_source="miner_key", amount_microusd=0, call_doc={})
    assert refused["status"] == "refused" and refused["reason"] == "key_preflight"
    for instance in instances + [setup]:
        instance.close()


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_openrouter_capacity_dispatch_uniqueness_and_settlement_rules(connect):
    round_id = "arena-2026-09-02-or"
    setup = ArenaStore(PsycopgTransport(connect))
    runners, parts = open_round(setup, round_id, participants=1, prefix="or")
    miner = parts[0]["miner_hotkey"]
    response, token, _, _ = claim(setup, round_id, runners[0], parallelism=8)
    lease_hash = hash_lease_token(token)
    # Observed remaining 10_000_000: 9 reservations of 1_000_000 fit alongside nothing else; the 11th fails.
    results = []
    lock = threading.Lock()
    instances = [ArenaStore(PsycopgTransport(connect)), ArenaStore(PsycopgTransport(connect))]

    def reserve(index: int):
        identity = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=0, action_sequence=index, operation_id="openrouter.chat", request_hash=sha("o%d" % index))
        result = instances[index % 2].reserve_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity, operation_id="openrouter.chat", provider="openrouter", funding_source="miner_key", amount_microusd=1_000_000, call_doc={})
        with lock:
            results.append((index, result["status"]))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(reserve, range(15)))
    statuses = [status for _, status in results]
    assert statuses.count("reserved") == 10 and statuses.count("refused") == 5
    account = setup.get_account(miner)
    assert account["outstanding_openrouter_reservation_microusd"] == 10_000_000
    reserved_index = next(index for index, status in results if status == "reserved")
    identity = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=0, action_sequence=reserved_index, operation_id="openrouter.chat", request_hash=sha("o%d" % reserved_index))
    # Two instances mark the same dispatch: exactly one non-idempotent dispatch.
    dispatch_results = []

    def dispatch(index: int):
        result = instances[index % 2].mark_dispatched(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity)
        with lock:
            dispatch_results.append(result["idempotent"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(dispatch, range(4)))
    assert dispatch_results.count(False) == 1 and dispatch_results.count(True) == 3
    assert len([row for row in setup.list_ledger(call_identity=identity) if row["entry_kind"] == "dispatch"]) == 1
    # Settlement cannot exceed the reservation.
    with pytest.raises(ArenaStoreError):
        setup.settle_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity, actual_microusd=1_000_001, terminal_response={}, event=None)
    settled = setup.settle_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=identity, actual_microusd=250_000, terminal_response={"usage": {"total_tokens": 10}}, event=None)
    assert settled["status"] == "settled" and settled["released_microusd"] == 750_000
    account = setup.get_account(miner)
    assert account["outstanding_openrouter_reservation_microusd"] == 9_000_000 and account["settled_since_preflight_microusd"] == 250_000
    # A worker-reported uncertain call consumes its full reservation and a late settle cannot release it.
    uncertain_index = next(index for index, status in results if status == "reserved" and index != reserved_index)
    uid = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=0, action_sequence=uncertain_index, operation_id="openrouter.chat", request_hash=sha("o%d" % uncertain_index))
    assert setup.mark_dispatched(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=uid)["status"] == "dispatched"
    assert setup.mark_uncertain(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=uid, call_doc={"why": "timeout"}, event=None)["status"] == "uncertain"
    late = setup.settle_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=uid, actual_microusd=0, terminal_response={}, event=None)
    assert late["status"] == "uncertain" and late["amount_microusd"] == 1_000_000
    account = setup.get_account(miner)
    assert account["settled_since_preflight_microusd"] == 1_250_000 and account["outstanding_openrouter_reservation_microusd"] == 8_000_000
    # Settling a merely reserved (never dispatched) call is refused; a preflight refresh resets settled-since.
    rid = next(index for index, status in results if status == "reserved" and index not in (reserved_index, uncertain_index))
    r_identity = contracts.provider_call_identity(assignment_id=response["assignment_id"], icp_position=0, action_sequence=rid, operation_id="openrouter.chat", request_hash=sha("o%d" % rid))
    assert setup.settle_call(run_id=response["run_id"], lease_token_hash=lease_hash, call_identity=r_identity, actual_microusd=1, terminal_response={}, event=None)["status"] == "reserved"
    refreshed = setup.record_preflight(miner, "openrouter", {"preflight_status": "ok", "provider": "openrouter", "key_hash": sha("key" + parts[0]["submission_id"] + "openrouter")[7:], "limit_microusd": 20_000_000, "limit_remaining_microusd": 12_000_000, "usage_microusd": 8_000_000})
    assert refreshed["settled_since_preflight_microusd"] == 0 and refreshed["observed_limit_remaining_microusd"] == 12_000_000
    with pytest.raises(ArenaStoreError):
        setup.record_preflight(miner, "openrouter", {"preflight_status": "ok", "provider": "openrouter", "key_hash": "0" * 64})
    for instance in instances + [setup]:
        instance.close()


def test_events_are_ordered_hash_chained_and_replay_safe(store):
    round_id = "arena-2026-09-02-ev"
    runners, parts = open_round(store, round_id, participants=1, prefix="ev")
    response, token, _, _ = claim(store, round_id, runners[0], parallelism=8)
    run_id, lease_hash = response["run_id"], hash_lease_token(token)
    first = make_event(response, 0, "", event_type="process_started")
    second = make_event(response, 1, first["event_hash"], event_type="stdout")
    result = store.append_events(run_id=run_id, lease_token_hash=lease_hash, events=[first, second])
    assert result["status"] == "appended" and result["event_cursor"] == 2 and result["event_head_hash"] == second["event_hash"]
    replay = store.append_events(run_id=run_id, lease_token_hash=lease_hash, events=[first, second])
    assert replay["status"] == "existing" and replay["event_cursor"] == 2
    with pytest.raises(ArenaStoreError):
        store.append_events(run_id=run_id, lease_token_hash=lease_hash, events=[make_event(response, 5, second["event_hash"])])
    with pytest.raises(ArenaStoreError):
        store.append_events(run_id=run_id, lease_token_hash=lease_hash, events=[make_event(response, 2, sha("wrong prev"))])
    with pytest.raises(ArenaStoreError):
        store.append_events(run_id=run_id, lease_token_hash=lease_hash, events=[dict(first, event_hash=sha("tamper"))])
    third = make_event(response, 2, second["event_hash"], event_type="trajectory")
    assert store.append_events(run_id=run_id, lease_token_hash=lease_hash, events=[third])["event_cursor"] == 3
    stored = store.list_events(run_id)
    assert [row["sequence"] for row in stored] == [0, 1, 2]
    assert [row["event_hash"] for row in stored] == [first["event_hash"], second["event_hash"], third["event_hash"]]
