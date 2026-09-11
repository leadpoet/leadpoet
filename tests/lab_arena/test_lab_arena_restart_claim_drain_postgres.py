"""PostgreSQL contract tests for the canonical restart claim drain."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    LAB_ARENA_RESTART_CLAIM_DRAIN_MIGRATION,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    commit_round,
    complete,
    encrypted_runtime_credentials,
    frozen_participants,
    hotkey,
    open_round,
    round_config,
    source_submission_doc,
    stage_positions,
)
from tests.postgres_migration_harness import SCRIPTS


CANDIDATE = "a" * 40
GUARD = "lab_arena_restart_guard:" + "b" * 64
OWNER = "lab_arena_restart_owner:" + "c" * 64
NEW_CANDIDATE = "d" * 40
NEW_GUARD = "lab_arena_restart_guard:" + "e" * 64
OLD_CLAIM_BARRIER = "lab-arena-test-old-182-before-round-read"


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration()


def _store(psycopg2, dsn):
    return ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)), lease_ttl_seconds=120)


def _rpc(connection, function, *values):
    placeholders = ",".join(["%s"] * len(values))
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT public.{function}({placeholders})", values)
        return cursor.fetchone()[0]


def _historical_frozen_participants(
    store: ArenaStore, round_id: str, count: int, *, prefix: str
):
    """Create frozen participants before the code-review RPC existed."""

    participants = []
    for index in range(count):
        miner = hotkey("%s-miner-%d" % (prefix, index))
        submission_id = "%s-sub-%d" % (prefix, index)
        assert store.register_submission(
            round_id,
            submission_id,
            miner,
            source_submission_doc(round_id, submission_id),
        )["status"] == "registered"
        assert store.accept_submission_with_credentials(
            round_id,
            submission_id,
            miner,
            encrypted_runtime_credentials(submission_id),
        )["status"] == "ok"
        assert store.update_submission(
            round_id, submission_id, "accepted", "frozen", {"is_king": False}
        )["status"] == "ok"
        participants.append(
            {
                "submission_id": submission_id,
                "miner_hotkey": miner,
                "is_king": False,
            }
        )
    return participants


def _historical_open_round(
    store: ArenaStore,
    round_id: str,
    *,
    participants: int,
    runners: int,
    prefix: str,
):
    """Open a round against the intentionally pre-review migration set."""

    runner_keys = [hotkey("%s-runner-%d" % (prefix, index)) for index in range(runners)]
    assert store.create_round(round_id, round_config(round_id, runner_keys))[
        "status"
    ] == "created"
    parts = _historical_frozen_participants(
        store, round_id, participants, prefix=prefix
    )
    commit_round(store, round_id, parts)
    assert store.open_stage(round_id, 1, parts, stage_positions(1))["status"] == "ok"
    return runner_keys, parts


def _acquire(connection, *, guard=GUARD, owner=OWNER, generation=0, scope="gateway"):
    return _rpc(
        connection,
        "lab_arena_acquire_restart_guard_v1",
        guard,
        owner,
        generation,
        600,
        CANDIDATE,
        scope,
        "test-restart",
    )


def _quiescence(connection, generation=1):
    return _rpc(
        connection, "lab_arena_restart_quiescence_v1",
        GUARD, OWNER, generation,
    )


def _install_instrumented_old_claim(connection):
    source = (SCRIPTS / "182-lab-arena-source-execution.sql").read_text()
    start = source.index("CREATE OR REPLACE FUNCTION public.lab_arena_claim_assignment(")
    end_marker = ") OWNER TO lab_arena_owner;"
    end = source.index(end_marker, start) + len(end_marker)
    claim_source = source[start:end]
    needle = (
        "  SELECT * INTO v_round FROM public.lab_arena_rounds "
        "WHERE round_id = p_round_id FOR SHARE;"
    )
    assert claim_source.count(needle) == 1
    claim_source = claim_source.replace(
        needle,
        "  PERFORM pg_catalog.pg_advisory_xact_lock("
        "pg_catalog.hashtextextended('" + OLD_CLAIM_BARRIER + "', 0));\n" + needle,
    )
    with connection.cursor() as cursor:
        cursor.execute(claim_source)


def _wait_for_advisory_barrier(connection, application_name):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT wait_event_type, wait_event FROM pg_catalog.pg_stat_activity "
                "WHERE application_name = %s AND state = 'active'",
                (application_name,),
            )
            row = cursor.fetchone()
        if row == ("Lock", "advisory"):
            return
        time.sleep(0.01)
    raise AssertionError("old migration-182 claim did not enter the test barrier")


def test_pause_preserves_duplicate_replay_and_accepted_receipt(database):
    psycopg2, dsn = database
    store = _store(psycopg2, dsn)
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    try:
        runners, _ = open_round(store, "arena-2099-01-01", participants=1, runners=1, prefix="drain-a")
        leased, token, request_id, request_hash = claim(store, "arena-2099-01-01", runners[0])
        state = _acquire(admin)
        assert state["drain"]["captured_count"] == 1
        replay = store.claim_assignment(
            round_id="arena-2099-01-01", runner_hotkey=runners[0],
            declared_parallelism=1, slot_ceiling=8, excluded_miner_hotkeys=[],
            request_id=request_id, request_hash=request_hash,
            lease_token_hash=hash_lease_token(token), lease_ttl_seconds=120,
        )
        assert replay == leased
        assert claim(store, "arena-2099-01-01", runners[0], parallelism=2)[0]["status"] == "paused"
        assert complete(store, leased["run_id"], hash_lease_token(token), "accepted", output_ref="arena/result.json")["status"] == "accepted"
        drain = _quiescence(admin)
        assert drain["preserved"] is True
        assert drain["accepted_receipt_count"] == 1
        assert drain["reported_terminal_receipt_count"] == 0
        _rpc(admin, "lab_arena_authorize_restart_phase_v1", GUARD, OWNER, 1, "gateway_destructive")
        _rpc(admin, "lab_arena_mark_restart_ready_v1", GUARD, OWNER, 1, "gateway_ready")
        released = _rpc(admin, "lab_arena_release_restart_guard_v1", GUARD, OWNER, 1, "test-release")
        assert released["guard_present"] is False
    finally:
        store._transport.close()
        admin.close()


def test_reported_failure_is_preserved_but_expiry_aborts_and_restores(database):
    psycopg2, dsn = database
    store = _store(psycopg2, dsn)
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    try:
        runners, _ = open_round(store, "arena-2099-01-02", participants=1, runners=1, prefix="drain-r")
        leased, token, _, _ = claim(store, "arena-2099-01-02", runners[0])
        _acquire(admin, scope="validator")
        call_identity = contracts.provider_call_identity(
            attempt=1,
            assignment_id=leased["assignment_id"],
            icp_position=leased["icp_position"],
            action_sequence=1,
            operation_id="openrouter.chat",
            request_hash="sha256:" + "1" * 64,
        )
        lease_hash = hash_lease_token(token)
        assert store.reserve_call(
            run_id=leased["run_id"], lease_token_hash=lease_hash,
            call_identity=call_identity, operation_id="openrouter.chat",
            provider="openrouter", funding_source="miner_key",
            amount_microusd=1, call_doc={}, lease_ttl_seconds=120,
        )["status"] == "reserved"
        assert store.mark_dispatched(
            run_id=leased["run_id"], lease_token_hash=lease_hash,
            call_identity=call_identity,
        )["status"] == "dispatched"
        assert store.mark_uncertain(
            run_id=leased["run_id"], lease_token_hash=lease_hash,
            call_identity=call_identity, call_doc={},
        )["status"] == "uncertain"
        complete(store, leased["run_id"], hash_lease_token(token), "model_error")
        drain = _quiescence(admin)
        assert drain["preserved"] is True
        assert drain["reported_terminal_receipt_count"] == 1
        assert drain["pending_retry_count"] == 1
        _rpc(admin, "lab_arena_authorize_restart_phase_v1", GUARD, OWNER, 1, "validator_destructive")
        _rpc(admin, "lab_arena_mark_restart_ready_v1", GUARD, OWNER, 1, "validator_ready")
        _rpc(admin, "lab_arena_release_restart_guard_v1", GUARD, OWNER, 1, "test-release")

        retry, _, _, _ = claim(store, "arena-2099-01-02", runners[0])
        assert retry["attempt"] == 2
        state = _acquire(admin, generation=1)
        assert state["guard_generation"] == 2
        with admin.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE run_id = %s",
                (retry["run_id"],),
            )
        store.expire_leases("arena-2099-01-02")
        drain = _quiescence(admin, 2)
        assert drain["preserved"] is False
        assert drain["lost_or_mutated_count"] == 1
        state = _rpc(admin, "lab_arena_abort_restart_guard_v1", GUARD, OWNER, 2, "test-abort")
        assert state["paused"] is False
        assert state["guard_present"] is False
    finally:
        store._transport.close()
        admin.close()


def test_owner_expiry_cas_and_idempotent_reapply_preserve_guard(database):
    psycopg2, dsn = database
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    try:
        state = _acquire(admin)
        assert state["guard_generation"] == 1
        with admin.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_restart_claim_control SET guard_expires_at = clock_timestamp() - interval '1 second' WHERE singleton"
            )
        with pytest.raises(psycopg2.Error, match="owned_by_another_invocation"):
            _acquire(
                admin,
                guard="lab_arena_restart_guard:" + "d" * 64,
                owner="lab_arena_restart_owner:" + "e" * 64,
                generation=1,
            )
        renewed = _acquire(admin, generation=1)
        assert renewed["guard_generation"] == 1
        with admin.cursor() as cursor:
            cursor.execute((SCRIPTS / LAB_ARENA_RESTART_CLAIM_DRAIN_MIGRATION).read_text())
        retained = _rpc(admin, "lab_arena_restart_guard_state_v1")
        assert retained["guard_generation"] == 1
        assert retained["guard_present"] is True
        assert retained["restart_phase"] == "draining"
    finally:
        admin.close()


def test_post_destructive_exact_owner_retarget_preserves_state(database):
    psycopg2, dsn = database
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_restart_claim_control SET "
                "operator_paused = TRUE, pause_reason = 'operator' WHERE singleton"
            )
        _acquire(admin, scope="all")
        _rpc(
            admin, "lab_arena_authorize_restart_phase_v1",
            GUARD, OWNER, 1, "gateway_destructive",
        )
        state = _rpc(
            admin, "lab_arena_retarget_restart_guard_v1",
            GUARD, OWNER, 1, NEW_GUARD, NEW_CANDIDATE, "all", 600,
            "test-retarget",
        )
        assert state["guard_generation"] == 2
        assert state["candidate_commit"] == NEW_CANDIDATE
        assert state["restart_phase"] == "gateway_destructive"
        assert state["operator_paused"] is True
        assert state["drain"]["captured_count"] == 0
        with pytest.raises(psycopg2.Error, match="retarget_forbidden"):
            _rpc(
                admin, "lab_arena_retarget_restart_guard_v1",
                GUARD, OWNER, 1, NEW_GUARD, NEW_CANDIDATE, "all", 600,
                "stale-retarget",
            )
    finally:
        admin.close()


def test_retarget_rejects_draining_null_and_lost_snapshot(database):
    psycopg2, dsn = database
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    valid = [
        GUARD, OWNER, 1, NEW_GUARD, NEW_CANDIDATE, "gateway", 600,
        "test-retarget",
    ]
    try:
        _acquire(admin)
        with pytest.raises(psycopg2.Error, match="retarget_forbidden"):
            _rpc(admin, "lab_arena_retarget_restart_guard_v1", *valid)
        for index in range(len(valid)):
            values = list(valid)
            values[index] = None
            with pytest.raises(
                psycopg2.Error, match="lab_arena_restart_guard_input_invalid"
            ):
                _rpc(admin, "lab_arena_retarget_restart_guard_v1", *values)

        _rpc(
            admin, "lab_arena_authorize_restart_phase_v1",
            GUARD, OWNER, 1, "gateway_destructive",
        )
        with admin.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_restart_claim_control SET "
                "captured_leases = '[{\"run_id\":\"missing\","
                "\"lease_generation\":1}]'::jsonb WHERE singleton"
            )
        with pytest.raises(psycopg2.Error, match="drain_not_preserved"):
            _rpc(admin, "lab_arena_retarget_restart_guard_v1", *valid)
    finally:
        admin.close()


def test_paired_phase_order_and_operator_pause_are_preserved(database):
    psycopg2, dsn = database
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_restart_claim_control SET operator_paused = TRUE, pause_reason = 'operator' WHERE singleton"
            )
        state = _acquire(admin, scope="all")
        assert state["paused"] is True and state["operator_paused"] is True
        _rpc(admin, "lab_arena_authorize_restart_phase_v1", GUARD, OWNER, 1, "gateway_destructive")
        _rpc(admin, "lab_arena_mark_restart_ready_v1", GUARD, OWNER, 1, "gateway_ready")
        _rpc(admin, "lab_arena_authorize_restart_phase_v1", GUARD, OWNER, 1, "validator_destructive")
        _rpc(admin, "lab_arena_mark_restart_ready_v1", GUARD, OWNER, 1, "validator_ready")
        state = _rpc(admin, "lab_arena_release_restart_guard_v1", GUARD, OWNER, 1, "test-release")
        assert state["paused"] is True
        assert state["operator_paused"] is True
        assert state["guard_present"] is False
    finally:
        admin.close()


@pytest.mark.parametrize(
    ("scope", "phase"),
    [
        ("gateway", "gateway_destructive"),
        ("gateway", "gateway_ready"),
        ("validator", "validator_destructive"),
        ("validator", "validator_ready"),
        ("all", "gateway_destructive"),
        ("all", "gateway_ready"),
        ("all", "validator_destructive"),
        ("all", "validator_ready"),
    ],
)
def test_exact_owner_normal_canonical_retry_repeats_full_phase_path(
    database, scope, phase,
):
    psycopg2, dsn = database
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    try:
        _acquire(admin, scope=scope)
        if scope in ("gateway", "all"):
            _rpc(
                admin, "lab_arena_authorize_restart_phase_v1",
                GUARD, OWNER, 1, "gateway_destructive",
            )
            if phase != "gateway_destructive":
                _rpc(
                    admin, "lab_arena_mark_restart_ready_v1",
                    GUARD, OWNER, 1, "gateway_ready",
                )
        if scope in ("validator", "all") and phase in (
            "validator_destructive", "validator_ready",
        ):
            _rpc(
                admin, "lab_arena_authorize_restart_phase_v1",
                GUARD, OWNER, 1, "validator_destructive",
            )
            if phase == "validator_ready":
                _rpc(
                    admin, "lab_arena_mark_restart_ready_v1",
                    GUARD, OWNER, 1, "validator_ready",
                )
        retained = _acquire(admin, generation=1, scope=scope)
        assert retained["guard_generation"] == 1
        assert retained["restart_phase"] == phase

        if scope in ("gateway", "all"):
            _rpc(
                admin, "lab_arena_authorize_restart_phase_v1",
                GUARD, OWNER, 1, "gateway_destructive",
            )
            _rpc(
                admin, "lab_arena_mark_restart_ready_v1",
                GUARD, OWNER, 1, "gateway_ready",
            )
        if scope in ("validator", "all"):
            _rpc(
                admin, "lab_arena_authorize_restart_phase_v1",
                GUARD, OWNER, 1, "validator_destructive",
            )
            _rpc(
                admin, "lab_arena_mark_restart_ready_v1",
                GUARD, OWNER, 1, "validator_ready",
            )
        released = _rpc(
            admin, "lab_arena_release_restart_guard_v1",
            GUARD, OWNER, 1, "test-release",
        )
        assert released["guard_present"] is False
    finally:
        admin.close()


def test_guard_rpc_null_inputs_fail_closed(database):
    psycopg2, dsn = database
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    valid_calls = {
        "lab_arena_restart_quiescence_v1": [GUARD, OWNER, 1],
        "lab_arena_authorize_restart_phase_v1": [
            GUARD, OWNER, 1, "gateway_destructive",
        ],
        "lab_arena_mark_restart_ready_v1": [GUARD, OWNER, 1, "gateway_ready"],
        "lab_arena_abort_restart_guard_v1": [GUARD, OWNER, 1, "test-abort"],
        "lab_arena_release_restart_guard_v1": [GUARD, OWNER, 1, "test-release"],
    }
    try:
        _acquire(admin)
        for function, valid in valid_calls.items():
            for index in range(len(valid)):
                values = list(valid)
                values[index] = None
                with pytest.raises(
                    psycopg2.Error, match="lab_arena_restart_guard_input_invalid"
                ):
                    _rpc(admin, function, *values)
        state = _rpc(admin, "lab_arena_restart_guard_state_v1")
        assert state["guard_present"] is True
        assert state["restart_phase"] == "draining"
    finally:
        admin.close()


def test_kind_incompatible_terminal_receipt_is_loss(database):
    psycopg2, dsn = database
    store = _store(psycopg2, dsn)
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    try:
        runners, _ = open_round(
            store, "arena-2099-01-06", participants=1, runners=1,
            prefix="drain-kind",
        )
        leased, _, _, _ = claim(store, "arena-2099-01-06", runners[0])
        _acquire(admin)
        with admin.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status = 'failed', "
                "terminal_cause = 'judge_error', "
                "result_doc = '{\"terminal_status\":\"judge_error\"}'::jsonb, "
                "lease_token_hash = NULL, lease_expires_at = NULL "
                "WHERE run_id = %s",
                (leased["run_id"],),
            )
        drain = _quiescence(admin)
        assert drain["preserved"] is False
        assert drain["reported_terminal_receipt_count"] == 0
        assert drain["lost_or_mutated_count"] == 1
    finally:
        store._transport.close()
        admin.close()


def test_claim_and_guard_acquisition_are_serialized_without_deadlock(database):
    psycopg2, dsn = database
    store = _store(psycopg2, dsn)
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    try:
        runners, _ = open_round(store, "arena-2099-01-03", participants=1, runners=1, prefix="drain-c")
        barrier = threading.Barrier(2)

        def run_claim():
            barrier.wait()
            return claim(store, "arena-2099-01-03", runners[0])[0]

        def run_acquire():
            connection = psycopg2.connect(**dsn)
            connection.autocommit = True
            try:
                barrier.wait()
                return _acquire(connection)
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            claimed = pool.submit(run_claim)
            guarded = pool.submit(run_acquire)
            claim_result = claimed.result(timeout=10)
            guard_result = guarded.result(timeout=10)
        assert claim_result["status"] in ("leased", "paused")
        assert guard_result["drain"]["captured_count"] == (1 if claim_result["status"] == "leased" else 0)
    finally:
        store._transport.close()
        admin.close()


@pytest.mark.parametrize("guard_wins", [True, False])
def test_old_claim_queued_before_first_round_read_is_rejected_or_captured(guard_wins):
    migrations = DEFAULT_MIGRATIONS[:DEFAULT_MIGRATIONS.index(LAB_ARENA_RESTART_CLAIM_DRAIN_MIGRATION)]
    database = database_with_lab_arena_migration(migrations)
    psycopg2, dsn = next(database)
    store = _store(psycopg2, dsn)
    blocker = psycopg2.connect(**dsn)
    blocker.autocommit = False
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    try:
        runners = [hotkey("drain-old-runner")]
        store.create_round(
            "arena-2099-01-04",
            round_config("arena-2099-01-04", runners),
        )
        parts = _historical_frozen_participants(
            store, "arena-2099-01-04", 1, prefix="drain-old"
        )
        commit_round(store, "arena-2099-01-04", parts)
        _install_instrumented_old_claim(admin)
        with blocker.cursor() as cursor:
            cursor.execute(
                "SELECT pg_catalog.pg_advisory_xact_lock("
                "pg_catalog.hashtextextended(%s, 0))",
                (OLD_CLAIM_BARRIER,),
            )

        def old_claim():
            claim_dsn = dict(dsn)
            claim_dsn["application_name"] = "old182_bootstrap_claim"
            local = _store(psycopg2, claim_dsn)
            try:
                return claim(local, "arena-2099-01-04", runners[0])
            finally:
                local._transport.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            queued_claim = pool.submit(old_claim)
            _wait_for_advisory_barrier(admin, "old182_bootstrap_claim")
            # The old function body is now executing but has not read a round
            # relation. The first install can take both relation locks NOWAIT.
            with admin.cursor() as cursor:
                cursor.execute(
                    (SCRIPTS / LAB_ARENA_RESTART_CLAIM_DRAIN_MIGRATION).read_text()
                )
            opened = store.open_stage(
                "arena-2099-01-04", 1, parts, stage_positions(1)
            )
            assert opened["status"] == "ok"

            if guard_wins:
                guarded = _acquire(admin)
                assert guarded["drain"]["captured_count"] == 0
                blocker.commit()
                with pytest.raises(ArenaStoreError, match="lab_arena_claims_paused"):
                    queued_claim.result(timeout=10)
                with admin.cursor() as cursor:
                    cursor.execute(
                        "SELECT status, attempt, lease_generation "
                        "FROM public.lab_arena_runs"
                    )
                    rows = cursor.fetchall()
                assert rows and all(row == ("pending", 1, 0) for row in rows)
            else:
                blocker.commit()
                leased, token, _, _ = queued_claim.result(timeout=10)
                assert leased["status"] == "leased"
                guarded = _acquire(admin)
                assert guarded["drain"]["captured_count"] == 1
                completed = complete(
                    store, leased["run_id"], hash_lease_token(token),
                    "accepted", output_ref="arena/old182-result.json",
                )
                assert completed["status"] == "accepted"
    finally:
        blocker.rollback()
        blocker.close()
        admin.close()
        store._transport.close()
        database.close()


def test_first_install_nowait_never_deadlocks_live_completion():
    migrations = DEFAULT_MIGRATIONS[:DEFAULT_MIGRATIONS.index(LAB_ARENA_RESTART_CLAIM_DRAIN_MIGRATION)]
    database = database_with_lab_arena_migration(migrations)
    psycopg2, dsn = next(database)
    store = _store(psycopg2, dsn)
    completion = psycopg2.connect(**dsn)
    completion.autocommit = False
    try:
        runners, _ = _historical_open_round(
            store, "arena-2099-01-05", participants=1, runners=1,
            prefix="drain-nowait",
        )
        leased, token, _, _ = claim(store, "arena-2099-01-05", runners[0])
        # This row lock is the first lock taken by migration-185 completion.
        # The migration must lose immediately rather than wait while holding
        # an incompatible rounds lock.
        with completion.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM public.lab_arena_runs WHERE run_id = %s FOR UPDATE",
                (leased["run_id"],),
            )

        def migrate():
            connection = psycopg2.connect(**dsn)
            connection.autocommit = True
            try:
                with connection.cursor() as cursor:
                    cursor.execute((SCRIPTS / LAB_ARENA_RESTART_CLAIM_DRAIN_MIGRATION).read_text())
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=1) as pool:
            migrating = pool.submit(migrate)
            with pytest.raises(psycopg2.errors.LockNotAvailable):
                migrating.result(timeout=3)
        # The completion transaction was not selected as a deadlock victim.
        completion.rollback()
        result = complete(
            store, leased["run_id"], hash_lease_token(token),
            "accepted", output_ref="arena/live-completion.json",
        )
        assert result["status"] == "accepted"
    finally:
        completion.rollback()
        completion.close()
        store._transport.close()
        database.close()


def test_restart_guard_privileges_ignore_permissive_default_function_grants():
    migrations = DEFAULT_MIGRATIONS[:DEFAULT_MIGRATIONS.index(LAB_ARENA_RESTART_CLAIM_DRAIN_MIGRATION)]
    database = database_with_lab_arena_migration(migrations)
    psycopg2, dsn = next(database)
    admin = psycopg2.connect(**dsn)
    admin.autocommit = True
    claim_signature = (
        "public.lab_arena_claim_assignment(text,text,integer,integer,text[],"
        "text,text,text,integer)"
    )
    roles = ("anon", "authenticated", "service_role", "lab_arena_service")

    def function_privilege(role, signature):
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT pg_catalog.has_function_privilege(%s, %s, 'EXECUTE')",
                (role, signature),
            )
            return cursor.fetchone()[0]

    try:
        before_claim = {
            role: function_privilege(role, claim_signature) for role in roles
        }
        with admin.cursor() as cursor:
            cursor.execute(
                "ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public "
                "GRANT EXECUTE ON FUNCTIONS TO anon, authenticated"
            )
            cursor.execute(
                (SCRIPTS / LAB_ARENA_RESTART_CLAIM_DRAIN_MIGRATION).read_text()
            )

        public_rpcs = (
            "public.lab_arena_restart_guard_state_v1()",
            "public.lab_arena_acquire_restart_guard_v1(text,text,bigint,integer,text,text,text)",
            "public.lab_arena_retarget_restart_guard_v1(text,text,bigint,text,text,text,integer,text)",
            "public.lab_arena_restart_quiescence_v1(text,text,bigint)",
            "public.lab_arena_authorize_restart_phase_v1(text,text,bigint,text)",
            "public.lab_arena_mark_restart_ready_v1(text,text,bigint,text)",
            "public.lab_arena_abort_restart_guard_v1(text,text,bigint,text)",
            "public.lab_arena_release_restart_guard_v1(text,text,bigint,text)",
        )
        internal_functions = (
            "public.lab_arena_restart_claim_gate_v1()",
            "public.lab_arena__restart_drain_state_v1()",
        )
        for signature in public_rpcs:
            assert function_privilege("anon", signature) is False
            assert function_privilege("authenticated", signature) is False
            assert function_privilege("service_role", signature) is True
            assert function_privilege("lab_arena_service", signature) is True
        for signature in internal_functions:
            for role in roles:
                assert function_privilege(role, signature) is False
        with admin.cursor() as cursor:
            for role in roles:
                cursor.execute(
                    "SELECT pg_catalog.has_table_privilege(%s, "
                    "'public.lab_arena_restart_claim_control', 'SELECT,UPDATE')",
                    (role,),
                )
                assert cursor.fetchone()[0] is False
        after_claim = {
            role: function_privilege(role, claim_signature) for role in roles
        }
        assert after_claim == before_claim
    finally:
        admin.close()
        database.close()
