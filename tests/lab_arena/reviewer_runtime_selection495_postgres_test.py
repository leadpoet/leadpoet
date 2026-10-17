"""Ordinary score claims atomically pin runtime deployment; accepted data stays fixed."""
from pathlib import Path

import pytest

from lab_arena import contracts, judgment_cache, scoring
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import judgment_cache_postgres_test as cache
from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_migration_postgres import (
    _commit_plan, _execute_everything, claim, commit_round, frozen_participants,
    hotkey, round_config,
)

SQL = Path(__file__).parents[2] / "scripts/495-lab-arena-reviewer-runtime-selection.sql"
REGISTER = "public.lab_arena_register_reviewer_runtime_v1(text,text,text,text)"
GRADING = "sha256:" + "a" * 64
GRADING_REF = "registry.example/lab/scorer@" + GRADING


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(DEFAULT_MIGRATIONS + (
        "212-lab-arena-accepted-judgment-cache.sql",
        "216-lab-arena-validator-participation.sql",
    ))


def prepare(store, label):
    rid = "arena-2026-10-16-runtime" + label
    runner = hotkey("runtime-" + label)
    configuration = round_config(rid, [runner])
    configuration["integrity_policy"] = "arena_integrity_v1"
    assert store.create_round(rid, configuration)["status"] == "created"
    parts = frozen_participants(store, rid, 2, prefix="runtime-" + label)
    commit_round(store, rid, parts)
    assert store.open_stage(rid, 1, parts, list(contracts.stage_positions(1)))["status"] == "ok"
    executions = _execute_everything(store, rid, runner)
    assert store.close_stage(rid, 1)["status"] == "closed"
    _commit_plan(store, rid, 1)
    items = cache._items(executions, parts, round_id=rid)
    assert store.open_scoring(rid, 1, items, integrity_cache=True)["assignments"] == 4
    return rid, runner


def accept(store, lease, token):
    source = store.get_run(lease["run_id"])
    evidence = judgment_cache.build_evidence_snapshot(
        output=scoring.build_scoring_output(lease["scored_run_id"], [
            {"company_name": "same", "final_score": 81.0},
        ]), cache_scope=source["judgment_scope_doc"],
        source_score_run_id=lease["run_id"], source_scored_run_id=lease["scored_run_id"],
        source_output_ref="arena/x/scores/" + lease["run_id"] + ".json",
        source_runner_hotkey=source["runner_hotkey"],
        runner_authority_exclusions=lease["runner_authority_exclusions"],
    )
    assert store.complete_attempt(
        run_id=lease["run_id"], lease_token_hash=hash_lease_token(token),
        result={"terminal_status": "accepted"}, terminal_cause="accepted",
        output_ref=evidence["source_output_ref"], judgment_evidence=evidence,
        judgment_evidence_hash=contracts.document_hash(evidence),
    )["status"] == "accepted"


def immutable_snapshot(cur):
    cur.execute("SELECT to_jsonb(r)-'reviewer_runtime_doc' FROM public.lab_arena_runs r "
                "WHERE status='accepted' ORDER BY run_id")
    runs = cur.fetchall()
    cur.execute("SELECT to_jsonb(c) FROM public.lab_arena_judgment_cache c ORDER BY cache_key")
    caches = cur.fetchall()
    cur.execute("SELECT round_id,configuration_doc FROM public.lab_arena_rounds ORDER BY round_id")
    return runs, caches, cur.fetchall()


def register(cur, marker):
    digest = "sha256:" + marker * 64
    reference = "registry.example/arena/runtime@" + digest
    cur.execute("SET ROLE lab_arena_service")
    cur.execute("SELECT public.lab_arena_register_reviewer_runtime_v1(%s,%s,%s,%s)",
                (GRADING, GRADING_REF, digest, reference))
    result = cur.fetchone()[0]
    cur.execute("RESET ROLE")
    return result


def test_migration_replay_exact_claim_replay_mapping_update_and_preserved_evidence(database):
    psycopg, dsn = database
    transport = PsycopgTransport(lambda: psycopg.connect(**dsn))
    store = ArenaStore(transport)
    try:
        old_round, old_runner = prepare(store, "old")
        accepted, token, _, _ = claim(store, old_round, old_runner, excluded=[old_runner])
        accept(store, accepted, token)
        old_lease, _, _, _ = claim(store, old_round, old_runner, excluded=[old_runner])
        new_round, new_runner = prepare(store, "new")
        with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
            before = immutable_snapshot(cur)
            # Apply under the same non-superuser ownership/default-grant shape
            # as hosted migrations, then repeat the exact migration.
            cur.execute("CREATE ROLE arena_runtime_migrator NOLOGIN NOSUPERUSER INHERIT")
            cur.execute("GRANT lab_arena_owner TO arena_runtime_migrator")
            cur.execute("GRANT USAGE,CREATE ON SCHEMA public TO arena_runtime_migrator WITH GRANT OPTION")
            cur.execute("ALTER DEFAULT PRIVILEGES FOR ROLE arena_runtime_migrator IN SCHEMA public "
                        "GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role")
            cur.execute("SET ROLE arena_runtime_migrator")
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
            cur.execute("RESET ROLE")
            assert immutable_snapshot(cur) == before
            for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
                cur.execute("SELECT has_table_privilege(%s,'public.lab_arena_reviewer_runtimes','SELECT'),"
                            "has_table_privilege(%s,'public.lab_arena_reviewer_runtimes','INSERT')", (role, role))
                assert cur.fetchone() == (False, False)
                cur.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, REGISTER))
                assert cur.fetchone()[0] == (role == "lab_arena_service")
            first = register(cur, "b")
            assert register(cur, "b") == first  # Replay does not move activation time.
            assert immutable_snapshot(cur) == before
        assert store.get_run(old_lease["run_id"])["reviewer_runtime_doc"] is None
        lease, lease_token, request_id, request_hash = claim(store, new_round, new_runner, excluded=[new_runner])
        persisted = store.get_run(lease["run_id"])
        binding = persisted["reviewer_runtime_doc"]
        assert binding["runtime_image_digest"] == "sha256:" + "b" * 64
        assert binding["grading_image_digest"] == GRADING
        assert binding["lease_generation"] == lease["lease_generation"]
        assert persisted["claim_response"]["reviewer_runtime_doc"] == binding
        replay_before = store.recover_claim_response(round_id=new_round, runner_hotkey=new_runner,
                                                     request_id=request_id, request_hash=request_hash)
        with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
            register(cur, "c")
            assert immutable_snapshot(cur) == before
        replay_after = store.recover_claim_response(round_id=new_round, runner_hotkey=new_runner,
                                                    request_id=request_id, request_hash=request_hash)
        assert replay_after == replay_before
        replayed_claim = store.claim_assignment(
            round_id=new_round, runner_hotkey=new_runner, declared_parallelism=1,
            slot_ceiling=8, excluded_miner_hotkeys=[new_runner], request_id=request_id,
            request_hash=request_hash, lease_token_hash=hash_lease_token(lease_token),
        )
        assert replayed_claim == replay_after
        assert store.get_run(lease["run_id"])["reviewer_runtime_doc"] == binding
        assert store.get_run(old_lease["run_id"])["reviewer_runtime_doc"] is None
        # A later ordinary score lease receives the new map. Old accepted
        # evidence and its key remain valid without aliases or a rekey.
        later, _, _, _ = claim(store, new_round, new_runner, parallelism=8, excluded=[new_runner])
        assert store.get_run(later["run_id"])["reviewer_runtime_doc"]["runtime_image_digest"] == "sha256:" + "c" * 64
        with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
            assert immutable_snapshot(cur) == before
            with pytest.raises(psycopg.Error, match="pin is immutable"):
                cur.execute("UPDATE public.lab_arena_runs SET reviewer_runtime_doc=NULL WHERE run_id=%s", (lease["run_id"],))
            conn.rollback()
            with pytest.raises(psycopg.Error, match="pin is immutable"):
                cur.execute("UPDATE public.lab_arena_runs SET reviewer_runtime_doc='{}' WHERE run_id=%s", (accepted["run_id"],))
            conn.rollback()
            assert immutable_snapshot(cur) == before
        execute_round = "arena-2026-10-17-runtimeexec"
        configuration = round_config(execute_round, [new_runner])
        assert store.create_round(execute_round, configuration)["status"] == "created"
        parts = frozen_participants(store, execute_round, 1, prefix="runtimeexec")
        commit_round(store, execute_round, parts)
        assert store.open_stage(execute_round, 1, parts, list(contracts.stage_positions(1)))["status"] == "ok"
        execute_lease, _, _, _ = claim(store, execute_round, new_runner)
        assert execute_lease["kind"] == "execute"
        assert store.get_run(execute_lease["run_id"])["reviewer_runtime_doc"] is None
        assert "reviewer_runtime_doc" not in execute_lease
    finally:
        transport.close()
