"""Promotion RPC grants admit only the canonical service, without body changes."""
import json
from pathlib import Path

import pytest

from tests.lab_arena import complete_baseline_final_authority_postgres_test as finality
from tests.lab_arena.test_lab_arena_reward_migration_postgres import MINER_A


SQL = Path(__file__).resolve().parents[2] / "scripts/583-lab-arena-promotion-rpc-privileges.sql"
SIGNATURES = ("public.lab_arena_prepare_promotion(text,jsonb)", "public.lab_arena_complete_promotion(text,jsonb)")
DENIED = ("anon", "authenticated", "service_role")
PLAN = {"commit": "1" * 40, "main_before": "2" * 40, "lab_before": "3" * 40, "timestamp": "2026-10-13T00:00:00Z"}


@pytest.fixture(scope="module")
def database():
    yield from finality.database.__wrapped__()


@pytest.fixture()
def restored(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for signature in SIGNATURES:
            cur.execute("REVOKE ALL ON FUNCTION " + signature + " FROM PUBLIC,anon,authenticated,service_role,lab_arena_service")
            cur.execute("GRANT EXECUTE ON FUNCTION " + signature + " TO anon,authenticated,service_role,lab_arena_service")
        cur.execute("GRANT USAGE ON SCHEMA public TO anon,authenticated,service_role,lab_arena_service")
    return database


def metadata(cur):
    cur.execute("SELECT p.oid::regprocedure::text,to_jsonb(p) FROM pg_proc p WHERE p.pronamespace='public'::regnamespace ORDER BY p.oid")
    return dict(cur.fetchall())


def guards(cur):
    cur.execute("SELECT tgname,tgenabled,pg_get_triggerdef(oid) FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass ORDER BY tgname")
    return cur.fetchall()


def apply(conn):
    with conn.cursor() as cur:
        cur.execute(SQL.read_text())


def call(cur, name, rid):
    cur.execute("SELECT public." + name + "(%s,%s::jsonb)", (rid, json.dumps(PLAN)))
    return cur.fetchone()[0]


@pytest.mark.parametrize("public_grant", [False, True])
def test_exact_acl_repair_is_idempotent_and_preserves_every_function_and_guard(restored, public_grant):
    with restored[0].connect(**restored[1]) as conn, conn.cursor() as cur:
        if public_grant:
            for signature in SIGNATURES:
                cur.execute("GRANT EXECUTE ON FUNCTION " + signature + " TO PUBLIC")
        before, trigger_before = metadata(cur), guards(cur)
        cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_rounds r ORDER BY round_id")
        rows = cur.fetchall()
        for _ in range(2):
            apply(conn)
            after = metadata(cur)
            assert set(after) == set(before)
            changed = {signature for signature in before if before[signature] != after[signature]}
            assert changed == {signature.removeprefix("public.") for signature in SIGNATURES}
            for signature in changed:
                old, new = before[signature].copy(), after[signature].copy()
                assert old.pop("proacl") != new.pop("proacl")
                assert old == new
            assert guards(cur) == trigger_before
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_rounds r ORDER BY round_id")
            assert cur.fetchall() == rows
            for signature in SIGNATURES:
                for role in (*DENIED, "lab_arena_service", "lab_arena_owner"):
                    cur.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, signature))
                    assert cur.fetchone() == (role not in DENIED,)


def test_real_denied_roles_cannot_promote_and_canonical_service_keeps_valid_flow(restored):
    # The established baseline fixture creates accepted baseline execution and
    # judgment rows. Only fixture construction bypasses triggers; all promotion
    # calls below run with every current round guard enabled.
    fixture = finality.connection.__wrapped__(restored)
    conn = next(fixture)
    try:
        with conn.cursor() as cur:
            rid = "arena-2026-10-13-finalgate"
            cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king) VALUES ('final-miner',%s,%s,'frozen',false)", (rid, MINER_A))
            publication = {"king_decision": {"outcome": "crowned", "winner_submission_id": "final-miner", "king_submission_id": "final-miner", "king_hotkey": MINER_A}}
            cur.execute("UPDATE public.lab_arena_rounds SET status='published',publication_doc=%s::jsonb WHERE round_id=%s", (json.dumps(publication), rid))
            cur.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
            assert finality.ready(cur)
            for role in DENIED:
                cur.execute("SAVEPOINT before_untrusted_call")
                cur.execute("SET LOCAL ROLE " + role)
                assert call(cur, "lab_arena_prepare_promotion", rid)["status"] == "prepared"
                assert call(cur, "lab_arena_complete_promotion", rid)["status"] == "promoted"
                cur.execute("ROLLBACK TO SAVEPOINT before_untrusted_call")
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s", (rid,))
            before = cur.fetchone()[0]
            apply(conn)
            cur.execute("BEGIN")
            for role in DENIED:
                for name in ("lab_arena_prepare_promotion", "lab_arena_complete_promotion"):
                    cur.execute("SAVEPOINT denied_call")
                    cur.execute("SET LOCAL ROLE " + role)
                    with pytest.raises(restored[0].errors.InsufficientPrivilege, match="permission denied for function") as denied:
                        call(cur, name, rid)
                    assert denied.value.pgcode == "42501"
                    cur.execute("ROLLBACK TO SAVEPOINT denied_call")
            cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s", (rid,))
            assert cur.fetchone()[0] == before
            # The ACL change must not bypass the baseline or publication gates
            # for the retained canonical caller.
            for change, expected in (
                ("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='provider_error' WHERE run_id='execute-0'", "lab_arena_final_baseline_incomplete"),
                ("UPDATE public.lab_arena_rounds SET status='scored' WHERE round_id='arena-2026-10-13-finalgate'", "lab_arena_promotion_not_required"),
                ("UPDATE public.lab_arena_rounds SET publication_doc=jsonb_set(publication_doc,'{king_decision,winner_submission_id}','\"missing\"') WHERE round_id='arena-2026-10-13-finalgate'", "lab_arena_promotion_winner_invalid"),
            ):
                cur.execute("SAVEPOINT preserved_guard")
                cur.execute("SET LOCAL session_replication_role=replica")
                cur.execute(change)
                cur.execute("SET LOCAL session_replication_role=origin")
                cur.execute("SET LOCAL ROLE lab_arena_service")
                with pytest.raises(restored[0].Error, match=expected):
                    call(cur, "lab_arena_prepare_promotion", rid)
                cur.execute("ROLLBACK TO SAVEPOINT preserved_guard")
            cur.execute("SET LOCAL ROLE lab_arena_service")
            assert call(cur, "lab_arena_prepare_promotion", rid)["status"] == "prepared"
            assert call(cur, "lab_arena_prepare_promotion", rid)["status"] == "existing"
            assert call(cur, "lab_arena_complete_promotion", rid)["status"] == "promoted"
            assert call(cur, "lab_arena_complete_promotion", rid)["status"] == "existing"
            cur.execute("RESET ROLE")
            cur.execute("SELECT publication_doc,reward_basis_doc,reward_activated_at FROM public.lab_arena_rounds WHERE round_id=%s", (rid,))
            assert cur.fetchone() == (publication, None, None)
    finally:
        fixture.close()


@pytest.mark.parametrize("fault", ["owner", "invoker", "path", "body", "missing", "inherited"])
def test_unreviewed_authority_or_inherited_access_rolls_back_atomically(restored, fault):
    with restored[0].connect(**restored[1]) as conn, conn.cursor() as cur:
        before = metadata(cur)
        signature = SIGNATURES[1]  # The first function's ACL change must roll back too.
        if fault == "owner":
            cur.execute("ALTER FUNCTION " + signature + " OWNER TO postgres")
        elif fault == "invoker":
            cur.execute("ALTER FUNCTION " + signature + " SECURITY INVOKER")
        elif fault == "path":
            cur.execute("ALTER FUNCTION " + signature + " SET search_path=public,pg_catalog")
        elif fault == "body":
            cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            cur.execute(cur.fetchone()[0].replace("DECLARE", "-- unreviewed drift\nDECLARE", 1))
        elif fault == "missing":
            cur.execute("DROP FUNCTION " + signature)
        else:
            cur.execute("GRANT lab_arena_owner TO anon")
        with pytest.raises(restored[0].Error, match="promotion583 (authority|privilege readback) differs"):
            apply(conn)
        conn.rollback()
        assert metadata(cur) == before


def test_owner_member_can_apply_without_superuser(restored):
    with restored[0].connect(**restored[1]) as conn, conn.cursor() as cur:
        cur.execute("CREATE ROLE promotion583_migrator NOLOGIN NOSUPERUSER INHERIT")
        cur.execute("GRANT lab_arena_owner TO promotion583_migrator")
        conn.commit()
        cur.execute("SET ROLE promotion583_migrator")
        cur.execute("SELECT rolsuper FROM pg_roles WHERE rolname=current_user")
        assert cur.fetchone() == (False,)
        apply(conn)
        cur.execute("RESET ROLE")
        for signature in SIGNATURES:
            cur.execute("SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE'),has_function_privilege('anon',%s,'EXECUTE')", (signature, signature))
            assert cur.fetchone() == (True, False)
