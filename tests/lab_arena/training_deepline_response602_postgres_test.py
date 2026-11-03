"""Actual private computed projection through PostgreSQL and PostgREST."""

import copy
from functools import partial
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
from uuid import uuid4

import httpx
import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PostgrestTransport, PsycopgTransport
from tests.lab_arena.test_lab_arena_postgrest_route import JWT_SECRET, mint_jwt
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey
from tests.lab_arena.training_recovered_deepline_response_test import recovered_values


MIGRATION = Path(__file__).resolve().parents[2] / "scripts/602-lab-arena-training-deepline-response.sql"
SIGNATURE = "public.lab_arena_training_deepline_response_v1(public.lab_arena_ledger)"
PG_IMAGE = "public.ecr.aws/supabase/postgres:17.6.1.158"
REST_IMAGE = "public.ecr.aws/supabase/postgrest:v14.16"


def _docker(*args):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-500:]
    return result.stdout.strip()


def _port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _cached_database(migrations, *, network, setup_sql=""):
    """Blank PostgreSQL17 from the cached image, with the standard SQL shim."""
    from tests.postgres_migration_harness import SCRIPTS
    pg = pytest.importorskip("psycopg2")
    name, port = "training602-pg-" + uuid4().hex[:10], _port()
    _docker("run", "--rm", "--detach", "--pull", "never", "--name", name,
            "--network", network, "--network-alias", "database", "--user", "postgres",
            "--publish", "127.0.0.1:%d:5432" % port, "--entrypoint", "sh", PG_IMAGE,
            "-c", "initdb -D /tmp/training602pg -A trust -U postgres >/dev/null "
            "&& printf 'host all all 0.0.0.0/0 trust\\n' >>/tmp/training602pg/pg_hba.conf "
            "&& exec postgres -D /tmp/training602pg -h 0.0.0.0")
    dsn = dict(host="127.0.0.1", port=port, user="postgres", dbname="postgres")
    connection = None
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                connection = pg.connect(**dsn)
                break
            except pg.OperationalError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        connection.autocommit = True
        with connection.cursor() as cur:
            cur.execute("CREATE SCHEMA extensions; CREATE EXTENSION pgcrypto WITH SCHEMA extensions; "
                        "CREATE ROLE anon NOLOGIN; CREATE ROLE authenticated NOLOGIN; CREATE ROLE service_role NOLOGIN")
            if setup_sql:
                cur.execute(setup_sql)
            for filename in migrations:
                cur.execute((SCRIPTS / filename).read_text())
        connection.close()
        connection = None
        yield pg, dsn
    finally:
        if connection is not None:
            connection.close()
        _docker("rm", "--force", name)


@pytest.fixture(scope="module")
def stack():
    """Cached PG17.6/PostgREST14.16; excludes nginx and opaque-key ingress.

    The production transport still generates the selection and parses the
    actual HTTP result. A test hook removes /rest/v1 and supplies a local JWT
    because this isolated contract stack has no ingress proxy.
    """
    from tests.lab_arena import lab_arena_pg_harness as arena_pg
    if shutil.which("docker") is None:
        pytest.skip("Docker is unavailable for the isolated cached-image contract")
    for image in (PG_IMAGE, REST_IMAGE):
        cached = subprocess.run(["docker", "image", "inspect", image],
                                capture_output=True, text=True, timeout=10)
        if cached.returncode != 0:
            pytest.skip("isolated contract image is not cached: " + image)
    network = "training602-net-" + uuid4().hex[:10]
    rest_name, port = "training602-rest-" + uuid4().hex[:10], _port()
    _docker("network", "create", network)
    database, connection, rest_started = None, None, False
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(arena_pg, "_database_with_migrations", partial(_cached_database, network=network))
            patch.delenv("LAB_ARENA_PG_LOCAL", raising=False)
            database = arena_pg.current_service_database()
            pg, dsn = next(database)
        connection = pg.connect(**dsn)
        connection.autocommit = True
        with connection.cursor() as cur:
            cur.execute("SHOW server_version_num")
            assert cur.fetchone()[0] == "170006"
            cur.execute("CREATE ROLE training602_auth LOGIN NOINHERIT; "
                        "GRANT anon,authenticated,service_role,lab_arena_service TO training602_auth")
        _docker("run", "--rm", "--detach", "--pull", "never", "--name", rest_name,
                "--network", network, "--publish", "127.0.0.1:%d:3000" % port,
                "--env", "PGRST_DB_URI=postgres://training602_auth@database:5432/postgres",
                "--env", "PGRST_DB_SCHEMAS=public", "--env", "PGRST_DB_ANON_ROLE=anon",
                "--env", "PGRST_JWT_SECRET=" + JWT_SECRET, REST_IMAGE)
        rest_started = True
        base_url = "http://127.0.0.1:%d" % port
        deadline = time.monotonic() + 30
        while True:
            try:
                response = httpx.get(base_url + "/", timeout=1)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                pytest.fail("isolated PostgREST did not become ready")
            time.sleep(0.1)
        yield {"base_url": base_url, "connection": connection}
    finally:
        if connection is not None:
            connection.close()
        if rest_started:
            _docker("rm", "--force", rest_name)
        if database is not None:
            database.close()
        _docker("network", "rm", network)


def make_transport(stack, role):
    def direct_path(request):
        assert request.url.path.startswith("/rest/v1/")
        request.url = request.url.copy_with(path=request.url.path[len("/rest/v1"):])
        request.headers["Authorization"] = "Bearer " + mint_jwt(role)

    return PostgrestTransport(stack["base_url"], service_key="sb_secret_Local_contract602",
                              http_client=httpx.Client(timeout=10, trust_env=False,
                                                       event_hooks={"request": [direct_path]}))


def _security_state(cur):
    cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
    schema = cur.fetchone()
    cur.execute("SELECT c.relname,c.relacl::text,c.relrowsecurity,c.relforcerowsecurity "
                "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname='public' ORDER BY c.relname")
    tables = cur.fetchall()
    cur.execute("SELECT schemaname,tablename,policyname,roles,cmd,qual,with_check "
                "FROM pg_policies WHERE schemaname='public' ORDER BY tablename,policyname")
    return schema, tables, cur.fetchall()


@pytest.fixture(scope="module")
def projected_stack(stack):
    with stack["connection"].cursor() as cur:
        before = _security_state(cur)
        cur.execute(MIGRATION.read_text())
        assert _security_state(cur) == before
        cur.execute(MIGRATION.read_text())
        assert _security_state(cur) == before
        # Minimal retained rows in this disposable database. No production
        # writer or provider is invoked; this test exercises the read boundary.
        cur.execute("INSERT INTO public.lab_arena_rounds(round_id,configuration_doc) "
                    "VALUES ('arena-2026-11-02-reader602','{}')")
        cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey) "
                    "VALUES ('reader602','arena-2026-11-02-reader602',%s)", (hotkey("reader602"),))
    return stack


def _seed(stack, index, *, recovery_patch=None, head_patch=None, no_recovery=False, kind="settlement"):
    rows = copy.deepcopy(recovered_values()["ledger_rows"])
    identity = "sha256:" + format(index, "064x")
    run_id = "reader602:%d" % index
    start = 6020000 + index * 10
    recovered = rows[-1].pop("training_recovered_response")
    for offset, row in enumerate(rows, 1):
        row.update(entry_id=start + offset, call_identity=identity, run_id=run_id,
                   round_id="arena-2026-11-02-reader602", submission_id="reader602",
                   miner_hotkey=hotkey("reader602"), stage=1, funding_source="host")
        row["entry_doc"]["deepline_execution_key"] = "arena:" + identity[7:]
    rows[-1]["entry_kind"] = kind
    if head_patch:
        rows[-1].update(head_patch)
    recovered.update(call_identity=identity, run_id=run_id,
                     reservation_entry_id=start + 1, terminal_entry_id=start + 3)
    if recovery_patch:
        recovered.update(recovery_patch)
    with stack["connection"].cursor() as cur:
        cur.execute("INSERT INTO public.lab_arena_runs "
                    "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt) "
                    "VALUES (%s,%s,'arena-2026-11-02-reader602','reader602',%s,1,0,1)",
                    (run_id, run_id, hotkey("reader602")))
        if recovered["run_id"] != run_id:
            cur.execute("INSERT INTO public.lab_arena_runs "
                        "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt) "
                        "VALUES (%s,%s,'arena-2026-11-02-reader602','reader602',%s,1,0,1)",
                        (recovered["run_id"], recovered["run_id"], hotkey("reader602")))
        for row in rows:
            cur.execute("INSERT INTO public.lab_arena_ledger "
                        "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,"
                        "provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)",
                        tuple(row.get(key) for key in
                              ("entry_id", "entry_kind", "miner_hotkey", "round_id", "submission_id", "run_id",
                               "stage", "call_identity", "provider", "operation_id", "funding_source", "amount_microusd"))
                        + (json.dumps(row["entry_doc"]), json.dumps(row.get("terminal_response"))))
        if not no_recovery:
            cur.execute("INSERT INTO public.lab_arena_deepline_call_responses "
                        "(call_identity,reservation_entry_id,run_id,request_id,terminal_response,created_at) "
                        "VALUES (%s,%s,%s,%s,%s::jsonb,%s)",
                        (recovered["call_identity"], recovered["reservation_entry_id"], recovered["run_id"],
                         recovered["request_id"], json.dumps(recovered["terminal_response"]), recovered["created_at"]))
    return rows, recovered


def _rows(transport, run_id):
    deadline = time.monotonic() + 10
    while True:
        try:
            return ArenaStore(transport).list_ledger(run_id=run_id, training_responses=True)
        except ArenaStoreError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def test_actual_sql_and_postgrest_keep_original_bill_and_same_recovery(projected_stack):
    stack = projected_stack
    rows, recovered = _seed(stack, 1)
    psycopg2 = pytest.importorskip("psycopg2")
    transport = PsycopgTransport(lambda: psycopg2.connect(stack["connection"].dsn))
    rest = make_transport(stack, "lab_arena_service")
    try:
        direct = _rows(transport, rows[0]["run_id"])
        remote = _rows(rest, rows[0]["run_id"])
        assert direct == remote
        assert all(row["training_recovered_response"] is None for row in direct[:-1])
        ref = direct[-1]["training_recovered_response"]
        assert ref["table"] == "lab_arena_deepline_call_responses"
        assert ref["call_identity"] == recovered["call_identity"]
        assert ref["terminal_response"] == recovered["terminal_response"]
        assert ref["terminal_entry_id"] == rows[-1]["entry_id"]
        assert direct[-1]["terminal_response"] == rows[-1]["terminal_response"]
        assert direct[-1]["terminal_response"]["status"] == 502
        assert direct[-1]["amount_microusd"] == 42
        ordinary = ArenaStore(rest).list_ledger(run_id=rows[0]["run_id"])
        assert "training_recovered_response" not in ordinary[-1]
        with stack["connection"].cursor() as cur:
            cur.execute("SELECT public.lab_arena__call_state_view(l,r)->'terminal_response' "
                        "FROM public.lab_arena_ledger l JOIN public.lab_arena_runs r USING(run_id) "
                        "WHERE l.entry_id=%s", (rows[-1]["entry_id"],))
            assert cur.fetchone()[0] == ref["terminal_response"]
    finally:
        transport.close()
        rest.close()


@pytest.mark.parametrize("patch", ["no_recovery", "wrong_request", "wrong_reservation", "wrong_tool", "old_head", "uncertain", "wrong_run"])
def test_saved_identity_and_current_head_controls(projected_stack, patch):
    stack = projected_stack
    index = 10 + ["no_recovery", "wrong_request", "wrong_reservation", "wrong_tool", "old_head", "uncertain", "wrong_run"].index(patch)
    options = {}
    if patch == "no_recovery":
        options["no_recovery"] = True
    elif patch == "wrong_request":
        options["recovery_patch"] = {"request_id": "foreign-native"}
    elif patch == "wrong_reservation":
        options["recovery_patch"] = {"reservation_entry_id": 6020000 + index * 10 + 2}
    elif patch == "wrong_run":
        options["recovery_patch"] = {"run_id": "reader602:foreign"}
    elif patch == "wrong_tool":
        terminal = copy.deepcopy(recovered_values()["ledger_rows"][-1]["training_recovered_response"]["terminal_response"])
        terminal["provider_cost"]["operation"] = "exa.search"
        options["recovery_patch"] = {"terminal_response": terminal}
    elif patch == "uncertain":
        options["kind"] = "uncertain"
    rows, recovered = _seed(stack, index, **options)
    if patch == "old_head":
        with stack["connection"].cursor() as cur:
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,"
                        "stage,call_identity,provider,operation_id,funding_source,amount_microusd) "
                        "SELECT %s,'recovery',miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,"
                        "operation_id,funding_source,0 FROM public.lab_arena_ledger WHERE entry_id=%s",
                        (rows[-1]["entry_id"] + 1, rows[-1]["entry_id"]))
    rest = make_transport(stack, "lab_arena_service")
    try:
        projected = _rows(rest, rows[0]["run_id"])
        if patch == "uncertain":
            assert projected[-1]["training_recovered_response"]["terminal_response"] == recovered["terminal_response"]
            assert projected[-1]["entry_kind"] == "uncertain" and projected[-1]["amount_microusd"] == 42
        else:
            assert all(row["training_recovered_response"] is None for row in projected)
    finally:
        rest.close()


def test_malicious_composite_cannot_override_authoritative_row(projected_stack):
    stack = projected_stack
    rows, _ = _seed(stack, 30)
    with stack["connection"].cursor() as cur:
        cur.execute("SELECT to_jsonb(l) FROM public.lab_arena_ledger l WHERE entry_id=%s", (rows[-1]["entry_id"],))
        authoritative = cur.fetchone()[0]
        for field, value in (("run_id", "foreign-run"), ("call_identity", "sha256:" + "f" * 64),
                             ("entry_doc", {}), ("terminal_response", {}), ("entry_id", 999999999)):
            forged = {**authoritative, field: value}
            cur.execute("SET ROLE lab_arena_service")
            try:
                cur.execute("SELECT public.lab_arena_training_deepline_response_v1("
                            "jsonb_populate_record(NULL::public.lab_arena_ledger,%s::jsonb))", (json.dumps(forged),))
                assert cur.fetchone()[0] is None
            finally:
                cur.execute("RESET ROLE")
            response = httpx.post(stack["base_url"] + "/rpc/lab_arena_training_deepline_response_v1",
                                  headers={"Authorization": "Bearer " + mint_jwt("lab_arena_service")},
                                  json={"p_ledger": forged})
            assert response.status_code == 200 and response.json() is None


@pytest.mark.parametrize("change", ["body", "overload", "acl", "strict"])
def test_migration_refuses_unknown_existing_function(projected_stack, change):
    connection = projected_stack["connection"]
    with connection.cursor() as cur:
        cur.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
        definition = cur.fetchone()[0]
        before = _security_state(cur)
        try:
            if change == "body":
                cur.execute("CREATE OR REPLACE FUNCTION public.lab_arena_training_deepline_response_v1("
                            "p_ledger public.lab_arena_ledger) RETURNS jsonb LANGUAGE plpgsql STABLE "
                            "SECURITY DEFINER SET search_path=pg_catalog,public AS $$ BEGIN RETURN '{}'::jsonb; END $$")
            elif change == "overload":
                cur.execute("CREATE FUNCTION public.lab_arena_training_deepline_response_v1(text) "
                            "RETURNS jsonb LANGUAGE sql AS $$ SELECT '{}'::jsonb $$")
            elif change == "acl":
                cur.execute("GRANT EXECUTE ON FUNCTION " + SIGNATURE + " TO anon")
            else:
                cur.execute("ALTER FUNCTION " + SIGNATURE + " STRICT")
            with pytest.raises(Exception, match="lab_arena_training_response_preimage_invalid"):
                cur.execute(MIGRATION.read_text())
            cur.execute("ROLLBACK")
            assert _security_state(cur) == before
        finally:
            cur.execute(definition)
            cur.execute("DROP FUNCTION IF EXISTS public.lab_arena_training_deepline_response_v1(text)")
            cur.execute("REVOKE EXECUTE ON FUNCTION " + SIGNATURE + " FROM anon")


def test_transaction_failure_restores_temporary_schema_create_grant(projected_stack):
    with projected_stack["connection"].cursor() as cur:
        before = _security_state(cur)
        cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
        assert cur.fetchone() == (False,)
        # Force a disposable failure immediately after the temporary grant,
        # before the normal restore block. The exact migration otherwise runs.
        failing = MIGRATION.read_text().replace(
            "  ALTER FUNCTION public.lab_arena_training_deepline_response_v1(",
            "  RAISE EXCEPTION 'injected602failure';\n"
            "  ALTER FUNCTION public.lab_arena_training_deepline_response_v1(", 1)
        with pytest.raises(Exception, match="injected602failure"):
            cur.execute(failing)
        cur.execute("ROLLBACK")
        assert _security_state(cur) == before
        cur.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
        assert cur.fetchone() == (False,)


def test_projection_and_raw_table_privileges_are_closed(projected_stack):
    stack = projected_stack
    with stack["connection"].cursor() as cur:
        cur.execute("SELECT prosecdef,provolatile,proconfig,pg_get_userbyid(proowner) "
                    "FROM pg_proc WHERE oid=%s::regprocedure", (SIGNATURE,))
        assert cur.fetchone() == (True, "s", ["search_path=pg_catalog, public"], "lab_arena_owner")
        for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
            cur.execute("SELECT has_function_privilege(%s,%s,'EXECUTE'),"
                        "has_table_privilege(%s,'public.lab_arena_deepline_call_responses','SELECT')",
                        (role, SIGNATURE, role))
            assert cur.fetchone() == (role == "lab_arena_service", False)
        cur.execute("SELECT EXISTS(SELECT 1 FROM pg_proc p,aclexplode(p.proacl) a "
                    "WHERE p.oid=%s::regprocedure AND a.grantee=0 AND a.privilege_type='EXECUTE')", (SIGNATURE,))
        assert cur.fetchone() == (False,)
    for role in ("lab_arena_service", "service_role", "anon", "authenticated"):
        headers = {"Authorization": "Bearer " + mint_jwt(role)}
        raw = httpx.get(stack["base_url"] + "/lab_arena_deepline_call_responses", headers=headers)
        assert raw.status_code in (401, 403, 404)
        if role != "lab_arena_service":
            rpc = httpx.post(stack["base_url"] + "/rpc/lab_arena_training_deepline_response_v1",
                             headers=headers, json={"p_ledger": {"entry_id": 6020013}})
            assert rpc.status_code in (401, 403, 404)
