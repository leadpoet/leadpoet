"""Production opaque keys and /rest/v1 through nginx/PostgREST14.5/PostgreSQL17.

No positive request injects a backend JWT or rewrites its path in Python.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

import httpx
import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, ArenaStoreError, PostgrestTransport, TABLES

DOCKER = shutil.which("docker")
POSTGREST_IMAGE = "postgrest/postgrest:v14.5@sha256:b574528fe109c8343c1247155734d03df8c34b462f342dca0ccc20244fc36ef9"
POSTGRES_IMAGE = "postgres:17.9@sha256:2a0d0fe14825b0939f78a8cad5cd4e6aa68bf94d0e5dd96e24b6d23af4315545"
JWT_SECRET = "lab-arena-local-jwt-secret-" + "x" * 40

pytestmark = pytest.mark.skipif(DOCKER is None, reason="Docker is unavailable")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def mint_jwt(role: str, *, secret: str = JWT_SECRET, expires_in: int = 3600) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64(json.dumps({"role": role, "iss": "leadpoet-lab-arena-test", "exp": int(time.time()) + expires_in}, separators=(",", ":")).encode())
    signature = _b64(hmac.new(secret.encode(), ("%s.%s" % (header, payload)).encode(), hashlib.sha256).digest())
    return "%s.%s.%s" % (header, payload, signature)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([DOCKER, *args], capture_output=True, text=True, timeout=timeout, check=False)


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """Current SQL on PostgreSQL17, real opaque-key nginx, PostgREST14.5."""
    from functools import partial
    import os
    from scripts import aws_subnet_api as api
    from tests import postgres_migration_harness as pg_harness
    from tests.lab_arena import lab_arena_pg_harness as arena_pg
    psycopg2 = pytest.importorskip("psycopg2")
    suffix = uuid4().hex[:10]
    network = "lab-arena-api-net-" + suffix
    rest_name = "lab-arena-api-rest-" + suffix
    proxy_name = "lab-arena-api-proxy-" + suffix
    started = []
    database = None
    connection = None
    with pytest.MonkeyPatch.context() as patch:
        try:
            assert _docker("network", "create", network).returncode == 0
            started.append(("network", network))
            factory = partial(pg_harness._database_with_migrations,
                              image=POSTGRES_IMAGE, network=network)
            patch.setattr(arena_pg, "_database_with_migrations", factory)
            patch.delenv("LAB_ARENA_PG_LOCAL", raising=False)
            database = arena_pg.current_service_database()
            pg, dsn = next(database)
            connection = pg.connect(**dsn)
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute("SHOW server_version_num")
                assert cursor.fetchone()[0] == "170009"
                # The schema fixture seeds historical repair controls. HTTP
                # behavior needs fresh test rounds, not those archived rows.
                cursor.execute("SET session_replication_role=replica")
                cursor.execute("TRUNCATE public.lab_arena_rounds CASCADE")
                cursor.execute("SET session_replication_role=origin")
                cursor.execute("""
                    CREATE ROLE authenticator LOGIN PASSWORD 'authenticator-pw'
                      NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
                    GRANT anon, service_role, lab_arena_service TO authenticator
                      WITH INHERIT FALSE, SET TRUE;
                    -- Match source request defaults. PostgREST applies each
                    -- request role's overrides above the login's fallback.
                    ALTER ROLE authenticator IN DATABASE postgres SET statement_timeout='8s';
                    ALTER ROLE authenticator IN DATABASE postgres SET lock_timeout='8s';
                    ALTER ROLE authenticator IN DATABASE postgres SET idle_in_transaction_session_timeout='0';
                    ALTER ROLE anon SET statement_timeout='10s';
                    CREATE TABLE public.lab_arena_test_other_table (id INT);
                    CREATE FUNCTION public.api_test_identity() RETURNS jsonb
                    LANGUAGE sql STABLE SECURITY INVOKER AS $$
                      SELECT jsonb_build_object('role',current_user,
                        'claims',current_setting('request.jwt.claims',true)::jsonb,
                        'statement_timeout',current_setting('statement_timeout'),
                        'lock_timeout',current_setting('lock_timeout'),
                        'idle_timeout',current_setting('idle_in_transaction_session_timeout'))
                    $$;
                    REVOKE ALL ON FUNCTION public.api_test_identity() FROM PUBLIC;
                    GRANT EXECUTE ON FUNCTION public.api_test_identity()
                      TO anon,service_role,lab_arena_service;
                    CREATE FUNCTION public.api_test_row_limit() RETURNS SETOF integer
                    LANGUAGE sql STABLE SECURITY INVOKER AS $$
                      SELECT generate_series(1,1001)
                    $$;
                    REVOKE ALL ON FUNCTION public.api_test_row_limit() FROM PUBLIC;
                    GRANT EXECUTE ON FUNCTION public.api_test_row_limit()
                      TO anon,service_role,lab_arena_service;
                    CREATE FUNCTION public.api_test_payload(p_body text) RETURNS integer
                    LANGUAGE sql IMMUTABLE SECURITY INVOKER AS $$ SELECT octet_length(p_body) $$;
                    REVOKE ALL ON FUNCTION public.api_test_payload(text) FROM PUBLIC;
                    GRANT EXECUTE ON FUNCTION public.api_test_payload(text) TO lab_arena_service;
                    CREATE FUNCTION public.api_test_sqlstate() RETURNS void
                    LANGUAGE plpgsql SECURITY INVOKER AS $$ BEGIN
                      RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='api_contract_test';
                    END $$;
                    REVOKE ALL ON FUNCTION public.api_test_sqlstate() FROM PUBLIC;
                    GRANT EXECUTE ON FUNCTION public.api_test_sqlstate() TO lab_arena_service;
                    -- Retained legacy PUBLIC grants must not widen the reader's
                    -- opaque-key API surface, even for SECURITY DEFINER RPCs.
                    CREATE FUNCTION public.api_test_legacy_public() RETURNS text
                    LANGUAGE sql STABLE SECURITY DEFINER
                    SET search_path=pg_catalog AS $$ SELECT 'legacy-public'::text $$;
                    GRANT EXECUTE ON FUNCTION public.api_test_legacy_public() TO PUBLIC;
                    CREATE TABLE public.api_test_public_table (id integer);
                    INSERT INTO public.api_test_public_table VALUES (1);
                    GRANT SELECT ON public.api_test_public_table TO PUBLIC;
                    CREATE SCHEMA IF NOT EXISTS auth;
                    CREATE FUNCTION auth.role() RETURNS text LANGUAGE sql STABLE AS $$
                      SELECT coalesce(nullif(current_setting('request.jwt.claim.role',true),''),
                        nullif(current_setting('request.jwt.claims',true),'')::jsonb->>'role')
                    $$;
                    GRANT USAGE ON SCHEMA auth TO service_role;
                    GRANT SELECT,INSERT,UPDATE ON public.qualification_private_icp_sets TO service_role;
                    CREATE POLICY api_test_service_claim ON public.qualification_private_icp_sets
                      TO service_role USING (auth.role()='service_role')
                      WITH CHECK (auth.role()='service_role');
                """)
                cursor.execute((Path(__file__).resolve().parents[2] /
                    "scripts/593-lab-arena-capture-summary-reader.sql").read_text())
                for migration in ("595-lab-arena-capture-called-icps.sql",
                                  "598-lab-arena-capture-summary-v2.sql"):
                    cursor.execute((Path(__file__).resolve().parents[2] / "scripts" / migration).read_text())
            directory = tmp_path_factory.mktemp("private-api-" + suffix)
            directory.chmod(0o700)
            values = {
                "arena_key": "sb_secret_Arena_Aa1_" + suffix * 4,
                "gateway_key": "sb_secret_Gateway_Bb2_" + suffix * 4,
                "public_key": "sb_publishable_Public_Cc3_" + suffix * 4,
                "capture_reader_key": "sb_secret_Capture_Dd4_" + suffix * 4,
                "jwt_secret": JWT_SECRET,
                "db_uri": "postgres://authenticator:authenticator-pw@database:5432/postgres",
            }
            nginx = directory / "nginx.conf"
            rest = directory / "postgrest.conf"
            nginx.write_text(api.render_nginx(values, upstream=rest_name + ":3000", listen="0.0.0.0:8081"))
            rest.write_text(api.render_postgrest(values, host="0.0.0.0"))
            nginx.chmod(0o600)
            rest.chmod(0o600)
            uid = str(os.getuid())
            common = ("--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
                      "--user", uid, "--log-driver", "none", "--network", network)
            assert _docker("run", "--rm", "--detach", "--name", rest_name, *common,
                "--volume", str(rest) + ":/run/postgrest.conf:ro",
                api.POSTGREST_IMAGE, "postgrest", "/run/postgrest.conf").returncode == 0
            started.append(("container", rest_name))
            proxy_port = _free_port()
            assert _docker("run", "--rm", "--detach", "--name", proxy_name, *common,
                "--publish", "127.0.0.1:%d:8081" % proxy_port,
                "--tmpfs", "/tmp:uid=" + uid + ",mode=0700",
                "--volume", str(nginx) + ":/run/nginx.conf:ro", "--entrypoint", "nginx",
                api.NGINX_IMAGE, "-c", "/run/nginx.conf", "-g", "daemon off;").returncode == 0
            started.append(("container", proxy_name))
            base_url = "http://127.0.0.1:%d" % proxy_port
            with httpx.Client(timeout=10, trust_env=False) as client:
                deadline = time.monotonic() + 60
                last_status = None
                while time.monotonic() < deadline:
                    try:
                        response = client.post(base_url + "/rest/v1/rpc/api_test_identity",
                            headers={"apikey": values["arena_key"]}, json={})
                        last_status = response.status_code
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.25)
                else:
                    states = [_docker("inspect", "--format", "{{.State.Status}}:{{.State.ExitCode}}", name).stdout.strip()
                              for name in (rest_name, proxy_name)]
                    pytest.fail("private compatibility pair did not become ready: HTTP %s, states %s" %
                                (last_status, states))
            yield {"base_url": base_url, "connection": connection, "values": values,
                   "proxy_name": proxy_name, "rest_name": rest_name}
        finally:
            if connection is not None:
                connection.close()
            for kind, name in reversed(started):
                if kind == "container":
                    _docker("rm", "--force", name, timeout=60)
            if database is not None:
                database.close()
            for kind, name in reversed(started):
                if kind == "network":
                    _docker("network", "rm", name, timeout=60)


def make_transport(stack, role: str, *, jwt_secret: str = JWT_SECRET) -> PostgrestTransport:
    # Every positive request uses exactly the production apikey-only transport.
    names = {"lab_arena_service": "arena_key", "service_role": "gateway_key", "anon": "public_key"}
    key = stack["values"].get(names.get(role, ""), "sb_secret_Unknown_" + "z" * 32)
    transport = PostgrestTransport(stack["base_url"], service_key=stack["values"]["arena_key"])
    transport._headers["apikey"] = key
    if jwt_secret != JWT_SECRET:
        transport._headers["Authorization"] = "Bearer " + mint_jwt(role, secret=jwt_secret)
    return transport


def test_service_role_authorization_reaches_whoami_and_functions_through_postgrest(stack):
    transport = make_transport(stack, "lab_arena_service")
    store = ArenaStore(transport)
    identity = store.require_service_role()
    assert identity["current_user"] == "lab_arena_service" and identity["jwt_role"] == "lab_arena_service"
    assert identity["rolsuper"] is False and identity["rolbypassrls"] is False and identity["session_user"] == "authenticator"
    assert store.list_rounds() == []
    with pytest.raises(ArenaStoreError):
        transport.select("lab_arena_test_other_table")  # not an Arena table: refused client-side
    response = transport._client.get(stack["base_url"] + "/rest/v1/lab_arena_test_other_table", headers=transport._headers)
    assert response.status_code in (401, 403, 404), response.text


def test_postgrest_round_queries_isolate_shadow_and_live_modes(stack):
    transport = make_transport(stack, "lab_arena_service")
    store = ArenaStore(transport)
    suffix = uuid4().hex[:10]
    shadow_id = "arena-2026-09-04-%s" % suffix
    live_id = "arena-2026-09-05-%s" % suffix
    shadow_open_ids = ["arena-2026-09-06-%s%02d" % (suffix[:8], index) for index in range(21)]
    shadow_published_ids = ["arena-2026-09-07-%s%02d" % (suffix[:8], index) for index in range(21)]
    live_published_id = "arena-2026-09-08-%s" % suffix
    try:
        # Seed enough rows to make an unfiltered limit-20 query fail to find
        # the live row. These rows are local PostgREST fixture data only.
        with stack["connection"].cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            for round_id, mode, rewards in ((shadow_id, "shadow", False), (live_id, "live", True)):
                cursor.execute(
                    "INSERT INTO public.lab_arena_rounds (round_id,configuration_doc,rewards_enabled) VALUES (%s,%s::jsonb,%s)",
                    (round_id, json.dumps({"mode": mode, "rewards_enabled": rewards}), rewards),
                )
            for round_id in shadow_open_ids:
                cursor.execute(
                    "INSERT INTO public.lab_arena_rounds (round_id, configuration_doc, rewards_enabled) VALUES (%s, %s::jsonb, FALSE)",
                    (round_id, '{"mode":"shadow","rewards_enabled":false}'),
                )
            for index, round_id in enumerate(shadow_published_ids):
                cursor.execute(
                    "INSERT INTO public.lab_arena_rounds (round_id, status, configuration_doc, rewards_enabled, king_outcome, effective_reward_epoch, reward_basis_hash, reward_basis_doc, signing_key_doc, reward_activated_at, published_at) VALUES (%s, 'published', %s::jsonb, FALSE, 'no_king', %s, %s, %s::jsonb, %s::jsonb, clock_timestamp(), clock_timestamp())",
                    (
                        round_id,
                        '{"mode":"shadow","rewards_enabled":false,"baseline_hotkey":"baseline"}',
                        10000 + index,
                        "sha256:" + ("%02x" % index) * 32,
                        '{"schema_version":"leadpoet.lab_arena.reward_basis.v1","king_outcome":"no_king"}',
                        '{"public_key_hash":"sha256:' + ("a" * 64) + '"}',
                    ),
                )
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds (round_id, status, configuration_doc, rewards_enabled, king_outcome, effective_reward_epoch, reward_basis_hash, reward_basis_doc, signing_key_doc, reward_activated_at, published_at) VALUES (%s, 'published', %s::jsonb, TRUE, 'no_king', 9000, %s, %s::jsonb, %s::jsonb, clock_timestamp(), clock_timestamp())",
                (
                    live_published_id,
                    '{"mode":"live","rewards_enabled":true,"baseline_hotkey":"baseline"}',
                    "sha256:" + "b" * 64,
                    '{"schema_version":"leadpoet.lab_arena.reward_basis.v1","king_outcome":"no_king"}',
                    '{"public_key_hash":"sha256:' + ("c" * 64) + '"}',
                ),
            )
            cursor.execute("SET session_replication_role=origin")
        stack["connection"].commit()
        shadow_rounds = {row["round_id"] for row in store.list_rounds(status="open", mode="shadow", limit=20)}
        live_rounds = {row["round_id"] for row in store.list_rounds(status="open", mode="live", limit=20)}
        assert len(shadow_rounds) == 20
        assert shadow_rounds <= {shadow_id, *shadow_open_ids}
        assert live_id in live_rounds and shadow_id not in live_rounds
        assert live_id not in {
            row["round_id"] for row in store.list_rounds(status="open", limit=20)
        }
        # Reward reads select activated, reward-enabled bases across statuses.
        # Disabled shadow publications cannot displace the live commitment.
        assert live_published_id in {
            row["round_id"] for row in store.published_reward_bases(limit=20)
        }
        shadow_bases = store.published_reward_bases(mode="shadow", limit=20)
        assert shadow_bases == []
        live_bases = store.published_reward_bases(mode="live", limit=20)
        assert {row["round_id"] for row in live_bases} == {live_published_id}
    finally:
        # Only synthetic pagination rows in this disposable database.
        with stack["connection"].cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("TRUNCATE public.lab_arena_rounds CASCADE")
            cursor.execute("SET session_replication_role=origin")
        transport.close()


def test_anon_and_service_role_tokens_are_denied_on_arena_tables_and_functions(stack):
    for role in ("anon", "service_role", "authenticated"):
        transport = make_transport(stack, role)
        headers = transport._headers
        client = transport._client
        assert client.get(stack["base_url"] + "/rest/v1/lab_arena_rounds", headers=headers).status_code in (401, 403), role
        rpc = client.post(stack["base_url"] + "/rest/v1/rpc/lab_arena_whoami", headers=headers, content=b"{}")
        assert rpc.status_code in (401, 403, 404), (role, rpc.text)
        create = client.post(stack["base_url"] + "/rest/v1/rpc/lab_arena_create_round", headers=headers, content=json.dumps({"p_round_id": "arena-2026-09-03", "p_configuration_doc": {}}).encode())
        assert create.status_code in (401, 403, 404), (role, create.text)


def test_wrong_secret_or_unknown_role_is_rejected(stack):
    forged = make_transport(
        stack,
        "lab_arena_service",
        jwt_secret="wrong-secret-" + "y" * 40,
    )
    response = forged._client.post(stack["base_url"] + "/rest/v1/rpc/lab_arena_whoami", headers=forged._headers, content=b"{}")
    assert response.status_code == 401
    unknown = make_transport(stack, "postgres")
    response = unknown._client.post(stack["base_url"] + "/rest/v1/rpc/lab_arena_whoami", headers=unknown._headers, content=b"{}")
    assert response.status_code in (401, 403), response.text


@pytest.mark.parametrize("variant", [
    "missing", "unknown", "case_changed", "duplicate_key", "duplicate_header_case",
    "mismatched_bearer", "forged_role", "bearer_only", "raw_bearer", "duplicate_bearer",
])
def test_edge_rejects_credential_confusion(stack, variant):
    values = stack["values"]
    key = values["arena_key"]
    headers = [("apikey", key)]
    if variant == "missing":
        headers = []
    elif variant == "unknown":
        headers = [("apikey", key + "unknown")]
    elif variant == "case_changed":
        headers = [("apikey", key.replace("Arena_Aa1", "arena_aa1"))]
    elif variant == "duplicate_key":
        headers += [("apikey", key)]
    elif variant == "duplicate_header_case":
        headers += [("ApiKey", values["gateway_key"])]
    elif variant == "mismatched_bearer":
        headers += [("Authorization", "Bearer " + values["gateway_key"])]
    elif variant == "forged_role":
        headers += [("Authorization", "Bearer " + mint_jwt("postgres"))]
    elif variant == "bearer_only":
        headers = [("Authorization", "Bearer " + key)]
    elif variant == "raw_bearer":
        headers += [("Authorization", key)]
    elif variant == "duplicate_bearer":
        headers += [("Authorization", "Bearer " + key), ("Authorization", "Bearer " + key)]
    response = httpx.post(stack["base_url"] + "/rest/v1/rpc/api_test_identity", headers=headers, json={})
    assert response.status_code in (400, 401)
    assert key not in response.text and values["gateway_key"] not in response.text


def test_exact_three_roles_and_untrusted_claim_headers(stack):
    values = stack["values"]
    expected = {"arena_key": "lab_arena_service", "gateway_key": "service_role", "public_key": "anon"}
    for name, role in expected.items():
        for authorization in (None, "Bearer " + values[name], "bEaReR " + values[name]):
            headers = {"apikey": values[name], "X-JWT-Claims": '{"role":"postgres"}',
                       "X-Forwarded-Authorization": "Bearer " + mint_jwt("postgres")}
            if authorization:
                headers["Authorization"] = authorization
            response = httpx.post(stack["base_url"] + "/rest/v1/rpc/api_test_identity", headers=headers, json={})
            assert response.status_code == 200
            result = response.json()
            assert result["role"] == result["claims"]["role"] == role
            assert result["claims"]["aud"] == "leadpoet-subnet-api"
            if role == "lab_arena_service":
                assert (result["statement_timeout"], result["lock_timeout"], result["idle_timeout"]) == ("30s", "5s", "1min")


@pytest.mark.parametrize("key_name,role,timeouts", [
    ("arena_key", "lab_arena_service", ("30s", "5s", "1min")),
    ("gateway_key", "service_role", ("8s", "8s", "0")),
    ("public_key", "anon", ("10s", "8s", "0")),
])
def test_request_role_timeouts_preserve_source_defaults(stack, key_name, role, timeouts):
    response = httpx.post(stack["base_url"] + "/rest/v1/rpc/api_test_identity",
                          headers={"apikey": stack["values"][key_name]}, json={})
    assert response.status_code == 200
    result = response.json()
    assert result["role"] == result["claims"]["role"] == role
    assert tuple(result[name] for name in
                 ("statement_timeout", "lock_timeout", "idle_timeout")) == timeouts


@pytest.mark.parametrize("key_name", ["arena_key", "gateway_key", "public_key"])
def test_source_row_cap_and_next_page_use_actual_ingress(stack, key_name):
    url = stack["base_url"] + "/rest/v1/rpc/api_test_row_limit"
    headers = {"apikey": stack["values"][key_name]}
    first = httpx.get(url, headers=headers, params={"limit": 1001})
    assert first.status_code == 200
    assert first.json() == list(range(1, 1001))
    assert first.headers["content-range"] == "0-999/*"
    last = httpx.get(url, headers={**headers, "Range": "1000-1000"})
    assert last.status_code in (200, 206)
    assert last.json() == [1001]
    assert last.headers["content-range"] == "1000-1000/*"


def test_arena_direct_writes_and_private_icps_are_denied(stack):
    with httpx.Client(base_url=stack["base_url"], timeout=10) as client:
        for name in ("arena_key", "gateway_key", "public_key"):
            headers = {"apikey": stack["values"][name]}
            for table in TABLES:
                role = {"arena_key": "lab_arena_service", "gateway_key": "service_role", "public_key": "anon"}[name]
                with stack["connection"].cursor() as cursor:
                    cursor.execute("SELECT has_table_privilege(%s,%s,'INSERT')", (role, "public." + table))
                    assert cursor.fetchone() == (False,)
                response = client.post("/rest/v1/" + table, headers=headers, json=[])
                if table == "lab_arena_published_results_v1" and response.status_code == 500:
                    # PostgreSQL checks this non-updatable view's shape before
                    # INSERT ACLs. Preserve its actual SQLSTATE/error mapping.
                    assert response.json()["code"] == "55000"
                else:
                    assert response.status_code in (401, 403, 404), (name, table, response.status_code)
            if name != "gateway_key":
                response = client.get("/rest/v1/qualification_private_icp_sets", headers=headers)
                assert response.status_code in (401, 403)


def test_gateway_sdk_upsert_patch_and_public_helper_use_actual_ingress(stack, monkeypatch):
    from gateway.db.client import create_http1_sync_client
    from Leadpoet.utils import public_supabase
    client = create_http1_sync_client(stack["base_url"], stack["values"]["gateway_key"])
    try:
        table = client.table("qualification_private_icp_sets")
        data = {"set_id": 20991001, "icps": [], "is_active": False}
        assert table.upsert(data, on_conflict="set_id").execute().data[0]["set_id"] == data["set_id"]
        table.update({"is_active": True}).eq("set_id", data["set_id"]).execute()
        result = table.select("set_id,is_active").eq("set_id", data["set_id"]).execute().data
        assert result == [{"set_id": data["set_id"], "is_active": True}]
    finally:
        client.postgrest.session.close()
    monkeypatch.setattr(public_supabase, "SUPABASE_URL", stack["base_url"])
    monkeypatch.setattr(public_supabase, "SUPABASE_ANON_KEY", stack["values"]["public_key"])
    assert public_supabase.call_public_rpc("api_test_identity")["role"] == "anon"


def test_large_json_sqlstate_and_non_rest_paths_keep_contract(stack):
    headers = {"apikey": stack["values"]["arena_key"]}
    payload = "x" * (2 * 1024 * 1024)
    response = httpx.post(stack["base_url"] + "/rest/v1/rpc/api_test_payload",
                          headers=headers, json={"p_body": payload}, timeout=15)
    assert response.status_code == 200 and response.json() == len(payload)
    response = httpx.post(stack["base_url"] + "/rest/v1/rpc/api_test_sqlstate", headers=headers, json={})
    assert response.status_code == 400 and response.json()["code"] == "22023"
    for path in ("/rpc/api_test_identity", "/auth/v1", "/storage/v1", "/rest/v1-not/rpc/api_test_identity"):
        assert httpx.get(stack["base_url"] + path, headers=headers).status_code == 404
    for name in (stack["proxy_name"], stack["rest_name"]):
        logs = _docker("logs", name)
        assert not logs.stdout.strip()


def test_private_backend_has_no_anonymous_fallback(stack):
    response = _docker("exec", stack["proxy_name"], "wget", "-q", "-O", "-",
                       "http://" + stack["rest_name"] + ":3000/rpc/api_test_identity")
    assert response.returncode != 0 and "401" in response.stderr


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_capture_reader_opaque_key_returns_only_aggregates_and_cannot_escalate(stack, version):
    headers = {"apikey": stack["values"]["capture_reader_key"]}
    arguments = {"p_round_id": "capture-reader-route", "p_submission_id": "baseline-route"}
    with httpx.Client(base_url=stack["base_url"], timeout=10, trust_env=False) as client:
        response = client.post("/rest/v1/rpc/lab_arena_capture_summary_" + version, headers=headers, json=arguments)
        assert response.status_code == 200, response.text
        assert response.json() == [{"round_id": arguments["p_round_id"],
            "submission_id": arguments["p_submission_id"], "execute_runs": 0, "icp_positions": [],
            "model_calls": 0, "usage_calls": 0, "reasoning_tokens": None,
            "first_call_at": None, "last_call_at": None,
            **({"input_tokens": None, "output_tokens": None, "cached_input_tokens": None,
                "token_calls": 0, "models": []} if version == "v2" else {})}]
        for table in ("lab_arena_ledger", "lab_arena_runs", "lab_arena_submissions"):
            assert client.get("/rest/v1/" + table + "?limit=1", headers=headers).status_code == 403
            assert client.post("/rest/v1/" + table, headers=headers, json={}).status_code == 403
        assert client.post("/rest/v1/rpc/lab_arena_create_round", headers=headers,
                           json={"p_round_id": "reader-must-not-create", "p_configuration": {}}).status_code in (403, 404)
        for name in ("arena_key", "gateway_key", "public_key"):
            assert client.post("/rest/v1/rpc/lab_arena_capture_summary_" + version,
                headers={"apikey": stack["values"][name]}, json=arguments).status_code == 403
            assert client.post("/rest/v1/rpc/lab_arena_capture_summary_" + version,
                headers={**headers, "Authorization": "Bearer " + stack["values"][name]},
                json=arguments).status_code == 401
        assert client.post("/rest/v1/rpc/lab_arena_capture_summary_" + version,
            headers={**headers, "X-JWT-Claims": '{"role":"service_role"}'}, json=arguments).status_code == 200
        assert client.get("/rest/v1/lab_arena_ledger?limit=1",
            headers={**headers, "X-JWT-Claims": '{"role":"service_role"}'}).status_code == 403


def test_capture_reader_cannot_reach_retained_public_privileges(stack):
    with stack["connection"].cursor() as cursor:
        cursor.execute("SELECT has_function_privilege('leadpoet_capture_reader',"
                       "'public.api_test_legacy_public()','EXECUTE'),"
                       "has_table_privilege('leadpoet_capture_reader',"
                       "'public.api_test_public_table','SELECT')")
        assert cursor.fetchone() == (True, True)
    with httpx.Client(base_url=stack["base_url"], timeout=10, trust_env=False) as client:
        for name in ("arena_key", "gateway_key", "public_key"):
            response = client.post("/rest/v1/rpc/api_test_legacy_public",
                                   headers={"apikey": stack["values"][name]}, json={})
            assert response.status_code == 200 and response.json() == "legacy-public"
        headers = {"apikey": stack["values"]["capture_reader_key"]}
        assert client.post("/rest/v1/rpc/api_test_legacy_public", headers=headers, json={}).status_code == 403
        assert client.get("/rest/v1/api_test_public_table", headers=headers).status_code == 403


@pytest.mark.parametrize("method", ["GET", "HEAD", "PUT", "PATCH", "DELETE", "OPTIONS"])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_capture_reader_rejects_other_methods(stack, version, method):
    response = httpx.request(method, stack["base_url"] + "/rest/v1/rpc/lab_arena_capture_summary_" + version,
        headers={"apikey": stack["values"]["capture_reader_key"]}, timeout=10, trust_env=False)
    assert response.status_code == 403


@pytest.mark.parametrize("path", [
    "/rest/v1/rpc/%6cab_arena_capture_summary_v1",
    "/rest%2fv1/rpc/lab_arena_capture_summary_v1",
    "/rest/v1//rpc/lab_arena_capture_summary_v1",
    "/rest/v1/rpc/./lab_arena_capture_summary_v1",
    "/rest/v1/other/../rpc/lab_arena_capture_summary_v1",
    "/rest/v1/rpc/lab_arena_capture_summary_v1/",
    "/rest/v1/rpc/LAB_ARENA_CAPTURE_SUMMARY_V1",
    "/rest/v1/rpc/lab_arena_capture_summary_v1?select=*",
])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_capture_reader_rejects_noncanonical_request_targets(stack, version, path):
    path = path.replace("_v1", "_" + version).replace("_V1", "_" + version.upper())
    # Keep the original target verbatim; HTTP client URL normalization must
    # not remove dot segments before nginx sees this regression input.
    address = urlsplit(stack["base_url"])
    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=10)
    try:
        connection.request("POST", path, body=json.dumps({"p_round_id": "route", "p_submission_id": "route"}),
            headers={"apikey": stack["values"]["capture_reader_key"], "Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("header,value", [
    ("Accept-Profile", "extensions"), ("Content-Profile", "extensions"),
    ("X-HTTP-Method-Override", "GET"), ("X-HTTP-Method", "DELETE"),
    ("X-Method-Override", "PATCH"),
    ("X-Original-URL", "/rest/v1/rpc/api_test_legacy_public"),
    ("X-Rewrite-URL", "/rest/v1/rpc/api_test_legacy_public"),
])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_capture_reader_rejects_schema_method_and_path_override_headers(stack, version, header, value):
    response = httpx.post(stack["base_url"] + "/rest/v1/rpc/lab_arena_capture_summary_" + version,
        headers={"apikey": stack["values"]["capture_reader_key"], header: value},
        json={"p_round_id": "route", "p_submission_id": "route"}, timeout=10, trust_env=False)
    assert response.status_code == 403


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_capture_reader_accepts_explicit_public_profiles(stack, version):
    response = httpx.post(stack["base_url"] + "/rest/v1/rpc/lab_arena_capture_summary_" + version,
        headers={"apikey": stack["values"]["capture_reader_key"],
                 "Accept-Profile": "public", "Content-Profile": "public"},
        json={"p_round_id": "route", "p_submission_id": "route"}, timeout=10, trust_env=False)
    assert response.status_code == 200


def test_capture_reader_request_timeout_is_applied_by_postgrest(stack):
    # Fixture-only introspection at the sole allowed RPC name. Granting another
    # function must no longer make it reachable through the opaque reader key.
    with stack["connection"].cursor() as cursor:
        cursor.execute("SELECT pg_get_functiondef('public.lab_arena_capture_summary_v1(text,text)'::regprocedure)")
        original = cursor.fetchone()[0]
        cursor.execute("""
            CREATE OR REPLACE FUNCTION public.lab_arena_capture_summary_v1(p_round_id text,p_submission_id text)
            RETURNS TABLE (round_id text,submission_id text,execute_runs integer,icp_positions integer[],
                model_calls integer,usage_calls integer,reasoning_tokens bigint,
                first_call_at timestamptz,last_call_at timestamptz)
            LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$
                SELECT current_setting('role'),current_setting('statement_timeout'),
                    0,'{}'::integer[],0,0,NULL::bigint,NULL::timestamptz,NULL::timestamptz
            $$
        """)
    try:
        response = httpx.post(stack["base_url"] + "/rest/v1/rpc/lab_arena_capture_summary_v1",
            headers={"apikey": stack["values"]["capture_reader_key"]},
            json={"p_round_id": "route", "p_submission_id": "route"}, timeout=10, trust_env=False)
        assert response.status_code == 200, response.text
        assert response.json()[0]["round_id"] == "leadpoet_capture_reader"
        assert response.json()[0]["submission_id"] == "1min"
    finally:
        with stack["connection"].cursor() as cursor:
            cursor.execute(original)


def test_capture_reader_baseline_and_miner_usage_reaches_http_without_private_bodies(stack):
    round_id = "capture-route-" + uuid4().hex[:10]
    with stack["connection"].cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        for submission, kind, reported in (("baseline", "execute", 123), ("miner", "execute", 456),
                                          ("miner", "score", 9999)):
            run_id = round_id + submission + kind
            cursor.execute("INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,submission_id,"
                "miner_hotkey,stage,icp_position,attempt,kind,scoring_generation_floor) "
                "VALUES (%s,%s,%s,%s,'private-test-hotkey',1,0,1,%s,0)",
                (run_id, run_id, round_id, submission, kind))
            terminal = {"call_succeeded": True, "status": 200, "body_b64": base64.b64encode(json.dumps({
                "model": "test/model", "usage": {"reasoning_tokens": reported, "prompt_tokens": 1000,
                    "completion_tokens": reported + 10, "prompt_tokens_details": {"cached_tokens": 200}},
                "private_reasoning": "private-test-reasoning"
            }).encode()).decode()}
            for entry_kind in ("reservation", "settlement"):
                cursor.execute("INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,round_id,"
                    "submission_id,run_id,stage,call_identity,provider,operation_id,amount_microusd,terminal_response) "
                    "VALUES (%s,'private-test-hotkey',%s,%s,%s,1,%s,'openrouter','openrouter.chat',0,%s)",
                    (entry_kind, round_id, submission, run_id, 'sha256:' + hashlib.sha256(run_id.encode()).hexdigest(),
                     json.dumps(terminal) if entry_kind == "settlement" else None))
        cursor.execute("SET session_replication_role=origin")
    with httpx.Client(base_url=stack["base_url"], timeout=10, trust_env=False) as client:
        for submission, reported in (("baseline", 123), ("miner", 456)):
            response = client.post("/rest/v1/rpc/lab_arena_capture_summary_v1",
                headers={"apikey": stack["values"]["capture_reader_key"]},
                json={"p_round_id": round_id, "p_submission_id": submission})
            assert response.status_code == 200, response.text
            assert len(response.json()) == 1
            summary = response.json()[0]
            assert (summary["execute_runs"], summary["model_calls"], summary["usage_calls"],
                    summary["reasoning_tokens"]) == (1, 1, 1, reported)
            assert "private-test" not in response.text and "body_b64" not in response.text
            v2 = client.post("/rest/v1/rpc/lab_arena_capture_summary_v2",
                headers={"apikey": stack["values"]["capture_reader_key"]},
                json={"p_round_id": round_id, "p_submission_id": submission})
            assert v2.status_code == 200, v2.text
            summary2 = v2.json()[0]
            assert {key: summary2[key] for key in summary} == summary
            assert {key: summary2[key] for key in ("input_tokens", "output_tokens", "cached_input_tokens", "token_calls", "models")} == {
                "input_tokens": 1000, "output_tokens": reported + 10, "cached_input_tokens": 200,
                "token_calls": 1, "models": [{"model": "test/model", "calls": 1,
                    "input_tokens": 1000, "output_tokens": reported + 10, "reasoning_tokens": reported}]}
            assert "private-test" not in v2.text and "body_b64" not in v2.text


def test_long_run_filter_and_postgrest_content_location_survive_target(stack):
    run_ids = ["run-" + "x" * 36 + "-%03d" % number for number in range(215)]
    params = [("select", "run_id"), ("run_id", "in.(" + ",".join(run_ids) + ")"), ("limit", "1")]
    with httpx.Client(timeout=10, trust_env=False) as client:
        request = client.build_request("GET", stack["base_url"] + "/rest/v1/lab_arena_runs",
                                       params=params, headers={"apikey": stack["values"]["arena_key"]})
        assert 8192 < len(request.url.raw_path) < 16384
        response = client.send(request)
    assert response.status_code == 200
    assert response.json() == []
    assert len(response.headers["Content-Location"]) > 8192
    assert "run_id=in." in response.headers["Content-Location"]
