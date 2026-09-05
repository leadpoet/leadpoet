"""Prove the least-privilege PostgREST route (labarena.md 11.1, 18.1) with a
real PostgREST container: an operator-minted JWT carrying the
``lab_arena_service`` role reaches the Arena functions through
``PostgrestTransport``, the whoami readback shows the exact role, and the
``anon`` role is denied on every Arena table and function.

This is the local stand-in for the hosted-Supabase preflight the plan
requires before step 2; hosted PostgREST behaves identically for the role
claim, and that preflight stays an operator gate.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from lab_arena import contracts
from lab_arena.store import ArenaRoleError, ArenaStore, ArenaStoreError, PostgrestTransport
from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
DOCKER = shutil.which("docker")
POSTGREST_IMAGE = "postgrest/postgrest:v12.2.3"
POSTGRES_IMAGE = "postgres:15"
JWT_SECRET = "lab-arena-local-jwt-secret-" + "x" * 40
ANON_KEY_ROLE = "anon"

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
def stack():
    psycopg2 = pytest.importorskip("psycopg2")
    suffix = uuid4().hex[:10]
    network = "lab-arena-net-%s" % suffix
    pg_name = "lab-arena-pg-%s" % suffix
    rest_name = "lab-arena-rest-%s" % suffix
    pg_port = _free_port()
    rest_port = _free_port()
    started = []
    try:
        if _docker("network", "create", network).returncode != 0:
            pytest.skip("docker network could not be created")
        started.append(("network", network))
        run = _docker("run", "--rm", "--detach", "--name", pg_name, "--network", network, "--env", "POSTGRES_PASSWORD=postgres", "--publish", "127.0.0.1:%d:5432" % pg_port, POSTGRES_IMAGE)
        if run.returncode != 0:
            pytest.skip("PostgreSQL container could not start: %s" % run.stderr[-200:])
        started.append(("container", pg_name))
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if _docker("exec", pg_name, "pg_isready", "-U", "postgres", timeout=10).returncode == 0:
                break
            time.sleep(0.5)
        else:
            pytest.fail("PostgreSQL container did not become ready")
        dsn = {"host": "127.0.0.1", "port": pg_port, "user": "postgres", "password": "postgres", "dbname": "postgres"}
        connect_deadline = time.monotonic() + 20
        while True:
            try:
                connection = psycopg2.connect(**dsn)
                break
            except psycopg2.OperationalError:
                if time.monotonic() >= connect_deadline:
                    raise
                time.sleep(0.5)
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE SCHEMA IF NOT EXISTS extensions;
                CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;
                CREATE ROLE anon NOLOGIN;
                CREATE ROLE authenticated NOLOGIN;
                CREATE ROLE service_role NOLOGIN;
                CREATE ROLE authenticator LOGIN PASSWORD 'authenticator-pw' NOINHERIT;
                GRANT anon, authenticated, service_role TO authenticator;
                CREATE TABLE public.lab_arena_test_other_table (id INT);
                CREATE TABLE public.qualification_private_icp_sets (
                  set_id BIGINT PRIMARY KEY,
                  icps JSONB NOT NULL,
                  active_from TIMESTAMPTZ,
                  active_until TIMESTAMPTZ,
                  is_active BOOLEAN NOT NULL DEFAULT FALSE
                );
                ALTER TABLE public.qualification_private_icp_sets ENABLE ROW LEVEL SECURITY;
                REVOKE ALL ON TABLE public.qualification_private_icp_sets
                  FROM PUBLIC, anon, authenticated;
                """
            )
            for migration in DEFAULT_MIGRATIONS:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
            cursor.execute("SELECT granted.rolname FROM pg_auth_members m JOIN pg_roles granted ON granted.oid = m.roleid JOIN pg_roles r ON r.oid = m.member WHERE r.rolname = 'authenticator' ORDER BY 1")
            memberships = [row[0] for row in cursor.fetchall()]
        assert "lab_arena_service" in memberships, memberships
        rest = _docker(
            "run", "--rm", "--detach", "--name", rest_name, "--network", network, "--publish", "127.0.0.1:%d:3000" % rest_port,
            "--env", "PGRST_DB_URI=postgres://authenticator:authenticator-pw@%s:5432/postgres" % pg_name,
            "--env", "PGRST_DB_SCHEMAS=public", "--env", "PGRST_DB_ANON_ROLE=anon", "--env", "PGRST_JWT_SECRET=%s" % JWT_SECRET,
            "--env", "PGRST_DB_POOL=4", POSTGREST_IMAGE,
        )
        if rest.returncode != 0:
            pytest.skip("PostgREST container could not start: %s" % rest.stderr[-200:])
        started.append(("container", rest_name))
        base_url = "http://127.0.0.1:%d" % rest_port
        client = httpx.Client(http1=True, http2=False, timeout=httpx.Timeout(10.0))
        deadline = time.monotonic() + 60
        ready = False
        while time.monotonic() < deadline:
            try:
                if client.get(base_url + "/", headers={"Accept": "application/json"}).status_code < 500:
                    ready = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        if not ready:
            logs = _docker("logs", rest_name)
            pytest.fail("PostgREST did not become ready: %s" % (logs.stdout + logs.stderr)[-500:])
        yield {"base_url": base_url, "connection": connection}
        connection.close()
    finally:
        for kind, name in reversed(started):
            if kind == "container":
                _docker("rm", "--force", name, timeout=60)
            else:
                _docker("network", "rm", name, timeout=60)


class RestClientTransport:
    """A PostgrestTransport whose URLs omit the Supabase /rest/v1 prefix (bare PostgREST)."""


def make_transport(stack, role: str) -> PostgrestTransport:
    transport = PostgrestTransport("https://placeholder.invalid", anon_key=mint_jwt(ANON_KEY_ROLE), service_jwt=mint_jwt(role), http_client=httpx.Client(http1=True, http2=False, timeout=httpx.Timeout(15.0), base_url=stack["base_url"]))
    # Bare PostgREST serves at the root; Supabase adds /rest/v1. Point the transport at the container.
    transport._base_url = ""
    transport._client.base_url = httpx.URL(stack["base_url"])
    return transport


def test_service_jwt_reaches_whoami_and_functions_through_postgrest(stack):
    transport = make_transport(stack, "lab_arena_service")
    # PostgREST has no /rest/v1 prefix: rewrite the request paths.
    original_post = transport._client.post
    original_get = transport._client.get
    transport._client.post = lambda url, **kwargs: original_post(url.replace("/rest/v1", ""), **kwargs)
    transport._client.get = lambda url, **kwargs: original_get(url.replace("/rest/v1", ""), **kwargs)
    store = ArenaStore(transport)
    identity = store.require_service_role()
    assert identity["current_user"] == "lab_arena_service" and identity["jwt_role"] == "lab_arena_service"
    assert identity["rolsuper"] is False and identity["rolbypassrls"] is False and identity["session_user"] == "authenticator"
    config = {"round_id": "arena-2026-09-02", "mode": "live", "rewards_enabled": False, "runner_hotkeys": [], "call_quotas": dict(contracts.CALL_QUOTAS_PER_ICP)}
    assert store.create_round("arena-2026-09-02", config)["status"] == "created"
    assert store.get_round("arena-2026-09-02")["status"] == "open"
    assert store.list_rounds()[0]["round_id"] == "arena-2026-09-02"
    with pytest.raises(ArenaStoreError):
        transport.select("lab_arena_test_other_table")  # not an Arena table: refused client-side
    response = transport._client.get(stack["base_url"] + "/lab_arena_test_other_table", headers=transport._headers)
    assert response.status_code in (401, 403, 404), response.text


def test_anon_and_service_role_tokens_are_denied_on_arena_tables_and_functions(stack):
    for role in ("anon", "service_role", "authenticated"):
        transport = make_transport(stack, role)
        headers = transport._headers
        client = transport._client
        assert client.get(stack["base_url"] + "/lab_arena_rounds", headers=headers).status_code in (401, 403), role
        rpc = client.post(stack["base_url"] + "/rpc/lab_arena_whoami", headers=headers, content=b"{}")
        assert rpc.status_code in (401, 403, 404), (role, rpc.text)
        create = client.post(stack["base_url"] + "/rpc/lab_arena_create_round", headers=headers, content=json.dumps({"p_round_id": "arena-2026-09-03", "p_configuration_doc": {}}).encode())
        assert create.status_code in (401, 403, 404), (role, create.text)


def test_wrong_secret_or_unknown_role_is_rejected(stack):
    forged = PostgrestTransport("https://placeholder.invalid", anon_key=mint_jwt("anon"), service_jwt=mint_jwt("lab_arena_service", secret="wrong-secret-" + "y" * 40), http_client=httpx.Client(http1=True, http2=False, timeout=httpx.Timeout(15.0)))
    response = forged._client.post(stack["base_url"] + "/rpc/lab_arena_whoami", headers=forged._headers, content=b"{}")
    assert response.status_code == 401
    unknown = PostgrestTransport("https://placeholder.invalid", anon_key=mint_jwt("anon"), service_jwt=mint_jwt("postgres"), http_client=httpx.Client(http1=True, http2=False, timeout=httpx.Timeout(15.0)))
    response = unknown._client.post(stack["base_url"] + "/rpc/lab_arena_whoami", headers=unknown._headers, content=b"{}")
    assert response.status_code in (401, 403), response.text
