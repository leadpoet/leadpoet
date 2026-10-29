"""Prove the least-privilege PostgREST route with a real container.

A test-only client Authorization header carries the
``lab_arena_service`` role while ``PostgrestTransport`` keeps its production
scoped-key headers. The whoami readback shows the exact role, and unauthorized
roles are denied on every Arena table and function.

This is the local stand-in for the hosted-Supabase role preflight. Hosted
PostgREST uses the same role claim behavior.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import shutil
import socket
import subprocess
import sys
import time
from uuid import uuid4

import httpx
import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PostgrestTransport
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena import lab_arena_pg_harness
from tests.lab_arena.test_lab_arena_migration_postgres import round_config

DOCKER = shutil.which("docker")
POSTGREST_IMAGE = "postgrest/postgrest:v12.2.3"
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
def stack():
    # Use the same exact current schema as the service tests. Keep real
    # PostgREST role switching, HTTP coercion, RLS and RPC execution.
    database = current_service_database()
    connection = None
    rest_name = "lab-arena-rest-" + uuid4().hex[:10]
    rest_port = _free_port()
    started = False
    try:
        # A Docker database exposes its password-authenticated host port to
        # PostgREST on both Linux CI and Docker Desktop.
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("LAB_ARENA_PG_LOCAL", "0")
            seed_database = lab_arena_pg_harness._database_with_migrations

            def with_authenticator(migrations, *, setup_sql="", **kwargs):
                # The real Arena migration must grant its scoped role to an
                # existing authenticator, as hosted Supabase does.
                setup_sql += "\nCREATE ROLE authenticator LOGIN PASSWORD 'authenticator-pw' NOINHERIT; GRANT anon,authenticated,service_role TO authenticator;"
                return seed_database(migrations, setup_sql=setup_sql, **kwargs)

            patch.setattr(lab_arena_pg_harness, "_database_with_migrations", with_authenticator)
            psycopg2, dsn = next(database)
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("SELECT granted.rolname FROM pg_auth_members m JOIN pg_roles granted ON granted.oid=m.roleid JOIN pg_roles r ON r.oid=m.member WHERE r.rolname='authenticator' ORDER BY 1")
            assert "lab_arena_service" in [row[0] for row in cursor.fetchall()]
            cursor.execute("CREATE TABLE public.lab_arena_test_other_table (id INT)")
        linux = sys.platform.startswith("linux")
        host = "127.0.0.1" if linux else "host.docker.internal"
        networking = ("--network", "host") if linux else (
            "--publish", "127.0.0.1:%d:%d" % (rest_port, rest_port))
        rest = _docker(
            "run", "--rm", "--detach", "--name", rest_name, *networking,
            "--env", "PGRST_DB_URI=postgres://authenticator:authenticator-pw@%s:%d/postgres" % (host, dsn["port"]),
            "--env", "PGRST_SERVER_PORT=%d" % rest_port,
            "--env", "PGRST_DB_SCHEMAS=public", "--env", "PGRST_DB_ANON_ROLE=anon", "--env", "PGRST_JWT_SECRET=%s" % JWT_SECRET,
            "--env", "PGRST_DB_POOL=4", POSTGREST_IMAGE,
        )
        assert rest.returncode == 0, "PostgREST container could not start: %s" % rest.stderr[-200:]
        started = True
        base_url = "http://127.0.0.1:%d" % rest_port
        with httpx.Client(http1=True, http2=False, timeout=httpx.Timeout(10.0)) as client:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                try:
                    if client.get(base_url + "/", headers={"Accept": "application/json"}).status_code < 500:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            else:
                logs = _docker("logs", rest_name)
                pytest.fail("PostgREST did not become ready: %s" % (logs.stdout + logs.stderr)[-500:])
        yield {"base_url": base_url, "connection": connection}
    finally:
        if started:
            _docker("rm", "--force", rest_name, timeout=60)
        if connection is not None:
            connection.close()
        database.close()


def make_transport(
    stack, role: str, *, jwt_secret: str = JWT_SECRET
) -> PostgrestTransport:
    client = httpx.Client(
        http1=True,
        http2=False,
        timeout=httpx.Timeout(15.0),
        base_url=stack["base_url"],
        headers={"Authorization": "Bearer " + mint_jwt(role, secret=jwt_secret)},
    )
    transport = PostgrestTransport(
        "https://placeholder.invalid",
        service_key="sb_secret_local_postgrest_test",
        http_client=client,
    )
    # Bare PostgREST serves at the root; Supabase adds /rest/v1. Point the transport at the container.
    transport._base_url = ""
    transport._client.base_url = httpx.URL(stack["base_url"])
    return transport


def test_service_role_authorization_reaches_whoami_and_functions_through_postgrest(stack):
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
    config = round_config("arena-2026-09-02", [], mode="live", rewards_enabled=False)
    assert store.create_round("arena-2026-09-02", config)["status"] == "created"
    assert store.get_round("arena-2026-09-02")["status"] == "open"
    assert store.list_rounds()[0]["round_id"] == "arena-2026-09-02"
    with pytest.raises(ArenaStoreError):
        transport.select("lab_arena_test_other_table")  # not an Arena table: refused client-side
    response = transport._client.get(stack["base_url"] + "/lab_arena_test_other_table", headers=transport._headers)
    assert response.status_code in (401, 403, 404), response.text


def test_postgrest_round_queries_isolate_shadow_and_live_modes(stack):
    transport = make_transport(stack, "lab_arena_service")
    original_post = transport._client.post
    original_get = transport._client.get
    transport._client.post = lambda url, **kwargs: original_post(url.replace("/rest/v1", ""), **kwargs)
    transport._client.get = lambda url, **kwargs: original_get(url.replace("/rest/v1", ""), **kwargs)
    store = ArenaStore(transport)
    suffix = uuid4().hex[:10]
    shadow_id = "arena-2026-09-04-%s" % suffix
    live_id = "arena-2026-09-05-%s" % suffix
    shadow_open_ids = ["arena-2026-09-06-%s%02d" % (suffix[:8], index) for index in range(21)]
    shadow_published_ids = ["arena-2026-09-07-%s%02d" % (suffix[:8], index) for index in range(21)]
    live_published_id = "arena-2026-09-08-%s" % suffix
    try:
        for round_id, mode, rewards in (
            (shadow_id, "shadow", False),
            (live_id, "live", True),
        ):
            assert store.create_round(
                round_id,
                round_config(round_id, [], mode=mode, rewards_enabled=rewards),
            )["status"] == "created"
        # Seed enough rows to make an unfiltered limit-20 query fail to find
        # the live row. These rows are local PostgREST fixture data only.
        with stack["connection"].cursor() as cursor:
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
        # The module-scoped Postgres container is disposable. Arena rounds are
        # intentionally write-once, so cleanup must not disable that trigger.
        for round_id in (shadow_id, live_id):
            transport.rpc(
                "lab_arena_cancel_round",
                {"p_round_id": round_id, "p_reason": "operator_abort"},
            )
        transport.close()


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
    forged = make_transport(
        stack,
        "lab_arena_service",
        jwt_secret="wrong-secret-" + "y" * 40,
    )
    response = forged._client.post(stack["base_url"] + "/rpc/lab_arena_whoami", headers=forged._headers, content=b"{}")
    assert response.status_code == 401
    unknown = make_transport(stack, "postgres")
    response = unknown._client.post(stack["base_url"] + "/rpc/lab_arena_whoami", headers=unknown._headers, content=b"{}")
    assert response.status_code in (401, 403), response.text
