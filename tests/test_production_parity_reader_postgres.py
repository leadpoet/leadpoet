from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psycopg2
from psycopg2 import OperationalError
import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PostgrestTransport
from leadpoet_canonical.production_parity import (
    CONTRACT_SCHEMA_VERSION,
    ProductionParityError,
    SNAPSHOT_SCHEMA_VERSION,
    sha256_bytes,
    sha256_json,
)
from scripts import production_parity_snapshot as parity_snapshot
from scripts import run_production_parity_fast as fast_parity
from scripts.run_production_parity_full_host import _ClonePostgrestPrefixAdapter
from scripts.materialize_production_parity_secrets import _jwt
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS as ARENA_MIGRATIONS,
    _DAILY_SOURCE_SHIM_SQL,
)
from tests.postgres_migration_harness import (
    HISTORICAL_SOURCE_ADD_UPGRADE_MIGRATIONS,
    _database_with_migrations,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts/156-production-parity-readonly-role.sql"
DATABASE = "leadpoet_rehearsal"
READER = "leadpoet_parity_reader"
PASSWORD = "a" * 64
ROTATED_PASSWORD = "b" * 64


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture(scope="module")
def postgres() -> dict[str, object]:
    name = f"leadpoet-parity-reader-{uuid.uuid4().hex[:12]}"
    try:
        _docker(
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--publish",
            "127.0.0.1::5432",
            "--env",
            "POSTGRES_PASSWORD=postgres",
            "postgres:15",
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"disposable PostgreSQL is unavailable: {exc}")
    try:
        port_output = _docker("port", name, "5432/tcp").stdout.strip()
        port = int(port_output.rsplit(":", 1)[1])
        admin = {
            "host": "127.0.0.1",
            "port": port,
            "dbname": "postgres",
            "user": "postgres",
            "password": "postgres",
        }
        deadline = time.monotonic() + 45
        while True:
            try:
                connection = psycopg2.connect(**admin, connect_timeout=2)
                connection.autocommit = True
                with connection.cursor() as cursor:
                    cursor.execute(f'CREATE DATABASE "{DATABASE}"')
                connection.close()
                break
            except OperationalError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
        target = {**admin, "dbname": DATABASE}
        connection = psycopg2.connect(**target)
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE ROLE anon NOLOGIN; "
                "CREATE ROLE authenticated NOLOGIN; "
                "CREATE ROLE service_role NOLOGIN; "
                "CREATE TABLE public.parity_source "
                "(id bigint PRIMARY KEY, value text NOT NULL); "
                "INSERT INTO public.parity_source VALUES (1, 'shape'); "
                "CREATE TABLE public.research_lab_finalized_allocation_epochs_v2 "
                "(netuid integer NOT NULL, epoch_id integer NOT NULL); "
                "INSERT INTO public.research_lab_finalized_allocation_epochs_v2 "
                "VALUES (71, 25000); "
                "CREATE SEQUENCE public.parity_sequence"
            )
        connection.close()
        yield {"admin": target, "port": port}
    finally:
        _docker("rm", "--force", name, check=False)


def _admin(postgres: dict[str, object]):
    connection = psycopg2.connect(**postgres["admin"])
    connection.autocommit = True
    return connection


def _reader(postgres: dict[str, object], password: str = PASSWORD):
    return psycopg2.connect(
        host="127.0.0.1",
        port=postgres["port"],
        dbname=DATABASE,
        user=READER,
        password=password,
        connect_timeout=3,
    )


def _apply_migration(postgres: dict[str, object]) -> None:
    connection = _admin(postgres)
    try:
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    finally:
        connection.close()


def _bind(postgres: dict[str, object], password: str) -> dict[str, object]:
    connection = _admin(postgres)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public."
                "leadpoet_set_production_parity_reader_password_v1(%s)",
                (password,),
            )
            value = cursor.fetchone()[0]
            return json.loads(value) if isinstance(value, str) else value
    finally:
        connection.close()


def _contract(postgres: dict[str, object]) -> dict[str, object]:
    connection = _admin(postgres)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.leadpoet_production_parity_reader_contract_v1()"
            )
            value = cursor.fetchone()[0]
            return json.loads(value) if isinstance(value, str) else value
    finally:
        connection.close()


def _snapshot_contract() -> dict[str, object]:
    candidate_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD^"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    migrations = parity_snapshot._source_migrations(
        root=ROOT,
        source_sha=candidate_sha,
        candidate_sha=candidate_sha,
    )
    body = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "base_sha": base_sha,
        "candidate_sha": candidate_sha,
        "changed_paths": [],
        "risk": {
            "class": "low",
            "full_physical_required": False,
            "reasons": ["documentation_or_test_only"],
        },
        "source_commitments": [
            {
                "path": "leadpoet_canonical/production_parity.py",
                "sha256": sha256_bytes(
                    subprocess.run(
                        [
                            "git",
                            "show",
                            f"{candidate_sha}:leadpoet_canonical/production_parity.py",
                        ],
                        cwd=ROOT,
                        capture_output=True,
                        check=True,
                    ).stdout
                ),
            }
        ],
        "migrations": migrations,
        "behavior_contract_hash": "sha256:" + "1" * 64,
        "protected_manifest_hash": "sha256:" + "2" * 64,
        "historical_oracle_hash": "sha256:" + "3" * 64,
        "runtime_config_hash": "sha256:" + "4" * 64,
        "runtime_config_keys": [],
        "policy_commitments": {},
    }
    return {**body, "contract_hash": sha256_json(body)}


def _pinned_postgres_image() -> str:
    raw = _docker(
        "image",
        "inspect",
        "postgres:15",
        "--format",
        "{{json .RepoDigests}}",
    ).stdout.strip()
    digests = json.loads(raw)
    assert isinstance(digests, list) and digests
    return str(digests[0])


def _pinned_postgrest_image() -> str:
    raw = _docker(
        "image",
        "inspect",
        "postgrest/postgrest:v12.2.8",
        "--format",
        "{{json .RepoDigests}}",
    ).stdout.strip()
    digests = json.loads(raw)
    assert isinstance(digests, list) and digests
    return str(digests[0])


def test_reader_migration_is_clone_safe_read_only_and_idempotent(postgres):
    _apply_migration(postgres)
    initial = _contract(postgres)
    assert initial == {
        "schema_version": "leadpoet.production-parity-reader-contract.v1",
        "database_name": DATABASE,
        "reader_role": READER,
        "login_enabled": False,
        "superuser": False,
        "bypass_rls": True,
        "createdb": False,
        "createrole": False,
        "inherit": False,
        "replication": False,
        "connection_limit": 2,
        "default_read_only": True,
        "membership_count": 0,
        "schema_create_capable": False,
        "table_write_capable": False,
        "sequence_write_capable": False,
    }
    assert _bind(postgres, PASSWORD)["status"] == "bound"

    reader = _reader(postgres)
    try:
        reader.autocommit = True
        with reader.cursor() as cursor:
            cursor.execute(
                "SELECT current_user, current_setting('transaction_read_only'), "
                "(SELECT value FROM public.parity_source WHERE id = 1)"
            )
            assert cursor.fetchone() == (READER, "on", "shape")
            with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
                cursor.execute(
                    "INSERT INTO public.parity_source VALUES (2, 'forbidden')"
                )
    finally:
        reader.close()

    with pytest.raises(OperationalError):
        _reader(postgres, ROTATED_PASSWORD)

    # Reapplying the exact migration preserves the existing LOGIN/password.
    _apply_migration(postgres)
    assert _contract(postgres)["login_enabled"] is True
    connection = _reader(postgres)
    connection.close()
    with pytest.raises(OperationalError):
        _reader(postgres, ROTATED_PASSWORD)


def test_binder_failure_commits_nologin_for_unsafe_existing_role(postgres):
    connection = _admin(postgres)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"GRANT service_role TO {READER}")
    finally:
        connection.close()

    assert _bind(postgres, ROTATED_PASSWORD)["status"] == "disabled"
    assert _contract(postgres)["login_enabled"] is False
    with pytest.raises(OperationalError):
        _reader(postgres, PASSWORD)
    with pytest.raises(OperationalError):
        _reader(postgres, ROTATED_PASSWORD)

    # Even malformed input disables a pre-existing LOGIN before validation.
    connection = _admin(postgres)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"REVOKE service_role FROM {READER}")
            cursor.execute(
                f"ALTER ROLE {READER} LOGIN PASSWORD %s",
                (PASSWORD,),
            )
    finally:
        connection.close()
    assert _bind(postgres, "not-a-password")["status"] == "disabled"
    assert _contract(postgres)["login_enabled"] is False


def test_migration_refuses_an_existing_superuser_collision(postgres):
    _apply_migration(postgres)
    connection = _admin(postgres)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"ALTER ROLE {READER} SUPERUSER")
    finally:
        connection.close()

    try:
        with pytest.raises(
            psycopg2.errors.RaiseException,
            match="unexpectedly superuser",
        ):
            _apply_migration(postgres)
        assert _contract(postgres)["superuser"] is True
    finally:
        connection = _admin(postgres)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"ALTER ROLE {READER} NOSUPERUSER")
        finally:
            connection.close()
        _apply_migration(postgres)


def test_snapshot_v6_real_capture_verify_restore_is_candidate_bound(
    postgres,
    monkeypatch,
    tmp_path: Path,
):
    _apply_migration(postgres)
    assert _bind(postgres, PASSWORD)["status"] == "bound"
    target_database = "leadpoet_parity_snapshot_v6"
    connection = _admin(postgres)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'DROP DATABASE IF EXISTS "{target_database}"')
            cursor.execute(f'CREATE DATABASE "{target_database}"')
    finally:
        connection.close()

    contract = _snapshot_contract()
    contract_path = tmp_path / "contract.json"
    manifest_path = tmp_path / "snapshot-manifest.json"
    archive_path = tmp_path / "production.dump"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    original_postgres_env = parity_snapshot._postgres_env
    client_host = (
        "host.docker.internal" if sys.platform == "darwin" else "127.0.0.1"
    )

    def local_postgres_env(dsn: str, *, read_only: bool):
        env, host = original_postgres_env(dsn, read_only=read_only)
        env["PGSSLMODE"] = "disable"
        env["PGHOST"] = client_host
        return env, host

    monkeypatch.setattr(parity_snapshot, "_postgres_env", local_postgres_env)
    postgres_image = _pinned_postgres_image()
    source_dsn = (
        f"postgresql://{READER}:{PASSWORD}@production.test:"
        f"{postgres['port']}/{DATABASE}"
    )
    target_dsn = (
        "postgresql://postgres:postgres@127.0.0.1:"
        f"{postgres['port']}/{target_database}"
    )

    manifest = parity_snapshot.capture_snapshot(
        contract_path=contract_path,
        archive_path=archive_path,
        manifest_path=manifest_path,
        dsn=source_dsn,
        expected_production_host="production.test",
        ttl_hours=1,
        source_sha=str(contract["base_sha"]),
        postgres_image=postgres_image,
    )
    evidence = parity_snapshot.verify_snapshot(
        contract_path=contract_path,
        manifest_path=manifest_path,
        archive_path=archive_path,
        expected_production_host="production.test",
        postgres_image=postgres_image,
    )
    restored = parity_snapshot.restore_snapshot(
        root=ROOT,
        contract_path=contract_path,
        manifest_path=manifest_path,
        archive_path=archive_path,
        target_dsn=target_dsn,
        production_host="production.test",
        postgres_image=postgres_image,
    )

    assert manifest["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert manifest["archive"]["ownership"] == "preserved"
    assert manifest["archive"]["acl"] == "preserved"
    assert set(manifest["database"]) == {
        "server_version_num",
        "relation_count",
        "total_relation_bytes",
        "largest_relation_bytes",
        "capture_utc_date",
        "target_rebenchmark_date",
        "source_role",
        "weight_history_scope",
    }
    assert manifest["database"]["weight_history_scope"] == {
        "netuid": 71,
        "start_epoch": 25000,
        "end_epoch": 25000,
        "expected_rows": 1,
    }
    assert evidence["candidate_sha"] == contract["candidate_sha"]
    assert restored["manifest_hash"] == manifest["manifest_hash"]
    assert restored["migration_delta"] == []
    clone = psycopg2.connect(
        host="127.0.0.1",
        port=postgres["port"],
        dbname=target_database,
        user="postgres",
        password="postgres",
    )
    try:
        with clone.cursor() as cursor:
            cursor.execute("SELECT id, value FROM public.parity_source")
            assert cursor.fetchall() == [(1, "shape")]
    finally:
        clone.close()

    wrong_candidate = dict(contract)
    wrong_candidate["candidate_sha"] = "f" * 40
    wrong_body = {
        key: value
        for key, value in wrong_candidate.items()
        if key != "contract_hash"
    }
    wrong_candidate["contract_hash"] = sha256_json(wrong_body)
    contract_path.write_text(json.dumps(wrong_candidate), encoding="utf-8")
    with pytest.raises(ProductionParityError, match="capture commit differs"):
        parity_snapshot.verify_snapshot(
            contract_path=contract_path,
            manifest_path=manifest_path,
            archive_path=archive_path,
            postgres_image=postgres_image,
        )

    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with archive_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ProductionParityError, match="archive size differs"):
        parity_snapshot.verify_snapshot(
            contract_path=contract_path,
            manifest_path=manifest_path,
            archive_path=archive_path,
            postgres_image=postgres_image,
        )


@pytest.mark.parametrize(
    ("capture_mode", "expected_probe_rows"),
    (("full", 1), ("schema-only", 0)),
)
def test_snapshot_v6_preserves_native_acl_owners_and_arena_postgrest(
    monkeypatch,
    tmp_path: Path,
    capture_mode: str,
    expected_probe_rows: int,
):
    source_generator = _database_with_migrations(
        HISTORICAL_SOURCE_ADD_UPGRADE_MIGRATIONS
        + tuple(ARENA_MIGRATIONS)
        + ("198-retire-research-lab-source-add-schema.sql",),
        setup_sql=_DAILY_SOURCE_SHIM_SQL,
    )
    psycopg2_module, source = next(source_generator)
    database = None
    prefix_adapter = None
    try:
        connection = psycopg2_module.connect(**source)
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT public.leadpoet_set_production_parity_reader_password_v1(%s)",
                (PASSWORD,),
            )
            assert cursor.fetchone()[0]["status"] == "bound"
            cursor.execute(
                """
                DO $$ BEGIN
                  CREATE ROLE supabase_admin NOLOGIN NOINHERIT;
                EXCEPTION WHEN duplicate_object THEN NULL;
                END $$;
                CREATE TABLE public.native_acl_probe (
                  id INTEGER PRIMARY KEY,
                  value TEXT NOT NULL
                );
                INSERT INTO public.native_acl_probe VALUES (1, 'preserved');
                ALTER TABLE public.native_acl_probe OWNER TO supabase_admin;
                GRANT SELECT ON public.native_acl_probe TO leadpoet_parity_reader;
                ALTER DEFAULT PRIVILEGES FOR ROLE supabase_admin IN SCHEMA public
                  GRANT SELECT ON TABLES TO leadpoet_parity_reader;
                CREATE TABLE IF NOT EXISTS
                  public.research_lab_finalized_allocation_epochs_v2 (
                    netuid INTEGER NOT NULL,
                    epoch_id INTEGER NOT NULL
                  );
                INSERT INTO public.research_lab_finalized_allocation_epochs_v2
                  (netuid, epoch_id) VALUES (71, 25000);
                GRANT SELECT ON public.research_lab_finalized_allocation_epochs_v2
                  TO leadpoet_parity_reader;
                """
            )
        connection.close()

        contract = _snapshot_contract()
        contract_path = tmp_path / "contract.json"
        manifest_path = tmp_path / "manifest.json"
        archive_path = tmp_path / "snapshot.dump"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        original_postgres_env = parity_snapshot._postgres_env
        client_host = (
            "host.docker.internal" if sys.platform == "darwin" else "127.0.0.1"
        )

        def local_postgres_env(dsn: str, *, read_only: bool):
            env, host = original_postgres_env(dsn, read_only=read_only)
            env["PGSSLMODE"] = "disable"
            env["PGHOST"] = client_host
            return env, host

        monkeypatch.setattr(parity_snapshot, "_postgres_env", local_postgres_env)
        postgres_image = _pinned_postgres_image()
        postgrest_image = _pinned_postgrest_image()
        source_dsn = (
            f"postgresql://{READER}:{PASSWORD}@production.test:"
            f"{source['port']}/postgres"
        )
        manifest = parity_snapshot.capture_snapshot(
            contract_path=contract_path,
            archive_path=archive_path,
            manifest_path=manifest_path,
            dsn=source_dsn,
            expected_production_host="production.test",
            ttl_hours=1,
            source_sha=str(contract["base_sha"]),
            capture_mode=capture_mode,
            postgres_image=postgres_image,
        )
        database = fast_parity._DockerDatabase(
            candidate_sha=str(contract["candidate_sha"]),
            postgres_image=postgres_image,
            postgrest_image=postgrest_image,
        )
        database.start()
        prerequisites = database.prepare_snapshot_restore()
        assert prerequisites["supabase_admin_role"] is True
        assert prerequisites["parity_reader_placeholder_role"] is True
        database._psql("DROP ROLE supabase_admin")
        with pytest.raises(
            ProductionParityError,
            match="isolated production snapshot restore failed",
        ):
            parity_snapshot.restore_snapshot(
                root=ROOT,
                contract_path=contract_path,
                manifest_path=manifest_path,
                archive_path=archive_path,
                target_dsn=database.target_dsn,
                production_host="production.test",
                postgres_image=postgres_image,
            )
        prerequisites = database.prepare_snapshot_restore()
        assert prerequisites["supabase_admin_role"] is True
        restored = parity_snapshot.restore_snapshot(
            root=ROOT,
            contract_path=contract_path,
            manifest_path=manifest_path,
            archive_path=archive_path,
            target_dsn=database.target_dsn,
            production_host="production.test",
            postgres_image=postgres_image,
        )
        assert manifest["archive"]["ownership"] == "preserved"
        assert manifest["archive"]["acl"] == "preserved"
        assert restored["migration_delta"] == []

        ownership = json.loads(
            database._psql(
                """
                SELECT json_build_object(
                  'public_schema_owner', pg_catalog.pg_get_userbyid(namespace.nspowner),
                  'probe_owner', pg_catalog.pg_get_userbyid(probe.relowner),
                  'probe_reader_select', pg_catalog.has_table_privilege(
                    'leadpoet_parity_reader', probe.oid, 'SELECT'
                  ),
                  'probe_rows', (SELECT count(*) FROM public.native_acl_probe),
                  'arena_schema_owner', pg_catalog.pg_get_userbyid(function.proowner),
                  'arena_schema_security_definer', function.prosecdef,
                  'default_acl_preserved', EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_default_acl AS defaults
                    JOIN pg_catalog.pg_roles AS owner ON owner.oid = defaults.defaclrole
                    JOIN pg_catalog.pg_roles AS grantee
                      ON grantee.rolname = 'leadpoet_parity_reader'
                    CROSS JOIN LATERAL pg_catalog.aclexplode(defaults.defaclacl) AS acl
                    WHERE owner.rolname = 'supabase_admin'
                      AND defaults.defaclnamespace = namespace.oid
                      AND defaults.defaclobjtype = 'r'
                      AND acl.grantee = grantee.oid
                      AND acl.privilege_type = 'SELECT'
                  )
                )::text
                FROM pg_catalog.pg_namespace AS namespace
                JOIN pg_catalog.pg_class AS probe
                  ON probe.relnamespace = namespace.oid
                 AND probe.relname = 'native_acl_probe'
                JOIN pg_catalog.pg_proc AS function
                  ON function.oid =
                     'public.lab_arena_schema_version_v1()'::regprocedure
                WHERE namespace.nspname = 'public';
                """
            )
        )
        assert ownership == {
            "public_schema_owner": "pg_database_owner",
            "probe_owner": "supabase_admin",
            "probe_reader_select": True,
            "probe_rows": expected_probe_rows,
            "arena_schema_owner": "lab_arena_owner",
            "arena_schema_security_definer": True,
            "default_acl_preserved": True,
        }

        local_postgrest_url, service_token = database.start_postgrest()
        prefix_adapter = _ClonePostgrestPrefixAdapter(
            upstream_origin=local_postgrest_url,
            public_origin="https://test.cloudfront.net",
            listen_host="127.0.0.1",
            listen_port=0,
        )
        prefix_evidence = prefix_adapter.start()
        supabase_url = "http://127.0.0.1:%d" % prefix_evidence["listen_port"]
        arena_token = _jwt(database.jwt_secret, "lab_arena_service")
        arena_transport = PostgrestTransport(
            supabase_url,
            anon_key=arena_token,
            service_jwt=arena_token,
            timeout_seconds=10,
        )
        arena = ArenaStore(arena_transport)
        try:
            assert arena.require_service_role()["current_user"] == "lab_arena_service"
            schema = arena._transport.rpc("lab_arena_schema_version_v1", {})
            assert schema["version"] == 197
            for table in (
                "lab_arena_rounds",
                "lab_arena_submissions",
                "lab_arena_runs",
                "lab_arena_ledger",
            ):
                assert arena._transport.select(table, limit=1) == []
            for function, params in (
                ("lab_arena_expire_leases", {"p_round_id": "arena-0000-00-00"}),
                ("lab_arena_close_stage", {"p_round_id": "arena-0000-00-00", "p_stage": 1}),
                ("lab_arena_cancel_round", {"p_round_id": "arena-0000-00-00", "p_reason": "startup-probe"}),
            ):
                with pytest.raises(ArenaStoreError, match="lab_arena_round_missing"):
                    arena._transport.rpc(function, params)
            assert arena.current_daily_icp_set(20990101)["status"] == "unavailable"
        finally:
            arena.close()

        for role_token in (
            service_token,
            _jwt(database.jwt_secret, "anon"),
            _jwt(database.jwt_secret, "authenticated"),
        ):
            transport = PostgrestTransport(
                supabase_url,
                anon_key=role_token,
                service_jwt=role_token,
                timeout_seconds=10,
            )
            try:
                with pytest.raises(ArenaStoreError):
                    transport.select("lab_arena_rounds", limit=1)
            finally:
                transport.close()

        with psycopg2_module.connect(**source) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM pg_catalog.pg_class c
                    JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public'
                      AND c.relname LIKE 'research_lab_source_add%'
                    """
                )
                assert cursor.fetchone() == (0,)
    finally:
        if prefix_adapter is not None:
            prefix_adapter.cleanup()
        if database is not None:
            database.cleanup()
        source_generator.close()
