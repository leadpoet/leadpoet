from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import uuid

import psycopg2
from psycopg2 import OperationalError
import pytest


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
