"""Execute the SOURCE_ADD admission-control migration on real PostgreSQL."""

from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "145-research-lab-source-add-admission-control.sql"
DOCKER = shutil.which("docker")
pytestmark = pytest.mark.skipif(DOCKER is None, reason="Docker is unavailable")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture(scope="module")
def postgres():
    psycopg2 = pytest.importorskip("psycopg2")
    port = _free_port()
    container = "source-add-control-%s" % uuid4().hex[:12]
    started = False
    try:
        result = subprocess.run(
            [
                str(DOCKER),
                "run",
                "--rm",
                "--detach",
                "--name",
                container,
                "--env",
                "POSTGRES_PASSWORD=postgres",
                "--publish",
                "127.0.0.1:%d:5432" % port,
                "postgres:15",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip("PostgreSQL container could not start: %s" % result.stderr[-300:])
        started = True
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            ready = subprocess.run(
                [str(DOCKER), "exec", container, "pg_isready", "-U", "postgres"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        else:
            pytest.fail("PostgreSQL container did not become ready")

        dsn = {
            "host": "127.0.0.1",
            "port": port,
            "user": "postgres",
            "password": "postgres",
            "dbname": "postgres",
        }
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE ROLE anon NOLOGIN;
                CREATE ROLE authenticated NOLOGIN;
                CREATE ROLE service_role NOLOGIN;
                CREATE TABLE public.research_lab_source_add_control (
                    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
                    paused BOOLEAN NOT NULL DEFAULT TRUE,
                    reason TEXT NOT NULL DEFAULT 'test_default',
                    actor_ref TEXT NOT NULL DEFAULT 'operator:test',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                INSERT INTO public.research_lab_source_add_control (singleton)
                VALUES (TRUE);
                CREATE TABLE public.research_lab_source_add_work_items (
                    work_id TEXT PRIMARY KEY,
                    submission_id TEXT NOT NULL,
                    adapter_id TEXT NOT NULL,
                    work_kind TEXT NOT NULL,
                    work_status TEXT NOT NULL,
                    job_doc JSONB NOT NULL DEFAULT '{}'::JSONB
                );
                """
            )
            migration_sql = MIGRATION.read_text(encoding="utf-8")
            cursor.execute(migration_sql)
            cursor.execute(migration_sql)
            cursor.execute(
                "SELECT public.research_lab_source_add_admission_control_contract_v1()"
            )
            contract = cursor.fetchone()[0]
            assert contract["control_row_present"] is True
            assert contract["trigger_enabled"] is True
        connection.close()
        yield psycopg2, dsn
    finally:
        if started:
            subprocess.run(
                [str(DOCKER), "rm", "--force", container],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )


def _insert_work(cursor, suffix: str, *, miner_admission: bool) -> None:
    job_doc = (
        '{"admission_kind":"miner_submission"}' if miner_admission else "{}"
    )
    cursor.execute(
        """
        INSERT INTO public.research_lab_source_add_work_items (
            work_id, submission_id, adapter_id, work_kind, work_status, job_doc
        ) VALUES (%s, %s, %s, 'provenance', 'queued', %s::JSONB)
        """,
        (
            "source_add_work:%s" % suffix,
            "source_add_submission:%s" % suffix,
            "adapter:%s" % suffix,
            job_doc,
        ),
    )


def test_paused_control_rejects_miner_admission_but_allows_recheck(postgres):
    psycopg2, dsn = postgres
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        with pytest.raises(psycopg2.Error) as rejected:
            _insert_work(cursor, "1111111111111111", miner_admission=True)
        assert rejected.value.pgcode == "55000"
        _insert_work(cursor, "2222222222222222", miner_admission=False)
        cursor.execute(
            "SELECT COUNT(*) FROM public.research_lab_source_add_work_items"
        )
        assert cursor.fetchone()[0] == 1
    connection.close()


def test_resume_and_pause_are_authoritative_for_direct_rpc_admission(postgres):
    psycopg2, dsn = postgres
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
            ("integration resume", "operator:integration"),
        )
        _insert_work(cursor, "3333333333333333", miner_admission=True)
        cursor.execute(
            "SELECT public.research_lab_source_add_set_paused(TRUE, %s, %s)",
            ("integration pause", "operator:integration"),
        )
        with pytest.raises(psycopg2.Error) as rejected:
            _insert_work(cursor, "4444444444444444", miner_admission=True)
        assert rejected.value.pgcode == "55000"
    connection.close()


def test_concurrent_pause_linearizes_before_waiting_admission(postgres):
    psycopg2, dsn = postgres
    setup = psycopg2.connect(**dsn)
    setup.autocommit = True
    with setup.cursor() as cursor:
        cursor.execute(
            "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
            ("concurrency resume", "operator:integration"),
        )
    setup.close()

    pause_connection = psycopg2.connect(**dsn)
    pause_connection.autocommit = False
    with pause_connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.research_lab_source_add_set_paused(TRUE, %s, %s)",
            ("concurrency pause", "operator:integration"),
        )

    started = threading.Event()
    finished = threading.Event()
    outcome: dict[str, str | None] = {"pgcode": None}

    def insert_while_pause_is_uncommitted() -> None:
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET statement_timeout = '10s'")
                started.set()
                _insert_work(cursor, "5555555555555555", miner_admission=True)
        except psycopg2.Error as exc:
            outcome["pgcode"] = exc.pgcode
        finally:
            connection.close()
            finished.set()

    worker = threading.Thread(target=insert_while_pause_is_uncommitted)
    worker.start()
    assert started.wait(timeout=5)
    time.sleep(0.2)
    assert not finished.is_set()
    pause_connection.commit()
    pause_connection.close()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert outcome["pgcode"] == "55000"


def test_missing_control_row_fails_closed(postgres):
    psycopg2, dsn = postgres
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM public.research_lab_source_add_control")
        with pytest.raises(psycopg2.Error) as rejected:
            _insert_work(cursor, "6666666666666666", miner_admission=True)
        assert rejected.value.pgcode == "55000"
        with pytest.raises(psycopg2.Error, match="control row is unavailable"):
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused(FALSE, %s, %s)",
                ("missing row", "operator:integration"),
            )
    connection.close()
