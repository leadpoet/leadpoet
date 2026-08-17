"""Execute the private-model activation generation guard on real PostgreSQL."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_GUARD = ROOT / "scripts" / "60-research-lab-one-active-version-guard.sql"
MIGRATION = ROOT / "scripts" / "153-research-lab-private-model-lineage-generation.sql"
DOCKER = shutil.which("docker")
pytestmark = pytest.mark.skipif(DOCKER is None, reason="Docker is unavailable")
PROTOCOL = "leadpoet.private-model-activation.v1"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture(scope="module")
def postgres():
    psycopg2 = pytest.importorskip("psycopg2")
    port = _free_port()
    container = "private-lineage-%s" % uuid4().hex[:12]
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
            pytest.skip(
                "PostgreSQL container could not start: %s" % result.stderr[-300:]
            )
        started = True
        dsn = {
            "host": "127.0.0.1",
            "port": port,
            "user": "postgres",
            "password": "postgres",
            "dbname": "postgres",
        }
        connection = None
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
                try:
                    connection = psycopg2.connect(**dsn)
                    break
                except psycopg2.OperationalError:
                    pass
            time.sleep(0.25)
        else:
            pytest.fail("PostgreSQL container did not become ready")
        assert connection is not None
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE ROLE anon NOLOGIN;
                CREATE ROLE authenticated NOLOGIN;
                CREATE ROLE service_role NOLOGIN;
                CREATE TABLE public.research_lab_private_model_versions (
                    private_model_version_id TEXT PRIMARY KEY
                );
                CREATE TABLE public.research_lab_private_model_version_events (
                    event_id UUID PRIMARY KEY,
                    private_model_version_id TEXT NOT NULL REFERENCES
                        public.research_lab_private_model_versions,
                    seq INTEGER NOT NULL CHECK (seq >= 0),
                    event_type TEXT NOT NULL,
                    version_status TEXT NOT NULL,
                    reason TEXT,
                    event_doc JSONB NOT NULL DEFAULT '{}'::JSONB,
                    anchored_hash TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (private_model_version_id, seq)
                );
                GRANT USAGE ON SCHEMA public TO service_role, anon, authenticated;
                GRANT SELECT, INSERT ON
                    public.research_lab_private_model_version_events
                    TO service_role;
                """
            )
            cursor.execute(BASE_GUARD.read_text(encoding="utf-8"))
            migration_sql = MIGRATION.read_text(encoding="utf-8")
            cursor.execute(migration_sql)
            cursor.execute(migration_sql)
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


@pytest.fixture(autouse=True)
def empty_lineage(postgres):
    psycopg2, dsn = postgres
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute("TRUNCATE public.research_lab_private_model_version_events")
        cursor.execute("TRUNCATE public.research_lab_private_model_versions CASCADE")
    connection.close()


def _version(cursor, label: str) -> str:
    version_id = "private_model_version:" + label
    cursor.execute(
        "INSERT INTO public.research_lab_private_model_versions VALUES (%s)",
        (version_id,),
    )
    return version_id


def _insert_event(
    cursor,
    *,
    version_id: str,
    seq: int,
    status: str,
    event_doc: object | None = None,
) -> None:
    identity = uuid4().hex
    cursor.execute(
        """
        INSERT INTO public.research_lab_private_model_version_events (
            event_id, private_model_version_id, seq, event_type,
            version_status, event_doc, anchored_hash
        ) VALUES (%s, %s, %s, %s, %s, %s::JSONB, %s)
        """,
        (
            str(uuid4()),
            version_id,
            seq,
            status,
            status,
            json.dumps({} if event_doc is None else event_doc),
            "sha256:" + identity.ljust(64, "0"),
        ),
    )


def _active_doc(generation: object) -> dict[str, object]:
    return {
        "activation_protocol_version": PROTOCOL,
        "expected_global_lineage_generation": generation,
    }


def test_rpc_is_invoker_only_and_denies_untrusted_roles(postgres) -> None:
    psycopg2, dsn = postgres
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT p.prosecdef,
                   has_function_privilege('anon', p.oid, 'EXECUTE'),
                   has_function_privilege('authenticated', p.oid, 'EXECUTE'),
                   has_function_privilege('service_role', p.oid, 'EXECUTE')
              FROM pg_catalog.pg_proc p
             WHERE p.oid =
                'public.research_lab_private_model_lineage_generation()'::regprocedure
            """
        )
        assert cursor.fetchone() == (False, False, False, True)
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT generation FROM public.research_lab_private_model_lineage_generation()"
        )
        assert cursor.fetchone()[0] == 0
        cursor.execute("RESET ROLE")
        cursor.execute("SET ROLE anon")
        with pytest.raises(psycopg2.Error) as denied:
            cursor.execute(
                "SELECT generation FROM public.research_lab_private_model_lineage_generation()"
            )
        assert denied.value.pgcode == "42501"
    connection.close()


@pytest.mark.parametrize(
    "event_doc,pgcode",
    [
        ({}, "23514"),
        ({"activation_protocol_version": PROTOCOL}, "23514"),
        (
            {
                "activation_protocol_version": (
                    "leadpoet.private-model-activation.v0"
                ),
                "expected_global_lineage_generation": 0,
            },
            "23514",
        ),
        (_active_doc("0"), "23514"),
        (_active_doc(True), "23514"),
        (_active_doc(None), "23514"),
        (_active_doc(-1), "23514"),
        (_active_doc(1.0), "23514"),
        (_active_doc(9223372036854775808), "22003"),
        (_active_doc(1), "40001"),
    ],
)
def test_active_generation_is_strict_and_excludes_new_row(
    postgres, event_doc, pgcode
) -> None:
    psycopg2, dsn = postgres
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        version_id = _version(cursor, uuid4().hex)
        with pytest.raises(psycopg2.Error) as rejected:
            _insert_event(
                cursor,
                version_id=version_id,
                seq=0,
                status="active",
                event_doc=event_doc,
            )
        assert rejected.value.pgcode == pgcode
        cursor.execute(
            "SELECT generation FROM public.research_lab_private_model_lineage_generation()"
        )
        assert cursor.fetchone()[0] == 0
    connection.close()


def test_exact_generation_accepts_and_increments_once(postgres) -> None:
    psycopg2, dsn = postgres
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        first = _version(cursor, "first")
        second = _version(cursor, "second")
        _insert_event(
            cursor,
            version_id=first,
            seq=0,
            status="active",
            event_doc=_active_doc(0),
        )
        _insert_event(
            cursor,
            version_id=first,
            seq=1,
            status="superseded",
            event_doc=_active_doc(1),
        )
        _insert_event(
            cursor,
            version_id=second,
            seq=0,
            status="active",
            event_doc=_active_doc(2),
        )
        cursor.execute(
            "SELECT generation FROM public.research_lab_private_model_lineage_generation()"
        )
        assert cursor.fetchone()[0] == 3
    connection.close()


@pytest.mark.parametrize(
    "status", ["bootstrap", "superseded", "failed", "tombstoned"]
)
def test_every_nonactive_status_uses_the_global_lock(postgres, status) -> None:
    psycopg2, dsn = postgres
    holder = psycopg2.connect(**dsn)
    holder.autocommit = False
    with holder.cursor() as cursor:
        held_version = _version(cursor, "held-" + status)
        _insert_event(
            cursor,
            version_id=held_version,
            seq=0,
            status=status,
            event_doc=_active_doc(0),
        )

    waiter = psycopg2.connect(**dsn)
    waiter.autocommit = True
    with waiter.cursor() as cursor:
        cursor.execute("SET statement_timeout = '250ms'")
        waiting_version = _version(cursor, "waiting-" + status)
        with pytest.raises(psycopg2.Error) as blocked:
            _insert_event(
                cursor,
                version_id=waiting_version,
                seq=0,
                status=status,
                event_doc=_active_doc(0),
            )
        assert blocked.value.pgcode == "57014"
    holder.commit()
    holder.close()
    with waiter.cursor() as cursor:
        cursor.execute("SET statement_timeout = 0")
        _insert_event(
            cursor,
            version_id=waiting_version,
            seq=0,
            status=status,
            event_doc=_active_doc(1),
        )
    waiter.close()


def test_concurrent_active_inserts_serialize_and_reject_stale_generation(
    postgres,
) -> None:
    psycopg2, dsn = postgres
    first = psycopg2.connect(**dsn)
    first.autocommit = False
    with first.cursor() as cursor:
        first_version = _version(cursor, "concurrent-first")
        _insert_event(
            cursor,
            version_id=first_version,
            seq=0,
            status="active",
            event_doc=_active_doc(0),
        )

    setup = psycopg2.connect(**dsn)
    setup.autocommit = True
    with setup.cursor() as cursor:
        second_version = _version(cursor, "concurrent-second")
    setup.close()
    started = threading.Event()
    finished = threading.Event()
    result: dict[str, str | None] = {"pgcode": None}

    def insert_second() -> None:
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET statement_timeout = '5s'")
                started.set()
                _insert_event(
                    cursor,
                    version_id=second_version,
                    seq=0,
                    status="active",
                    event_doc=_active_doc(0),
                )
        except psycopg2.Error as exc:
            result["pgcode"] = exc.pgcode
        finally:
            connection.close()
            finished.set()

    worker = threading.Thread(target=insert_second)
    worker.start()
    assert started.wait(timeout=5)
    time.sleep(0.2)
    assert not finished.is_set()
    first.commit()
    first.close()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert result["pgcode"] == "40001"


def test_markerless_supersede_is_rejected_and_leaves_active_model(postgres) -> None:
    psycopg2, dsn = postgres
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        version_id = _version(cursor, "markerless-supersede")
        _insert_event(
            cursor,
            version_id=version_id,
            seq=0,
            status="active",
            event_doc=_active_doc(0),
        )
        with pytest.raises(psycopg2.Error) as rejected:
            _insert_event(
                cursor,
                version_id=version_id,
                seq=1,
                status="superseded",
            )
        assert rejected.value.pgcode == "23514"
        cursor.execute(
            """
            SELECT version_status
              FROM public.research_lab_private_model_version_events
             WHERE private_model_version_id = %s
             ORDER BY seq DESC
             LIMIT 1
            """,
            (version_id,),
        )
        assert cursor.fetchone()[0] == "active"
        cursor.execute(
            "SELECT generation FROM "
            "public.research_lab_private_model_lineage_generation()"
        )
        assert cursor.fetchone()[0] == 1
    connection.close()


def test_supersede_rejects_stale_generation_and_accepts_exact_v1(postgres) -> None:
    psycopg2, dsn = postgres
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        version_id = _version(cursor, "exact-supersede")
        _insert_event(
            cursor,
            version_id=version_id,
            seq=0,
            status="active",
            event_doc=_active_doc(0),
        )
        with pytest.raises(psycopg2.Error) as rejected:
            _insert_event(
                cursor,
                version_id=version_id,
                seq=1,
                status="superseded",
                event_doc=_active_doc(0),
            )
        assert rejected.value.pgcode == "40001"
        _insert_event(
            cursor,
            version_id=version_id,
            seq=1,
            status="superseded",
            event_doc=_active_doc(1),
        )
        cursor.execute(
            """
            SELECT version_status
              FROM public.research_lab_private_model_version_events
             WHERE private_model_version_id = %s
             ORDER BY seq DESC
             LIMIT 1
            """,
            (version_id,),
        )
        assert cursor.fetchone()[0] == "superseded"
    connection.close()


def test_stale_zero_active_reconcile_cannot_cross_lineage_generations(
    postgres,
) -> None:
    psycopg2, dsn = postgres
    setup = psycopg2.connect(**dsn)
    setup.autocommit = True
    with setup.cursor() as cursor:
        stale_target = _version(cursor, "stale-target")
        intervening = _version(cursor, "intervening")
        _insert_event(
            cursor,
            version_id=stale_target,
            seq=0,
            status="superseded",
            event_doc=_active_doc(0),
        )
    setup.close()

    stale_reconcile = psycopg2.connect(**dsn)
    stale_reconcile.autocommit = True
    with stale_reconcile.cursor() as cursor:
        cursor.execute(
            "SELECT generation FROM "
            "public.research_lab_private_model_lineage_generation()"
        )
        stale_generation = cursor.fetchone()[0]
    assert stale_generation == 1

    newer_generation = psycopg2.connect(**dsn)
    newer_generation.autocommit = True
    with newer_generation.cursor() as cursor:
        _insert_event(
            cursor,
            version_id=intervening,
            seq=0,
            status="active",
            event_doc=_active_doc(1),
        )
        _insert_event(
            cursor,
            version_id=intervening,
            seq=1,
            status="superseded",
            event_doc=_active_doc(2),
        )
    newer_generation.close()

    with stale_reconcile.cursor() as cursor:
        with pytest.raises(psycopg2.Error) as rejected:
            _insert_event(
                cursor,
                version_id=stale_target,
                seq=1,
                status="active",
                event_doc=_active_doc(stale_generation),
            )
        assert rejected.value.pgcode == "40001"
        cursor.execute(
            """
            SELECT COUNT(*)
              FROM (
                SELECT DISTINCT ON (private_model_version_id)
                       private_model_version_id, version_status
                  FROM public.research_lab_private_model_version_events
                 ORDER BY private_model_version_id, seq DESC, created_at DESC
              ) latest
             WHERE version_status = 'active'
            """
        )
        assert cursor.fetchone()[0] == 0
    stale_reconcile.close()
