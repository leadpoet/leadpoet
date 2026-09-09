"""Shared disposable PostgreSQL fixture for migration contract tests."""
from __future__ import annotations
import shutil
import socket
import subprocess
import time
from pathlib import Path
from uuid import uuid4
import pytest
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DOCKER = shutil.which("docker")
HISTORICAL_SOURCE_ADD_UPGRADE_MIGRATIONS = (
    "72-research-lab-source-experiments.sql",
    "74-research-lab-source-add-provenance-precheck.sql",
    "78-research-lab-source-add-catalog-provisioning.sql",
    "79-research-lab-source-add-llm-leg2-evidence.sql",
    "82-research-lab-source-add-llm-only-leg2.sql",
    "84-expand-source-add-source-kinds.sql",
    "86-research-lab-attested-v2-authority.sql",
    "104-research-lab-attested-result-replay-v2.sql",
    "96-research-lab-source-add-functional-workflow.sql",
    "145-research-lab-source-add-admission-control.sql",
    "169-research-lab-source-add-post-accept-leg1.sql",
    "170-research-lab-source-add-provider-origin-uniqueness.sql",
    "171-research-lab-source-add-duplicate-privacy.sql",
    "172-research-lab-source-add-claim-control.sql",
    "173-research-lab-source-add-leg1-release-policy.sql",
    "174-research-lab-source-add-restart-state-restore.sql",
    "175-research-lab-source-add-provenance-leg1.sql",
    "176-research-lab-source-add-provenance-origin-repair.sql",
    "177-research-lab-source-add-provenance-authority-acl.sql",
    "178-research-lab-source-add-miner-status.sql",
    "186-research-lab-source-add-provisioned-status.sql",
)

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _database_with_migrations(migrations, *, setup_sql=""):
    psycopg2 = pytest.importorskip("psycopg2")
    port = _free_port()
    container = "migration-contract-%s" % uuid4().hex[:12]
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
        connect_deadline = time.monotonic() + 15
        while True:
            try:
                connection = psycopg2.connect(**dsn)
                break
            except psycopg2.OperationalError:
                if time.monotonic() >= connect_deadline:
                    raise
                time.sleep(0.25)
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE SCHEMA IF NOT EXISTS extensions;
                CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;
                CREATE ROLE anon NOLOGIN;
                CREATE ROLE authenticated NOLOGIN;
                CREATE ROLE service_role NOLOGIN;
                CREATE TABLE public.research_lab_auto_research_loop_events (
                    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    event_type TEXT NOT NULL,
                    CONSTRAINT research_lab_auto_research_loop_events_event_type_check
                        CHECK (event_type = 'loop_started')
                );
                """
            )
            if setup_sql:
                cursor.execute(setup_sql)
            for migration in migrations:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
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
