from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import pytest

from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_RPCS,
    _role_purpose_pairs_from_constraint_v1,
)
from leadpoet_canonical.attested_v2 import ROLE_PURPOSES


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_NAME = "scripts/152-research-lab-candidate-hybrid-purposes.sql"
MIGRATION = ROOT / MIGRATION_NAME
SQL = MIGRATION.read_text(encoding="utf-8")
CONTRACT_RPC = "research_lab_candidate_hybrid_purpose_contract_v1"


def _sql_text(value: str) -> str:
    return "'%s'" % value.replace("'", "''")


def test_candidate_hybrid_purpose_migration_matches_canonical_contract() -> None:
    assert re.search(r"\bBEGIN\s*;", SQL)
    assert re.search(r"\bCOMMIT\s*;\s*$", SQL)
    assert "SET LOCAL lock_timeout = '5s'" in SQL
    assert "NOT VALID" in SQL
    assert "VALIDATE CONSTRAINT" in SQL
    assert not re.search(
        r"^\s*(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+",
        SQL,
        re.MULTILINE,
    )
    assert (MIGRATION_NAME, CONTRACT_RPC) in REQUIRED_SUPABASE_V2_RPCS

    for role, expected_purposes in ROLE_PURPOSES.items():
        match = re.search(
            rf"role = '{re.escape(role)}' AND purpose IN \((.*?)\n\s*\)\)",
            SQL,
            re.DOTALL,
        )
        assert match is not None, role
        assert set(re.findall(r"'([^']+)'", match.group(1))) == set(
            expected_purposes
        )

    assert re.search(
        r"REVOKE\s+ALL\s+ON\s+FUNCTION[\s\S]+?"
        r"research_lab_candidate_hybrid_purpose_contract_v1\(\)"
        r"[\s\S]+?FROM\s+PUBLIC,\s*anon,\s*authenticated",
        SQL,
        re.IGNORECASE,
    )
    assert re.search(
        r"GRANT\s+EXECUTE\s+ON\s+FUNCTION[\s\S]+?"
        r"research_lab_candidate_hybrid_purpose_contract_v1\(\)"
        r"[\s\S]+?TO\s+service_role",
        SQL,
        re.IGNORECASE,
    )


def test_candidate_hybrid_purpose_migration_is_idempotent_and_fail_closed() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for the PostgreSQL contract test")
    info = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=15
    )
    if info.returncode != 0:
        pytest.skip("Docker daemon is unavailable")

    container = "leadpoet-hybrid-purpose-%s" % uuid.uuid4().hex[:12]

    def psql(statement: str, *, expect_success: bool = True):
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container,
                "psql",
                "-X",
                "-A",
                "-t",
                "-U",
                "postgres",
                "-d",
                "leadpoet",
                "-v",
                "ON_ERROR_STOP=1",
            ],
            input=statement,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if expect_success and result.returncode != 0:
            raise AssertionError(result.stderr)
        if not expect_success and result.returncode == 0:
            raise AssertionError("statement unexpectedly succeeded")
        return result

    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--name",
                container,
                "--env",
                "POSTGRES_PASSWORD=postgres",
                "--env",
                "POSTGRES_DB=leadpoet",
                "postgres:15-alpine",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            ready = subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "psql",
                    "-X",
                    "-qAt",
                    "-U",
                    "postgres",
                    "-d",
                    "leadpoet",
                    "-c",
                    "SELECT 1",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        else:
            raise AssertionError("PostgreSQL did not become ready")

        psql(
            """
            CREATE ROLE anon;
            CREATE ROLE authenticated;
            CREATE ROLE service_role;
            CREATE TABLE public.research_lab_attested_execution_receipts_v2 (
                role TEXT NOT NULL,
                purpose TEXT NOT NULL
            );
            """
        )
        psql(SQL)
        psql(SQL)

        values = ",".join(
            "(%s,%s)" % (_sql_text(role), _sql_text(purpose))
            for role, purposes in ROLE_PURPOSES.items()
            for purpose in sorted(purposes)
        )
        psql(
            "BEGIN; INSERT INTO "
            "public.research_lab_attested_execution_receipts_v2(role,purpose) "
            f"VALUES {values}; ROLLBACK;"
        )
        psql(
            "INSERT INTO public.research_lab_attested_execution_receipts_v2"
            "(role,purpose) VALUES "
            "('gateway_autoresearch','research_lab.candidate_hybrid_test.v2');",
            expect_success=False,
        )
        psql(
            "INSERT INTO public.research_lab_attested_execution_receipts_v2"
            "(role,purpose) VALUES "
            "('gateway_scoring','research_lab.unknown_hybrid.v2');",
            expect_success=False,
        )

        encoded = psql(
            "SELECT public.research_lab_candidate_hybrid_purpose_contract_v1();"
        ).stdout.strip()
        contract = json.loads(encoded)
        assert contract["constraint_valid"] is True
        assert contract["constraint_name"] == (
            "research_lab_attested_execution_receipts_v2_role_purpose_check"
        )
        assert _role_purpose_pairs_from_constraint_v1(
            contract["constraint_definition"]
        ) == {
            role: frozenset(purposes)
            for role, purposes in ROLE_PURPOSES.items()
        }
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            capture_output=True,
            text=True,
            timeout=15,
        )
