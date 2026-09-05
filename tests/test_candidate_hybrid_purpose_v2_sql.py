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
)
from leadpoet_canonical.attested_v2 import ROLE_PURPOSES
from tests.historical_sql_purpose_contract import (
    canonical_purposes_before_routing_experiment_v2,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_NAME = "scripts/152-research-lab-candidate-hybrid-purposes.sql"
MIGRATION = ROOT / MIGRATION_NAME
SQL = MIGRATION.read_text(encoding="utf-8")
UPGRADE_MIGRATION_NAME = "scripts/154-research-lab-model-compatibility-purpose.sql"
UPGRADE_SQL = (ROOT / UPGRADE_MIGRATION_NAME).read_text(encoding="utf-8")
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
    assert (UPGRADE_MIGRATION_NAME, CONTRACT_RPC) not in REQUIRED_SUPABASE_V2_RPCS

    for role, expected_purposes in ROLE_PURPOSES.items():
        match = re.search(
            rf"role = '{re.escape(role)}' AND purpose IN \((.*?)\n\s*\)\)",
            SQL,
            re.DOTALL,
        )
        assert match is not None, role
        historical_purposes = canonical_purposes_before_routing_experiment_v2(
            role
        )
        if role == "gateway_scoring":
            historical_purposes.difference_update(
                {
                    "research_lab.model_compatibility.v2",
                    "research_lab.routing_experiment.v2",
                    "research_lab.routing_model_binding_observation.v2",
                    "research_lab.routing_provider_evidence.v2",
                }
            )
        assert set(re.findall(r"'([^']+)'", match.group(1))) == historical_purposes

    assert re.search(
        r"REVOKE\s+ALL\s+ON\s+FUNCTION[\s\S]+?"
        r"research_lab_candidate_hybrid_purpose_contract_v1\(\)"
        r"[\s\S]+?FROM\s+PUBLIC,\s*anon,\s*authenticated",
        SQL,
        re.IGNORECASE,
    )


def test_model_compatibility_purpose_upgrade_matches_canonical_contract() -> None:
    assert re.search(r"\bBEGIN\s*;", UPGRADE_SQL)
    assert re.search(r"\bCOMMIT\s*;\s*$", UPGRADE_SQL)
    assert "SET LOCAL lock_timeout = '5s'" in UPGRADE_SQL
    assert "NOT VALID" in UPGRADE_SQL
    assert "VALIDATE CONSTRAINT" in UPGRADE_SQL

    for role, expected_purposes in ROLE_PURPOSES.items():
        match = re.search(
            rf"role = '{re.escape(role)}' AND purpose IN \((.*?)\n\s*\)",
            UPGRADE_SQL,
            re.DOTALL,
        )
        assert match is not None, role
        historical_purposes = canonical_purposes_before_routing_experiment_v2(
            role
        )
        if role == "gateway_scoring":
            historical_purposes.add("research_lab.model_compatibility.v2")
        assert set(re.findall(r"'([^']+)'", match.group(1))) == historical_purposes
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
        psql(UPGRADE_SQL)
        psql(UPGRADE_SQL)

        expected_historical_purposes = {
            role: canonical_purposes_before_routing_experiment_v2(role)
            for role in ROLE_PURPOSES
        }
        expected_historical_purposes["gateway_scoring"].add(
            "research_lab.model_compatibility.v2"
        )
        values = ",".join(
            "(%s,%s)" % (_sql_text(role), _sql_text(purpose))
            for role, purposes in expected_historical_purposes.items()
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
        historical_clauses = re.findall(
            r"\(role = '([^']+)'::text\)\s+AND\s+"
            r"\(purpose = ANY \(ARRAY\[(.*?)\]\)\)",
            contract["constraint_definition"],
            flags=re.DOTALL,
        )
        historical_pairs = {
            role: frozenset(
                re.findall(r"'([^']+)'::text", encoded_purposes)
            )
            for role, encoded_purposes in historical_clauses
        }
        assert historical_pairs == {
            role: frozenset(purposes)
            for role, purposes in expected_historical_purposes.items()
        }
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            capture_output=True,
            text=True,
            timeout=15,
        )
