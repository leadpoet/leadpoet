"""Disposable-PostgreSQL contract for V3 durable weight-precompute storage."""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_RPCS,
    REQUIRED_SUPABASE_V2_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
SQL = (ROOT / "scripts" / "149-research-lab-weight-precompute-store.sql").read_text()


def _sha(seed: str) -> str:
    return "sha256:" + (seed * 64)[:64]


def _complete_receipt_map() -> str:
    names = (
        "research_lab_allocation", "champions", "reimbursements",
        "source_add_rewards", "fulfillment_rewards", "leaderboard", "bans",
        "sourcing_history", "anomaly_adjustments",
    )
    return "{" + ",".join('"%s":"%s"' % (name, _sha(str(index))) for index, name in enumerate(names)) + "}"


def test_weight_precompute_store_is_in_the_release_schema_gate() -> None:
    migration = "scripts/149-research-lab-weight-precompute-store.sql"
    assert {
        relation
        for declared, relation, _columns in REQUIRED_SUPABASE_V2_SCHEMA
        if declared == migration
    } == {
        "research_lab_weight_precompute_runs_v3",
        "research_lab_weight_precompute_input_sets_v3",
        "research_lab_weight_precompute_stage_events_v3",
        "research_lab_weight_precompute_run_current_v3",
    }
    assert {
        function
        for declared, function in REQUIRED_SUPABASE_V2_RPCS
        if declared == migration
    } == {
        "begin_research_lab_weight_precompute_run_v3",
        "record_research_lab_weight_precompute_input_set_v3",
        "append_research_lab_weight_precompute_stage_event_v3",
        "research_lab_weight_precompute_readback_v3",
        "research_lab_weight_precompute_store_contract_v3",
    }


def test_weight_precompute_store_is_append_only_service_only_and_idempotent() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for the PostgreSQL migration contract")
    if subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=15
    ).returncode != 0:
        pytest.skip("Docker daemon is unavailable")

    container = "leadpoet-weight-precompute-%s" % uuid.uuid4().hex[:12]

    def psql(statement: str, *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                "docker", "exec", "-i", container, "psql", "-X", "-A", "-t",
                "-U", "postgres", "-d", "leadpoet", "-v", "ON_ERROR_STOP=1",
            ],
            input=statement,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if expect_success:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0, result.stdout
        return result

    run_id = "10000000-0000-4000-8000-000000000001"
    event_id = "20000000-0000-4000-8000-000000000001"
    try:
        subprocess.run(
            [
                "docker", "run", "--detach", "--rm", "--name", container,
                "--env", "POSTGRES_PASSWORD=postgres", "--env", "POSTGRES_DB=leadpoet",
                "postgres:15",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        for _ in range(80):
            ready = subprocess.run(
                ["docker", "exec", container, "pg_isready", "-U", "postgres", "-d", "leadpoet"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        else:
            raise AssertionError("PostgreSQL 15 did not become ready")

        psql("CREATE ROLE anon NOLOGIN; CREATE ROLE authenticated NOLOGIN; CREATE ROLE service_role NOLOGIN;")
        psql(SQL)
        psql(SQL)  # The numbered migration is safe to apply again.

        contract = psql("SET ROLE service_role; SELECT public.research_lab_weight_precompute_store_contract_v3();")
        assert "leadpoet.weight_precompute_store.v3" in contract.stdout
        assert "leadpoet.published_weight_bundle.v2" in contract.stdout

        no_anon_read = psql(
            "SET ROLE anon; SELECT public.research_lab_weight_precompute_store_contract_v3();",
            expect_success=False,
        )
        assert "permission denied" in no_anon_read.stderr

        begin = """
SET ROLE service_role;
SELECT precompute_run_id FROM public.begin_research_lab_weight_precompute_run_v3(
    '%s', '%s', 71, 123, '%s', '%s', 456, '%s', '%s', '{"source":"predeadline"}'::jsonb
);
""" % (run_id, "0x" + "a" * 64, _sha("b"), _sha("c"), "c" * 40, _sha("d"))
        assert run_id in psql(begin).stdout
        assert run_id in psql(begin).stdout  # Exact replay returns the same run.

        conflicting_replay = psql(
            begin.replace(" 456,", " 457,"), expect_success=False
        )
        assert "research_lab_weight_precompute_run_replay_differs" in conflicting_replay.stderr

        input_set_hash = _sha("e")
        record_input = """
SET ROLE service_role;
SELECT precompute_run_id FROM public.record_research_lab_weight_precompute_input_set_v3(
    '%s', '%s', '%s', '%s', '%s'::jsonb,
    '{"complete":true}'::jsonb
);
""" % (run_id, input_set_hash, _sha("f"), _sha("1"), _complete_receipt_map())
        assert run_id in psql(record_input).stdout
        assert run_id in psql(record_input).stdout
        incomplete_input_set = psql(
            "SET ROLE service_role; SELECT public.research_lab_weight_precompute_complete_input_set_v3("
            "'{\"chain_state\":\"%s\"}'::jsonb);" % _sha("4")
        )
        assert incomplete_input_set.stdout.strip().endswith("f")

        append_event = """
SET ROLE service_role;
SELECT stage_event_id FROM public.append_research_lab_weight_precompute_stage_event_v3(
    '%s', '%s', 0, '%s', 'input_set_complete', 'succeeded', '%s',
    '{"readback":"ready"}'::jsonb
);
""" % (event_id, run_id, input_set_hash, _sha("3"))
        assert event_id in psql(append_event).stdout
        assert event_id in psql(append_event).stdout

        readback = psql(
            "SET ROLE service_role; SELECT public.research_lab_weight_precompute_readback_v3('%s');" % run_id
        )
        assert '"complete_input_set"' in readback.stdout
        assert '"input_set_complete"' in readback.stdout
        assert input_set_hash in readback.stdout

        no_anon_rpc = psql(
            "SET ROLE anon; SELECT public.research_lab_weight_precompute_readback_v3('%s');" % run_id,
            expect_success=False,
        )
        assert "permission denied" in no_anon_rpc.stderr
        no_service_mutation = psql(
            "SET ROLE service_role; UPDATE public.research_lab_weight_precompute_runs_v3 SET netuid = 1;",
            expect_success=False,
        )
        assert "permission denied" in no_service_mutation.stderr

        # Disable RLS only in this disposable database to prove the immutable
        # trigger also protects a table owner with direct mutation rights.
        immutable = psql(
            "ALTER TABLE public.research_lab_weight_precompute_runs_v3 DISABLE ROW LEVEL SECURITY; "
            "UPDATE public.research_lab_weight_precompute_runs_v3 SET netuid = 1;",
            expect_success=False,
        )
        assert "research_lab_weight_precompute_append_only" in immutable.stderr
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            capture_output=True,
            text=True,
            timeout=30,
        )
