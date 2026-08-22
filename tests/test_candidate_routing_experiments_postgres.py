"""Native PostgreSQL checks for migration 162.

The test is skipped when a local PostgreSQL server is not installed. It never
uses Docker or a configured application database.
"""

from __future__ import annotations

import json
import getpass
from pathlib import Path
import shutil
import socket as socket_module
import subprocess
import tempfile

import pytest

from research_lab.canonical import sha256_json
from research_lab.candidate_routing_experiments import CandidateWaterfallMetric


MIGRATION = (
    Path(__file__).parents[1]
    / "scripts"
    / "162-research-lab-candidate-routing-experiments.sql"
)
HASH_PREFIX = "sha256:"
EXPERIMENT_1 = HASH_PREFIX + "e" * 64
EXPERIMENT_2 = HASH_PREFIX + "f" * 64


def _postgres_bin(name: str) -> Path | None:
    candidates = []
    candidates.extend(
        Path(root) / "bin" / name
        for root in (
            "/opt/homebrew/opt/postgresql@17",
            "/usr/local/opt/postgresql@17",
        )
    )
    executable = shutil.which(name)
    if executable:
        candidates.append(Path(executable))
    for candidate in candidates:
        path = candidate.resolve()
        if not path.exists():
            continue
        if name in {"initdb", "pg_ctl"} and not (path.parent / "postgres").exists():
            continue
        return path
    return None


def _free_tcp_port() -> str:
    with socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return str(listener.getsockname()[1])


@pytest.fixture()
def postgres162():
    initdb = _postgres_bin("initdb")
    pg_ctl = _postgres_bin("pg_ctl")
    psql_bin = _postgres_bin("psql")
    if initdb is None or pg_ctl is None or psql_bin is None:
        pytest.skip("native PostgreSQL server binaries are unavailable")
    temp_root = Path(tempfile.mkdtemp(prefix="lab162-pg.", dir="/private/tmp"))
    data = temp_root / "data"
    socket = temp_root / "socket"
    log = temp_root / "postgres.log"
    socket.mkdir()
    port = _free_tcp_port()
    db_user = getpass.getuser()
    try:
        subprocess.run(
            [
                str(initdb),
                "-D",
                str(data),
                "--auth=trust",
                "--no-locale",
                "--encoding=UTF8",
                "--set=shared_memory_type=mmap",
                "--set=dynamic_shared_memory_type=mmap",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(
            [
                str(pg_ctl),
                "-D",
                str(data),
                "-l",
                str(log),
                "-o",
                f"-k {socket} -p {port} -c listen_addresses=''"
                " -c shared_memory_type=mmap -c dynamic_shared_memory_type=mmap",
                "start",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        try:
            stop_result = subprocess.run(
                [str(pg_ctl), "-D", str(data), "-w", "stop"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            stop_result = None
        if stop_result is not None and stop_result.returncode == 0:
            shutil.rmtree(temp_root, ignore_errors=True)
        detail = str(getattr(exc, "stderr", "") or "").strip().splitlines()
        pytest.skip(
            "native PostgreSQL could not start: "
            f"{type(exc).__name__}: {detail[-1] if detail else 'no stderr'}"
        )

    def run_psql(sql: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(psql_bin),
                "--no-psqlrc",
                "-X",
                "--host",
                str(socket),
                "--port",
                port,
                "--username",
                db_user,
                "--dbname",
                "postgres",
                "--set",
                "ON_ERROR_STOP=1",
            ],
            input=sql,
            check=check,
            capture_output=True,
            text=True,
            timeout=30,
        )

    try:
        run_psql(
            """
            CREATE SCHEMA extensions;
            CREATE EXTENSION IF NOT EXISTS pgcrypto SCHEMA extensions;
            CREATE ROLE anon NOLOGIN;
            CREATE ROLE authenticated NOLOGIN;
            CREATE ROLE service_role NOLOGIN;
            CREATE TABLE public.research_lab_routing_experiments_v2 (
                experiment_hash TEXT PRIMARY KEY
            );
            CREATE TABLE public.research_lab_routing_decision_receipts_v2 (
                receipt_id TEXT PRIMARY KEY,
                experiment_hash TEXT NOT NULL
            );
            CREATE TABLE public.research_lab_routing_evaluation_receipts_v2 (
                receipt_id TEXT PRIMARY KEY,
                experiment_hash TEXT NOT NULL
            );
            """
        )
        # These are the exact canonical hash functions supplied by migration
        # 157, reduced to their function definitions for this isolated test.
        run_psql(
            """
            CREATE OR REPLACE FUNCTION public.research_lab_routing_canonical_jsonb_v2(
                p_value JSONB
            ) RETURNS TEXT
            LANGUAGE plpgsql IMMUTABLE STRICT
            SET search_path = pg_catalog, public
            AS $canonical_json$
            BEGIN
                CASE pg_catalog.jsonb_typeof(p_value)
                    WHEN 'object' THEN
                        RETURN (
                            SELECT '{' || coalesce(
                                pg_catalog.string_agg(
                                    pg_catalog.to_jsonb(entry.key)::TEXT || ':' ||
                                    public.research_lab_routing_canonical_jsonb_v2(entry.value),
                                    ',' ORDER BY entry.key COLLATE "C"
                                ), ''
                            ) || '}'
                            FROM pg_catalog.jsonb_each(p_value) AS entry(key, value)
                        );
                    WHEN 'array' THEN
                        RETURN (
                            SELECT '[' || coalesce(
                                pg_catalog.string_agg(
                                    public.research_lab_routing_canonical_jsonb_v2(entry.value),
                                    ',' ORDER BY entry.ordinality
                                ), ''
                            ) || ']'
                            FROM pg_catalog.jsonb_array_elements(p_value)
                                WITH ORDINALITY AS entry(value, ordinality)
                        );
                    ELSE
                        RETURN p_value::TEXT;
                END CASE;
            END;
            $canonical_json$;
            CREATE OR REPLACE FUNCTION public.research_lab_routing_jsonb_hash_v2(
                p_value JSONB
            ) RETURNS TEXT
            LANGUAGE sql IMMUTABLE STRICT
            SET search_path = pg_catalog, public
            AS $jsonb_hash$
                SELECT 'sha256:' || pg_catalog.encode(
                    extensions.digest(
                        pg_catalog.convert_to(
                            public.research_lab_routing_canonical_jsonb_v2(p_value),
                            'UTF8'
                        ),
                        'sha256'
                    ),
                    'hex'
                )
            $jsonb_hash$;
            """
        )
        run_psql(MIGRATION.read_text(encoding="utf-8"))
        yield run_psql
    finally:
        stop_result = subprocess.run(
            [str(pg_ctl), "-D", str(data), "-w", "stop"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if stop_result.returncode != 0:
            raise RuntimeError(
                "native PostgreSQL did not stop cleanly: "
                + (stop_result.stderr or stop_result.stdout).strip()
            )
        shutil.rmtree(temp_root, ignore_errors=True)


def _receipt_row(
    *,
    experiment_hash: str = EXPERIMENT_1,
    decision_receipt_id: str = "routing_decision:" + "b" * 16,
    provider_receipt_ref: str = "provider_receipt:" + "c" * 16,
    attempt_sequence: int = 0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "contract_version": "leadpoet.candidate_waterfall_receipt_sidecar:v1",
        "experiment_id": "exp-1",
        "experiment_hash": experiment_hash,
        "variant_id": "baseline",
        "artifact_key": HASH_PREFIX + "a" * 64,
        "decision_receipt_id": decision_receipt_id,
        "provider_receipt_ref": provider_receipt_ref,
        "unit_ref": "icp.cal",
        "binding_id": "binding.registry",
        "tool_id": "candidate.registry_feed",
        "execution_mode": "fixture",
        "provider_outcome": "verified",
        "decision_plan_hash": HASH_PREFIX + "d" * 64,
        "decision_route_hash": HASH_PREFIX + "1" * 64,
        "model_contract_sha256": "2" * 64,
        "model_plan_sha256": "3" * 64,
        "stop_policy_sha256": "4" * 64,
        "attempt_receipt_sha256": ("5" * 63) + str(attempt_sequence),
        "prior_attempt_receipt_sha256": "",
        "attempt_chain_sha256": "6" * 64,
        "verification_receipt_sha256": "7" * 64,
        "step_order": 0,
        "attempt_sequence": attempt_sequence,
        "target_verified_qualified_count": 1,
        "disposition": "succeeded",
        "outcome_code": "verified",
        "provider_call_count": 1,
        "billed_credit_microunits": 25,
        "latency_ms": 250,
        "raw_count": 1,
        "normalized_count": 1,
        "unique_count": 1,
        "verified_qualified_count": 1,
        "published_count": 1,
        "immutable": True,
    }
    row["receipt_hash"] = sha256_json(row)
    row["receipt_id"] = "candidate_waterfall:" + str(row["receipt_hash"])[7:31]
    row["receipt_doc"] = {**row}
    return row


def _insert_receipt(
    psql,
    row: dict[str, object],
    *,
    check: bool = True,
):
    columns = tuple(key for key in row if key != "immutable")
    encoded = json.dumps(row, separators=(",", ":"))
    assert "$candidate$" not in encoded
    selected = ",".join(columns)
    result = psql(
        "INSERT INTO public.research_lab_candidate_waterfall_receipts (%s) "
        "SELECT %s FROM json_populate_record(NULL::public."
        "research_lab_candidate_waterfall_receipts, $candidate$%s$candidate$::json);"
        % (selected, selected, encoded),
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr)
    return result


def _metric_row(
    *,
    experiment_hash: str = EXPERIMENT_1,
    evaluation_receipt_id: str = "routing_evaluation_v2:" + "a" * 16,
    variant_id: str = "baseline",
    waterfall_receipt_ref: str = "candidate_waterfall:" + "b" * 24,
) -> dict[str, object]:
    metric = CandidateWaterfallMetric(
        evaluation_receipt_id=evaluation_receipt_id,
        experiment_id="exp-1",
        experiment_hash=experiment_hash,
        variant_id=variant_id,
        split="calibration",
        target_verified_qualified_count=1,
        unit_count=1,
        fulfilled_unit_count=1,
        waterfall_attempt_count=1,
        provider_call_count=1,
        total_billed_credit_microunits=25,
        total_latency_ms=250,
        raw_count=1,
        normalized_count=1,
        unique_count=1,
        verified_qualified_count=1,
        published_count=1,
        failed_attempt_count=0,
        missed_attempt_count=0,
        fulfillment_rate=1.0,
        verification_rate=1.0,
        publication_rate=1.0,
        verified_qualified_per_credit=40_000.0,
        waterfall_receipt_refs=(waterfall_receipt_ref,),
        provider_receipt_refs=("provider_receipt:" + "c" * 16,),
        decision_receipt_refs=("routing_decision:" + "b" * 16,),
    )
    return {**metric.to_dict(), "metric_doc": metric.to_dict()}


def _insert_metric(
    psql,
    row: dict[str, object],
    *,
    check: bool = True,
):
    columns = tuple(
        key
        for key in row
        if key
        not in {
            "metric_doc",
            "immutable",
            "waterfall_receipt_refs",
            "provider_receipt_refs",
            "decision_receipt_refs",
        }
    )
    encoded = json.dumps(row, separators=(",", ":"))
    assert "$candidate$" not in encoded
    selected = ",".join(columns + ("metric_doc",))
    result = psql(
        "INSERT INTO public.research_lab_candidate_waterfall_metrics (%s) "
        "SELECT %s FROM json_populate_record(NULL::public."
        "research_lab_candidate_waterfall_metrics, $candidate$%s$candidate$::json);"
        % (selected, selected, encoded),
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr)
    return result


def test_migration_162_accepts_exact_row_and_rejects_mutation(
    postgres162,
):
    psql = postgres162
    psql(
        "INSERT INTO public.research_lab_routing_experiments_v2 VALUES ('%s'); "
        "INSERT INTO public.research_lab_routing_decision_receipts_v2 VALUES ('%s','%s');"
        % (EXPERIMENT_1, "routing_decision:" + "b" * 16, EXPERIMENT_1)
    )
    row = _receipt_row()
    _insert_receipt(psql, row)
    mutated = psql(
        "UPDATE public.research_lab_candidate_waterfall_receipts "
        "SET published_count = 0;",
        check=False,
    )
    assert mutated.returncode != 0
    assert "append-only" in mutated.stderr


def test_migration_162_rejects_forged_hash_duplicate_provider_and_cross_lineage(
    postgres162,
):
    psql = postgres162
    psql(
        "INSERT INTO public.research_lab_routing_experiments_v2 VALUES ('%s'), ('%s'); "
        "INSERT INTO public.research_lab_routing_decision_receipts_v2 VALUES "
        "('%s','%s'), ('%s','%s');"
        % (
            EXPERIMENT_1,
            EXPERIMENT_2,
            "routing_decision:" + "b" * 16,
            EXPERIMENT_1,
            "routing_decision:" + "d" * 16,
            EXPERIMENT_2,
        )
    )
    row = _receipt_row()
    _insert_receipt(psql, row)

    forged = dict(row)
    forged["receipt_hash"] = HASH_PREFIX + "9" * 64
    forged["receipt_id"] = "candidate_waterfall:" + "9" * 24
    forged["receipt_doc"] = {**forged}
    forged_result = _insert_receipt(psql, forged, check=False)
    assert forged_result.returncode != 0
    assert "check constraint" in forged_result.stderr.lower()

    duplicate = _receipt_row(attempt_sequence=1)
    _insert_duplicate_result = _insert_receipt(psql, duplicate, check=False)
    assert _insert_duplicate_result.returncode != 0
    assert "idx_research_lab_candidate_waterfall_provider_receipt" in _insert_duplicate_result.stderr

    cross_lineage = _receipt_row(
        decision_receipt_id="routing_decision:" + "d" * 16,
        provider_receipt_ref="provider_receipt:" + "d" * 16,
        attempt_sequence=1,
    )
    cross_result = _insert_receipt(psql, cross_lineage, check=False)
    assert cross_result.returncode != 0
    assert "foreign key" in cross_result.stderr.lower()


def test_migration_162_accepts_python_metric_and_rejects_metric_integrity_and_lineage(
    postgres162,
):
    psql = postgres162
    evaluation_1 = "routing_evaluation_v2:" + "a" * 16
    evaluation_2 = "routing_evaluation_v2:" + "b" * 16
    psql(
        "INSERT INTO public.research_lab_routing_experiments_v2 VALUES ('%s'), ('%s'); "
        "INSERT INTO public.research_lab_routing_decision_receipts_v2 VALUES "
        "('%s','%s'); "
        "INSERT INTO public.research_lab_routing_evaluation_receipts_v2 VALUES "
        "('%s','%s'), ('%s','%s');"
        % (
            EXPERIMENT_1,
            EXPERIMENT_2,
            "routing_decision:" + "b" * 16,
            EXPERIMENT_1,
            evaluation_1,
            EXPERIMENT_1,
            evaluation_2,
            EXPERIMENT_2,
        )
    )
    receipt = _receipt_row()
    _insert_receipt(psql, receipt)
    # Rebuild the Python metric after binding its exact receipt reference. This
    # proves the database canonical JSONB hash agrees with the producer hash,
    # including the floating-point metric fields.
    metric = _metric_row(
        evaluation_receipt_id=evaluation_1,
        waterfall_receipt_ref=str(receipt["receipt_id"]),
    )
    _insert_metric(psql, metric)

    forged_hash = dict(metric)
    forged_hash["metric_hash"] = HASH_PREFIX + "9" * 64
    forged_hash["metric_id"] = "candidate_metric:" + "9" * 24
    forged_hash["metric_doc"] = {
        **metric["metric_doc"],
        "metric_hash": forged_hash["metric_hash"],
        "metric_id": forged_hash["metric_id"],
    }
    forged_hash_result = _insert_metric(psql, forged_hash, check=False)
    assert forged_hash_result.returncode != 0
    assert "check constraint" in forged_hash_result.stderr.lower()

    forged_id = _metric_row(
        evaluation_receipt_id=evaluation_1,
        variant_id="candidate",
        waterfall_receipt_ref=str(receipt["receipt_id"]),
    )
    forged_id["metric_id"] = "candidate_metric:" + "9" * 24
    forged_id["metric_doc"] = {
        **forged_id["metric_doc"],
        "metric_id": forged_id["metric_id"],
    }
    forged_id_result = _insert_metric(psql, forged_id, check=False)
    assert forged_id_result.returncode != 0
    assert "check constraint" in forged_id_result.stderr.lower()

    cross_lineage = _metric_row(
        experiment_hash=EXPERIMENT_1,
        evaluation_receipt_id=evaluation_2,
        variant_id="candidate",
    )
    cross_result = _insert_metric(psql, cross_lineage, check=False)
    assert cross_result.returncode != 0
    assert "foreign key" in cross_result.stderr.lower()
