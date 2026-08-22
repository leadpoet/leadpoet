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
CLAIM_KEY = HASH_PREFIX + "c" * 64
BASELINE_DECISION = "routing_decision:" + "b" * 16
BASELINE_PROVIDER = "provider_receipt:" + "c" * 16
BASELINE_EVALUATION = "routing_evaluation_v2:" + "a" * 16


def _json_literal(value: object, tag: str) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True)
    marker = f"${tag}$"
    assert marker not in encoded
    return f"{marker}{encoded}{marker}::jsonb"


def _experiment_spec(*, variants: tuple[str, ...] = ("baseline", "candidate")) -> dict[str, object]:
    return {
        "input": {
            "stage": "candidate_acquisition",
            "target_verified_qualified_count": 1,
            "calibration_unit_refs": ["icp.cal"],
            "holdout_unit_refs": ["icp.hold"],
        },
        "baseline_variant_id": "baseline",
        "variants": [
            {
                "variant_id": variant,
                "binding_ids": ["binding.registry"],
                "stage": "candidate_acquisition",
                "artifact": {
                    "branch": "main" if variant == "baseline" else "leadpoet-lab",
                    "commit_sha": "m" * 40,
                    "manifest_hash": "n" * 64,
                },
            }
            for variant in variants
        ],
        "provider_bindings": [
            {
                "binding_id": "binding.registry",
                "tool_id": "candidate.registry_feed",
            }
        ],
    }


def _decision_doc(
    *,
    decision_receipt_id: str = BASELINE_DECISION,
    provider_receipt_refs: tuple[str, ...] = (BASELINE_PROVIDER,),
    variant_id: str = "baseline",
    unit_ref: str = "icp.cal",
    experiment_id: str = "exp-1",
) -> dict[str, object]:
    return {
        "receipt_id": decision_receipt_id,
        "experiment_id": experiment_id,
        "variant_id": variant_id,
        "artifact_key": HASH_PREFIX + "a" * 64,
        "stage": "candidate_acquisition",
        "unit_ref": unit_ref,
        "plan_hash": HASH_PREFIX + "3" * 64,
        "route_hash": HASH_PREFIX + "1" * 64,
        "attempted_tool_ids": (
            ["candidate.registry_feed"] if provider_receipt_refs else []
        ),
        "skipped_tool_reasons": (
            [] if provider_receipt_refs else [["candidate.registry_feed", "unavailable"]]
        ),
        "provider_receipt_refs": list(provider_receipt_refs),
        "execution_mode": "fixture",
    }


def _evaluation_doc(
    *,
    evaluation_receipt_id: str = BASELINE_EVALUATION,
    baseline_decisions: tuple[str, ...] = (BASELINE_DECISION,),
    baseline_providers: tuple[str, ...] = (BASELINE_PROVIDER,),
    candidate_decisions: tuple[str, ...] = (),
    candidate_providers: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "receipt_id": evaluation_receipt_id,
        "experiment_hash": EXPERIMENT_1,
        "variants": [
            {
                "variant_id": "baseline",
                "decision_receipt_refs": list(baseline_decisions),
                "provider_receipt_refs": list(baseline_providers),
            },
            {
                "variant_id": "candidate",
                "decision_receipt_refs": list(candidate_decisions),
                "provider_receipt_refs": list(candidate_providers),
            },
        ],
    }


def _insert_authority(
    psql,
    *,
    experiment_hash: str = EXPERIMENT_1,
    experiment_id: str = "exp-1",
    decision_receipt_id: str = BASELINE_DECISION,
    provider_receipt_ref: str = BASELINE_PROVIDER,
    evaluation_receipt_id: str | None = BASELINE_EVALUATION,
    variant_id: str = "baseline",
    unit_ref: str = "icp.cal",
    create_provider: bool = True,
) -> None:
    decision_doc = _decision_doc(
        decision_receipt_id=decision_receipt_id,
        provider_receipt_refs=((provider_receipt_ref,) if provider_receipt_ref else ()),
        variant_id=variant_id,
        unit_ref=unit_ref,
    )
    evaluation_doc = _evaluation_doc(
        evaluation_receipt_id=evaluation_receipt_id or BASELINE_EVALUATION,
        baseline_decisions=(decision_receipt_id,),
        baseline_providers=((provider_receipt_ref,) if provider_receipt_ref else ()),
    )
    evaluation_doc["experiment_hash"] = experiment_hash
    statements = [
        "INSERT INTO public.research_lab_routing_experiments_v2 "
        "(experiment_hash, experiment_id, spec_doc) VALUES "
        f"('{experiment_hash}','{experiment_id}',{_json_literal(_experiment_spec(), 'spec')}) "
        "ON CONFLICT (experiment_hash) DO NOTHING",
        "INSERT INTO public.research_lab_routing_decision_receipts_v2 "
        "(receipt_id, experiment_hash, variant_id, unit_ref, plan_hash, route_hash, decision_doc) VALUES "
        f"('{decision_receipt_id}','{experiment_hash}','{variant_id}','{unit_ref}',"
        f"'{HASH_PREFIX + '3' * 64}','{HASH_PREFIX + '1' * 64}',"
        f"{_json_literal(decision_doc, 'decision')})",
    ]
    if provider_receipt_ref and create_provider:
        statements.append(
            "INSERT INTO public.research_lab_routing_provider_attempts_v2 "
            "(provider_receipt_ref, experiment_hash, binding_id, tool_id, variant_id, unit_ref, "
            "outcome, billing_state, authoritative_billed_credit_microunits, latency_ms, "
            "execution_mode, attempt_doc) VALUES "
            f"('{provider_receipt_ref}','{experiment_hash}','binding.registry',"
            f"'candidate.registry_feed','{variant_id}','{unit_ref}','verified','known',25,250,'fixture',"
            f"{_json_literal({'provider_receipt': {'call_count': 1}}, 'attempt')})"
        )
    if evaluation_receipt_id:
        statements.append(
            "INSERT INTO public.research_lab_routing_evaluation_receipts_v2 "
            "(receipt_id, experiment_hash, evaluation_doc) VALUES "
            f"('{evaluation_receipt_id}','{experiment_hash}',"
            f"{_json_literal(evaluation_doc, 'evaluation')})"
        )
    psql(";".join(statements) + ";")


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

    def run_psql(
        sql: str,
        *,
        check: bool = True,
        tuples_only: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        output_args = ["-A", "-t"] if tuples_only else []
        return subprocess.run(
            [
                str(psql_bin),
                "--no-psqlrc",
                "-X",
                *output_args,
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
                experiment_hash TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                spec_doc JSONB NOT NULL
            );
            CREATE TABLE public.research_lab_routing_decision_receipts_v2 (
                receipt_id TEXT PRIMARY KEY,
                experiment_hash TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                unit_ref TEXT NOT NULL,
                plan_hash TEXT NOT NULL,
                route_hash TEXT NOT NULL,
                decision_doc JSONB NOT NULL
            );
            CREATE TABLE public.research_lab_routing_evaluation_receipts_v2 (
                receipt_id TEXT PRIMARY KEY,
                experiment_hash TEXT NOT NULL,
                evaluation_doc JSONB NOT NULL
            );
            CREATE TABLE public.research_lab_routing_provider_attempts_v2 (
                provider_receipt_ref TEXT PRIMARY KEY,
                experiment_hash TEXT NOT NULL,
                binding_id TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                unit_ref TEXT NOT NULL,
                outcome TEXT NOT NULL,
                billing_state TEXT NOT NULL,
                authoritative_billed_credit_microunits BIGINT,
                latency_ms BIGINT NOT NULL,
                execution_mode TEXT NOT NULL,
                attempt_doc JSONB NOT NULL
            );
            CREATE TABLE public.research_lab_routing_experiment_events_v2 (
                event_hash TEXT PRIMARY KEY,
                experiment_hash TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_doc JSONB NOT NULL
            );
            CREATE OR REPLACE FUNCTION public.research_lab_routing_reject_secret_doc_v2(
                JSONB, TEXT
            ) RETURNS VOID LANGUAGE sql AS 'SELECT NULL::VOID';
            CREATE OR REPLACE FUNCTION public.research_lab_routing_assert_claim_v3(
                TEXT, TEXT, BIGINT
            ) RETURNS VOID LANGUAGE sql AS 'SELECT NULL::VOID';
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
        try:
            run_psql(MIGRATION.read_text(encoding="utf-8"))
        except subprocess.CalledProcessError as exc:
            raise AssertionError(exc.stderr or exc.stdout) from exc
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
    decision_receipt_id: str = BASELINE_DECISION,
    provider_receipt_ref: str = BASELINE_PROVIDER,
    attempt_sequence: int = 0,
    variant_id: str = "baseline",
    unit_ref: str = "icp.cal",
) -> dict[str, object]:
    skipped = not provider_receipt_ref
    publication_projection_hash = "8" * 64
    terminal_publication_projection_hash = sha256_json(
        [publication_projection_hash]
    ).split(":", 1)[1]
    row: dict[str, object] = {
        "contract_version": "leadpoet.candidate_waterfall_receipt_sidecar:v1",
        "experiment_id": "exp-1",
        "experiment_hash": experiment_hash,
        "variant_id": variant_id,
        "artifact_key": HASH_PREFIX + "a" * 64,
        "decision_receipt_id": decision_receipt_id,
        "provider_receipt_ref": provider_receipt_ref,
        "unit_ref": unit_ref,
        "binding_id": "binding.registry",
        "tool_id": "candidate.registry_feed",
        "execution_mode": "fixture",
        "provider_outcome": "skipped" if skipped else "verified",
        "decision_plan_hash": HASH_PREFIX + "3" * 64,
        "decision_route_hash": HASH_PREFIX + "1" * 64,
        "model_contract_sha256": "2" * 64,
        "model_plan_sha256": "3" * 64,
        "stop_policy_sha256": "4" * 64,
        "publication_projection_sha256": publication_projection_hash,
        "attempt_receipt_sha256": ("5" * 63) + str(attempt_sequence),
        "prior_attempt_receipt_sha256": "",
        "attempt_chain_sha256": sha256_json(
            [(("5" * 63) + str(attempt_sequence))]
        ).split(":", 1)[1],
        "verification_receipt_sha256": "" if skipped else sha256_json(["7" * 64]).split(":", 1)[1],
        "company_verification_receipt_sha256s": [] if skipped else ["7" * 64],
        "step_order": 0,
        "attempt_sequence": attempt_sequence,
        "target_verified_qualified_count": 1,
        "disposition": "skipped" if skipped else "succeeded",
        "outcome_code": "unavailable" if skipped else "verified",
        "provider_call_count": 0 if skipped else 1,
        "billed_credit_microunits": 0 if skipped else 25,
        "latency_ms": 0 if skipped else 250,
        "raw_count": 0 if skipped else 1,
        "normalized_count": 0 if skipped else 1,
        "unique_count": 0 if skipped else 1,
        "verified_qualified_count": 0 if skipped else 1,
        "published_count": 0,
        "immutable": True,
    }
    row["step_order"] = attempt_sequence
    terminal_projection = {
        "tool_id": row["tool_id"],
        "outcome": row["provider_outcome"],
        "disposition": row["disposition"],
        "reason_code": row["outcome_code"],
        "provider_receipt_ref": row["provider_receipt_ref"],
        "provider_outcome": row["provider_outcome"],
        "raw_candidate_count": row["raw_count"],
        "normalized_candidate_count": row["normalized_count"],
        "unique_candidate_count": row["unique_count"],
        "verified_qualified_candidate_count": row["verified_qualified_count"],
        "company_verification_receipt_sha256s": row["company_verification_receipt_sha256s"],
        "verification_receipt_sha256": row["verification_receipt_sha256"],
        "publication_projection_sha256": publication_projection_hash,
        "published_count": 0,
        "provider_call_count": row["provider_call_count"],
        "billed_credit_microunits": row["billed_credit_microunits"],
        "latency_ms": row["latency_ms"],
        "candidate_plan_sha256": row["model_plan_sha256"],
        "stop_policy_sha256": row["stop_policy_sha256"],
        "attempt_sha256": row["attempt_receipt_sha256"],
        "prior_attempt_receipt_sha256": row["prior_attempt_receipt_sha256"],
        "attempt_chain_sha256": row["attempt_chain_sha256"],
        "step_order": row["step_order"],
        "attempt_sequence": row["attempt_sequence"],
    }
    terminal_identity = {
        "experiment_id": row["experiment_id"],
        "experiment_hash": row["experiment_hash"],
        "variant_id": row["variant_id"],
        "unit_ref": row["unit_ref"],
        "artifact_key": row["artifact_key"],
        "artifact_branch": "main" if row["variant_id"] == "baseline" else "leadpoet-lab",
        "artifact_commit_sha": "m" * 40,
        "artifact_manifest_hash": "n" * 64,
        "release_identity_sha256": HASH_PREFIX + "6" * 64,
        "binding_contracts_sha256": HASH_PREFIX + "b" * 64,
        "candidate_waterfall_contract_sha256": HASH_PREFIX + "a" * 64,
        "start_request_sha256": "9" * 64,
        "decision_receipt_id": row["decision_receipt_id"],
        "target_verified_qualified_count": 1,
        "disposition": row["disposition"],
        "stop_policy_sha256": row["stop_policy_sha256"],
        "verification_receipt_sha256": row["verification_receipt_sha256"],
        "attempt_chain_sha256": row["attempt_chain_sha256"],
        "publication_projection_sha256": terminal_publication_projection_hash,
        "terminal_result_sha256": HASH_PREFIX + "5" * 64,
        "model_receipt_sha256": "2" * 64,
        "orchestration_receipt_sha256": "3" * 64,
        "candidate_waterfall_sha256": "4" * 64,
        "candidate_plan_sha256": row["model_plan_sha256"],
        "stop_reason": "completed",
        "provider_receipt_refs": [row["provider_receipt_ref"]],
        "verification_receipt_refs": [row["company_verification_receipt_sha256s"]],
        "attempt_receipt_sha256s": [row["attempt_receipt_sha256"]],
        "attempt_chain_sha256s": [row["attempt_chain_sha256"]],
        "attempt_projections": [terminal_projection],
        "provider_call_count": row["provider_call_count"],
        "billed_credit_microunits": row["billed_credit_microunits"],
        "latency_ms": row["latency_ms"],
        "raw_count": row["raw_count"],
        "normalized_count": row["normalized_count"],
        "unique_count": row["unique_count"],
        "verified_qualified_count": row["verified_qualified_count"],
        "published_count": 0,
        "immutable": True,
        "contract_version": "leadpoet.candidate_model_unit_terminal_authority:v1",
    }
    terminal_hash = sha256_json(terminal_identity)
    terminal_doc = {
        **terminal_identity,
        "receipt_id": "candidate_model_terminal:" + terminal_hash[7:31],
        "receipt_hash": terminal_hash,
    }
    row["model_terminal_receipt_id"] = terminal_doc["receipt_id"]
    row["model_terminal_receipt_hash"] = terminal_doc["receipt_hash"]
    row["receipt_hash"] = sha256_json(row)
    row["receipt_id"] = "candidate_waterfall:" + str(row["receipt_hash"])[7:31]
    row["receipt_doc"] = {**row}
    row["terminal_doc"] = terminal_doc
    return row


def _insert_terminal(psql, row: dict[str, object], *, check: bool = True):
    terminal = row["terminal_doc"]
    assert isinstance(terminal, dict)
    columns = tuple(
        key
        for key in terminal
        if key not in {"immutable", "receipt_id", "receipt_hash"}
    ) + ("receipt_id", "receipt_hash", "terminal_doc")
    terminal_row = {**terminal, "terminal_doc": terminal}
    encoded = json.dumps(terminal_row, separators=(",", ":"))
    selected = ",".join(columns)
    result = psql(
        "INSERT INTO public.research_lab_candidate_model_unit_terminals (%s) "
        "SELECT %s FROM json_populate_record(NULL::public."
        "research_lab_candidate_model_unit_terminals, $terminal$%s$terminal$::json) "
        "ON CONFLICT (receipt_id) DO NOTHING;"
        % (selected, selected, encoded),
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr)
    return result


def _insert_receipt(
    psql,
    row: dict[str, object],
    *,
    check: bool = True,
):
    if "terminal_doc" in row:
        _insert_terminal(psql, row, check=False)
    columns = tuple(key for key in row if key not in {"immutable", "terminal_doc"})
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
        published_count=0,
        failed_attempt_count=0,
        missed_attempt_count=0,
        fulfillment_rate=1.0,
        verification_rate=1.0,
        publication_rate=0.0,
        verified_qualified_per_credit=40_000.0,
        waterfall_receipt_refs=(waterfall_receipt_ref,),
        provider_receipt_refs=(BASELINE_PROVIDER,),
        decision_receipt_refs=(BASELINE_DECISION,),
    )
    return {**metric.to_dict(), "metric_doc": metric.to_dict()}


def _insert_metric(
    psql,
    row: dict[str, object],
    *,
    check: bool = True,
):
    statement = _metric_insert_statement(row)
    result = psql(statement, check=False)
    if check and result.returncode != 0:
        raise AssertionError(result.stderr)
    return result


def _metric_insert_statement(row: dict[str, object]) -> str:
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
    return (
        "INSERT INTO public.research_lab_candidate_waterfall_metrics (%s) "
        "SELECT %s FROM json_populate_record(NULL::public."
        "research_lab_candidate_waterfall_metrics, $candidate$%s$candidate$::json);"
        % (selected, selected, encoded)
    )


def _rehash_receipt_doc(doc: dict[str, object]) -> dict[str, object]:
    identity = {
        key: value
        for key, value in doc.items()
        if key not in {"receipt_id", "receipt_hash", "receipt_doc"}
    }
    receipt_hash = sha256_json(identity)
    return {
        **identity,
        "receipt_id": "candidate_waterfall:" + receipt_hash[7:31],
        "receipt_hash": receipt_hash,
    }


def _append_receipt_rpc(psql, doc: dict[str, object], *, check: bool = True):
    terminal_exists = psql(
        "SELECT EXISTS (SELECT 1 FROM public.research_lab_candidate_model_unit_terminals "
        f"WHERE receipt_id = '{doc['model_terminal_receipt_id']}');",
        tuples_only=True,
    ).stdout.strip()
    if terminal_exists != "t":
        row = dict(doc)
        row["terminal_doc"] = _receipt_row(
            experiment_hash=str(doc["experiment_hash"]),
            decision_receipt_id=str(doc["decision_receipt_id"]),
            provider_receipt_ref=str(doc["provider_receipt_ref"]),
            variant_id=str(doc["variant_id"]),
            unit_ref=str(doc["unit_ref"]),
        )["terminal_doc"]
        _insert_terminal(psql, row, check=False)
    return psql(
        "SELECT public.research_lab_candidate_append_waterfall_receipt_v1("
        f"'{doc['receipt_id']}','{doc['receipt_hash']}','{doc['experiment_hash']}',"
        f"'{CLAIM_KEY}',1,{_json_literal(doc, 'receipt_rpc')});",
        check=check,
    )


def _authoritative_metric_doc(
    psql,
    *,
    variant_id: str = "baseline",
    split: str = "calibration",
) -> dict[str, object]:
    sql = f"""
        WITH identity AS (
            SELECT public.research_lab_candidate_metric_projection_v1(
                '{EXPERIMENT_1}', '{BASELINE_EVALUATION}',
                '{variant_id}', '{split}', 1
            ) AS doc
        ), hashed AS (
            SELECT doc, public.research_lab_routing_jsonb_hash_v2(doc) AS metric_hash
              FROM identity
        )
        SELECT (
            doc || pg_catalog.jsonb_build_object(
                'metric_hash', metric_hash,
                'metric_id', 'candidate_metric:' || pg_catalog.substr(metric_hash, 8, 24)
            )
        )::TEXT
          FROM hashed;
    """
    result = psql(sql, tuples_only=True)
    return json.loads(result.stdout.strip())


def _rehash_metric_doc(psql, identity: dict[str, object]) -> dict[str, object]:
    result = psql(
        "WITH identity AS (SELECT "
        f"{_json_literal(identity, 'metric_identity')} AS doc), "
        "hashed AS (SELECT doc, public.research_lab_routing_jsonb_hash_v2(doc) AS metric_hash "
        "FROM identity) SELECT (doc || pg_catalog.jsonb_build_object("
        "'metric_hash', metric_hash, 'metric_id', 'candidate_metric:' || "
        "pg_catalog.substr(metric_hash, 8, 24)))::TEXT FROM hashed;",
        tuples_only=True,
    )
    return json.loads(result.stdout.strip())


def _append_metric_rpc(psql, doc: dict[str, object], *, check: bool = True):
    return psql(
        "SELECT public.research_lab_candidate_append_waterfall_metric_v1("
        f"'{doc['metric_id']}','{doc['metric_hash']}','{doc['experiment_hash']}',"
        f"'{CLAIM_KEY}',1,{_json_literal(doc, 'metric_rpc')});",
        check=check,
    )


def test_migration_162_accepts_exact_row_and_rejects_mutation(
    postgres162,
):
    psql = postgres162
    _insert_authority(psql)
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
    _insert_authority(psql, evaluation_receipt_id=None)
    _insert_authority(
        psql,
        experiment_hash=EXPERIMENT_2,
        decision_receipt_id="routing_decision:" + "d" * 16,
        provider_receipt_ref="provider_receipt:" + "d" * 16,
        evaluation_receipt_id=None,
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

    duplicate_decision = "routing_decision:" + "8" * 16
    _insert_authority(
        psql,
        evaluation_receipt_id=None,
        decision_receipt_id=duplicate_decision,
        provider_receipt_ref=BASELINE_PROVIDER,
        unit_ref="icp.hold",
        create_provider=False,
    )
    duplicate = _receipt_row(
        decision_receipt_id=duplicate_decision,
        unit_ref="icp.hold",
    )
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
    _insert_authority(psql, evaluation_receipt_id=evaluation_1)
    _insert_authority(
        psql,
        experiment_hash=EXPERIMENT_2,
        decision_receipt_id="routing_decision:" + "d" * 16,
        provider_receipt_ref="provider_receipt:" + "d" * 16,
        evaluation_receipt_id=evaluation_2,
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


def test_migration_162_receipt_rpc_rejects_rehashed_parent_and_projection_drift(
    postgres162,
):
    psql = postgres162
    _insert_authority(psql)
    base = dict(_receipt_row()["receipt_doc"])
    mutations: tuple[tuple[str, object], ...] = (
        ("artifact_key", HASH_PREFIX + "9" * 64),
        ("decision_plan_hash", HASH_PREFIX + "8" * 64),
        ("decision_route_hash", HASH_PREFIX + "7" * 64),
        ("execution_mode", "replay"),
        ("provider_outcome", "source_miss"),
        ("provider_receipt_ref", "provider_receipt:" + "9" * 16),
        ("provider_call_count", 2),
        ("billed_credit_microunits", 26),
        ("latency_ms", 251),
        ("published_count", 1),
    )
    for field_name, forged_value in mutations:
        forged = _rehash_receipt_doc({**base, field_name: forged_value})
        result = _append_receipt_rpc(psql, forged, check=False)
        assert result.returncode != 0, field_name
        assert (
            "authoritative" in result.stderr.lower()
            or "terminal" in result.stderr.lower()
            or "check constraint" in result.stderr.lower()
        )

    reordered = _rehash_receipt_doc(
        {
            **base,
            "company_verification_receipt_sha256s": ["8" * 64, "7" * 64],
        }
    )
    reordered_result = _append_receipt_rpc(psql, reordered, check=False)
    assert reordered_result.returncode != 0
    assert "projection_not_authoritative" in reordered_result.stderr

    duplicate_verification_refs = ["7" * 64, "7" * 64]
    duplicated = _rehash_receipt_doc(
        {
            **base,
            "raw_count": 2,
            "normalized_count": 2,
            "unique_count": 2,
            "verified_qualified_count": 2,
            "company_verification_receipt_sha256s": duplicate_verification_refs,
            "verification_receipt_sha256": sha256_json(
                duplicate_verification_refs
            ).split(":", 1)[1],
        }
    )
    duplicated_result = _append_receipt_rpc(psql, duplicated, check=False)
    assert duplicated_result.returncode != 0
    assert "projection_not_authoritative" in duplicated_result.stderr

    first = _append_receipt_rpc(psql, base, check=False)
    assert first.returncode == 0, first.stderr
    replay = _append_receipt_rpc(psql, base, check=False)
    assert replay.returncode == 0, replay.stderr
    assert '"idempotent": false' in first.stdout.lower()
    assert '"idempotent": true' in replay.stdout.lower()


def test_migration_162_metric_rpc_recomputes_all_numbers_and_reference_sets(
    postgres162,
):
    psql = postgres162
    _insert_authority(psql)
    _append_receipt_rpc(psql, dict(_receipt_row()["receipt_doc"]))
    authoritative = _authoritative_metric_doc(psql)
    identity = {
        key: value
        for key, value in authoritative.items()
        if key not in {"metric_id", "metric_hash"}
    }
    mutations: tuple[tuple[str, object], ...] = (
        ("target_verified_qualified_count", 2),
        ("unit_count", 2),
        ("fulfilled_unit_count", 0),
        ("waterfall_attempt_count", 0),
        ("provider_call_count", 2),
        ("total_billed_credit_microunits", 26),
        ("total_latency_ms", 251),
        ("raw_count", 2),
        ("normalized_count", 0),
        ("unique_count", 0),
        ("verified_qualified_count", 0),
        ("published_count", 1),
        ("failed_attempt_count", 1),
        ("missed_attempt_count", 1),
        ("fulfillment_rate", 0.5),
        ("verification_rate", 0.5),
        ("publication_rate", 0.5),
        ("verified_qualified_per_credit", 1.0),
        ("waterfall_receipt_refs", []),
        ("provider_receipt_refs", []),
        ("decision_receipt_refs", []),
    )
    for field_name, forged_value in mutations:
        forged = _rehash_metric_doc(
            psql,
            {**identity, field_name: forged_value},
        )
        result = _append_metric_rpc(psql, forged, check=False)
        assert result.returncode != 0, field_name
        assert (
            "candidate_metric_projection" in result.stderr
            or "metric_not_authoritative" in result.stderr
            or "check constraint" in result.stderr.lower()
        ), field_name

    duplicate_refs = _rehash_metric_doc(
        psql,
        {
            **identity,
            "waterfall_receipt_refs": identity["waterfall_receipt_refs"] * 2,
        },
    )
    duplicate_result = _append_metric_rpc(psql, duplicate_refs, check=False)
    assert duplicate_result.returncode != 0

    first = _append_metric_rpc(psql, authoritative, check=False)
    assert first.returncode == 0, first.stderr
    replay = _append_metric_rpc(psql, authoritative, check=False)
    assert replay.returncode == 0, replay.stderr
    assert '"idempotent": false' in first.stdout.lower()
    assert '"idempotent": true' in replay.stdout.lower()


def test_migration_162_skipped_receipt_rpc_requires_zero_provider_authority(
    postgres162,
):
    psql = postgres162
    _insert_authority(psql, provider_receipt_ref="")
    _insert_terminal(psql, _receipt_row(provider_receipt_ref=""))
    base = dict(_receipt_row(provider_receipt_ref="")["receipt_doc"])
    skipped = _rehash_receipt_doc(
        {
            **base,
            "provider_receipt_ref": "",
            "provider_outcome": "skipped",
            "disposition": "skipped",
            "provider_call_count": 0,
            "billed_credit_microunits": 0,
            "latency_ms": 0,
            "raw_count": 0,
            "normalized_count": 0,
            "unique_count": 0,
            "verified_qualified_count": 0,
            "verification_receipt_sha256": "",
            "company_verification_receipt_sha256s": [],
        }
    )
    mutations: tuple[tuple[str, object], ...] = (
        ("provider_receipt_ref", BASELINE_PROVIDER),
        ("provider_outcome", "verified"),
        ("provider_call_count", 1),
        ("billed_credit_microunits", 1),
        ("latency_ms", 1),
        ("raw_count", 1),
    )
    for field_name, forged_value in mutations:
        forged = _rehash_receipt_doc({**skipped, field_name: forged_value})
        result = _append_receipt_rpc(psql, forged, check=False)
        assert result.returncode != 0, field_name
        assert (
            "authoritative" in result.stderr.lower()
            or "terminal" in result.stderr.lower()
            or "check constraint" in result.stderr.lower()
        )

    valid = _append_receipt_rpc(psql, skipped, check=False)
    assert valid.returncode == 0, valid.stderr


def test_migration_162_skipped_receipt_rejects_attempted_tool(
    postgres162,
):
    psql = postgres162
    _insert_authority(psql, provider_receipt_ref="", evaluation_receipt_id=None)
    skipped = _rehash_receipt_doc(
        {
            **_receipt_row(provider_receipt_ref="")["receipt_doc"],
            "provider_outcome": "skipped",
            "disposition": "skipped",
            "provider_call_count": 0,
            "billed_credit_microunits": 0,
            "latency_ms": 0,
            "raw_count": 0,
            "normalized_count": 0,
            "unique_count": 0,
            "verified_qualified_count": 0,
            "verification_receipt_sha256": "",
            "company_verification_receipt_sha256s": [],
        }
    )
    psql(
        "UPDATE public.research_lab_routing_decision_receipts_v2 "
        "SET decision_doc = jsonb_set(decision_doc, '{attempted_tool_ids}', "
        "'[\"candidate.registry_feed\"]'::jsonb) "
        f"WHERE receipt_id = '{BASELINE_DECISION}';"
    )
    attempted = _append_receipt_rpc(psql, skipped, check=False)
    assert attempted.returncode != 0
    assert "skipped_receipt_not_authoritative" in attempted.stderr


def test_migration_162_receipt_requires_exact_verification_reference_cardinality(
    postgres162,
):
    psql = postgres162
    _insert_authority(psql, evaluation_receipt_id=None)
    invalid = _rehash_receipt_doc(
        {
            **_receipt_row(),
            "verified_qualified_count": 2,
            "company_verification_receipt_sha256s": ["7" * 64],
        }
    )
    result = _append_receipt_rpc(psql, invalid, check=False)
    assert result.returncode != 0
    assert "projection_not_authoritative" in result.stderr


def test_migration_162_terminal_rejects_reused_verification_across_attempts(
    postgres162,
):
    psql = postgres162
    second_provider = "provider_receipt:" + "d" * 16
    _insert_authority(psql, evaluation_receipt_id=None)
    psql(
        "UPDATE public.research_lab_routing_decision_receipts_v2 "
        "SET decision_doc = jsonb_set(decision_doc, '{provider_receipt_refs}', "
        f"'[\"{BASELINE_PROVIDER}\",\"{second_provider}\"]'::jsonb) "
        f"WHERE receipt_id = '{BASELINE_DECISION}';"
        "INSERT INTO public.research_lab_routing_provider_attempts_v2 "
        "(provider_receipt_ref, experiment_hash, binding_id, tool_id, "
        "variant_id, unit_ref, outcome, billing_state, "
        "authoritative_billed_credit_microunits, latency_ms, execution_mode, "
        "attempt_doc) VALUES "
        f"('{second_provider}','{EXPERIMENT_1}','binding.registry',"
        "'candidate.registry_feed','baseline','icp.cal','verified','known',"
        "25,250,'fixture','{\"provider_receipt\":{\"call_count\":1}}'::jsonb);"
    )

    terminal = dict(_receipt_row()["terminal_doc"])
    first_projection = dict(terminal["attempt_projections"][0])
    second_attempt = "6" * 64
    second_chain = sha256_json(
        [terminal["attempt_receipt_sha256s"][0], second_attempt]
    ).split(":", 1)[1]
    second_publication = "a" * 64
    second_projection = {
        **first_projection,
        "provider_receipt_ref": second_provider,
        "attempt_sha256": second_attempt,
        "prior_attempt_receipt_sha256": terminal[
            "attempt_receipt_sha256s"
        ][0],
        "attempt_chain_sha256": second_chain,
        "publication_projection_sha256": second_publication,
        "step_order": 1,
        "attempt_sequence": 1,
    }
    verification_ref = terminal["verification_receipt_refs"][0]
    terminal_identity = {
        **{
            key: value
            for key, value in terminal.items()
            if key not in {"receipt_id", "receipt_hash"}
        },
        "provider_receipt_refs": [BASELINE_PROVIDER, second_provider],
        "verification_receipt_refs": [verification_ref, verification_ref],
        "attempt_receipt_sha256s": [
            terminal["attempt_receipt_sha256s"][0],
            second_attempt,
        ],
        "attempt_chain_sha256s": [
            terminal["attempt_chain_sha256s"][0],
            second_chain,
        ],
        "attempt_projections": [first_projection, second_projection],
        "verification_receipt_sha256": sha256_json(
            [verification_ref[0], verification_ref[0]]
        ).split(":", 1)[1],
        "attempt_chain_sha256": second_chain,
        "publication_projection_sha256": sha256_json(
            [
                first_projection["publication_projection_sha256"],
                second_publication,
            ]
        ).split(":", 1)[1],
        "provider_call_count": 2,
        "billed_credit_microunits": 50,
        "latency_ms": 500,
        "raw_count": 2,
        "normalized_count": 2,
        "unique_count": 2,
        "verified_qualified_count": 2,
    }
    terminal_hash = sha256_json(terminal_identity)
    terminal_doc = {
        **terminal_identity,
        "receipt_id": "candidate_model_terminal:" + terminal_hash[7:31],
        "receipt_hash": terminal_hash,
    }
    _insert_terminal(psql, {"terminal_doc": terminal_doc})

    result = psql(
        "SELECT public.research_lab_candidate_assert_model_unit_terminal_v1("
        f"'{EXPERIMENT_1}','baseline','icp.cal');",
        check=False,
    )
    assert result.returncode != 0
    assert "verification_receipt_duplicated" in result.stderr


def test_migration_162_recomputes_attempt_chain_and_requires_model_parent(
    postgres162,
):
    psql = postgres162
    _insert_authority(psql)
    valid = _receipt_row()
    _insert_terminal(psql, valid)
    forged = _rehash_receipt_doc(
        {**valid["receipt_doc"], "attempt_chain_sha256": "9" * 64}
    )
    forged_insert = _insert_receipt(
        psql, {**forged, "receipt_doc": {**forged}}, check=False
    )
    assert forged_insert.returncode == 0, forged_insert.stderr
    projection = psql(
        "SELECT public.research_lab_candidate_metric_projection_v1("
        f"'{EXPERIMENT_1}', '{BASELINE_EVALUATION}', 'baseline', 'calibration', 1);",
        check=False,
    )
    assert projection.returncode != 0
    assert "attempt_chain_invalid" in projection.stderr

    missing_parent = psql(
        "SELECT public.research_lab_candidate_assert_model_waterfall_authority_v1("
        f"'{EXPERIMENT_1}');",
        check=False,
    )
    assert missing_parent.returncode != 0
    assert "candidate_model_terminal_coverage_differs" in missing_parent.stderr


def test_migration_162_derives_target_and_rejects_empty_split_coverage(
    postgres162,
):
    psql = postgres162
    _insert_authority(psql)
    psql(
        "UPDATE public.research_lab_routing_experiments_v2 "
        "SET spec_doc = spec_doc #- '{input,target_verified_qualified_count}' "
        f"WHERE experiment_hash = '{EXPERIMENT_1}';"
    )
    missing_target = _append_receipt_rpc(
        psql, dict(_receipt_row()["receipt_doc"]), check=False
    )
    assert missing_target.returncode != 0
    assert "target_authority_missing" in missing_target.stderr

    psql(
        "UPDATE public.research_lab_routing_experiments_v2 "
        "SET spec_doc = jsonb_set(spec_doc, '{input,target_verified_qualified_count}', '1'::jsonb) "
        f"WHERE experiment_hash = '{EXPERIMENT_1}';"
    )
    empty_split = psql(
        "SELECT public.research_lab_candidate_metric_projection_v1("
        f"'{EXPERIMENT_1}', '{BASELINE_EVALUATION}', 'candidate', 'calibration', 1);",
        check=False,
    )
    assert empty_split.returncode != 0
    assert "receipt_coverage_missing" in empty_split.stderr


def test_migration_162_promotion_recomputes_stored_metrics_after_privileged_drift(
    postgres162,
):
    psql = postgres162
    authorities = (
        ("baseline", "icp.cal", BASELINE_DECISION, BASELINE_PROVIDER),
        ("baseline", "icp.hold", "routing_decision:" + "e" * 16, "provider_receipt:" + "d" * 16),
        ("candidate", "icp.cal", "routing_decision:" + "f" * 16, "provider_receipt:" + "e" * 16),
        ("candidate", "icp.hold", "routing_decision:" + "1" * 16, "provider_receipt:" + "2" * 16),
    )
    for variant_id, unit_ref, decision_id, provider_ref in authorities:
        _insert_authority(
            psql,
            evaluation_receipt_id=None,
            decision_receipt_id=decision_id,
            provider_receipt_ref=provider_ref,
            variant_id=variant_id,
            unit_ref=unit_ref,
        )
    evaluation_doc = _evaluation_doc(
        baseline_decisions=(BASELINE_DECISION, "routing_decision:" + "e" * 16),
        baseline_providers=(BASELINE_PROVIDER, "provider_receipt:" + "d" * 16),
        candidate_decisions=("routing_decision:" + "f" * 16, "routing_decision:" + "1" * 16),
        candidate_providers=("provider_receipt:" + "e" * 16, "provider_receipt:" + "2" * 16),
    )
    psql(
        "INSERT INTO public.research_lab_routing_evaluation_receipts_v2 "
        "(receipt_id, experiment_hash, evaluation_doc) VALUES "
        f"('{BASELINE_EVALUATION}','{EXPERIMENT_1}',"
        f"{_json_literal(evaluation_doc, 'complete_evaluation')});"
    )
    for variant_id, unit_ref, decision_id, provider_ref in authorities:
        row = _receipt_row(
            decision_receipt_id=decision_id,
            provider_receipt_ref=provider_ref,
            variant_id=variant_id,
            unit_ref=unit_ref,
        )
        _append_receipt_rpc(psql, dict(row["receipt_doc"]))
    metric_docs: dict[tuple[str, str], dict[str, object]] = {}
    for variant_id in ("baseline", "candidate"):
        for split in ("calibration", "holdout"):
            metric_doc = _authoritative_metric_doc(
                psql,
                variant_id=variant_id,
                split=split,
            )
            _append_metric_rpc(psql, metric_doc)
            metric_docs[(variant_id, split)] = metric_doc

    baseline = metric_docs[("baseline", "calibration")]
    forged_identity = {
        key: value
        for key, value in baseline.items()
        if key not in {"metric_id", "metric_hash"}
    }
    forged = _rehash_metric_doc(
        psql,
        {
            **forged_identity,
            "total_latency_ms": int(forged_identity["total_latency_ms"]) + 1,
        },
    )
    forged_row = {**forged, "metric_doc": forged}
    forged_event = psql(
        "BEGIN;"
        + _metric_insert_statement(forged_row)
        + "INSERT INTO public.research_lab_routing_experiment_events_v2 "
        "(event_hash, experiment_hash, event_type, event_doc) VALUES "
        f"('event-forged','{EXPERIMENT_1}','promoted',"
        f"{_json_literal({'evaluation_receipt_id': BASELINE_EVALUATION}, 'forged_event')});"
        "COMMIT;",
        check=False,
    )
    assert forged_event.returncode != 0
    assert "promotion_metric_not_authoritative" in forged_event.stderr

    valid_event = psql(
        "INSERT INTO public.research_lab_routing_experiment_events_v2 "
        "(event_hash, experiment_hash, event_type, event_doc) VALUES "
        f"('event-valid','{EXPERIMENT_1}','promoted',"
        f"{_json_literal({'evaluation_receipt_id': BASELINE_EVALUATION}, 'valid_event')});",
        check=False,
    )
    assert valid_event.returncode == 0, valid_event.stderr
