"""Exact recovery283 terminal fixture; sealed same-round recovery284."""
from __future__ import annotations

import hashlib
import json
import re
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import arena_sep17_baseline_recovery284 as operator
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.sep17_baseline_recovery278_postgres_test import _insert_rows
from tests.lab_arena.sep17_baseline_recovery283_postgres_test import (
    restore_terminal as restore_recovery283_terminal,
)


ROOT = Path(__file__).parents[2]
PRIVATE = Path("/private/tmp/leadpoet-tyche-rerun-evidence-20260915/private")
SNAPSHOT = PRIVATE / "sep17-recovery284-terminal-snapshot-20260917T120911.884492Z.json"
SNAPSHOT_SHA = "958107b0543404fefacd1d3dcb0deddb1ab4738d3106606cfe677827f5c5669f"
BANK_SNAPSHOT = PRIVATE / "sep17-original-bank.json"
MIGRATION_283 = ROOT / "scripts/283-arena-2026-09-17-baseline-recovery.sql"
MIGRATION = ROOT / "scripts/284-arena-2026-09-17-baseline-recovery.sql"
TEMPLATE = Path(str(MIGRATION) + ".template")
ROUND = operator.ROUND
BASELINE = operator.BASELINE
ARCHIVE = "arena-2026-09-17-rerun283archive"
ARCHIVE_BASELINE = "baseline-2026-09-17-rerun283archive"
pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in (SNAPSHOT, BANK_SNAPSHOT)),
    reason=(
        "Recovery284 requires the protected exact recovery283 terminal snapshot "
        "and frozen ICP bank; release gates require zero skips."
    ),
)


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def captured():
    assert stat.S_IMODE(SNAPSHOT.stat().st_mode) == 0o600
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == SNAPSHOT_SHA
    receipt = json.loads(SNAPSHOT.read_text())
    assert receipt["schema_version"] == "leadpoet.sep17_recovery284_terminal_snapshot.v1"
    assert receipt["read_only"] is True and receipt["production_writes"] is False
    assert receipt["captured"]["gate"]["usable"] is True
    assert receipt["expected_source"] == {
        "source_ref": operator.TERMINAL_SOURCE_REF,
        "source_size_bytes": operator.TERMINAL_SOURCE_SIZE,
        "source_sha256": operator.TERMINAL_SOURCE_SHA256,
        "source_commit": operator.TERMINAL_SOURCE_COMMIT,
    }
    protected = receipt["captured"]["protected_rows"]
    assert {key: len(protected[key]) for key in (
        "rounds", "submissions", "runs", "ledger",
    )} == {"rounds": 5, "submissions": 15, "runs": 121, "ledger": 18272}
    assert all(isinstance(protected[key], str) and protected[key] for key in (
        "authority278", "audit278", "authority282", "audit282",
        "authority283", "audit283",
    ))
    assert receipt["captured"]["proof"]["baseline_runs"]["count"] == 22
    return receipt["captured"]


def replacements(snapshot):
    proof = snapshot["proof"]
    round_doc = proof["round"]
    baseline = proof["baseline"]
    runs = proof["baseline_runs"]
    ledger = proof["baseline_ledger"]
    submissions = proof["submissions"]
    nonbaseline_submissions = proof["nonbaseline_submissions"]
    nonbaseline_ledger = proof["nonbaseline_ledger"]
    prior = proof["prior283"]
    return {
        "__RECOVERY_SOURCE_SIZE_BYTES__": str(operator.SOURCE_SIZE),
        "__RECOVERY_SOURCE_SHA256__": operator.SOURCE_SHA256,
        "__RECOVERY_SOURCE_COMMIT__": operator.SOURCE_COMMIT,
        "__TERMINAL_ROUND_HASH__": round_doc["round_hash"],
        "__TERMINAL_BASELINE_SUBMISSION_HASH__": baseline["hash"],
        "__TERMINAL_BASELINE_RUNS_HASH__": runs["hash"],
        "__TERMINAL_BASELINE_LEDGER_HASH__": ledger["hash"],
        "__TERMINAL_NONBASELINE_LEDGER_HASH__": nonbaseline_ledger["hash"],
        "__TERMINAL_SUBMISSIONS_HASH__": submissions["hash"],
        "__TERMINAL_NONBASELINE_SUBMISSIONS_HASH__": nonbaseline_submissions["hash"],
        "__TERMINAL_BASELINE_RUN_COUNT__": str(runs["count"]),
        "__TERMINAL_BASELINE_LEDGER_COUNT__": str(ledger["count"]),
        "__TERMINAL_NONBASELINE_LEDGER_COUNT__": str(nonbaseline_ledger["count"]),
        "__TERMINAL_NONBASELINE_LEDGER_MAX_ENTRY_ID__": str(nonbaseline_ledger["max_entry_id"]),
        "__TERMINAL_BASELINE_LEDGER_MAX_ENTRY_ID__": str(ledger["max_entry_id"]),
        "__TERMINAL_BASELINE_SETTLED_MICROUSD__": str(ledger["settled_microusd"]),
        "__TERMINAL_BASELINE_UNCERTAIN_MICROUSD__": str(ledger["uncertain_microusd"]),
        "__TERMINAL_SUBMISSION_COUNT__": str(submissions["count"]),
        "__TERMINAL_NONBASELINE_SUBMISSION_COUNT__": str(nonbaseline_submissions["count"]),
        "__TERMINAL_EXECUTE_COST_JSON__": compact(proof["execute_cost"]),
        "__TERMINAL_SCORE_COST_JSON__": compact(proof["score_cost"]),
        "__TERMINAL_STATUS_GENERATION__": str(round_doc["status_generation"]),
        "__TERMINAL_STAGE_GENERATION__": str(round_doc["stage_generation"]),
        "__TERMINAL_CANCEL_REASON__": round_doc["cancel_reason"],
        "__OLD_SCHEDULE_JSON__": compact(round_doc["schedule"]),
        "__FORWARD_SCHEDULE_JSON__": compact(operator.FORWARD_SCHEDULE),
        "__PRIOR283_AUTHORITY_HASH__": prior["authority_hash"],
        "__PRIOR283_AUDIT_HASH__": prior["audit_hash"],
    }


def test_template_matches_operator_and_exact_protected_terminal():
    snapshot = captured()
    rendered = TEMPLATE.read_text()
    for token, value in replacements(snapshot).items():
        assert token in rendered
        rendered = rendered.replace(token, value)
    assert re.search(r"__[A-Z0-9_]+__", rendered) is None
    assert rendered == MIGRATION.read_text()


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def restore_terminal(connection):
    snapshot = captured()
    restore_recovery283_terminal(connection)
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION_283.read_text())
    connection.commit()
    rows = snapshot["protected_rows"]
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_sep17_baseline_recovery283_audit,"
            "public.lab_arena_sep17_baseline_recovery283_authority,"
            "public.lab_arena_sep17_baseline_recovery282_audit,"
            "public.lab_arena_sep17_baseline_recovery282_authority,"
            "public.lab_arena_sep17_baseline_recovery278_audit,"
            "public.lab_arena_sep17_baseline_recovery278_authority,"
            "public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
        for table, key in (
            ("lab_arena_rounds", "rounds"),
            ("lab_arena_submissions", "submissions"),
            ("lab_arena_runs", "runs"),
            ("lab_arena_ledger", "ledger"),
            ("lab_arena_sep17_baseline_recovery278_authority", "authority278"),
            ("lab_arena_sep17_baseline_recovery278_audit", "audit278"),
            ("lab_arena_sep17_baseline_recovery282_authority", "authority282"),
            ("lab_arena_sep17_baseline_recovery282_audit", "audit282"),
            ("lab_arena_sep17_baseline_recovery283_authority", "authority283"),
            ("lab_arena_sep17_baseline_recovery283_audit", "audit283"),
        ):
            raw = rows[key]
            documents = [json.loads(item) for item in (
                raw if isinstance(raw, list) else [raw]
            )]
            for start in range(0, len(documents), 250):
                _insert_rows(cursor, table, documents[start:start + 250])
        cursor.execute(
            "SELECT setval('public.lab_arena_ledger_entry_id_seq',"
            "(SELECT max(entry_id) FROM public.lab_arena_ledger),true)"
        )
        cursor.execute("SET session_replication_role=origin")
        cursor.execute(
            "SELECT public.lab_arena_sep17_recovery278_archive_valid_v1(),"
            "public.lab_arena_sep17_recovery278_nonbaseline_ledger_valid_v1(),"
            "public.lab_arena_sep17_recovery282_archive_valid_v1(),"
            "public.lab_arena_sep17_recovery282_nonbaseline_ledger_valid_v1(),"
            "public.lab_arena_sep17_recovery283_archive_valid_v1(),"
            "public.lab_arena_sep17_recovery283_nonbaseline_ledger_valid_v1()"
        )
        assert cursor.fetchone() == (True, True, True, True, True, True)
    connection.commit()
    return snapshot


def table_hash(cursor, table, where, order):
    cursor.execute(
        "SELECT 'sha256:'||encode(extensions.digest(coalesce(string_agg("
        "encode(extensions.digest(to_jsonb(x)::text,'sha256'),'hex'),'' ORDER BY "
        + order + "),''),'sha256'),'hex') FROM public." + table + " x WHERE " + where
    )
    return cursor.fetchone()[0]


def preserved_hashes(connection):
    with connection.cursor() as cursor:
        return {
            "archive_rounds": table_hash(
                cursor, "lab_arena_rounds",
                "round_id IN ('arena-2026-09-17-archive','arena-2026-09-17-rerun278archive',"
                "'arena-2026-09-17-rerun282archive')",
                "round_id",
            ),
            "archive_submissions": table_hash(
                cursor, "lab_arena_submissions",
                "round_id IN ('arena-2026-09-17-archive','arena-2026-09-17-rerun278archive',"
                "'arena-2026-09-17-rerun282archive')",
                "round_id||':'||submission_id",
            ),
            "archive_runs": table_hash(
                cursor, "lab_arena_runs",
                "round_id IN ('arena-2026-09-17-archive','arena-2026-09-17-rerun278archive',"
                "'arena-2026-09-17-rerun282archive')",
                "round_id||':'||run_id",
            ),
            "archive_ledger": table_hash(
                cursor, "lab_arena_ledger",
                "round_id IN ('arena-2026-09-17-archive','arena-2026-09-17-rerun278archive',"
                "'arena-2026-09-17-rerun282archive')",
                "entry_id",
            ),
            "sep18_round": table_hash(cursor, "lab_arena_rounds", "round_id='arena-2026-09-18'", "round_id"),
            "sep18_submissions": table_hash(cursor, "lab_arena_submissions", "round_id='arena-2026-09-18'", "submission_id"),
            "sep18_runs": table_hash(cursor, "lab_arena_runs", "round_id='arena-2026-09-18'", "run_id"),
            "sep18_ledger": table_hash(cursor, "lab_arena_ledger", "round_id='arena-2026-09-18'", "entry_id"),
            "authority278": table_hash(cursor, "lab_arena_sep17_baseline_recovery278_authority", "TRUE", "round_id"),
            "audit278": table_hash(cursor, "lab_arena_sep17_baseline_recovery278_audit", "TRUE", "round_id"),
            "authority282": table_hash(cursor, "lab_arena_sep17_baseline_recovery282_authority", "TRUE", "round_id"),
            "audit282": table_hash(cursor, "lab_arena_sep17_baseline_recovery282_audit", "TRUE", "round_id"),
            "authority283": table_hash(cursor, "lab_arena_sep17_baseline_recovery283_authority", "TRUE", "round_id"),
            "audit283": table_hash(cursor, "lab_arena_sep17_baseline_recovery283_audit", "TRUE", "round_id"),
            "miners": table_hash(
                cursor, "lab_arena_submissions",
                "round_id='arena-2026-09-17' AND submission_id<>'baseline-2026-09-17'",
                "submission_id",
            ),
            "miner_ledger": table_hash(
                cursor, "lab_arena_ledger",
                "round_id='arena-2026-09-17' AND submission_id<>'baseline-2026-09-17'",
                "entry_id",
            ),
        }


def prepare(cursor, **overrides):
    args = {
        "size": operator.SOURCE_SIZE,
        "sha": operator.SOURCE_SHA256,
        "commit": operator.SOURCE_COMMIT,
        "bank": operator.BANK_SHA256,
        "schedule": operator.FORWARD_SCHEDULE,
    }
    args.update(overrides)
    cursor.execute(
        "SELECT public.lab_arena_prepare_sep17_baseline_recovery284_v1("
        "%s,%s,%s,%s,%s::jsonb)",
        (args["size"], args["sha"], args["commit"], args["bank"],
         json.dumps(args["schedule"])),
    )
    return cursor.fetchone()[0]


def test_sealed_prepare_preserves_history_and_scores_positive(
    database, tmp_path, monkeypatch,
):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        snapshot = restore_terminal(connection)
        before = preserved_hashes(connection)
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
        connection.commit()
        assert preserved_hashes(connection) == before

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT has_function_privilege('lab_arena_service',"
                "'public.lab_arena_prepare_sep17_baseline_recovery284_v1(bigint,text,text,text,jsonb)',"
                "'EXECUTE'),has_function_privilege('service_role',"
                "'public.lab_arena_prepare_sep17_baseline_recovery284_v1(bigint,text,text,text,jsonb)',"
                "'EXECUTE'),has_table_privilege('lab_arena_service',"
                "'public.lab_arena_sep17_baseline_recovery284_authority','SELECT')"
            )
            assert cursor.fetchone() == (True, False, False)
            cursor.execute(
                "SELECT count(*) FROM pg_catalog.pg_trigger WHERE tgname="
                "'lab_arena_sep17_recovery284_archive_guard' AND NOT tgisinternal"
            )
            assert cursor.fetchone() == (1,)

        with pytest.raises(Exception):
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE public.lab_arena_sep17_baseline_recovery284_authority "
                    "SET execute_namespace='unsealed' WHERE round_id=%s", (ROUND,)
                )
        connection.rollback()
        assert preserved_hashes(connection) == before

        # Source, active-state, cost, miner, and prior-audit drift must fail atomically.
        with pytest.raises(Exception):
            with connection.cursor() as cursor:
                prepare(cursor, sha="0" * 64)
        connection.rollback()
        for statement in (
            "UPDATE public.lab_arena_runs SET status='pending' "
            "WHERE round_id='arena-2026-09-17' AND icp_position=0",
            "UPDATE public.lab_arena_submissions SET created_at=created_at+interval '1 second' "
            "WHERE round_id='arena-2026-09-17' AND submission_id<>'baseline-2026-09-17'",
            "UPDATE public.lab_arena_sep17_baseline_recovery283_audit "
            "SET started_at=started_at+interval '1 second'",
        ):
            with pytest.raises(Exception):
                with connection.cursor() as cursor:
                    cursor.execute("SET LOCAL session_replication_role=replica")
                    cursor.execute(statement)
                    cursor.execute("SET LOCAL session_replication_role=origin")
                    prepare(cursor)
            connection.rollback()
            assert preserved_hashes(connection) == before

        with pytest.raises(Exception):
            with connection.cursor() as cursor:
                cursor.execute(
                    "ALTER FUNCTION public.lab_arena__successful_call_cost_state(text,text,text) "
                    "RENAME TO recovery284_test_original_cost_state"
                )
                cursor.execute(
                    "CREATE FUNCTION public.lab_arena__successful_call_cost_state("
                    "p_submission_id text,p_kind text,p_provider text DEFAULT NULL) RETURNS jsonb "
                    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$ "
                    "SELECT public.recovery284_test_original_cost_state("
                    "p_submission_id,p_kind,p_provider)||jsonb_build_object('inflight_calls',1) $$"
                )
                prepare(cursor)
        connection.rollback()
        assert preserved_hashes(connection) == before

        with connection.cursor() as cursor:
            result = prepare(cursor)
            assert result["status"] == "prepared"
            assert result["archived_runs"] == snapshot["proof"]["baseline_runs"]["count"]
            assert result["archived_ledger_entries"] == snapshot["proof"]["baseline_ledger"]["count"]
            assert result["preserved_nonparticipant_submissions"] == 8
            assert prepare(cursor)["status"] == "existing"
        connection.commit()

        assert preserved_hashes(connection) == before
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),max(attempt),"
                "count(output_ref),count(*) FILTER (WHERE status='pending') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20, 1, 0, 20)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_ledger "
                "WHERE round_id=%s AND submission_id=%s", (ROUND, BASELINE)
            )
            assert cursor.fetchone() == (0,)
            cursor.execute(
                "SELECT count(*),count(*) FILTER (WHERE status='accepted' AND output_ref IS NOT NULL) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s",
                (ARCHIVE, ARCHIVE_BASELINE),
            )
            assert cursor.fetchone() == (
                snapshot["proof"]["baseline_runs"]["count"], 8,
            )
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_ledger "
                "WHERE round_id=%s AND submission_id=%s", (ARCHIVE, ARCHIVE_BASELINE)
            )
            assert cursor.fetchone() == (snapshot["proof"]["baseline_ledger"]["count"],)
            cursor.execute(
                "SELECT public.lab_arena_sep17_recovery284_prior_valid_v1(),"
                "public.lab_arena_sep17_recovery284_archive_valid_v1(),"
                "public.lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1()"
            )
            assert cursor.fetchone() == (True, True, True)

        # The archive trigger must roll back a corrupted archive and the update
        # that exposed it. This exercises the installed trigger, not only the RPC.
        with pytest.raises(Exception, match="archive seal differs"):
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL session_replication_role=replica")
                cursor.execute(
                    "UPDATE public.lab_arena_runs SET output_ref=output_ref||'-corrupt' "
                    "WHERE round_id=%s AND submission_id=%s AND run_id=("
                    "SELECT min(run_id) FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND submission_id=%s "
                    "AND output_ref IS NOT NULL)",
                    (ARCHIVE, ARCHIVE_BASELINE, ARCHIVE, ARCHIVE_BASELINE),
                )
                cursor.execute("SET LOCAL session_replication_role=origin")
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET updated_at=updated_at "
                    "WHERE round_id=%s", (ROUND,)
                )
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SELECT public.lab_arena_sep17_recovery284_archive_valid_v1()")
            assert cursor.fetchone() == (True,)

        from lab_arena import contracts, scoring
        from tests.lab_arena.sep16_native_baseline_rerun_postgres_test import (
            _proof_breakdown, _proof_company, _proof_execution,
        )
        from tests.lab_arena.test_lab_arena_service_round import Harness

        harness = Harness(lambda: psycopg2.connect(**dsn), tmp_path,
                          challengers=[], runners=["recovery284-proof"])
        service, objects = harness.service, harness.objects
        harness.clock.now = datetime(2026, 9, 17, 16, tzinfo=timezone.utc)
        bank = json.loads(BANK_SNAPSHOT.read_text())
        assert hashlib.sha256(
            contracts.canonical_json(bank["icps"]).encode()
        ).hexdigest() == operator.BANK_SHA256
        objects.put(operator.BANK_REF, json.dumps(bank).encode())
        icps = bank["icps"]
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "SELECT run_id,icp_position FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute' ORDER BY icp_position", (ROUND,)
            )
            executions = cursor.fetchall()
            assert len(executions) == 20
            for run_id, position in executions:
                _proof_execution(objects, icps[position], 0, position, run_id)
                cursor.execute(
                    "UPDATE public.lab_arena_runs SET status='accepted',"
                    "terminal_cause='accepted',output_ref=%s WHERE run_id=%s",
                    ("arena/output/%s.json" % run_id, run_id),
                )
            cursor.execute("SET LOCAL session_replication_role=origin")
        connection.commit()

        def fixture_judge_boundary(run, *, icp, companies, policy):
            document = json.loads(objects.get(run["output_ref"]))
            validated = scoring.validate_scoring_output_document(document)
            return scoring.validate_breakdowns_for_item(
                validated["breakdowns"], icp=icp, companies=companies,
                max_scored_companies=int(policy["max_scored_companies"]),
                integrity_policy=True, contacts_required=True,
            )

        monkeypatch.setattr(service, "_verified_breakdowns", fixture_judge_boundary)
        assert service.close_stage(ROUND, 1)["status"] == "ok"
        for stage in (1, 2):
            assert service.open_scoring(ROUND, stage)["assignments"] == 10
            score_runs = service.store.list_runs(ROUND, stage=stage, kind="score")
            assert len(score_runs) == 10
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL session_replication_role=replica")
                for run in score_runs:
                    position = int(run["icp_position"])
                    company = _proof_company(icps[position], 0, position)
                    document = scoring.build_scoring_output(
                        run["scored_run_id"], [_proof_breakdown(company, 40)]
                    )
                    ref = "arena/score/%s.json" % run["run_id"]
                    objects.put(ref, json.dumps(document).encode())
                    cursor.execute(
                        "UPDATE public.lab_arena_runs SET status='accepted',"
                        "terminal_cause='accepted',output_ref=%s WHERE run_id=%s",
                        (ref, run["run_id"]),
                    )
                cursor.execute("SET LOCAL session_replication_role=origin")
            connection.commit()
            assert service.close_scoring(ROUND, stage)["status"] == "closed"
            assert service.score_stage(ROUND, stage)["status"] == "ok"
            if stage == 1:
                assert service.open_stage(ROUND, 2)["status"] == "ok"
                assert service.close_stage(ROUND, 2)["status"] == "ok"
        assert service.publish(ROUND)["status"] == "ok"
        row = service.store.get_round(ROUND)
        ranking = row["publication_doc"]["final_ranking"]
        assert row["status"] == "published" and len(ranking) == 1
        assert ranking[0]["submission_id"] == BASELINE
        assert ranking[0]["eligible"] is True and ranking[0]["final_score"] > 0
        assert preserved_hashes(connection) == before
        with connection.cursor() as cursor:
            assert prepare(cursor)["status"] == "existing"
            cursor.execute(
                "SELECT public.lab_arena_sep17_recovery284_prior_valid_v1(),"
                "public.lab_arena_sep17_recovery284_archive_valid_v1(),"
                "public.lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1()"
            )
            assert cursor.fetchone() == (True, True, True)
    finally:
        connection.close()
