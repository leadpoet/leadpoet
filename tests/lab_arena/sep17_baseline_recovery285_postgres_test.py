"""Protected recovery284 terminal fixture; minimal same-round recovery285."""
from __future__ import annotations

import hashlib
import json
import re
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab_arena.store import hash_lease_token, new_lease_token
from scripts import arena_sep17_baseline_recovery285 as operator
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.sep17_baseline_recovery278_postgres_test import _insert_rows
from tests.lab_arena.sep17_baseline_recovery284_postgres_test import (
    restore_terminal as restore_recovery284_terminal,
)

ROOT = Path(__file__).parents[2]
PRIVATE = Path("/private/tmp/leadpoet-tyche-rerun-evidence-20260915/private")
SNAPSHOT = PRIVATE / "sep17-recovery285-terminal-snapshot-20260917T145112.070964Z.json"
SNAPSHOT_SHA = "755719b3b924ba0be68ef683a312d4da260234e8140aae31a9b4e542240ded06"
BANK_SNAPSHOT = PRIVATE / "sep17-original-bank.json"
MIGRATION_284 = ROOT / "scripts/284-arena-2026-09-17-baseline-recovery.sql"
MIGRATION = ROOT / "scripts/285-arena-2026-09-17-baseline-recovery.sql"
TEMPLATE = Path(str(MIGRATION) + ".template")
ROUND = operator.ROUND
BASELINE = operator.BASELINE
ARCHIVE = "arena-2026-09-17-rerun284archive"
ARCHIVE_BASELINE = "baseline-2026-09-17-rerun284archive"
TEST_SOURCE_SIZE = 600001
TEST_SOURCE_SHA = "b" * 64
TEST_SOURCE_COMMIT = "a" * 40
pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in (SNAPSHOT, BANK_SNAPSHOT)),
    reason="Recovery285 release gate requires the protected terminal fixture and bank.",
)


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def captured():
    assert stat.S_IMODE(SNAPSHOT.stat().st_mode) == 0o600
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == SNAPSHOT_SHA
    receipt = json.loads(SNAPSHOT.read_text())
    assert receipt["schema_version"] == "leadpoet.sep17_recovery285_terminal_snapshot.v1"
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
    )} == {"rounds": 6, "submissions": 16, "runs": 144, "ledger": 23676}
    assert all(isinstance(protected[key], str) and protected[key] for key in (
        "authority278", "audit278", "authority282", "audit282",
        "authority283", "audit283", "authority284", "audit284",
    ))
    proof = receipt["captured"]["proof"]
    assert proof["baseline_runs"]["count"] == 23
    assert proof["baseline_runs"]["assignments"] == 20
    assert proof["baseline_runs"]["active_count"] == 0
    assert proof["nonbaseline_submissions"]["count"] == 8
    submissions = [json.loads(value) for value in protected["submissions"]]
    miners = [row for row in submissions if row["round_id"] == ROUND
              and row["submission_id"] != BASELINE]
    assert len(miners) == 8 and {row["status"] for row in miners} == {"rejected"}
    assert len({row["miner_hotkey"] for row in miners}) == 5
    return receipt["captured"]


def terminal_round(snapshot):
    rows = [json.loads(value) for value in snapshot["protected_rows"]["rounds"]]
    return next(row for row in rows if row["round_id"] == ROUND)


def replacements(snapshot):
    row = terminal_round(snapshot)
    return {
        "__TERMINAL_CONFIGURATION_JSON__": compact(row["configuration_doc"]),
        "__TERMINAL_PARTICIPANTS_JSON__": compact(row["participants"]),
        "__TERMINAL_BASELINE_LEDGER_COUNT__": str(
            snapshot["proof"]["baseline_ledger"]["count"]
        ),
    }


def test_template_matches_exact_protected_terminal():
    rendered = TEMPLATE.read_text()
    for token, value in replacements(captured()).items():
        assert token in rendered
        rendered = rendered.replace(token, value)
    assert re.search(r"__[A-Z0-9_]+__", rendered) is None
    assert rendered == MIGRATION.read_text()


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def restore_terminal(connection):
    snapshot = captured()
    restore_recovery284_terminal(connection)
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION_284.read_text())
    connection.commit()
    rows = snapshot["protected_rows"]
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_sep17_baseline_recovery284_audit,"
            "public.lab_arena_sep17_baseline_recovery284_authority,"
            "public.lab_arena_sep17_baseline_recovery283_audit,"
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
            ("lab_arena_sep17_baseline_recovery284_authority", "authority284"),
            ("lab_arena_sep17_baseline_recovery284_audit", "audit284"),
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
            "SELECT public.lab_arena_sep17_recovery284_archive_valid_v1(),"
            "public.lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1()"
        )
        assert cursor.fetchone() == (True, True)
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
                "round_id IN ('arena-2026-09-17-archive',"
                "'arena-2026-09-17-rerun278archive',"
                "'arena-2026-09-17-rerun282archive',"
                "'arena-2026-09-17-rerun283archive')", "round_id",
            ),
            "archive_submissions": table_hash(
                cursor, "lab_arena_submissions",
                "round_id IN ('arena-2026-09-17-archive',"
                "'arena-2026-09-17-rerun278archive',"
                "'arena-2026-09-17-rerun282archive',"
                "'arena-2026-09-17-rerun283archive')", "round_id||':'||submission_id",
            ),
            "archive_runs": table_hash(
                cursor, "lab_arena_runs",
                "round_id IN ('arena-2026-09-17-archive',"
                "'arena-2026-09-17-rerun278archive',"
                "'arena-2026-09-17-rerun282archive',"
                "'arena-2026-09-17-rerun283archive')", "round_id||':'||run_id",
            ),
            "archive_ledger": table_hash(
                cursor, "lab_arena_ledger",
                "round_id IN ('arena-2026-09-17-archive',"
                "'arena-2026-09-17-rerun278archive',"
                "'arena-2026-09-17-rerun282archive',"
                "'arena-2026-09-17-rerun283archive')", "entry_id",
            ),
            "sep18_round": table_hash(cursor, "lab_arena_rounds", "round_id='arena-2026-09-18'", "round_id"),
            "sep18_submissions": table_hash(cursor, "lab_arena_submissions", "round_id='arena-2026-09-18'", "submission_id"),
            "sep18_runs": table_hash(cursor, "lab_arena_runs", "round_id='arena-2026-09-18'", "run_id"),
            "sep18_ledger": table_hash(cursor, "lab_arena_ledger", "round_id='arena-2026-09-18'", "entry_id"),
            "miners": table_hash(cursor, "lab_arena_submissions", "round_id='arena-2026-09-17' AND submission_id<>'baseline-2026-09-17'", "submission_id"),
            "miner_runs": table_hash(cursor, "lab_arena_runs", "round_id='arena-2026-09-17' AND submission_id<>'baseline-2026-09-17'", "run_id"),
            "miner_ledger": table_hash(cursor, "lab_arena_ledger", "round_id='arena-2026-09-17' AND submission_id<>'baseline-2026-09-17'", "entry_id"),
            "weights": table_hash(
                cursor, "lab_arena_accepted_weight_states", "TRUE",
                "network||':'||netuid::text||':'||epoch::text",
            ),
            **{
                name: table_hash(cursor, "lab_arena_sep17_baseline_recovery%s_%s" % (number, kind), "TRUE", "round_id")
                for number in ("278", "282", "283", "284")
                for kind, name in (("authority", "authority" + number), ("audit", "audit" + number))
            },
        }


def prepare(cursor, **overrides):
    args = {
        "size": TEST_SOURCE_SIZE, "sha": TEST_SOURCE_SHA,
        "commit": TEST_SOURCE_COMMIT, "bank": operator.BANK_SHA256,
        "schedule": operator.FORWARD_SCHEDULE,
    }
    args.update(overrides)
    cursor.execute(
        "SELECT public.lab_arena_prepare_sep17_baseline_recovery285_v1("
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
                "'public.lab_arena_prepare_sep17_baseline_recovery285_v1(bigint,text,text,text,jsonb)',"
                "'EXECUTE'),has_function_privilege('service_role',"
                "'public.lab_arena_prepare_sep17_baseline_recovery285_v1(bigint,text,text,text,jsonb)',"
                "'EXECUTE'),pg_catalog.to_regclass("
                "'public.lab_arena_sep17_baseline_recovery285_authority') IS NULL,"
                "pg_catalog.to_regclass("
                "'public.lab_arena_sep17_baseline_recovery285_audit') IS NULL"
            )
            assert cursor.fetchone() == (True, False, True, True)
            cursor.execute(
                "SELECT count(*) FROM pg_catalog.pg_trigger WHERE tgname LIKE "
                "'lab_arena_sep17_recovery285%' AND NOT tgisinternal"
            )
            assert cursor.fetchone() == (0,)

        # Null/malformed facts and each terminal-state drift fail atomically.
        for overrides in (
            {"size": None}, {"sha": None}, {"commit": None},
            {"size": 0}, {"sha": "0" * 63}, {"commit": "0" * 39},
            {"bank": "0" * 64},
            {"schedule": {**operator.FORWARD_SCHEDULE,
                          "stage_1_close": "2026-09-17T21:59:59Z"}},
        ):
            with pytest.raises(Exception):
                with connection.cursor() as cursor:
                    prepare(cursor, **overrides)
            connection.rollback()
            assert preserved_hashes(connection) == before
        for statement in (
            "UPDATE public.lab_arena_runs SET status='pending' "
            "WHERE round_id='arena-2026-09-17' AND icp_position=0",
            "UPDATE public.lab_arena_rounds SET status_generation=10 "
            "WHERE round_id='arena-2026-09-17'",
            "UPDATE public.lab_arena_rounds SET cancel_reason=NULL "
            "WHERE round_id='arena-2026-09-17'",
            "UPDATE public.lab_arena_rounds SET evaluation_date=NULL "
            "WHERE round_id='arena-2026-09-17'",
            "UPDATE public.lab_arena_rounds SET icp_set_date=NULL "
            "WHERE round_id='arena-2026-09-17'",
            "UPDATE public.lab_arena_rounds SET benchmark_ref=NULL "
            "WHERE round_id='arena-2026-09-17'",
            "UPDATE public.lab_arena_submissions SET source_ref='wrong' "
            "WHERE round_id='arena-2026-09-17' AND submission_id='baseline-2026-09-17'",
            "UPDATE public.lab_arena_rounds SET configuration_doc="
            "configuration_doc||'{\"unexpected\":true}'::jsonb "
            "WHERE round_id='arena-2026-09-17'",
        ):
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SET LOCAL session_replication_role=replica")
                    cursor.execute(statement)
                    cursor.execute("SET LOCAL session_replication_role=origin")
                    prepare(cursor)
            except Exception:
                pass
            else:
                pytest.fail("drift was admitted: " + statement.split(" WHERE")[0])
            connection.rollback()
            assert preserved_hashes(connection) == before

        with pytest.raises(Exception):
            with connection.cursor() as cursor:
                cursor.execute(
                    "ALTER FUNCTION public.lab_arena__successful_call_cost_state(text,text,text) "
                    "RENAME TO recovery285_test_original_cost_state"
                )
                cursor.execute(
                    "CREATE FUNCTION public.lab_arena__successful_call_cost_state("
                    "p_submission_id text,p_kind text,p_provider text DEFAULT NULL) RETURNS jsonb "
                    "LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$ "
                    "SELECT public.recovery285_test_original_cost_state("
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

        from lab_arena import contracts, scoring
        from tests.lab_arena.sep16_native_baseline_rerun_postgres_test import (
            _proof_breakdown, _proof_company, _proof_execution,
        )
        from tests.lab_arena.test_lab_arena_service_round import Harness

        harness = Harness(lambda: psycopg2.connect(**dsn), tmp_path,
                          challengers=[], runners=["recovery285-proof"])
        service, objects = harness.service, harness.objects
        harness.clock.now = datetime(2026, 9, 17, 16, tzinfo=timezone.utc)
        bank = json.loads(BANK_SNAPSHOT.read_text())
        assert hashlib.sha256(
            contracts.canonical_json(bank["icps"]).encode()
        ).hexdigest() == operator.BANK_SHA256
        objects.put(operator.BANK_REF, json.dumps(bank).encode())
        icps = bank["icps"]
        def claim_execute_batch(expected_positions):
            leases = []
            for _ in expected_positions:
                token = new_lease_token()
                request_id = contracts.new_request_id()
                claim = service.store.claim_assignment(
                    round_id=ROUND,
                    runner_hotkey=harness.runner_keys[0],
                    declared_parallelism=10,
                    slot_ceiling=20,
                    excluded_miner_hotkeys=[],
                    request_id=request_id,
                    request_hash=contracts.document_hash({"request_id": request_id}),
                    lease_token_hash=hash_lease_token(token),
                )
                assert claim["status"] == "leased" and claim["kind"] == "execute"
                leases.append((claim, token))
            assert sorted(int(claim["icp_position"]) for claim, _ in leases) == list(
                expected_positions
            )
            assert service.store.get_round(ROUND)["status"] == "stage1"
            for claim, token in leases:
                position = int(claim["icp_position"])
                _proof_execution(objects, icps[position], 0, position, claim["run_id"])
                assert service.store.complete_attempt(
                    run_id=claim["run_id"],
                    lease_token_hash=hash_lease_token(token),
                    result={"terminal_status": "accepted"},
                    terminal_cause="accepted",
                    output_ref="arena/output/%s.json" % claim["run_id"],
                )["status"] == "accepted"

        claim_execute_batch(range(0, 10))
        claim_execute_batch(range(10, 20))

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
            scored_positions = []
            for _ in range(10):
                token = new_lease_token()
                request_id = contracts.new_request_id()
                run = service.store.claim_assignment(
                    round_id=ROUND,
                    runner_hotkey=harness.runner_keys[0],
                    declared_parallelism=1,
                    slot_ceiling=20,
                    excluded_miner_hotkeys=[],
                    request_id=request_id,
                    request_hash=contracts.document_hash({"request_id": request_id}),
                    lease_token_hash=hash_lease_token(token),
                )
                assert run["status"] == "leased" and run["kind"] == "score"
                position = int(run["icp_position"])
                scored_positions.append(position)
                company = _proof_company(icps[position], 0, position)
                document = scoring.build_scoring_output(
                    run["scored_run_id"], [_proof_breakdown(company, 40)]
                )
                ref = "arena/score/%s.json" % run["run_id"]
                objects.put(ref, json.dumps(document).encode())
                assert service.store.complete_attempt(
                    run_id=run["run_id"],
                    lease_token_hash=hash_lease_token(token),
                    result={"terminal_status": "accepted"},
                    terminal_cause="accepted",
                    output_ref=ref,
                )["status"] == "accepted"
            assert len(set(scored_positions)) == 10
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
