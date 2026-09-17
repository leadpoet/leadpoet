"""PostgreSQL proof for the exact Sep16 recovery275 template."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from lab_arena import scoring
from lab_arena.store import FUNCTION_SIGNATURES, hash_lease_token
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.sep16_baseline_recovery_postgres_test import (
    ARCHIVE_ROUND as RERUN265_ARCHIVE,
    ARCHIVE_SUBMISSION as RERUN265_ARCHIVE_SUBMISSION,
    _fail_both_attempts_and_cancel,
    _install_and_recover,
    _prepare_latest,
    _render_recovery,
    _shift_schedule,
    _single_hash,
)
from tests.lab_arena.sep16_baseline_recovery273_postgres_test import (
    RECOVERY_SOURCE_COMMIT as TERMINAL_SOURCE_COMMIT,
    RECOVERY_SOURCE_REF as TERMINAL_SOURCE_REF,
    RECOVERY_SOURCE_SHA as TERMINAL_SOURCE_SHA,
    RECOVERY_SOURCE_SIZE as TERMINAL_SOURCE_SIZE,
    _install_and_recover273 as _install_prior_recovery273,
    _prepare_terminal_recovery272,
    _render_recovery273 as _render_prior_recovery273,
)
from tests.lab_arena.sep16_baseline_recovery272_postgres_test import (
    ARCHIVE_ROUND as RERUN269_ARCHIVE,
    ARCHIVE_SUBMISSION as RERUN269_ARCHIVE_SUBMISSION,
)
from tests.lab_arena.sep16_native_baseline_rerun_postgres_test import (
    BASELINE,
    MIGRATIONS,
    ROUND,
    _proof_breakdown,
    _proof_company,
    _proof_execution,
    _seed_observed_sep16,
    _service_scoring_items,
    _state_seal,
)
from tests.lab_arena.sep16_latest_champion_reseal_postgres_test import (
    NEW_IMAGE,
    NEW_IMAGE_REF,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim
from tests.lab_arena.test_lab_arena_service_round import Harness


ARCHIVE_ROUND = "arena-2026-09-16-rerun273archive"
ARCHIVE_SUBMISSION = "baseline-2026-09-16-native-rerun273-archive"
RECOVERY_SOURCE_REF = (
    "arena/arena-2026-09-16/sources/"
    "baseline-2026-09-16-recovery275.tar.gz"
)
RECOVERY_SOURCE_SIZE = 556089
RECOVERY_SOURCE_SHA = (
    "dd877baf3f1210480b8bb1a0fdd62c9b0d75a1478da705a0db043d98978ffe67"
)
RECOVERY_SOURCE_COMMIT = "396bcb277ce831fd92e31a80f556d2996812bbf4"
RESEALED_SOURCE_SIZE = 556500
RESEALED_SOURCE_SHA = (
    "c86df33ae0f21164b46f9ab6b8e7dbcb116b7a53c49dbc43603a49b04067d104"
)
RESEALED_SOURCE_COMMIT = "2b8386b835de02739ddef6c182bdeba43c6ad467"
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/275-arena-2026-09-16-baseline-recovery.sql.template"
)
MIGRATION = TEMPLATE.with_suffix("")
RESEAL_TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/276-arena-2026-09-16-recovery275-source-reseal.sql.template"
)
RESEAL_MIGRATION = RESEAL_TEMPLATE.with_suffix("")
FORWARD_SCHEDULE = {
    "benchmark_deadline": "2026-09-16T23:15:00Z",
    "final_scoring_close": "2026-09-17T08:15:00Z",
    "publication_deadline": "2026-09-17T08:45:00Z",
    "stage_1_close": "2026-09-17T05:00:00Z",
    "stage_1_scoring_close": "2026-09-17T06:30:00Z",
    "stage_1_start": "2026-09-16T23:15:01Z",
    "stage_2_close": "2026-09-17T06:45:00Z",
    "stage_2_start": "2026-09-17T06:35:00Z",
    "submission_cutoff": "2026-09-16T00:00:00Z",
    "submission_open": "2026-09-15T00:00:00Z",
}
PRIOR_TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/273-arena-2026-09-16-baseline-recovery.sql.template"
)
PROTECTED_SNAPSHOT_SHA256 = (
    "0904d3479b8c81c85b840c1815955b225ad3398e7574ac9f42e7bb1800f2e23c"
)
PRIOR_PROTECTED_SNAPSHOT_SHA256 = (
    "517bea5e5bb40ea13795f11f6d648c0d7c064a0f6605b8451d4e23075bbbeb9b"
)
DEFAULT_PROTECTED_SNAPSHOT = Path(
    "/private/tmp/leadpoet-tyche-rerun-evidence-20260915/private/"
    "sep16-recovery274-terminal-snapshot-20260917T001506.440257Z.json"
)
DEFAULT_PRIOR_PROTECTED_SNAPSHOT = Path(
    "/private/tmp/leadpoet-tyche-rerun-evidence-20260915/private/"
    "sep16-recovery273-terminal-snapshot-20260916T223031.080958Z.json"
)


@pytest.fixture
def connect():
    generator = database_with_lab_arena_migration(MIGRATIONS)
    psycopg2, dsn = next(generator)
    try:
        yield lambda: psycopg2.connect(**dsn)
    finally:
        generator.close()


def _append_catalog_settlements(connection, *, identity_offset=0):
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER")
        cursor.execute(
            "SELECT run_id,miner_hotkey,stage FROM public.lab_arena_runs "
            "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
            "ORDER BY run_id LIMIT 2",
            (ROUND, BASELINE),
        )
        rows = cursor.fetchall()
        assert len(rows) == 2
        for index, (run_id, hotkey, stage) in enumerate(rows):
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,"
                "round_id,submission_id,run_id,stage,call_identity,provider,"
                "operation_id,funding_source,amount_microusd,entry_doc,"
                "terminal_response) VALUES ('settlement',%s,%s,%s,%s,%s,%s,"
                "'deepline','deepline.execute','host',0,%s::jsonb,%s::jsonb)",
                (
                    hotkey,
                    ROUND,
                    BASELINE,
                    run_id,
                    stage,
                    "sha256:" + str(index + identity_offset + 1) * 64,
                    json.dumps(
                        {
                            "late_reconciliation": True,
                            "sep16_generic_http_catalog_reconciliation": True,
                                "reconciled_uncertainty_entry_id": (
                                    900000 + identity_offset + index
                                ),
                        }
                    ),
                    json.dumps(
                        {
                            "status": 200,
                            "call_succeeded": True,
                            "provider_cost": {
                                "basis": "fixture_free_catalog",
                                "units": "0",
                                "unit_name": "credits",
                                "operation": "generic_http_request",
                            },
                        }
                    ),
                ),
            )
        cursor.execute("ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER")
    connection.commit()


def _fail_recovery269_and_cancel(connection):
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',"
            "terminal_cause='provider_error' WHERE round_id=%s "
            "AND submission_id=%s AND kind='execute' AND status='pending' "
            "RETURNING assignment_id,miner_hotkey,stage,icp_position,stage_generation",
            (ROUND, BASELINE),
        )
        first_attempts = cursor.fetchall()
        assert len(first_attempts) == 20
        for assignment, hotkey, stage, position, generation in first_attempts:
            cursor.execute(
                "INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,"
                "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
                "terminal_cause,stage_generation) VALUES "
                "(%s,%s,%s,%s,%s,%s,%s,2,'execute','failed','provider_error',%s)",
                (
                    assignment + ":2", assignment, ROUND, BASELINE, hotkey,
                    stage, position, generation,
                ),
            )
        cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
        cursor.execute("SELECT public.lab_arena_close_parallel_execution_v1(%s)", (ROUND,))
        result = cursor.fetchone()[0]
        assert result["status"] == "cancelled"
        assert result["incomplete_assignments"] == 20
    connection.commit()


def _prepare_terminal_recovery269(connection, objects=None):
    old_schedule, hotkeys, ids = _seed_observed_sep16(connection, objects)
    _prepare_latest(connection, old_schedule)
    _fail_both_attempts_and_cancel(connection)
    schedule269 = _shift_schedule(old_schedule, minutes=45)
    migration269 = _render_recovery(connection, schedule269)
    _install_and_recover(connection, migration269, schedule269)
    _fail_recovery269_and_cancel(connection)
    _append_catalog_settlements(connection, identity_offset=2)
    return schedule269, hotkeys, ids


def _prepare_terminal_recovery273(connection, objects=None):
    schedule272, hotkeys, ids = _prepare_terminal_recovery272(connection, objects)
    schedule273 = _shift_schedule(schedule272, minutes=90)
    migration273 = _render_prior_recovery273(connection, schedule273)
    _install_prior_recovery273(connection, migration273, schedule273)
    _fail_recovery269_and_cancel(connection)
    # Production recovery273 was cancelled by the canonical operator RPC.
    # This fixture creates the same terminal 40-run shape directly, so bind
    # its cancellation reason to that sealed production contract.
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='cancelled',cancel_reason='operator' "
            "WHERE round_id=%s", (ROUND,),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
    connection.commit()
    return schedule273, hotkeys, ids


def _fixture_template():
    """Derive a synthetic 275 fixture while the production source seal is open."""

    rendered = PRIOR_TEMPLATE.read_text(encoding="utf-8")
    rendered = rendered.replace("recovery273", "__RECOVERY_SELF__")
    rendered = rendered.replace("rerun273", "__RERUN_SELF__")
    rendered = rendered.replace("recovery272", "recovery273")
    rendered = rendered.replace("rerun272", "rerun273")
    rendered = rendered.replace("__RECOVERY_SELF__", "recovery275")
    rendered = rendered.replace("__RERUN_SELF__", "rerun275")
    rendered = rendered.replace("before 273", "before 275")
    rendered = rendered.replace("fourth baseline rerun", "fifth baseline rerun")
    rendered = rendered.replace("third native rerun", "fourth native rerun")
    rendered = rendered.replace("533090", str(TERMINAL_SOURCE_SIZE))
    rendered = rendered.replace(
        "72a008e2cacde52921c4be94951839edccc321d90de07741010a3100463eef32",
        TERMINAL_SOURCE_SHA,
    )
    rendered = rendered.replace(
        "0a19bd28177628d01b5ce1f4fc54cc459d7d4f51",
        TERMINAL_SOURCE_COMMIT,
    )
    rendered = rendered.replace(
        "OR NOT public.lab_arena_sep16_recovery_archive_valid_v1()\n"
        "     OR NOT public.lab_arena_sep16_recovery273_archive_valid_v1()",
        "OR NOT public.lab_arena_sep16_recovery_archive_valid_v1()\n"
        "     OR NOT public.lab_arena_sep16_recovery272_archive_valid_v1()\n"
        "     OR NOT public.lab_arena_sep16_recovery273_archive_valid_v1()",
    ).replace(
        "OR NOT public.lab_arena_sep16_recovery_archive_valid_v1()\n"
        "       OR NOT public.lab_arena_sep16_recovery273_archive_valid_v1()",
        "OR NOT public.lab_arena_sep16_recovery_archive_valid_v1()\n"
        "       OR NOT public.lab_arena_sep16_recovery272_archive_valid_v1()\n"
        "       OR NOT public.lab_arena_sep16_recovery273_archive_valid_v1()",
    )
    production = TEMPLATE.read_text(encoding="utf-8")
    block_start = "DO $bind_sep16_recovery275_archive_guard$"
    block_end = "$bind_sep16_recovery275_archive_guard$;"
    fixture_start = rendered.index(block_start)
    fixture_end = rendered.index(block_end, fixture_start) + len(block_end)
    production_start = production.index(block_start)
    production_end = production.index(block_end, production_start) + len(block_end)
    rendered = (
        rendered[:fixture_start]
        + production[production_start:production_end]
        + rendered[fixture_end:]
    )
    return rendered


def _load_protected_snapshot(path, expected_sha256):
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected_sha256
    document = json.loads(payload)
    assert document["read_only"] is True
    assert document["production_writes"] is False
    assert document["paid_calls"] == 0
    assert document["captured"]["gate"]["usable"] is True
    return document["captured"]


def _protected_paths():
    snapshot = Path(
        os.environ.get("SEP16_RECOVERY275_PROTECTED_SNAPSHOT", DEFAULT_PROTECTED_SNAPSHOT)
    )
    prior = Path(
        os.environ.get(
            "SEP16_RECOVERY275_PRIOR_PROTECTED_SNAPSHOT",
            DEFAULT_PRIOR_PROTECTED_SNAPSHOT,
        )
    )
    if not snapshot.is_file() or not prior.is_file():
        pytest.skip("protected Sep16 recovery snapshots are unavailable")
    return snapshot, prior


def _restore_protected_rows(connection, captured, prior_captured):
    rows = captured["protected_rows"]
    prior_rows = prior_captured["protected_rows"]
    authority_rows = {
        "lab_arena_sep16_baseline_rerun_audit": rows["prior_rerun_audit"],
        "lab_arena_sep16_rerun_release_authority": rows["release_authority"],
        "lab_arena_sep16_baseline_recovery_authority": rows["recovery_authority"],
        "lab_arena_sep16_baseline_recovery_audit": rows["recovery_audit"],
        "lab_arena_sep16_baseline_recovery272_authority": prior_rows[
            "recovery272_authority"
        ],
        "lab_arena_sep16_baseline_recovery272_audit": prior_rows[
            "recovery272_audit"
        ],
        "lab_arena_sep16_baseline_recovery273_authority": rows[
            "recovery273_authority"
        ],
        "lab_arena_sep16_baseline_recovery273_audit": rows["recovery273_audit"],
    }
    def insert_rows(cursor, table, documents):
        cursor.execute(
            "SELECT attname FROM pg_catalog.pg_attribute "
            "WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped "
            "AND attgenerated='' ORDER BY attnum",
            ("public." + table,),
        )
        columns = [row[0] for row in cursor.fetchall()]
        assert columns and all(name.replace("_", "").isalnum() for name in columns)
        names = ",".join(columns)
        selected = ",".join("row_value." + name for name in columns)
        cursor.execute(
            "INSERT INTO public." + table + " (" + names + ") SELECT "
            + selected + " FROM pg_catalog.jsonb_populate_recordset("
            "NULL::public." + table + ",%s::jsonb) AS row_value",
            (json.dumps(documents),),
        )

    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
        for table in authority_rows:
            cursor.execute("TRUNCATE public." + table)
            if table.endswith("_authority"):
                # The predecessor fixture rendered row-specific CHECK values.
                # Remove those fixture-only seals before restoring the exact
                # protected rows; archive functions still verify every hash.
                cursor.execute(
                    "SELECT conname FROM pg_catalog.pg_constraint "
                    "WHERE conrelid=%s::regclass AND contype='c'",
                    ("public." + table,),
                )
                constraints = [row[0] for row in cursor.fetchall()]
                assert all(name.replace("_", "").isalnum() for name in constraints)
                for constraint in constraints:
                    cursor.execute(
                        "ALTER TABLE public." + table
                        + " DROP CONSTRAINT " + constraint
                    )
        for table, key in (
            ("lab_arena_rounds", "rounds"),
            ("lab_arena_submissions", "submissions"),
            ("lab_arena_runs", "runs"),
            ("lab_arena_ledger", "ledger"),
        ):
            decoded = [json.loads(value) for value in rows[key]]
            for start in range(0, len(decoded), 500):
                insert_rows(cursor, table, decoded[start : start + 500])
        for table, value in authority_rows.items():
            insert_rows(cursor, table, [json.loads(value)])
        cursor.execute(
            "SELECT pg_catalog.setval('public.lab_arena_ledger_entry_id_seq',"
            "(SELECT max(entry_id) FROM public.lab_arena_ledger),true)"
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _render_protected_recovery275(captured, new_schedule):
    assert new_schedule == FORWARD_SCHEDULE
    rendered = MIGRATION.read_text(encoding="utf-8")
    assert "__SEALED_" not in rendered
    assert str(RECOVERY_SOURCE_SIZE) in rendered
    assert RECOVERY_SOURCE_SHA in rendered
    assert RECOVERY_SOURCE_COMMIT in rendered
    seal = captured["seal"]
    assert seal["terminal_baseline_run_count"] == 27
    assert seal["terminal_baseline_ledger_count"] == 4428
    assert seal["terminal_status_generation"] == 20
    assert seal["terminal_stage_generation"] == 16
    return rendered


def _render_recovery275(connection, new_schedule):
    seal = _state_seal(connection)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT status_generation,stage_generation,cancel_reason,"
            "configuration_doc->'schedule' FROM public.lab_arena_rounds "
            "WHERE round_id=%s",
            (ROUND,),
        )
        status_generation, stage_generation, reason, old_schedule = cursor.fetchone()
        cursor.execute(
            "SELECT count(*),coalesce(max(entry_id),0),"
            "coalesce(sum(amount_microusd) FILTER (WHERE entry_kind='settlement'),0),"
            "coalesce(sum(amount_microusd) FILTER (WHERE entry_kind='uncertain'),0) "
            "FROM public.lab_arena_ledger WHERE round_id=%s AND submission_id=%s",
            (ROUND, BASELINE),
        )
        ledger_count, ledger_max, settled, uncertain = cursor.fetchone()
        cursor.execute(
            "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
            "AND submission_id=%s",
            (ROUND, BASELINE),
        )
        run_count = cursor.fetchone()[0]
    if seal["baseline_ledger"] is None:
        seal["baseline_ledger"] = "sha256:" + hashlib.sha256(b"").hexdigest()
    values = {
        "__SEALED_RECOVERY_SOURCE_SIZE_BYTES__": str(RECOVERY_SOURCE_SIZE),
        "__SEALED_RECOVERY_SOURCE_SHA256__": RECOVERY_SOURCE_SHA,
        "__SEALED_RECOVERY_SOURCE_COMMIT__": RECOVERY_SOURCE_COMMIT,
        "__SEALED_TERMINAL_ROUND_HASH__": seal["round"],
        "__SEALED_TERMINAL_BASELINE_SUBMISSION_HASH__": seal["baseline_submission"],
        "__SEALED_TERMINAL_BASELINE_RUNS_HASH__": seal["baseline_runs"],
        "__SEALED_TERMINAL_BASELINE_LEDGER_HASH__": seal["baseline_ledger"],
        "__SEALED_TERMINAL_BASELINE_RUN_COUNT__": str(run_count),
        "__SEALED_TERMINAL_BASELINE_LEDGER_COUNT__": str(ledger_count),
        "__SEALED_TERMINAL_BASELINE_LEDGER_MAX_ENTRY_ID__": str(ledger_max),
        "__SEALED_TERMINAL_BASELINE_SETTLED_MICROUSD__": str(settled),
        "__SEALED_TERMINAL_BASELINE_UNCERTAIN_MICROUSD__": str(uncertain),
        "__SEALED_CHALLENGER_RUNS_HASH__": seal["challenger_runs"],
        "__SEALED_CHALLENGER_SUBMISSIONS_HASH__": seal["challenger_submissions"],
        "__SEALED_CHALLENGER_LEDGER_HASH__": seal["challenger_ledger"],
        "__SEALED_CHALLENGER_LEDGER_MAX_ENTRY_ID__": str(seal["challenger_ledger_max"]),
        "__SEALED_PRIOR_RERUN_AUDIT_HASH__": _single_hash(
            connection, "lab_arena_sep16_baseline_rerun_audit", f"round_id='{ROUND}'"
        ),
        "__SEALED_RELEASE_AUTHORITY_HASH__": _single_hash(
            connection, "lab_arena_sep16_rerun_release_authority", f"round_id='{ROUND}'"
        ),
        "__SEALED_PRIOR_RECOVERY_AUTHORITY_HASH__": _single_hash(
            connection, "lab_arena_sep16_baseline_recovery273_authority", f"round_id='{ROUND}'"
        ),
        "__SEALED_PRIOR_RECOVERY_AUDIT_HASH__": _single_hash(
            connection, "lab_arena_sep16_baseline_recovery273_audit", f"round_id='{ROUND}'"
        ),
        "__SEALED_TERMINAL_STATUS_GENERATION__": str(status_generation),
        "__SEALED_TERMINAL_STAGE_GENERATION__": str(stage_generation),
        "__SEALED_TERMINAL_CANCEL_REASON__": reason,
        "__SEALED_OLD_FORWARD_SCHEDULE_JSON__": json.dumps(
            old_schedule, sort_keys=True, separators=(",", ":")
        ),
        "__SEALED_NEW_FORWARD_SCHEDULE_JSON__": json.dumps(
            new_schedule, sort_keys=True, separators=(",", ":")
        ),
    }
    rendered = _fixture_template()
    for marker, value in values.items():
        assert rendered.count(marker) >= 1
        assert isinstance(value, str), marker
        rendered = rendered.replace(marker, value)
    assert "__SEALED_" not in rendered
    assert TERMINAL_SOURCE_REF in rendered
    assert str(TERMINAL_SOURCE_SIZE) in rendered
    assert TERMINAL_SOURCE_SHA in rendered
    assert TERMINAL_SOURCE_COMMIT in rendered
    return rendered


def _render_recovery276(connection=None):
    rendered = RESEAL_TEMPLATE.read_text(encoding="utf-8")
    if connection is not None:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_catalog.to_jsonb(row_value)-'authorized_at'-"
                "'recovery_source_size_bytes'-'recovery_source_sha256'-"
                "'recovery_source_commit' FROM "
                "public.lab_arena_sep16_baseline_recovery275_authority AS row_value"
            )
            expected = json.dumps(
                cursor.fetchone()[0], sort_keys=True, separators=(",", ":")
            )
        start = rendered.index("$expected$") + len("$expected$")
        end = rendered.index("$expected$", start)
        rendered = rendered[:start] + expected + rendered[end:]
    values = {
        "__RESEALED_RECOVERY_SOURCE_SIZE_BYTES__": str(RESEALED_SOURCE_SIZE),
        "__RESEALED_RECOVERY_SOURCE_SHA256__": RESEALED_SOURCE_SHA,
        "__RESEALED_RECOVERY_SOURCE_COMMIT__": RESEALED_SOURCE_COMMIT,
    }
    for marker, value in values.items():
        assert rendered.count(marker) >= 1
        rendered = rendered.replace(marker, value)
    assert "__RESEALED_" not in rendered
    return rendered


def _install_and_recover275(
    connection,
    sql,
    schedule,
    *,
    source_size=RECOVERY_SOURCE_SIZE,
    source_sha=RECOVERY_SOURCE_SHA,
    source_commit=RECOVERY_SOURCE_COMMIT,
    install_sql=True,
):
    with connection.cursor() as cursor:
        if install_sql:
            cursor.execute(sql)
            cursor.execute(sql)
        arguments = (
            source_size,
            source_sha,
            source_commit,
            json.dumps(schedule),
        )
        cursor.execute(
            "SELECT public.lab_arena_prepare_sep16_baseline_recovery275_v1("
            "%s,%s,%s,%s::jsonb)",
            arguments,
        )
        prepared = cursor.fetchone()[0]
        cursor.execute(
            "SELECT public.lab_arena_prepare_sep16_baseline_recovery275_v1("
            "%s,%s,%s,%s::jsonb)",
            arguments,
        )
        existing = cursor.fetchone()[0]
    connection.commit()
    assert prepared["status"] == "prepared"
    assert existing["status"] == "existing"
    return prepared


def _accept_recovery275_scores(connection, objects, stage, icps):
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
        cursor.execute(
            "SELECT run_id,scored_run_id,icp_position FROM public.lab_arena_runs "
            "WHERE round_id=%s AND submission_id=%s AND kind='score' AND stage=%s "
            "AND assignment_id LIKE '%%:score:rerun275' ORDER BY icp_position",
            (ROUND, BASELINE, stage),
        )
        rows = cursor.fetchall()
        assert len(rows) == 10
        for run_id, scored_run_id, position in rows:
            company = _proof_company(icps[position], 0, position)
            breakdown = _proof_breakdown(company, 40)
            ref = "arena/score/%s.json" % run_id
            objects.put(
                ref,
                json.dumps(scoring.build_scoring_output(scored_run_id, [breakdown])).encode(),
            )
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "terminal_cause='accepted',output_ref=%s WHERE run_id=%s",
                (ref, run_id),
            )
        cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")


def test_recovery275_exact_migration_has_published_source_and_schedule_seal():
    template = TEMPLATE.read_text(encoding="utf-8")
    assert template.count("__SEALED_RECOVERY_SOURCE_SIZE_BYTES__") == 2
    assert template.count("__SEALED_RECOVERY_SOURCE_SHA256__") == 2
    assert template.count("__SEALED_RECOVERY_SOURCE_COMMIT__") == 2
    assert template.count("__SEALED_NEW_FORWARD_SCHEDULE_JSON__") == 2

    migration = MIGRATION.read_text(encoding="utf-8")
    schedule = json.dumps(FORWARD_SCHEDULE, sort_keys=True, separators=(",", ":"))
    assert "__SEALED_" not in migration
    assert migration.count(str(RECOVERY_SOURCE_SIZE)) == 2
    assert migration.count(RECOVERY_SOURCE_SHA) == 2
    assert migration.count(RECOVERY_SOURCE_COMMIT) == 2
    assert migration.count(schedule) == 2
    assert "terminal_baseline_run_count = 27" in migration
    assert "terminal_baseline_ledger_count = 4428" in migration

    reseal_template = RESEAL_TEMPLATE.read_text(encoding="utf-8")
    assert reseal_template.count("__RESEALED_RECOVERY_SOURCE_SIZE_BYTES__") >= 1
    assert reseal_template.count("__RESEALED_RECOVERY_SOURCE_SHA256__") >= 1
    assert reseal_template.count("__RESEALED_RECOVERY_SOURCE_COMMIT__") >= 1
    reseal = RESEAL_MIGRATION.read_text(encoding="utf-8")
    assert "__RESEALED_" not in reseal
    assert str(RESEALED_SOURCE_SIZE) in reseal
    assert RESEALED_SOURCE_SHA in reseal
    assert RESEALED_SOURCE_COMMIT in reseal


def test_recovery276_reseals_only_unused_recovery275_and_fails_closed(connect):
    connection = connect()
    try:
        schedule272, _hotkeys, _ids = _prepare_terminal_recovery273(connection)
        schedule275 = _shift_schedule(schedule272, minutes=90)
        migration275 = _render_recovery275(connection, schedule275)
        with connection.cursor() as cursor:
            cursor.execute(migration275)
        connection.commit()
        migration276 = _render_recovery276(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT relowner,relrowsecurity,relacl FROM pg_catalog.pg_class "
                "WHERE oid='public.lab_arena_sep16_baseline_recovery275_authority'"
                "::regclass"
            )
            acl_before = cursor.fetchone()
            cursor.execute(
                "SELECT pg_catalog.to_jsonb(row_value)-'authorized_at'-"
                "'recovery_source_size_bytes'-'recovery_source_sha256'-"
                "'recovery_source_commit' FROM "
                "public.lab_arena_sep16_baseline_recovery275_authority AS row_value"
            )
            non_source_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_prepare_sep16_baseline_recovery275_v1("
                "bigint,text,text,jsonb)'::regprocedure)"
            )
            function_before = cursor.fetchone()[0]
        terminal_before = _state_seal(connection)

        with connection.cursor() as cursor:
            cursor.execute(migration276)
            cursor.execute(migration276)
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT recovery_source_size_bytes,recovery_source_sha256,"
                "recovery_source_commit FROM "
                "public.lab_arena_sep16_baseline_recovery275_authority"
            )
            assert cursor.fetchone() == (
                RESEALED_SOURCE_SIZE,
                RESEALED_SOURCE_SHA,
                RESEALED_SOURCE_COMMIT,
            )
            cursor.execute(
                "SELECT conname,pg_get_constraintdef(oid,false) FROM "
                "pg_catalog.pg_constraint WHERE conrelid="
                "'public.lab_arena_sep16_baseline_recovery275_authority'::regclass "
                "AND conname LIKE 'lab_arena_recovery275_source_%_ck' "
                "ORDER BY conname"
            )
            assert cursor.fetchall() == [
                (
                    "lab_arena_recovery275_source_commit_ck",
                    "CHECK ((recovery_source_commit = "
                    f"'{RESEALED_SOURCE_COMMIT}'::text))",
                ),
                (
                    "lab_arena_recovery275_source_sha256_ck",
                    "CHECK ((recovery_source_sha256 = "
                    f"'{RESEALED_SOURCE_SHA}'::text))",
                ),
                (
                    "lab_arena_recovery275_source_size_ck",
                    f"CHECK ((recovery_source_size_bytes = {RESEALED_SOURCE_SIZE}))",
                ),
            ]
            cursor.execute(
                "SELECT relowner,relrowsecurity,relacl FROM pg_catalog.pg_class "
                "WHERE oid='public.lab_arena_sep16_baseline_recovery275_authority'"
                "::regclass"
            )
            assert cursor.fetchone() == acl_before
            cursor.execute(
                "SELECT pg_catalog.to_jsonb(row_value)-'authorized_at'-"
                "'recovery_source_size_bytes'-'recovery_source_sha256'-"
                "'recovery_source_commit' FROM "
                "public.lab_arena_sep16_baseline_recovery275_authority AS row_value"
            )
            assert cursor.fetchone()[0] == non_source_before
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_prepare_sep16_baseline_recovery275_v1("
                "bigint,text,text,jsonb)'::regprocedure)"
            )
            assert cursor.fetchone()[0] == function_before
            cursor.execute(
                "SELECT count(*) FROM "
                "public.lab_arena_sep16_baseline_recovery275_audit"
            )
            assert cursor.fetchone() == (0,)
        assert _state_seal(connection) == terminal_before

        with connection.cursor() as cursor, pytest.raises(
            Exception, match="source differs"
        ):
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep16_baseline_recovery275_v1("
                "%s,%s,%s,%s::jsonb)",
                (
                    RECOVERY_SOURCE_SIZE,
                    RECOVERY_SOURCE_SHA,
                    RECOVERY_SOURCE_COMMIT,
                    json.dumps(schedule275),
                ),
            )
        connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE public.lab_arena_sep16_baseline_recovery275_authority "
                "DROP CONSTRAINT lab_arena_recovery275_source_commit_ck"
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_sep16_baseline_recovery275_authority "
                "ADD CONSTRAINT lab_arena_recovery275_source_commit_ck "
                "CHECK (pg_catalog.length(recovery_source_commit) > 0)"
            )
            with pytest.raises(Exception, match="source authority or constraints differ"):
                cursor.execute(migration276)
        connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET cancel_reason='drift' "
                "WHERE round_id=%s",
                (ROUND,),
            )
            with pytest.raises(Exception, match="sealed terminal state differs"):
                cursor.execute(migration276)
        connection.rollback()

        prepared = _install_and_recover275(
            connection,
            migration275,
            schedule275,
            source_size=RESEALED_SOURCE_SIZE,
            source_sha=RESEALED_SOURCE_SHA,
            source_commit=RESEALED_SOURCE_COMMIT,
            install_sql=False,
        )
        assert prepared["baseline_execute_assignments"] == 20
        with connection.cursor() as cursor, pytest.raises(
            Exception, match="already started"
        ):
            cursor.execute(migration276)
        connection.rollback()
    finally:
        connection.close()


def test_recovery275_replays_exact_protected_terminal_snapshot_twice(connect):
    snapshot_path, prior_path = _protected_paths()
    captured = _load_protected_snapshot(snapshot_path, PROTECTED_SNAPSHOT_SHA256)
    prior_captured = _load_protected_snapshot(
        prior_path, PRIOR_PROTECTED_SNAPSHOT_SHA256
    )
    connection = connect()
    try:
        # Install the reviewed predecessor functions and tables, then replace
        # the synthetic rows with the one-statement protected production read.
        _prepare_terminal_recovery273(connection)
        _restore_protected_rows(connection, captured, prior_captured)
        seal = captured["seal"]
        restored_seal = _state_seal(connection)
        assert {key: restored_seal[key] for key in (
            "round", "baseline_submission", "baseline_runs", "baseline_ledger",
            "challenger_runs", "challenger_submissions", "challenger_ledger",
            "challenger_ledger_max",
        )} == {
            "round": seal["terminal_round_hash"],
            "baseline_submission": seal["terminal_baseline_submission_hash"],
            "baseline_runs": seal["terminal_baseline_runs_hash"],
            "baseline_ledger": seal["terminal_baseline_ledger_hash"],
            "challenger_runs": seal["challenger_runs_hash"],
            "challenger_submissions": seal["challenger_submissions_hash"],
            "challenger_ledger": seal["challenger_ledger_hash"],
            "challenger_ledger_max": seal["challenger_ledger_max_entry_id"],
        }
        protected = {
            table: _single_hash(connection, table, f"round_id='{ROUND}'")
            for table in (
                "lab_arena_sep16_baseline_rerun_audit",
                "lab_arena_sep16_rerun_release_authority",
                "lab_arena_sep16_baseline_recovery_authority",
                "lab_arena_sep16_baseline_recovery_audit",
                "lab_arena_sep16_baseline_recovery272_authority",
                "lab_arena_sep16_baseline_recovery272_audit",
                "lab_arena_sep16_baseline_recovery273_authority",
                "lab_arena_sep16_baseline_recovery273_audit",
            )
        }
        schedule = FORWARD_SCHEDULE
        migration = _render_protected_recovery275(captured, schedule)

        # A terminal ledger append after the snapshot must invalidate the seal.
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,"
                "round_id,submission_id,run_id,stage,call_identity,provider,"
                "operation_id,funding_source,amount_microusd,entry_doc,"
                "terminal_response) SELECT 'settlement',miner_hotkey,round_id,"
                "submission_id,run_id,stage,%s,provider,operation_id,funding_source,"
                "0,'{}'::jsonb,'{}'::jsonb FROM public.lab_arena_ledger "
                "WHERE round_id=%s AND submission_id=%s ORDER BY entry_id LIMIT 1",
                ("sha256:" + "9" * 64, ROUND, BASELINE),
            )
            with pytest.raises(Exception, match="terminal native rerun seal differs"):
                cursor.execute(migration)
        connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(RESEAL_MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(RESEAL_MIGRATION.read_text(encoding="utf-8"))
        connection.commit()

        wrong = (
            RECOVERY_SOURCE_SIZE,
            RECOVERY_SOURCE_SHA,
            RECOVERY_SOURCE_COMMIT,
            json.dumps(schedule),
        )
        with connection.cursor() as cursor, pytest.raises(
            Exception, match="source differs"
        ):
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep16_baseline_recovery275_v1("
                "%s,%s,%s,%s::jsonb)",
                wrong,
            )
        connection.rollback()

        arguments = (
            RESEALED_SOURCE_SIZE,
            RESEALED_SOURCE_SHA,
            RESEALED_SOURCE_COMMIT,
            json.dumps(schedule),
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep16_baseline_recovery275_v1("
                "%s,%s,%s,%s::jsonb)",
                arguments,
            )
            prepared = cursor.fetchone()[0]
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep16_baseline_recovery275_v1("
                "%s,%s,%s,%s::jsonb)",
                arguments,
            )
            assert cursor.fetchone()[0]["status"] == "existing"
        connection.commit()
        assert prepared["baseline_execute_assignments"] == 20
        assert prepared["archived_runs"] == 27
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_ledger "
                "WHERE round_id=%s AND submission_id=%s",
                (ARCHIVE_ROUND, ARCHIVE_SUBMISSION),
            )
            assert cursor.fetchone() == (4428,)
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),max(attempt) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s "
                "AND kind='execute'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20, 1)
            cursor.execute(
                "SELECT public.lab_arena_sep16_recovery275_archive_valid_v1(),"
                "public.lab_arena_sep16_recovery273_archive_valid_v1(),"
                "public.lab_arena_sep16_recovery272_archive_valid_v1(),"
                "public.lab_arena_sep16_recovery_archive_valid_v1()"
            )
            assert cursor.fetchone() == (True, True, True, True)
        for table, digest in protected.items():
            assert _single_hash(connection, table, f"round_id='{ROUND}'") == digest
    finally:
        connection.close()


def test_recovery275_preserves_history_and_publishes_positive(connect, tmp_path, monkeypatch):
    connection = connect()
    harness = Harness(connect, tmp_path, challengers=[], runners=["recovery275-proof"])
    objects = harness.objects
    service = harness.service
    icps = daily_icps()
    try:
        schedule272, hotkeys, ids = _prepare_terminal_recovery273(connection, objects)
        terminal_seal = _state_seal(connection)
        protected = {
            "rerun265_archive": _single_hash(
                connection, "lab_arena_sep16_baseline_recovery_audit", f"round_id='{ROUND}'"
            ),
            "recovery_authority": _single_hash(
                connection, "lab_arena_sep16_baseline_recovery_authority", f"round_id='{ROUND}'"
            ),
            "recovery273_authority": _single_hash(
                connection, "lab_arena_sep16_baseline_recovery273_authority", f"round_id='{ROUND}'"
            ),
            "recovery273_audit": _single_hash(
                connection, "lab_arena_sep16_baseline_recovery273_audit", f"round_id='{ROUND}'"
            ),
            "challenger_runs": terminal_seal["challenger_runs"],
            "challenger_submissions": terminal_seal["challenger_submissions"],
            "challenger_ledger": terminal_seal["challenger_ledger"],
        }
        schedule273 = _shift_schedule(schedule272, minutes=90)
        migration273 = _render_recovery275(connection, schedule273)
        with connection.cursor() as cursor:
            cursor.execute(migration273)
            cursor.execute(_render_recovery276(connection))
        connection.commit()
        prepared = _install_and_recover275(
            connection,
            migration273,
            schedule273,
            source_size=RESEALED_SOURCE_SIZE,
            source_sha=RESEALED_SOURCE_SHA,
            source_commit=RESEALED_SOURCE_COMMIT,
            install_sql=False,
        )
        assert prepared["archived_runs"] == 40
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT has_table_privilege('lab_arena_service',"
                "'public.lab_arena_sep16_baseline_recovery275_authority','SELECT')"
            )
            assert cursor.fetchone() == (False,)
            cursor.execute(
                "SELECT has_function_privilege('lab_arena_service',"
                "'public.lab_arena_prepare_sep16_baseline_recovery275_v1("
                "bigint,text,text,jsonb)','EXECUTE'),"
                "has_function_privilege('service_role',"
                "'public.lab_arena_prepare_sep16_baseline_recovery275_v1("
                "bigint,text,text,jsonb)','EXECUTE'),"
                "has_function_privilege('anon',"
                "'public.lab_arena_prepare_sep16_baseline_recovery275_v1("
                "bigint,text,text,jsonb)','EXECUTE'),"
                "has_function_privilege('authenticated',"
                "'public.lab_arena_prepare_sep16_baseline_recovery275_v1("
                "bigint,text,text,jsonb)','EXECUTE')"
            )
            assert cursor.fetchone() == (True, False, False, False)
        # The deployed 4738 daemon does not need the authority table or the
        # one-off admin RPC after recovery preparation completes. Once
        # preparation is complete, all execution, scoring, aggregation, and
        # publication use the existing generic runtime calls and DB guards.
        monkeypatch.delitem(
            FUNCTION_SIGNATURES,
            "lab_arena_prepare_sep16_baseline_recovery275_v1",
        )
        prepared_state = _state_seal(connection)
        for arguments, message in (
            (
                (RESEALED_SOURCE_SIZE + 1, RESEALED_SOURCE_SHA,
                 RESEALED_SOURCE_COMMIT, json.dumps(schedule273)),
                "source differs",
            ),
            (
                (RESEALED_SOURCE_SIZE, RESEALED_SOURCE_SHA,
                 RESEALED_SOURCE_COMMIT,
                 json.dumps({**schedule273, "stage_1_close": "changed"})),
                "schedule differs",
            ),
        ):
            with connection.cursor() as cursor, pytest.raises(Exception, match=message):
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep16_baseline_recovery275_v1("
                    "%s,%s,%s,%s::jsonb)",
                    arguments,
                )
            connection.rollback()
            assert _state_seal(connection) == prepared_state
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT p.oid::regprocedure::text,pg_get_functiondef(p.oid) "
                "FROM pg_proc p WHERE p.oid = "
                "ANY(ARRAY["
                "'public.lab_arena_open_scoring_sep16_baseline_only_v1(text,smallint,jsonb)'::regprocedure,"
                "'public.lab_arena_sep16_rerun_score_namespace_guard_v1()'::regprocedure,"
                "'public.lab_arena_open_sep16_baseline_scoring_v1(text,smallint,jsonb)'::regprocedure,"
                "'public.lab_arena_sep16_rerun_publication_guard_v1()'::regprocedure"
                "]) ORDER BY p.oid"
            )
            definitions = dict(cursor.fetchall())
            assert len(definitions) == 4
            expected_namespace_counts = {
                "lab_arena_open_scoring_sep16_baseline_only_v1(text,smallint,jsonb)": 1,
                "lab_arena_sep16_rerun_score_namespace_guard_v1()": 1,
                "lab_arena_open_sep16_baseline_scoring_v1(text,smallint,jsonb)": 3,
                "lab_arena_sep16_rerun_publication_guard_v1()": 6,
            }
            assert {
                name: definition.count(":rerun275")
                for name, definition in definitions.items()
            } == expected_namespace_counts
            assert all(":rerun273" not in definition for definition in definitions.values())
            publication = next(
                definition for definition in definitions.values()
                if "lab_arena_sep16_recovery275_archive_valid_v1" in definition
            )
            archive_clause = (
                "OR NOT public.lab_arena_sep16_recovery_archive_valid_v1()\n"
                "       OR NOT public.lab_arena_sep16_recovery272_archive_valid_v1()\n"
                "       OR NOT public.lab_arena_sep16_recovery273_archive_valid_v1()\n"
                "       OR NOT public.lab_arena_sep16_recovery275_archive_valid_v1()"
            )
            assert publication.count(archive_clause) == 1
            assert publication.count(
                "lab_arena_sep16_recovery_archive_valid_v1()"
            ) == 1
            assert publication.count(
                "lab_arena_sep16_recovery272_archive_valid_v1()"
            ) == 1
            assert publication.count(
                "lab_arena_sep16_recovery273_archive_valid_v1()"
            ) == 1
            assert publication.count(
                "lab_arena_sep16_recovery275_archive_valid_v1()"
            ) == 1
            assert "success_unresolved_calls" in publication
            assert "inflight_calls" in publication
            cursor.execute("SELECT public.lab_arena_sep16_recovery_archive_valid_v1()")
            assert cursor.fetchone() == (True,)
            cursor.execute("SELECT public.lab_arena_sep16_recovery273_archive_valid_v1()")
            assert cursor.fetchone() == (True,)
            cursor.execute("SELECT public.lab_arena_sep16_recovery275_archive_valid_v1()")
            assert cursor.fetchone() == (True,)
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),max(attempt) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s "
                "AND kind='execute'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20, 1)
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),max(attempt) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s "
                "AND kind='execute'",
                (ARCHIVE_ROUND, ARCHIVE_SUBMISSION),
            )
            assert cursor.fetchone() == (40, 20, 2)
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=%s "
                "AND submission_id=%s AND entry_doc->>"
                "'sep16_generic_http_catalog_reconciliation'='true'",
                (RERUN269_ARCHIVE, RERUN269_ARCHIVE_SUBMISSION),
            )
            assert cursor.fetchone() == (2,)
            cursor.execute(
                "SELECT source_ref,source_size_bytes,submission_doc->>'source_sha256',"
                "submission_doc->>'source_commit' FROM public.lab_arena_submissions "
                "WHERE submission_id=%s",
                (BASELINE,),
            )
            assert cursor.fetchone() == (
                RECOVERY_SOURCE_REF,
                RESEALED_SOURCE_SIZE,
                RESEALED_SOURCE_SHA,
                RESEALED_SOURCE_COMMIT,
            )
        assert _single_hash(
            connection, "lab_arena_sep16_baseline_recovery_audit", f"round_id='{ROUND}'"
        ) == protected["rerun265_archive"]
        assert _single_hash(
            connection, "lab_arena_sep16_baseline_recovery_authority", f"round_id='{ROUND}'"
        ) == protected["recovery_authority"]
        assert _single_hash(
            connection, "lab_arena_sep16_baseline_recovery273_authority", f"round_id='{ROUND}'"
        ) == protected["recovery273_authority"]
        assert _single_hash(
            connection, "lab_arena_sep16_baseline_recovery273_audit", f"round_id='{ROUND}'"
        ) == protected["recovery273_audit"]
        after_prepare = _state_seal(connection)
        assert after_prepare["challenger_runs"] == protected["challenger_runs"]
        assert (
            after_prepare["challenger_submissions"]
            == protected["challenger_submissions"]
        )
        assert after_prepare["challenger_ledger"] == protected["challenger_ledger"]
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT reward_basis_hash,configuration_doc->>'max_attempts_per_assignment',"
                "configuration_doc->>'parallel_twenty_icp_execution',"
                "configuration_doc->>'sourcing_cost_eligibility_policy' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (
                "sha256:b19a147f1c8cc8365e3cf7dcc96b827ffffdd6c1e8fad65ad9fc25c8895e493f",
                "2",
                "true",
                "successful_calls_v1",
            )

        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute(
                "SELECT run_id,icp_position FROM public.lab_arena_runs "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
                "ORDER BY icp_position",
                (ROUND, BASELINE),
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
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
        connection.commit()

        original_verified = service._verified_breakdowns

        def verified_fixture(run, *, icp, companies, policy):
            if run["submission_id"] != BASELINE:
                return original_verified(run, icp=icp, companies=companies, policy=policy)
            document = json.loads(objects.get(run["output_ref"]).decode())
            validated = scoring.validate_scoring_output_document(document)
            return scoring.validate_breakdowns_for_item(
                validated["breakdowns"], icp=icp, companies=companies,
                max_scored_companies=int(policy["max_scored_companies"]),
                integrity_policy=True, contacts_required=True,
            )

        monkeypatch.setattr(service, "_verified_breakdowns", verified_fixture)
        assert service.close_stage(ROUND, 1)["status"] == "ok"
        for stage in (1, 2):
            items = _service_scoring_items(connection, stage, hotkeys)
            for item in items:
                if item["submission_id"] == BASELINE:
                    item["judgment_scope_doc"]["scorer_image_digest"] = NEW_IMAGE
                    item["judgment_scope_doc"]["scorer_image_reference"] = NEW_IMAGE_REF
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT public.lab_arena_open_sep16_baseline_scoring_v1("
                    "%s,%s::smallint,%s::jsonb)",
                    (ROUND, stage, json.dumps(items)),
                )
                assert cursor.fetchone()[0]["assignments"] == 10
            connection.commit()
            _accept_recovery275_scores(connection, objects, stage, icps)
            connection.commit()
            assert service.close_scoring(ROUND, stage)["status"] == "closed"
            assert service.score_stage(ROUND, stage)["status"] == "ok"
            if stage == 1:
                assert service.open_stage(ROUND, 2)["status"] == "ok"
                assert service.close_stage(ROUND, 2)["status"] == "ok"
        published = service.publish(ROUND)
        assert published["status"] == "ok"
        row = service.store.get_round(ROUND)
        baseline = next(
            item for item in row["publication_doc"]["final_ranking"]
            if item["submission_id"] == BASELINE
        )
        assert baseline["final_score"] > 0
        assert row["status"] == "published"
        assert ids[1:] == row["finalists"]
    finally:
        connection.close()


@pytest.mark.parametrize("drift", ("namespace", "archive-clause", "prior-duplicate"))
def test_recovery275_refuses_partial_function_rewrite(connect, drift):
    connection = connect()
    try:
        schedule272, _hotkeys, _ids = _prepare_terminal_recovery273(connection)
        schedule273 = _shift_schedule(schedule272, minutes=90)
        migration273 = _render_recovery275(connection, schedule273)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_sep16_rerun_publication_guard_v1()'::regprocedure)"
            )
            definition = cursor.fetchone()[0]
            if drift == "namespace":
                changed = definition.replace(":rerun273", ":rerun275", 1)
            elif drift == "archive-clause":
                old = (
                    "OR NOT public.lab_arena_sep16_recovery_archive_valid_v1()"
                )
                changed = definition.replace(
                    old,
                    old + "\n       OR pg_catalog.quote_literal("
                    "'lab_arena_sep16_recovery275_archive_valid_v1()') = ''",
                    1,
                )
            else:
                old = "OR NOT public.lab_arena_sep16_recovery273_archive_valid_v1()"
                changed = definition.replace(old, old + "\n       " + old, 1)
            assert changed != definition
            cursor.execute(changed)
        connection.commit()
        with connection.cursor() as cursor, pytest.raises(
            Exception,
            match=(
                "namespace function shape differs|archive guard shape differs|"
                "recovery275 guard shape differs"
            ),
        ):
            cursor.execute(migration273)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('public.lab_arena_sep16_baseline_recovery275_audit')"
            )
            assert cursor.fetchone() == (None,)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "drift", ("active", "success-unresolved", "prior-archive", "prior272-archive")
)
def test_recovery275_seal_fails_closed_on_terminal_drift(connect, drift):
    connection = connect()
    try:
        schedule272, _hotkeys, _ids = _prepare_terminal_recovery273(connection)
        if drift == "active":
            with connection.cursor() as cursor:
                cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
                cursor.execute(
                    "UPDATE public.lab_arena_runs SET status='pending',terminal_cause=NULL "
                    "WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND submission_id=%s)",
                    (ROUND, BASELINE),
                )
                cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
            connection.commit()
        elif drift == "success-unresolved":
            with connection.cursor() as cursor:
                cursor.execute("ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER")
                identity = "sha256:" + "7" * 64
                common = (
                    "miner_hotkey,round_id,submission_id,run_id,stage,call_identity,"
                    "provider,operation_id,funding_source,amount_microusd,entry_doc"
                )
                for kind, amount, document in (
                    ("reservation", 1000, {}),
                    ("dispatch", 0, {}),
                    (
                        "uncertain",
                        0,
                        {
                            "reason": "worker_reported",
                            "call": {
                                "reason": "missing_provider_cost",
                                "call_succeeded": True,
                            },
                        },
                    ),
                ):
                    cursor.execute(
                        "INSERT INTO public.lab_arena_ledger (entry_kind," + common + ") "
                        "SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,"
                        "%s,'deepline','deepline.execute','host',%s,%s::jsonb "
                        "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s "
                        "ORDER BY run_id LIMIT 1",
                        (kind, identity, amount, json.dumps(document), ROUND, BASELINE),
                    )
                cursor.execute("ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER")
                cursor.execute(
                    "SELECT public.lab_arena__successful_call_cost_state(%s,'execute',NULL)",
                    (BASELINE,),
                )
                assert cursor.fetchone()[0]["success_unresolved_calls"] > 0
            connection.commit()
        schedule273 = _shift_schedule(schedule272, minutes=90)
        migration273 = _render_recovery275(connection, schedule273)
        if drift in ("prior-archive", "prior272-archive"):
            with connection.cursor() as cursor:
                cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
                cursor.execute(
                    "UPDATE public.lab_arena_runs SET terminal_cause='model_error' "
                    "WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs "
                    "WHERE round_id=%s)",
                    (
                        RERUN265_ARCHIVE
                        if drift == "prior-archive"
                        else RERUN269_ARCHIVE,
                    ),
                )
                cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
            connection.commit()
        with connection.cursor() as cursor, pytest.raises(Exception):
            cursor.execute(migration273)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('public.lab_arena_sep16_baseline_recovery275_audit')"
            )
            assert cursor.fetchone() == (None,)
    finally:
        connection.close()


def test_recovery275_uses_normal_two_attempt_limit(connect, tmp_path):
    connection = connect()
    harness = Harness(connect, tmp_path, challengers=[], runners=["retry273-proof"])
    store = harness.service.store
    try:
        schedule272, hotkeys, _ids = _prepare_terminal_recovery273(connection)
        schedule273 = _shift_schedule(schedule272, minutes=90)
        _install_and_recover275(
            connection, _render_recovery275(connection, schedule273), schedule273
        )
        first, token, _request_id, _request_hash = claim(
            store, ROUND, hotkeys[5], parallelism=1, ceiling=10,
            excluded=[hotkeys[5]],
        )
        assert first["attempt"] == 1
        assert first["assignment_id"].endswith(":rerun275")
        assert store.complete_attempt(
            run_id=first["run_id"], lease_token_hash=hash_lease_token(token),
            result={"terminal_status": "provider_error"},
            terminal_cause="provider_error", output_ref="",
        )["status"] == "failed"
        second, second_token, _request_id, _request_hash = claim(
            store, ROUND, hotkeys[5], parallelism=1, ceiling=10,
            excluded=[hotkeys[5]],
        )
        assert second["assignment_id"] == first["assignment_id"]
        assert second["attempt"] == 2
        assert store.complete_attempt(
            run_id=second["run_id"], lease_token_hash=hash_lease_token(second_token),
            result={"terminal_status": "model_error"},
            terminal_cause="model_error", output_ref="",
        )["status"] == "failed"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),max(attempt) FROM public.lab_arena_runs "
                "WHERE assignment_id=%s",
                (first["assignment_id"],),
            )
            assert cursor.fetchone() == (2, 2)
    finally:
        connection.close()
