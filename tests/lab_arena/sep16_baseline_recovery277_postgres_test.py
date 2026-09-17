"""PostgreSQL proof for the exact Sep16 recovery277 quota transition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.sep16_baseline_recovery275_postgres_test import (
    DEFAULT_PRIOR_PROTECTED_SNAPSHOT,
    DEFAULT_PROTECTED_SNAPSHOT,
    MIGRATION as RECOVERY275_MIGRATION,
    MIGRATIONS,
    PRIOR_PROTECTED_SNAPSHOT_SHA256,
    PROTECTED_SNAPSHOT_SHA256,
    RESEAL_MIGRATION,
    _load_protected_snapshot,
    _prepare_terminal_recovery273,
    _restore_protected_rows,
)
from tests.lab_arena.sep16_native_baseline_rerun_postgres_test import (
    BASELINE,
    ROUND,
    _state_seal,
)


ARCHIVE_ROUND = "arena-2026-09-16-rerun275archive"
ARCHIVE_SUBMISSION = "baseline-2026-09-16-native-rerun275-archive"
SOURCE_REF = (
    "arena/arena-2026-09-16/sources/"
    "baseline-2026-09-16-recovery277.tar.gz"
)
SOURCE_SIZE = 561102
SOURCE_SHA = "962ef646cd63e51e1f7fcdac85cc47a510389c42ce4ed787b685927bbf436425"
SOURCE_COMMIT = "f46e03cc8dba1d1b7ca0292bad0e1a71f849b4d7"
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/277-arena-2026-09-16-baseline-recovery.sql.template"
)
MIGRATION = TEMPLATE.with_suffix("")
FORWARD_SCHEDULE = {
    "submission_open": "2026-09-15T00:00:00Z",
    "submission_cutoff": "2026-09-16T00:00:00Z",
    "benchmark_deadline": "2026-09-17T04:00:00Z",
    "stage_1_start": "2026-09-17T04:00:01Z",
    "stage_1_close": "2026-09-17T12:00:00Z",
    "stage_1_scoring_close": "2026-09-17T14:00:00Z",
    "stage_2_start": "2026-09-17T14:05:00Z",
    "stage_2_close": "2026-09-17T14:20:00Z",
    "final_scoring_close": "2026-09-17T16:20:00Z",
    "publication_deadline": "2026-09-17T16:50:00Z",
}
CURRENT_SNAPSHOT = Path(
    "/private/tmp/leadpoet-tyche-rerun-evidence-20260915/private/"
    "sep16-recovery277-terminal-snapshot-20260917T031200.178092Z.json"
)
CURRENT_SNAPSHOT_SHA256 = (
    "00f8d0ff327b5248b23d73ca1f35823984fb05fa68f082109d32eaebc4064f12"
)


@pytest.fixture
def connect():
    generator = database_with_lab_arena_migration(MIGRATIONS)
    psycopg2, dsn = next(generator)
    try:
        yield lambda: psycopg2.connect(**dsn)
    finally:
        generator.close()


def _insert_rows(cursor, table: str, documents: list[dict]) -> None:
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


def _restore_current_recovery275_rows(connection, captured: dict) -> None:
    rows = captured["protected_rows"]
    authority_rows = {
        "lab_arena_sep16_baseline_rerun_audit": rows["prior_rerun_audit"],
        "lab_arena_sep16_rerun_release_authority": rows["release_authority"],
        "lab_arena_sep16_baseline_recovery_authority": rows["recovery_authority"],
        "lab_arena_sep16_baseline_recovery_audit": rows["recovery_audit"],
        "lab_arena_sep16_baseline_recovery275_authority": rows[
            "recovery275_authority"
        ],
        "lab_arena_sep16_baseline_recovery275_audit": rows["recovery275_audit"],
    }
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
        for table in authority_rows:
            cursor.execute("TRUNCATE public." + table)
        for table, key in (
            ("lab_arena_rounds", "rounds"),
            ("lab_arena_submissions", "submissions"),
            ("lab_arena_runs", "runs"),
            ("lab_arena_ledger", "ledger"),
        ):
            decoded = [json.loads(value) for value in rows[key]]
            for start in range(0, len(decoded), 500):
                _insert_rows(cursor, table, decoded[start : start + 500])
        for table, value in authority_rows.items():
            _insert_rows(cursor, table, [json.loads(value)])
        cursor.execute(
            "SELECT pg_catalog.setval('public.lab_arena_ledger_entry_id_seq',"
            "(SELECT max(entry_id) FROM public.lab_arena_ledger),true)"
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _forward_schedule(old_schedule: dict[str, str]) -> dict[str, str]:
    assert FORWARD_SCHEDULE["submission_open"] == old_schedule["submission_open"]
    assert FORWARD_SCHEDULE["submission_cutoff"] == old_schedule["submission_cutoff"]
    return dict(FORWARD_SCHEDULE)


def _render(captured: dict, schedule: dict[str, str]) -> str:
    seal = captured["seal"]
    values = {
        "__SEALED_RECOVERY_SOURCE_SIZE_BYTES__": str(SOURCE_SIZE),
        "__SEALED_RECOVERY_SOURCE_SHA256__": SOURCE_SHA,
        "__SEALED_RECOVERY_SOURCE_COMMIT__": SOURCE_COMMIT,
        "__SEALED_TERMINAL_SOURCE_SIZE_BYTES__": "556500",
        "__SEALED_TERMINAL_SOURCE_SHA256__": (
            "c86df33ae0f21164b46f9ab6b8e7dbcb116b7a53c49dbc43603a49b04067d104"
        ),
        "__SEALED_TERMINAL_SOURCE_COMMIT__": (
            "2b8386b835de02739ddef6c182bdeba43c6ad467"
        ),
        "__SEALED_TERMINAL_ROUND_HASH__": seal["terminal_round_hash"],
        "__SEALED_TERMINAL_BASELINE_SUBMISSION_HASH__": seal[
            "terminal_baseline_submission_hash"
        ],
        "__SEALED_TERMINAL_BASELINE_RUNS_HASH__": seal[
            "terminal_baseline_runs_hash"
        ],
        "__SEALED_TERMINAL_BASELINE_LEDGER_HASH__": seal[
            "terminal_baseline_ledger_hash"
        ],
        "__SEALED_TERMINAL_BASELINE_RUN_COUNT__": str(
            seal["terminal_baseline_run_count"]
        ),
        "__SEALED_TERMINAL_BASELINE_LEDGER_COUNT__": str(
            seal["terminal_baseline_ledger_count"]
        ),
        "__SEALED_TERMINAL_BASELINE_LEDGER_MAX_ENTRY_ID__": str(
            seal["terminal_baseline_ledger_max_entry_id"]
        ),
        "__SEALED_TERMINAL_BASELINE_SETTLED_MICROUSD__": str(
            seal["terminal_baseline_settled_microusd"]
        ),
        "__SEALED_TERMINAL_BASELINE_UNCERTAIN_MICROUSD__": str(
            seal["terminal_baseline_uncertain_microusd"]
        ),
        "__SEALED_CHALLENGER_RUNS_HASH__": seal["challenger_runs_hash"],
        "__SEALED_CHALLENGER_SUBMISSIONS_HASH__": seal[
            "challenger_submissions_hash"
        ],
        "__SEALED_CHALLENGER_LEDGER_HASH__": seal["challenger_ledger_hash"],
        "__SEALED_CHALLENGER_LEDGER_MAX_ENTRY_ID__": str(
            seal["challenger_ledger_max_entry_id"]
        ),
        "__SEALED_PRIOR_RERUN_AUDIT_HASH__": seal["prior_rerun_audit_hash"],
        "__SEALED_RELEASE_AUTHORITY_HASH__": seal["release_authority_hash"],
        "__SEALED_PRIOR_RECOVERY_AUTHORITY_HASH__": seal[
            "prior_recovery_authority_hash"
        ],
        "__SEALED_PRIOR_RECOVERY_AUDIT_HASH__": seal[
            "prior_recovery_audit_hash"
        ],
        "__SEALED_TERMINAL_STATUS_GENERATION__": str(
            seal["terminal_status_generation"]
        ),
        "__SEALED_TERMINAL_STAGE_GENERATION__": str(
            seal["terminal_stage_generation"]
        ),
        "__SEALED_TERMINAL_CANCEL_REASON__": seal["terminal_cancel_reason"],
        "__SEALED_OLD_FORWARD_SCHEDULE_JSON__": json.dumps(
            seal["old_schedule"], sort_keys=True, separators=(",", ":")
        ),
        "__SEALED_NEW_FORWARD_SCHEDULE_JSON__": json.dumps(
            schedule, sort_keys=True, separators=(",", ":")
        ),
    }
    rendered = TEMPLATE.read_text(encoding="utf-8")
    for marker, value in values.items():
        assert rendered.count(marker) >= 1, marker
        rendered = rendered.replace(marker, value)
    assert "__SEALED_" not in rendered
    return rendered


def test_recovery277_replays_cancelled_275_and_changes_only_schedule_and_quota(
    connect,
):
    assert hashlib.sha256(CURRENT_SNAPSHOT.read_bytes()).hexdigest() == (
        CURRENT_SNAPSHOT_SHA256
    )
    current = _load_protected_snapshot(CURRENT_SNAPSHOT, CURRENT_SNAPSHOT_SHA256)
    prior275 = _load_protected_snapshot(
        DEFAULT_PROTECTED_SNAPSHOT, PROTECTED_SNAPSHOT_SHA256
    )
    prior273 = _load_protected_snapshot(
        DEFAULT_PRIOR_PROTECTED_SNAPSHOT, PRIOR_PROTECTED_SNAPSHOT_SHA256
    )
    connection = connect()
    try:
        _prepare_terminal_recovery273(connection)
        _restore_protected_rows(connection, prior275, prior273)
        with connection.cursor() as cursor:
            cursor.execute(RECOVERY275_MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(RESEAL_MIGRATION.read_text(encoding="utf-8"))
        connection.commit()
        _restore_current_recovery275_rows(connection, current)

        restored = _state_seal(connection)
        seal = current["seal"]
        assert restored["round"] == seal["terminal_round_hash"]
        assert restored["baseline_submission"] == seal[
            "terminal_baseline_submission_hash"
        ]
        assert restored["baseline_runs"] == seal["terminal_baseline_runs_hash"]
        assert restored["baseline_ledger"] == seal["terminal_baseline_ledger_hash"]

        schedule = _forward_schedule(seal["old_schedule"])
        migration = _render(current, schedule)
        assert migration == MIGRATION.read_text(encoding="utf-8")
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE OR REPLACE FUNCTION "
                "public.lab_arena__successful_call_cost_state("
                "p_submission_id TEXT,p_kind TEXT,p_provider TEXT DEFAULT NULL) "
                "RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER "
                "SET search_path=pg_catalog,public AS $$ SELECT '{}'::jsonb $$"
            )
            with pytest.raises(Exception, match="terminal native rerun seal differs"):
                cursor.execute(migration)
        connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT configuration_doc,reward_basis_hash "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            old_configuration, reward_basis = cursor.fetchone()
            assert old_configuration["call_quotas"]["openrouter"] == 60

            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET configuration_doc="
                "jsonb_set(configuration_doc,'{call_quotas,openrouter}','200') "
                "WHERE round_id=%s",
                (ROUND,),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
            with pytest.raises(Exception, match="terminal native rerun seal differs"):
                cursor.execute(migration)
        connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute(migration)
            cursor.execute(migration)
            arguments = (SOURCE_SIZE, SOURCE_SHA, SOURCE_COMMIT, json.dumps(schedule))
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep16_baseline_recovery277_v1("
                "%s,%s,%s,%s::jsonb)",
                arguments,
            )
            prepared = cursor.fetchone()[0]
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep16_baseline_recovery277_v1("
                "%s,%s,%s,%s::jsonb)",
                arguments,
            )
            assert cursor.fetchone()[0]["status"] == "existing"
        connection.commit()

        assert prepared["status"] == "prepared"
        assert prepared["baseline_execute_assignments"] == 20
        assert prepared["openrouter_calls_per_icp"] == 200
        arguments = (SOURCE_SIZE, SOURCE_SHA, SOURCE_COMMIT, json.dumps(schedule))
        for path, value in (
            ("{call_quotas,openrouter}", "199"),
            ("{schedule,benchmark_deadline}", '"2026-09-17T04:01:00Z"'),
        ):
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL session_replication_role=replica")
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET configuration_doc="
                    "jsonb_set(configuration_doc,%s,%s::jsonb) WHERE round_id=%s",
                    (path, value, ROUND),
                )
                cursor.execute("SET LOCAL session_replication_role=origin")
                with pytest.raises(Exception, match="replay differs"):
                    cursor.execute(
                        "SELECT public.lab_arena_prepare_sep16_baseline_recovery277_v1("
                        "%s,%s,%s,%s::jsonb)",
                        arguments,
                    )
            connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT configuration_doc,reward_basis_hash "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            new_configuration, new_reward_basis = cursor.fetchone()
            assert new_configuration["call_quotas"]["openrouter"] == 200
            assert new_configuration["schedule"] == schedule
            assert {
                key: value
                for key, value in new_configuration.items()
                if key not in {"schedule", "call_quotas"}
            } == {
                key: value
                for key, value in old_configuration.items()
                if key not in {"schedule", "call_quotas"}
            }
            assert {
                key: value
                for key, value in new_configuration["call_quotas"].items()
                if key != "openrouter"
            } == {
                key: value
                for key, value in old_configuration["call_quotas"].items()
                if key != "openrouter"
            }
            assert new_reward_basis == reward_basis
            cursor.execute(
                "SELECT bank_sha256 FROM "
                "public.lab_arena_sep16_baseline_recovery277_authority "
                "WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (
                "42b417604acf15e612570687e8fbccef82fe7b0808364a6491b4a2a4bcb07390",
            )
            cursor.execute(
                "SELECT configuration_doc#>>'{call_quotas,openrouter}' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ARCHIVE_ROUND,),
            )
            assert cursor.fetchone() == ("60",)
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),max(attempt) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s "
                "AND kind='execute' AND status='pending' "
                "AND assignment_id LIKE '%%:rerun277'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20, 1)
            cursor.execute(
                "SELECT source_ref,source_size_bytes,submission_doc->>'source_sha256',"
                "submission_doc->>'source_commit' FROM public.lab_arena_submissions "
                "WHERE round_id=%s AND submission_id=%s",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (SOURCE_REF, SOURCE_SIZE, SOURCE_SHA, SOURCE_COMMIT)
            cursor.execute(
                "SELECT count(*),coalesce(sum(amount_microusd) FILTER "
                "(WHERE entry_kind='settlement'),0),coalesce(sum(amount_microusd) "
                "FILTER (WHERE entry_kind='uncertain'),0) FROM public.lab_arena_ledger "
                "WHERE round_id=%s AND submission_id=%s",
                (ARCHIVE_ROUND, ARCHIVE_SUBMISSION),
            )
            assert cursor.fetchone() == (
                seal["terminal_baseline_ledger_count"],
                seal["terminal_baseline_settled_microusd"],
                seal["terminal_baseline_uncertain_microusd"],
            )
            cursor.execute(
                "SELECT public.lab_arena_sep16_recovery277_archive_valid_v1(),"
                "public.lab_arena_sep16_recovery275_archive_valid_v1(),"
                "public.lab_arena_sep16_recovery273_archive_valid_v1(),"
                "public.lab_arena_sep16_recovery272_archive_valid_v1(),"
                "public.lab_arena_sep16_recovery_archive_valid_v1()"
            )
            assert cursor.fetchone() == (True, True, True, True, True)
    finally:
        connection.close()
