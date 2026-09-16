"""PostgreSQL proof for the exact Sep16 failed-baseline recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import scoring
from lab_arena.store import hash_lease_token
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.sep16_latest_champion_reseal_postgres_test import (
    NEW_COMMIT,
    NEW_IMAGE,
    NEW_IMAGE_REF,
    NEW_SHA,
    NEW_SIZE,
    _insert_owner266_authority,
    _render_reseal,
    _shift_schedule,
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
from tests.lab_arena.test_lab_arena_service_round import Harness
from tests.lab_arena.test_lab_arena_migration_postgres import claim


ARCHIVE_ROUND = "arena-2026-09-16-rerun265archive"
ARCHIVE_SUBMISSION = "baseline-2026-09-16-native-rerun265-archive"
TERMINAL_SOURCE_REF = (
    "arena/arena-2026-09-16/sources/"
    "baseline-2026-09-16-native-rerun268.tar.gz"
)
RECOVERY_SOURCE_REF = (
    "arena/arena-2026-09-16/sources/baseline-2026-09-16-recovery269.tar.gz"
)
RECOVERY_SOURCE_SIZE = 525000
RECOVERY_SOURCE_SHA = "d" * 64
RECOVERY_SOURCE_COMMIT = "e" * 40
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/269-arena-2026-09-16-baseline-recovery.sql.template"
)


@pytest.fixture
def connect():
    generator = database_with_lab_arena_migration(MIGRATIONS)
    psycopg2, dsn = next(generator)
    try:
        yield lambda: psycopg2.connect(**dsn)
    finally:
        generator.close()


def _prepare_latest(connection, old_schedule):
    seal = _state_seal(connection)
    _insert_owner266_authority(connection, old_schedule, seal)
    prepared_schedule = _shift_schedule(old_schedule)
    reseal = _render_reseal(connection, old_schedule, prepared_schedule)
    with connection.cursor() as cursor:
        cursor.execute(reseal)
        cursor.execute(
            "SELECT public.lab_arena_prepare_sep16_baseline_rerun_v1("
            "%s,%s,%s,%s,%s::jsonb)",
            (
                NEW_SIZE,
                NEW_SHA,
                NEW_COMMIT,
                "42b417604acf15e612570687e8fbccef82fe7b0808364a6491b4a2a4bcb07390",
                json.dumps(prepared_schedule),
            ),
        )
        assert cursor.fetchone()[0]["status"] == "prepared"
    connection.commit()
    return prepared_schedule


def _fail_both_attempts_and_cancel(connection):
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',"
            "terminal_cause='provider_error' WHERE round_id=%s "
            "AND submission_id=%s AND kind='execute' AND status='pending' "
            "RETURNING assignment_id,miner_hotkey,stage,icp_position,"
            "stage_generation",
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
                    assignment + ":2",
                    assignment,
                    ROUND,
                    BASELINE,
                    hotkey,
                    stage,
                    position,
                    generation,
                ),
            )
        cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
        failed_run = first_attempts[0][0] + ":2"
        call_identity = "sha256:" + "9" * 64
        request_id = "ctx-tool-" + call_identity[7:39]
        credential = "sha256:" + "a" * 64
        common = (
            "miner_hotkey,round_id,submission_id,run_id,stage,call_identity,"
            "provider,operation_id,funding_source,amount_microusd,entry_doc"
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger (entry_kind," + common + ") "
            "SELECT 'reservation',miner_hotkey,round_id,submission_id,run_id,"
            "stage,%s,'deepline','deepline.execute','host',5000000,%s::jsonb "
            "FROM public.lab_arena_runs WHERE run_id=%s",
            (
                call_identity,
                json.dumps(
                    {
                        "deepline_request_id": request_id,
                        "tool": "find_companies",
                        "credential_fingerprint": credential,
                    }
                ),
                failed_run,
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger (entry_kind," + common + ") "
            "SELECT 'dispatch',miner_hotkey,round_id,submission_id,run_id,stage,"
            "%s,'deepline','deepline.execute','host',0,'{}'::jsonb "
            "FROM public.lab_arena_runs WHERE run_id=%s",
            (call_identity, failed_run),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger (entry_kind," + common + ") "
            "SELECT 'uncertain',miner_hotkey,round_id,submission_id,run_id,stage,"
            "%s,'deepline','deepline.execute','host',7000,%s::jsonb "
            "FROM public.lab_arena_runs WHERE run_id=%s",
            (
                call_identity,
                json.dumps(
                    {
                        "reason": "worker_reported",
                        "call": {
                            "reason": "missing_provider_cost",
                            "call_succeeded": False,
                            "deepline_request_id": request_id,
                            "deepline_operation": "find_companies",
                            "credential_fingerprint": credential,
                        },
                    }
                ),
                failed_run,
            ),
        )
        cursor.execute(
            "SELECT public.lab_arena_close_parallel_execution_v1(%s)", (ROUND,)
        )
        closed = cursor.fetchone()[0]
        assert closed["status"] == "cancelled"
        assert closed["incomplete_assignments"] == 20
    connection.commit()


def _row_hash(connection, table, predicate, order_by, *, strip_ids=False):
    expression = "to_jsonb(row_value)"
    if strip_ids:
        expression += " - 'round_id' - 'submission_id'"
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 'sha256:' || encode(extensions.digest(COALESCE("
            "string_agg(encode(extensions.digest((%s)::text,'sha256'),'hex'),'' "
            "ORDER BY %s),''),'sha256'),'hex') FROM public.%s AS row_value "
            "WHERE %s" % (expression, order_by, table, predicate)
        )
        return cursor.fetchone()[0]


def _single_hash(connection, table, predicate):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT 'sha256:' || encode(extensions.digest(to_jsonb(row_value)::text,"
            "'sha256'),'hex') FROM public.%s AS row_value WHERE %s"
            % (table, predicate)
        )
        return cursor.fetchone()[0]


def _render_recovery(connection, new_schedule):
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
            "FROM public.lab_arena_ledger "
            "WHERE round_id=%s AND submission_id=%s",
            (ROUND, BASELINE),
        )
        ledger_count, ledger_max, settled, uncertain = cursor.fetchone()
        cursor.execute(
            "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
            "AND submission_id=%s",
            (ROUND, BASELINE),
        )
        run_count = cursor.fetchone()[0]
    values = {
        "__SEALED_RECOVERY_SOURCE_SIZE_BYTES__": str(RECOVERY_SOURCE_SIZE),
        "__SEALED_RECOVERY_SOURCE_SHA256__": RECOVERY_SOURCE_SHA,
        "__SEALED_RECOVERY_SOURCE_COMMIT__": RECOVERY_SOURCE_COMMIT,
        "__SEALED_TERMINAL_ROUND_HASH__": seal["round"],
        "__SEALED_TERMINAL_BASELINE_SUBMISSION_HASH__": seal[
            "baseline_submission"
        ],
        "__SEALED_TERMINAL_BASELINE_RUNS_HASH__": seal["baseline_runs"],
        "__SEALED_TERMINAL_BASELINE_LEDGER_HASH__": seal["baseline_ledger"],
        "__SEALED_TERMINAL_BASELINE_RUN_COUNT__": str(run_count),
        "__SEALED_TERMINAL_BASELINE_LEDGER_COUNT__": str(ledger_count),
        "__SEALED_TERMINAL_BASELINE_LEDGER_MAX_ENTRY_ID__": str(ledger_max),
        "__SEALED_TERMINAL_BASELINE_SETTLED_MICROUSD__": str(settled),
        "__SEALED_TERMINAL_BASELINE_UNCERTAIN_MICROUSD__": str(uncertain),
        "__SEALED_CHALLENGER_RUNS_HASH__": seal["challenger_runs"],
        "__SEALED_CHALLENGER_SUBMISSIONS_HASH__": seal[
            "challenger_submissions"
        ],
        "__SEALED_CHALLENGER_LEDGER_HASH__": seal["challenger_ledger"],
        "__SEALED_CHALLENGER_LEDGER_MAX_ENTRY_ID__": str(
            seal["challenger_ledger_max"]
        ),
        "__SEALED_PRIOR_RERUN_AUDIT_HASH__": _single_hash(
            connection,
            "lab_arena_sep16_baseline_rerun_audit",
            "round_id='%s'" % ROUND,
        ),
        "__SEALED_RELEASE_AUTHORITY_HASH__": _single_hash(
            connection,
            "lab_arena_sep16_rerun_release_authority",
            "round_id='%s'" % ROUND,
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
    rendered = TEMPLATE.read_text(encoding="utf-8")
    for marker, value in values.items():
        assert rendered.count(marker) >= 1
        rendered = rendered.replace(marker, value)
    assert "__SEALED_" not in rendered
    return rendered


def _install_and_recover(connection, sql, schedule):
    with connection.cursor() as cursor:
        cursor.execute(sql)
        cursor.execute(
            "SELECT public.lab_arena_prepare_sep16_baseline_recovery_v1("
            "%s,%s,%s,%s::jsonb)",
            (
                RECOVERY_SOURCE_SIZE,
                RECOVERY_SOURCE_SHA,
                RECOVERY_SOURCE_COMMIT,
                json.dumps(schedule),
            ),
        )
        prepared = cursor.fetchone()[0]
        cursor.execute(
            "SELECT public.lab_arena_prepare_sep16_baseline_recovery_v1("
            "%s,%s,%s,%s::jsonb)",
            (
                RECOVERY_SOURCE_SIZE,
                RECOVERY_SOURCE_SHA,
                RECOVERY_SOURCE_COMMIT,
                json.dumps(schedule),
            ),
        )
        existing = cursor.fetchone()[0]
    connection.commit()
    assert prepared["status"] == "prepared"
    assert existing["status"] == "existing"
    return prepared


def _accept_recovery_scores(connection, objects, stage, icps):
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
        cursor.execute(
            "SELECT run_id,scored_run_id,icp_position FROM public.lab_arena_runs "
            "WHERE round_id=%s AND submission_id=%s AND kind='score' AND stage=%s "
            "AND assignment_id LIKE '%%:score:rerun269' ORDER BY icp_position",
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
                json.dumps(
                    scoring.build_scoring_output(scored_run_id, [breakdown])
                ).encode(),
            )
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "terminal_cause='accepted',output_ref=%s WHERE run_id=%s",
                (ref, run_id),
            )
        cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")


def test_recovery_archives_failure_and_publishes_positive_with_all_guards(
    connect, tmp_path, monkeypatch
):
    connection = connect()
    harness = Harness(connect, tmp_path, challengers=[], runners=["recovery-proof"])
    objects = harness.objects
    service = harness.service
    icps = daily_icps()
    try:
        old_schedule, hotkeys, ids = _seed_observed_sep16(connection, objects)
        _prepare_latest(connection, old_schedule)
        _fail_both_attempts_and_cancel(connection)
        recovery_schedule = _shift_schedule(old_schedule, minutes=45)
        migration = _render_recovery(connection, recovery_schedule)
        protected_prior_audit = _single_hash(
            connection,
            "lab_arena_sep16_baseline_rerun_audit",
            "round_id='%s'" % ROUND,
        )
        prepared = _install_and_recover(
            connection, migration, recovery_schedule
        )
        assert prepared["archived_runs"] == 40
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_functiondef(p.oid) FROM pg_proc p WHERE p.oid = "
                "ANY(ARRAY["
                "'public.lab_arena_open_scoring_sep16_baseline_only_v1(text,smallint,jsonb)'::regprocedure,"
                "'public.lab_arena_sep16_rerun_score_namespace_guard_v1()'::regprocedure,"
                "'public.lab_arena_open_sep16_baseline_scoring_v1(text,smallint,jsonb)'::regprocedure,"
                "'public.lab_arena_sep16_rerun_publication_guard_v1()'::regprocedure"
                "]) ORDER BY p.oid"
            )
            definitions = [row[0] for row in cursor.fetchall()]
            assert len(definitions) == 4
            assert all(":rerun269" in definition for definition in definitions)
            assert all(":rerun265" not in definition for definition in definitions)
            publication_definition = next(
                definition
                for definition in definitions
                if "lab_arena_sep16_recovery_archive_valid_v1" in definition
            )
            assert "success_unresolved_calls" in publication_definition
            assert "inflight_calls" in publication_definition
            assert "(v_execute_cost ->> 'uncertain_calls')" not in (
                publication_definition
            )
            cursor.execute(
                "SELECT configuration_doc->>'max_attempts_per_assignment',"
                "configuration_doc->>'parallel_twenty_icp_execution' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == ("2", "true")
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),max(attempt) "
                "FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20, 1)
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),max(attempt) "
                "FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='execute'",
                (ARCHIVE_ROUND, ARCHIVE_SUBMISSION),
            )
            assert cursor.fetchone() == (40, 20, 2)
            cursor.execute(
                "SELECT public.lab_arena_sep16_recovery_archive_valid_v1()"
            )
            assert cursor.fetchone() == (True,)
            cursor.execute(
                "SELECT source_ref,source_size_bytes,"
                "submission_doc->>'source_ref',"
                "submission_doc->>'source_sha256',"
                "submission_doc->>'source_commit' "
                "FROM public.lab_arena_submissions WHERE submission_id=%s",
                (BASELINE,),
            )
            assert cursor.fetchone() == (
                RECOVERY_SOURCE_REF,
                RECOVERY_SOURCE_SIZE,
                RECOVERY_SOURCE_REF,
                RECOVERY_SOURCE_SHA,
                RECOVERY_SOURCE_COMMIT,
            )
            cursor.execute(
                "SELECT source_ref,source_size_bytes,"
                "submission_doc->>'source_ref',"
                "submission_doc->>'source_sha256',"
                "submission_doc->>'source_commit' "
                "FROM public.lab_arena_submissions WHERE submission_id=%s",
                (ARCHIVE_SUBMISSION,),
            )
            assert cursor.fetchone() == (
                TERMINAL_SOURCE_REF,
                NEW_SIZE,
                TERMINAL_SOURCE_REF,
                NEW_SHA,
                NEW_COMMIT,
            )
        assert _single_hash(
            connection,
            "lab_arena_sep16_baseline_rerun_audit",
            "round_id='%s'" % ROUND,
        ) == protected_prior_audit

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
                return original_verified(
                    run, icp=icp, companies=companies, policy=policy
                )
            document = json.loads(objects.get(run["output_ref"]).decode())
            validated = scoring.validate_scoring_output_document(document)
            assert validated["scored_run_id"] == run["scored_run_id"]
            return scoring.validate_breakdowns_for_item(
                validated["breakdowns"],
                icp=icp,
                companies=companies,
                max_scored_companies=int(policy["max_scored_companies"]),
                integrity_policy=True,
                contacts_required=True,
            )

        monkeypatch.setattr(service, "_verified_breakdowns", verified_fixture)
        assert service.close_stage(ROUND, 1)["status"] == "ok"
        for stage in (1, 2):
            items = _service_scoring_items(connection, stage, hotkeys)
            for item in items:
                if item["submission_id"] == BASELINE:
                    item["judgment_scope_doc"]["scorer_image_digest"] = NEW_IMAGE
                    item["judgment_scope_doc"][
                        "scorer_image_reference"
                    ] = NEW_IMAGE_REF
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT public.lab_arena_open_sep16_baseline_scoring_v1("
                    "%s,%s::smallint,%s::jsonb)",
                    (ROUND, stage, json.dumps(items)),
                )
                assert cursor.fetchone()[0]["assignments"] == 10
            connection.commit()
            _accept_recovery_scores(connection, objects, stage, icps)
            connection.commit()
            assert service.close_scoring(ROUND, stage)["status"] == "closed"
            assert service.score_stage(ROUND, stage)["status"] == "ok"
            if stage == 1:
                assert service.open_stage(ROUND, 2)["status"] == "ok"
                assert service.close_stage(ROUND, 2)["status"] == "ok"

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT tgname,tgenabled FROM pg_trigger WHERE "
                "tgrelid='public.lab_arena_rounds'::regclass AND tgname IN "
                "('lab_arena_integrity_publication_guard',"
                "'lab_arena_sep16_rerun_publication_guard') ORDER BY tgname"
            )
            assert cursor.fetchall() == [
                ("lab_arena_integrity_publication_guard", "O"),
                ("lab_arena_sep16_rerun_publication_guard", "O"),
            ]
        published = service.publish(ROUND)
        assert published["status"] == "ok"
        row = service.store.get_round(ROUND)
        baseline = next(
            item
            for item in row["publication_doc"]["final_ranking"]
            if item["submission_id"] == BASELINE
        )
        assert baseline["final_score"] > 0
        assert row["status"] == "published"
        assert row["publication_doc"]["final_ranking"]
        assert ids[1:] == row["finalists"]
    finally:
        connection.close()


def test_recovery_seal_rejects_active_or_changed_terminal_state(connect):
    connection = connect()
    try:
        old_schedule, _hotkeys, _ids = _seed_observed_sep16(connection)
        _prepare_latest(connection, old_schedule)
        _fail_both_attempts_and_cancel(connection)
        recovery_schedule = _shift_schedule(old_schedule, minutes=45)
        migration = _render_recovery(connection, recovery_schedule)
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='pending',"
                "terminal_cause=NULL WHERE run_id=(SELECT min(run_id) FROM "
                "public.lab_arena_runs WHERE round_id=%s AND submission_id=%s)",
                (ROUND, BASELINE),
            )
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
        connection.commit()
        with connection.cursor() as cursor, pytest.raises(Exception):
            cursor.execute(migration)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('public.lab_arena_sep16_baseline_recovery_audit')"
            )
            assert cursor.fetchone() == (None,)
    finally:
        connection.close()


def test_archive_allows_only_exact_late_failed_call_reconciliation(connect):
    connection = connect()
    try:
        old_schedule, _hotkeys, _ids = _seed_observed_sep16(connection)
        _prepare_latest(connection, old_schedule)
        _fail_both_attempts_and_cancel(connection)
        recovery_schedule = _shift_schedule(old_schedule, minutes=45)
        migration = _render_recovery(connection, recovery_schedule)
        _install_and_recover(connection, migration, recovery_schedule)
        sealed_recovered_state = _state_seal(connection)
        wrong_sources = (
            (RECOVERY_SOURCE_SIZE + 1, RECOVERY_SOURCE_SHA, RECOVERY_SOURCE_COMMIT),
            (RECOVERY_SOURCE_SIZE, "0" * 64, RECOVERY_SOURCE_COMMIT),
            (RECOVERY_SOURCE_SIZE, RECOVERY_SOURCE_SHA, "0" * 40),
        )
        for source_size, source_sha, source_commit in wrong_sources:
            with connection.cursor() as cursor, pytest.raises(
                Exception, match="source differs"
            ):
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep16_baseline_recovery_v1("
                    "%s,%s,%s,%s::jsonb)",
                    (
                        source_size,
                        source_sha,
                        source_commit,
                        json.dumps(recovery_schedule),
                    ),
                )
            connection.rollback()
            assert _state_seal(connection) == sealed_recovered_state
        with connection.cursor() as cursor:
            with pytest.raises(Exception, match="schedule differs"):
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep16_baseline_recovery_v1("
                    "%s,%s,%s,%s::jsonb)",
                    (
                        RECOVERY_SOURCE_SIZE,
                        RECOVERY_SOURCE_SHA,
                        RECOVERY_SOURCE_COMMIT,
                        json.dumps(
                            {**recovery_schedule, "stage_1_close": "changed"}
                        ),
                    ),
                )
            connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT entry_id,miner_hotkey,run_id,stage,call_identity,provider,"
                "operation_id,funding_source,"
                "entry_doc#>>'{call,deepline_request_id}',"
                "entry_doc#>>'{call,deepline_operation}',"
                "entry_doc#>>'{call,credential_fingerprint}' "
                "FROM public.lab_arena_ledger WHERE round_id=%s "
                "AND submission_id=%s AND entry_kind='uncertain'",
                (ARCHIVE_ROUND, ARCHIVE_SUBMISSION),
            )
            (
                entry_id,
                hotkey,
                run_id,
                stage,
                identity,
                provider,
                operation_id,
                funding_source,
                request_id,
                operation,
                credential,
            ) = cursor.fetchone()
            cursor.execute(
                "SELECT public.lab_arena_reconcile_deepline_cost_v1("
                "%s,%s,%s,%s,%s,%s,%s,7000,'0.07')",
                (
                    ARCHIVE_ROUND,
                    run_id,
                    identity,
                    entry_id,
                    request_id,
                    operation,
                    credential,
                ),
            )
            assert cursor.fetchone()[0]["status"] == "settled"
            cursor.execute(
                "SELECT public.lab_arena_sep16_recovery_archive_valid_v1()"
            )
            assert cursor.fetchone() == (True,)
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger (entry_kind,miner_hotkey,"
                "round_id,submission_id,run_id,stage,call_identity,provider,"
                "operation_id,funding_source,amount_microusd,terminal_response,"
                "entry_doc) VALUES ('settlement',%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                "0,%s::jsonb,'{}'::jsonb)",
                (
                    hotkey,
                    ARCHIVE_ROUND,
                    ARCHIVE_SUBMISSION,
                    run_id,
                    stage,
                    "sha256:" + "8" * 64,
                    provider,
                    operation_id,
                    funding_source,
                    json.dumps({"call_succeeded": False}),
                ),
            )
            cursor.execute(
                "SELECT public.lab_arena_sep16_recovery_archive_valid_v1()"
            )
            assert cursor.fetchone() == (False,)
        connection.rollback()
    finally:
        connection.close()


def test_recovery_assignment_uses_normal_claim_and_two_attempt_limit(
    connect, tmp_path
):
    connection = connect()
    harness = Harness(connect, tmp_path, challengers=[], runners=["retry-proof"])
    store = harness.service.store
    try:
        old_schedule, hotkeys, _ids = _seed_observed_sep16(connection)
        _prepare_latest(connection, old_schedule)
        _fail_both_attempts_and_cancel(connection)
        recovery_schedule = _shift_schedule(old_schedule, minutes=45)
        migration = _render_recovery(connection, recovery_schedule)
        _install_and_recover(connection, migration, recovery_schedule)

        first, token, _request_id, _request_hash = claim(
            store,
            ROUND,
            hotkeys[5],
            parallelism=1,
            ceiling=10,
            excluded=[hotkeys[5]],
        )
        assert first["status"] == "leased"
        assert first["attempt"] == 1
        assert first["assignment_id"].endswith(":rerun269")
        failed = store.complete_attempt(
            run_id=first["run_id"],
            lease_token_hash=hash_lease_token(token),
            result={"terminal_status": "provider_error"},
            terminal_cause="provider_error",
            output_ref="",
        )
        assert failed["status"] == "failed"
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',"
                "terminal_cause='model_error' WHERE round_id=%s "
                "AND submission_id=%s AND kind='execute' AND status='pending' "
                "AND assignment_id<>%s",
                (ROUND, BASELINE, first["assignment_id"]),
            )
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
        connection.commit()

        second, second_token, _request_id, _request_hash = claim(
            store,
            ROUND,
            hotkeys[5],
            parallelism=1,
            ceiling=10,
            excluded=[hotkeys[5]],
        )
        assert second["assignment_id"] == first["assignment_id"]
        assert second["attempt"] == 2
        second_failed = store.complete_attempt(
            run_id=second["run_id"],
            lease_token_hash=hash_lease_token(second_token),
            result={"terminal_status": "provider_error"},
            terminal_cause="provider_error",
            output_ref="",
        )
        assert second_failed["status"] == "failed"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),max(attempt) FROM public.lab_arena_runs "
                "WHERE assignment_id=%s",
                (first["assignment_id"],),
            )
            assert cursor.fetchone() == (2, 2)
    finally:
        connection.close()
