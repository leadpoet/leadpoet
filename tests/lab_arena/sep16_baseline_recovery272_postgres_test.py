"""PostgreSQL proof for the exact Sep16 recovery272 template."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import scoring
from lab_arena.store import FUNCTION_SIGNATURES, hash_lease_token
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.sep16_baseline_recovery_postgres_test import (
    ARCHIVE_ROUND as RERUN265_ARCHIVE,
    ARCHIVE_SUBMISSION as RERUN265_ARCHIVE_SUBMISSION,
    RECOVERY_SOURCE_COMMIT as TERMINAL_SOURCE_COMMIT,
    RECOVERY_SOURCE_REF as TERMINAL_SOURCE_REF,
    RECOVERY_SOURCE_SHA as TERMINAL_SOURCE_SHA,
    RECOVERY_SOURCE_SIZE as TERMINAL_SOURCE_SIZE,
    _fail_both_attempts_and_cancel,
    _install_and_recover,
    _prepare_latest,
    _render_recovery,
    _shift_schedule,
    _single_hash,
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


ARCHIVE_ROUND = "arena-2026-09-16-rerun269archive"
ARCHIVE_SUBMISSION = "baseline-2026-09-16-native-rerun269-archive"
RECOVERY_SOURCE_REF = (
    "arena/arena-2026-09-16/sources/"
    "baseline-2026-09-16-recovery272.tar.gz"
)
# Template-only fixture values. Production SQL must be rendered from the
# separately verified public champion archive before it can be applied.
RECOVERY_SOURCE_SIZE = 533090
RECOVERY_SOURCE_SHA = (
    "72a008e2cacde52921c4be94951839edccc321d90de07741010a3100463eef32"
)
RECOVERY_SOURCE_COMMIT = "0a19bd28177628d01b5ce1f4fc54cc459d7d4f51"
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/272-arena-2026-09-16-baseline-recovery.sql.template"
)


@pytest.fixture
def connect():
    generator = database_with_lab_arena_migration(MIGRATIONS)
    psycopg2, dsn = next(generator)
    try:
        yield lambda: psycopg2.connect(**dsn)
    finally:
        generator.close()


def _append_catalog_settlements(connection):
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
                    "sha256:" + str(index + 1) * 64,
                    json.dumps(
                        {
                            "late_reconciliation": True,
                            "sep16_generic_http_catalog_reconciliation": True,
                            "reconciled_uncertainty_entry_id": 900000 + index,
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
    _append_catalog_settlements(connection)
    return schedule269, hotkeys, ids


def _render_recovery272(connection, new_schedule):
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
            connection, "lab_arena_sep16_baseline_recovery_authority", f"round_id='{ROUND}'"
        ),
        "__SEALED_PRIOR_RECOVERY_AUDIT_HASH__": _single_hash(
            connection, "lab_arena_sep16_baseline_recovery_audit", f"round_id='{ROUND}'"
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


def _install_and_recover272(connection, sql, schedule):
    with connection.cursor() as cursor:
        cursor.execute(sql)
        cursor.execute(sql)
        arguments = (
            RECOVERY_SOURCE_SIZE,
            RECOVERY_SOURCE_SHA,
            RECOVERY_SOURCE_COMMIT,
            json.dumps(schedule),
        )
        cursor.execute(
            "SELECT public.lab_arena_prepare_sep16_baseline_recovery272_v1("
            "%s,%s,%s,%s::jsonb)",
            arguments,
        )
        prepared = cursor.fetchone()[0]
        cursor.execute(
            "SELECT public.lab_arena_prepare_sep16_baseline_recovery272_v1("
            "%s,%s,%s,%s::jsonb)",
            arguments,
        )
        existing = cursor.fetchone()[0]
    connection.commit()
    assert prepared["status"] == "prepared"
    assert existing["status"] == "existing"
    return prepared


def _accept_recovery272_scores(connection, objects, stage, icps):
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
        cursor.execute(
            "SELECT run_id,scored_run_id,icp_position FROM public.lab_arena_runs "
            "WHERE round_id=%s AND submission_id=%s AND kind='score' AND stage=%s "
            "AND assignment_id LIKE '%%:score:rerun272' ORDER BY icp_position",
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


def test_recovery272_preserves_history_and_publishes_positive(connect, tmp_path, monkeypatch):
    connection = connect()
    harness = Harness(connect, tmp_path, challengers=[], runners=["recovery272-proof"])
    objects = harness.objects
    service = harness.service
    icps = daily_icps()
    try:
        schedule269, hotkeys, ids = _prepare_terminal_recovery269(connection, objects)
        protected = {
            "rerun265_archive": _single_hash(
                connection, "lab_arena_sep16_baseline_recovery_audit", f"round_id='{ROUND}'"
            ),
            "recovery_authority": _single_hash(
                connection, "lab_arena_sep16_baseline_recovery_authority", f"round_id='{ROUND}'"
            ),
        }
        schedule272 = _shift_schedule(schedule269, minutes=90)
        migration272 = _render_recovery272(connection, schedule272)
        prepared = _install_and_recover272(connection, migration272, schedule272)
        assert prepared["archived_runs"] == 40
        # The deployed 4738 daemon does not need the one-off admin RPC. Once
        # preparation is complete, all execution, scoring, aggregation, and
        # publication use the existing generic runtime calls and DB guards.
        monkeypatch.delitem(
            FUNCTION_SIGNATURES,
            "lab_arena_prepare_sep16_baseline_recovery272_v1",
        )
        prepared_state = _state_seal(connection)
        for arguments, message in (
            (
                (RECOVERY_SOURCE_SIZE + 1, RECOVERY_SOURCE_SHA,
                 RECOVERY_SOURCE_COMMIT, json.dumps(schedule272)),
                "source differs",
            ),
            (
                (RECOVERY_SOURCE_SIZE, RECOVERY_SOURCE_SHA,
                 RECOVERY_SOURCE_COMMIT,
                 json.dumps({**schedule272, "stage_1_close": "changed"})),
                "schedule differs",
            ),
        ):
            with connection.cursor() as cursor, pytest.raises(Exception, match=message):
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep16_baseline_recovery272_v1("
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
                name: definition.count(":rerun272")
                for name, definition in definitions.items()
            } == expected_namespace_counts
            assert all(":rerun269" not in definition for definition in definitions.values())
            publication = next(
                definition for definition in definitions.values()
                if "lab_arena_sep16_recovery272_archive_valid_v1" in definition
            )
            archive_clause = (
                "OR NOT public.lab_arena_sep16_recovery_archive_valid_v1()\n"
                "       OR NOT public.lab_arena_sep16_recovery272_archive_valid_v1()"
            )
            assert publication.count(archive_clause) == 1
            assert publication.count(
                "lab_arena_sep16_recovery_archive_valid_v1()"
            ) == 1
            assert publication.count(
                "lab_arena_sep16_recovery272_archive_valid_v1()"
            ) == 1
            assert "success_unresolved_calls" in publication
            assert "inflight_calls" in publication
            cursor.execute("SELECT public.lab_arena_sep16_recovery_archive_valid_v1()")
            assert cursor.fetchone() == (True,)
            cursor.execute("SELECT public.lab_arena_sep16_recovery272_archive_valid_v1()")
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
                (ARCHIVE_ROUND, ARCHIVE_SUBMISSION),
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
                RECOVERY_SOURCE_SIZE,
                RECOVERY_SOURCE_SHA,
                RECOVERY_SOURCE_COMMIT,
            )
        assert _single_hash(
            connection, "lab_arena_sep16_baseline_recovery_audit", f"round_id='{ROUND}'"
        ) == protected["rerun265_archive"]
        assert _single_hash(
            connection, "lab_arena_sep16_baseline_recovery_authority", f"round_id='{ROUND}'"
        ) == protected["recovery_authority"]

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
            _accept_recovery272_scores(connection, objects, stage, icps)
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


@pytest.mark.parametrize("drift", ("namespace", "archive-clause"))
def test_recovery272_refuses_partial_function_rewrite(connect, drift):
    connection = connect()
    try:
        schedule269, _hotkeys, _ids = _prepare_terminal_recovery269(connection)
        schedule272 = _shift_schedule(schedule269, minutes=90)
        migration272 = _render_recovery272(connection, schedule272)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_sep16_rerun_publication_guard_v1()'::regprocedure)"
            )
            definition = cursor.fetchone()[0]
            if drift == "namespace":
                changed = definition.replace(":rerun269", ":rerun272", 1)
            else:
                old = (
                    "OR NOT public.lab_arena_sep16_recovery_archive_valid_v1()"
                )
                changed = definition.replace(
                    old,
                    old + "\n       OR pg_catalog.quote_literal("
                    "'lab_arena_sep16_recovery272_archive_valid_v1()') = ''",
                    1,
                )
            assert changed != definition
            cursor.execute(changed)
        connection.commit()
        with connection.cursor() as cursor, pytest.raises(
            Exception, match="namespace function shape differs|recovery272 guard shape differs"
        ):
            cursor.execute(migration272)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('public.lab_arena_sep16_baseline_recovery272_audit')"
            )
            assert cursor.fetchone() == (None,)
    finally:
        connection.close()


@pytest.mark.parametrize("drift", ("active", "success-unresolved", "prior-archive"))
def test_recovery272_seal_fails_closed_on_terminal_drift(connect, drift):
    connection = connect()
    try:
        schedule269, _hotkeys, _ids = _prepare_terminal_recovery269(connection)
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
        schedule272 = _shift_schedule(schedule269, minutes=90)
        migration272 = _render_recovery272(connection, schedule272)
        if drift == "prior-archive":
            with connection.cursor() as cursor:
                cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
                cursor.execute(
                    "UPDATE public.lab_arena_runs SET terminal_cause='model_error' "
                    "WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs "
                    "WHERE round_id=%s)",
                    (RERUN265_ARCHIVE,),
                )
                cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
            connection.commit()
        with connection.cursor() as cursor, pytest.raises(Exception):
            cursor.execute(migration272)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('public.lab_arena_sep16_baseline_recovery272_audit')"
            )
            assert cursor.fetchone() == (None,)
    finally:
        connection.close()


def test_recovery272_uses_normal_two_attempt_limit(connect, tmp_path):
    connection = connect()
    harness = Harness(connect, tmp_path, challengers=[], runners=["retry272-proof"])
    store = harness.service.store
    try:
        schedule269, hotkeys, _ids = _prepare_terminal_recovery269(connection)
        schedule272 = _shift_schedule(schedule269, minutes=90)
        _install_and_recover272(
            connection, _render_recovery272(connection, schedule272), schedule272
        )
        first, token, _request_id, _request_hash = claim(
            store, ROUND, hotkeys[5], parallelism=1, ceiling=10,
            excluded=[hotkeys[5]],
        )
        assert first["attempt"] == 1
        assert first["assignment_id"].endswith(":rerun272")
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
