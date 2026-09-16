"""PostgreSQL proof for the exact Sep16 latest-champion reseal."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lab_arena import scoring
from scripts import arena_sep16_native_rerun as operator
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.sep16_native_baseline_rerun_postgres_test import (
    BASELINE,
    BANK_HASH,
    MIGRATIONS,
    ROUND,
    _seed_observed_sep16,
    _stage1_scoring_items,
    _state_seal,
)


OLD_REF = (
    "arena/arena-2026-09-16/sources/"
    "baseline-2026-09-16-native-rerun265.tar.gz"
)
NEW_REF = (
    "arena/arena-2026-09-16/sources/"
    "baseline-2026-09-16-native-rerun268.tar.gz"
)
OLD_SIZE = 523337
OLD_SHA = "9170a9c6551b927ac311a415da632babd91e9e6ba049cdcfd72c3f753a496576"
OLD_COMMIT = "b718c5d2f4d02b804a148dae37d511a84ba38ffd"
NEW_SIZE = 524266
NEW_SHA = "c8d188c4766008ac949c5ff794536fc905589f46657d4b1ed27f3a22385f11b0"
NEW_COMMIT = "3afe71508636254eb31a172d94f5a63f4f84a5c2"
OLD_IMAGE = "sha256:ee84f274ba24b07fa204c03535b21aac3c030c72aff0fcf7918d88d3c2c8ddee"
NEW_IMAGE = "sha256:31ac47b38b4291396765b897df48e758741b8eb3bb9697be4d789e1d71597385"
NEW_IMAGE_REF = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
    "sourcing-model@" + NEW_IMAGE
)
SCORING_TREE = "f8452314cc8b1529fbfa8b7fc9345143c1b7d731"
RUNTIME = "9d97e2ae295f715209b8fdfef2b6ddebdc46622e"
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/268-arena-2026-09-16-latest-champion-reseal.sql.template"
)


@pytest.fixture
def connect():
    generator = database_with_lab_arena_migration(MIGRATIONS)
    psycopg2, dsn = next(generator)
    try:
        yield lambda: psycopg2.connect(**dsn)
    finally:
        generator.close()


def _shift_schedule(schedule, minutes=15):
    result = dict(schedule)
    for key in (
        "benchmark_deadline",
        "stage_1_start",
        "stage_1_close",
        "stage_1_scoring_close",
        "stage_2_start",
        "stage_2_close",
        "final_scoring_close",
        "publication_deadline",
    ):
        value = datetime.fromisoformat(result[key].replace("Z", "+00:00"))
        result[key] = (value + timedelta(minutes=minutes)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return result


def _table_hashes(connection):
    values = {}
    for table, key in (
        ("lab_arena_rounds", "round_id"),
        ("lab_arena_submissions", "submission_id"),
        ("lab_arena_runs", "run_id"),
        ("lab_arena_ledger", "entry_id"),
    ):
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 'sha256:' || encode(extensions.digest(COALESCE("
                "string_agg(encode(extensions.digest(to_jsonb(row_value)::text,"
                "'sha256'),'hex'),'' ORDER BY %s),''),'sha256'),'hex') "
                "FROM public.%s AS row_value WHERE round_id=%%s" % (key, table),
                (ROUND,),
            )
            values[table] = cursor.fetchone()[0]
    return values


def _insert_owner266_authority(connection, schedule, seal):
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_sep16_rerun_release_authority ("
            "round_id,source_ref,source_size_bytes,source_sha256,source_commit,"
            "champion_model_main_commit,champion_model_lab_commit,bank_sha256,"
            "old_round_hash,old_baseline_submission_hash,old_baseline_runs_hash,"
            "old_baseline_ledger_hash,old_challenger_runs_hash,"
            "old_challenger_submissions_hash,old_challenger_ledger_hash,"
            "old_challenger_ledger_max_entry_id,old_baseline_actual_microusd,"
            "old_scorer_image_digest,new_scorer_image_reference,"
            "new_scorer_image_digest,scoring_tree_hash,native_runtime_commit,"
            "verified_parallel_runner_slots,forward_schedule) VALUES ("
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
            "%s,%s,%s,%s,10,%s::jsonb)",
            (
                ROUND, OLD_REF, OLD_SIZE, OLD_SHA, OLD_COMMIT, OLD_COMMIT,
                OLD_COMMIT, BANK_HASH, seal["round"],
                seal["baseline_submission"], seal["baseline_runs"],
                seal["baseline_ledger"], seal["challenger_runs"],
                seal["challenger_submissions"], seal["challenger_ledger"],
                seal["challenger_ledger_max"], seal["baseline_actual"],
                OLD_IMAGE, NEW_IMAGE_REF, NEW_IMAGE, SCORING_TREE, RUNTIME,
                json.dumps(schedule),
            ),
        )
    connection.commit()


def _render_reseal(connection, old_schedule, new_schedule):
    seal = _state_seal(connection)
    tables = _table_hashes(connection)
    values = {
        "__SEALED_SOURCE_SIZE_BYTES__": str(NEW_SIZE),
        "__SEALED_SOURCE_SHA256__": NEW_SHA,
        "__SEALED_CHAMPION_MODEL_COMMIT__": NEW_COMMIT,
        "__SEALED_OLD_ROUND_HASH__": seal["round"],
        "__SEALED_OLD_BASELINE_SUBMISSION_HASH__": seal["baseline_submission"],
        "__SEALED_OLD_BASELINE_RUNS_HASH__": seal["baseline_runs"],
        "__SEALED_OLD_BASELINE_LEDGER_HASH__": seal["baseline_ledger"],
        "__SEALED_OLD_CHALLENGER_RUNS_HASH__": seal["challenger_runs"],
        "__SEALED_OLD_CHALLENGER_SUBMISSIONS_HASH__": seal[
            "challenger_submissions"
        ],
        "__SEALED_OLD_CHALLENGER_LEDGER_HASH__": seal["challenger_ledger"],
        "__SEALED_OLD_CHALLENGER_LEDGER_MAX_ENTRY_ID__": str(
            seal["challenger_ledger_max"]
        ),
        "__SEALED_OLD_BASELINE_ACTUAL_MICROUSD__": str(seal["baseline_actual"]),
        "__SEALED_ROUNDS_TABLE_HASH__": tables["lab_arena_rounds"],
        "__SEALED_SUBMISSIONS_TABLE_HASH__": tables["lab_arena_submissions"],
        "__SEALED_RUNS_TABLE_HASH__": tables["lab_arena_runs"],
        "__SEALED_LEDGER_TABLE_HASH__": tables["lab_arena_ledger"],
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


def _catalog_state(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT p.proowner,p.proacl,pg_get_functiondef(p.oid) FROM pg_proc p "
            "WHERE p.oid='public.lab_arena_prepare_sep16_baseline_rerun_v1("
            "bigint,text,text,text,jsonb)'::regprocedure"
        )
        function_state = cursor.fetchone()
        cursor.execute(
            "SELECT c.relowner,c.relacl,c.relrowsecurity FROM pg_class c WHERE "
            "c.oid='public.lab_arena_sep16_rerun_release_authority'::regclass"
        )
        table_state = cursor.fetchone()
        cursor.execute(
            "SELECT tgname,tgenabled,pg_get_triggerdef(oid) FROM pg_trigger WHERE "
            "tgrelid='public.lab_arena_sep16_rerun_release_authority'::regclass "
            "AND NOT tgisinternal ORDER BY tgname"
        )
        triggers = cursor.fetchall()
    return function_state, table_state, triggers


def test_reseal_is_exact_idempotent_and_preserves_protected_rows(connect):
    connection = connect()
    try:
        old_schedule, _hotkeys, _ids = _seed_observed_sep16(connection)
        new_schedule = _shift_schedule(old_schedule)
        seal = _state_seal(connection)
        _insert_owner266_authority(connection, old_schedule, seal)
        sql = _render_reseal(connection, old_schedule, new_schedule)
        rows_before = _table_hashes(connection)
        catalog_before = _catalog_state(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT authorized_at FROM "
                "public.lab_arena_sep16_rerun_release_authority"
            )
            old_authorized_at = cursor.fetchone()[0]
            cursor.execute(sql)
            cursor.execute(
                "SELECT source_ref,source_size_bytes,source_sha256,source_commit,"
                "champion_model_main_commit,champion_model_lab_commit,"
                "forward_schedule,authorized_at FROM "
                "public.lab_arena_sep16_rerun_release_authority"
            )
            first = cursor.fetchone()
            cursor.execute(sql)
            cursor.execute(
                "SELECT source_ref,source_size_bytes,source_sha256,source_commit,"
                "champion_model_main_commit,champion_model_lab_commit,"
                "forward_schedule,authorized_at FROM "
                "public.lab_arena_sep16_rerun_release_authority"
            )
            second = cursor.fetchone()
        assert first == second
        assert first[:6] == (
            NEW_REF, NEW_SIZE, NEW_SHA, NEW_COMMIT, NEW_COMMIT, NEW_COMMIT
        )
        assert first[6] == new_schedule
        assert first[7] > old_authorized_at
        assert _table_hashes(connection) == rows_before
        catalog_after = _catalog_state(connection)
        assert catalog_after[0][:2] == catalog_before[0][:2]
        assert catalog_after[1:] == catalog_before[1:]
        assert catalog_after[0][2].replace(NEW_REF, OLD_REF) == catalog_before[0][2]
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_sep16_baseline_rerun_audit"
            )
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()


@pytest.mark.parametrize("tamper", ["protected_row", "authority", "prepared"])
def test_reseal_refuses_changed_or_prepared_state(connect, tamper):
    connection = connect()
    try:
        old_schedule, _hotkeys, _ids = _seed_observed_sep16(connection)
        new_schedule = _shift_schedule(old_schedule)
        seal = _state_seal(connection)
        _insert_owner266_authority(connection, old_schedule, seal)
        sql = _render_reseal(connection, old_schedule, new_schedule)
        if tamper == "protected_row":
            with connection.cursor() as cursor:
                cursor.execute(
                    "ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER"
                )
                cursor.execute(
                    "UPDATE public.lab_arena_submissions SET source_size_bytes="
                    "source_size_bytes+1 WHERE round_id=%s AND submission_id<>%s",
                    (ROUND, BASELINE),
                )
                cursor.execute(
                    "ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER"
                )
            connection.commit()
        elif tamper == "authority":
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE public.lab_arena_sep16_rerun_release_authority SET "
                    "source_sha256=%s WHERE round_id=%s", (NEW_SHA, ROUND)
                )
            connection.commit()
        else:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep16_baseline_rerun_v1("
                    "%s,%s,%s,%s,%s::jsonb)",
                    (OLD_SIZE, OLD_SHA, OLD_COMMIT, BANK_HASH, json.dumps(old_schedule)),
                )
                assert cursor.fetchone()[0]["status"] == "prepared"
            connection.commit()
        with connection.cursor() as cursor:
            with pytest.raises(Exception):
                cursor.execute(sql)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT source_ref,source_commit FROM "
                "public.lab_arena_sep16_rerun_release_authority"
            )
            assert cursor.fetchone() == (OLD_REF, OLD_COMMIT)
    finally:
        connection.close()


def _stage2_scoring_items(connection, hotkeys):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT run_id,submission_id,stage,icp_position,attempt,kind,status,"
            "output_ref,terminal_cause FROM public.lab_arena_runs "
            "WHERE round_id=%s AND stage=2 AND kind='execute'", (ROUND,)
        )
        names = (
            "run_id", "submission_id", "stage", "icp_position", "attempt",
            "kind", "status", "output_ref", "terminal_cause",
        )
        plan = scoring.build_scoring_plan(
            round_id=ROUND,
            stage=2,
            runs=[dict(zip(names, record)) for record in cursor.fetchall()],
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='stage2_closed',"
            "status_generation=16,stage_generation=12,"
            "stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s",
            (json.dumps(plan), ROUND),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
    connection.commit()
    items = []
    for index, planned in enumerate(plan["work_items"]):
        item = dict(planned)
        if item["submission_id"] == BASELINE:
            cache_key = "sha256:" + ("%064x" % (2100 + index))
            input_hash = "sha256:" + ("%064x" % (2200 + index))
            item.update(
                judgment_cache_key=cache_key,
                judgment_input_hash=input_hash,
                judgment_scope_doc={
                    "cache_key": cache_key,
                    "scoring_input_hash": input_hash,
                    "round_id": ROUND,
                    "network_name": "finney",
                    "netuid": 71,
                    "integrity_policy": "arena_integrity_v1",
                    "evaluation_date": "2026-09-16",
                    "scorer_image_digest": NEW_IMAGE,
                    "scorer_image_reference": NEW_IMAGE_REF,
                },
                judgment_group_leader=True,
                judgment_group_miner_hotkeys=[hotkeys[0]],
            )
        items.append(item)
    return items


def test_resealed_source_prepares_scores_both_stages_and_publishes(connect):
    connection = connect()
    try:
        old_schedule, hotkeys, ids = _seed_observed_sep16(connection)
        new_schedule = _shift_schedule(old_schedule)
        seal = _state_seal(connection)
        _insert_owner266_authority(connection, old_schedule, seal)
        sql = _render_reseal(connection, old_schedule, new_schedule)
        with connection.cursor() as cursor:
            cursor.execute(sql)
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep16_baseline_rerun_v1("
                "%s,%s,%s,%s,%s::jsonb)",
                (NEW_SIZE, NEW_SHA, NEW_COMMIT, BANK_HASH, json.dumps(new_schedule)),
            )
            assert cursor.fetchone()[0]["status"] == "prepared"
            cursor.execute(
                "SELECT source_ref,source_size_bytes,submission_doc->>'source_sha256',"
                "submission_doc->>'source_commit' FROM public.lab_arena_submissions "
                "WHERE submission_id=%s", (BASELINE,)
            )
            assert cursor.fetchone() == (
                operator.SOURCE_REF, NEW_SIZE, NEW_SHA, NEW_COMMIT
            )
        connection.commit()

        items = _stage1_scoring_items(connection, hotkeys, ids[1:])
        for item in items:
            if item["submission_id"] == BASELINE:
                item["judgment_scope_doc"]["scorer_image_digest"] = NEW_IMAGE
                item["judgment_scope_doc"]["scorer_image_reference"] = NEW_IMAGE_REF
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_open_sep16_baseline_scoring_v1("
                "%s,1::smallint,%s::jsonb)", (ROUND, json.dumps(items))
            )
            assert cursor.fetchone()[0]["assignments"] == 10
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "terminal_cause='accepted',per_icp_score=1.0,"
                "output_ref='arena/score/' || run_id || '.json' WHERE round_id=%s "
                "AND submission_id=%s AND kind='score' AND stage=1",
                (ROUND, BASELINE),
            )
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
        connection.commit()

        items = _stage2_scoring_items(connection, hotkeys)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_open_sep16_baseline_scoring_v1("
                "%s,2::smallint,%s::jsonb)", (ROUND, json.dumps(items))
            )
            assert cursor.fetchone()[0]["assignments"] == 10
            cursor.execute("ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER")
            cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "terminal_cause='accepted',per_icp_score=1.0,"
                "output_ref='arena/score/' || run_id || '.json' WHERE round_id=%s "
                "AND submission_id=%s AND kind='score' AND stage=2",
                (ROUND, BASELINE),
            )
            cursor.execute(
                "SELECT old_round_doc->'publication_doc' FROM "
                "public.lab_arena_sep16_baseline_rerun_audit WHERE round_id=%s",
                (ROUND,),
            )
            publication = cursor.fetchone()[0]
            baseline = next(
                row for row in publication["final_ranking"]
                if row["submission_id"] == BASELINE
            )
            baseline["final_score"] = 1.0
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='scored',"
                "status_generation=18,stage_generation=12,"
                "publication_doc=%s::jsonb WHERE round_id=%s",
                (json.dumps(publication), ROUND),
            )
            cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
            cursor.execute("ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER")
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT tgenabled FROM pg_trigger WHERE "
                "tgrelid='public.lab_arena_rounds'::regclass AND "
                "tgname='lab_arena_sep16_rerun_publication_guard'"
            )
            assert cursor.fetchone() == ("O",)
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='published',"
                "status_generation=status_generation+1,published_at=now(),"
                "publication_doc=%s::jsonb WHERE round_id=%s RETURNING status",
                (json.dumps(publication), ROUND),
            )
            assert cursor.fetchone() == ("published",)
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='score' AND status='accepted'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20,)
        connection.commit()
    finally:
        connection.close()
