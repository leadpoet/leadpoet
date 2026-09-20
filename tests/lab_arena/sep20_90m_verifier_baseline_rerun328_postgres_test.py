"""Sealed Sep20 90-minute verifier recovery in disposable PostgreSQL."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.lab_arena import sep18_published_rerun295_postgres_test as lifecycle
from tests.lab_arena import sep19_terminal309_newjudge_rerun310_postgres_test as rerun310
from tests.lab_arena import sep20_terminal_native_parity_baseline_rerun326_postgres_test as rerun326
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


ROUND = "arena-2026-09-20"
BASELINE = "baseline-2026-09-20"
PRIOR_ARCHIVE = ROUND + "-r326archive"
ARCHIVE = ROUND + "-r328archive"
SOURCE_REF = (
    "arena/arena-2026-09-20/sources/"
    "baseline-2026-09-20-rerun328-5d492f17.tar.gz"
)
SOURCE_SIZE = 854_708
SOURCE_SHA256 = "f1f24e24e0cd736d8c9695640093f4d5f28c360e2773b9824dd5010fe5100f3e"
SOURCE_COMMIT = "5d492f1715e27c8d9b919bcc3ad69d1be5160524"
NEW_SCORER_DIGEST = "sha256:" + "e" * 64
NEW_SCORER_REFERENCE = "registry.example/lab/scorer@" + NEW_SCORER_DIGEST
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/328-arena-2026-09-20-90m-verifier-baseline-rerun.sql.template"
)


@pytest.fixture(scope="module")
def database():
    assert CURRENT_SERVICE_MIGRATIONS[-3:] == (
        "294-lab-arena-retire-open-cost-backfill.sql",
        "301-lab-arena-score-payer-boundary.sql",
        "326-lab-arena-exhausted-provider-error-isolation.sql",
    )
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS[:-3]
        + (
            "289-lab-arena-per-icp-cost-policy.sql",
            "292-lab-arena-null-final-score-publication.sql",
        )
        + CURRENT_SERVICE_MIGRATIONS[-3:]
    )


def _schedule(*, start_in_minutes: int) -> dict[str, str]:
    start = datetime.now(timezone.utc) + timedelta(minutes=start_in_minutes)
    stamp = lambda value: value.isoformat().replace("+00:00", "Z")
    return {
        "submission_open": "2026-09-19T00:00:00Z",
        "submission_cutoff": "2026-09-20T00:00:00Z",
        "benchmark_deadline": stamp(start),
        "stage_1_start": stamp(start + timedelta(seconds=1)),
        "stage_1_close": stamp(start + timedelta(hours=7)),
        "stage_1_scoring_close": stamp(start + timedelta(hours=10)),
        "stage_2_start": stamp(start + timedelta(hours=10, seconds=1)),
        "stage_2_close": stamp(start + timedelta(hours=11)),
        "final_scoring_close": stamp(start + timedelta(hours=14)),
        "publication_deadline": stamp(start + timedelta(hours=14, seconds=1)),
    }


def _capacity_proof(schedule: dict[str, str]) -> None:
    parsed = {
        key: datetime.fromisoformat(value.replace("Z", "+00:00"))
        for key, value in schedule.items()
        if key not in {"submission_open", "submission_cutoff"}
    }
    origin = parsed["benchmark_deadline"]
    assert parsed["stage_1_start"] == origin + timedelta(seconds=1)
    assert parsed["stage_1_close"] == origin + timedelta(hours=7)
    assert parsed["stage_1_scoring_close"] == origin + timedelta(hours=10)
    assert parsed["stage_2_start"] == origin + timedelta(hours=10, seconds=1)
    assert parsed["stage_2_close"] == origin + timedelta(hours=11)
    assert parsed["final_scoring_close"] == origin + timedelta(hours=14)
    assert parsed["publication_deadline"] == origin + timedelta(
        hours=14, seconds=1
    )
    execution_capacity = 20 * 2 * (5400 + 60) / 10
    stage_one_judge_capacity = 50 * 2 * (900 + 60) / 10
    assert execution_capacity <= 7 * 3600
    assert stage_one_judge_capacity <= 3 * 3600
    assert list(parsed.values()) == sorted(parsed.values())


def _scalar(cursor, query: str, parameters=()) -> str:
    cursor.execute(query, parameters)
    return str(cursor.fetchone()[0])


def _json(cursor, query: str, parameters=()):
    cursor.execute(query, parameters)
    return cursor.fetchone()[0]


def _template_block(name: str) -> str:
    matches = re.findall(
        rf"\${re.escape(name)}\$(.*?)\${re.escape(name)}\$",
        TEMPLATE.read_text(),
        re.DOTALL,
    )
    assert len(matches) == 1
    return matches[0]


def _render(cursor, schedule: dict[str, str]) -> tuple[str, str, str]:
    cursor.execute(
        "SELECT status,cancel_reason FROM public.lab_arena_rounds WHERE round_id=%s",
        (ROUND,),
    )
    status, cancel_reason = cursor.fetchone()
    cursor.execute(
        "SELECT pg_get_functiondef("
        "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
    )
    scoring_definition = cursor.fetchone()[0]
    anchor = _template_block("anchor")
    replacement = _template_block("replacement")
    assert scoring_definition.count(anchor) == 1
    patched_scoring_definition = scoring_definition.replace(anchor, replacement)
    values = {
        "__NEW_SCORER_REFERENCE__": NEW_SCORER_REFERENCE,
        "__NEW_SCORER_DIGEST__": NEW_SCORER_DIGEST,
        "__RERUN_SCHEDULE_JSON__": json.dumps(
            schedule, sort_keys=True, separators=(",", ":")
        ),
        "__TERMINAL_STATUS__": status,
        "__TERMINAL_CANCEL_REASON_SQL__": (
            "NULL" if cancel_reason is None else "'" + cancel_reason + "'"
        ),
        "__SCORING_DEFINITION_SHA256__": hashlib.sha256(
            scoring_definition.encode()
        ).hexdigest(),
        "__PATCHED_SCORING_DEFINITION_SHA256__": hashlib.sha256(
            patched_scoring_definition.encode()
        ).hexdigest(),
        "__TERMINAL_ROUND_SHA256__": _scalar(
            cursor,
            "SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex') "
            "FROM public.lab_arena_rounds r WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_BASELINE_SHA256__": _scalar(
            cursor,
            "SELECT encode(extensions.digest(to_jsonb(s)::text,'sha256'),'hex') "
            "FROM public.lab_arena_submissions s WHERE submission_id=%s",
            (BASELINE,),
        ),
        "__TERMINAL_SUBMISSIONS_SHA256__": _scalar(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode("
            "extensions.digest(to_jsonb(s)::text,'sha256'),'hex'),'' ORDER BY "
            "submission_id),''),'sha256'),'hex') FROM public.lab_arena_submissions s "
            "WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_SUBMISSION_COUNT__": _scalar(
            cursor,
            "SELECT count(*) FROM public.lab_arena_submissions WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_RUNS_SHA256__": _scalar(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode("
            "extensions.digest(to_jsonb(r)::text,'sha256'),'hex'),'' ORDER BY "
            "run_id),''),'sha256'),'hex') FROM public.lab_arena_runs r "
            "WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_RUN_COUNT__": _scalar(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_LEDGER_SHA256__": _scalar(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode("
            "extensions.digest(to_jsonb(l)::text,'sha256'),'hex'),'' ORDER BY "
            "entry_id),''),'sha256'),'hex') FROM public.lab_arena_ledger l "
            "WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_LEDGER_COUNT__": _scalar(
            cursor,
            "SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_COMPANY_JUDGMENTS_SHA256__": _scalar(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode("
            "extensions.digest(to_jsonb(j)::text,'sha256'),'hex'),'' ORDER BY "
            "cache_key,authority_slot),''),'sha256'),'hex') "
            "FROM public.lab_arena_company_judgments j",
        ),
        "__TERMINAL_COMPANY_JUDGMENT_COUNT__": _scalar(
            cursor, "SELECT count(*) FROM public.lab_arena_company_judgments"
        ),
        "__TERMINAL_JUDGMENT_CACHE_SHA256__": _scalar(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode("
            "extensions.digest(to_jsonb(c)::text,'sha256'),'hex'),'' ORDER BY "
            "cache_key),''),'sha256'),'hex') FROM public.lab_arena_judgment_cache c",
        ),
        "__TERMINAL_JUDGMENT_CACHE_COUNT__": _scalar(
            cursor, "SELECT count(*) FROM public.lab_arena_judgment_cache"
        ),
        "__TERMINAL_SCORE_LEDGER_COUNT__": _scalar(
            cursor,
            "SELECT count(*) FROM public.lab_arena_ledger l WHERE l.round_id=%s "
            "AND EXISTS(SELECT 1 FROM public.lab_arena_runs r WHERE r.run_id=l.run_id "
            "AND r.round_id=%s AND r.kind='score')",
            (ROUND, ROUND),
        ),
        "__TERMINAL_BASELINE_LEDGER_COUNT__": _scalar(
            cursor,
            "SELECT count(*) FROM public.lab_arena_ledger l WHERE l.round_id=%s "
            "AND l.submission_id=%s AND EXISTS(SELECT 1 FROM public.lab_arena_runs r "
            "WHERE r.run_id=l.run_id AND r.round_id=%s AND r.kind='execute')",
            (ROUND, BASELINE, ROUND),
        ),
    }
    body = TEMPLATE.read_text()
    for marker, value in values.items():
        body = body.replace(marker, value)
    assert re.search(r"__[A-Z0-9_]+__", body) is None
    return body, scoring_definition, patched_scoring_definition


def _put_retained_outputs(harness, connection) -> list[tuple[str, str, int]]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT run_id,output_ref,icp_position FROM public.lab_arena_runs "
            "WHERE round_id=%s AND kind='execute' AND submission_id<>%s "
            "ORDER BY run_id",
            (ROUND, BASELINE),
        )
        retained = cursor.fetchall()
    icps = daily_icps()
    harness.objects.put(
        f"arena/{ROUND}/benchmark.json",
        json.dumps(
            {
                "schema_version": "leadpoet.lab_arena.benchmark.v1",
                "round_id": ROUND,
                "icps": icps,
            }
        ).encode(),
    )
    for index, (run_id, output_ref, position) in enumerate(retained, 1):
        rerun310._proof_execution_v2(
            harness.objects, icps[position], index, position, run_id
        )
        harness.objects.put(
            output_ref, harness.objects.get(f"arena/output/{run_id}.json")
        )
    return retained


def _publish_rerun326(connection, harness, monkeypatch) -> None:
    old_schedule = rerun326._schedule()
    rerun326._seed_terminal(
        connection, old_schedule, runner_hotkeys=harness.runner_keys
    )
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc="
            "jsonb_set(configuration_doc,'{checkpoint_deadline_policy}',"
            "'\"atomic_checkpoint_45m_v1\"'::jsonb,true) WHERE round_id=%s",
            (ROUND,),
        )
        cursor.execute("SET session_replication_role=origin")
        rendered, _, _ = rerun326._render(cursor, old_schedule)
        cursor.execute(rendered)
    connection.commit()
    _put_retained_outputs(harness, connection)
    monkeypatch.setattr(lifecycle, "ROUND", ROUND)
    monkeypatch.setattr(lifecycle, "BASELINE", BASELINE)
    monkeypatch.setattr(lifecycle, "_proof_execution", rerun310._proof_execution_v2)
    lifecycle._drive_cycle(
        harness.service, harness.objects, daily_icps(), harness.runner_keys[0]
    )
    assert harness.service.publish(ROUND)["status"] == "ok"
    # The shared lifecycle deliberately injects one retry before success. The
    # live rerun326 seal has exactly the twenty accepted native executions, so
    # retain that synthetic failed attempt and its costs in the prior archive.
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
            "AND submission_id=%s AND kind='execute' AND status='failed'",
            (ROUND, BASELINE),
        )
        failed = [row[0] for row in cursor.fetchall()]
        assert len(failed) == 1
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_ledger SET round_id=%s,submission_id=%s "
            "WHERE run_id=%s",
            (PRIOR_ARCHIVE, BASELINE + ":r326archive", failed[0]),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET round_id=%s,submission_id=%s "
            "WHERE run_id=%s",
            (PRIOR_ARCHIVE, BASELINE + ":r326archive", failed[0]),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _round_scope_snapshot(cursor, round_id: str):
    return _json(
        cursor,
        "SELECT jsonb_build_object("
        "'round',(SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s),"
        "'submissions',(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) "
        "FROM public.lab_arena_submissions s WHERE round_id=%s),"
        "'runs',(SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) "
        "FROM public.lab_arena_runs r WHERE round_id=%s),"
        "'ledger',(SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
        "FROM public.lab_arena_ledger l WHERE round_id=%s))",
        (round_id, round_id, round_id, round_id),
    )


def test_sep20_rerun328_seals_transition_and_publishes_100_fresh_scores(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    schedule = _schedule(start_in_minutes=10)
    _capacity_proof(schedule)
    harness = lifecycle.Harness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _publish_rerun326(connection, harness, monkeypatch)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE submission_id=%s),"
                "count(*) FILTER(WHERE submission_id<>%s) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (BASELINE, BASELINE, ROUND),
            )
            assert cursor.fetchone() == (100, 20, 80)
            prior_archive_before = _round_scope_snapshot(cursor, PRIOR_ARCHIVE)
            company_judgments_before = _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(j) ORDER BY cache_key,authority_slot) "
                "FROM public.lab_arena_company_judgments j",
            )
            judgment_cache_before = _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(c) ORDER BY cache_key) "
                "FROM public.lab_arena_judgment_cache c",
            )
            old_cache_keys = set(
                _json(
                    cursor,
                    "SELECT coalesce(jsonb_agg(cache_key),'[]'::jsonb) "
                    "FROM public.lab_arena_judgment_cache",
                )
            )
            miner_submissions_before = _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) "
                "FROM public.lab_arena_submissions s WHERE round_id=%s "
                "AND submission_id<>%s",
                (ROUND, BASELINE),
            )
            miner_runs_before = _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(r)-'per_icp_score'-'qualification_doc'-"
                "'updated_at' ORDER BY run_id) FROM public.lab_arena_runs r "
                "WHERE round_id=%s AND kind='execute' AND submission_id<>%s",
                (ROUND, BASELINE),
            )
            miner_ledger_before = _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id=%s "
                "AND submission_id<>%s AND NOT EXISTS(SELECT 1 "
                "FROM public.lab_arena_runs r WHERE r.run_id=l.run_id "
                "AND r.round_id=%s AND r.kind='score')",
                (ROUND, BASELINE, ROUND),
            )
            old_config = _json(
                cursor,
                "SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            terminal_derived = _json(
                cursor,
                "SELECT jsonb_build_object('publication_doc',publication_doc,"
                "'king_outcome',king_outcome,'king_hotkey',king_hotkey,"
                "'king_start_epoch',king_start_epoch,"
                "'effective_reward_epoch',effective_reward_epoch,"
                "'reward_basis_hash',reward_basis_hash,"
                "'reward_basis_doc',reward_basis_doc,'signing_key_doc',signing_key_doc,"
                "'reward_activated_at',reward_activated_at,"
                "'promotion_required',promotion_required,"
                "'promotion_doc',promotion_doc,"
                "'baseline_promoted_at',baseline_promoted_at) "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            old_counts = _json(
                cursor,
                "SELECT jsonb_build_array("
                "(SELECT count(*) FROM public.lab_arena_rounds),"
                "(SELECT count(*) FROM public.lab_arena_submissions),"
                "(SELECT count(*) FROM public.lab_arena_runs),"
                "(SELECT count(*) FROM public.lab_arena_ledger))",
            )
            rendered, scoring_before, scoring_after = _render(cursor, schedule)
            cursor.execute(
                "SELECT run_id,result_doc FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute' ORDER BY run_id LIMIT 1",
                (ROUND,),
            )
            drift_run_id, drift_result_doc = cursor.fetchone()
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb "
                "WHERE run_id=%s",
                (drift_run_id,),
            )
            cursor.execute("SET session_replication_role=origin")
        connection.commit()
        with pytest.raises(psycopg2.Error, match="terminal (preimage|history) differs"):
            with connection.cursor() as cursor:
                cursor.execute(rendered)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET result_doc=%s::jsonb "
                "WHERE run_id=%s",
                (json.dumps(drift_result_doc), drift_run_id),
            )
            cursor.execute("SET session_replication_role=origin")
        connection.commit()

        with connection.cursor() as cursor:
            rendered, scoring_before, scoring_after = _render(cursor, schedule)
            cursor.execute(rendered)
            cursor.execute(rendered)
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
            )
            definition = cursor.fetchone()[0]
            assert definition == scoring_after
            assert definition.count(":score:rerun328") == 1
            assert scoring_after.replace(
                _template_block("replacement"), _template_block("anchor")
            ) == scoring_before
            assert _round_scope_snapshot(cursor, PRIOR_ARCHIVE) == prior_archive_before
            assert _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(j) ORDER BY cache_key,authority_slot) "
                "FROM public.lab_arena_company_judgments j",
            ) == company_judgments_before
            assert _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(c) ORDER BY cache_key) "
                "FROM public.lab_arena_judgment_cache c",
            ) == judgment_cache_before
            assert _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) "
                "FROM public.lab_arena_submissions s WHERE round_id=%s "
                "AND submission_id<>%s",
                (ROUND, BASELINE),
            ) == miner_submissions_before
            assert _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(r)-'per_icp_score'-'qualification_doc'-"
                "'updated_at' ORDER BY run_id) FROM public.lab_arena_runs r "
                "WHERE round_id=%s AND kind='execute' AND submission_id<>%s",
                (ROUND, BASELINE),
            ) == miner_runs_before
            assert _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id=%s "
                "AND submission_id<>%s",
                (ROUND, BASELINE),
            ) == miner_ledger_before
            new_counts = _json(
                cursor,
                "SELECT jsonb_build_array("
                "(SELECT count(*) FROM public.lab_arena_rounds),"
                "(SELECT count(*) FROM public.lab_arena_submissions),"
                "(SELECT count(*) FROM public.lab_arena_runs),"
                "(SELECT count(*) FROM public.lab_arena_ledger))",
            )
            assert new_counts == [old_counts[0] + 1, old_counts[1] + 8, old_counts[2] + 20, old_counts[3]]
            cursor.execute(
                "SELECT configuration_doc,source_ref,source_size_bytes,"
                "submission_doc->>'source_sha256',submission_doc->>'source_commit' "
                "FROM public.lab_arena_rounds r JOIN public.lab_arena_submissions s "
                "ON s.round_id=r.round_id WHERE r.round_id=%s AND s.submission_id=%s",
                (ROUND, BASELINE),
            )
            new_config, source_ref, source_size, source_sha, source_commit = cursor.fetchone()
            mutable = {
                "schedule",
                "scorer_image_digest",
                "scorer_image_reference",
                "checkpoint_deadline_policy",
                "icp_wall_clock_seconds",
                "lease_ttl_seconds",
            }
            assert {key: value for key, value in new_config.items() if key not in mutable} == {
                key: value for key, value in old_config.items() if key not in mutable
            }
            assert (
                new_config["checkpoint_deadline_policy"],
                new_config["icp_wall_clock_seconds"],
                new_config["lease_ttl_seconds"],
                new_config["scorer_image_digest"],
                new_config["scorer_image_reference"],
                new_config["schedule"],
            ) == (
                "atomic_checkpoint_90m_v1",
                5400,
                6300,
                NEW_SCORER_DIGEST,
                NEW_SCORER_REFERENCE,
                schedule,
            )
            assert (source_ref, source_size, source_sha, source_commit) == (
                SOURCE_REF,
                SOURCE_SIZE,
                SOURCE_SHA256,
                SOURCE_COMMIT,
            )
            archived_derived = _json(
                cursor,
                "SELECT jsonb_build_object('publication_doc',publication_doc,"
                "'king_outcome',configuration_doc->'archived_king_outcome',"
                "'king_hotkey',configuration_doc->'archived_king_hotkey',"
                "'king_start_epoch',configuration_doc->'archived_king_start_epoch',"
                "'effective_reward_epoch',"
                "configuration_doc->'archived_effective_reward_epoch',"
                "'reward_basis_hash',configuration_doc->'archived_reward_basis_hash',"
                "'reward_basis_doc',configuration_doc->'archived_reward_basis_doc',"
                "'signing_key_doc',configuration_doc->'archived_signing_key_doc',"
                "'reward_activated_at',configuration_doc->'archived_reward_activated_at',"
                "'promotion_required',configuration_doc->'archived_promotion_required',"
                "'promotion_doc',configuration_doc->'archived_promotion_doc',"
                "'baseline_promoted_at',configuration_doc->'archived_baseline_promoted_at') "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ARCHIVE,),
            )
            assert archived_derived == terminal_derived
            assert _scalar(
                cursor,
                "SELECT jsonb_array_length(configuration_doc->"
                "'archived_execution_judgments') FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (ARCHIVE,),
            ) == "100"
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE round_id=%s AND kind='score'),"
                "count(*) FILTER(WHERE round_id=%s AND kind='execute' "
                "AND submission_id=%s),count(*) FILTER(WHERE round_id=%s "
                "AND kind='execute' AND submission_id<>%s),"
                "count(*) FILTER(WHERE round_id=%s AND kind='execute' "
                "AND submission_id<>%s AND per_icp_score IS NULL "
                "AND qualification_doc IS NULL) FROM public.lab_arena_runs "
                "WHERE round_id IN(%s,%s)",
                (ARCHIVE, ARCHIVE, BASELINE + ":r328archive", ROUND, BASELINE, ROUND, BASELINE, ROUND, ARCHIVE),
            )
            assert cursor.fetchone() == (220, 100, 20, 80, 80)
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )
            with pytest.raises(psycopg2.Error, match="100 fresh distinct scores"):
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET status='published',"
                    "publication_doc='{\"invalid\":true}'::jsonb WHERE round_id=%s",
                    (ROUND,),
                )
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
                "lab_arena_integrity_publication_guard"
            )
            cursor.execute(
                "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER "
                "lab_arena_rounds_write_once"
            )
        connection.commit()

        lifecycle._drive_cycle(
            harness.service, harness.objects, daily_icps(), harness.runner_keys[0]
        )
        publication = harness.service.publish(ROUND)
        assert publication["status"] == "ok"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),"
                "count(DISTINCT scored_run_id),bool_and(status='accepted'),"
                "bool_and(assignment_id LIKE '%%:score:rerun328'),"
                "bool_and(judgment_scope_doc->>'scorer_image_digest'=%s),"
                "bool_and(judgment_scope_doc->>'scorer_image_reference'=%s) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (NEW_SCORER_DIGEST, NEW_SCORER_REFERENCE, ROUND),
            )
            assert cursor.fetchone() == (100, 100, 100, True, True, True, True)
            cursor.execute(
                "SELECT coalesce(jsonb_agg(judgment_cache_key),'[]'::jsonb) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            new_cache_keys = set(cursor.fetchone()[0])
            assert len(new_cache_keys) > 0
            assert new_cache_keys.isdisjoint(old_cache_keys)
            cursor.execute(
                "SELECT status,jsonb_array_length(publication_doc->'final_ranking'),"
                "(SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='execute' AND per_icp_score IS NOT NULL "
                "AND qualification_doc IS NOT NULL) "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND, ROUND),
            )
            assert cursor.fetchone() == ("published", 5, 100)
            published_before = _round_scope_snapshot(cursor, ROUND)
            archive_before = _round_scope_snapshot(cursor, ARCHIVE)
            cursor.execute(rendered)
            assert _round_scope_snapshot(cursor, ROUND) == published_before
            assert _round_scope_snapshot(cursor, ARCHIVE) == archive_before


def test_sep20_rerun328_template_is_inactive_and_narrow():
    body = TEMPLATE.read_text()
    assert TEMPLATE.name not in CURRENT_SERVICE_MIGRATIONS
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
    assert "arena-2026-09-20-r326archive" in body
    assert "arena-2026-09-20-r328archive" in body
    assert "score:rerun328" in body
    assert "atomic_checkpoint_90m_v1" in body
    assert "icp_wall_clock_seconds',5400" in body
    assert "lease_ttl_seconds',6300" in body
    assert '"openrouter":500' in body
    assert "4000000" in body and "800000" in body
    assert SOURCE_REF in body
    assert str(SOURCE_SIZE) in body
    assert SOURCE_SHA256 in body
    assert SOURCE_COMMIT in body
    assert "archived_execution_judgments" in body
    assert "Sep20 rerun328 requires 100 fresh distinct scores" in body
    assert "lab_arena_company_judgments" in body
    assert "lab_arena_judgment_cache" in body
    assert "moved_baseline_ledger<>__TERMINAL_BASELINE_LEDGER_COUNT__" in body
    assert "moved_score_ledger<>__TERMINAL_SCORE_LEDGER_COUNT__" in body
    assert "INTERVAL '7 hours'" in body
    assert "INTERVAL '10 hours'" in body
    assert "INTERVAL '14 hours 1 second'" in body
    assert len(set(re.findall(r"__[A-Z0-9_]+__", body))) == 21
