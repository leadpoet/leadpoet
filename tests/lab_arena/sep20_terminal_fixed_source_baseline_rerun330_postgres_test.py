"""Sealed Sep20 fixed-source rerun330 in disposable PostgreSQL."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.lab_arena import sep18_published_rerun295_postgres_test as lifecycle
from tests.lab_arena import sep19_terminal309_newjudge_rerun310_postgres_test as rerun310
from tests.lab_arena import sep20_90m_verifier_baseline_rerun328_postgres_test as rerun328
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


ROUND = "arena-2026-09-20"
BASELINE = "baseline-2026-09-20"
PRIOR_ARCHIVES = (ROUND + "-r326archive", ROUND + "-r328archive")
ARCHIVE = ROUND + "-r330archive"
SCORER_DIGEST = "sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f"
SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + SCORER_DIGEST
)
SOURCE_REF = (
    "arena/arena-2026-09-20/sources/"
    "baseline-2026-09-20-rerun330-51f1c3c8.tar.gz"
)
SOURCE_SIZE = 857_598
SOURCE_SHA256 = "f403292595dd753e8398ad46de388dd75165ebe1265526cfffd7ef7c6c3e72a4"
SOURCE_COMMIT = "51f1c3c8e17a436ec2ccfb80162beae7fc1f02b2"
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/330-arena-2026-09-20-terminal-fixed-source-baseline-rerun.sql.template"
)
RENDERED = TEMPLATE.with_suffix("")


@pytest.fixture(scope="module")
def database():
    assert CURRENT_SERVICE_MIGRATIONS[-3:] == (
        "294-lab-arena-retire-open-cost-backfill.sql",
        "301-lab-arena-score-payer-boundary.sql",
        "326-lab-arena-exhausted-provider-error-isolation.sql",
    )
    migrations = (
        CURRENT_SERVICE_MIGRATIONS[:-3]
        + (
            "264-lab-arena-codex-cost-reconciliation.sql",
            "289-lab-arena-per-icp-cost-policy.sql",
            "292-lab-arena-null-final-score-publication.sql",
        )
        + CURRENT_SERVICE_MIGRATIONS[-3:-1]
        + (
            "311-lab-arena-per-icp-closed-billing-reconciliation.sql",
            "312-lab-arena-temporary-hold-admission.sql",
            "314-lab-arena-openrouter-web-search-reservation.sql",
            "319-lab-arena-quota-sourcing-cost.sql",
            "321-lab-arena-confirmed-cost-admission.sql",
            CURRENT_SERVICE_MIGRATIONS[-1],
            "329-lab-arena-explicit-90m-lease.sql",
        )
    )
    for psycopg2, dsn in database_with_lab_arena_migration(migrations):
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            signatures = {
                "lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)": (
                    "4c5eb83c3daddfee5bfa2c33eaf33002be07bcd2f7a6fc23b5ff700d5980100a"
                ),
                "lab_arena_mark_uncertain(text,text,text,jsonb,integer)": (
                    "da6b73e012983bb59ed83be2976e435af8ea5c0c9fea1b6ceca177cd892b8d99"
                ),
                "lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)": (
                    "eb08656f73d4421be399a7bb24599a2b3d63912511372d6738e7d989dd134263"
                ),
                "lab_arena_settle_call(text,text,text,bigint,jsonb,integer)": (
                    "7f8537fa88aadd0cfe796c1ce326a6fc0b16a30f76d1f51fc800bd5e18052356"
                ),
            }
            cursor.execute(
                "SELECT p.oid::regprocedure::text,encode(extensions.digest("
                "pg_get_functiondef(p.oid),'sha256'),'hex') FROM pg_proc p "
                "WHERE p.oid=ANY(%s::regprocedure[]) ORDER BY 1",
                (list(signatures),),
            )
            assert dict(cursor.fetchall()) == signatures
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_reserve_call(text,text,text,text,text,text,"
                "bigint,jsonb,integer)'::regprocedure),pg_get_functiondef("
                "'public.lab_arena_icp_cost_eligibility(text,text,integer,integer)'"
                "::regprocedure)"
            )
            reserve_definition, eligibility_definition = cursor.fetchone()
            assert "lab_arena_confirmed_cost_admission" in reserve_definition
            assert "lab_arena_confirmed_score_admission" in reserve_definition
            assert "lab_arena_confirmed_cost_eligibility" in eligibility_definition
        yield psycopg2, dsn


def _schedule(*, start_in_minutes: int = 10) -> dict[str, str]:
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


def _assert_exact_render_shape(rendered: str) -> dict[str, str]:
    template = TEMPLATE.read_text()
    marker_pattern = re.compile(r"__[A-Z0-9_]+__")
    parts = marker_pattern.split(template)
    markers = marker_pattern.findall(template)
    pattern = [re.escape(parts[0])]
    marker_groups: dict[str, str] = {}
    for index, marker in enumerate(markers):
        group = marker_groups.get(marker)
        if group is None:
            group = f"seal_{len(marker_groups)}"
            marker_groups[marker] = group
            pattern.append(f"(?P<{group}>.*?)")
        else:
            pattern.append(f"(?P={group})")
        pattern.append(re.escape(parts[index + 1]))
    match = re.fullmatch("".join(pattern), rendered, re.DOTALL)
    assert match is not None, "rendered recovery SQL changes non-placeholder bytes"
    values = {marker: match.group(group) for marker, group in marker_groups.items()}
    assert all(value and "__" not in value for value in values.values())
    return values


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
    patched_definition = scoring_definition.replace(anchor, replacement)
    values = {
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
            patched_definition.encode()
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
        "__TERMINAL_SCORE_RUN_COUNT__": _scalar(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
            "AND kind='score'",
            (ROUND,),
        ),
        "__TERMINAL_BASELINE_RUN_COUNT__": _scalar(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
            "AND kind='execute' AND submission_id=%s",
            (ROUND, BASELINE),
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
    assert set(_assert_exact_render_shape(body)) == set(values)
    if RENDERED.exists():
        assert set(_assert_exact_render_shape(RENDERED.read_text())) == set(values)
    return body, scoring_definition, patched_definition


def _scope_snapshot(cursor, round_id: str):
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


def _publish_rerun328(connection, harness, monkeypatch) -> None:
    rerun328._publish_rerun326(connection, harness, monkeypatch)
    monkeypatch.setattr(rerun328, "NEW_SCORER_DIGEST", SCORER_DIGEST)
    monkeypatch.setattr(rerun328, "NEW_SCORER_REFERENCE", SCORER_REFERENCE)
    schedule = rerun328._schedule(start_in_minutes=10)
    with connection.cursor() as cursor:
        rendered, _, _ = rerun328._render(cursor, schedule)
        cursor.execute(rendered)
    connection.commit()
    # The shared lifecycle calls the store directly. Match the frozen 90-minute
    # round so its execute and score claims, reserves and settlements cross the
    # exact 329 RPC guards instead of the store's historical 3600-second default.
    harness.service.store._lease_ttl_seconds = 6300
    lifecycle._drive_cycle(
        harness.service, harness.objects, daily_icps(), harness.runner_keys[0]
    )
    assert harness.service.publish(ROUND)["status"] == "ok"


def _inject_exhausted_provider_zero_and_failed_unknown(connection) -> str:
    """Model the terminal state explicitly accepted by migration 326."""
    identity = "sha256:" + hashlib.sha256(b"rerun330-failed-unknown").hexdigest()
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "SELECT run_id,assignment_id,miner_hotkey,stage,icp_position,"
            "stage_generation FROM public.lab_arena_runs WHERE round_id=%s "
            "AND submission_id=%s AND kind='execute' AND icp_position=19 "
            "AND status='accepted'",
            (ROUND, BASELINE),
        )
        old_run, assignment, hotkey, stage, position, generation = cursor.fetchone()
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',"
            "terminal_cause='provider_error',result_doc="
            "'{\"terminal_status\":\"provider_error\"}'::jsonb,output_ref=NULL,"
            "per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s",
            (old_run,),
        )
        failed_run = assignment + ":2"
        cursor.execute(
            "INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,"
            "submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,"
            "runner_hotkey,terminal_cause,result_doc,stage_generation) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,2,'execute','failed',%s,"
            "'provider_error','{\"terminal_status\":\"provider_error\"}'::jsonb,%s)",
            (
                failed_run,
                assignment,
                ROUND,
                BASELINE,
                hotkey,
                stage,
                position,
                hotkey,
                generation,
            ),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET round_id=%s,submission_id=%s "
            "WHERE round_id=%s AND kind='score' AND scored_run_id=%s",
            (PRIOR_ARCHIVES[1], BASELINE + ":r328archive", ROUND, old_run),
        )
        assert cursor.rowcount == 1
        cursor.execute(
            "WITH entries(entry_kind,ordinality) AS ("
            "VALUES('reservation',1),('dispatch',2),('uncertain',3)) "
            "INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,"
            "submission_id,run_id,stage,call_identity,provider,operation_id,"
            "funding_source,amount_microusd,entry_doc,terminal_response) "
            "SELECT entry_kind,%s,%s,%s,%s,%s,%s,'openrouter',"
            "'openrouter.responses','host',CASE WHEN entry_kind='uncertain' "
            "THEN 70000 ELSE 0 END,CASE WHEN entry_kind='uncertain' THEN "
            "'{\"call\":{\"call_succeeded\":false},\"reason\":\"missing_provider_cost\"}'::jsonb "
            "ELSE '{}'::jsonb END,CASE WHEN entry_kind='uncertain' THEN "
            "'{\"status\":503,\"call_succeeded\":false}'::jsonb ELSE NULL END "
            "FROM entries ORDER BY ordinality",
            (hotkey, ROUND, BASELINE, failed_run, stage, identity),
        )
        cursor.execute("SET session_replication_role=origin")
        cursor.execute(
            "SELECT public.lab_arena__successful_call_cost_state(%s,'execute',NULL)",
            (BASELINE,),
        )
        cost = cursor.fetchone()[0]
        assert cost["inflight_calls"] == 0
        assert cost["success_unresolved_calls"] == 0
        assert cost["uncertain_calls"] >= 1
    connection.commit()
    return identity


def test_sep20_rerun330_preserves_history_and_reuses_unchanged_scorer_cache(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    schedule = _schedule()
    harness = lifecycle.Harness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _publish_rerun328(connection, harness, monkeypatch)
        failed_identity = _inject_exhausted_provider_zero_and_failed_unknown(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT entry_kind,amount_microusd FROM public.lab_arena_ledger "
                "WHERE round_id=%s AND submission_id=%s "
                "AND entry_kind IN('reservation','settlement') ORDER BY entry_id",
                (ROUND, BASELINE),
            )
            confirmed_cost_rows = cursor.fetchall()
            assert any(
                kind == "reservation" and amount == 0
                for kind, amount in confirmed_cost_rows
            )
            assert any(
                kind == "settlement" and amount > 0
                for kind, amount in confirmed_cost_rows
            )
            prior_before = {
                round_id: _scope_snapshot(cursor, round_id)
                for round_id in PRIOR_ARCHIVES
            }
            publication_before = _json(
                cursor,
                "SELECT publication_doc FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            judgments_before = _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(j) ORDER BY cache_key,authority_slot) "
                "FROM public.lab_arena_company_judgments j",
            )
            cache_before = _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(c) ORDER BY cache_key) "
                "FROM public.lab_arena_judgment_cache c",
            )
            miner_cache_keys = set(
                _json(
                    cursor,
                    "SELECT jsonb_agg(judgment_cache_key) FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND kind='score' AND submission_id<>%s",
                    (ROUND, BASELINE),
                )
            )
            terminal_baseline_runs = _scalar(
                cursor,
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='execute' AND submission_id=%s",
                (ROUND, BASELINE),
            )
            terminal_score_runs = _scalar(
                cursor,
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='score'",
                (ROUND,),
            )
            miner_before = _json(
                cursor,
                "SELECT jsonb_build_object("
                "'submissions',(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) "
                "FROM public.lab_arena_submissions s WHERE round_id=%s AND submission_id<>%s),"
                "'runs',(SELECT jsonb_agg(to_jsonb(r)-'per_icp_score'-"
                "'qualification_doc'-'updated_at' ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='execute' "
                "AND submission_id<>%s),"
                "'ledger',(SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id=%s AND submission_id<>%s "
                "AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs r "
                "WHERE r.run_id=l.run_id AND r.round_id=%s AND r.kind='score')))",
                (ROUND, BASELINE, ROUND, BASELINE, ROUND, BASELINE, ROUND),
            )
            rendered, scoring_before, scoring_after = _render(cursor, schedule)
            cursor.execute(rendered)
            cursor.execute(rendered)
            assert _scalar(
                cursor,
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='execute' AND submission_id=%s",
                (ARCHIVE, BASELINE + ":r330archive"),
            ) == terminal_baseline_runs
            assert _scalar(
                cursor,
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='score'",
                (ARCHIVE,),
            ) == terminal_score_runs
            assert all(
                _scope_snapshot(cursor, round_id) == prior_before[round_id]
                for round_id in PRIOR_ARCHIVES
            )
            assert _json(
                cursor,
                "SELECT publication_doc FROM public.lab_arena_rounds WHERE round_id=%s",
                (ARCHIVE,),
            ) == publication_before
            assert _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(j) ORDER BY cache_key,authority_slot) "
                "FROM public.lab_arena_company_judgments j",
            ) == judgments_before
            assert _json(
                cursor,
                "SELECT jsonb_agg(to_jsonb(c) ORDER BY cache_key) "
                "FROM public.lab_arena_judgment_cache c",
            ) == cache_before
            miner_after = _json(
                cursor,
                "SELECT jsonb_build_object("
                "'submissions',(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) "
                "FROM public.lab_arena_submissions s WHERE round_id=%s AND submission_id<>%s),"
                "'runs',(SELECT jsonb_agg(to_jsonb(r)-'per_icp_score'-"
                "'qualification_doc'-'updated_at' ORDER BY run_id) "
                "FROM public.lab_arena_runs r WHERE round_id=%s AND kind='execute' "
                "AND submission_id<>%s),"
                "'ledger',(SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id=%s AND submission_id<>%s))",
                (ROUND, BASELINE, ROUND, BASELINE, ROUND, BASELINE),
            )
            assert miner_after == miner_before
            cursor.execute(
                "SELECT configuration_doc,source_ref,source_size_bytes,"
                "submission_doc->>'source_sha256',submission_doc->>'source_commit' "
                "FROM public.lab_arena_rounds r JOIN public.lab_arena_submissions s "
                "ON s.round_id=r.round_id WHERE r.round_id=%s AND s.submission_id=%s",
                (ROUND, BASELINE),
            )
            config, ref, size, sha256, commit = cursor.fetchone()
            assert config["call_quotas"] == {
                "deepline": 200,
                "openrouter": 2000,
                "scrapingdog": 200,
            }
            assert config["checkpoint_deadline_policy"] == "atomic_checkpoint_90m_v1"
            assert config["icp_wall_clock_seconds"] == 5400
            assert config["lease_ttl_seconds"] == 6300
            assert config["scorer_image_digest"] == SCORER_DIGEST
            assert (ref, size, sha256, commit) == (
                SOURCE_REF,
                SOURCE_SIZE,
                SOURCE_SHA256,
                SOURCE_COMMIT,
            )
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
            )
            assert cursor.fetchone()[0] == scoring_after
            assert scoring_after.replace(
                _template_block("replacement"), _template_block("anchor")
            ) == scoring_before
            assert _scalar(
                cursor,
                "SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=%s "
                "AND call_identity=%s",
                (ARCHIVE, failed_identity),
            ) == "3"
        connection.commit()

        lifecycle._drive_cycle(
            harness.service, harness.objects, daily_icps(), harness.runner_keys[0]
        )
        assert harness.service.publish(ROUND)["status"] == "ok"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),"
                "bool_and(status='accepted'),"
                "bool_and(assignment_id LIKE '%%:score:rerun330') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            assert cursor.fetchone() == (100, 100, True, True)
            reused_miner_keys = set(
                _json(
                    cursor,
                    "SELECT jsonb_agg(judgment_cache_key) FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND kind='score' AND submission_id<>%s",
                    (ROUND, BASELINE),
                )
            )
            assert reused_miner_keys == miner_cache_keys
            assert _json(
                cursor,
                "SELECT publication_doc FROM public.lab_arena_rounds WHERE round_id=%s",
                (ARCHIVE,),
            ) == publication_before


def test_sep20_rerun330_template_is_inactive_and_narrow():
    body = TEMPLATE.read_text()
    assert TEMPLATE.name not in CURRENT_SERVICE_MIGRATIONS
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
    assert all(round_id in body for round_id in PRIOR_ARCHIVES)
    assert ARCHIVE in body and "score:rerun330" in body
    assert '"openrouter":500' in body and '"openrouter":2000' in body
    assert SCORER_DIGEST in body and SCORER_REFERENCE in body
    assert "atomic_checkpoint_90m_v1" in body
    assert "icp_wall_clock_seconds')::INTEGER<>5400" in body
    assert "lease_ttl_seconds')::INTEGER<>6300" in body
    assert "inflight_calls" in body and "success_unresolved_calls" in body
    assert "uncertain_calls" not in body
    assert "output_ref IS NOT NULL)<>100" not in body
    assert "output_ref IS NOT NULL)<>80" in body
    assert "__TERMINAL_SCORE_RUN_COUNT__" in body
    assert "__TERMINAL_BASELINE_RUN_COUNT__" in body
    assert "archived_execution_judgments" in body
    assert "CREATE TRIGGER" not in body
    assert "INTERVAL '7 hours'" in body
    assert "INTERVAL '10 hours'" in body
    assert "INTERVAL '14 hours 1 second'" in body
    assert SOURCE_REF in body and str(SOURCE_SIZE) in body
    assert SOURCE_SHA256 in body and SOURCE_COMMIT in body
    assert len(set(re.findall(r"__[A-Z0-9_]+__", body))) == 21
    if RENDERED.exists():
        values = _assert_exact_render_shape(RENDERED.read_text())
        assert len(values) == 21
        assert re.search(r"__[A-Z0-9_]+__", RENDERED.read_text()) is None
