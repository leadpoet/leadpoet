"""Sealed Sep20 same-source rejudge-only rerun330 in disposable PostgreSQL."""

from __future__ import annotations

import copy
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
TERMINAL_SCORER_DIGEST = "sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f"
TERMINAL_SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + TERMINAL_SCORER_DIGEST
)
# Fixture-only identity proves the template binds a supplied digest and reference.
NEW_SCORER_DIGEST = "sha256:" + "4a" * 32
NEW_SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@"
    + NEW_SCORER_DIGEST
)
SOURCE_REF = (
    "arena/arena-2026-09-20/sources/"
    "baseline-2026-09-20-rerun328-5d492f17.tar.gz"
)
SOURCE_SIZE = 854_708
SOURCE_SHA256 = "f1f24e24e0cd736d8c9695640093f4d5f28c360e2773b9824dd5010fe5100f3e"
SOURCE_COMMIT = "5d492f1715e27c8d9b919bcc3ad69d1be5160524"
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/330-arena-2026-09-20-terminal-fixed-source-baseline-rerun.sql.template"
)
RENDERED = TEMPLATE.with_suffix("")
AUTHORITY_HOLD = (
    Path(__file__).parents[2]
    / "scripts/331-arena-2026-09-20-promotion-reward-hold.sql"
)


class IsolatedHarness(lifecycle.Harness):
    def objects_key(self) -> str:
        return "rerun330-" + self.tmp.name


@pytest.fixture()
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


class ArtifactProofError(RuntimeError):
    pass


def _execution_artifact_proof(cursor, objects) -> str:
    cursor.execute(
        "SELECT run_id,output_ref FROM public.lab_arena_runs WHERE round_id=%s "
        "AND kind='execute' AND status='accepted' AND terminal_cause='accepted' "
        "ORDER BY run_id",
        (ROUND,),
    )
    rows = cursor.fetchall()
    if len(rows) != 98:
        raise ArtifactProofError("expected 98 accepted execution artifacts")
    items = []
    for run_id, output_ref in rows:
        try:
            body = objects.get(str(output_ref))
            parsed = json.loads(body.decode("utf-8"))
        except (OSError, KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise ArtifactProofError("execution artifact is not hash-readable") from exc
        if not isinstance(parsed, dict):
            raise ArtifactProofError("execution artifact is not an object")
        items.append(
            {
                "run_id": str(run_id),
                "output_ref": str(output_ref),
                "object_sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    encoded = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _render(cursor, schedule: dict[str, str], objects) -> tuple[str, str, str]:
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
        "__NEW_SCORER_IMAGE_DIGEST__": NEW_SCORER_DIGEST,
        "__NEW_SCORER_IMAGE_REFERENCE__": NEW_SCORER_REFERENCE,
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
        "__TERMINAL_EXECUTION_ARTIFACTS_SHA256__": _execution_artifact_proof(
            cursor, objects
        ),
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
        "__TERMINAL_BASELINE_ACCEPTED_RUN_COUNT__": _scalar(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
            "AND kind='execute' AND submission_id=%s AND status='accepted' "
            "AND terminal_cause='accepted'",
            (ROUND, BASELINE),
        ),
        "__TERMINAL_BASELINE_FAILED_RUN_COUNT__": _scalar(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
            "AND kind='execute' AND submission_id=%s AND status='failed'",
            (ROUND, BASELINE),
        ),
        "__TERMINAL_ACCEPTED_EXECUTION_COUNT__": _scalar(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
            "AND kind='execute' AND status='accepted' "
            "AND terminal_cause='accepted'",
            (ROUND,),
        ),
        "__TERMINAL_EXECUTION_ARTIFACT_COUNT__": _scalar(
            cursor,
            "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
            "AND kind='execute' AND status='accepted' "
            "AND terminal_cause='accepted' AND output_ref IS NOT NULL",
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


def _protected_database_snapshot(cursor):
    return _json(
        cursor,
        "SELECT jsonb_build_object("
        "'rounds',(SELECT jsonb_agg(to_jsonb(r) ORDER BY round_id) "
        "FROM public.lab_arena_rounds r),"
        "'submissions',(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) "
        "FROM public.lab_arena_submissions s),"
        "'runs',(SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) "
        "FROM public.lab_arena_runs r),"
        "'ledger',(SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
        "FROM public.lab_arena_ledger l),"
        "'judgments',(SELECT jsonb_agg(to_jsonb(j) ORDER BY cache_key,authority_slot) "
        "FROM public.lab_arena_company_judgments j),"
        "'cache',(SELECT jsonb_agg(to_jsonb(c) ORDER BY cache_key) "
        "FROM public.lab_arena_judgment_cache c),"
        "'reservations',(SELECT jsonb_agg(to_jsonb(x) ORDER BY run_id) "
        "FROM public.lab_arena_company_judgment_reservations x),"
        "'scoring_definition',pg_get_functiondef("
        "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure))",
    )


def _migration_body(rendered: str) -> str:
    prefix, remainder = rendered.split("BEGIN;\n", 1)
    body, suffix = remainder.rsplit("COMMIT;\n", 1)
    assert prefix.strip().startswith("-- Render only") and not suffix.strip()
    return body


def _execution_snapshot(cursor):
    return _json(
        cursor,
        "SELECT jsonb_build_object("
        "'submissions',(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) "
        "FROM public.lab_arena_submissions s WHERE round_id=%s),"
        "'runs',(SELECT jsonb_agg(to_jsonb(r)-'per_icp_score'-"
        "'qualification_doc'-'updated_at' ORDER BY run_id) FROM public.lab_arena_runs r "
        "WHERE round_id=%s AND kind='execute'),"
        "'ledger',(SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
        "FROM public.lab_arena_ledger l WHERE round_id=%s AND EXISTS("
        "SELECT 1 FROM public.lab_arena_runs r WHERE r.round_id=%s "
        "AND r.kind='execute' AND r.run_id=l.run_id)))",
        (ROUND, ROUND, ROUND, ROUND),
    )


def _publish_rerun328(
    connection, harness, monkeypatch, *, zero_baseline: bool = False
) -> None:
    rerun328._publish_rerun326(connection, harness, monkeypatch)
    monkeypatch.setattr(rerun328, "NEW_SCORER_DIGEST", TERMINAL_SCORER_DIGEST)
    monkeypatch.setattr(rerun328, "NEW_SCORER_REFERENCE", TERMINAL_SCORER_REFERENCE)
    schedule = rerun328._schedule(start_in_minutes=10)
    with connection.cursor() as cursor:
        rendered, _, _ = rerun328._render(cursor, schedule)
        cursor.execute(rendered)
    connection.commit()
    # The shared lifecycle calls the store directly. Match the frozen 90-minute
    # round so its execute and score claims, reserves and settlements cross the
    # exact 329 RPC guards instead of the store's historical 3600-second default.
    harness.service.store._lease_ttl_seconds = 6300
    if zero_baseline:
        normal_breakdown = lifecycle._proof_breakdown

        def zero_baseline_breakdown(company, score):
            return normal_breakdown(company, 0 if float(score) > 0 else 40)

        monkeypatch.setattr(
            lifecycle, "_proof_breakdown", zero_baseline_breakdown
        )
    lifecycle._drive_cycle(
        harness.service, harness.objects, daily_icps(), harness.runner_keys[0]
    )
    assert harness.service.publish(ROUND)["status"] == "ok"
    if zero_baseline:
        monkeypatch.setattr(lifecycle, "_proof_breakdown", normal_breakdown)
        with connection.cursor() as cursor:
            cursor.execute(AUTHORITY_HOLD.read_text())
            cursor.execute(AUTHORITY_HOLD.read_text())
        connection.commit()


def _inject_exhausted_provider_zero_and_failed_unknown(connection) -> str:
    """Model 24 baseline attempts: 18 accepted and six terminal failures."""
    identity = "sha256:" + hashlib.sha256(b"rerun330-failed-unknown").hexdigest()
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")

        def row_for(position):
            return _json(
                cursor,
                "SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s "
                "AND submission_id=%s AND kind='execute' AND icp_position=%s "
                "AND status='accepted'",
                (ROUND, BASELINE, position),
            )

        def insert_attempt(document, *, status, cause, keep_output):
            retry = copy.deepcopy(document)
            retry.update(
                {
                    "run_id": document["assignment_id"] + ":2",
                    "attempt": 2,
                    "status": status,
                    "terminal_cause": cause,
                    "claim_request_id": None,
                    "claim_request_hash": None,
                    "claim_response": None,
                    "lease_token_hash": None,
                    "lease_expires_at": None,
                    "per_icp_score": document["per_icp_score"] if keep_output else None,
                    "qualification_doc": document["qualification_doc"] if keep_output else None,
                    "output_ref": document["output_ref"] if keep_output else None,
                    "result_doc": (
                        document["result_doc"]
                        if keep_output
                        else {"terminal_status": cause}
                    ),
                }
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_runs SELECT "
                "(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*",
                (json.dumps(retry),),
            )
            return retry["run_id"]

        # The seed already has one accepted retry. Add one more retained retry.
        for position in (17,):
            accepted = row_for(position)
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',"
                "terminal_cause='provider_error',result_doc="
                "'{\"terminal_status\":\"provider_error\"}'::jsonb,output_ref=NULL,"
                "per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s",
                (accepted["run_id"],),
            )
            retry_run = insert_attempt(
                accepted, status="accepted", cause="accepted", keep_output=True
            )
            cursor.execute(
                "UPDATE public.lab_arena_runs SET scored_run_id=%s "
                "WHERE round_id=%s AND kind='score' AND scored_run_id=%s",
                (retry_run, ROUND, accepted["run_id"]),
            )
            assert cursor.rowcount == 1

        # Position 18 exhausted with a model-owned zero on both attempts.
        model_zero = row_for(18)
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',"
            "terminal_cause='budget_exhausted',result_doc="
            "'{\"terminal_status\":\"budget_exhausted\"}'::jsonb,output_ref=NULL,"
            "per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s",
            (model_zero["run_id"],),
        )
        insert_attempt(
            model_zero, status="failed", cause="budget_exhausted", keep_output=False
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET round_id=%s,submission_id=%s "
            "WHERE round_id=%s AND kind='score' AND scored_run_id=%s",
            (PRIOR_ARCHIVES[1], BASELINE + ":r328archive", ROUND, model_zero["run_id"]),
        )
        assert cursor.rowcount == 1

        # Position 19 exhausted provider retries; its failed call has unknown cost.
        provider_zero = row_for(19)
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='failed',"
            "terminal_cause='provider_error',result_doc="
            "'{\"terminal_status\":\"provider_error\"}'::jsonb,output_ref=NULL,"
            "per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s",
            (provider_zero["run_id"],),
        )
        failed_run = insert_attempt(
            provider_zero, status="failed", cause="provider_error", keep_output=False
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET round_id=%s,submission_id=%s "
            "WHERE round_id=%s AND kind='score' AND scored_run_id=%s",
            (PRIOR_ARCHIVES[1], BASELINE + ":r328archive", ROUND, provider_zero["run_id"]),
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
            (
                provider_zero["miner_hotkey"],
                ROUND,
                BASELINE,
                failed_run,
                provider_zero["stage"],
                identity,
            ),
        )
        cursor.execute("SET session_replication_role=origin")
        cursor.execute(
            "SELECT count(*),count(*) FILTER(WHERE status='accepted'),"
            "count(*) FILTER(WHERE status='failed') FROM public.lab_arena_runs "
            "WHERE round_id=%s AND submission_id=%s AND kind='execute'",
            (ROUND, BASELINE),
        )
        assert cursor.fetchone() == (24, 18, 6)
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


def _drive_rejudge_cycle(service, objects, runner_hotkey) -> None:
    """Use the real parallel close and score lifecycle without execute claims."""
    assert service.close_stage(ROUND, 1)["status"] == "ok"
    for stage, expected in ((1, 50), (2, 48)):
        assert service.open_scoring(ROUND, stage)["assignments"] == expected
        while True:
            pending = [
                row
                for row in service.store.list_runs(ROUND, stage=stage, kind="score")
                if row["status"] != "accepted"
            ]
            if not pending:
                break
            token = lifecycle.new_lease_token()
            request_id = lifecycle.contracts.new_request_id()
            run = service.store.claim_assignment(
                round_id=ROUND,
                runner_hotkey=runner_hotkey,
                declared_parallelism=1,
                slot_ceiling=20,
                excluded_miner_hotkeys=[runner_hotkey],
                request_id=request_id,
                request_hash=lifecycle.contracts.document_hash(
                    {"request_id": request_id}
                ),
                lease_token_hash=lifecycle.hash_lease_token(token),
            )
            assert run["status"] == "leased" and run["kind"] == "score"
            scored_run = service.store.get_run(run["scored_run_id"])
            executed = json.loads(objects.get(scored_run["output_ref"]).decode())
            document = lifecycle.scoring.build_scoring_output(
                run["scored_run_id"],
                [
                    lifecycle._proof_breakdown(
                        executed["companies"][0],
                        40 if run["submission_id"] == BASELINE else 0,
                    )
                ],
            )
            ref = "arena/score/%s.json" % run["run_id"]
            objects.put(ref, json.dumps(document).encode())
            stored = service.store.get_run(run["run_id"])
            evidence = lifecycle.judgment_cache.build_evidence_snapshot(
                output=document,
                cache_scope=stored["judgment_scope_doc"],
                source_score_run_id=run["run_id"],
                source_scored_run_id=run["scored_run_id"],
                source_output_ref=ref,
                source_runner_hotkey=runner_hotkey,
                runner_authority_exclusions=run["runner_authority_exclusions"],
            )
            assert service.store.complete_attempt(
                run_id=run["run_id"],
                lease_token_hash=lifecycle.hash_lease_token(token),
                result={"terminal_status": "accepted"},
                terminal_cause="accepted",
                output_ref=ref,
                judgment_evidence=evidence,
                judgment_evidence_hash=lifecycle.contracts.document_hash(evidence),
            )["status"] == "accepted"
        assert service.close_scoring(ROUND, stage)["status"] == "closed"
        scored = service.score_stage(ROUND, stage)
        assert scored["status"] == "ok", scored
        if stage == 1:
            assert service.open_stage(ROUND, 2)["status"] == "ok"
            assert service.close_stage(ROUND, 2)["status"] == "ok"


def test_sep20_rerun330_preserves_history_and_isolates_new_scorer_cache(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    schedule = _schedule()
    harness = IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _publish_rerun328(
            connection, harness, monkeypatch, zero_baseline=True
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT entry->>'submission_id',"
                "(entry->>'final_score')::double precision "
                "FROM public.lab_arena_rounds r CROSS JOIN LATERAL "
                "jsonb_array_elements(r.publication_doc->'final_ranking') entry "
                "WHERE r.round_id=%s ORDER BY entry->>'submission_id'",
                (ROUND,),
            )
            published_scores = dict(cursor.fetchall())
            assert published_scores[BASELINE] == 0
            assert all(
                score > 0
                for submission_id, score in published_scores.items()
                if submission_id != BASELINE
            )
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
            historical_award_before = _json(
                cursor,
                "SELECT jsonb_build_object('king_outcome',king_outcome,"
                "'king_hotkey',king_hotkey,'king_start_epoch',king_start_epoch,"
                "'effective_reward_epoch',effective_reward_epoch,"
                "'reward_basis_hash',reward_basis_hash,"
                "'reward_basis_doc',reward_basis_doc,"
                "'signing_key_doc',signing_key_doc,"
                "'reward_activated_at',reward_activated_at,"
                "'promotion_doc',promotion_doc,"
                "'baseline_promoted_at',baseline_promoted_at) "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert historical_award_before["king_outcome"] is not None
            assert historical_award_before["reward_activated_at"] is None
            assert historical_award_before["baseline_promoted_at"] is None
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
            old_score_cache_keys = set(
                _json(
                    cursor,
                    "SELECT jsonb_agg(judgment_cache_key) FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND kind='score'",
                    (ROUND,),
                )
            )
            terminal_score_runs = _scalar(
                cursor,
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='score'",
                (ROUND,),
            )
            execution_before = _execution_snapshot(cursor)
            artifact_proof = _execution_artifact_proof(cursor, harness.objects)
            rendered, scoring_before, scoring_after = _render(
                cursor, schedule, harness.objects
            )
            cursor.execute(rendered)
            cursor.execute(rendered)
            assert _scalar(
                cursor,
                "SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='execute'",
                (ARCHIVE,),
            ) == "0"
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
                "SELECT jsonb_build_object('king_outcome',king_outcome,"
                "'king_hotkey',king_hotkey,'king_start_epoch',king_start_epoch,"
                "'effective_reward_epoch',configuration_doc->'archived_effective_reward_epoch',"
                "'reward_basis_hash',configuration_doc->'archived_reward_basis_hash',"
                "'reward_basis_doc',configuration_doc->'archived_reward_basis_doc',"
                "'signing_key_doc',configuration_doc->'archived_signing_key_doc',"
                "'reward_activated_at',configuration_doc->'archived_reward_activated_at',"
                "'promotion_doc',configuration_doc->'archived_promotion_doc',"
                "'baseline_promoted_at',configuration_doc->'archived_baseline_promoted_at') "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ARCHIVE,),
            ) == historical_award_before
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
            assert _execution_snapshot(cursor) == execution_before
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
                "openrouter": 500,
                "scrapingdog": 200,
            }
            assert config["checkpoint_deadline_policy"] == "atomic_checkpoint_90m_v1"
            assert config["icp_wall_clock_seconds"] == 5400
            assert config["lease_ttl_seconds"] == 6300
            assert config["scorer_image_digest"] == NEW_SCORER_DIGEST
            assert config["scorer_image_reference"] == NEW_SCORER_REFERENCE
            assert (ref, size, sha256, commit) == (
                SOURCE_REF,
                SOURCE_SIZE,
                SOURCE_SHA256,
                SOURCE_COMMIT,
            )
            assert config["schedule"] == schedule
            assert _scalar(
                cursor,
                "SELECT configuration_doc->>'archived_execution_artifacts_sha256' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ARCHIVE,),
            ) == artifact_proof
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
                (ROUND, failed_identity),
            ) == "3"
        connection.commit()

        _drive_rejudge_cycle(
            harness.service, harness.objects, harness.runner_keys[0]
        )
        assert harness.service.publish(ROUND)["status"] == "ok"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id),"
                "bool_and(status='accepted'),"
                "bool_and(assignment_id LIKE '%%:score:rerun330'),"
                "bool_and(judgment_scope_doc->>'scorer_image_digest'=%s),"
                "bool_and(judgment_scope_doc->>'scorer_image_reference'=%s) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (NEW_SCORER_DIGEST, NEW_SCORER_REFERENCE, ROUND),
            )
            assert cursor.fetchone() == (98, 98, True, True, True, True)
            new_miner_keys = set(
                _json(
                    cursor,
                    "SELECT jsonb_agg(judgment_cache_key) FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND kind='score' AND submission_id<>%s",
                    (ROUND, BASELINE),
                )
            )
            assert len(new_miner_keys) == len(miner_cache_keys)
            assert new_miner_keys.isdisjoint(miner_cache_keys)
            new_score_cache_keys = set(
                _json(
                    cursor,
                    "SELECT jsonb_agg(judgment_cache_key) FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND kind='score'",
                    (ROUND,),
                )
            )
            assert None not in new_score_cache_keys
            assert new_score_cache_keys.isdisjoint(old_score_cache_keys)
            assert _json(
                cursor,
                "SELECT publication_doc FROM public.lab_arena_rounds WHERE round_id=%s",
                (ARCHIVE,),
            ) == publication_before
            assert _execution_snapshot(cursor) == execution_before
            cursor.execute(
                "SELECT (entry->>'final_score')::numeric FROM "
                "public.lab_arena_rounds r CROSS JOIN LATERAL "
                "jsonb_array_elements(r.publication_doc->'final_ranking') entry "
                "WHERE r.round_id=%s AND entry->>'submission_id'=%s",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] > 0
            cursor.execute(
                "SELECT effective_reward_epoch,reward_activated_at,"
                "promotion_doc,baseline_promoted_at FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (None, None, None, None)
            # The exact migration remains idempotent after its new positive publication.
            cursor.execute(rendered)


def test_sep20_rerun330_refuses_genuine_positive_publication_without_mutation(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    schedule = _schedule()
    harness = IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _publish_rerun328(connection, harness, monkeypatch)
        _inject_exhausted_provider_zero_and_failed_unknown(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),min((entry->>'final_score')::double precision) "
                "FROM public.lab_arena_rounds r CROSS JOIN LATERAL "
                "jsonb_array_elements(r.publication_doc->'final_ranking') entry "
                "WHERE r.round_id=%s AND entry->>'submission_id'=%s",
                (ROUND, BASELINE),
            )
            count, baseline_score = cursor.fetchone()
            assert count == 1 and baseline_score > 0
            rendered, _, _ = _render(cursor, schedule, harness.objects)
            before = _protected_database_snapshot(cursor)
        connection.commit()

        with pytest.raises(
            psycopg2.Error,
            match="requires exactly one zero published baseline",
        ):
            with connection.cursor() as cursor:
                cursor.execute(rendered)
        connection.rollback()

        with connection.cursor() as cursor:
            assert _protected_database_snapshot(cursor) == before
            assert _scalar(
                cursor,
                "SELECT count(*) FROM public.lab_arena_rounds WHERE round_id=%s",
                (ARCHIVE,),
            ) == "0"


def test_sep20_rerun330_refuses_malformed_publication_worker_loss_and_award_replay(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    schedule = _schedule()
    harness = IsolatedHarness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _publish_rerun328(connection, harness, monkeypatch, zero_baseline=True)
        _inject_exhausted_provider_zero_and_failed_unknown(connection)
        with connection.cursor() as cursor:
            original_get = harness.objects.get
            cursor.execute(
                "SELECT output_ref FROM public.lab_arena_runs WHERE round_id=%s "
                "AND kind='execute' AND status='accepted' ORDER BY run_id LIMIT 1",
                (ROUND,),
            )
            unreadable_ref = cursor.fetchone()[0]

            def fail_one(ref):
                if ref == unreadable_ref:
                    raise KeyError(ref)
                return original_get(ref)

            monkeypatch.setattr(harness.objects, "get", fail_one)
            with pytest.raises(ArtifactProofError, match="not hash-readable"):
                _render(cursor, schedule, harness.objects)
            monkeypatch.setattr(harness.objects, "get", original_get)

            original_publication = _json(
                cursor,
                "SELECT publication_doc FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )

            for label, value in (
                ("missing", None),
                ("nonnumeric", "zero"),
                ("negative", -1),
            ):
                publication = copy.deepcopy(original_publication)
                ranking = publication["final_ranking"]
                if value is None:
                    publication["final_ranking"] = [
                        item for item in ranking if item["submission_id"] != BASELINE
                    ]
                else:
                    next(
                        item for item in ranking if item["submission_id"] == BASELINE
                    )["final_score"] = value
                cursor.execute("SAVEPOINT malformed_publication")
                cursor.execute("SET LOCAL session_replication_role=replica")
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET publication_doc=%s::jsonb "
                    "WHERE round_id=%s",
                    (json.dumps(publication), ROUND),
                )
                cursor.execute("SET LOCAL session_replication_role=origin")
                rendered, _, _ = _render(cursor, schedule, harness.objects)
                before = _protected_database_snapshot(cursor)
                cursor.execute("SAVEPOINT refused_migration")
                with pytest.raises(
                    psycopg2.Error,
                    match="requires exactly one zero published baseline",
                ):
                    cursor.execute(_migration_body(rendered))
                cursor.execute("ROLLBACK TO SAVEPOINT refused_migration")
                assert _protected_database_snapshot(cursor) == before, label
                cursor.execute("ROLLBACK TO SAVEPOINT malformed_publication")

            for label, mutation in (
                (
                    "worker_loss",
                    "UPDATE public.lab_arena_runs SET terminal_cause='worker_lost' "
                    "WHERE round_id=%s AND submission_id=%s AND kind='execute' "
                    "AND icp_position=18 AND status='failed'",
                ),
                (
                    "promoted",
                    "UPDATE public.lab_arena_rounds SET promotion_doc='{}'::jsonb "
                    "WHERE round_id=%s",
                ),
                (
                    "reward_activated",
                    "UPDATE public.lab_arena_rounds SET reward_activated_at=clock_timestamp() "
                    "WHERE round_id=%s",
                ),
            ):
                cursor.execute("SAVEPOINT invalid_terminal")
                cursor.execute("SET LOCAL session_replication_role=replica")
                parameters = (ROUND, BASELINE) if label == "worker_loss" else (ROUND,)
                cursor.execute(mutation, parameters)
                assert cursor.rowcount == (2 if label == "worker_loss" else 1)
                cursor.execute("SET LOCAL session_replication_role=origin")
                rendered, _, _ = _render(cursor, schedule, harness.objects)
                before = _protected_database_snapshot(cursor)
                cursor.execute("SAVEPOINT refused_migration")
                with pytest.raises(psycopg2.Error, match="terminal preimage differs"):
                    cursor.execute(_migration_body(rendered))
                cursor.execute("ROLLBACK TO SAVEPOINT refused_migration")
                assert _protected_database_snapshot(cursor) == before, label
                cursor.execute("ROLLBACK TO SAVEPOINT invalid_terminal")


def test_sep20_rerun330_template_is_inactive_and_narrow():
    body = TEMPLATE.read_text()
    assert TEMPLATE.name not in CURRENT_SERVICE_MIGRATIONS
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
    assert all(round_id in body for round_id in PRIOR_ARCHIVES)
    assert ARCHIVE in body and "score:rerun330" in body
    assert '"openrouter":500' in body and '"openrouter":2000' not in body
    assert TERMINAL_SCORER_DIGEST in body and TERMINAL_SCORER_REFERENCE in body
    assert "__NEW_SCORER_IMAGE_DIGEST__" in body
    assert "__NEW_SCORER_IMAGE_REFERENCE__" in body
    assert "atomic_checkpoint_90m_v1" in body
    assert "icp_wall_clock_seconds')::INTEGER<>5400" in body
    assert "lease_ttl_seconds')::INTEGER<>6300" in body
    assert "inflight_calls" in body and "success_unresolved_calls" in body
    assert "uncertain_calls" not in body
    assert "requires exactly one zero published baseline" in body
    assert "jsonb_typeof(entry->'final_score')='number'" in body
    assert "output_ref IS NOT NULL)<>80" in body
    assert "archived_execution_artifacts_sha256" in body
    assert "331-arena-2026-09-20-promotion-reward-hold" in body
    assert "sep20_rerun328_promotion_reward_hold" in body
    assert "worker_lost" not in body
    assert "__TERMINAL_SCORE_RUN_COUNT__" in body
    assert "__TERMINAL_BASELINE_RUN_COUNT__" in body
    assert "__TERMINAL_BASELINE_ACCEPTED_RUN_COUNT__" in body
    assert "__TERMINAL_BASELINE_FAILED_RUN_COUNT__" in body
    assert "__TERMINAL_ACCEPTED_EXECUTION_COUNT__" in body
    assert "__TERMINAL_EXECUTION_ARTIFACT_COUNT__" in body
    assert "archived_execution_judgments" in body
    assert "INSERT INTO public.lab_arena_runs" not in body
    assert "kind='execute' AND submission_id='baseline-2026-09-20';" not in body
    assert "CREATE TRIGGER" not in body
    assert "INTERVAL '7 hours'" in body
    assert "INTERVAL '10 hours'" in body
    assert "INTERVAL '14 hours 1 second'" in body
    assert SOURCE_REF in body and str(SOURCE_SIZE) in body
    assert SOURCE_SHA256 in body and SOURCE_COMMIT in body
    marker_count = len(set(re.findall(r"__[A-Z0-9_]+__", body)))
    assert marker_count == 27
    if RENDERED.exists():
        values = _assert_exact_render_shape(RENDERED.read_text())
        assert len(values) == marker_count
        assert re.search(r"__[A-Z0-9_]+__", RENDERED.read_text()) is None
