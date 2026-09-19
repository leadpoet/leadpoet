"""Focused PostgreSQL proof for the terminal-only Sep19 rerun310 template."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import pytest

from lab_arena import contact_policy, contracts, output, scoring, verify
from qualification.scoring.arena_integrity import canonical_company_identity
from tests.lab_arena import sep19_baseline_recovery309_postgres_test as recovery309
from tests.lab_arena import sep18_published_rerun295_postgres_test as lifecycle
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "scripts/310-arena-2026-09-19-terminal309-newjudge-rerun.sql.template"
RENDERED = ROOT / "scripts/310-arena-2026-09-19-terminal309-newjudge-rerun.sql"
ROUND = "arena-2026-09-19"
BASELINE = "baseline-2026-09-19"
NEW_SOURCE_REF = "arena/arena-2026-09-19/sources/baseline-2026-09-19-rerun310.tar.gz"
NEW_SOURCE_SIZE = 688_770
NEW_SOURCE_SHA = "a67901a72a90965dd002a442236f44d113f81eda93f3778525c69a7dedf2cead"
NEW_SOURCE_COMMIT = "5e6d881882a2dae2be3ca783060f33e0380dd0de"
NEW_SCORER_DIGEST = "sha256:" + "3" * 64
NEW_SCORER_REFERENCE = "registry.test/judge@" + NEW_SCORER_DIGEST
BENCHMARK = {
    "schema_version": "leadpoet.lab_arena.benchmark.v1",
    "round_id": ROUND,
    "icps": daily_icps(),
}
BENCHMARK_BYTES = json.dumps(BENCHMARK).encode()
BANK_SHA = "sha256:" + hashlib.sha256(BENCHMARK_BYTES).hexdigest()


def _test_migrations() -> tuple[str, ...]:
    assert CURRENT_SERVICE_MIGRATIONS[-2:] == (
        "294-lab-arena-retire-open-cost-backfill.sql",
        "301-lab-arena-score-payer-boundary.sql",
    )
    return (
        CURRENT_SERVICE_MIGRATIONS[:-2]
        + (
            "289-lab-arena-per-icp-cost-policy.sql",
            "292-lab-arena-null-final-score-publication.sql",
        )
        + CURRENT_SERVICE_MIGRATIONS[-2:]
    )


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        _test_migrations()
    )


@pytest.fixture
def negative_database():
    yield from database_with_lab_arena_migration(
        _test_migrations()
    )


def _schedule() -> dict:
    start = datetime.now(timezone.utc) + timedelta(seconds=5)
    stamp = lambda value: value.isoformat().replace("+00:00", "Z")
    return {
        "submission_open": "2026-09-18T00:00:00Z",
        "submission_cutoff": "2026-09-19T00:00:00Z",
        "benchmark_deadline": stamp(start),
        "stage_1_start": stamp(start + timedelta(seconds=1)),
        "stage_1_close": stamp(start + timedelta(hours=3, seconds=1)),
        "stage_1_scoring_close": stamp(start + timedelta(hours=6)),
        "stage_2_start": stamp(start + timedelta(hours=6, seconds=1)),
        "stage_2_close": stamp(start + timedelta(hours=9, seconds=1)),
        "final_scoring_close": stamp(start + timedelta(hours=12)),
        "publication_deadline": stamp(start + timedelta(hours=12, seconds=1)),
    }


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha(cursor, query: str, parameters=()) -> str:
    cursor.execute(query, parameters)
    return cursor.fetchone()[0]


def _scorer_patch(definition: str) -> str:
    guard = "  FOR v_item IN"
    assignment = "    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;"
    assert definition.count(guard) == 1
    assert definition.count(assignment) == 1
    definition = definition.replace(
        guard,
        """  IF p_round_id='arena-2026-09-19'
     AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id='arena-2026-09-19-r310archive')
     AND public.lab_arena_sep19_rerun310_active_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_sep19_rerun310_frozen_state_invalid'
      USING ERRCODE='22023';
  END IF;
  FOR v_item IN""",
    )
    return definition.replace(
        assignment,
        """    IF p_round_id='arena-2026-09-19'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-19-r310archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun310';
    END IF;
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;""",
    )


def _proof_execution_v2(objects, icp, index, position, execute_id):
    company = lifecycle._proof_company(icp, index, position)
    company.pop("intent_details")
    company["fit_summary"] = "The company matches the requested industry and size."
    company["fit_evidence_urls"] = [company["company_website"]]
    for signal in company["intent_signals"]:
        signal["why_now"] = "The current operating milestone supports timely outreach."
        signal["snippet"] = signal["description"]
    document = output.output_document_from_bytes(
        json.dumps({
            "schema_version": contracts.CONTACT_OUTPUT_DOCUMENT_SCHEMA_VERSION,
            "companies": [company],
        }).encode(),
        expected_schema_version=contracts.CONTACT_OUTPUT_DOCUMENT_SCHEMA_VERSION,
    )
    objects.put("arena/output/%s.json" % execute_id, json.dumps(document).encode())
    breakdown = lifecycle._proof_breakdown(company, 40 if index == 0 else 0)
    breakdown["verifier_gate_receipts"] = [
        receipt for receipt in breakdown["verifier_gate_receipts"]
        if receipt["gate"] != "intent_details"
    ]
    row = verify.scored_row(
        "proof", position, execute_id, icp, document["companies"], [breakdown],
        scoring.build_scorer_policy(
            scoring_adapter_version=contact_policy.SCORING_ADAPTER,
        ),
    )
    identity = canonical_company_identity(company).key
    receipt = {"companies": [{
        "company_index": 0,
        "company_identity_key": identity,
        "company_qualified": breakdown["company_qualified"],
        "duplicate_company": False,
        "contact_qualified": breakdown["contact_qualified"],
    }]}
    return row["per_icp_score"], receipt


def _seed_published_recovery309(conn) -> None:
    recovery309._seed_terminal(conn)
    schedule = recovery309._schedule()
    with conn.cursor() as cursor:
        cursor.execute(recovery309._function_ddl(schedule))
        cursor.execute(
            "SELECT public.lab_arena_prepare_sep19_recovery309_v1(%s::jsonb)",
            (json.dumps(schedule),),
        )
        assert cursor.fetchone()[0]["status"] == "prepared"
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            """
            UPDATE public.lab_arena_runs
            SET status='accepted',terminal_cause='accepted',runner_hotkey=miner_hotkey,
                output_ref='arena/test/rerun310/'||run_id||'.json',
                result_doc='{}'::jsonb,participation_accepted_at=clock_timestamp(),
                per_icp_score=0::double precision,
                qualification_doc=jsonb_build_object('fixture','terminal309')
            WHERE round_id=%s AND kind='execute'
            """,
            (ROUND,),
        )
        assert cursor.rowcount == 100
        cursor.execute(
            """
            INSERT INTO public.lab_arena_runs(
              run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
              icp_position,attempt,kind,status,stage_generation,runner_hotkey,
              terminal_cause,result_doc
            )
            SELECT assignment_id||':2',assignment_id,round_id,submission_id,
                   miner_hotkey,stage,icp_position,2,kind,'failed',stage_generation,
                   miner_hotkey,'provider_error','{}'::jsonb
            FROM public.lab_arena_runs
            WHERE round_id=%s AND kind='execute' AND submission_id=%s
              AND icp_position IN (14,15) AND attempt=1
            """,
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 2
        cursor.execute(
            """
            INSERT INTO public.lab_arena_runs(
              run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
              icp_position,attempt,kind,status,stage_generation,runner_hotkey,
              terminal_cause,output_ref,scored_run_id,judgment_cache_key,
              judgment_input_hash,judgment_scope_doc
            )
            SELECT execute.round_id||':'||execute.submission_id||':'||execute.stage::text||':'||
                     execute.icp_position::text||':score:1',
                   execute.round_id||':'||execute.submission_id||':'||execute.stage::text||':'||
                     execute.icp_position::text||':score',
                   execute.round_id,execute.submission_id,execute.miner_hotkey,
                   execute.stage,execute.icp_position,1,
                   'score','accepted',execute.stage_generation,execute.miner_hotkey,'accepted',
                   'arena/test/judgment/'||execute.run_id||'.json',execute.run_id,
                   'sha256:'||repeat('a',64),'sha256:'||repeat('b',64),
                   jsonb_build_object(
                     'scorer_image_digest',round.configuration_doc->>'scorer_image_digest',
                     'scorer_image_reference',round.configuration_doc->>'scorer_image_reference')
            FROM public.lab_arena_runs execute
            JOIN public.lab_arena_rounds round USING(round_id)
            WHERE execute.round_id=%s AND execute.kind='execute'
              AND execute.status='accepted'
            """,
            (ROUND,),
        )
        assert cursor.rowcount == 100
        cursor.execute(
            """
            INSERT INTO public.lab_arena_runs(
              run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
              icp_position,attempt,kind,status,stage_generation,runner_hotkey,
              terminal_cause,scored_run_id,result_doc
            )
            SELECT assignment_id||':2',assignment_id,round_id,submission_id,
                   miner_hotkey,stage,icp_position,2,kind,'failed',stage_generation,
                   miner_hotkey,'provider_error',scored_run_id,'{}'::jsonb
            FROM public.lab_arena_runs
            WHERE round_id=%s AND kind='score' ORDER BY run_id LIMIT 1
            """,
            (ROUND,),
        )
        assert cursor.rowcount == 1
        cursor.execute(
            """
            WITH calls AS (
              SELECT n,'sha256:'||md5('failed-score-'||n::text)||
                md5('failed-score-'||n::text) AS identity
              FROM generate_series(1,18) n
            ), entries AS (
              SELECT n,identity,kind FROM calls
              CROSS JOIN (VALUES('reservation'),('dispatch'),('uncertain')) k(kind)
            )
            INSERT INTO public.lab_arena_ledger(
              entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
              call_identity,provider,operation_id,funding_source,amount_microusd,
              entry_doc
            )
            SELECT entries.kind,run.miner_hotkey,run.round_id,run.submission_id,
                   run.run_id,run.stage,identity,'openrouter','openrouter.score','host',
                   CASE WHEN entries.kind='uncertain' AND n=1 THEN 4102563 ELSE 0 END,
                   CASE WHEN entries.kind='uncertain'
                     THEN '{"call":{"call_succeeded":false}}'::jsonb ELSE '{}'::jsonb END
            FROM entries
            CROSS JOIN LATERAL (
              SELECT * FROM public.lab_arena_runs
              WHERE round_id=%s AND kind='score' ORDER BY run_id LIMIT 1
            ) run
            """,
            (ROUND,),
        )
        assert cursor.rowcount == 54
        for stage in (1, 2):
            cursor.execute(
                """
                SELECT jsonb_build_object(
                  'schema_version','leadpoet.lab_arena.scoring_plan.v1',
                  'round_id',%s,'stage',%s,
                  'work_items',jsonb_agg(jsonb_build_object(
                    'submission_id',submission_id,'icp_position',icp_position,
                    'scored_run_id',run_id,'output_ref',output_ref)
                    ORDER BY submission_id,icp_position),'zero_rows','[]'::jsonb)
                FROM public.lab_arena_runs
                WHERE round_id=%s AND kind='execute' AND stage=%s
                  AND status='accepted'
                """,
                (ROUND, stage, ROUND, stage),
            )
            plan = cursor.fetchone()[0]
            cursor.execute(
                f"UPDATE public.lab_arena_rounds SET stage{stage}_scoring_plan_doc=%s::jsonb WHERE round_id=%s",
                (json.dumps(plan), ROUND),
            )
        cursor.execute(
            """
            UPDATE public.lab_arena_rounds
            SET status='published',published_at=clock_timestamp(),
                configuration_doc=configuration_doc||jsonb_build_object(
                  'runner_slot_ceiling',20,'parallel_twenty_icp_execution',true,
                  'execution_cap_microusd',80000000,
                  'execution_icp_cap_microusd',4000000,
                  'scoring_cap_microusd',50000000,
                  'cost_per_company_microusd',800000,
                  'baseline_hotkey',(SELECT miner_hotkey FROM public.lab_arena_submissions
                    WHERE round_id=%s AND submission_id=%s)),
                publication_doc=jsonb_build_object(
                  'schema_version','leadpoet.lab_arena.publication.v1',
                  'king_decision',jsonb_build_object('outcome','no_king')),
                king_outcome='no_king',king_hotkey=NULL,king_start_epoch=0,
                effective_reward_epoch=25266,
                reward_basis_hash='sha256:'||repeat('c',64),
                reward_basis_doc=jsonb_build_object('fixture','activated309'),
                signing_key_doc=jsonb_build_object('fixture','activated309'),
                reward_activated_at=clock_timestamp(),promotion_required=TRUE,
                confirmation_bank_hash=%s,
                cancel_reason=NULL
            WHERE round_id=%s
            """,
            (ROUND, BASELINE, BANK_SHA, ROUND),
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set("
            "configuration_doc,'{scorer_policy}',%s::jsonb,true) WHERE round_id=%s",
            (
                json.dumps(scoring.build_scorer_policy(
                    scoring_adapter_version=contact_policy.SCORING_ADAPTER,
                )),
                ROUND,
            ),
        )
        cursor.execute(
            """
            INSERT INTO public.lab_arena_accepted_weight_states(
              network,netuid,epoch,state_hash,state_doc
            ) VALUES(
              'finney',71,25266,'sha256:'||repeat('d',64),
              jsonb_build_object(
                'schema_version','leadpoet.arena.accepted_weight_state.v1',
                'network','finney','genesis_hash','0xfixture','netuid',71,
                'epoch',25266,'valid_from_block',1,'valid_until_block',2,
                'reward_basis',jsonb_build_object('fixture','activated309'),
                'burn_hotkey','fixture','issued_at','2026-09-19T06:34:47Z',
                'state_hash','sha256:'||repeat('d',64),
                'signature',jsonb_build_object('fixture','activated309')))
            """
        )
        cursor.execute(
            "UPDATE public.lab_arena_submissions SET code_review_status='passed',"
            "code_review_attempts=1,code_review_doc='{}'::jsonb,"
            "code_review_claim='sha256:'||repeat('e',64),"
            "code_review_started_at=clock_timestamp(),code_review_expires_at=NULL "
            "WHERE round_id=%s AND submission_id<>%s",
            (ROUND, BASELINE),
        )
        assert cursor.rowcount == 4
        cursor.execute("SET session_replication_role=origin")
    conn.commit()


def _render(cursor, schedule: dict) -> str:
    cursor.execute(
        "SELECT pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
    )
    scorer_definition = cursor.fetchone()[0]
    patched_definition = _scorer_patch(scorer_definition)
    values = {
        "__NEW_SOURCE_REF__": NEW_SOURCE_REF,
        "__NEW_SOURCE_SIZE_BYTES__": str(NEW_SOURCE_SIZE),
        "__NEW_SOURCE_SHA256__": NEW_SOURCE_SHA,
        "__NEW_SOURCE_COMMIT__": NEW_SOURCE_COMMIT,
        "__NEW_SCORER_DIGEST__": NEW_SCORER_DIGEST,
        "__NEW_SCORER_REFERENCE__": NEW_SCORER_REFERENCE,
        "__RECOVERY_SCHEDULE_JSON__": _compact(schedule),
        "__SCORING_DEFINITION_SHA256__": hashlib.sha256(
            scorer_definition.encode()
        ).hexdigest(),
        "__PATCHED_SCORING_DEFINITION_SHA256__": hashlib.sha256(
            patched_definition.encode()
        ).hexdigest(),
        "__TERMINAL_EXECUTE_RUN_COUNT__": "102",
        "__TERMINAL_BASELINE_RUN_COUNT__": "22",
        "__TERMINAL_SCORE_RUN_COUNT__": "101",
        "__TERMINAL_ROUND_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex') "
            "FROM public.lab_arena_rounds r WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_BASELINE_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(to_jsonb(s)::text,'sha256'),'hex') "
            "FROM public.lab_arena_submissions s WHERE round_id=%s AND submission_id=%s",
            (ROUND, BASELINE),
        ),
        "__TERMINAL_MINER_SUBMISSIONS_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode(extensions.digest(to_jsonb(s)::text,'sha256'),'hex'),'' ORDER BY submission_id),''),'sha256'),'hex') FROM public.lab_arena_submissions s WHERE round_id=%s AND submission_id<>%s",
            (ROUND, BASELINE),
        ),
        "__TERMINAL_RUNS_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex') FROM public.lab_arena_runs r WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_LEDGER_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(coalesce(string_agg(encode(extensions.digest(to_jsonb(l)::text,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex') FROM public.lab_arena_ledger l WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_BANK_STATE_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(jsonb_build_object("
            "'confirmation_bank_ref',confirmation_bank_ref,"
            "'confirmation_bank_hash',confirmation_bank_hash,"
            "'confirmation_cohort',confirmation_cohort)::text,'sha256'),'hex') "
            "FROM public.lab_arena_rounds WHERE round_id=%s",
            (ROUND,),
        ),
        "__TERMINAL_REWARD_AUTHORITY_SHA256__": _sha(
            cursor,
            "SELECT encode(extensions.digest(jsonb_build_object("
            "'rewards_enabled',rewards_enabled,"
            "'effective_reward_epoch',effective_reward_epoch,"
            "'reward_basis_hash',reward_basis_hash,'reward_basis_doc',reward_basis_doc,"
            "'signing_key_doc',signing_key_doc,'reward_activated_at',reward_activated_at,"
            "'king_outcome',king_outcome,'king_hotkey',king_hotkey,"
            "'king_start_epoch',king_start_epoch,'promotion_required',promotion_required,"
            "'promotion_doc',promotion_doc,'baseline_promoted_at',baseline_promoted_at)"
            "::text,'sha256'),'hex') FROM public.lab_arena_rounds WHERE round_id=%s",
            (ROUND,),
        ),
    }
    sql = TEMPLATE.read_text()
    assert set(re.findall(r"__[A-Z0-9_]+__", sql)) == set(values)
    for marker, value in values.items():
        sql = sql.replace(marker, value)
    assert re.search(r"__[A-Z0-9_]+__", sql) is None
    return sql


def test_terminal_recovery_archives_judgments_and_retains_miner_sources(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    conn = psycopg2.connect(**dsn)
    try:
        _seed_published_recovery309(conn)
        schedule = _schedule()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r "
                "WHERE round_id=%s AND submission_id<>%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            miner_runs_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l "
                "WHERE round_id=%s AND submission_id<>%s AND NOT EXISTS("
                "SELECT 1 FROM public.lab_arena_runs s WHERE s.round_id=%s "
                "AND s.kind='score' AND s.run_id=l.run_id)",
                (ROUND, BASELINE, ROUND),
            )
            miner_ledger_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT public.lab_arena__successful_call_cost_state(submission_id,'score',NULL) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' "
                "ORDER BY run_id LIMIT 1",
                (ROUND,),
            )
            failed_cost = cursor.fetchone()[0]
            assert failed_cost["inflight_calls"] == 0
            assert failed_cost["success_unresolved_calls"] == 0
            assert failed_cost["uncertain_calls"] == 18
            assert failed_cost["reserved_or_uncertain_microusd"] == 4_102_563
            cursor.execute(
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r "
                "WHERE round_id='arena-2026-09-19-r309archive'"
            )
            recovery309_archive_before = cursor.fetchone()[0]
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r310archive"),
            )
            prior_ledger_before = cursor.fetchone()[0]
            assert prior_ledger_before
            cursor.execute(
                "SELECT rewards_enabled,effective_reward_epoch,reward_basis_hash,"
                "reward_basis_doc,signing_key_doc,reward_activated_at,king_outcome,"
                "king_hotkey,king_start_epoch,promotion_required,promotion_doc,"
                "baseline_promoted_at FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            reward_state_before = cursor.fetchone()
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(w) ORDER BY network,netuid,epoch) "
                "FROM public.lab_arena_accepted_weight_states w"
            )
            accepted_weights_before = cursor.fetchone()[0]
            rendered = _render(cursor, schedule)
            cursor.execute(rendered)
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE status='pending'),"
                "bool_and(assignment_id LIKE '%%:rerun310') FROM public.lab_arena_runs "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20, True)
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE kind='score') FROM public.lab_arena_runs "
                "WHERE round_id='arena-2026-09-19-r310archive'"
            )
            assert cursor.fetchone() == (123, 101)
            cursor.execute(
                "SELECT count(*),coalesce(sum(amount_microusd),0) "
                "FROM public.lab_arena_ledger WHERE round_id=%s "
                "AND entry_kind='uncertain' AND operation_id='openrouter.score'",
                (ROUND + "-r310archive",),
            )
            assert cursor.fetchone() == (18, 4_102_563)
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r "
                "WHERE round_id=%s AND submission_id<>%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            miner_runs_after = cursor.fetchone()[0]
            for before, after in zip(miner_runs_before, miner_runs_after, strict=True):
                for field in ("per_icp_score", "qualification_doc", "updated_at"):
                    before.pop(field, None)
                    after.pop(field, None)
            assert miner_runs_after == miner_runs_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l "
                "WHERE round_id=%s AND submission_id<>%s",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] == miner_ledger_before
            cursor.execute(
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r "
                "WHERE round_id='arena-2026-09-19-r309archive'"
            )
            assert cursor.fetchone()[0] == recovery309_archive_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) "
                "FROM public.lab_arena_ledger l WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r310archive"),
            )
            assert cursor.fetchone()[0] == prior_ledger_before
            cursor.execute(
                "SELECT rewards_enabled,effective_reward_epoch,reward_basis_hash,"
                "reward_basis_doc,signing_key_doc,reward_activated_at,king_outcome,"
                "king_hotkey,king_start_epoch,promotion_required,promotion_doc,"
                "baseline_promoted_at FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == reward_state_before
            cursor.execute(
                "SELECT jsonb_agg(to_jsonb(w) ORDER BY network,netuid,epoch) "
                "FROM public.lab_arena_accepted_weight_states w"
            )
            assert cursor.fetchone()[0] == accepted_weights_before
            cursor.execute(
                "SELECT source_ref,source_size_bytes,submission_doc->>'source_commit' "
                "FROM public.lab_arena_submissions WHERE round_id=%s AND submission_id=%s",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (
                NEW_SOURCE_REF,
                NEW_SOURCE_SIZE,
                NEW_SOURCE_COMMIT,
            )
            # Applying the exact rendered migration again must accept the already
            # patched scorer and preserve the prepared state.
            cursor.execute(rendered)
            cursor.execute(
                "SELECT public.lab_arena_prepare_sep19_rerun310_v1(%s,%s,%s,%s::jsonb,%s,%s)",
                (
                    NEW_SOURCE_SIZE,
                    NEW_SOURCE_SHA,
                    NEW_SOURCE_COMMIT,
                    json.dumps(schedule),
                    NEW_SCORER_DIGEST,
                    NEW_SCORER_REFERENCE,
                ),
            )
            assert cursor.fetchone()[0]["status"] == "existing"
            cursor.execute("BEGIN")
            cursor.execute("SAVEPOINT frozen_state_negative")
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET confirmation_bank_hash=%s WHERE round_id=%s",
                ("sha256:" + "0" * 64, ROUND),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
            with pytest.raises(psycopg2.Error, match="existing Sep19 rerun310 differs"):
                cursor.execute(
                    "SELECT public.lab_arena_prepare_sep19_rerun310_v1(%s,%s,%s,%s::jsonb,%s,%s)",
                    (NEW_SOURCE_SIZE, NEW_SOURCE_SHA, NEW_SOURCE_COMMIT,
                     json.dumps(schedule), NEW_SCORER_DIGEST, NEW_SCORER_REFERENCE),
                )
            cursor.execute("ROLLBACK TO SAVEPOINT frozen_state_negative")
        conn.commit()

        monkeypatch.setattr(lifecycle, "ROUND", ROUND)
        monkeypatch.setattr(lifecycle, "BASELINE", BASELINE)
        monkeypatch.setattr(lifecycle, "_proof_execution", _proof_execution_v2)
        proof_breakdown = lifecycle._proof_breakdown
        monkeypatch.setattr(
            lifecycle,
            "_proof_breakdown",
            lambda company, score: proof_breakdown(company, 40 if score else 39),
        )
        harness = lifecycle.Harness(
            lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha", "beta"]
        )
        harness.round_id = ROUND
        harness.objects.put(f"arena/{ROUND}/benchmark.json", BENCHMARK_BYTES)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT run_id,output_ref,icp_position FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute' AND submission_id<>%s",
                (ROUND, BASELINE),
            )
            for index, (run_id, output_ref, position) in enumerate(cursor.fetchall(), 1):
                _proof_execution_v2(
                    harness.objects, BENCHMARK["icps"][position], index, position, run_id
                )
                harness.objects.put(
                    output_ref, harness.objects.get(f"arena/output/{run_id}.json")
                )
            cursor.execute(
                "SELECT r.status,r.stage_generation,"
                "(SELECT count(*) FROM public.lab_arena_runs x WHERE x.round_id=r.round_id "
                "AND x.submission_id=%s AND x.kind='execute' AND x.stage=1 "
                "AND x.status='pending' AND x.stage_generation=r.stage_generation),"
                "(SELECT bool_and(s.status='frozen' AND s.is_king AND "
                "s.miner_hotkey=r.configuration_doc->>'baseline_hotkey') "
                "FROM public.lab_arena_submissions s WHERE s.round_id=r.round_id "
                "AND s.submission_id=%s),"
                "r.configuration_doc->'runner_hotkeys' ? %s,"
                "r.configuration_doc ? 'company_quality_policy' "
                "FROM public.lab_arena_rounds r WHERE r.round_id=%s",
                (BASELINE, BASELINE, harness.runner_keys[1], ROUND),
            )
            assert cursor.fetchone() == ("stage1", 3, 10, True, None, False)
        conn.commit()
        lifecycle._drive_cycle(
            harness.service, harness.objects, BENCHMARK["icps"], harness.runner_keys[1]
        )
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT count(*),count(*) FILTER(WHERE assignment_id LIKE '%%:score:rerun310'),"
                "bool_and(judgment_scope_doc->>'scorer_image_digest'=%s),"
                "bool_and(judgment_scope_doc->>'scorer_image_reference'=%s) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND status='accepted'",
                (NEW_SCORER_DIGEST, NEW_SCORER_REFERENCE, ROUND),
            )
            assert cursor.fetchone() == (100, 100, True, True)
        conn.commit()
        assert harness.service.publish(ROUND)["status"] == "ok"
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT rewards_enabled,effective_reward_epoch,reward_basis_hash,"
                "reward_basis_doc,signing_key_doc,reward_activated_at,king_outcome,"
                "king_hotkey,king_start_epoch,promotion_required,promotion_doc,"
                "baseline_promoted_at FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == reward_state_before
            cursor.execute(
                "SELECT publication_doc,participants FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            publication, participants = cursor.fetchone()
            assert publication["king_decision"]["outcome"] == "no_king"
            assert len(publication["final_ranking"]) == 5
            assert all(row["final_score"] > 0 for row in publication["final_ranking"])
            baseline_score = next(
                row["final_score"] for row in publication["final_ranking"]
                if row["submission_id"] == BASELINE
            )
            assert all(
                row["final_score"] < baseline_score + 1
                for row in publication["final_ranking"]
                if row["submission_id"] != BASELINE
            )
            cursor.execute(
                "CREATE TEMP TABLE round_guard_probe AS TABLE public.lab_arena_rounds WITH NO DATA"
            )
            cursor.execute(
                "INSERT INTO round_guard_probe SELECT * FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            cursor.execute(
                "UPDATE round_guard_probe SET status='scored' WHERE round_id=%s",
                (ROUND,),
            )
            challenger = next(item for item in participants if not item.get("is_king"))
            changed = {
                "outcome": "crowned",
                "king_submission_id": challenger["submission_id"],
                "king_hotkey": challenger["miner_hotkey"],
                "winner_submission_id": challenger["submission_id"],
            }
            cursor.execute(
                "CREATE TRIGGER round_guard_probe BEFORE UPDATE ON round_guard_probe "
                "FOR EACH ROW EXECUTE FUNCTION public.lab_arena_rounds_write_once_v1()"
            )
            with pytest.raises(
                psycopg2.Error,
                match="round publication and commitment columns are write-once",
            ):
                cursor.execute(
                    "UPDATE round_guard_probe SET status='published',publication_doc=jsonb_set("
                    "publication_doc,'{king_decision}',%s::jsonb,true),"
                    "king_outcome='crowned',king_hotkey=%s WHERE round_id=%s",
                    (json.dumps(changed), challenger["miner_hotkey"], ROUND),
                )
        conn.commit()
    finally:
        conn.close()


def test_template_is_terminal_only_and_has_no_release_values():
    body = TEMPLATE.read_text()
    assert body.count("CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring_v2") == 0
    assert "pg_get_functiondef" in body and "__SCORING_DEFINITION_SHA256__" in body
    assert "ELSIF current_hash<>'__PATCHED_SCORING_DEFINITION_SHA256__'" in body
    assert "status NOT IN('published','cancelled')" in body
    assert "inflight_calls" in body and "success_unresolved_calls" in body
    assert "uncertain_calls" not in body
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
    assert "UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL" in body
    assert "submission_id<>'baseline-2026-09-19'" in body
    assert "publication_guard" not in body
    assert "lab_arena_accepted_weight_states" in body
    assert "__TERMINAL_BANK_STATE_SHA256__" in body
    assert "__TERMINAL_REWARD_AUTHORITY_SHA256__" in body
    assert "active_round.configuration_doc ? 'company_quality_policy'" in body
    assert "__NEW_SOURCE_COMMIT__" in body and "__NEW_SCORER_DIGEST__" in body
    assert body.count("jsonb_build_array(entry_id,xmin::TEXT,ctid::TEXT)") == 2
    assert "transaction-local physical tuple" in body


@pytest.mark.parametrize(
    "mutation",
    (
        """
        UPDATE public.lab_arena_ledger SET amount_microusd=amount_microusd
        WHERE entry_id=(SELECT min(entry_id) FROM public.lab_arena_ledger
          WHERE round_id NOT IN('arena-2026-09-19','arena-2026-09-19-r310archive'));
        """,
        """
        WITH moved AS (
          DELETE FROM public.lab_arena_ledger
          WHERE entry_id=(SELECT min(entry_id) FROM public.lab_arena_ledger
            WHERE round_id NOT IN('arena-2026-09-19','arena-2026-09-19-r310archive'))
          RETURNING *
        ) INSERT INTO public.lab_arena_ledger SELECT * FROM moved;
        """,
    ),
    ids=("same-value-update", "balanced-delete-insert"),
)
def test_transaction_tuple_fingerprint_rejects_non_target_rewrite(
    negative_database, mutation
):
    psycopg2, dsn = negative_database
    conn = psycopg2.connect(**dsn)
    try:
        _seed_published_recovery309(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_ledger WHERE round_id NOT IN(%s,%s)",
                (ROUND, ROUND + "-r310archive"),
            )
            assert cursor.fetchone()[0] > 0
            rendered = _render(cursor, _schedule())
            seam = " ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;"
            assert rendered.count(seam) == 1
            attacked = rendered.replace(seam, mutation + seam, 1)
            with pytest.raises(
                psycopg2.Error, match="Sep19 rerun310 atomic preservation differs"
            ):
                cursor.execute(attacked)
        conn.rollback()
    finally:
        conn.close()


def test_rendered_migration_has_exact_reviewed_identity():
    raw = RENDERED.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "fad535e80727e70a0b21eb86cb459a30e229da0c933eb628dbd3e647077ff8a6"
    )
    body = raw.decode()
    assert re.search(r"__[A-Z0-9_]+__", body) is None
    assert "sha256:19eae396b4168866a573122fce175316596e394049b51b544f5ee4262eaf5ec8" in body
    assert "5e6d881882a2dae2be3ca783060f33e0380dd0de" in body
    assert '"benchmark_deadline":"2026-09-19T07:45:00Z"' in body
