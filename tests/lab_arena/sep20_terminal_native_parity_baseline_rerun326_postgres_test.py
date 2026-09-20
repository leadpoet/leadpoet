"""Sealed Sep20 native-parity baseline recovery against disposable PostgreSQL."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.lab_arena import sep18_published_rerun295_postgres_test as lifecycle
from tests.lab_arena import sep19_terminal309_newjudge_rerun310_postgres_test as rerun310
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey


ROUND = "arena-2026-09-20"
BASELINE = "baseline-2026-09-20"
ARCHIVE = ROUND + "-r326archive"
TEMPLATE = (
    Path(__file__).parents[2]
    / "scripts/327-arena-2026-09-20-terminal-native-parity-baseline-rerun.sql.template"
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


def _schedule() -> dict[str, str]:
    start = datetime.now(timezone.utc) + timedelta(minutes=5)
    stamp = lambda value: value.isoformat().replace("+00:00", "Z")
    return {
        "submission_open": "2026-09-19T00:00:00Z",
        "submission_cutoff": "2026-09-20T00:00:00Z",
        "benchmark_deadline": stamp(start),
        "stage_1_start": stamp(start + timedelta(seconds=1)),
        "stage_1_close": stamp(start + timedelta(hours=3, seconds=1)),
        "stage_1_scoring_close": stamp(start + timedelta(hours=6)),
        "stage_2_start": stamp(start + timedelta(hours=6, seconds=1)),
        "stage_2_close": stamp(start + timedelta(hours=9, seconds=1)),
        "final_scoring_close": stamp(start + timedelta(hours=12)),
        "publication_deadline": stamp(start + timedelta(hours=12, seconds=1)),
    }


def _configuration(
    schedule: dict[str, str], *, baseline_hotkey: str, runner_hotkeys: list[str]
) -> dict:
    return {
        "schema_version": "leadpoet.lab_arena.round_configuration.v1",
        "round_id": ROUND,
        "mode": "live",
        "network_name": "finney",
        "netuid": 71,
        "rewards_enabled": True,
        "call_quotas": {"scrapingdog": 200, "deepline": 200, "openrouter": 200},
        "scoring_call_quotas": {"scrapingdog": 150, "deepline": 40, "openrouter": 120},
        "sourcing_cost_eligibility_policy": "successful_calls_per_icp_v1",
        "integrity_policy": "arena_integrity_v1",
        "contact_policy": "contacts_v1",
        "execution_cap_microusd": 80_000_000,
        "execution_icp_cap_microusd": 4_000_000,
        "cost_per_company_microusd": 800_000,
        "icp_wall_clock_seconds": 2700,
        "lease_ttl_seconds": 3600,
        "scoring_wall_clock_seconds": 900,
        "stage_1_icp_count": 10,
        "stage_2_icp_count": 10,
        "parallel_twenty_icp_execution": True,
        "runner_slot_ceiling": 20,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": "registry.example/lab/scorer@sha256:" + "a" * 64,
        "scorer_policy": lifecycle.scoring.build_scorer_policy(
            scoring_adapter_version="qualification_contacts_v3"
        ),
        "baseline_hotkey": baseline_hotkey,
        "runner_hotkeys": runner_hotkeys,
        "schedule": schedule,
    }


def _seed_terminal(connection, schedule: dict[str, str], *, runner_hotkeys: list[str]) -> None:
    miners = [hotkey(f"sep20-r326-miner-{index}") for index in range(4)]
    baseline_hotkey = hotkey("sep20-r326-baseline")
    frozen = [
        {
            "submission_id": f"sep20-miner-{index}",
            "miner_hotkey": miner,
            "source_ref": f"arena/{ROUND}/sources/miner-{index}.tar.gz",
            "source_size_bytes": 1000 + index,
            "is_king": False,
        }
        for index, miner in enumerate(miners)
    ]
    frozen.append(
        {
            "submission_id": BASELINE,
            "miner_hotkey": baseline_hotkey,
            "source_ref": f"arena/{ROUND}/sources/{BASELINE}-2fcfa34.tar.gz",
            "source_size_bytes": 861_023,
            "is_king": True,
        }
    )
    participants = [dict(item) for item in frozen]
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            """
            INSERT INTO public.lab_arena_rounds(
              round_id,status,status_generation,stage_generation,
              configuration_doc,rewards_enabled,participants,benchmark_ref,
              evaluation_date,icp_set_date,publication_doc,published_at,
              king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
              reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
              promotion_required,promotion_doc,baseline_promoted_at,cancel_reason)
            VALUES(%s,'published',4,3,%s::jsonb,TRUE,%s::jsonb,%s,%s,%s,
              '{"fixture":"stale-terminal-publication"}'::jsonb,clock_timestamp(),
              'crowned',%s,100,101,%s,%s::jsonb,%s::jsonb,clock_timestamp(),
              FALSE,%s::jsonb,clock_timestamp(),NULL)
            """,
            (
                ROUND,
                json.dumps(
                    _configuration(
                        schedule,
                        baseline_hotkey=baseline_hotkey,
                        runner_hotkeys=runner_hotkeys,
                    )
                ),
                json.dumps(participants),
                f"arena/{ROUND}/benchmark.json",
                "2026-09-20",
                "2026-09-19",
                miners[0],
                "sha256:" + "c" * 64,
                json.dumps({"fixture": "stale-reward-basis"}),
                json.dumps({"fixture": "stale-signing-key"}),
                json.dumps({"fixture": "stale-promotion"}),
            ),
        )
        for item in frozen:
            submission_doc = {
                "source_ref": item["source_ref"],
                "source_size_bytes": item["source_size_bytes"],
                "source_sha256": "b" * 64,
                "source_commit": "2fcfa34d22db214b0317b13665ef68493cb9ee6c",
            }
            cursor.execute(
                """
                INSERT INTO public.lab_arena_submissions(
                  submission_id,round_id,miner_hotkey,status,is_king,
                  submission_doc,source_ref,source_size_bytes,code_review_status,
                  code_review_attempts,code_review_doc,code_review_claim,
                  code_review_started_at)
                VALUES(%s,%s,%s,'frozen',%s,%s::jsonb,%s,%s,%s,%s,%s::jsonb,
                  %s,%s)
                """,
                (
                    item["submission_id"],
                    ROUND,
                    item["miner_hotkey"],
                    item["is_king"],
                    json.dumps(submission_doc),
                    item["source_ref"],
                    item["source_size_bytes"],
                    "pending" if item["is_king"] else "passed",
                    0 if item["is_king"] else 1,
                    None if item["is_king"] else json.dumps({"verdict": "pass"}),
                    None if item["is_king"] else "sha256:" + "d" * 64,
                    None if item["is_king"] else datetime.now(timezone.utc),
                ),
            )
        for index in range(3):
            cursor.execute(
                """
                INSERT INTO public.lab_arena_submissions(
                  submission_id,round_id,miner_hotkey,status,is_king)
                VALUES(%s,%s,%s,'rejected',FALSE)
                """,
                (f"sep20-rejected-{index}", ROUND, hotkey(f"sep20-rejected-{index}")),
            )
        for submission in frozen[:4]:
            for position in range(20):
                stage = 1 if position < 10 else 2
                assignment = f"{ROUND}:{submission['submission_id']}:{stage}:{position}"
                cursor.execute(
                    """
                    INSERT INTO public.lab_arena_runs(
                      run_id,assignment_id,round_id,submission_id,miner_hotkey,
                      stage,icp_position,attempt,kind,status,runner_hotkey,
                      terminal_cause,result_doc,output_ref,per_icp_score,
                      qualification_doc,stage_generation)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,1,'execute','accepted',%s,
                           'accepted','{"terminal_status":"accepted"}'::jsonb,
                           %s,1.25,'{"qualified":true}'::jsonb,3)
                    """,
                    (
                        assignment + ":1",
                        assignment,
                        ROUND,
                        submission["submission_id"],
                        submission["miner_hotkey"],
                        stage,
                        position,
                        submission["miner_hotkey"],
                        f"arena/{ROUND}/outputs/{assignment}:1.json",
                    ),
                )
        for position in range(20):
            stage = 1 if position < 10 else 2
            assignment = f"{ROUND}:{BASELINE}:{stage}:{position}"
            accepted = position == 0
            cursor.execute(
                """
                INSERT INTO public.lab_arena_runs(
                  run_id,assignment_id,round_id,submission_id,miner_hotkey,
                  stage,icp_position,attempt,kind,status,runner_hotkey,
                  terminal_cause,result_doc,output_ref,stage_generation)
                VALUES(%s,%s,%s,%s,%s,%s,%s,1,'execute',%s,%s,%s,%s::jsonb,%s,3)
                """,
                (
                    assignment + ":1",
                    assignment,
                    ROUND,
                    BASELINE,
                    baseline_hotkey,
                    stage,
                    position,
                    "accepted" if accepted else "failed",
                    baseline_hotkey,
                    "accepted" if accepted else "budget_exhausted",
                    json.dumps({"terminal_status": "accepted" if accepted else "budget_exhausted"}),
                    f"arena/{ROUND}/outputs/{assignment}:1.json" if accepted else None,
                ),
            )
        score_assignment = f"{ROUND}:{BASELINE}:1:0:score"
        scored_run = f"{ROUND}:{BASELINE}:1:0:1"
        cursor.execute(
            """
            INSERT INTO public.lab_arena_runs(
              run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
              icp_position,attempt,kind,status,runner_hotkey,terminal_cause,
              result_doc,output_ref,scored_run_id,stage_generation)
            VALUES(%s,%s,%s,%s,%s,1,0,1,'score','accepted',%s,'accepted',
                   '{"terminal_status":"accepted"}'::jsonb,%s,%s,3)
            """,
            (
                score_assignment + ":1",
                score_assignment,
                ROUND,
                BASELINE,
                baseline_hotkey,
                baseline_hotkey,
                f"arena/{ROUND}/scores/{score_assignment}:1.json",
                scored_run,
            ),
        )
        for index in range(14):
            cursor.execute(
                """
                INSERT INTO public.lab_arena_ledger(
                  entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
                  call_identity,provider,operation_id,funding_source,
                  amount_microusd,entry_doc,terminal_response)
                VALUES('settlement',%s,%s,%s,%s,1,%s,'openrouter',
                  'openrouter.responses','host',1000,'{}'::jsonb,
                  '{"status":200,"call_succeeded":true}'::jsonb)
                """,
                (
                    frozen[0]["miner_hotkey"],
                    ROUND,
                    frozen[0]["submission_id"],
                    f"{ROUND}:{frozen[0]['submission_id']}:1:0:1",
                    "sha256:" + hashlib.sha256(f"miner-{index}".encode()).hexdigest(),
                ),
            )
        for label, run_id, submission_id in (
            ("baseline", scored_run, BASELINE),
            ("score", score_assignment + ":1", BASELINE),
        ):
            cursor.execute(
                """
                INSERT INTO public.lab_arena_ledger(
                  entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
                  call_identity,provider,operation_id,funding_source,
                  amount_microusd,entry_doc,terminal_response)
                VALUES('settlement',%s,%s,%s,%s,1,%s,'openrouter',
                  'openrouter.responses','host',60000,'{}'::jsonb,
                  '{"status":200,"call_succeeded":true}'::jsonb)
                """,
                (
                    baseline_hotkey,
                    ROUND,
                    submission_id,
                    run_id,
                    "sha256:" + hashlib.sha256(label.encode()).hexdigest(),
                ),
            )
        judgment_key = "sha256:" + hashlib.sha256(b"archived-judgment").hexdigest()
        company_hash = "sha256:" + hashlib.sha256(b"archived-company").hexdigest()
        evidence = {
            "cache_key": judgment_key,
            "company_input_hash": company_hash,
            "authority_slot": 0,
            "source_score_run_id": score_assignment + ":1",
            "source_scored_run_id": scored_run,
            "source_runner_hotkey": baseline_hotkey,
            "runner_authority_exclusions": [baseline_hotkey],
            "raw_judgment": {},
        }
        cursor.execute(
            """
            INSERT INTO public.lab_arena_company_judgments(
              cache_key,authority_slot,scope_doc,company_input_hash,
              evidence_hash,evidence_doc,source_score_run_id,
              source_scored_run_id,source_runner_hotkey)
            VALUES(%s,0,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s)
            """,
            (
                judgment_key,
                json.dumps({"cache_key": judgment_key, "company_input_hash": company_hash}),
                company_hash,
                "sha256:" + hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
                json.dumps(evidence),
                score_assignment + ":1",
                scored_run,
                baseline_hotkey,
            ),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _scalar(cursor, query: str) -> str:
    cursor.execute(query)
    return str(cursor.fetchone()[0])


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
        "__RERUN_SCHEDULE_JSON__": json.dumps(schedule, sort_keys=True, separators=(",", ":")),
        "__TERMINAL_STATUS__": status,
        "__TERMINAL_CANCEL_REASON_SQL__": "NULL" if cancel_reason is None else "'" + cancel_reason + "'",
        "__SCORING_DEFINITION_SHA256__": hashlib.sha256(scoring_definition.encode()).hexdigest(),
        "__PATCHED_SCORING_DEFINITION_SHA256__": hashlib.sha256(patched_scoring_definition.encode()).hexdigest(),
        "__TERMINAL_ROUND_SHA256__": _scalar(cursor, "SELECT encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex') FROM public.lab_arena_rounds r WHERE round_id='arena-2026-09-20'"),
        "__TERMINAL_BASELINE_SHA256__": _scalar(cursor, "SELECT encode(extensions.digest(to_jsonb(s)::text,'sha256'),'hex') FROM public.lab_arena_submissions s WHERE submission_id='baseline-2026-09-20'"),
        "__TERMINAL_MINER_SUBMISSIONS_SHA256__": _scalar(cursor, "SELECT encode(extensions.digest(coalesce(string_agg(encode(extensions.digest(to_jsonb(s)::text,'sha256'),'hex'),'' ORDER BY submission_id),''),'sha256'),'hex') FROM public.lab_arena_submissions s WHERE round_id='arena-2026-09-20' AND submission_id<>'baseline-2026-09-20'"),
        "__TERMINAL_RUNS_SHA256__": _scalar(cursor, "SELECT encode(extensions.digest(coalesce(string_agg(encode(extensions.digest(to_jsonb(r)::text,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex') FROM public.lab_arena_runs r WHERE round_id='arena-2026-09-20'"),
        "__TERMINAL_LEDGER_SHA256__": _scalar(cursor, "SELECT encode(extensions.digest(coalesce(string_agg(encode(extensions.digest(to_jsonb(l)::text,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex') FROM public.lab_arena_ledger l WHERE round_id='arena-2026-09-20'"),
        "__TERMINAL_EXECUTE_RUN_COUNT__": _scalar(cursor, "SELECT count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-20' AND kind='execute'"),
        "__TERMINAL_SCORE_RUN_COUNT__": _scalar(cursor, "SELECT count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-20' AND kind='score'"),
        "__TERMINAL_BASELINE_RUN_COUNT__": _scalar(cursor, "SELECT count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-20' AND kind='execute' AND submission_id='baseline-2026-09-20'"),
        "__TERMINAL_BASELINE_ACCEPTED_RUN_COUNT__": _scalar(cursor, "SELECT count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-20' AND kind='execute' AND submission_id='baseline-2026-09-20' AND status='accepted'"),
        "__TERMINAL_BASELINE_FAILED_RUN_COUNT__": _scalar(cursor, "SELECT count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-20' AND kind='execute' AND submission_id='baseline-2026-09-20' AND status='failed'"),
        "__TERMINAL_BASELINE_EXECUTE_LEDGER_COUNT__": _scalar(cursor, "SELECT count(*) FROM public.lab_arena_ledger WHERE round_id='arena-2026-09-20' AND submission_id='baseline-2026-09-20' AND run_id IN(SELECT run_id FROM public.lab_arena_runs WHERE round_id='arena-2026-09-20' AND kind='execute')"),
        "__TERMINAL_SCORE_LEDGER_COUNT__": _scalar(cursor, "SELECT count(*) FROM public.lab_arena_ledger l WHERE round_id='arena-2026-09-20' AND EXISTS(SELECT 1 FROM public.lab_arena_runs r WHERE r.run_id=l.run_id AND r.kind='score')"),
    }
    body = TEMPLATE.read_text()
    for marker, value in values.items():
        body = body.replace(marker, value)
    assert re.search(r"__[A-Z0-9_]+__", body) is None
    return body, scoring_definition, patched_scoring_definition


def test_sep20_recovery_rejects_drift_then_publishes_fresh_scores(
    database, tmp_path, monkeypatch
):
    psycopg2, dsn = database
    schedule = _schedule()
    harness = lifecycle.Harness(
        lambda: psycopg2.connect(**dsn), tmp_path, challengers=[], runners=["alpha"]
    )
    harness.round_id = ROUND
    with psycopg2.connect(**dsn) as connection:
        _seed_terminal(connection, schedule, runner_hotkeys=harness.runner_keys)
        with connection.cursor() as cursor:
            rendered, scoring_before, scoring_after = _render(cursor, schedule)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb "
                "WHERE run_id=%s",
                (f"{ROUND}:{BASELINE}:1:0:1",),
            )
            cursor.execute("SET session_replication_role=origin")
        connection.commit()
        with pytest.raises(psycopg2.Error, match="terminal history differs"):
            with connection.cursor() as cursor:
                cursor.execute(rendered)
        connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET result_doc=%s::jsonb WHERE run_id=%s",
                (json.dumps({"terminal_status": "accepted"}), f"{ROUND}:{BASELINE}:1:0:1"),
            )
            cursor.execute("SET session_replication_role=origin")
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(rendered)
            cursor.execute(rendered)
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
            )
            assert cursor.fetchone()[0] == scoring_after
            assert scoring_after.replace(
                _template_block("replacement"), _template_block("anchor")
            ) == scoring_before
            cursor.execute(
                """
                SELECT
                  (SELECT configuration_doc->'call_quotas' FROM public.lab_arena_rounds WHERE round_id=%s),
                  (SELECT count(DISTINCT assignment_id) FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s AND kind='execute'),
                  (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND submission_id<>%s AND kind='execute' AND status='accepted'),
                  (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND submission_id=%s AND kind='execute'),
                  (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'),
                  (SELECT count(*) FROM public.lab_arena_submissions WHERE round_id=%s),
                  (SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=%s),
                  (SELECT count(*) FROM public.lab_arena_company_judgments
                    WHERE source_score_run_id=%s),
                  (SELECT source_ref FROM public.lab_arena_submissions WHERE submission_id=%s)
                """,
                (
                    ROUND, ROUND, BASELINE, ROUND, BASELINE, ARCHIVE,
                    BASELINE + ":r326archive", ARCHIVE, ARCHIVE, ARCHIVE,
                    f"{ROUND}:{BASELINE}:1:0:score:1", BASELINE,
                ),
            )
            row = cursor.fetchone()
        assert row == (
            {"deepline": 200, "openrouter": 500, "scrapingdog": 200},
            20,
            80,
            20,
            1,
            8,
            2,
            1,
            "arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun326-c26122a7.tar.gz",
        )

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT king_outcome,king_hotkey,king_start_epoch,"
                "effective_reward_epoch,reward_basis_hash,reward_basis_doc,"
                "signing_key_doc,reward_activated_at,promotion_required,"
                "promotion_doc,baseline_promoted_at FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone() == (
                None, None, None, None, None, None, None, None, True, None, None
            )
            cursor.execute(
                "SELECT configuration_doc->>'archived_king_outcome',"
                "configuration_doc->>'archived_reward_basis_hash',"
                "configuration_doc->'archived_promotion_doc' "
                "FROM public.lab_arena_rounds WHERE round_id=%s",
                (ARCHIVE,),
            )
            assert cursor.fetchone() == (
                "crowned",
                "sha256:" + "c" * 64,
                {"fixture": "stale-promotion"},
            )
            cursor.execute(
                "SELECT run_id,output_ref,icp_position FROM public.lab_arena_runs "
                "WHERE round_id=%s AND kind='execute' AND submission_id<>%s",
                (ROUND, BASELINE),
            )
            retained = cursor.fetchall()

        icps = daily_icps()
        benchmark = {
            "schema_version": "leadpoet.lab_arena.benchmark.v1",
            "round_id": ROUND,
            "icps": icps,
        }
        harness.objects.put(
            f"arena/{ROUND}/benchmark.json", json.dumps(benchmark).encode()
        )
        for index, (run_id, output_ref, position) in enumerate(retained, 1):
            rerun310._proof_execution_v2(
                harness.objects, icps[position], index, position, run_id
            )
            harness.objects.put(
                output_ref, harness.objects.get(f"arena/output/{run_id}.json")
            )

        monkeypatch.setattr(lifecycle, "ROUND", ROUND)
        monkeypatch.setattr(lifecycle, "BASELINE", BASELINE)
        monkeypatch.setattr(lifecycle, "_proof_execution", rerun310._proof_execution_v2)
        lifecycle._drive_cycle(
            harness.service, harness.objects, icps, harness.runner_keys[0]
        )
        publication = harness.service.publish(ROUND)
        assert publication["status"] == "ok"

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status,publication_doc->'king_decision'->>'outcome',"
                "jsonb_array_length(publication_doc->'final_ranking'),"
                "king_outcome,reward_basis_hash,reward_basis_doc,signing_key_doc,"
                "reward_activated_at,promotion_required,promotion_doc,"
                "baseline_promoted_at FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            published = cursor.fetchone()
            cursor.execute(
                "SELECT array_agg(per_icp_score ORDER BY icp_position),"
                "jsonb_agg(qualification_doc ORDER BY icp_position) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' "
                "AND submission_id=%s AND status='accepted'",
                (ROUND, BASELINE),
            )
            baseline_scores = cursor.fetchone()
            assert published[:4] == ("published", "no_king", 5, "no_king"), (
                published,
                baseline_scores,
            )
            assert published[4:] == (None, None, None, None, True, None, None)
            cursor.execute(
                "SELECT count(*),min(per_icp_score),"
                "bool_and(qualification_doc IS NOT NULL) "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' "
                "AND submission_id=%s AND status='accepted'",
                (ROUND, BASELINE),
            )
            count, minimum_score, persisted = cursor.fetchone()
            assert count == 20 and minimum_score > 0 and persisted is True
            cursor.execute(
                "SELECT count(*),bool_and(status='accepted'),"
                "bool_and(assignment_id LIKE '%%:score:rerun326') "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score'",
                (ROUND,),
            )
            assert cursor.fetchone() == (100, True, True)
            cursor.execute(
                "SELECT (entry->>'final_score')::double precision "
                "FROM public.lab_arena_rounds r,"
                "jsonb_array_elements(r.publication_doc->'final_ranking') entry "
                "WHERE r.round_id=%s AND entry->>'submission_id'=%s",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone()[0] > 0
            cursor.execute(
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
                (ROUND,),
            )
            published_before = cursor.fetchone()[0]
            cursor.execute(rendered)
            cursor.execute(
                "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
                (ROUND,),
            )
            assert cursor.fetchone()[0] == published_before


def test_sep20_recovery_template_is_inactive_and_narrow():
    body = TEMPLATE.read_text()
    assert TEMPLATE.name not in CURRENT_SERVICE_MIGRATIONS
    rendered_path = TEMPLATE.with_suffix("")
    if rendered_path.exists():
        rendered = rendered_path.read_text()
        assert re.search(r"__[A-Z0-9_]+__", rendered) is None
        assert rendered.count("BEGIN;") == 1 and rendered.count("COMMIT;") == 1
        assert "score:rerun326" in rendered
        assert "c26122a7287c2e9366c2b6a897b8d56a8fab41b1" in rendered
        assert "61e60812b4ca4506c7c1ea7ee7a7678750e50ad5adec17ca500f44708f71b8b4" in rendered
    assert "arena-2026-09-19" not in body
    assert "DELETE FROM" not in body and "TRUNCATE " not in body
    assert "openrouter\":500" in body
    assert "execution_icp_cap_microusd" in body and "4000000" in body
    assert "cost_per_company_microusd" in body and "800000" in body
    assert "icp_wall_clock_seconds" in body and "2700" in body
    assert "lease_ttl_seconds" in body and "3600" in body
    assert "archived_execution_judgments" in body
    for field in (
        "archived_reward_basis_hash",
        "archived_reward_basis_doc",
        "archived_signing_key_doc",
        "archived_effective_reward_epoch",
        "archived_reward_activated_at",
        "archived_king_outcome",
        "archived_king_hotkey",
        "archived_king_start_epoch",
        "archived_promotion_required",
        "archived_promotion_doc",
        "archived_baseline_promoted_at",
    ):
        assert field in body
    assert "moved_baseline_ledger<>__TERMINAL_BASELINE_EXECUTE_LEDGER_COUNT__" in body
    assert "score:rerun326" in body
    assert "c26122a7287c2e9366c2b6a897b8d56a8fab41b1" in body
    assert "61e60812b4ca4506c7c1ea7ee7a7678750e50ad5adec17ca500f44708f71b8b4" in body
    assert len(set(re.findall(r"__[A-Z0-9_]+__", body))) == 17
