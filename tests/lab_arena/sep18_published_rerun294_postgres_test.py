"""Hermetic PostgreSQL proof for the Sep18 published baseline rerun294."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab_arena import (
    contact_policy,
    contracts,
    intent_details_policy,
    judgment_cache,
    output,
    scoring,
    verify,
)
from lab_arena.store import hash_lease_token, new_lease_token
from qualification.scoring.arena_integrity import canonical_company_identity
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.per_icp_cost_admission_postgres_test import (
    _reserve as _reserve_cost_call,
    _settle as _settle_cost_call,
)
from tests.lab_arena.sep18_open_quota287_postgres_test import _configuration
from tests.lab_arena.test_lab_arena_service_round import Harness

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "scripts/294-arena-2026-09-18-published-baseline-rerun.sql.template"
ROUND = "arena-2026-09-18"
BASELINE = "baseline-2026-09-18"
ARCHIVE = "arena-2026-09-18-rerun291archive"
ARCHIVE_BASELINE = "baseline-2026-09-18-rerun291archive"
BANK_SHA = "6999fdd7bcc09f95943f29127471659d966a86bdb370863f4d895376863acf91"
OLD_SOURCE_SIZE = 604_847
OLD_SOURCE_SHA = "7e1bb0747014a57bc50f48f9f822d1a7c936682d63f06564e978d23f54eb7fc1"
OLD_SOURCE_COMMIT = "e5341f85829ad196b4a1cb58b38a34155697c8d4"
NEW_SOURCE_SIZE = 610_292
NEW_SOURCE_SHA = "d" * 64
NEW_SOURCE_COMMIT = "3" * 40
HISTORICAL_CALL_IDENTITY = "sha256:" + hashlib.sha256(
    b"rerun294-unrelated-historical-sentinel"
).hexdigest()


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS
        + (
            "289-lab-arena-per-icp-cost-policy.sql",
            "292-lab-arena-null-final-score-publication.sql",
        )
    )


def _compact(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _proof_company(icp, index, position):
    name = "Proof %d %d" % (index, position)
    domain = "proof-%d-%d.example.com" % (index, position)
    return {
        "company_name": name,
        "company_website": "https://" + domain,
        "company_linkedin": "https://www.linkedin.com/company/proof-%d-%d"
        % (index, position),
        "industry": icp["industry"],
        "employee_count": icp["employee_count"][0],
        "company_stage": str(icp.get("company_stage") or ""),
        "country": icp.get("country") or "United States",
        "state": "California",
        "intent_details": (
            "%s announced a relevant operating milestone in August 2026. "
            "That activity indicates current demand aligned with this ICP."
        ) % name,
        "intent_signals": [{
            "description": "Announced a relevant operating milestone",
            "url": "https://" + domain + "/news",
            "date": "2026-08-01",
            "matched_icp_signal": 0,
        }],
        "contact": {
            "full_name": "Alex Proof",
            "job_title": "VP Engineering",
            "email": "alex@" + domain,
        },
    }


def _proof_breakdown(company, score):
    identity = canonical_company_identity(company).key
    qualified = float(score) > 0
    checks = {key: {"status": "pass"} for key in (
        "claim", "identity", "source", "company", "role", "location",
        "email_attribution", "email_verification",
    )}
    return {
        "final_score": float(score),
        "company_index": 0,
        "company_identity_key": identity,
        "company_identity_alias_keys": [identity],
        "company_qualified": qualified,
        "duplicate_company": False,
        "contact_identity_key": "contact:" + identity,
        "contact_qualified": qualified,
        "email_status": "valid" if qualified else "invalid",
        "contact_verification": {
            "decision": "verified" if qualified else "mismatch",
            "subchecks": checks,
        },
        "verifier_gate_receipts": [
            {"gate": "company_fit", "decision": "match"},
            {"gate": "intent_details", "decision": "match"},
        ],
        "intent_signals_detail": [{
            "matched_icp_signal": 0,
            "after_decay": 50.0,
            "judge_verdict": {
                "decision": "verified",
                "verification_trace": {
                    "intent_verdict": {
                        "signal_evaluations": [{"signal_status": "supported"}]
                    }
                },
            },
        }],
        "failure_reason": "",
    }


def _proof_execution(objects, icp, index, position, execute_id, score_id=None):
    company = _proof_company(icp, index, position)
    document = output.output_document_from_bytes(
        json.dumps({
            "schema_version": intent_details_policy.OUTPUT_SCHEMA,
            "companies": [company],
        }).encode(),
        expected_schema_version=intent_details_policy.OUTPUT_SCHEMA,
    )
    objects.put("arena/output/%s.json" % execute_id, json.dumps(document).encode())
    breakdown = _proof_breakdown(company, 40 if index == 0 else 0)
    if score_id is not None:
        objects.put(
            "arena/score/%s.json" % score_id,
            json.dumps(scoring.build_scoring_output(execute_id, [breakdown])).encode(),
        )
    row = verify.scored_row(
        "proof", position, execute_id, icp, document["companies"], [breakdown],
        scoring.build_scorer_policy(
            scoring_adapter_version=contact_policy.SCORING_ADAPTER,
            intent_details=True,
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


def _publication_definition(cursor) -> str:
    cursor.execute(
        "SELECT pg_catalog.pg_get_functiondef("
        "'public.lab_arena__per_icp_publication_valid(text,jsonb)'"
        "::pg_catalog.regprocedure)"
    )
    return cursor.fetchone()[0]


def _hash(cursor, table: str, predicate: str, order: str, *, strip_identity=False):
    expression = "to_jsonb(row_value)"
    if strip_identity:
        expression += "-'round_id'-'submission_id'"
    cursor.execute(
        "SELECT 'sha256:'||encode(extensions.digest(coalesce(string_agg("
        "encode(extensions.digest((%s)::text,'sha256'),'hex'),'' ORDER BY %s),''),"
        "'sha256'),'hex') FROM public.%s AS row_value WHERE %s"
        % (expression, order, table, predicate)
    )
    return cursor.fetchone()[0]


def _install_sep17_score_namespace(cursor) -> str:
    cursor.execute(
        "SELECT pg_get_functiondef("
        "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)"
    )
    definition = cursor.fetchone()[0]
    old = """    v_assignment := p_round_id || ':' || v_scored.submission_id || ':' ||
      p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score';"""
    current = """    v_assignment := p_round_id || ':' || v_scored.submission_id || ':' ||
      p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score' ||
      CASE WHEN p_round_id = 'arena-2026-09-17'
                  AND v_scored.submission_id = 'baseline-2026-09-17'
                  AND EXISTS (
                    SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-17-rerun285archive'
                  )
           THEN ':rerun286' ELSE '' END;"""
    assert definition.count(old) == 1
    cursor.execute(definition.replace(old, current))
    cursor.execute(
        "SELECT encode(extensions.digest(pg_get_functiondef("
        "'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure),"
        "'sha256'),'hex')"
    )
    return cursor.fetchone()[0]


def _future_schedule(old):
    return {
        **old,
        "benchmark_deadline": "2099-01-01T00:00:00Z",
        "stage_1_start": "2099-01-01T00:00:01Z",
        "stage_1_close": "2099-01-01T01:00:00Z",
        "stage_1_scoring_close": "2099-01-01T02:00:00Z",
        "stage_2_start": "2099-01-01T02:00:01Z",
        "stage_2_close": "2099-01-01T03:00:00Z",
        "final_scoring_close": "2099-01-01T04:00:00Z",
        "publication_deadline": "2099-01-01T04:30:00Z",
    }


def _seed_published_terminal(connection, harness: Harness):
    config = _configuration(ROUND)
    config.update(
        schedule={**config["schedule"], "benchmark_deadline": "2026-09-18T18:30:00Z"},
        sourcing_cost_eligibility_policy="successful_calls_per_icp_v1",
        execution_icp_cap_microusd=4_000_000,
        call_quotas={"openrouter": 200, "deepline": 30, "scrapingdog": 30},
        runner_hotkeys=harness.runner_keys,
        baseline_hotkey=harness.baseline_hotkey,
        runner_slot_ceiling=20,
        scorer_image_digest="sha256:" + "e" * 64,
        scorer_image_reference="registry.example/scorer@sha256:" + "e" * 64,
    )
    participants = [{
        "submission_id": BASELINE if index == 0 else f"sep18-miner-{index}",
        "miner_hotkey": harness.baseline_hotkey if index == 0 else "5" + chr(65 + index) * 47,
        "is_king": index == 0,
        "source_ref": (
            f"arena/{ROUND}/sources/{BASELINE}.tar.gz" if index == 0
            else f"arena/{ROUND}/sources/sep18-miner-{index}.tar.gz"
        ),
        "source_size_bytes": OLD_SOURCE_SIZE if index == 0 else 1000 + index,
    } for index in range(5)]
    final_ranking = [{
        "rank": index + 1,
        "submission_id": participant["submission_id"],
        "final_score": 1.0 if participant["is_king"] else 0.0,
        "is_baseline": participant["is_king"],
        "eligible": True,
        "eligibility_reason": "eligible",
        "cost_summary": {},
    } for index, participant in enumerate(participants)]
    publication = {
        "schema_version": "leadpoet.lab_arena.publication.v1",
        "round_id": ROUND,
        "participants": [{
            "submission_id": item["submission_id"],
            "miner_hotkey": item["miner_hotkey"],
            "is_baseline": item["is_king"],
        } for item in participants],
        "stage1_ranking": [],
        "finalists": [item["submission_id"] for item in participants[1:]],
        "final_ranking": final_ranking,
        "king_decision": {
            "outcome": "no_king", "king_submission_id": None,
            "king_hotkey": "", "winner_submission_id": None,
        },
        "published_at": "2026-09-18T22:00:00Z",
    }
    reward_basis = {
        "schema_version": "leadpoet.lab_arena.reward_basis.v1",
        "round_id": ROUND, "king_outcome": "defended",
        "king_hotkey": participants[1]["miner_hotkey"],
        "king_start_epoch": 8247000, "effective_reward_epoch": 8248000,
        "published_at": publication["published_at"],
    }
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds RESTART IDENTITY CASCADE"
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds("
            "round_id,status,status_generation,stage_generation,configuration_doc,"
            "rewards_enabled,participants,benchmark_ref,evaluation_date,icp_set_date,"
            "stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,"
            "king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,"
            "reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,"
            "published_at,promotion_required,champion_funding_frozen,"
            "champion_submission_id,champion_hotkey,champion_fallback_providers) VALUES ("
            "%s,'published',12,8,%s::jsonb,TRUE,%s::jsonb,%s,'2026-09-18','2026-09-17',"
            "'{}'::jsonb,'{}'::jsonb,%s::jsonb,%s::jsonb,'no_king',NULL,8247000,"
            "8248000,%s,%s::jsonb,%s::jsonb,'2026-09-18T22:01:00Z',"
            "'2026-09-18T22:00:00Z',TRUE,TRUE,%s,%s,ARRAY['openrouter']::text[])",
            (
                ROUND, json.dumps(config), json.dumps(participants),
                f"arena/{ROUND}/benchmark.json",
                json.dumps(publication["finalists"]), json.dumps(publication),
                "sha256:" + "a" * 64, json.dumps(reward_basis),
                json.dumps({"public_key_hash": "sha256:" + "b" * 64}),
                participants[1]["submission_id"], participants[1]["miner_hotkey"],
            ),
        )
        for index, participant in enumerate(participants):
            source_doc = {
                "source_ref": participant["source_ref"],
                "source_size_bytes": participant["source_size_bytes"],
                "consent": {"public_rerun": True},
            }
            if participant["is_king"]:
                source_doc.update(source_sha256=OLD_SOURCE_SHA, source_commit=OLD_SOURCE_COMMIT)
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status,is_king,submission_doc,"
                "source_ref,source_size_bytes,consent,frozen_at,code_review_status,"
                "code_review_attempts,code_review_doc,code_review_claim,"
                "code_review_started_at) VALUES ("
                "%s,%s,%s,'frozen',%s,%s::jsonb,%s,%s,'{\"public_rerun\":true}'::jsonb,"
                "'2026-09-18T00:10:00Z',%s,%s,%s::jsonb,%s,%s)",
                (
                    participant["submission_id"], ROUND, participant["miner_hotkey"],
                    participant["is_king"], json.dumps(source_doc),
                    participant["source_ref"], participant["source_size_bytes"],
                    "pending" if participant["is_king"] else "passed",
                    0 if participant["is_king"] else 1,
                    None if participant["is_king"] else json.dumps({"decision": "passed"}),
                    None if participant["is_king"] else "sha256:" + "%064x" % (index + 9000),
                    None if participant["is_king"] else "2026-09-18T00:09:00Z",
                ),
            )
            for position in range(20):
                stage = 1 if position < 10 else 2
                assignment = f"{ROUND}:{participant['submission_id']}:{stage}:{position}:terminal"
                run_id = assignment + ":1"
                terminal_score, terminal_qualification = _proof_execution(
                    harness.objects, daily_icps()[position], index, position, run_id
                )
                score_assignment = assignment + ":score:terminal"
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs("
                    "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                    "icp_position,attempt,kind,status,stage_generation,terminal_cause,"
                    "output_ref,per_icp_score,qualification_doc) VALUES ("
                    "%s,%s,%s,%s,%s,%s,%s,1,'execute','accepted',8,'accepted',%s,%s,%s::jsonb)",
                    (
                        run_id, assignment, ROUND, participant["submission_id"],
                        participant["miner_hotkey"], stage, position,
                        f"arena/output/{run_id}.json",
                        terminal_score if participant["is_king"] else None,
                        json.dumps(terminal_qualification) if participant["is_king"] else None,
                    ),
                )
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs("
                    "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                    "icp_position,attempt,kind,status,stage_generation,terminal_cause,"
                    "output_ref,scored_run_id) VALUES ("
                    "%s,%s,%s,%s,%s,%s,%s,1,'score','accepted',8,'accepted',%s,%s)",
                    (
                        score_assignment + ":1", score_assignment, ROUND,
                        participant["submission_id"], participant["miner_hotkey"],
                        stage, position, f"arena/terminal/{run_id}-score.json", run_id,
                    ),
                )
        # This direct insert represents sealed historical terminal evidence; it
        # is not the active-cycle cost-path proof. IDs, amounts, and payloads
        # must survive the archive; routing IDs are the only allowed rewrite.
        first_run = f"{ROUND}:{BASELINE}:1:0:terminal:1"
        cursor.execute(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,stage_generation,terminal_cause) VALUES ("
            "%s,%s,%s,%s,%s,1,0,2,'execute','failed',8,'provider_error')",
            (
                f"{ROUND}:{BASELINE}:1:0:terminal:2",
                f"{ROUND}:{BASELINE}:1:0:terminal", ROUND, BASELINE,
                harness.baseline_hotkey,
            ),
        )
        for suffix, run_id, amount in (
            ("old-a", first_run, 700_000),
            ("old-b", f"{ROUND}:{BASELINE}:1:0:terminal:2", 800_000),
        ):
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd,"
                "entry_doc,terminal_response) VALUES ('settlement',%s,%s,%s,%s,1,%s,"
                "'openrouter','openrouter.responses','host',%s,%s::jsonb,%s::jsonb)",
                (
                    harness.baseline_hotkey, ROUND, BASELINE, run_id,
                    "sha256:" + hashlib.sha256(suffix.encode()).hexdigest(), amount,
                    json.dumps({"attempt_evidence": suffix, "paid": True}),
                    json.dumps({"call_succeeded": True}),
                ),
            )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger("
            "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,amount_microusd,"
            "entry_doc,terminal_response) VALUES ('settlement',%s,"
            "'arena-2026-08-01','historical-baseline','historical-run',1,%s,"
            "'openrouter','openrouter.responses','host',12345,%s::jsonb,%s::jsonb)",
            (
                harness.baseline_hotkey, HISTORICAL_CALL_IDENTITY,
                json.dumps({"sentinel": "unrelated-history"}),
                json.dumps({"call_succeeded": True}),
            ),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()
    return participants, publication


def _terminal_seals(cursor):
    cursor.execute("SELECT to_jsonb(row_value) FROM public.lab_arena_rounds AS row_value WHERE round_id=%s", (ROUND,))
    round_doc = cursor.fetchone()[0]
    cursor.execute(
        "SELECT to_jsonb(row_value) FROM public.lab_arena_submissions AS row_value "
        "WHERE round_id=%s AND submission_id=%s", (ROUND, BASELINE),
    )
    baseline = cursor.fetchone()[0]
    predicates = {
        "BASELINE_RUN": "round_id='%s' AND submission_id='%s'" % (ROUND, BASELINE),
        "BASELINE_LEDGER": "round_id='%s' AND submission_id='%s'" % (ROUND, BASELINE),
        "NONBASELINE_SUBMISSION": "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE),
        "NONBASELINE_RUN": "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE),
        "NONBASELINE_LEDGER": "round_id='%s' AND submission_id<>'%s'" % (ROUND, BASELINE),
    }
    counts = {}
    for key, table in (
        ("BASELINE_RUN", "lab_arena_runs"),
        ("BASELINE_LEDGER", "lab_arena_ledger"),
        ("NONBASELINE_SUBMISSION", "lab_arena_submissions"),
        ("NONBASELINE_RUN", "lab_arena_runs"),
        ("NONBASELINE_LEDGER", "lab_arena_ledger"),
    ):
        cursor.execute("SELECT count(*) FROM public.%s WHERE %s" % (table, predicates[key]))
        counts[key] = cursor.fetchone()[0]
    hashes = {
        "BASELINE_RUNS": _hash(cursor, "lab_arena_runs", predicates["BASELINE_RUN"], "run_id", strip_identity=True),
        "BASELINE_LEDGER": _hash(cursor, "lab_arena_ledger", predicates["BASELINE_LEDGER"], "entry_id", strip_identity=True),
        "NONBASELINE_SUBMISSIONS": _hash(cursor, "lab_arena_submissions", predicates["NONBASELINE_SUBMISSION"], "submission_id"),
        "NONBASELINE_RUNS": _hash(cursor, "lab_arena_runs", predicates["NONBASELINE_RUN"], "run_id"),
        "NONBASELINE_LEDGER": _hash(cursor, "lab_arena_ledger", predicates["NONBASELINE_LEDGER"], "entry_id"),
    }
    return round_doc, baseline, counts, hashes


def _render(cursor, round_doc, baseline, counts, hashes, schedule):
    values = {
        "__FORWARD_SCHEDULE_JSON__": _compact(schedule),
        "__SCORING_DEFINITION_SHA256__": _install_sep17_score_namespace(cursor),
        "__NEW_SOURCE_SIZE_BYTES__": str(NEW_SOURCE_SIZE),
        "__NEW_SOURCE_SHA256__": NEW_SOURCE_SHA,
        "__NEW_SOURCE_COMMIT__": NEW_SOURCE_COMMIT,
        "__TERMINAL_ROUND_JSON__": _compact(round_doc),
        "__TERMINAL_BASELINE_JSON__": _compact(baseline),
        "__TERMINAL_SOURCE_SIZE_BYTES__": str(OLD_SOURCE_SIZE),
        "__TERMINAL_SOURCE_SHA256__": OLD_SOURCE_SHA,
        "__TERMINAL_SOURCE_COMMIT__": OLD_SOURCE_COMMIT,
        "__TERMINAL_BASELINE_RUN_COUNT__": str(counts["BASELINE_RUN"]),
        "__TERMINAL_BASELINE_RUNS_HASH__": hashes["BASELINE_RUNS"],
        "__TERMINAL_BASELINE_LEDGER_COUNT__": str(counts["BASELINE_LEDGER"]),
        "__TERMINAL_BASELINE_LEDGER_HASH__": hashes["BASELINE_LEDGER"],
        "__TERMINAL_NONBASELINE_SUBMISSION_COUNT__": str(counts["NONBASELINE_SUBMISSION"]),
        "__TERMINAL_NONBASELINE_SUBMISSIONS_HASH__": hashes["NONBASELINE_SUBMISSIONS"],
        "__TERMINAL_NONBASELINE_RUN_COUNT__": str(counts["NONBASELINE_RUN"]),
        "__TERMINAL_NONBASELINE_RUNS_HASH__": hashes["NONBASELINE_RUNS"],
        "__TERMINAL_NONBASELINE_LEDGER_COUNT__": str(counts["NONBASELINE_LEDGER"]),
        "__TERMINAL_NONBASELINE_LEDGER_HASH__": hashes["NONBASELINE_LEDGER"],
    }
    sql = TEMPLATE.read_text()
    for token, value in values.items():
        assert token in sql
        sql = sql.replace(token, value)
    assert re.search(r"__[A-Z0-9_]+__", sql) is None
    return sql


def _prepare(cursor, schedule):
    cursor.execute(
        "SELECT public.lab_arena_prepare_sep18_published_rerun294_v1("
        "%s,%s,%s,%s,%s::jsonb)",
        (NEW_SOURCE_SIZE, NEW_SOURCE_SHA, NEW_SOURCE_COMMIT, BANK_SHA, json.dumps(schedule)),
    )
    return cursor.fetchone()[0]


def _snapshot_historical_ledger(cursor):
    cursor.execute(
        "SELECT to_jsonb(row_value) FROM public.lab_arena_ledger AS row_value "
        "WHERE call_identity=%s", (HISTORICAL_CALL_IDENTITY,),
    )
    row = cursor.fetchone()
    assert row is not None
    return row[0]


def _drive_cycle(service, objects, icps, runner_hotkey):
    token = new_lease_token()
    request_id = contracts.new_request_id()
    first = service.store.claim_assignment(
        round_id=ROUND, runner_hotkey=runner_hotkey,
        declared_parallelism=10, slot_ceiling=20, excluded_miner_hotkeys=[],
        request_id=request_id,
        request_hash=contracts.document_hash({"request_id": request_id}),
        lease_token_hash=hash_lease_token(token),
    )
    assert first["status"] == "leased" and first["kind"] == "execute"
    assert (first["icp_position"], first["attempt"]) == (0, 1)
    first_identity, first_reserved = _reserve_cost_call(
        service.store, first, token, "rerun294-icp0-attempt1", 300_000
    )
    assert first_reserved["status"] == "reserved"
    _settle_cost_call(
        service.store, first, token, first_identity, 300_000, succeeded=True
    )
    assert service.store.complete_attempt(
        run_id=first["run_id"], lease_token_hash=hash_lease_token(token),
        result={"terminal_status": "provider_error"},
        terminal_cause="provider_error", output_ref="",
    )["confirmation_attempt"] == 2
    accepted_positions = set()
    retry_cost_settled = False
    while len(accepted_positions) < 20:
        token = new_lease_token()
        request_id = contracts.new_request_id()
        claim = service.store.claim_assignment(
            round_id=ROUND, runner_hotkey=runner_hotkey,
            declared_parallelism=10, slot_ceiling=20, excluded_miner_hotkeys=[],
            request_id=request_id,
            request_hash=contracts.document_hash({"request_id": request_id}),
            lease_token_hash=hash_lease_token(token),
        )
        assert claim["status"] == "leased" and claim["kind"] == "execute"
        position = int(claim["icp_position"])
        if position == 0:
            assert claim["attempt"] == 2
            retry_identity, retry_reserved = _reserve_cost_call(
                service.store, claim, token, "rerun294-icp0-attempt2", 400_000
            )
            assert retry_reserved["status"] == "reserved"
            _settle_cost_call(
                service.store, claim, token, retry_identity, 400_000,
                succeeded=True,
            )
            retry_cost_settled = True
        _proof_execution(objects, icps[position], 9, position, claim["run_id"])
        assert service.store.complete_attempt(
            run_id=claim["run_id"], lease_token_hash=hash_lease_token(token),
            result={"terminal_status": "accepted"}, terminal_cause="accepted",
            output_ref="arena/output/%s.json" % claim["run_id"],
        )["status"] == "accepted"
        accepted_positions.add(position)
    assert retry_cost_settled
    assert service.close_stage(ROUND, 1)["status"] == "ok"
    for stage in (1, 2):
        assert service.open_scoring(ROUND, stage)["assignments"] == 50
        while True:
            pending = [
                row for row in service.store.list_runs(ROUND, stage=stage, kind="score")
                if row["status"] != "accepted"
            ]
            if not pending:
                break
            token = new_lease_token()
            request_id = contracts.new_request_id()
            run = service.store.claim_assignment(
                round_id=ROUND, runner_hotkey=runner_hotkey,
                declared_parallelism=1, slot_ceiling=20,
                excluded_miner_hotkeys=[runner_hotkey], request_id=request_id,
                request_hash=contracts.document_hash({"request_id": request_id}),
                lease_token_hash=hash_lease_token(token),
            )
            assert run["status"] == "leased" and run["kind"] == "score"
            position = int(run["icp_position"])
            scored_run = service.store.get_run(run["scored_run_id"])
            executed = json.loads(objects.get(scored_run["output_ref"]).decode())
            document = scoring.build_scoring_output(
                run["scored_run_id"],
                [_proof_breakdown(
                    executed["companies"][0],
                    40 if run["submission_id"] == BASELINE else 0,
                )],
            )
            ref = "arena/score/%s.json" % run["run_id"]
            objects.put(ref, json.dumps(document).encode())
            stored = service.store.get_run(run["run_id"])
            evidence = judgment_cache.build_evidence_snapshot(
                output=document, cache_scope=stored["judgment_scope_doc"],
                source_score_run_id=run["run_id"],
                source_scored_run_id=run["scored_run_id"],
                source_output_ref=ref, source_runner_hotkey=runner_hotkey,
                runner_authority_exclusions=run["runner_authority_exclusions"],
            )
            assert service.store.complete_attempt(
                run_id=run["run_id"], lease_token_hash=hash_lease_token(token),
                result={"terminal_status": "accepted"}, terminal_cause="accepted",
                output_ref=ref, judgment_evidence=evidence,
                judgment_evidence_hash=contracts.document_hash(evidence),
            )["status"] == "accepted"
        assert service.close_scoring(ROUND, stage)["status"] == "closed"
        scored = service.score_stage(ROUND, stage)
        assert scored["status"] == "ok", (
            scored, service.store.get_round(ROUND).get("cancel_reason")
        )
        if stage == 1:
            assert service.open_stage(ROUND, 2)["status"] == "ok"
            assert service.close_stage(ROUND, 2)["status"] == "ok"


def test_rerun294_full_published_transition_and_fail_closed_winner_guard(
    database, tmp_path,
):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        harness = Harness(lambda: psycopg2.connect(**dsn), tmp_path,
                          challengers=[], runners=["alpha"])
        harness.round_id = ROUND
        participants, terminal_publication = _seed_published_terminal(connection, harness)
        with connection.cursor() as cursor:
            round_doc, baseline, counts, hashes = _terminal_seals(cursor)
            schedule = _future_schedule(round_doc["configuration_doc"]["schedule"])
            cursor.execute(
                "SELECT coalesce(jsonb_agg(to_jsonb(row_value) ORDER BY entry_id),'[]'::jsonb) "
                "FROM public.lab_arena_ledger AS row_value WHERE round_id=%s AND submission_id=%s",
                (ROUND, BASELINE),
            )
            paid_before = cursor.fetchone()[0]
            historical_ledger_before = _snapshot_historical_ledger(cursor)
            authority_before = {
                key: round_doc.get(key) for key in (
                    "reward_basis_hash", "reward_basis_doc", "signing_key_doc",
                    "effective_reward_epoch", "reward_activated_at", "king_outcome",
                    "king_hotkey", "king_start_epoch", "promotion_required",
                    "promotion_doc", "baseline_promoted_at", "champion_funding_frozen",
                    "champion_submission_id", "champion_hotkey",
                    "champion_fallback_providers",
                )
            }
            publication_definition_before = _publication_definition(cursor)
            sql = _render(cursor, round_doc, baseline, counts, hashes, schedule)
            cursor.execute(sql)
            cursor.execute(sql)
            assert _publication_definition(cursor) == publication_definition_before
        connection.commit()

        with connection.cursor() as cursor:
            assert _prepare(cursor, schedule)["status"] == "prepared"
            assert _prepare(cursor, schedule)["status"] == "existing"
            cursor.execute(
                "SELECT count(*),min((state->>'settled_microusd')::bigint),"
                "max((state->>'reserved_or_uncertain_microusd')::bigint) FROM ("
                "SELECT public.lab_arena__successful_icp_cost_state(%s,%s,p) AS state "
                "FROM generate_series(0,19) p) costs",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 0, 0)
            cursor.execute(
                "SELECT coalesce(jsonb_agg(to_jsonb(row_value)-'round_id'-'submission_id' "
                "ORDER BY entry_id),'[]'::jsonb) FROM public.lab_arena_ledger AS row_value "
                "WHERE round_id=%s AND submission_id=%s", (ARCHIVE, ARCHIVE_BASELINE),
            )
            paid_after = cursor.fetchone()[0]
            assert paid_after == [
                {k: v for k, v in row.items() if k not in ("round_id", "submission_id")}
                for row in paid_before
            ]
            assert _snapshot_historical_ledger(cursor) == historical_ledger_before
        connection.commit()

        bank = {"schema_version": "leadpoet.lab_arena.benchmark.v1", "round_id": ROUND, "icps": daily_icps()}
        harness.objects.put(f"arena/{ROUND}/benchmark.json", json.dumps(bank).encode())
        harness.clock.now = datetime(2098, 12, 31, 23, 59, tzinfo=timezone.utc)
        _drive_cycle(harness.service, harness.objects, bank["icps"], harness.runner_keys[0])

        with connection.cursor() as cursor:
            # Two paid calls for separate attempts on the same ICP aggregate;
            # another ICP retains an independent fresh cap.
            cursor.execute(
                "SELECT run_id,attempt FROM public.lab_arena_runs WHERE round_id=%s "
                "AND submission_id=%s AND kind='execute' AND icp_position=0 "
                "ORDER BY attempt", (ROUND, BASELINE),
            )
            attempt_runs = cursor.fetchall()
            assert [row[1] for row in attempt_runs] == [1, 2]
            cursor.execute(
                "SELECT amount_microusd,terminal_response FROM public.lab_arena_ledger "
                "WHERE round_id=%s AND submission_id=%s AND entry_kind='settlement' "
                "ORDER BY amount_microusd", (ROUND, BASELINE),
            )
            active_settlements = cursor.fetchall()
            assert [row[0] for row in active_settlements] == [300_000, 400_000]
            assert all(row[1] == {"status": 200, "call_succeeded": True}
                       for row in active_settlements)
            cursor.execute(
                "SELECT public.lab_arena__successful_icp_cost_state(%s,%s,0),"
                "public.lab_arena__successful_icp_cost_state(%s,%s,1)",
                (ROUND, BASELINE, ROUND, BASELINE),
            )
            icp0, icp1 = cursor.fetchone()
            assert icp0["successful_microusd"] == 700_000
            assert icp0["successful_calls"] == 2
            assert icp1["successful_microusd"] == 0
            cursor.execute(
                "SELECT count(*) FILTER (WHERE kind='execute' AND per_icp_score IS NOT NULL),"
                "count(*) FILTER (WHERE kind='score' AND status='accepted' "
                "AND assignment_id NOT LIKE '%%:score:terminal') "
                "FROM public.lab_arena_runs WHERE round_id=%s", (ROUND,),
            )
            assert cursor.fetchone() == (100, 100)
        connection.commit()

        # A changed daily winner is rejected even when scores and source are
        # otherwise allowed to change. This probes only the one-off guard.
        with connection.cursor() as cursor:
            cursor.execute("CREATE TEMP TABLE round_guard_probe AS TABLE public.lab_arena_rounds WITH NO DATA")
            cursor.execute(
                "INSERT INTO round_guard_probe SELECT * FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            cursor.execute("UPDATE round_guard_probe SET status='scored' WHERE round_id=%s", (ROUND,))
            changed = {
                "outcome": "crowned",
                "king_submission_id": participants[1]["submission_id"],
                "king_hotkey": participants[1]["miner_hotkey"],
                "winner_submission_id": participants[1]["submission_id"],
            }
            cursor.execute(
                "UPDATE round_guard_probe SET publication_doc=jsonb_set("
                "publication_doc,'{king_decision}',%s::jsonb,true),king_outcome='crowned',"
                "king_hotkey=%s WHERE round_id=%s",
                (json.dumps(changed), participants[1]["miner_hotkey"], ROUND),
            )
            cursor.execute(
                "CREATE TRIGGER round_guard_probe BEFORE UPDATE ON round_guard_probe "
                "FOR EACH ROW EXECUTE FUNCTION "
                "public.lab_arena_sep18_published_rerun294_publication_guard_v1()"
            )
            with pytest.raises(Exception, match="sealed reward, winner"):
                cursor.execute("UPDATE round_guard_probe SET status='published' WHERE round_id=%s", (ROUND,))
        connection.rollback()

        assert harness.service.publish(ROUND)["status"] == "ok"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_jsonb(row_value),publication_doc FROM public.lab_arena_rounds AS row_value "
                "WHERE round_id=%s", (ROUND,),
            )
            final_round, publication = cursor.fetchone()
            authority_after = {key: final_round.get(key) for key in authority_before}
            assert authority_after == authority_before
            assert publication["king_decision"] == terminal_publication["king_decision"]
            assert len(publication["final_ranking"]) == 5
            assert all(row["eligible"] for row in publication["final_ranking"])
            old_score = next(
                row["final_score"] for row in terminal_publication["final_ranking"]
                if row["submission_id"] == BASELINE
            )
            new_score = next(
                row["final_score"] for row in publication["final_ranking"]
                if row["submission_id"] == BASELINE
            )
            assert new_score > old_score
            old_miners = {
                row["submission_id"]: row["final_score"]
                for row in terminal_publication["final_ranking"]
                if row["submission_id"] != BASELINE
            }
            new_miners = {
                row["submission_id"]: row["final_score"]
                for row in publication["final_ranking"]
                if row["submission_id"] != BASELINE
            }
            assert new_miners == old_miners
            assert final_round["configuration_doc"]["baseline_source_url"] == (
                "https://github.com/leadpoet/leadpoet-sales-agent/"
                "archive/refs/heads/lab.tar.gz"
            )
            assert _prepare(cursor, schedule)["status"] == "existing"
        connection.commit()
    finally:
        connection.close()
