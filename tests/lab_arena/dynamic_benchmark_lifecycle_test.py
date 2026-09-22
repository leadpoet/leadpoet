"""Dynamic benchmark rounds through the real service and disposable database."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pytest

from lab_arena import broker, contracts, runtime, scoring, shim, verify
from lab_arena.contact_evidence import source_key
from lab_arena.service import ServiceError
from qualification.scoring.competition import apply_company_judgment_context, _merge_contact_breakdown
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _verified_test_pool
from tests.lab_arena.test_lab_arena_service_round import Harness
from tests.lab_arena.per_icp_cost_round_test import PerIcpHarness
from tests.lab_arena.contact_round_test import _claim, _contact_result
from tests.test_arena_company_quality_policy import _positive_breakdown


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


@pytest.fixture
def connect(database):
    psycopg2, dsn = database
    return lambda: psycopg2.connect(**dsn)


def _bank(count: int) -> list[dict]:
    source = daily_icps()
    return [
        {
            **source[position % len(source)],
            "icp_id": f"icp_20260902_{position + 1:03d}",
            "prompt": f"Local fixture ICP {position + 1}",
        }
        for position in range(count)
    ]


def _harness(connect, tmp_path, *, count: int, margin: float, challengers: list[str], baseline_first: bool = False) -> Harness:
    harness = Harness(connect, tmp_path, challengers=challengers, runners=["alpha", "beta"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=count,
        promotion_margin=margin,
        runner_slot_ceiling=10,
        benchmark_disclosure_from="2026-01-01T00:00:00Z",
        execution_sequence_from="2026-01-01T00:00:00Z" if baseline_first else None,
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": int(kwargs["set_id"]), "icps": _bank(count),
    }
    if baseline_first:
        original_runner = harness.runner

        def verified_runner(index, parallel=4):
            runner = original_runner(index, parallel)
            runner._config.proxy_worker_pool = _verified_test_pool(parallel)
            return runner

        harness.runner = verified_runner
    return harness


def _start(harness: Harness, *, label: str, epoch: int) -> tuple[dict, list[dict]]:
    harness.chain.epoch = epoch
    harness.clock.now = datetime.now(timezone.utc)
    configuration = harness.service.create_round(
        datetime.now(timezone.utc) + timedelta(minutes=30),
        round_id=f"arena-2099-01-01-{label}",
    )
    harness.round_id = configuration["round_id"]
    for flavor in harness.challengers:
        harness.submit(flavor, harness.round_id)
    with pytest.raises(ServiceError, match="benchmark_not_public"):
        harness.service.public_benchmark(harness.round_id)
    for submission in harness.service.store.list_submissions(harness.round_id):
        with pytest.raises(ServiceError, match="source_not_public"):
            harness.service.public_submission_code(submission["submission_id"])
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    participants = harness.service.store.get_round(harness.round_id)["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    return configuration, participants


@pytest.mark.parametrize("baseline_first", [False, True], ids=["two_stage", "baseline_first"])
def test_fresh_ten_icp_round_scores_both_participants_and_publishes(connect, tmp_path, baseline_first):
    harness = _harness(connect, tmp_path, count=10, margin=0.5, challengers=["Miner"], baseline_first=baseline_first)
    configuration, participants = _start(
        harness, label="d10base" if baseline_first else "d10split",
        epoch=36012 if baseline_first else 36010,
    )
    assert (configuration["stage_1_icp_count"], configuration["stage_2_icp_count"]) == (5, 5)
    assert configuration["promotion_margin"] == 0.5
    assert (configuration.get("execution_sequence_policy") == contracts.BASELINE_SCORED_FIRST_POLICY) is baseline_first
    assert len(participants) == 2
    assert sum(bool(participant["is_king"]) for participant in participants) == 1
    public_bank = harness.service.public_benchmark(harness.round_id)
    assert public_bank["benchmark_icp_count"] == public_bank["public_icp_count"] == 10
    assert public_bank["private_icp_count"] == 0
    assert {icp["icp_position"] for icp in public_bank["icps"]} == set(range(10))
    assert all(icp["baseline_score"] is None for icp in public_bank["icps"])
    for participant in participants:
        with pytest.raises(ServiceError, match="results_not_public"):
            harness.service.public_results(harness.round_id, participant["submission_id"])
        assert harness.service.public_submission_code(participant["submission_id"])["files"]

    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = harness.service.advance_round(harness.round_id)
    assert opened["assignments"] == (10 if baseline_first else 5 * len(participants))
    harness.advance_until("published", runners=2)
    published = harness.service.store.get_round(harness.round_id)
    assert published["status"] == "published"
    assert published["configuration_doc"] == configuration
    ranking = published["publication_doc"]["final_ranking"]
    hotkeys = {participant["submission_id"]: participant["miner_hotkey"] for participant in participants}
    def decision_entry(entry):
        return {
            "submission_id": entry["submission_id"], "hotkey": hotkeys[entry["submission_id"]],
            "final_score": entry["final_score"], "is_king": entry["is_baseline"],
        }
    baseline_entry = decision_entry(next(entry for entry in ranking if entry["is_baseline"]))
    challengers = [decision_entry(entry) for entry in ranking if not entry["is_baseline"] and entry["eligible"]]
    assert published["publication_doc"]["king_decision"] == verify.king_decision(
        challengers, baseline_entry, configuration
    )
    for participant in participants:
        submission_id = participant["submission_id"]
        runs = harness.service.store.list_runs(harness.round_id, submission_id=submission_id, kind="execute")
        assert {run["icp_position"] for run in runs} == set(range(10))
        result = harness.service.public_results(harness.round_id, submission_id)
        assert result["benchmark_icp_count"] == 10
        scores = result["scores"]["stage_1"] + result["scores"]["stage_2"]
        assert {row["icp_position"] for row in scores} == set(range(10))
        assert result["submission_scores"]["final"] is not None
        assert harness.service.public_submission_code(submission_id)["files"]

    baseline = {"submission_id": "baseline", "hotkey": "baseline", "is_king": True, "final_score": 70.0}
    miner = {"submission_id": "miner", "hotkey": "miner", "is_king": False, "final_score": 70.5}
    assert verify.king_decision([miner], baseline, configuration)["outcome"] == "crowned"
    assert verify.king_decision([{**miner, "final_score": 70.49}], baseline, configuration)["outcome"] == "no_king"


def test_ten_parallel_icps_form_one_execution_wave(connect, tmp_path):
    harness = _harness(connect, tmp_path, count=10, margin=0.5, challengers=[])
    harness.service.config.defaults = replace(
        harness.service.config.defaults, parallel_twenty_icp_execution=True
    )
    configuration, participants = _start(harness, label="parallel10", epoch=36011)
    assert configuration["parallel_twenty_icp_execution"] is True
    assert len(participants) == 1
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == 10
    store = harness.service.store
    leases = [
        claim(store, harness.round_id, harness.runner_keys[0], parallelism=10, ceiling=10)[0]
        for _ in range(10)
    ]
    assert all(lease["status"] == "leased" for lease in leases)
    assert {lease["icp_position"] for lease in leases} == set(range(10))
    assert {lease["submission_id"] for lease in leases} == {participants[0]["submission_id"]}
    blocked = claim(store, harness.round_id, harness.runner_keys[1], parallelism=10, ceiling=10)[0]
    assert blocked["status"] == "no_pending"


@pytest.mark.parametrize("count", [5, 10, 15, 20, 30])
def test_current_per_icp_cost_policy_publishes_each_frozen_count_and_excludes_judge_cost(connect, tmp_path, count):
    harness = PerIcpHarness(connect, tmp_path, challengers=["Miner"], runners=["alpha", "beta"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=count, promotion_margin=0.5,
        runner_slot_ceiling=min(count, contracts.RUNNER_SLOT_CEILING),
        execution_sequence_from="2026-01-01T00:00:00Z",
        benchmark_disclosure_from="2026-01-01T00:00:00Z",
    )
    original_source = harness.service.config.daily_icp_source

    def counted_icps(**kwargs):
        source = original_source(**kwargs)
        bank = source["icps"]
        icps = [
            {**bank[position % len(bank)],
             "icp_id": f"icp_20260902_{position + 1:03d}",
             "prompt": f"Local fixture ICP {position + 1}"}
            for position in range(count)
        ]
        return {**source, "icps": icps}

    harness.service.config.daily_icp_source = counted_icps
    original_runner = harness.runner

    def verified_runner(index, parallel=4):
        runner = original_runner(index, parallel)
        runner._config.proxy_worker_pool = _verified_test_pool(parallel)
        return runner

    harness.runner = verified_runner
    original_run_icp = harness.sandbox.run_icp

    class ChargedJudgeTransport(type(harness.provider)):
        def send(self, **kwargs):
            request = json.loads(kwargs.get("body") or b"{}")
            if request.get("model") in scoring.DEFAULT_JUDGE_MODELS.values() and request["messages"][0]["content"] == "charged judge":
                payload = {
                    "model": request["model"], "usage": {"cost": "0.01"},
                    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                }
                return broker.ProviderResponse(200, {"content-type": "application/json"}, json.dumps(payload).encode())
            return super().send(**kwargs)

    harness.provider = ChargedJudgeTransport()

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            with harness.sandbox.lock:
                os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
                try:
                    status, _, _ = shim.dispatch("openrouter.chat", {
                        "model": next(iter(scoring.DEFAULT_JUDGE_MODELS.values())),
                        "messages": [{"role": "user", "content": "charged judge"}],
                        "max_tokens": 100,
                    }, 5000)
                    assert status == 200
                finally:
                    os.environ.pop(shim.WORKER_SOCKET_ENV, None)

            def judge(companies, buyer, _reference):
                company = companies[0]
                name = company["company_name"]
                raw = _positive_breakdown(name, urlsplit(company["company_website"]).hostname,
                                          name.lower().replace(" ", "-"))
                raw.update(final_score=30.0 if name.startswith("PublicBaseline") else 60.0,
                           intent_signal_raw=30.0, intent_signal_final=30.0)
                evidence = document["contact_source_evidence"][source_key(company)]
                _merge_contact_breakdown(raw, asyncio.run(_contact_result(company, buyer, evidence)))
                return apply_company_judgment_context(companies, [raw], contacts_required=True)

            judge.company_quality = judge.integrity_policy = judge.contacts_required = True
            full, new = scoring.score_quality_work_item(
                {"scored_run_id": document["scored_run_id"]}, icp=document["icp"],
                companies=document["companies"], scorer=judge,
                cache_context=document["company_judgment_cache"],
            )
            output = scoring.build_scoring_output(document["scored_run_id"], full, company_judgments=new)
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps(output).encode())

        source_result = original_run_icp(spec, **kwargs)
        assert source_result.output_bytes
        companies = json.loads(source_result.output_bytes)["companies"]
        for company in companies:
            company.update(company_linkedin="https://linkedin.com/company/" + company["company_name"].lower().replace(" ", "-"), state="CA")
            company["contact"] = _claim(company, valid_role=True)
        return runtime.fake_result(exit_code=0, output_bytes=json.dumps({"companies": companies}).encode())

    harness.sandbox.run_icp = run_icp
    configuration, participants = _start(harness, label=f"cost{count}", epoch=36100 + count)
    assert configuration["sourcing_cost_eligibility_policy"] == contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
    assert configuration["execution_sequence_policy"] == contracts.BASELINE_SCORED_FIRST_POLICY
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == count
    harness.advance_until("published", runners=2)
    published = harness.service.store.get_round(harness.round_id)
    assert published["status"] == "published"
    assert published["king_outcome"] == "crowned"
    miner_id = next(participant["submission_id"] for participant in participants if not participant["is_king"])
    assert published["publication_doc"]["king_decision"]["winner_submission_id"] == miner_id
    for entry in published["publication_doc"]["final_ranking"]:
        summary = entry["cost_summary"]
        assert len(summary["per_icp"]) == count
        assert {item["icp_position"] for item in summary["per_icp"]} == set(range(count))
        assert summary["judge"]["settled_microusd"] > 0
        assert summary["competition_sourcing_microusd"] == 0
        result = harness.service.public_results(harness.round_id, entry["submission_id"])
        assert result["benchmark_icp_count"] == count
        assert len(result["scores"]["stage_1"] + result["scores"]["stage_2"]) == count
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)",
                (harness.round_id, json.dumps(entry)),
            )
            assert cursor.fetchone()[0] is True
        malformed = deepcopy(entry)
        malformed["cost_summary"]["per_icp"].pop()
        duplicate = deepcopy(entry)
        duplicate["cost_summary"]["per_icp"][-1] = deepcopy(
            duplicate["cost_summary"]["per_icp"][0]
        )
        for invalid in (malformed, duplicate):
            with connect() as connection, connection.cursor() as cursor:
                with pytest.raises(Exception, match="lab_arena_publication_cost_report_mismatch"):
                    cursor.execute(
                        "SELECT public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)",
                        (harness.round_id, json.dumps(invalid)),
                    )
    public_submissions = harness.service.public_submissions(harness.round_id)["submissions"]
    baseline_public = next(item for item in public_submissions if item["is_baseline"])
    assert baseline_public["stage1_score"] is not None
    assert baseline_public["stage1_score"] == pytest.approx(baseline_public["final_score"])
    assert all(len(item["cost_summary"]["per_icp"]) == count for item in public_submissions)
    assert len(participants) == 2


@pytest.mark.parametrize("count", [15, 30])
def test_future_counts_freeze_and_open_matching_assignments(connect, tmp_path, count):
    harness = _harness(connect, tmp_path, count=count, margin=0.5, challengers=["Miner"])
    configuration, participants = _start(harness, label=f"dynamic{count}", epoch=36000 + count)
    assert contracts.benchmark_icp_count(configuration) == count
    assert (configuration["stage_1_icp_count"], configuration["stage_2_icp_count"]) == (
        (count + 1) // 2, count // 2,
    )
    assert len(harness.service.benchmark_icps(harness.round_id)) == count
    assert len(harness.service.public_benchmark(harness.round_id)["icps"]) == count
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = harness.service.advance_round(harness.round_id)
    assert opened["assignments"] == ((count + 1) // 2) * len(participants)
    stage_one = harness.service.store.list_runs(harness.round_id, stage=1, kind="execute")
    for participant in participants:
        assert {run["icp_position"] for run in stage_one if run["submission_id"] == participant["submission_id"]} == set(range((count + 1) // 2))


def test_explicit_historical_twenty_and_one_point_rule_remain_valid(connect, tmp_path):
    harness = _harness(connect, tmp_path, count=20, margin=1.0, challengers=[])
    configuration, participants = _start(harness, label="historical20", epoch=36020)
    assert len(participants) == 1
    assert (configuration["stage_1_icp_count"], configuration["stage_2_icp_count"]) == (10, 10)
    assert configuration["promotion_margin"] == 1.0
    assert len(harness.service.public_benchmark(harness.round_id)["icps"]) == 20
    baseline = {"submission_id": "baseline", "hotkey": "baseline", "is_king": True, "final_score": 70.0}
    miner = {"submission_id": "miner", "hotkey": "miner", "is_king": False, "final_score": 70.5}
    assert verify.king_decision([miner], baseline, configuration)["outcome"] == "no_king"
