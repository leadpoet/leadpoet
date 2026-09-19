"""Per-ICP costs through the real runner, broker, database and round lifecycle.

Only model/provider/judge and chain boundaries are controlled. No paid calls.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import replace
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from lab_arena import broker as br, contracts, runtime, scoring, shim, submission_runtime
from lab_arena.contact_evidence import source_key
from lab_arena.promotion import GitPromoter
from qualification.scoring.competition import apply_company_judgment_context, _merge_contact_breakdown
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.company_quality_round_test import QualityHarness
from tests.lab_arena.contact_round_test import _claim, _contact_icps, _contact_result
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.test_arena_company_quality_policy import _positive_breakdown


class SourcingTransport(fixtures.FakeProviderTransport):
    def __init__(self):
        super().__init__()
        self.dispatched = []

    def send(self, **kwargs):
        request = json.loads(kwargs.get("body") or b"{}")
        if request.get("model") == "openai/gpt-4o-mini":
            prompt = request["messages"][0]["content"]
            self.dispatched.append(prompt)
            cost, failure = prompt.split("|")
            failed = failure == "failed"
            response = {"model": request["model"], "usage": {"cost": cost}}
            response.update({"error": {"code": 503, "message": "Controlled failure"}} if failed else {
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": ""}}]})
            return br.ProviderResponse(503 if failed else 200, {"content-type": "application/json"}, json.dumps(response).encode())
        return super().send(**kwargs)


class PerIcpHarness(QualityHarness):
    def objects_key(self):
        return "per-icp-cost-" + self.tmp.name

    def build_service(self):
        service = super().build_service()
        service.config.defaults = replace(service.config.defaults, contacts_from="2026-01-01T00:00:00Z", rewards_enabled=True, per_icp_cost_policy=True)
        source = service.config.daily_icp_source
        service.config.daily_icp_source = lambda **kwargs: {**source(**kwargs), "icps": _contact_icps(source(**kwargs)["icps"])}
        self.provider = getattr(self, "provider", SourcingTransport())
        payer = submission_runtime.SubmissionProviderKeys(store=service.store,
            credentials=fixtures.FakeCredentialManager(), organizer_keys=fixtures.CANARY_KEYS)
        service.config.broker_factory = lambda *_: br.Broker(store=service.store,
            credential_for=payer.credential_for, funding_source_for=payer.funding_source_for,
            key_for=lambda provider: fixtures.CANARY_KEYS[provider], price_table=fixtures.price_table(),
            judge_models=tuple(scoring.DEFAULT_JUDGE_MODELS.values()), transport=self.provider, clock=self.clock)
        return service


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS
        + (
            "289-lab-arena-per-icp-cost-policy.sql",
            "292-lab-arena-null-final-score-publication.sql",
        )
    )


def test_per_icp_overshoot_preserves_output_other_icps_restart_and_rewards(database, tmp_path):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = PerIcpHarness(connect, tmp_path, challengers=["Healthy"], runners=["alpha"])
    original = harness.sandbox.run_icp
    observed = []

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            def judge(companies, buyer, _reference):
                company = companies[0]
                name = company["company_name"]
                raw = _positive_breakdown(name, urlsplit(company["company_website"]).hostname,
                                          name.lower().replace(" ", "-"))
                score = 30.12345678901234 if name.startswith("PublicBaseline") else 60.12345678901234
                raw.update(final_score=score, intent_signal_raw=score, intent_signal_final=score)
                raw["intent_signals_detail"][0].update(raw=score, after_decay=score)
                evidence = document["contact_source_evidence"][source_key(company)]
                _merge_contact_breakdown(raw, asyncio.run(_contact_result(company, buyer, evidence)))
                return apply_company_judgment_context(companies, [raw], contacts_required=True)
            judge.company_quality = judge.integrity_policy = judge.contacts_required = True
            full, new = scoring.score_quality_work_item({"scored_run_id": document["scored_run_id"]},
                icp=document["icp"], companies=document["companies"], scorer=judge,
                cache_context=document["company_judgment_cache"])
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps(
                scoring.build_scoring_output(document["scored_run_id"], full, company_judgments=new)).encode())
        base = original(spec, **kwargs)  # Existing fixture's successful zero-cost discovery.
        assert base.output_bytes
        rows = json.loads(base.output_bytes)["companies"]
        position = int(document["icp"]["icp_id"].rsplit("_", 1)[-1]) - 1
        if position in (1, 2):
            rows = rows[:2]
        for company in rows:
            company.update(company_linkedin="https://linkedin.com/company/" + company["company_name"].lower().replace(" ", "-"), state="CA")
            company["contact"] = _claim(company, valid_role=True)
        calls = {0: [("3.90", "ok"), ("0.25", "ok"), ("0.01", "refused")],
                 1: [("1.60", "ok")], 2: [("1.600001", "ok")],
                 3: [("1.00", "failed"), ("0.20", "ok")]}.get(position, [("0.10", "ok")])
        with harness.sandbox.lock:
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                for cost, outcome in calls:
                    status, _, _ = shim.dispatch("openrouter.chat", {
                        "model": "openai/gpt-4o-mini",
                        "messages": [{"role": "user", "content": cost + "|" + ("failed" if outcome == "failed" else "ok")}],
                        "max_tokens": 100}, 5000)
                    assert status == {"ok": 200, "failed": 502, "refused": 402}[outcome]
                    observed.append((position, outcome, status))
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        # Valid output remains available after the overshooting call and refusal.
        return runtime.fake_result(exit_code=0, output_bytes=json.dumps({"companies": rows}).encode())

    harness.sandbox.run_icp = run_icp
    participants = fixtures._start_round(harness, day=28, epoch=31800)
    assert participants == 2
    config = harness.service.store.get_round(harness.round_id)["configuration_doc"]
    fixtures._run_stage_one_to_scoring(harness, participants, runners=1)
    harness.service = harness.build_service()
    assert harness.service.store.get_round(harness.round_id)["configuration_doc"] == config
    harness.advance_until("published", runners=1)
    saved = harness.service.store.get_round(harness.round_id)
    assert saved["status"] == "published"
    runs = harness.service.store.list_runs(harness.round_id, kind="execute")
    assert len(runs) == 40 and all(run["terminal_cause"] == "accepted" for run in runs)
    assert all(run["per_icp_score"] > 0 for run in runs), "Preserve raw quality scores even for cost-ineligible ICPs"
    assert observed.count((0, "refused", 402)) == 2
    assert "0.01|ok" not in harness.provider.dispatched
    for result in saved["publication_doc"]["final_ranking"]:
        assert result["eligible"] is True, result
        raw = [run["per_icp_score"] for run in runs if run["submission_id"] == result["submission_id"]]
        selected = {run["icp_position"]: run for run in runs if run["submission_id"] == result["submission_id"]}
        expected = sum(float(run["per_icp_score"]) for pos, run in selected.items() if pos not in (0, 2)) / 20
        assert result["final_score"] == pytest.approx(expected)
        assert 0 < result["final_score"] < sum(raw) / 20
        summary = result["cost_summary"]
        assert summary["competition_sourcing_microusd"] == 9_150_001
        assert summary["execution"]["settled_microusd"] == 10_150_001
        per_icp = {item["icp_position"]: item for item in summary["per_icp"]}
        assert set(per_icp) == set(range(20))
        assert {pos for pos, item in per_icp.items() if not item["eligible"]} == {0, 2}
        assert per_icp[0]["competition_sourcing_microusd"] == 4_150_000
        assert per_icp[0]["eligibility_cap_microusd"] == 4_000_000
        assert per_icp[1]["eligibility_cap_microusd"] == 1_600_000
        assert per_icp[2]["eligibility_cap_microusd"] == 1_600_000
        public = harness.service.public_results(harness.round_id, result["submission_id"])
        assert len(public["scores"]["stage_1"] + public["scores"]["stage_2"]) == 20
    # The database independently rejects forged score or per-ICP eligibility.
    genuine = saved["publication_doc"]["final_ranking"][0]
    forged_documents = []
    for key in ("competition_sourcing_microusd", "eligibility_cap_microusd", "qualified_company_count"):
        forged = deepcopy(genuine)
        forged["cost_summary"]["per_icp"][0][key] += 1
        forged_documents.append(forged)
    forged = deepcopy(genuine)
    forged["cost_summary"]["per_icp"][0]["eligible"] = True
    forged_documents.append(forged)
    forged = deepcopy(genuine)
    forged["cost_summary"]["per_icp"].pop()
    forged_documents.append(forged)
    forged = deepcopy(genuine)
    forged["final_score"] += 1
    forged_documents.append(forged)
    for forged in forged_documents:
        with connect() as connection:
            with connection.cursor() as cursor, pytest.raises(psycopg2.Error) as rejected:
                cursor.execute("SELECT public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)",
                               (harness.round_id, json.dumps(forged)))
            assert rejected.value.pgcode == "22023"
            connection.rollback()
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)",
                       (harness.round_id, json.dumps(genuine)))
        assert cursor.fetchone()[0] is True
    assert saved["king_outcome"] == "crowned"
    promotion = tmp_path / "promotion"; promotion.mkdir()
    remote = fixtures.promotion_repository(promotion)
    harness.service.config.baseline_promoter_factory = lambda: GitPromoter(str(remote), tmp_path / "promotion-objects")
    harness.clock.now = datetime.fromisoformat(saved["published_at"].replace("Z", "+00:00"))
    assert harness.service.promote_pending_baselines() == {"status": "ok", "promoted": 1}
    assert subprocess.check_output(["git", "--git-dir", str(remote), "show", "lab:flavor.txt"]).decode() == "Healthy"
    assert harness.service.activate_reward(harness.round_id)["status"] == "activated"
    assert harness.service.activate_reward(harness.round_id)["status"] == "existing"
    fixtures.assert_canary_absent(harness, connect)


@pytest.mark.parametrize(
    ("stop_reason", "day", "epoch"),
    [("money_cap", 29, 31801), ("per_icp_quota", 30, 31802)],
)
def test_per_icp_budget_stop_after_provider_failure_does_not_cancel_round(
    database, tmp_path, stop_reason, day, epoch
):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = PerIcpHarness(connect, tmp_path, challengers=[], runners=["alpha"])
    original = harness.sandbox.run_icp
    observed = []
    deepline_quota = None

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            def judge(companies, buyer, _reference):
                company = companies[0]
                name = company["company_name"]
                raw = _positive_breakdown(
                    name,
                    urlsplit(company["company_website"]).hostname,
                    name.lower().replace(" ", "-"),
                )
                raw.update(
                    final_score=30,
                    intent_signal_raw=30,
                    intent_signal_final=30,
                )
                raw["intent_signals_detail"][0].update(raw=30, after_decay=30)
                evidence = document["contact_source_evidence"][source_key(company)]
                _merge_contact_breakdown(
                    raw,
                    asyncio.run(_contact_result(company, buyer, evidence)),
                )
                return apply_company_judgment_context(
                    companies, [raw], contacts_required=True
                )

            judge.company_quality = judge.integrity_policy = judge.contacts_required = True
            full, new = scoring.score_quality_work_item(
                {"scored_run_id": document["scored_run_id"]},
                icp=document["icp"],
                companies=document["companies"],
                scorer=judge,
                cache_context=document["company_judgment_cache"],
            )
            return runtime.fake_result(
                exit_code=0,
                output_bytes=json.dumps(
                    scoring.build_scoring_output(
                        document["scored_run_id"], full, company_judgments=new
                    )
                ).encode(),
            )

        submission_id = spec.source_dir.parent.name.removeprefix("submission-")
        position = int(document["icp"]["icp_id"].rsplit("_", 1)[-1]) - 1
        if harness.flavors[submission_id] == "PublicBaseline" and position == 0:
            with harness.sandbox.lock:
                os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
                try:
                    if stop_reason == "money_cap":
                        calls = (("1.00", "failed"), ("3.10", "ok"), ("0.01", "refused"))
                    else:
                        calls = (("1.00", "failed"),)
                    for cost, outcome in calls:
                        status, _, _ = shim.dispatch(
                            "openrouter.chat",
                            {
                                "model": "openai/gpt-4o-mini",
                                "messages": [{
                                    "role": "user",
                                    "content": cost + "|" + (
                                        "failed" if outcome == "failed" else "ok"
                                    ),
                                }],
                                "max_tokens": 100,
                            },
                            5000,
                        )
                        assert status == {
                            "ok": 200,
                            "failed": 502,
                            "refused": 402,
                        }[outcome]
                        observed.append((outcome, status))
                    if stop_reason == "per_icp_quota":
                        assert deepline_quota is not None
                        for index in range(deepline_quota + 1):
                            status, _, _ = shim.dispatch(
                                "deepline.execute",
                                {"tool": "exa_search", "payload": {"query": "fintech"}},
                                5000,
                            )
                            expected = 200 if index < deepline_quota else 402
                            assert status == expected
                            observed.append((
                                "quota_ok" if index < deepline_quota else "quota_refused",
                                status,
                            ))
                finally:
                    os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            return runtime.fake_result(
                exit_code=1,
                output_bytes=None,
                stderr=b"controlled no output after budget stop",
            )

        base = original(spec, **kwargs)
        assert base.output_bytes
        rows = json.loads(base.output_bytes)["companies"]
        for company in rows:
            company.update(
                company_linkedin=(
                    "https://linkedin.com/company/"
                    + company["company_name"].lower().replace(" ", "-")
                ),
                state="CA",
            )
            company["contact"] = _claim(company, valid_role=True)
        return runtime.fake_result(
            exit_code=0,
            output_bytes=json.dumps({"companies": rows}).encode(),
        )

    harness.sandbox.run_icp = run_icp
    participants = fixtures._start_round(harness, day=day, epoch=epoch)
    assert participants == 1
    deepline_quota = harness.service.store.get_round(harness.round_id)[
        "configuration_doc"
    ]["call_quotas"]["deepline"]
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == 10
    harness.run_stage_with_runners(1)
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    scoring_opened = harness.service.advance_round(harness.round_id)
    assert scoring_opened["assignments"] == 9
    assert harness.status() == "stage1_scoring"
    harness.advance_until("published", runners=1)

    saved = harness.service.store.get_round(harness.round_id)
    assert saved["status"] == "published"
    baseline_id = next(iter(harness.flavors))
    runs = harness.service.store.list_runs(
        harness.round_id, submission_id=baseline_id, kind="execute"
    )
    accepted = [run for run in runs if run["status"] == "accepted"]
    budget_stops = [
        run for run in runs if run["terminal_cause"] == "budget_exhausted"
    ]
    assert len(accepted) == 19
    assert len(budget_stops) == 1
    assert budget_stops[0]["icp_position"] == 0
    if stop_reason == "money_cap":
        assert observed == [("failed", 502), ("ok", 200), ("refused", 402)]
        assert "0.01|ok" not in harness.provider.dispatched
    else:
        assert observed.count(("quota_ok", 200)) == deepline_quota
        assert observed.count(("quota_refused", 402)) == 1
        refusal_entries = harness.service.store.list_ledger(
            run_id=budget_stops[0]["run_id"], entry_kind="refusal"
        )
        assert [entry["entry_doc"]["reason"] for entry in refusal_entries] == [
            "per_icp_quota"
        ]

    result = saved["publication_doc"]["final_ranking"][0]
    assert result["submission_id"] == baseline_id
    assert result["eligible"] is True
    assert result["final_score"] > 0
    per_icp = {item["icp_position"]: item for item in result["cost_summary"]["per_icp"]}
    assert per_icp[0]["eligible"] is (stop_reason == "per_icp_quota")
    assert all(per_icp[position]["eligible"] for position in range(1, 20))
    fixtures.assert_canary_absent(harness, connect)


def test_all_budget_exhausted_challenger_publishes_null_score(database, tmp_path):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = PerIcpHarness(
        connect, tmp_path, challengers=["Empty", "Healthy"], runners=["alpha"]
    )
    original = harness.sandbox.run_icp

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            def judge(companies, buyer, _reference):
                company = companies[0]
                name = company["company_name"]
                raw = _positive_breakdown(
                    name,
                    urlsplit(company["company_website"]).hostname,
                    name.lower().replace(" ", "-"),
                )
                raw.update(
                    final_score=30,
                    intent_signal_raw=30,
                    intent_signal_final=30,
                )
                raw["intent_signals_detail"][0].update(raw=30, after_decay=30)
                evidence = document["contact_source_evidence"][source_key(company)]
                _merge_contact_breakdown(
                    raw,
                    asyncio.run(_contact_result(company, buyer, evidence)),
                )
                return apply_company_judgment_context(
                    companies, [raw], contacts_required=True
                )

            judge.company_quality = judge.integrity_policy = judge.contacts_required = True
            full, new = scoring.score_quality_work_item(
                {"scored_run_id": document["scored_run_id"]},
                icp=document["icp"],
                companies=document["companies"],
                scorer=judge,
                cache_context=document["company_judgment_cache"],
            )
            return runtime.fake_result(
                exit_code=0,
                output_bytes=json.dumps(
                    scoring.build_scoring_output(
                        document["scored_run_id"], full, company_judgments=new
                    )
                ).encode(),
            )

        submission_id = spec.source_dir.parent.name.removeprefix("submission-")
        if harness.flavors[submission_id] == "Empty":
            with harness.sandbox.lock:
                os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
                try:
                    for cost, expected in (("4.10", 200), ("0.01", 402)):
                        status, _, _ = shim.dispatch(
                            "openrouter.chat",
                            {
                                "model": "openai/gpt-4o-mini",
                                "messages": [
                                    {"role": "user", "content": cost + "|ok"}
                                ],
                                "max_tokens": 100,
                            },
                            5000,
                        )
                        assert status == expected
                finally:
                    os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            return runtime.fake_result(
                exit_code=1,
                output_bytes=None,
                stderr=b"controlled no output after per-ICP budget stop",
            )

        result = original(spec, **kwargs)
        assert result.output_bytes
        rows = json.loads(result.output_bytes)["companies"]
        for company in rows:
            company.update(
                company_linkedin=(
                    "https://linkedin.com/company/"
                    + company["company_name"].lower().replace(" ", "-")
                ),
                state="CA",
            )
            company["contact"] = _claim(company, valid_role=True)
        return runtime.fake_result(
            exit_code=0,
            output_bytes=json.dumps({"companies": rows}).encode(),
        )

    harness.sandbox.run_icp = run_icp
    participants = fixtures._start_round(harness, day=31, epoch=31803)
    assert participants == 3
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == 30
    harness.run_stage_with_runners(1)
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    assert harness.service.advance_round(harness.round_id)["assignments"] == 20
    assert harness.status() == "stage1_scoring"
    harness.advance_until("published", runners=1)

    saved = harness.service.store.get_round(harness.round_id)
    empty_id = next(
        submission_id
        for submission_id, flavor in harness.flavors.items()
        if flavor == "Empty"
    )
    assert empty_id in saved["finalists"]
    empty_runs = harness.service.store.list_runs(
        harness.round_id, submission_id=empty_id, kind="execute"
    )
    assert len(empty_runs) == 20
    assert all(run["terminal_cause"] == "budget_exhausted" for run in empty_runs)
    assert all(run["per_icp_score"] == 0 for run in empty_runs)

    ranking = {
        row["submission_id"]: row
        for row in saved["publication_doc"]["final_ranking"]
    }
    assert len(ranking) == participants
    assert ranking[empty_id]["final_score"] is None
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)",
            (harness.round_id, json.dumps(ranking[empty_id])),
        )
        assert cursor.fetchone()[0] is True

        accepted_challenger = next(
            row
            for row in ranking.values()
            if not row["is_baseline"] and row["submission_id"] != empty_id
        )
        assert accepted_challenger["final_score"] > 0
        forged = deepcopy(accepted_challenger)
        forged["final_score"] = None
        with pytest.raises(psycopg2.Error) as rejected:
            cursor.execute(
                "SELECT public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)",
                (harness.round_id, json.dumps(forged)),
            )
        assert rejected.value.pgcode == "22023"
        connection.rollback()

        baseline = next(row for row in ranking.values() if row["is_baseline"])
        assert baseline["final_score"] > 0
        forged = deepcopy(baseline)
        forged["final_score"] = None
        with pytest.raises(psycopg2.Error) as rejected:
            cursor.execute(
                "SELECT public.lab_arena__per_icp_publication_valid(%s,%s::jsonb)",
                (harness.round_id, json.dumps(forged)),
            )
        assert rejected.value.pgcode == "22023"
        connection.rollback()

    fixtures.assert_canary_absent(harness, connect)


def test_null_score_publication_migration_replays_and_rolls_back(database):
    psycopg2, dsn = database
    migration = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "292-lab-arena-null-final-score-publication.sql"
    ).read_text(encoding="utf-8")
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.lab_arena__per_icp_publication_valid(text,jsonb)'"
                "::pg_catalog.regprocedure)"
            )
            original = cursor.fetchone()[0]
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.lab_arena__per_icp_publication_valid(text,jsonb)'"
                "::pg_catalog.regprocedure)"
            )
            assert cursor.fetchone()[0] == original

            failing = migration.replace(
                "RETURN TRUE;", "RETURN FALSE;", 1
            ).replace(
                "pg_catalog.strpos(v_definition, 'v_is_baseline') = 0 THEN",
                "pg_catalog.strpos(v_definition, 'deliberate_missing_marker') = 0 THEN",
                1,
            )
            with pytest.raises(psycopg2.Error):
                cursor.execute(failing)
            cursor.execute("ROLLBACK")
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.lab_arena__per_icp_publication_valid(text,jsonb)'"
                "::pg_catalog.regprocedure)"
            )
            assert cursor.fetchone()[0] == original
    finally:
        connection.close()
