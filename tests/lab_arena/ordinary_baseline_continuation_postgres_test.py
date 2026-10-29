"""Normal current baseline judges persist through the existing scored writer."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from lab_arena import contracts, runner, runtime, scoring
from lab_arena.api import create_app
from qualification.scoring.competition import apply_company_judgment_context
from tests.test_arena_company_quality_policy import _positive_breakdown
from lab_arena.service import ServiceError
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _enable_verified_proxy_runtime,
)


@pytest.fixture
def database():
    yield from current_service_database()


def _scored_round(database, tmp_path, *, flat, http_runner=False):
    psycopg2, dsn = database
    harness = IntegrityHarness(
        lambda: psycopg2.connect(**dsn), tmp_path,
        challengers=["Miner"], runners=["alpha", "beta"],
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=5,
        runner_slot_ceiling=5,
        execution_sequence_from="2026-01-01T00:00:00Z",
        company_judgment_from="2026-01-01T00:00:00Z",
        per_icp_cost_policy=not flat,
        flat_per_icp_cost_policy=flat,
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": int(kwargs["set_id"]),
        "icps": deepcopy(daily_icps()[:5]),
    }
    def judge(companies, icp, reference):
        rows = []
        for company in companies:
            name = company["company_name"]
            raw = _positive_breakdown(name, urlsplit(company["company_website"]).hostname,
                                      name.lower().replace(" ", "-"))
            value = 80 if name.startswith("PublicBaseline") else 10
            raw.update(final_score=value, intent_signal_raw=value, intent_signal_final=value)
            raw["intent_signals_detail"][0].update(raw=value, after_decay=value)
            rows.append(raw)
        return apply_company_judgment_context(companies, rows, company_quality=False)

    judge.integrity_policy = True
    original = harness.sandbox.run_icp

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") != scoring.SCORING_INPUT_SCHEMA_VERSION:
            result = original(spec, **kwargs)
            output = json.loads(result.output_bytes)
            for company in output["companies"]:
                company["company_linkedin"] = "https://linkedin.com/company/" + company["company_name"].lower().replace(" ", "-")
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps(output).encode())
        full, new = scoring.score_quality_work_item(
            {"scored_run_id": document["scored_run_id"]},
            icp=document["icp"], companies=document["companies"],
            scorer=judge,
            cache_context=document["company_judgment_cache"],
        )
        output = scoring.build_scoring_output(
            document["scored_run_id"], full, company_judgments=new,
        )
        return runtime.fake_result(exit_code=0, output_bytes=json.dumps(output).encode())

    harness.sandbox.run_icp = run_icp
    _enable_verified_proxy_runtime(harness)
    harness.clock.now = datetime.now(timezone.utc)
    config = harness.service.create_round(
        harness.clock.now + timedelta(minutes=30),
        round_id="arena-2099-01-01-normalflat" if flat else "arena-2099-01-01-normallegacy",
    )
    harness.round_id = config["round_id"]
    harness.submit("Miner", harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    committed = harness.service.store.get_round(harness.round_id)
    assert committed["evaluation_date"] >= "2026-10-15"
    assert committed["configuration_doc"]["company_judgment_policy"] == "company_judgment_v1"
    assert "company_quality_policy" not in committed["configuration_doc"]
    assert "contact_policy" not in committed["configuration_doc"]
    assert committed["configuration_doc"]["sourcing_cost_eligibility_policy"] == (
        contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY if flat
        else contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
    )
    for participant in committed["participants"]:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    if http_runner:
        paths = []
        with TestClient(create_app(harness.service)) as http:
            http.event_hooks["request"].append(lambda request: paths.append(request.url.path))
            harness.api_factory = lambda: runner.HttpArenaApiClient("http://localhost", client=http)
            try:
                harness.advance_until("scored", runners=2)
            finally:
                harness.api_factory = None
        assert "/arena/v1/runs/claim" in paths
        assert any(path.endswith("/source") for path in paths)
        assert any(path.endswith("/complete") for path in paths)
    else:
        harness.advance_until("scored", runners=2)
    return harness


def _assert_publication_http(harness, published):
    round_id = harness.round_id
    executions = harness.service.store.list_runs(round_id, kind="execute")
    with TestClient(create_app(harness.service)) as http:
        response = http.get("/arena/v1/current")
        assert response.status_code == 200, response.text
        assert response.json()["published_round"]["round_id"] == round_id
        response = http.get("/arena/v1/rounds/" + round_id)
        assert response.status_code == 200, response.text
        view = response.json()
        assert view["status"] == "published"
        assert view["company_judgment_policy"] == "company_judgment_v1"
        assert view["final_ranking"] == published["publication_doc"]["final_ranking"]
        for participant in published["publication_doc"]["final_ranking"]:
            submission_id = participant["submission_id"]
            response = http.get(f"/arena/v1/rounds/{round_id}/results/{submission_id}")
            assert response.status_code == 200, response.text
            result = response.json()
            expected = {run["run_id"]: run["per_icp_score"] for run in executions
                        if run["submission_id"] == submission_id}
            assert result["score_status"] == "published"
            assert result["public_icp_status"] == "ready"
            assert result["public_icp_count"] == len(expected) == 5
            scores = result["scores"]["stage_1"] + result["scores"]["stage_2"]
            assert {row["run_id"]: row["per_icp_score"] for row in scores} == expected
            assert set(result["outputs"]) == set(expected)
            assert result["submission_scores"]["final"] == participant["final_score"]


@pytest.mark.parametrize("flat", [False, True], ids=["legacy-cost", "flat-cost"])
@pytest.mark.parametrize("fault", [None, "missing_judge", "wrong_miner", "stale_generation"])
def test_current_ordinary_baseline_requires_bound_judges_before_publication(database, tmp_path, flat, fault):
    harness = _scored_round(database, tmp_path, flat=flat, http_runner=flat and fault is None)
    store = harness.service.store
    before = store.get_round(harness.round_id)
    executions = store.list_runs(harness.round_id, kind="execute")
    judges = store.list_runs(harness.round_id, kind="score")
    assert len(executions) == len(judges) == 10
    assert all(run["status"] == "accepted" for run in executions + judges)
    baseline = next(p["submission_id"] for p in before["participants"] if p["is_king"])
    source = next(run for run in executions if run["submission_id"] == baseline)
    judge = next(run for run in judges if run["scored_run_id"] == source["run_id"])
    assert judge["attempt"] == source["attempt"] == 1
    assert not any(harness.service._is_baseline_recovery(run, executions + judges, before)
                   for run in executions + judges)
    original_source_score = source["per_icp_score"]
    if fault:
        with harness.connect() as connection, connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            if fault == "missing_judge":
                cursor.execute("DELETE FROM public.lab_arena_runs WHERE run_id=%s", (judge["run_id"],))
            elif fault == "wrong_miner":
                cursor.execute("UPDATE public.lab_arena_runs SET miner_hotkey=%s WHERE run_id=%s",
                               (harness.runner_keys[0], judge["run_id"]))
            else:
                cursor.execute("UPDATE public.lab_arena_runs SET scoring_generation_floor=1 WHERE run_id=%s", (judge["run_id"],))
    if fault == "wrong_miner":
        with pytest.raises(ServiceError, match="(?:recovered_judgment|accepted_judgment)_binding_invalid"):
            harness.service.publish(harness.round_id)
    else:
        result = harness.service.publish(harness.round_id)
        assert result["status"] == ("retry" if fault else "ok"), result
    final = store.get_round(harness.round_id)
    if fault:
        assert final["status"] == "scored"
        assert final["publication_doc"] is None
        assert final["reward_activated_at"] is None
        assert store.get_run(source["run_id"])["per_icp_score"] == original_source_score
    else:
        assert final["status"] == "published"
        assert len(final["publication_doc"]["final_ranking"]) == 2
        assert store.get_run(source["run_id"])["per_icp_score"] == 80
        assert all(run["per_icp_score"] is not None
                   for run in store.list_runs(harness.round_id, kind="execute"))
        _assert_publication_http(harness, final)
    assert {run["run_id"] for run in store.list_runs(harness.round_id, kind="execute")} == {
        run["run_id"] for run in executions
    }
