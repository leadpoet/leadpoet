"""Local signed-HTTP/SQL lifecycle for captured source TLS finality.

The affected company's scorer, investigation, source reader and retry logic are
real. Unrelated companies retain this repository's fixed positive leaf fixture.
Only provider replies, source execution and external evidence are controlled.
"""
import base64
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
import os
from threading import Lock
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
import pytest

from lab_arena import broker as br, runtime, runner, scorer_entrypoint, scoring, shim
from lab_arena.api import create_app
from qualification.scoring import company_evidence_investigator as investigator, lead_scorer
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.firecrawl_tls_source_refusal_test import CAPTURED, URL
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.scorer_environment_fixture import isolated_scorer_environment
from tests.lab_arena.test_deepline_catalog import frozen, row
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.test_arena_score_integrity import _positive_breakdown
from tests.test_company_evidence_investigator import _complete_verdict, _finding


@pytest.fixture(scope="module")
def database():
    yield from current_service_database()


@pytest.mark.parametrize("mode", ["tls", "tls_bounded", "pre_fix_tls", "provider_unavailable", "exa_tls"])
def test_captured_tls_baseline_and_miner_real_scorer_publish(database, tmp_path, monkeypatch, mode):
    if mode not in {"tls", "exa_tls"}:
        monkeypatch.setattr(investigator, "MAX_FETCH_CALLS", 1)
    if mode == "pre_fix_tls":
        monkeypatch.setattr(br, "_deepline_firecrawl_tls_source_refusal", lambda *_: False)
    psycopg, dsn = database
    connect = lambda: psycopg.connect(**dsn)
    harness = IntegrityHarness(connect, tmp_path, challengers=["TLSMiner"], runners=["alpha"])
    harness.service.config.defaults = replace(harness.service.config.defaults, per_icp_cost_policy=True)
    # Keep one controlled fit obligation amid the normal twenty-ICP bank.
    icps = fixtures.daily_icps()
    icps[0].update(sub_industry="SaaS", employee_count=["11-50"], company_stage="Public",
                   required_attribute="", product_service="software")
    harness.service.config.daily_icp_source = lambda **kw: {"status": "ready", "set_id": int(kw["set_id"]), "icps": deepcopy(icps)}
    catalog = frozen(*[row(tool, provider=provider, categories=["research"],
                          inputSchema={"type": "object", "properties": {name: {} for name in ("query", "url", "formats", "onlyMainContent", "maxAge", "timeout", "storeInCache", "ids", "text", "maxAgeHours", "livecrawlTimeout")}})
                       for tool, provider in (("firecrawl_scrape", "firecrawl"), ("exa_contents", "exa"), ("exa_search", "exa"))])
    harness.service.config.deepline_catalog_source = lambda **_: catalog
    provider_counts = Counter()
    send = fixtures.FakeProviderTransport.send

    def source_reply(self, **request):
        wire = json.loads(request.get("body") or b"{}")
        tool = wire.get("operation")
        if request["method"] != "POST" or tool not in {"firecrawl_scrape", "exa_contents"}:
            return send(self, **request)
        provider_counts[tool] += 1
        native = "tls-lifecycle-%d" % sum(provider_counts.values())
        if tool == "firecrawl_scrape":
            document = deepcopy(CAPTURED)
            document.update(request_id=native, requestId=native, billing={"credits_charged": .02})
            document["tool_error"]["requestId"] = native
            document["executionRecovery"]["idempotencyKey"] = request["headers"]["idempotency-key"]
            status, credits = 502, .02
            if mode == "provider_unavailable":
                document["upstream_error_code"] = "INTERNAL_SERVER_ERROR"
                document["tool_error"]["retryable"] = True
        else:
            target = wire["payload"]["ids"][0]
            document = {"status": "completed", "job_id": native, "billing": {"credits_charged": .01},
                        "result": {"data": {"results": [], "statuses": [{"id": target, "status": "error",
                        "error": {"httpStatusCode": 500 if mode == "exa_tls" else 404,
                                  "tag": "CRAWL_SSL_ERROR" if mode == "exa_tls" else "CRAWL_HTTP_404"}}]}}}
            status, credits = 200, .01
        with self._deepline_lock:
            self._deepline_jobs.append({"id": native, "request_id": native, "operation": tool,
                "provider": wire["provider"], "credits": credits, "delta": -credits,
                "charge_state": "posted", "charge_finality": "final", "billing_mode": "deduct_on_settle", "metadata": {}})
        return br.ProviderResponse(status, {"content-type": "application/json"}, json.dumps(document).encode())

    monkeypatch.setattr(fixtures.FakeProviderTransport, "send", source_reply)
    broker_factory = harness.service.config.broker_factory

    def routed_broker(service, round_row):
        instance = broker_factory(service, round_row)
        instance._host_shadow_score_compat_round_id = round_row["round_id"]
        instance._host_shadow_score_compat_source_round_id = round_row["round_id"]
        return instance

    harness.service.config.broker_factory = routed_broker
    state = {"company": None, "targets": (), "document": None}
    attempts = Counter()
    native_score = lead_scorer.score_company_competition_intent
    native_investigation = investigator.investigate_company_evidence

    async def score_company(**kwargs):
        company = kwargs["company"]
        if state["document"]["icp"]["icp_id"] == icps[0]["icp_id"] and company.company_name.endswith(" Company 0"):
            state["company"] = company
            attempts[state["document"]["scored_run_id"]] += 1
            return await native_score(**kwargs)
        return _positive_breakdown(company.company_name, urlsplit(company.company_website).hostname)

    async def broad(**_):
        company = state["company"]
        return _complete_verdict(observed_company_name=company.company_name,
            observed_company_website=company.company_website, observed_company_linkedin="",
            observed_company_stage="Public", stage_matches=True, stage_evidence_url=URL,
            stage_evidence_quote="Company shares are listed on a public exchange."), ""

    async def prechecks(*_, **__):
        return lead_scorer.company_fit_match("controlled unrelated prechecks")

    async def investigate(**kwargs):
        state["targets"] = kwargs["targets"]
        return await native_investigation(**kwargs)

    class Session:
        def __init__(self, **_): self.turn = 0
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return False
        def get(self, url, **kwargs):
            assert url == "https://api.scrapingdog.com/scrape"
            status, headers, body = shim.dispatch("scrapingdog.scrape", {"url": kwargs["params"]["url"]}, 30_000)
            return shim._AiohttpResponse(request_url=url, response_url="", status=status, headers=headers, body=body)

    async def post(session, url, *, headers, payload):
        if url == "https://api.exa.ai/contents":
            status, _, body = shim.dispatch("exa.contents", payload, 30_000)
            return status, json.loads(body)
        session.turn += 1
        if session.turn == 1:
            name, arguments = "fetch_page", {"url": URL}
        else:
            name, arguments = "submit_findings", {"findings": [_finding(target, status="UNPROVEN",
                observed_value=None, evidence_url="", evidence_quote="") for target in state["targets"]]}
        return 200, {"choices": [{"message": {"tool_calls": [{"id": "turn-%d" % session.turn, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}

    async def search(*_, **__): return {"results": []}
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad)
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", prechecks)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigate)
    monkeypatch.setattr(investigator.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", search)
    original_runtime = harness.sandbox.run_icp
    outputs = []
    lock = Lock()

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") != scoring.SCORING_INPUT_SCHEMA_VERSION:
            return original_runtime(spec, **kwargs)
        with lock:
            state["document"] = document
            environment = dict(os.environ)
            for name in (*scoring.CREDENTIAL_ENV_NAMES, *document["scorer_policy"]["env_bindings"]):
                os.environ.pop(name, None)
            os.environ.update(scorer_entrypoint.PLACEHOLDER_CREDENTIALS)
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                output = scorer_entrypoint.score_input(document)
            finally:
                os.environ.clear(); os.environ.update(environment)
            outputs.append(deepcopy(output))
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps(output).encode())

    harness.sandbox.run_icp = run_icp
    build_runner = harness.runner
    harness.runner = lambda index, parallel=4: build_runner(index, parallel=1)
    replays = []
    with TestClient(create_app(harness.service)) as client:
        original_post = client.post

        def checked_post(url, *args, **kwargs):
            response = original_post(url, *args, **kwargs)
            if str(url).endswith("/provider") and response.status_code == 200:
                document = response.json()
                call = document.get("call", {})
                if call.get("operation_id") in {"scrapingdog.scrape", "exa.contents"}:
                    before = provider_counts.copy()
                    replay = original_post(url, *args, **kwargs).json()
                    assert replay["status"] == document["status"]
                    assert replay["body_b64"] == document["body_b64"]
                    assert replay["call"]["call_identity"] == call["call_identity"]
                    assert replay["call"]["idempotent"] is True
                    assert replay["call"]["actual_microusd"] == call["actual_microusd"]
                    assert provider_counts == before
                    replays.append(deepcopy(document))
            return response

        client.post = checked_post
        harness.api_factory = lambda: runner.HttpArenaApiClient("http://localhost", client=client)
        day = 25 + ("tls", "tls_bounded", "pre_fix_tls", "provider_unavailable", "exa_tls").index(mode)
        fixtures._start_round(harness, day=day, epoch=63000 + day)
        harness.advance_until("stage1_scoring", runners=1, max_steps=120)
        harness.run_stage_with_runners(1)
        if mode in {"pre_fix_tls", "provider_unavailable"}:
            failed = [output for output in outputs if "failure" in output]
            assert len(failed) == 2, (outputs, attempts, provider_counts)
            assert all(output["reason"] == "provider_error" for output in failed)
            assert len(attempts) == 2 and set(attempts.values()) == {scoring.MAX_JUDGE_RETRIES}
            failed_runs = [run for run in harness.service.store.list_runs(harness.round_id, kind="score") if run["status"] == "failed"]
            assert len(failed_runs) == 2 and all(run["per_icp_score"] is None for run in failed_runs)
            assert harness.service.store.get_round(harness.round_id)["publication_doc"] is None
            return
        harness.advance_until("scored", runners=1, max_steps=120)
        assert all("failure" not in output for output in outputs), outputs
        assert len(attempts) == 2 and set(attempts.values()) == {1}
        assert provider_counts == ({"firecrawl_scrape": 4, "exa_contents": 4} if mode in {"tls", "exa_tls"} else {"firecrawl_scrape": 2})
        affected = [output for output in outputs if output["scored_run_id"] in attempts]
        assert len(affected) == 2
        for output in affected:
            assert output["breakdowns"][0]["final_score"] == 0
            assert output["breakdowns"][0]["company_qualified"] is False
            assert all(row["final_score"] == 60 and row["company_qualified"] is True
                       for row in output["breakdowns"][1:])
        assert len(replays) == sum(provider_counts.values())
        assert {item["call"]["funding_source"] for item in replays} == {"host", "miner_key"}
        with connect() as connection, connection.cursor() as cursor:
            for item in replays:
                call = item["call"]
                cursor.execute("SELECT entry_kind,amount_microusd,terminal_response FROM public.lab_arena_ledger "
                               "WHERE call_identity=%s ORDER BY entry_id", (call["call_identity"],))
                ledger = cursor.fetchall()
                assert [entry[0] for entry in ledger] == ["reservation", "dispatch", "settlement"]
                assert ledger[-1][1] == call["actual_microusd"]
                terminal = ledger[-1][2]
                if call["operation_id"] == "scrapingdog.scrape":
                    assert item["status"] == terminal["status"] == 403
                    assert call["actual_microusd"] == 2_000
                    assert terminal["call_succeeded"] is False
                    raw = base64.b64decode(terminal["upstream_response_receipt"]["body_b64"])
                    assert json.loads(raw)["upstream_error_code"] == "SCRAPE_SSL_ERROR"
                else:
                    assert item["status"] == 200 and call["actual_microusd"] == 1_000
        scores = harness.service.store.list_runs(harness.round_id, kind="score")
        assert all(run["status"] == "accepted" and run["attempt"] == 1 for run in scores)
        harness.service = harness.build_service()
        published = harness.advance_until("published", runners=1, max_steps=120)
        assert published["publication_doc"]["final_ranking"]
        for submission in harness.service.store.list_submissions(harness.round_id):
            results = harness.service.public_results(harness.round_id, submission["submission_id"])
            assert len(results["outputs"]) == 20
        fixtures.assert_canary_absent(harness, connect)
