from __future__ import annotations

import base64
import json
import subprocess
import threading
import urllib.parse
import urllib.request
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gateway.research_lab import provider_evidence_proxy
from research_lab.canonical import sha256_json
from research_lab.eval.private_runtime import (
    DockerPrivateModelRunner,
    DockerPrivateModelSpec,
    PROVIDER_COST_EVALUATION_SCOPE_ENV,
)
from research_lab.eval.provider_costs import (
    DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    ProviderCostEstimate,
    ProviderCostLedger,
    decode_cost_event_header,
    estimate_provider_cost,
    extract_exa_cost_dollars,
    extract_openrouter_cost_dollars,
    openrouter_perplexity_pricing_fallback,
    openrouter_generation_id,
    scrapingdog_credits_for_path,
    scrapingdog_credits_for_url,
    summarize_provider_cost_trace_entries,
)


def _docker_cost_scope_for_seed(monkeypatch, evaluation_scope: str) -> str:
    captured_commands: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001
        captured_commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    image_digest = "123456789012.dkr.ecr.us-east-1.amazonaws.com/model@sha256:" + "a" * 64
    runner = DockerPrivateModelRunner(
        DockerPrivateModelSpec(
            image_digest=image_digest,
            pull_before_run=False,
            extra_env={PROVIDER_COST_EVALUATION_SCOPE_ENV: evaluation_scope},
        )
    )
    stdin_payload = {"icp": {"id": "icp-1"}, "context": {"mode": "private_baseline"}}
    runner._run_json(
        bootstrap="print([])",
        argv=("research_lab_adapter", "run_icp"),
        stdin_payload=stdin_payload,
    )
    command = captured_commands[-1]
    scope_args = [
        value
        for index, value in enumerate(command)
        if index > 0
        and command[index - 1] == "-e"
        and value.startswith("RESEARCH_LAB_PROVIDER_COST_SCOPE=")
    ]
    assert len(scope_args) == 1
    expected = sha256_json(
        {
            "image_digest": image_digest,
            "argv": ["research_lab_adapter", "run_icp"],
            "stdin_payload": stdin_payload,
            "evaluation_scope": evaluation_scope,
        }
    )
    assert scope_args[0] == f"RESEARCH_LAB_PROVIDER_COST_SCOPE={expected}"
    return scope_args[0]


def test_docker_provider_cost_scope_includes_evaluation_scope(monkeypatch):
    first = _docker_cost_scope_for_seed(monkeypatch, "sha256:" + "1" * 64)
    second = _docker_cost_scope_for_seed(monkeypatch, "sha256:" + "2" * 64)

    assert first != second


def test_scrapingdog_cost_map_uses_current_endpoint_credits():
    assert scrapingdog_credits_for_path("/profile") == 100
    assert scrapingdog_credits_for_path("/profile/post") == 5
    assert scrapingdog_credits_for_path("/google/ai") == 10
    assert scrapingdog_credits_for_path("/google-ai") == 10
    assert scrapingdog_credits_for_path("/youtube/transcripts") == 5
    assert scrapingdog_credits_for_path("/tiktok/profile") == 5
    assert scrapingdog_credits_for_path("/linkedinjobs") == 5
    assert scrapingdog_credits_for_path("/instagram/profile") == 15
    assert scrapingdog_credits_for_path("/unmapped") is None
    assert scrapingdog_credits_for_url("https://api.scrapingdog.com/profile?type=company&id=abc") == 10
    assert scrapingdog_credits_for_url("https://api.scrapingdog.com/profile?type=COMPANY&id=abc") == 10
    assert scrapingdog_credits_for_url("https://api.scrapingdog.com/profile?type=profile&id=abc") == 100
    assert scrapingdog_credits_for_url("https://api.scrapingdog.com/profile?id=abc") == 100
    assert scrapingdog_credits_for_url("https://api.scrapingdog.com/scrape?url=https%3A%2F%2Fexample.com") == 5
    assert scrapingdog_credits_for_url("https://api.scrapingdog.com/scrape?dynamic=true") == 5
    assert scrapingdog_credits_for_url("https://api.scrapingdog.com/scrape?premium=true") == 5
    assert scrapingdog_credits_for_url("https://api.scrapingdog.com/scrape?dynamic=true&premium=true") == 5
    assert scrapingdog_credits_for_url("https://api.scrapingdog.com/scrape?country=us") == 5
    assert scrapingdog_credits_for_url("https://api.scrapingdog.com/new/paid/endpoint") == 5


def test_scrapingdog_success_cost_and_unknown_endpoint_defaults_to_five_credits():
    profile = estimate_provider_cost(
        provider="sd",
        upstream_url="https://api.scrapingdog.com/profile?id=abc",
        status=200,
        response_body=b"{}",
        request_body=None,
        scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    )
    assert profile.billable
    assert profile.credits == 100
    assert profile.cost_usd == Decimal("0.00500")

    company = estimate_provider_cost(
        provider="sd",
        upstream_url="https://api.scrapingdog.com/profile?id=abc&type=company",
        status=200,
        response_body=b"{}",
        request_body=None,
        scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    )
    assert company.billable
    assert company.credits == 10
    assert company.cost_usd == Decimal("0.00050")

    unknown = estimate_provider_cost(
        provider="sd",
        upstream_url="https://api.scrapingdog.com/new/paid/endpoint",
        status=200,
        response_body=b"{}",
        request_body=None,
        scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    )
    assert unknown.billable
    assert unknown.cost_source == "scrapingdog_default_credit_map"
    assert unknown.credits == 5
    assert unknown.cost_usd == Decimal("0.00025")

    scrape = estimate_provider_cost(
        provider="sd",
        upstream_url="https://api.scrapingdog.com/scrape?dynamic=true&url=https%3A%2F%2Fexample.com",
        status=200,
        response_body=b"{}",
        request_body=None,
        scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    )
    assert scrape.billable
    assert scrape.credits == 5
    assert scrape.cost_usd == Decimal("0.00025")


def test_scrapingdog_company_profile_cost_matches_private_model_call_shape():
    params = {"api_key": "redacted", "type": "company", "id": "amazon"}
    private_model_url = "https://api.scrapingdog.com/profile?" + urllib.parse.urlencode(params)

    assert private_model_url.endswith("api_key=redacted&type=company&id=amazon")
    assert scrapingdog_credits_for_url(private_model_url) == 10

    estimate = estimate_provider_cost(
        provider="sd",
        upstream_url=private_model_url,
        status=200,
        response_body=b"{}",
        request_body=None,
        scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    )
    assert estimate.billable
    assert estimate.credits == 10
    assert estimate.cost_usd == Decimal("0.00050")

    for status in (402, 500):
        failed = estimate_provider_cost(
            provider="sd",
            upstream_url=private_model_url,
            status=status,
            response_body=b'{"error":"provider failed"}',
            request_body=None,
            scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
        )
        assert not failed.billable
        assert failed.credits == 0
        assert failed.cost_usd == Decimal("0")
        assert failed.cost_source == "provider_failure_zero_cost"


def test_failed_provider_call_adds_zero_cost():
    failed = estimate_provider_cost(
        provider="exa",
        upstream_url="https://api.exa.ai/search",
        status=503,
        response_body=b'{"error":"temporarily unavailable"}',
        request_body=None,
        scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    )
    assert not failed.billable
    assert failed.cost_usd == Decimal("0")
    assert not failed.tracking_failed

    for status in (402, 500):
        sd_failed = estimate_provider_cost(
            provider="sd",
            upstream_url="https://api.scrapingdog.com/scrape?url=https%3A%2F%2Fexample.com",
            status=status,
            response_body=b'{"error":"provider failed"}',
            request_body=None,
            scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
        )
        assert not sd_failed.billable
        assert sd_failed.cost_usd == Decimal("0")
        assert sd_failed.cost_source == "provider_failure_zero_cost"
        assert not sd_failed.tracking_failed


def test_exa_and_openrouter_cost_extraction():
    assert extract_exa_cost_dollars(b'{"results":[],"costDollars":{"total":0.017}}') == Decimal("0.017")

    inline_cost, inline_meta = extract_openrouter_cost_dollars(
        b'{"usage":{"prompt_tokens":12,"completion_tokens":3,"cost":0.0042}}'
    )
    assert inline_cost == Decimal("0.0042")
    assert inline_meta["prompt_tokens"] == 12
    assert inline_meta["completion_tokens"] == 3

    reconciled_cost, reconciled_meta = extract_openrouter_cost_dollars(
        b'{"data":{"total_cost":"0.0065","native_tokens_prompt":21,"native_tokens_completion":8}}'
    )
    assert reconciled_cost == Decimal("0.0065")
    assert reconciled_meta["prompt_tokens"] == 21
    assert reconciled_meta["completion_tokens"] == 8

    event_stream_cost, event_stream_meta = extract_openrouter_cost_dollars(
        b'data: {"id":"gen-123","choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"usage":{"prompt_tokens":31,"completion_tokens":9,"cost":"0.0071"}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert event_stream_cost == Decimal("0.0071")
    assert event_stream_meta["prompt_tokens"] == 31
    assert event_stream_meta["completion_tokens"] == 9
    assert openrouter_generation_id(
        b'data: {"id":"gen-123","choices":[{"delta":{"content":"hi"}}]}\n\n'
        b"data: [DONE]\n\n"
    ) == "gen-123"
    assert (
        openrouter_generation_id(
            b'{"data":{"id":"gen-nested-456","total_cost":"0.001"}}'
        )
        == "gen-nested-456"
    )
    assert (
        openrouter_generation_id(
            b'{"response":{"generationId":"gen-response-789"}}'
        )
        == "gen-response-789"
    )


def test_openrouter_perplexity_pricing_fallback_uses_model_specific_rates():
    sonar = openrouter_perplexity_pricing_fallback(
        model="perplexity/sonar",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert sonar is not None
    assert sonar.billable
    assert sonar.cost_usd == Decimal("0.0065")
    assert sonar.cost_source == "openrouter_perplexity_token_pricing_fallback"

    sonar_pro = openrouter_perplexity_pricing_fallback(
        model="perplexity/sonar-pro",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert sonar_pro is not None
    assert sonar_pro.cost_usd == Decimal("0.0155")

    deep_research = openrouter_perplexity_pricing_fallback(
        model="perplexity/sonar-deep-research",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert deep_research is not None
    assert deep_research.cost_usd == Decimal("0.011")

    assert (
        openrouter_perplexity_pricing_fallback(
            model="anthropic/claude-opus-4.1",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        is None
    )


def test_openrouter_missing_cost_zero_cost_for_non_perplexity_model():
    estimate = estimate_provider_cost(
        provider="or",
        upstream_url="https://openrouter.ai/api/v1/chat/completions",
        status=200,
        response_body=b'{"id":"gen-no-cost","usage":{"prompt_tokens":7,"completion_tokens":3}}',
        request_body=b'{"model":"google/gemini-2.5-pro"}',
        scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    )
    assert not estimate.billable
    assert estimate.cost_usd == Decimal("0")
    assert estimate.cost_source == "openrouter_missing_cost_zero_cost"
    assert not estimate.tracking_failed
    assert estimate.tracking_reason == "missing_openrouter_cost"


def test_openrouter_missing_cost_perplexity_fallback_from_response_usage():
    estimate = estimate_provider_cost(
        provider="or",
        upstream_url="https://openrouter.ai/api/v1/chat/completions",
        status=200,
        response_body=b'{"id":"gen-sonar-no-cost","usage":{"prompt_tokens":7,"completion_tokens":3}}',
        request_body=b'{"model":"perplexity/sonar"}',
        scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    )
    assert estimate.billable
    assert estimate.cost_usd == Decimal("0.00501")
    assert estimate.cost_source == "openrouter_perplexity_token_pricing_fallback"
    assert not estimate.tracking_failed
    assert estimate.generation_id == "gen-sonar-no-cost"


def test_exa_agent_running_poll_is_not_billable_until_completed():
    running = estimate_provider_cost(
        provider="exa",
        upstream_url="https://api.exa.ai/agent/runs/agent_run_123",
        status=200,
        response_body=b'{"object":"agent_run","id":"agent_run_123","status":"running","costDollars":{"total":0.1}}',
        request_body=None,
        scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    )
    assert not running.billable
    assert running.cost_usd == Decimal("0")
    assert running.cost_source == "exa_agent_nonterminal_poll_zero_cost"
    assert not running.tracking_failed

    completed = estimate_provider_cost(
        provider="exa",
        upstream_url="https://api.exa.ai/agent/runs/agent_run_123",
        status=200,
        response_body=b'{"object":"agent_run","id":"agent_run_123","status":"completed","costDollars":{"total":0.1}}',
        request_body=None,
        scrapingdog_credit_price_usd=DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    )
    assert completed.billable
    assert completed.cost_usd == Decimal("0.1")
    assert completed.cost_source == "exa_cost_dollars"


def test_ledger_allows_final_success_to_exceed_cap_then_blocks_later_call():
    ledger = ProviderCostLedger(scope="scope-1", cap_usd=Decimal("0.50"))
    first = ledger.record_live_event(
        provider="exa",
        request_fingerprint="a" * 64,
        status_code=200,
        estimate=ProviderCostEstimate(
            provider="exa",
            endpoint="/search",
            billable=True,
            cost_usd=Decimal("0.51"),
            cost_source="exa_cost_dollars",
        ),
    )
    assert first.cap_exceeded_after_success
    assert not first.cap_blocked

    summary_after_first = summarize_provider_cost_trace_entries(
        [{"provider_cost_event": first.to_doc()}]
    )
    assert summary_after_first["cap_exceeded_after_success"]
    assert not summary_after_first["cap_blocked"]
    assert summary_after_first["total_cost_usd"] == 0.51

    assert ledger.should_block_paid_call()
    second = ledger.block_event(
        provider="exa",
        endpoint="/search",
        request_fingerprint="b" * 64,
        reason="cost_cap_reached",
    )
    summary_after_second = summarize_provider_cost_trace_entries(
        [{"provider_cost_event": first.to_doc()}, {"provider_cost_event": second.to_doc()}]
    )
    assert summary_after_second["cap_blocked"]
    assert summary_after_second["blocked_call_count"] == 1


def test_openrouter_missing_cost_zero_event_does_not_block_later_paid_calls():
    ledger = ProviderCostLedger(scope="scope-tracking", cap_usd=Decimal("0.50"))
    zero_cost = ledger.record_live_event(
        provider="or",
        request_fingerprint="d" * 64,
        status_code=200,
        estimate=ProviderCostEstimate(
            provider="or",
            endpoint="/api/v1/chat/completions",
            model="google/gemini-2.5-pro",
            billable=False,
            cost_source="openrouter_missing_cost_zero_cost",
            tracking_reason="missing_openrouter_cost",
        ),
    )
    assert not zero_cost.tracking_failed
    assert not ledger.should_block_paid_call()

    paid = ledger.record_live_event(
        provider="exa",
        request_fingerprint="e" * 64,
        status_code=200,
        estimate=ProviderCostEstimate(
            provider="exa",
            endpoint="/search",
            billable=True,
            cost_usd=Decimal("0.01"),
            cost_source="exa_cost_dollars",
        ),
    )
    assert paid.billable
    assert paid.spent_before_usd == Decimal("0")
    assert paid.spent_after_usd == Decimal("0.01")


def test_cache_hit_adds_zero_cost_to_summary():
    ledger = ProviderCostLedger(scope="scope-2", cap_usd=Decimal("0.50"))
    event = ledger.cache_hit_event(
        provider="sd",
        endpoint="/profile",
        request_fingerprint="c" * 64,
        status_code=200,
    )
    summary = summarize_provider_cost_trace_entries([{"provider_cost_event": event.to_doc()}])
    assert summary["cache_hit_count"] == 1
    assert summary["total_cost_usd"] == 0.0
    assert summary["paid_call_count"] == 0


class _FakeProvider(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: ANN001
        pass

    def do_POST(self):  # noqa: N802
        body = json.dumps({"results": [], "costDollars": 0.0123}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeExaAgentProvider(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *args):  # noqa: ANN001
        pass

    def do_GET(self):  # noqa: N802
        type(self).calls += 1
        status = "running" if type(self).calls == 1 else "completed"
        cost = 0.1 if status == "completed" else 0.0
        body = json.dumps(
            {
                "object": "agent_run",
                "id": "agent_run_123",
                "status": status,
                "costDollars": {"total": cost},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeOpenRouterProvider(BaseHTTPRequestHandler):
    seen_completion_bodies: list[dict] = []
    generation_ids: list[str] = []

    def log_message(self, *args):  # noqa: ANN001
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        type(self).seen_completion_bodies.append(payload)
        body = json.dumps(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-OpenRouter-Generation-Id", "gen-sonar-cost-1")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/api/v1/generation":
            self.send_response(404)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        query = dict(urllib.parse.parse_qsl(parsed.query))
        type(self).generation_ids.append(query.get("id") or "")
        body = json.dumps(
            {
                "data": {
                    "total_cost": "0.0032",
                    "native_tokens_prompt": 11,
                    "native_tokens_completion": 5,
                }
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeOpenRouterBodyIdProvider(BaseHTTPRequestHandler):
    generation_ids: list[str] = []

    def log_message(self, *args):  # noqa: ANN001
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        body = json.dumps(
            {
                "id": "gen-sonar-body-cost-1",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/api/v1/generation":
            self.send_response(404)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        query = dict(urllib.parse.parse_qsl(parsed.query))
        type(self).generation_ids.append(query.get("id") or "")
        body = json.dumps(
            {
                "data": {
                    "total_cost": "0.0041",
                    "native_tokens_prompt": 19,
                    "native_tokens_completion": 6,
                }
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeOpenRouterGenerationTokensProvider(BaseHTTPRequestHandler):
    generation_ids: list[str] = []

    def log_message(self, *args):  # noqa: ANN001
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        body = json.dumps(
            {
                "id": "gen-sonar-token-priced-1",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/api/v1/generation":
            self.send_response(404)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        query = dict(urllib.parse.parse_qsl(parsed.query))
        type(self).generation_ids.append(query.get("id") or "")
        body = json.dumps(
            {
                "data": {
                    "id": query.get("id"),
                    "model": "perplexity/sonar-pro",
                    "tokens_prompt": 1000,
                    "tokens_completion": 500,
                }
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeOpenRouterMissingCostProvider(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *args):  # noqa: ANN001
        pass

    def do_POST(self):  # noqa: N802
        type(self).calls += 1
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        body = json.dumps(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_evidence_proxy_emits_cost_event_header_for_live_success():
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeProvider)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = None
    try:
        registry = [
            provider_evidence_proxy.ProviderRegistryEntry(
                id="exa",
                base_url=f"http://127.0.0.1:{upstream.server_address[1]}",
                auth_kind="none",
            )
        ]
        proxy, _store, proxy_thread = provider_evidence_proxy.serve_evidence_proxy(
            host="127.0.0.1",
            port=0,
            registry=registry,
        )
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_address[1]}/exa/search",
            data=b'{"query":"redacted"}',
            headers={
                "Content-Type": "application/json",
                "X-Research-Lab-Cost-Scope": "test-scope",
                "X-Research-Lab-Cost-Cap-Usd": "0.50",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            event = decode_cost_event_header(response.headers.get("X-Research-Lab-Provider-Cost-Event"))
            assert event is not None
            assert event["provider"] == "exa"
            assert event["evidence"] == "recorded"
            assert event["billable"] is True
            assert event["cost_usd"] == 0.0123
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        upstream.shutdown()
        upstream.server_close()


def test_evidence_proxy_reconciles_openrouter_generation_cost_from_header():
    _FakeOpenRouterProvider.seen_completion_bodies = []
    _FakeOpenRouterProvider.generation_ids = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenRouterProvider)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = None
    try:
        registry = [
            provider_evidence_proxy.ProviderRegistryEntry(
                id="or",
                base_url=f"http://127.0.0.1:{upstream.server_address[1]}",
                auth_kind="none",
            )
        ]
        proxy, _store, _proxy_thread = provider_evidence_proxy.serve_evidence_proxy(
            host="127.0.0.1",
            port=0,
            registry=registry,
        )
        request_body = json.dumps(
            {
                "model": "perplexity/sonar",
                "messages": [{"role": "user", "content": "redacted"}],
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_address[1]}/or/api/v1/chat/completions",
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "X-Research-Lab-Cost-Scope": "or-scope",
                "X-Research-Lab-Cost-Cap-Usd": "0.50",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            event = decode_cost_event_header(response.headers.get("X-Research-Lab-Provider-Cost-Event"))

        assert _FakeOpenRouterProvider.seen_completion_bodies
        upstream_body = _FakeOpenRouterProvider.seen_completion_bodies[0]
        assert upstream_body["model"] == "perplexity/sonar"
        assert upstream_body["usage"]["include"] is True
        assert _FakeOpenRouterProvider.generation_ids == ["gen-sonar-cost-1"]
        assert event is not None
        assert event["provider"] == "or"
        assert event["billable"] is True
        assert event["cost_usd"] == 0.0032
        assert event["cost_source"] == "openrouter_generation_reconciliation"
        assert event["tracking_failed"] is False
        assert event["model"] == "perplexity/sonar"
        assert event["prompt_tokens"] == 11
        assert event["completion_tokens"] == 5
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        upstream.shutdown()
        upstream.server_close()


def test_evidence_proxy_reconciles_openrouter_generation_cost_from_body_id():
    _FakeOpenRouterBodyIdProvider.generation_ids = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenRouterBodyIdProvider)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = None
    try:
        registry = [
            provider_evidence_proxy.ProviderRegistryEntry(
                id="or",
                base_url=f"http://127.0.0.1:{upstream.server_address[1]}",
                auth_kind="none",
            )
        ]
        proxy, _store, _proxy_thread = provider_evidence_proxy.serve_evidence_proxy(
            host="127.0.0.1",
            port=0,
            registry=registry,
        )
        request_body = json.dumps(
            {
                "model": "perplexity/sonar",
                "messages": [{"role": "user", "content": "redacted"}],
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_address[1]}/or/api/v1/chat/completions",
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "X-Research-Lab-Cost-Scope": "or-body-scope",
                "X-Research-Lab-Cost-Cap-Usd": "0.50",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            event = decode_cost_event_header(response.headers.get("X-Research-Lab-Provider-Cost-Event"))

        assert _FakeOpenRouterBodyIdProvider.generation_ids == ["gen-sonar-body-cost-1"]
        assert event is not None
        assert event["provider"] == "or"
        assert event["billable"] is True
        assert event["cost_usd"] == 0.0041
        assert event["cost_source"] == "openrouter_generation_reconciliation"
        assert event["tracking_failed"] is False
        assert event["generation_id"] == "gen-sonar-body-cost-1"
        assert event["prompt_tokens"] == 19
        assert event["completion_tokens"] == 6
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        upstream.shutdown()
        upstream.server_close()


def test_evidence_proxy_falls_back_to_perplexity_pricing_when_generation_has_tokens_no_cost():
    _FakeOpenRouterGenerationTokensProvider.generation_ids = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenRouterGenerationTokensProvider)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = None
    try:
        registry = [
            provider_evidence_proxy.ProviderRegistryEntry(
                id="or",
                base_url=f"http://127.0.0.1:{upstream.server_address[1]}",
                auth_kind="none",
            )
        ]
        proxy, _store, _proxy_thread = provider_evidence_proxy.serve_evidence_proxy(
            host="127.0.0.1",
            port=0,
            registry=registry,
        )
        request_body = json.dumps(
            {
                "model": "perplexity/sonar-pro",
                "messages": [{"role": "user", "content": "redacted"}],
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_address[1]}/or/api/v1/chat/completions",
            data=request_body,
            headers={
                "Content-Type": "application/json",
                "X-Research-Lab-Cost-Scope": "or-token-priced-scope",
                "X-Research-Lab-Cost-Cap-Usd": "0.50",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            event = decode_cost_event_header(response.headers.get("X-Research-Lab-Provider-Cost-Event"))

        assert _FakeOpenRouterGenerationTokensProvider.generation_ids == ["gen-sonar-token-priced-1"]
        assert event is not None
        assert event["provider"] == "or"
        assert event["billable"] is True
        assert event["cost_usd"] == 0.0155
        assert event["cost_source"] == "openrouter_perplexity_token_pricing_fallback"
        assert event["tracking_failed"] is False
        assert event["generation_id"] == "gen-sonar-token-priced-1"
        assert event["prompt_tokens"] == 1000
        assert event["completion_tokens"] == 500
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        upstream.shutdown()
        upstream.server_close()


def test_evidence_proxy_missing_openrouter_cost_zero_cost_does_not_block_later_calls():
    _FakeOpenRouterMissingCostProvider.calls = 0
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenRouterMissingCostProvider)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = None
    try:
        registry = [
            provider_evidence_proxy.ProviderRegistryEntry(
                id="or",
                base_url=f"http://127.0.0.1:{upstream.server_address[1]}",
                auth_kind="none",
            )
        ]
        proxy, _store, _proxy_thread = provider_evidence_proxy.serve_evidence_proxy(
            host="127.0.0.1",
            port=0,
            registry=registry,
        )
        url = f"http://127.0.0.1:{proxy.server_address[1]}/or/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "X-Research-Lab-Cost-Scope": "or-missing-cost-scope",
            "X-Research-Lab-Cost-Cap-Usd": "0.50",
        }
        first_req = urllib.request.Request(
            url,
            data=json.dumps(
                {
                    "model": "google/gemini-2.5-pro",
                    "messages": [{"role": "user", "content": "first"}],
                }
            ).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(first_req, timeout=5) as response:
            first_event = decode_cost_event_header(response.headers.get("X-Research-Lab-Provider-Cost-Event"))
        assert first_event is not None
        assert first_event["tracking_failed"] is False
        assert first_event["tracking_reason"] == "missing_openrouter_cost"
        assert first_event["cost_source"] == "openrouter_missing_cost_zero_cost"
        assert first_event["spent_after_usd"] == 0.0

        second_req = urllib.request.Request(
            url,
            data=json.dumps(
                {
                    "model": "google/gemini-2.5-pro",
                    "messages": [{"role": "user", "content": "second"}],
                }
            ).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(second_req, timeout=5) as response:
            second_event = decode_cost_event_header(response.headers.get("X-Research-Lab-Provider-Cost-Event"))

        assert second_event is not None
        assert second_event["tracking_failed"] is False
        assert second_event["cost_source"] == "openrouter_missing_cost_zero_cost"
        assert second_event["spent_before_usd"] == 0.0
        assert second_event["spent_after_usd"] == 0.0
        assert _FakeOpenRouterMissingCostProvider.calls == 2
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        upstream.shutdown()
        upstream.server_close()


def test_evidence_proxy_does_not_cache_nonterminal_exa_agent_poll():
    _FakeExaAgentProvider.calls = 0
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeExaAgentProvider)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    upstream_base = f"http://127.0.0.1:{upstream.server_address[1]}"
    proxy = None
    try:
        registry = [
            provider_evidence_proxy.ProviderRegistryEntry(
                id="exa",
                base_url=upstream_base,
                auth_kind="none",
            )
        ]
        proxy, store, _proxy_thread = provider_evidence_proxy.serve_evidence_proxy(
            host="127.0.0.1",
            port=0,
            registry=registry,
        )
        url = f"http://127.0.0.1:{proxy.server_address[1]}/exa/agent/runs/agent_run_123"
        req = urllib.request.Request(
            url,
            headers={
                "X-Research-Lab-Cost-Scope": "agent-scope",
                "X-Research-Lab-Cost-Cap-Usd": "0.50",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            first = json.loads(response.read().decode())
            first_event = decode_cost_event_header(response.headers.get("X-Research-Lab-Provider-Cost-Event"))
        assert first["status"] == "running"
        assert first_event["evidence"] == "live_unrecorded"
        assert first_event["billable"] is False
        assert not store.lookup(
            provider_evidence_proxy.canonical_request_fingerprint(
                "GET",
                upstream_base + "/agent/runs/agent_run_123",
                None,
            )
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            second = json.loads(response.read().decode())
            second_event = decode_cost_event_header(response.headers.get("X-Research-Lab-Provider-Cost-Event"))
        assert second["status"] == "completed"
        assert second_event["evidence"] == "recorded"
        assert second_event["billable"] is True
        assert second_event["cost_usd"] == 0.1
        assert _FakeExaAgentProvider.calls == 2
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        upstream.shutdown()
        upstream.server_close()


def test_evidence_store_ignores_preexisting_nonterminal_exa_agent_cache(tmp_path):
    fingerprint = "d" * 64
    cache_path = tmp_path / "day_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "utc_day": provider_evidence_proxy._utc_day(),
                "entries": {
                    fingerprint: {
                        "status": 200,
                        "body_b64": base64.b64encode(
                            b'{"object":"agent_run","id":"agent_run_123","status":"running"}'
                        ).decode("ascii"),
                        "outcome": "success",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    store = provider_evidence_proxy.EvidenceStore(day_cache_path=str(cache_path))

    assert store.lookup(fingerprint) is None
