from __future__ import annotations

import base64
import json
import threading
import urllib.request
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gateway.research_lab import provider_evidence_proxy
from research_lab.eval.provider_costs import (
    DEFAULT_SCRAPINGDOG_COST_PER_CREDIT_USD,
    ProviderCostEstimate,
    ProviderCostLedger,
    decode_cost_event_header,
    estimate_provider_cost,
    extract_exa_cost_dollars,
    extract_openrouter_cost_dollars,
    openrouter_generation_id,
    scrapingdog_credits_for_path,
    scrapingdog_credits_for_url,
    summarize_provider_cost_trace_entries,
)


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


def test_ledger_preserves_prior_tracking_failure_reason_for_later_blocks():
    ledger = ProviderCostLedger(scope="scope-tracking", cap_usd=Decimal("0.50"))
    failed = ledger.record_live_event(
        provider="or",
        request_fingerprint="d" * 64,
        status_code=200,
        estimate=ProviderCostEstimate(
            provider="or",
            endpoint="/api/v1/chat/completions",
            model="perplexity/sonar",
            tracking_failed=True,
            tracking_reason="missing_openrouter_cost",
        ),
    )
    assert failed.tracking_failed
    assert ledger.should_block_paid_call()
    assert ledger.block_reason() == "missing_openrouter_cost"

    blocked = ledger.block_event(
        provider="exa",
        endpoint="/search",
        request_fingerprint="e" * 64,
        reason=ledger.block_reason(),
    )
    doc = blocked.to_doc()
    assert doc["tracking_failed"]
    assert doc["tracking_reason"] == "missing_openrouter_cost"


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
