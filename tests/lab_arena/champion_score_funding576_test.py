"""Promoted score routes keep the effective provider's immutable payer."""

from dataclasses import replace

import pytest

from tests.lab_arena.firecrawl_diagnostic_adapter_test import URL, _envelope, _score_context
from tests.lab_arena.test_lab_arena_broker import DL_KEY, make_broker
from tests.lab_arena.test_submission_runtime import resolver


@pytest.mark.parametrize("flagged", [(), ("deepline",), ("scrapingdog",),
                                    ("openrouter",), ("deepline", "scrapingdog")])
def test_verified_fallback_keeps_route_and_bills_only_effective_provider(flagged):
    selected = []
    eligibility = []

    def funding(_context, provider):
        selected.append(provider)
        return "host" if provider in flagged else "miner_key"

    def credential(_context, provider):
        assert provider == "deepline"
        return DL_KEY

    broker, ledger, transport = make_broker(
        provider_funding_source_for=funding, credential_for=credential,
        champion_score_for=lambda context: eligibility.append(context.kind) or True,
    )
    transport.responses = [(200, _envelope("<html>Company source.</html>"))]
    result = broker.execute(_score_context(), operation_id="scrapingdog.scrape",
                            parameters={"url": URL}, action_sequence=0, timeout_ms=30_000)
    assert result.status == 200
    assert result.call["provider"] == "deepline"
    assert result.call["adapter"] == "firecrawl_raw_html"
    expected = "host" if "deepline" in flagged else "miner_key"
    assert result.call["funding_source"] == expected
    assert selected == ["scrapingdog", "deepline"]
    assert eligibility == (["score"] if "scrapingdog" in flagged else [])
    call = next(iter(ledger.calls.values()))
    assert call["funding_source"] == expected
    assert call["provider"] == "deepline"
    assert call["actual"] == 2_000
    replay = broker.execute(_score_context(), operation_id="scrapingdog.scrape",
                            parameters={"url": URL}, action_sequence=0, timeout_ms=30_000)
    assert replay.status == 200 and replay.call["idempotent"] is True
    assert len(transport.sent) == len(ledger.calls) == 1


@pytest.mark.parametrize("kind", ["execute", "score"])
@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("champion", [False, True])
def test_route_eligibility_requires_database_champion_score(kind, baseline, champion):
    keys, context, _ = resolver(baseline=baseline)
    context = replace(context, kind=kind)
    keys._store.funding = {
        "status": "available", "funding_source": "host", "champion_funding": champion,
        "credential_submission_id": None, "credential_miner_hotkey": None,
        "restart_required": False,
    }
    # Database champion identity is authoritative; model input cannot set it.
    assert keys.champion_score_for(context) is (kind == "score" and baseline and champion)


def test_unpromoted_host_score_keeps_native_route():
    broker, _, transport = make_broker(
        provider_funding_source_for=lambda *_: "host",
        champion_score_for=lambda _: False,
    )
    transport.responses = [(200, {"body": "native"})]
    result = broker.execute(_score_context(), operation_id="scrapingdog.scrape",
                            parameters={"url": URL}, action_sequence=0, timeout_ms=30_000)
    assert result.status == 200
    assert result.call["provider"] == "scrapingdog"
    assert "adapter" not in result.call
