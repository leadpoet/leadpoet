"""Credential-isolation regression checks; these are not live-run evidence."""

from dataclasses import replace
import json

import pytest

from lab_arena import broker as br
from test_lab_arena_broker import CHAT, CONTEXT, FakeLedgerStore, FakeTransport, make_broker


def test_miner_score_scrape_uses_only_deepline_and_keeps_requested_identity():
    source_url = "https://example.com/about"
    envelope = {
        "job_id": "test",
        "status": "completed",
        "result": {
            "data": {
                "rawHtml": "<!doctype html><head><title>Acme</title></head>",
                "metadata": {
                    "sourceURL": source_url,
                    "url": source_url,
                    "statusCode": 200,
                },
            }
        },
        "billing": {"cost_usd": 0.002},
    }
    providers = []
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(200, envelope)]),
        credential_for=lambda context, provider: (
            providers.append(provider) or "miner-deepline-key"
        ),
        funding_source_for=lambda context: "miner_key",
    )
    context = replace(CONTEXT, kind="score", round_id="arena-2026-09-04")
    result = broker.execute(
        context,
        operation_id="scrapingdog.scrape",
        parameters={"url": source_url},
        action_sequence=0,
        timeout_ms=60_000,
    )
    assert result.status == 200 and b"<title>Acme</title>" in result.body
    assert providers == ["deepline"]
    sent = transport.sent[0]
    assert sent["url"].endswith("/api/v2/integrations/firecrawl_scrape/execute")
    assert sent["headers"]["authorization"] == "Bearer miner-deepline-key"
    assert json.loads(sent["body"])["payload"]["formats"] == ["rawHtml"]
    assert result.call["operation_id"] == "scrapingdog.scrape"
    assert result.call["effective_operation_id"] == "deepline.execute"
    assert result.call["provider"] == "deepline"
    assert result.call["actual_microusd"] == 2000
    assert ledger.calls[result.call["call_identity"]]["provider"] == "deepline"


def test_miner_execution_does_not_fall_back_to_a_host_scrapingdog_key():
    providers = []

    def miner_credential(_context, provider):
        providers.append(provider)
        raise br.BrokerError("miner_provider_not_configured")

    broker, ledger, transport = make_broker(
        credential_for=miner_credential,
        funding_source_for=lambda context: "miner_key",
    )
    result = broker.execute(
        CONTEXT,
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=0,
        timeout_ms=5000,
    )
    assert result.status == 400
    assert result.call["error_code"] == "miner_provider_not_configured"
    assert providers == ["scrapingdog"]
    assert not ledger.calls and not transport.sent


def test_host_funded_score_keeps_the_direct_scrapingdog_route():
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(200, b"<html>baseline</html>")]),
        funding_source_for=lambda context: "host",
    )
    result = broker.execute(
        replace(CONTEXT, kind="score", round_id="arena-2026-09-04"),
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=0,
        timeout_ms=5000,
    )
    assert result.status == 200 and b"baseline" in result.body
    assert transport.sent[0]["url"].startswith(
        "https://api.scrapingdog.com/scrape?"
    )
    assert result.call["provider"] == "scrapingdog"
    assert ledger.calls[result.call["call_identity"]]["provider"] == "scrapingdog"


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_each_submission_pays_with_its_own_key(kind):
    keys = {"s1": "miner-one-runtime-key", "s2": "miner-two-runtime-key"}
    broker, ledger, transport = make_broker(
        credential_for=lambda context, provider: keys[context.submission_id],
        funding_source_for=lambda context: "miner_key",
        judge_models=[CHAT["model"]],
    )
    for index, submission_id in enumerate(keys):
        context = replace(CONTEXT, submission_id=submission_id, assignment_id=f"assignment-{index}", kind=kind)
        result = broker.execute(context, operation_id="openrouter.chat", parameters=CHAT, action_sequence=index, timeout_ms=5000)
        assert result.status == 200
        assert result.call["funding_source"] == "miner_key"
        assert transport.sent[-1]["headers"]["authorization"] == "Bearer " + keys[submission_id]
        assert not any(secret in repr(result.to_document()) for secret in keys.values())
    assert all(call["funding_source"] == "miner_key" for call in ledger.calls.values())


def test_missing_key_never_dispatches_or_uses_host_key():
    def unavailable(context, provider):
        raise br.BrokerError("miner_credentials_unavailable")

    broker, ledger, transport = make_broker(
        credential_for=unavailable,
        funding_source_for=lambda context: "miner_key",
    )
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert result.status == 402
    assert result.call["error_code"] == "miner_credentials_unavailable"
    assert result.call["funding_source"] == "miner_key"
    assert not ledger.calls and not transport.sent


@pytest.mark.parametrize("status", [401, 402, 403])
def test_miner_key_refusal_is_not_an_organizer_outage(status):
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(status, {"error": "refused"})]),
        credential_for=lambda context, provider: "miner-runtime-key",
        funding_source_for=lambda context: "miner_key",
    )
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert result.call["error_code"] == "miner_credentials_unavailable"
    assert result.call["provider_status"] == status
    assert ledger.log == ["reserve", "dispatch", "settle"]
    replay = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert replay.call["error_code"] == "miner_credentials_unavailable"
    assert replay.body == result.body and replay.status == result.status
    assert len(transport.sent) == 1


def test_provider_cannot_echo_runtime_key_into_output_or_storage():
    secret = "miner-runtime-key-never-publish"
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(200, {"content": secret})]),
        credential_for=lambda context, provider: secret,
        funding_source_for=lambda context: "miner_key",
    )
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert result.status == 502
    assert secret not in repr(result.to_document())
    assert secret not in repr(ledger.calls)
