"""Credential-isolation regression checks; these are not live-run evidence."""

from dataclasses import replace
import base64
import json
from urllib.parse import quote

import pytest

from lab_arena import broker as br
from lab_arena import operations
from test_lab_arena_broker import CHAT, CONTEXT, FakeLedgerStore, FakeTransport, make_broker


def test_miner_score_scrape_preserves_only_the_validated_final_url_on_replay():
    source_url = "https://example.com/about"
    final_url = "https://www.example.net/about"
    envelope = {
        "job_id": "test",
        "status": "completed",
        "result": {
            "data": {
                "rawHtml": "<!doctype html><head><title>Acme</title></head>",
                "metadata": {
                    "sourceURL": source_url,
                    "url": final_url,
                    "statusCode": 200,
                },
            }
        },
        "billing": {"credits_charged": 0.02, "cost_usd": 0.002},
    }
    providers = []
    class HeaderInjectionTransport(FakeTransport):
        def send(self, **kwargs):
            response = super().send(**kwargs)
            return br.ProviderResponse(
                response.status,
                {
                    **response.headers,
                    operations.TRUSTED_RESPONSE_URL_HEADER: "https://attacker.example/",
                },
                response.body,
            )

    broker, ledger, transport = make_broker(
        transport=HeaderInjectionTransport([(200, envelope)]),
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
    assert result.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == final_url
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
    terminal = ledger.calls[result.call["call_identity"]]["terminal"]
    assert set(terminal) == {"status", "headers", "body_b64"}
    assert terminal["headers"][operations.TRUSTED_RESPONSE_URL_HEADER] == final_url
    assert "attacker.example" not in repr(result.to_document())
    assert "attacker.example" not in repr(terminal)

    replay = broker.execute(
        context,
        operation_id="scrapingdog.scrape",
        parameters={"url": source_url},
        action_sequence=0,
        timeout_ms=60_000,
    )
    assert replay.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == final_url
    assert replay.call["idempotent"] is True
    assert [call["method"] for call in transport.sent] == ["POST", "GET"]
    assert transport.sent[1]["url"] == br.DEEPLINE_BILLING_HISTORY_URL


@pytest.mark.parametrize("percent_encoded", [False, True])
def test_miner_score_final_url_cannot_expose_its_runtime_key(percent_encoded):
    secret = "miner+deepline/key=never-publish"
    exposed = quote(secret, safe="") if percent_encoded else secret
    source_url = "https://example.com/about"
    envelope = {
        "job_id": "test",
        "status": "completed",
        "result": {
            "data": {
                "rawHtml": "<html><title>Example Company</title></html>",
                "metadata": {
                    "sourceURL": source_url,
                    "url": f"https://example.net/about?token={exposed}",
                    "statusCode": 200,
                },
            }
        },
        "billing": {"credits_charged": 0.02, "cost_usd": 0.002},
    }
    broker, ledger, _transport = make_broker(
        transport=FakeTransport([(200, envelope)]),
        credential_for=lambda _context, _provider: secret,
        funding_source_for=lambda _context: "miner_key",
    )

    result = broker.execute(
        replace(CONTEXT, kind="score", round_id="arena-2026-09-04"),
        operation_id="scrapingdog.scrape",
        parameters={"url": source_url},
        action_sequence=0,
        timeout_ms=60_000,
    )

    assert result.status == 502
    assert operations.TRUSTED_RESPONSE_URL_HEADER not in result.headers
    assert secret not in repr(result.to_document())
    assert secret not in repr(ledger.calls)
    assert exposed not in repr(ledger.calls)


@pytest.mark.parametrize("corruption", ("empty", "null", "secret"))
def test_cached_final_url_fails_closed_when_invalid_or_secret_bearing(corruption):
    secret = "miner+deepline/key=never-publish"
    source_url = "https://example.com/about"
    envelope = {
        "job_id": "test",
        "status": "completed",
        "result": {
            "data": {
                "rawHtml": "<html><title>Example Company</title></html>",
                "metadata": {
                    "sourceURL": source_url,
                    "url": "https://example.net/about",
                    "statusCode": 200,
                },
            }
        },
        "billing": {"credits_charged": 0.02, "cost_usd": 0.002},
    }
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(200, envelope)]),
        credential_for=lambda _context, _provider: secret,
        funding_source_for=lambda _context: "miner_key",
    )
    context = replace(CONTEXT, kind="score", round_id="arena-2026-09-04")
    args = {
        "operation_id": "scrapingdog.scrape",
        "parameters": {"url": source_url},
        "action_sequence": 0,
        "timeout_ms": 60_000,
    }
    first = broker.execute(context, **args)
    terminal = ledger.calls[first.call["call_identity"]]["terminal"]
    corrupted_url = {
        "empty": "",
        "null": None,
        "secret": "https://example.net/about?token=" + quote(secret, safe=""),
    }[corruption]
    terminal["headers"][operations.TRUSTED_RESPONSE_URL_HEADER] = corrupted_url

    replay = broker.execute(context, **args)

    assert replay.status == 503
    assert json.loads(replay.body) == {"error": {"code": "broker_unavailable"}}
    assert operations.TRUSTED_RESPONSE_URL_HEADER not in replay.headers
    assert secret not in repr(replay.to_document())
    assert [call["method"] for call in transport.sent] == ["POST", "GET"]
    assert transport.sent[1]["url"] == br.DEEPLINE_BILLING_HISTORY_URL


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
    assert operations.TRUSTED_RESPONSE_URL_HEADER not in result.headers

    terminal = ledger.calls[result.call["call_identity"]]["terminal"]
    terminal["headers"][operations.TRUSTED_RESPONSE_URL_HEADER] = (
        "https://attacker.example/"
    )
    replay = broker.execute(
        replace(CONTEXT, kind="score", round_id="arena-2026-09-04"),
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/about"},
        action_sequence=0,
        timeout_ms=5000,
    )
    assert replay.status == 503
    assert json.loads(replay.body) == {"error": {"code": "broker_unavailable"}}


@pytest.mark.parametrize("kind", ["execute", "score"])
def test_each_submission_pays_with_its_own_key(kind):
    keys = {"s1": "miner-one-runtime-key", "s2": "miner-two-runtime-key"}
    provider_response = {"choices": [], "usage": {"cost": "0.00000345"}}
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(200, provider_response), (200, provider_response)]),
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


@pytest.mark.parametrize("operation_id,parameters", [
    ("openrouter.chat", CHAT),
    ("deepline.execute", {"tool": "exa_search", "payload": {"query": "Acme"}}),
    ("exa.search", {"query": "Acme"}),
])
@pytest.mark.parametrize("location", ["value", "key", "duplicate_member"])
@pytest.mark.parametrize("percent_encoded", [False, True])
def test_json_escaped_credential_echo_is_blocked_before_storage(operation_id, parameters, location, percent_encoded):
    secret = "synthetic-arena-runtime-key-0123456789"
    echoed = "".join("%%%02x" % ord(character) for character in secret) if percent_encoded else secret
    escaped = "".join("\\u%04x" % ord(character) for character in echoed)
    inner = ('{"nested":["prefix ' + escaped + ' suffix"]}' if location == "value"
             else '{"' + escaped + '":"value"}')
    if location == "duplicate_member":
        inner = '{"value":"' + escaped + '","value":"safe"}'
    body = ('{"status":"completed","result":{"data":' + inner + '}}').encode()
    assert secret.encode() not in body
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(200, body)]),
        credential_for=lambda context, provider: secret,
        funding_source_for=lambda context: "miner_key",
    )
    args = dict(operation_id=operation_id, parameters=parameters, action_sequence=0, timeout_ms=5000)
    result = broker.execute(CONTEXT, **args)
    assert result.status == 502
    assert secret not in str(json.loads(result.body))
    call = ledger.calls[result.call["call_identity"]]
    assert call["kind"] == "uncertain" and "terminal" not in call
    assert secret not in repr(call)
    replay = broker.execute(CONTEXT, **args)
    assert replay.status == 409 and json.loads(replay.body) == {"error": {"code": "call_uncertain"}}
    assert len(transport.sent) == 1


def test_valid_json_escapes_without_a_credential_keep_the_response():
    body = b'{"choices":[{"message":{"content":"Acme\\u0020Inc"}}],"usage":{"cost":"0.00000345"}}'
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(200, body)]),
        credential_for=lambda context, provider: "synthetic-arena-runtime-key",
        funding_source_for=lambda context: "miner_key",
    )
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert result.status == 200 and result.body == body


def test_provider_credential_echo_in_a_header_is_blocked():
    secret = "synthetic-arena-runtime-key"

    class HeaderEchoTransport:
        def send(self, **kwargs):
            return br.ProviderResponse(200, {"content-type": "text/" + secret}, b"safe")

    broker, ledger, _ = make_broker(
        transport=HeaderEchoTransport(),
        credential_for=lambda context, provider: secret,
    )
    result = broker.execute(CONTEXT, operation_id="scrapingdog.scrape", parameters={"url": "https://example.com"}, action_sequence=0, timeout_ms=5000)
    assert result.status == 502
    assert secret not in repr(result.to_document())
    assert secret not in repr(ledger.calls)


def test_json_too_deep_to_inspect_is_uncertain_without_fake_cost():
    body = b"[" * 2000 + b"0" + b"]" * 2000
    broker, ledger, _ = make_broker(transport=FakeTransport([(200, body)]))
    result = broker.execute(CONTEXT, operation_id="openrouter.chat", parameters=CHAT, action_sequence=0, timeout_ms=5000)
    assert result.status == 502
    assert result.call["outcome"] == "uncertain"
    assert ledger.log == ["reserve", "dispatch", "uncertain"]
