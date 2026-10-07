"""Credential-isolation regression checks; these are not live-run evidence."""

from dataclasses import replace
import base64
import json
from urllib.parse import quote

import pytest

from lab_arena import broker as br
from lab_arena import operations
from test_lab_arena_broker import (
    CHAT, CONTEXT, FakeLedgerStore, FakeTransport, deepline_history,
    deepline_history_entry, make_broker,
)


def firecrawl_enrichment_denial(request_id="iad1::firecrawl-enrichment-denied"):
    return {
        "code": "PROVIDER_AUTHORIZATION_FAILED",
        "credential_owner": "deepline_managed",
        "credential_source": "env",
        "error": "private upstream denial",
        "error_category": "provider_auth",
        "failure_origin": "provider",
        "operation": "firecrawl_scrape",
        "provider": "firecrawl",
        "requestId": request_id,
        "request_id": request_id,
        "tool_error": {
            "code": "PROVIDER_AUTHORIZATION_FAILED",
            "category": "authentication",
            "origin": "provider",
            "statusCode": 403,
            "provider": "firecrawl",
            "operation": "firecrawl_scrape",
        },
        "upstream_error_code": "THIRD_PARTY_DATA_ENRICHMENT_NOT_ENABLED",
        "upstream_status": 403,
    }


def _firecrawl_failed_zero_history(request_id):
    entry = deepline_history_entry(
        request_id, "firecrawl_scrape", 0,
        charge_state="failed", provider="firecrawl",
    )
    entry.update({"status": "error", "delta": 0})
    return deepline_history(entry)


def test_miner_score_managed_firecrawl_enrichment_denial_is_request_refusal():
    request_id = "iad1::firecrawl-enrichment-denied"
    denial = firecrawl_enrichment_denial(request_id)
    marked = []
    broker, ledger, transport = make_broker(
        transport=FakeTransport([
            (403, denial), (200, _firecrawl_failed_zero_history(request_id)),
        ]),
        credential_for=lambda _context, _provider: "miner-deepline-key",
        funding_source_for=lambda _context: "miner_key",
        retry_miner_credential_for=lambda _context: True,
        mark_provider_fallback=lambda context, provider, evidence: (
            marked.append((context, provider, evidence)) or {"status": "marked"}
        ),
    )
    context = replace(CONTEXT, kind="score", round_id="arena-2026-10-01")
    arguments = dict(
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://www.linkedin.com/company/edvistasinc"},
        action_sequence=56,
        timeout_ms=60_000,
    )

    result = broker.execute(context, **arguments)
    replay = broker.execute(context, **arguments)

    assert result.status == 403
    assert json.loads(result.body) == {
        "error": {"code": "provider_request_refused"}
    }
    assert result.call["error_code"] == "provider_request_refused"
    assert result.call["provider_status"] == 403
    assert result.call["actual_microusd"] == 0
    assert result.call["cost_basis"] == "deepline_exact_request_credits_x_0.10_usd"
    assert result.call["operation_id"] == "scrapingdog.scrape"
    assert result.call["effective_operation_id"] == "deepline.execute"
    assert marked == []
    assert replay.call["idempotent"] is True
    assert replay.call["error_code"] == "provider_request_refused"
    assert replay.body == result.body
    assert [sent["method"] for sent in transport.sent] == ["POST", "GET"]
    terminal = ledger.calls[result.call["call_identity"]]["terminal"]
    assert "account_failure_evidence" not in terminal
    assert denial["error"] not in repr(result.to_document())
    assert denial["error"] not in repr(ledger.calls)


def test_miner_score_firecrawl_denial_without_verified_cost_stays_uncertain():
    denial = firecrawl_enrichment_denial()
    broker, ledger, _transport = make_broker(
        transport=FakeTransport([(403, denial), (200, deepline_history())]),
        credential_for=lambda _context, _provider: "miner-deepline-key",
        funding_source_for=lambda _context: "miner_key",
    )
    result = broker.execute(
        replace(CONTEXT, kind="score", round_id="arena-2026-10-01"),
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://www.linkedin.com/company/edvistasinc"},
        action_sequence=0,
        timeout_ms=60_000,
    )
    assert result.status == 403
    assert result.call["error_code"] == "provider_request_refused"
    assert result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call
    assert ledger.log == ["reserve", "dispatch", "uncertain"]


def test_miner_score_firecrawl_public_page_still_succeeds():
    envelope = {
        "job_id": "test-public-page",
        "status": "completed",
        "result": {"data": {
            "rawHtml": "<html>Example Domain</html>",
            "metadata": {
                "sourceURL": "https://example.com/",
                "url": "https://example.com/",
                "statusCode": 200,
            },
        }},
        "billing": {"credits_charged": 0},
    }
    broker, _ledger, transport = make_broker(
        transport=FakeTransport([(200, envelope)]),
        credential_for=lambda _context, _provider: "miner-deepline-key",
        funding_source_for=lambda _context: "miner_key",
    )
    result = broker.execute(
        replace(CONTEXT, kind="score", round_id="arena-2026-10-01"),
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://example.com/"},
        action_sequence=0,
        timeout_ms=60_000,
    )
    assert result.status == 200
    assert b"Example Domain" in result.body
    assert result.call.get("error_code") is None
    assert result.call["actual_microusd"] == 0
    assert len(transport.sent) == 1


@pytest.mark.parametrize(
    "status,owner,upstream_code,expected_status,expected_error",
    [
        (401, "deepline_managed", "THIRD_PARTY_DATA_ENRICHMENT_NOT_ENABLED",
         402, "miner_credentials_unavailable"),
        (402, "deepline_managed", "THIRD_PARTY_DATA_ENRICHMENT_NOT_ENABLED",
         402, "miner_credentials_unavailable"),
        (403, "workspace", "THIRD_PARTY_DATA_ENRICHMENT_NOT_ENABLED",
         402, "miner_credentials_unavailable"),
        (403, "deepline_managed", "OTHER", 502, "provider_unavailable"),
    ],
)
def test_miner_score_firecrawl_account_denials_preserve_account_ownership(
    status, owner, upstream_code, expected_status, expected_error
):
    request_id = "iad1::firecrawl-account-denied"
    denial = firecrawl_enrichment_denial(request_id)
    denial["credential_owner"] = owner
    denial["upstream_error_code"] = upstream_code
    broker, _ledger, _transport = make_broker(
        transport=FakeTransport([
            (status, denial), (200, _firecrawl_failed_zero_history(request_id)),
        ]),
        credential_for=lambda _context, _provider: "miner-deepline-key",
        funding_source_for=lambda _context: "miner_key",
    )
    result = broker.execute(
        replace(CONTEXT, kind="score", round_id="arena-2026-10-01"),
        operation_id="scrapingdog.scrape",
        parameters={"url": "https://www.linkedin.com/company/edvistasinc"},
        action_sequence=0,
        timeout_ms=60_000,
    )
    assert result.status == expected_status
    assert result.call["error_code"] == expected_error
    assert result.call["provider_status"] == status


@pytest.mark.parametrize("change", [
    {"code": "OTHER"},
    {"credential_owner": "workspace"},
    {"credential_source": "workspace"},
    {"error_category": "authorization"},
    {"failure_origin": "workspace"},
    {"provider": "generic_http"},
    {"operation": "other"},
    {"upstream_status": 401},
    {"upstream_error_code": "OTHER"},
    {"tool_error": {"code": "OTHER"}},
    {"tool_error": {"category": "authorization"}},
    {"tool_error": {"origin": "workspace"}},
    {"tool_error": {"statusCode": 401}},
    {"tool_error": {"provider": "generic_http"}},
    {"tool_error": {"operation": "other"}},
])
def test_firecrawl_enrichment_refusal_rejects_inconsistent_metadata(change):
    denial = firecrawl_enrichment_denial()
    if "tool_error" in change:
        denial["tool_error"].update(change["tool_error"])
    else:
        denial.update(change)
    response = br.ProviderResponse(
        403, {"content-type": "application/json"},
        json.dumps(denial).encode("utf-8"),
    )
    assert not br._provider_request_refused(
        "deepline", {"tool": "firecrawl_scrape"}, response,
    )


@pytest.mark.parametrize("body,tool", [
    (b"{not json", "firecrawl_scrape"),
    (b"[]", "firecrawl_scrape"),
    (b"{}", "generic_http_request"),
])
def test_firecrawl_enrichment_refusal_rejects_malformed_or_other_tool(body, tool):
    response = br.ProviderResponse(403, {"content-type": "application/json"}, body)
    assert not br._provider_request_refused("deepline", {"tool": tool}, response)


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
    assert set(terminal) == {
        "status", "headers", "body_b64", "call_succeeded", "provider_cost",
    }
    assert terminal["call_succeeded"] is True
    assert terminal["provider_cost"] == {
        "basis": "deepline_billing_credits_charged_x_0.10_usd",
        "units": "0.02",
        "unit_name": "credits",
        "operation": "firecrawl_scrape",
        "request_id": "test",
    }
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
    assert [call["method"] for call in transport.sent] == ["POST"]


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
    assert [call["method"] for call in transport.sent] == ["POST"]


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


def test_openrouter_legacy_moderation_403_is_not_a_miner_credential_failure():
    flagged_input = "private prompt that must not be persisted"
    payload = {
        "choices": [
            {
                "finish_reason": "error",
                "error": {
                    "code": 403,
                    "message": "request rejected by moderation",
                    "metadata": {
                        "reasons": ["policy"],
                        "flagged_input": flagged_input,
                        "provider_name": "Upstream Provider",
                        "model_slug": "provider/model",
                    },
                },
            }
        ]
    }
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(200, payload)]),
        credential_for=lambda _context, _provider: "miner-runtime-key",
        funding_source_for=lambda _context: "miner_key",
    )
    arguments = dict(
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=5000,
    )

    result = broker.execute(CONTEXT, **arguments)
    replay = broker.execute(CONTEXT, **arguments)

    assert result.status == 403
    assert json.loads(result.body) == {
        "error": {"code": "provider_request_refused"}
    }
    assert result.call["error_code"] == "provider_request_refused"
    assert result.call["provider_status"] == 403
    assert replay.call["idempotent"] is True
    assert replay.call["error_code"] == "provider_request_refused"
    assert replay.call["provider_status"] == 403
    assert replay.body == result.body and replay.status == result.status
    assert len(transport.sent) == 1
    assert flagged_input not in repr(result.to_document())
    assert flagged_input not in repr(ledger.calls)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "error": {
                "code": 403,
                "metadata": {"provider_name": "Upstream Provider"},
            }
        },
        {
            "error": {
                "code": 403,
                "metadata": {"error_type": "refusal"},
            },
            "choices": [
                {
                    "finish_reason": "error",
                    "error": {"code": 403, "message": "account forbidden"},
                }
            ],
        },
        {
            "error": {
                "code": 403,
                "metadata": {
                    "error_type": "authentication",
                    "reasons": ["policy"],
                    "flagged_input": "input",
                    "provider_name": "Upstream Provider",
                    "model_slug": "provider/model",
                },
            }
        },
        {
            "error_type": "permission_denied",
            "error": {
                "code": 403,
                "metadata": {"patterns": ["blocked-pattern"]},
            },
        },
        {
            "error_type": "content_policy_violation",
            "error": {"code": "invalid_prompt"},
        },
        {
            "error": {
                "code": 403,
                "metadata": {"patterns": []},
            }
        },
        {
            "error_type": {"unexpected": "mapping"},
            "error": {"code": 403},
        },
    ],
    ids=(
        "provider_name_alone",
        "mixed_policy_and_account_errors",
        "typed_authentication_overrides_legacy_shape",
        "top_level_permission_overrides_patterns",
        "incoherent_responses_type_and_code",
        "empty_guardrail_patterns",
        "non_string_top_level_error_type",
    ),
)
def test_openrouter_ambiguous_403_errors_remain_miner_credential_failures(
    payload,
):
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(403, payload)]),
        credential_for=lambda _context, _provider: "miner-runtime-key",
        funding_source_for=lambda _context: "miner_key",
    )

    result = broker.execute(
        CONTEXT,
        operation_id="openrouter.chat",
        parameters=CHAT,
        action_sequence=0,
        timeout_ms=5000,
    )

    assert result.status == 402
    assert result.call["error_code"] == "miner_credentials_unavailable"
    assert ledger.log == ["reserve", "dispatch", "settle"]
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
    if operation_id == "openrouter.chat":
        assert [sent["method"] for sent in transport.sent] == ["POST"]
    else:
        # Recovery may read the existing execution by its safe key, never repost it.
        assert [sent["method"] for sent in transport.sent] == ["POST", "GET", "GET"]
        assert transport.sent[1]["url"].startswith(br.DEEPLINE_EXECUTION_BY_KEY_URL)
        assert transport.sent[2]["url"] == transport.sent[1]["url"]
    assert all(secret not in sent["url"] and secret.encode() not in sent["body"]
               for sent in transport.sent)
    assert secret not in repr(ledger.calls)


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
