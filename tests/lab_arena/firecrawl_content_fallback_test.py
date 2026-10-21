"""One source-bound rescue keeps every ledger charge and the caller deadline."""

import copy
import json
from types import SimpleNamespace

import pytest

from lab_arena import broker as br, operations
from tests.lab_arena.firecrawl_diagnostic_adapter_test import (
    DIAGNOSTIC, REDACTED, URL, _envelope, _score_context,
)
from tests.lab_arena.test_lab_arena_broker import (
    DL_KEY, FakeLedgerStore, FakeTransport, make_broker,
)


HTML = "<html><body>Acme provides an authoritative company page.</body></html>"
FINAL_URL = "https://example.com/company/about"


def _generic(*, html=HTML, credits=0, final_url=FINAL_URL):
    return {
        "job_id": "public-execute:generic-source-recovery",
        "status": "completed",
        "toolResponse": {"view": "rawV2", "rawV2": {
            "provider": "generic_http", "operation": "generic_http_request",
            "method": "GET", "requested_url": URL, "final_url": final_url,
            "status_code": 200, "ok": True,
            "headers": {"content-type": "text/html; charset=utf-8"}, "data": html,
        }},
        "billing": {"credits_charged": credits, "cost_usd": credits / 10},
    }


def _broker(responses, *, store=None, transport=None, **kwargs):
    return make_broker(
        store=store, transport=transport or FakeTransport(responses),
        credential_for=lambda *_: DL_KEY, funding_source_for=lambda *_: "miner_key",
        **kwargs,
    )


def _execute(broker, *, action_sequence=0):
    return broker.execute(_score_context(), operation_id="scrapingdog.scrape",
                          parameters={"url": URL}, action_sequence=action_sequence,
                          timeout_ms=30_000)


@pytest.mark.parametrize("placeholder", [DIAGNOSTIC, REDACTED, "", " \n "])
@pytest.mark.parametrize("credits", [0, 0.01])
def test_source_rescue_preserves_charges_identity_and_durable_replay(placeholder, credits):
    broker, ledger, transport = _broker([(200, _envelope(placeholder)), (200, _generic(credits=credits))])
    result = _execute(broker)
    assert result.status == 200 and result.body == HTML.encode()
    assert result.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == FINAL_URL
    assert len(transport.sent) == 2
    primary, fallback = list(ledger.calls.values())
    assert primary["kind"] == fallback["kind"] == "settlement"
    assert primary["actual"] == 2_000
    assert fallback["actual"] == int(credits * 100_000)
    assert primary["terminal"]["compatibility_failure_code"] == "firecrawl_source_content_unavailable"
    assert primary["terminal"]["call_succeeded"] is False
    assert fallback["terminal"]["call_succeeded"] is True
    assert primary["identity"] != fallback["identity"]
    assert primary["operation_id"] == fallback["operation_id"] == "scrapingdog.scrape"
    assert primary["call_doc"]["request_hash"] == fallback["call_doc"]["request_hash"]
    assert primary["call_doc"]["base_call_identity"] == fallback["call_doc"]["base_call_identity"]
    assert [primary["call_doc"]["provider_attempt"], fallback["call_doc"]["provider_attempt"]] == [1, 2]
    assert primary["call_doc"]["tool"] == "firecrawl_scrape"
    assert fallback["call_doc"]["tool"] == "generic_http_request"
    assert "x-deepline-execute-response-contract" not in transport.sent[0]["headers"]
    assert transport.sent[1]["headers"]["x-deepline-execute-response-contract"] == "raw-v2"
    assert json.loads(transport.sent[1]["body"])["payload"]["url"] == URL
    for _ in range(2):
        replay = _execute(broker)
        assert replay.status == 200 and replay.body == result.body
        assert replay.call["call_identity"] == result.call["call_identity"]
        assert replay.call["idempotent"] is True
    assert len(transport.sent) == ledger.log.count("settle") == 2
    assert sum(call["actual"] for call in ledger.calls.values()) == 2_000 + int(credits * 100_000)


def test_settled_primary_replay_retains_exact_trigger_before_rescue():
    broker, ledger, transport = _broker([(200, _envelope(REDACTED)), (200, _generic())])
    primary = broker._execute_once(
        _score_context(), operation_id="scrapingdog.scrape", parameters={"url": URL},
        action_sequence=0, timeout_ms=30_000, provider_attempt=1,
        champion_credential_retry=False,
    )
    replay = broker._execute_once(
        _score_context(), operation_id="scrapingdog.scrape", parameters={"url": URL},
        action_sequence=0, timeout_ms=30_000, provider_attempt=1,
        champion_credential_retry=False,
    )
    assert replay.call["compatibility_failure_code"] == primary.call["compatibility_failure_code"]
    assert replay.call["actual_microusd"] == 2_000
    assert len(transport.sent) == 1
    result = _execute(broker)
    assert result.status == 200 and len(transport.sent) == 2
    assert ledger.log.count("settle") == 2


@pytest.mark.parametrize("target_status", [401, 402, 403, 407, 429, 500, 502, 503, 504, 599])
def test_authenticated_failed_firecrawl_target_uses_one_source_rescue(target_status):
    payload = _envelope("<p>Target access failed.</p>", status=target_status)
    broker, ledger, transport = _broker([(200, payload), (200, _generic())])
    result = _execute(broker)
    assert result.status == 200 and result.body == HTML.encode()
    assert len(transport.sent) == len(ledger.calls) == 2
    primary, fallback = list(ledger.calls.values())
    assert primary["terminal"]["call_succeeded"] is False
    assert primary["terminal"]["compatibility_failure_code"] == "firecrawl_source_content_unavailable"
    assert primary["terminal"]["headers"][operations.TRUSTED_RESPONSE_URL_HEADER] == URL
    assert primary["actual"] == 2_000 and fallback["actual"] == 0
    assert json.loads(transport.sent[1]["body"])["payload"]["url"] == URL
    replay = _execute(broker)
    assert replay.status == 200 and replay.body == result.body
    assert replay.call["idempotent"] is True
    assert len(transport.sent) == ledger.log.count("settle") == 2


@pytest.mark.parametrize("target_status", [403, 429, 503])
@pytest.mark.parametrize("invalid", ["source_mismatch", "missing_status", "invalid_final", "missing_html"])
def test_failed_target_requires_complete_valid_source_proof(target_status, invalid):
    payload = _envelope("<p>Target access failed.</p>", status=target_status)
    data = payload["result"]["data"]
    if invalid == "source_mismatch":
        data["metadata"]["sourceURL"] = "https://wrong.example/"
    elif invalid == "missing_status":
        del data["metadata"]["statusCode"]
    elif invalid == "invalid_final":
        data["metadata"]["url"] = "https://127.0.0.1/"
    else:
        del data["rawHtml"]
    broker, ledger, transport = _broker([(200, payload)])
    result = _execute(broker)
    assert result.status == 502 and len(transport.sent) == len(ledger.calls) == 1
    assert result.call.get("compatibility_failure_code") != "firecrawl_source_content_unavailable"


@pytest.mark.parametrize("target_status", [404, 410])
def test_firecrawl_source_absence_never_opens_a_rescue(target_status):
    broker, ledger, transport = _broker([(200, _envelope("", status=target_status))])
    result = _execute(broker)
    assert result.status == target_status
    assert len(transport.sent) == len(ledger.calls) == 1
    assert "compatibility_failure_code" not in result.call


@pytest.mark.parametrize("mode", ["success", "url_mismatch", "missing_status", "authentication", "uncertain"])
def test_other_results_do_not_open_a_rescue(mode):
    payload = _envelope("<p>A valid source.</p>")
    status = 200
    if mode == "url_mismatch":
        payload = _envelope(REDACTED, source_url="https://wrong.example/")
    elif mode == "missing_status":
        payload = _envelope(REDACTED)
        del payload["result"]["data"]["metadata"]["statusCode"]
    elif mode == "authentication":
        status, payload = 401, {"error": "invalid authentication"}
    elif mode == "uncertain":
        payload = _envelope(REDACTED)
        payload["billing"] = {"pricing_status": "pending"}
    broker, ledger, transport = _broker([(status, payload)])
    result = _execute(broker)
    assert len([request for request in transport.sent if request["method"] == "POST"]) == 1
    assert len(ledger.calls) == 1
    if mode == "uncertain":
        assert result.call["outcome"] == "uncertain"
        assert "compatibility_failure_code" not in result.call


@pytest.mark.parametrize("failure", ["placeholder", "authentication"])
def test_rescue_never_opens_another_transport_or_credential_retry(failure):
    response = (200, _generic(html=REDACTED)) if failure == "placeholder" else (401, {"error": "invalid authentication"})
    broker, ledger, transport = _broker(
        [(200, _envelope(REDACTED)), response],
        retry_miner_credential_for=lambda *_: True,
    )
    result = _execute(broker)
    assert result.status != 200
    assert len([request for request in transport.sent if request["method"] == "POST"]) == 2
    assert len(ledger.calls) == 2


def test_existing_provider_call_quota_still_blocks_rescue():
    broker, ledger, transport = _broker(
        [(200, _envelope(REDACTED)), (200, _generic())],
        store=FakeLedgerStore(per_icp_quota=1),
    )
    result = _execute(broker)
    assert result.status == 402
    assert len(transport.sent) == 1
    assert len(ledger.calls) == 2
    assert list(ledger.calls.values())[-1]["kind"] == "refusal"


@pytest.mark.parametrize("elapsed", [15.0, 31.0])
def test_rescue_uses_only_remaining_original_deadline(monkeypatch, elapsed):
    clock = SimpleNamespace(now=0.0)

    class SlowPrimary(FakeTransport):
        def send(self, **request):
            result = super().send(**request)
            if len(self.sent) == 1:
                clock.now = elapsed
            return result

    monkeypatch.setattr(br, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=lambda *_: None))
    transport = SlowPrimary([(200, _envelope(REDACTED)), (200, _generic())])
    broker, ledger, _ = _broker([], transport=transport)
    result = _execute(broker)
    if elapsed > 30.0:
        assert result.status == 502 and len(transport.sent) == 1
        assert len(ledger.calls) == 1
    else:
        assert result.status == 200 and len(transport.sent) == 2
        assert transport.sent[1]["timeout"] <= 30.0 - elapsed
        assert json.loads(transport.sent[1]["body"])["payload"]["timeout_ms"] <= 15_000


@pytest.mark.parametrize("value", [None, "unknown_future_reason", [], 1])
def test_unknown_or_malformed_durable_trigger_fails_closed(value):
    broker, ledger, transport = _broker([(200, _envelope(REDACTED))])
    primary = broker._execute_once(
        _score_context(), operation_id="scrapingdog.scrape", parameters={"url": URL},
        action_sequence=0, timeout_ms=30_000, provider_attempt=1,
        champion_credential_retry=False,
    )
    saved = ledger.calls[primary.call["call_identity"]]
    saved["terminal"] = copy.deepcopy(saved["terminal"])
    saved["terminal"]["compatibility_failure_code"] = value
    result = _execute(broker)
    assert result.call["error_code"] == "broker_unavailable"
    assert len(transport.sent) == 1


@pytest.mark.parametrize("target_status", [
    100, 199, 301, 302, 304, 307, 308, 400, 401, 403, 408, 422, 429, 451, 500,
])
def test_retryable_generic_target_failure_keeps_existing_exa_fallback_status(target_status):
    payload = _generic()
    payload["toolResponse"]["rawV2"].update(status_code=target_status, ok=False)
    broker, ledger, transport = _broker([(200, _envelope(REDACTED)), (200, payload)])
    result = _execute(broker)
    assert result.status == 502
    assert result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert result.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == FINAL_URL
    assert len(transport.sent) == len(ledger.calls) == 2
    assert sum(call["actual"] for call in ledger.calls.values()) == 2_000


@pytest.mark.parametrize("target_status", [404, 410])
def test_generic_target_source_absence_stays_terminal_and_replays(target_status):
    payload = _generic()
    payload["toolResponse"]["rawV2"].update(status_code=target_status, ok=False)
    broker, ledger, transport = _broker([(200, _envelope(REDACTED)), (200, payload)])
    result = _execute(broker)
    assert result.status == target_status and result.body == b""
    assert result.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == FINAL_URL
    replay = _execute(broker)
    assert replay.status == target_status and replay.body == b""
    assert replay.call["idempotent"] is True
    assert len(transport.sent) == len(ledger.calls) == 2
    assert sum(call["actual"] for call in ledger.calls.values()) == 2_000


@pytest.mark.parametrize("lookup_finishes", [21.0, 31.0])
def test_recovery_keeps_exact_payload_and_shared_deadline(monkeypatch, lookup_finishes):
    clock = SimpleNamespace(now=0.0)

    class RecoveryStore(FakeLedgerStore):
        recovering = False

        def reserve_call(self, **kwargs):
            state = super().reserve_call(**kwargs)
            if self.recovering and kwargs["call_doc"].get("tool") == "generic_http_request":
                clock.now = 20.0
            return state

        def recover_deepline_response(self, **kwargs):
            return self.settle_call(**{key: kwargs[key] for key in (
                "run_id", "lease_token_hash", "call_identity", "actual_microusd",
                "terminal_response", "lease_ttl_seconds",
            )})

    ledger = RecoveryStore()
    broker, _, original_transport = _broker(
        [(200, _envelope(REDACTED)), (200, _generic())], store=ledger,
    )
    assert _execute(broker).status == 200
    fallback = next(call for call in ledger.calls.values()
                    if call["call_doc"].get("tool") == "generic_http_request")
    # Model a previously dispatched request whose response was lost. Its
    # original immutable reservation and execution key still own the work.
    fallback["kind"] = "dispatch"
    fallback.pop("terminal")
    ledger.recovering = True
    recovery_requests = []

    class RunningLookup:
        def send(self, **request):
            recovery_requests.append(request)
            if request["method"] == "POST":
                assert lookup_finishes < 30.0, "expired recovery must never POST"
                assert request["body"] == original_transport.sent[1]["body"]
                assert request["timeout_seconds"] <= 30.0 - lookup_finishes
                return br.ProviderResponse(200, {}, json.dumps(_generic()).encode())
            assert request["method"] == "GET" and request["timeout_seconds"] <= 10.0
            clock.now = lookup_finishes
            return br.ProviderResponse(200, {"x-deepline-idempotency-supported": "true"}, json.dumps({
                "requestId": _generic()["job_id"],
                "toolId": "generic_http_request",
                "executionRecovery": {
                    "idempotencyKey": fallback["call_doc"]["deepline_execution_key"],
                    "state": "running",
                },
            }).encode())

    broker._transport = RunningLookup()
    monkeypatch.setattr(br, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=lambda *_: None))
    result = _execute(broker)
    if lookup_finishes > 30.0:
        assert result.status == 409 and result.call["outcome"] == "uncertain"
        assert len(recovery_requests) == 1
    else:
        assert result.status == 200 and result.body == HTML.encode()
        assert len(recovery_requests) == 2
    assert len(original_transport.sent) == 2


def test_already_admitted_fallback_keeps_minimum_bound_after_slow_reserve(monkeypatch):
    clock = SimpleNamespace(now=0.0)

    class SlowReserve(FakeLedgerStore):
        def reserve_call(self, **kwargs):
            state = super().reserve_call(**kwargs)
            if kwargs["call_doc"].get("tool") == "generic_http_request":
                clock.now = 31.0
            return state

    monkeypatch.setattr(br, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=lambda *_: None))
    broker, ledger, transport = _broker(
        [(200, _envelope(REDACTED)), (200, _generic())], store=SlowReserve(),
    )
    assert _execute(broker).status == 200
    assert len(transport.sent) == 2
    assert transport.sent[1]["timeout"] == 0.001
    assert ledger.log.count("settle") == 2


@pytest.mark.parametrize("saved_timeout", [None, False, 0, 60_001, "1000"])
def test_malformed_saved_fallback_timeout_cannot_dispatch(saved_timeout):
    broker, ledger, transport = _broker([(200, _envelope(REDACTED)), (200, _generic())])
    assert _execute(broker).status == 200
    fallback = next(call for call in ledger.calls.values()
                    if call["call_doc"].get("tool") == "generic_http_request")
    fallback["call_doc"]["source_content_fallback_timeout_ms"] = saved_timeout
    assert _execute(broker).call["error_code"] == "broker_unavailable"
    assert len(transport.sent) == 2


# These source/redirect/status/placeholder fields reproduce the saved M&G
# Firecrawl receipt. The successful HTML is synthetic; no score is asserted.
MANDG_URL = "https://mandg.com/"
MANDG_FINAL = "https://www.mandg.com/"


def _mandg_primary():
    primary = _envelope(REDACTED, source_url=MANDG_URL)
    primary["result"]["data"]["metadata"]["url"] = MANDG_FINAL
    return primary


def _mandg_generic(*, requested_url=MANDG_FINAL):
    payload = _generic(final_url=MANDG_FINAL)
    payload["toolResponse"]["rawV2"]["requested_url"] = requested_url
    return payload


def _mandg_execute(broker):
    return broker.execute(_score_context(), operation_id="scrapingdog.scrape",
                          parameters={"url": MANDG_URL}, action_sequence=0,
                          timeout_ms=30_000)


def _mandg_once(broker, *, attempt=1, response_url=None):
    return broker._execute_once(
        _score_context(), operation_id="scrapingdog.scrape", parameters={"url": MANDG_URL},
        action_sequence=0, timeout_ms=30_000, provider_attempt=attempt,
        champion_credential_retry=False, source_content_fallback=attempt == 2,
        source_content_response_url=response_url,
    )


def test_saved_mandg_redirect_drives_one_rescue_with_original_action_identity():
    broker, ledger, transport = _broker([(200, _mandg_primary()), (200, _mandg_generic())])
    assert _mandg_execute(broker).body == HTML.encode()
    primary, fallback = list(ledger.calls.values())
    assert primary["terminal"]["headers"][operations.TRUSTED_RESPONSE_URL_HEADER] == MANDG_FINAL
    assert fallback["call_doc"]["source_content_fallback_url"] == MANDG_FINAL
    assert primary["call_doc"]["request_hash"] == fallback["call_doc"]["request_hash"]
    assert primary["call_doc"]["base_call_identity"] == fallback["call_doc"]["base_call_identity"]
    assert [json.loads(request["body"])["payload"]["url"] for request in transport.sent] == [MANDG_URL, MANDG_FINAL]
    assert [call["actual"] for call in ledger.calls.values()] == [2_000, 0]
    restarted, _, resumed_transport = _broker([], store=ledger)
    result = _mandg_execute(restarted)
    assert result.status == 200 and result.body == HTML.encode()
    assert result.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == MANDG_FINAL
    assert resumed_transport.sent == []


def test_restart_after_primary_settlement_uses_its_pinned_redirect():
    broker, ledger, original_transport = _broker([(200, _mandg_primary())])
    primary = _mandg_once(broker)
    assert primary.status == 502
    assert primary.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == MANDG_FINAL
    restarted, _, transport = _broker([(200, _mandg_generic())], store=ledger)
    assert _mandg_execute(restarted).status == 200
    assert len(original_transport.sent) == len(transport.sent) == 1
    assert json.loads(transport.sent[0]["body"])["payload"]["url"] == MANDG_FINAL


def test_old_primary_terminal_without_redirect_keeps_original_url():
    broker, ledger, _ = _broker([(200, _mandg_primary())])
    primary = _mandg_once(broker)
    ledger.calls[primary.call["call_identity"]]["terminal"]["headers"].pop(operations.TRUSTED_RESPONSE_URL_HEADER)
    restarted, _, transport = _broker([(200, _mandg_generic(requested_url=MANDG_URL))], store=ledger)
    assert _mandg_execute(restarted).status == 200
    assert json.loads(transport.sent[0]["body"])["payload"]["url"] == MANDG_URL


@pytest.mark.parametrize("state", ["settlement", "reservation", "dispatch"])
@pytest.mark.parametrize("legacy", [False, True])
def test_fallback_reservation_replays_exact_keyed_url_before_and_after_url_pinning(state, legacy):
    class RecoveryStore(FakeLedgerStore):
        def recover_deepline_response(self, **kwargs):
            return self.settle_call(**{key: kwargs[key] for key in (
                "run_id", "lease_token_hash", "call_identity", "actual_microusd",
                "terminal_response", "lease_ttl_seconds",
            )})

    actual_url = MANDG_URL if legacy else MANDG_FINAL
    broker, ledger, transport = _broker([
        (200, _mandg_primary()), (200, _mandg_generic(requested_url=actual_url)),
    ], store=RecoveryStore())
    _mandg_once(broker)
    # The old release always used the original source, even when its primary
    # knew a redirect. Construct that exact historical request and receipt.
    assert _mandg_once(broker, attempt=2, response_url=None if legacy else MANDG_FINAL).status == 200
    fallback = next(call for call in ledger.calls.values()
                    if call["call_doc"].get("provider_attempt") == 2)
    if legacy:
        fallback["call_doc"].pop("source_content_fallback_url")
    original_body = transport.sent[1]["body"]
    original_timeout = fallback["call_doc"]["source_content_fallback_timeout_ms"]
    if state != "settlement":
        fallback["kind"] = state
        fallback.pop("terminal")

    class ResumeTransport(FakeTransport):
        def send(self, **request):
            if request["method"] == "GET":
                self.sent.append(request)
                return br.ProviderResponse(200, {"x-deepline-idempotency-supported": "true"}, json.dumps({
                    "requestId": _generic()["job_id"], "toolId": "generic_http_request",
                    "executionRecovery": {"idempotencyKey": fallback["call_doc"]["deepline_execution_key"], "state": "running"},
                }).encode())
            assert request["body"] == original_body
            return super().send(**request)

    resumed_transport = ResumeTransport([(200, _mandg_generic(requested_url=actual_url))])
    restarted, _, _ = _broker([], store=ledger, transport=resumed_transport)
    result = _mandg_execute(restarted)
    assert result.status == 200 and result.body == HTML.encode()
    assert len(resumed_transport.sent) == {"settlement": 0, "reservation": 1, "dispatch": 2}[state]
    assert ("source_content_fallback_url" not in fallback["call_doc"]) is legacy
    assert fallback["call_doc"]["source_content_fallback_timeout_ms"] == original_timeout


@pytest.mark.parametrize("bad_url", [None, "https://metadata.google.internal/", "https://private.local/",
                                      "http://example.com/", "https://other-public.example/"])
def test_tampered_saved_redirect_destination_cannot_dispatch(bad_url):
    broker, ledger, transport = _broker([(200, _mandg_primary()), (200, _mandg_generic())])
    assert _mandg_execute(broker).status == 200
    fallback = next(call for call in ledger.calls.values() if call["call_doc"].get("provider_attempt") == 2)
    fallback["call_doc"]["source_content_fallback_url"] = bad_url
    assert _mandg_execute(broker).call["error_code"] == "broker_unavailable"
    assert len(transport.sent) == 2


@pytest.mark.parametrize("header_url", ["https://metadata.google.internal/", "https://private.local/",
                                        "https://other-public.example/"])
def test_tampered_primary_redirect_or_reservation_mismatch_fails_closed(header_url):
    broker, ledger, transport = _broker([(200, _mandg_primary()), (200, _mandg_generic())])
    assert _mandg_execute(broker).status == 200
    primary = next(call for call in ledger.calls.values() if call["call_doc"].get("provider_attempt") == 1)
    primary["terminal"]["headers"][operations.TRUSTED_RESPONSE_URL_HEADER] = header_url
    assert _mandg_execute(broker).call["error_code"] == "broker_unavailable"
    assert len(transport.sent) == 2


def test_wrong_generic_requested_url_cannot_substitute_for_observed_redirect():
    broker, ledger, transport = _broker([
        (200, _mandg_primary()), (200, _mandg_generic(requested_url=MANDG_URL)),
    ])
    result = _mandg_execute(broker)
    assert result.status == 502 and result.body == operations.GENERIC_UNAVAILABLE_BODY
    assert len(transport.sent) == len(ledger.calls) == 2


def test_untrusted_upstream_header_cannot_override_validated_redirect():
    broker, ledger, transport = _broker([
        (200, _mandg_primary(), {operations.TRUSTED_RESPONSE_URL_HEADER: "https://other-public.example/"}),
        (200, _mandg_generic()),
    ])
    assert _mandg_execute(broker).status == 200
    assert json.loads(transport.sent[1]["body"])["payload"]["url"] == MANDG_FINAL


def test_other_compatibility_failure_cannot_carry_redirect_authority():
    broker, ledger, transport = _broker([(200, _mandg_primary())])
    primary = _mandg_once(broker)
    terminal = ledger.calls[primary.call["call_identity"]]["terminal"]
    terminal["compatibility_failure_code"] = "missing_firecrawl_raw_html"
    assert _mandg_execute(broker).call["error_code"] == "broker_unavailable"
    assert len(transport.sent) == 1
