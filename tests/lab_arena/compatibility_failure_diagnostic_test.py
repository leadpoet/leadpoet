"""Fixed adapter diagnostics survive telemetry without changing provider results."""

import pytest

from lab_arena import operations, scoring_provider_compat as compat, trajectory
from tests.lab_arena.firecrawl_diagnostic_adapter_test import (
    DIAGNOSTIC, URL, _envelope, _score_context,
)
from tests.lab_arena.test_lab_arena_broker import DL_KEY, FakeTransport, make_broker


CODES = (
    "invalid_deepline_envelope", "incomplete_deepline_envelope",
    "missing_deepline_result", "missing_firecrawl_raw_html",
    "firecrawl_source_url_mismatch", "invalid_firecrawl_final_url",
    "missing_firecrawl_target_status", "firecrawl_source_content_unavailable",
)


class OtherCompatibilityError(compat.CompatibilityResponseError):
    pass


def _execute(payload):
    broker, ledger, transport = make_broker(
        transport=FakeTransport([(200, payload)]),
        credential_for=lambda *_: DL_KEY, funding_source_for=lambda *_: "miner_key",
    )
    result = broker.execute(
        _score_context(), operation_id="scrapingdog.scrape",
        parameters={"url": URL}, action_sequence=0, timeout_ms=30_000,
    )
    return result, ledger, transport


def _event(result):
    return trajectory.event("provider.response", trajectory.provider_response_content(
        result.to_document(), elapsed_ms=1,
    ))["content"]


@pytest.mark.parametrize("code", CODES)
def test_fixed_adapter_reason_survives_real_broker_and_trajectory(monkeypatch, code):
    def refused(*_, **__):
        raise compat.CompatibilityResponseError(code)

    monkeypatch.setattr(compat, "adapt_response_with_trusted_url", refused)
    result, ledger, transport = _execute(_envelope("<p>Acme</p>"))
    assert result.status == 502
    assert result.body == b'{"error":{"code":"provider_unavailable"}}'
    assert operations.TRUSTED_RESPONSE_URL_HEADER not in result.headers
    assert result.call["actual_microusd"] == 2_000
    content = _event(result)
    assert content["call"]["compatibility_failure_code"] == code
    assert content["http_status"] == 502
    assert content["call"]["actual_microusd"] == 2_000
    saved = ledger.calls[result.call["call_identity"]]
    assert saved["terminal"]["call_succeeded"] is False
    assert "compatibility_failure_code" not in saved["terminal"]
    assert "compatibility_failure_code" not in saved["terminal"]["headers"]
    assert ledger.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1


@pytest.mark.parametrize("reason", [
    "firecrawl_source_content_unavailable", "missing_firecrawl_raw_html",
    "invalid_firecrawl_final_url",
])
def test_native_adapter_emits_exact_reason_without_source_content(reason):
    payload = _envelope(DIAGNOSTIC)
    if reason == "missing_firecrawl_raw_html":
        payload["result"]["data"]["rawHtml"] = {"unexpected": "shape"}
    elif reason == "invalid_firecrawl_final_url":
        payload["result"]["data"]["metadata"]["url"] = "http://example.com/"
    result, _, _ = _execute(payload)
    assert _event(result)["call"]["compatibility_failure_code"] == reason
    assert result.body == b'{"error":{"code":"provider_unavailable"}}'


@pytest.mark.parametrize("error", [
    compat.CompatibilityResponseError("unknown_future_reason"),
    compat.CompatibilityResponseError("private credential text"),
    OtherCompatibilityError("firecrawl_source_content_unavailable"),
    ValueError("firecrawl_source_content_unavailable"),
])
def test_unknown_codes_and_untyped_exceptions_do_not_emit_text(monkeypatch, error):
    def refused(*_, **__):
        raise error

    monkeypatch.setattr(compat, "adapt_response_with_trusted_url", refused)
    result, ledger, _ = _execute(_envelope("<p>Acme</p>"))
    assert result.status == 502 and result.call["actual_microusd"] == 2_000
    assert "compatibility_failure_code" not in _event(result)["call"]
    assert "private credential text" not in repr(result.to_document())
    assert "private credential text" not in repr(ledger.calls)


def test_native_positive_status_body_url_and_cost_are_unchanged():
    html = "<p>Acme supplies software.</p>"
    result, ledger, _ = _execute(_envelope(html))
    assert result.status == 200 and result.body == html.encode()
    assert result.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == URL
    assert result.call["actual_microusd"] == 2_000
    assert "compatibility_failure_code" not in _event(result)["call"]
    assert ledger.calls[result.call["call_identity"]]["terminal"]["call_succeeded"] is True


def test_unknown_fee_stays_uncertain_without_new_diagnostic_field():
    payload = _envelope(DIAGNOSTIC)
    payload["billing"] = {"pricing_status": "pending"}
    result, ledger, _ = _execute(payload)
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert "actual_microusd" not in result.call
    assert "compatibility_failure_code" not in _event(result)["call"]
    assert ledger.calls[result.call["call_identity"]]["kind"] == "uncertain"
