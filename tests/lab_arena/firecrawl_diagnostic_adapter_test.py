"""A paid Firecrawl diagnostic must not masquerade as source content."""

import asyncio
import json

import pytest

from lab_arena import broker as br
from lab_arena import scoring_provider_compat as compat
from qualification.scoring import company_evidence_investigator as investigator
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT,
    DL_KEY,
    FakeLedgerStore,
    FakeTransport,
    make_broker,
)


URL = "https://example.com/about"
DIAGNOSTIC = "Provider account capacity details redacted."
REDACTED = "[REDACTED]"


def _route():
    return compat.route_for(
        kind="score", funding_source="miner_key",
        round_id="arena-2026-10-13", operation_id="scrapingdog.scrape",
        parameters={"url": URL},
    )


def _envelope(raw_html, *, source_url=URL, status=200):
    return {
        "job_id": "public-execute:e3dd106e-aed5-4816-b899-1166d4624c82",
        "status": "completed",
        "result": {"data": {
            "rawHtml": raw_html,
            "metadata": {"sourceURL": source_url, "url": URL, "statusCode": status},
        }},
        "billing": {"credits_charged": 0.02, "cost_usd": 0.002},
    }


@pytest.mark.parametrize("raw_html", [
    "", " \n ", DIAGNOSTIC, "  Provider  account\ncapacity details redacted.  ",
    REDACTED, " \n[REDACTED]\t ", "[redacted]",
])
def test_successful_firecrawl_requires_source_content(raw_html):
    with pytest.raises(compat.CompatibilityResponseError, match="firecrawl_source_content_unavailable"):
        compat.adapt_response_with_trusted_url(
            _route(), status=200, headers={},
            body=json.dumps(_envelope(raw_html)).encode(),
        )


@pytest.mark.parametrize("target_status", [401, 402, 403, 407, 429, 500, 502, 503, 504, 599])
def test_failed_target_retains_validated_source_proof_for_alternate_transport(target_status):
    payload = _envelope("<p>Target fetch failed.</p>", status=target_status)
    final_url = "https://example.com/about/team"
    payload["result"]["data"]["metadata"]["url"] = final_url
    with pytest.raises(compat.CompatibilityResponseError,
                       match="firecrawl_source_content_unavailable") as captured:
        compat.adapt_response_with_trusted_url(
            _route(), status=200, headers={}, body=json.dumps(payload).encode(),
        )
    assert captured.value.trusted_response_url == final_url


@pytest.mark.parametrize("raw_html", [
    "Short but valid page text.",
    "<article>Provider account capacity details redacted. This article explains the issue.</article>",
    "[REDACTED] appears in this valid page text.",
    "<article>[REDACTED]</article>",
    "[ REDACTED ]",
])
def test_valid_short_or_mentioning_page_survives(raw_html):
    status, _headers, body, response_url = compat.adapt_response_with_trusted_url(
        _route(), status=200, headers={},
        body=json.dumps(_envelope(raw_html)).encode(),
    )
    assert (status, body, response_url) == (200, raw_html.encode(), URL)


def test_live_shaped_redacted_envelope_with_valid_redirect_metadata_is_unavailable():
    payload = _envelope(REDACTED)
    payload["result"]["data"]["metadata"]["url"] = "https://example.com/en/global"
    assert len(payload["result"]["data"]["rawHtml"].encode()) == 10
    with pytest.raises(compat.CompatibilityResponseError, match="firecrawl_source_content_unavailable"):
        compat.adapt_response_with_trusted_url(
            _route(), status=200, headers={}, body=json.dumps(payload).encode(),
        )


@pytest.mark.parametrize("raw_html", [DIAGNOSTIC, REDACTED])
def test_wrong_url_and_missing_status_still_fail_before_content_check(raw_html):
    with pytest.raises(compat.CompatibilityResponseError, match="firecrawl_source_url_mismatch"):
        compat.adapt_response_with_trusted_url(
            _route(), status=200, headers={},
            body=json.dumps(_envelope(raw_html, source_url="https://other.example/")).encode(),
        )
    missing_status = _envelope(raw_html)
    del missing_status["result"]["data"]["metadata"]["statusCode"]
    with pytest.raises(compat.CompatibilityResponseError, match="missing_firecrawl_target_status"):
        compat.adapt_response_with_trusted_url(
            _route(), status=200, headers={}, body=json.dumps(missing_status).encode(),
        )


def _score_context():
    return br.RunContext(**{**CONTEXT.__dict__, "kind": "score", "round_id": "arena-2026-10-13"})


@pytest.mark.parametrize("raw_html", [DIAGNOSTIC, REDACTED])
def test_paid_diagnostic_settles_exact_cost_once_and_triggers_fetch_failure(monkeypatch, raw_html):
    store = FakeLedgerStore(openrouter_capacity=49_945_650)
    broker, _store, transport = make_broker(
        store=store, transport=FakeTransport([(200, _envelope(raw_html))]),
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key",
    )
    # Inspect the primary settlement separately from the bounded alternate
    # transport now selected by execute() after this exact settled failure.
    result = broker._execute_once(
        _score_context(), operation_id="scrapingdog.scrape",
        parameters={"url": URL}, action_sequence=0, timeout_ms=30_000,
        provider_attempt=1, champion_credential_retry=False,
    )
    assert result.status == 502 and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 2_000
    assert json.loads(result.body) == {"error": {"code": "provider_unavailable"}}
    saved = store.calls[result.call["call_identity"]]
    assert saved["kind"] == "settlement" and saved["actual"] == 2_000
    assert saved["terminal"]["call_succeeded"] is False
    assert saved["terminal"]["provider_cost"]["units"] == "0.02"
    assert store.log == ["reserve", "dispatch", "settle"]
    assert len(transport.sent) == 1

    replay = broker._execute_once(
        _score_context(), operation_id="scrapingdog.scrape",
        parameters={"url": URL}, action_sequence=0, timeout_ms=30_000,
        provider_attempt=1, champion_credential_retry=False,
    )
    assert replay.status == 502 and replay.body == result.body
    assert replay.call["call_identity"] == result.call["call_identity"]
    assert store.calls[result.call["call_identity"]]["actual"] == 2_000
    assert len(transport.sent) == 1 and store.log.count("settle") == 1

    async def fetch(_session, _url, **_kwargs):
        return result.status, URL, result.body.decode()

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-key")
    monkeypatch.setattr(investigator, "_fetch_bounded_html", fetch)
    page = asyncio.run(investigator._fetch_page(object(), URL))
    assert page == {"ok": False, "error": "http_502"}


@pytest.mark.parametrize("raw_html", [DIAGNOSTIC, REDACTED])
def test_diagnostic_with_delayed_billing_keeps_full_reservation(raw_html):
    payload = _envelope(raw_html)
    payload["billing"] = {"pricing_status": "pending"}
    transport = FakeTransport([(200, payload), (200, {"entries": "invalid"})])
    store = FakeLedgerStore(openrouter_capacity=49_945_650)
    broker, _store, _transport = make_broker(
        store=store, transport=transport,
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key",
    )
    result = broker.execute(
        _score_context(), operation_id="scrapingdog.scrape",
        parameters={"url": URL}, action_sequence=0, timeout_ms=30_000,
    )
    assert result.status == 502 and result.call["outcome"] == "uncertain"
    assert result.call["reserved_microusd"] == 49_945_650
    assert "actual_microusd" not in result.call
    saved = store.calls[result.call["call_identity"]]
    assert saved["kind"] == "uncertain" and saved["amount"] == 49_945_650
    assert saved["uncertain_doc"]["reason"] == "settle_failure"
    assert saved["uncertain_doc"]["call_succeeded"] is False
    assert saved["uncertain_doc"]["failure_stage"] == "response_adaptation"
    assert saved["uncertain_doc"]["error_class"] == "CompatibilityResponseError"
    for field in ("deepline_request_id", "deepline_execution_key", "credential_fingerprint"):
        assert saved["uncertain_doc"][field] == saved["call_doc"][field]
    assert saved["uncertain_doc"]["deepline_operation"] == "firecrawl_scrape"
    assert saved["uncertain_doc"]["deepline_job_id"] == payload["job_id"]
    assert store.log == ["reserve", "dispatch", "uncertain"]
    assert len([request for request in transport.sent if request["method"] == "POST"]) == 1

    class ExactBillTransport:
        def __init__(self):
            self.sent = []

        def send(self, **request):
            self.sent.append(request)
            assert request["method"] == "GET"
            if "/executions/by-key/" in request["url"]:
                document = {
                    "toolId": "firecrawl_scrape", "requestId": payload["job_id"],
                    "executionRecovery": {
                        "idempotencyKey": saved["uncertain_doc"]["deepline_execution_key"],
                        "state": "completed",
                    },
                }
                return br.ProviderResponse(
                    200, {"x-deepline-idempotency-supported": "true"},
                    json.dumps(document).encode(),
                )
            entry = {
                "id": "exact-charge", "request_id": payload["job_id"],
                "provider": "firecrawl", "operation": "firecrawl_scrape",
                "credits": 0.02, "delta": -0.02,
                "charge_state": "posted", "charge_finality": "final", "metadata": {},
            }
            return br.ProviderResponse(
                200, {"content-type": "application/json"},
                json.dumps({"recent": {
                    "request_id": payload["job_id"], "entries": [entry],
                }}).encode(),
            )

    exact = ExactBillTransport()
    broker._transport = exact
    settled = []

    def settle_exact(**kwargs):
        settled.append(kwargs)
        assert kwargs["call_identity"] == result.call["call_identity"]
        assert kwargs["request_id"] == saved["uncertain_doc"]["deepline_request_id"]
        assert kwargs["execution_key"] == saved["uncertain_doc"]["deepline_execution_key"]
        assert kwargs["credential_fingerprint"] == saved["uncertain_doc"]["credential_fingerprint"]
        assert kwargs["recovered_request_id"] == payload["job_id"]
        assert kwargs["actual_microusd"] == 2_000
        saved.update(kind="settlement", actual=kwargs["actual_microusd"])
        return {"status": "settled", "actual_microusd": kwargs["actual_microusd"]}

    store.reconcile_deepline_cost = settle_exact
    context = _score_context()
    candidate = {
        "uncertain_entry_id": 41, "round_id": context.round_id,
        "run_id": context.run_id, "submission_id": context.submission_id,
        "miner_hotkey": context.miner_hotkey,
        "assignment_id": context.assignment_id, "stage": context.stage,
        "icp_position": context.icp_position, "attempt": context.attempt,
        "kind": context.kind, "call_identity": result.call["call_identity"],
        "request_id": saved["uncertain_doc"]["deepline_request_id"],
        "credential_fingerprint": saved["uncertain_doc"]["credential_fingerprint"],
        "funding_source": "miner_key", "run_status": "leased",
        "lease_expires_at": "2026-10-13T11:00:00Z",
        "uncertain_at": "2026-10-13T10:50:00Z",
        "reservation_at": "2026-10-13T10:49:00Z",
        "operation": "firecrawl_scrape",
        "execution_key": saved["uncertain_doc"]["deepline_execution_key"],
        "billing_provider": "firecrawl",
        "operation_aliases": ["firecrawl_scrape", "scrape"],
    }
    assert broker.reconcile_deepline_cost(candidate)["status"] == "settled"
    assert saved["actual"] == 2_000 and len(settled) == 1
    assert [request["method"] for request in exact.sent] == ["GET", "GET"]
    assert len([request for request in transport.sent if request["method"] == "POST"]) == 1


@pytest.mark.parametrize("raw_html,target_status", [
    ("Short but valid page text.", 200),
    (DIAGNOSTIC, 404),
    (REDACTED, 404),
])
def test_zero_cost_or_failed_target_status_is_not_reclassified(raw_html, target_status):
    payload = _envelope(raw_html, status=target_status)
    payload["billing"] = {"credits_charged": 0}
    broker, store, _transport = make_broker(
        transport=FakeTransport([(200, payload)]),
        credential_for=lambda _context, _provider: DL_KEY,
        funding_source_for=lambda _context: "miner_key",
    )
    result = broker.execute(
        _score_context(), operation_id="scrapingdog.scrape",
        parameters={"url": URL}, action_sequence=0, timeout_ms=30_000,
    )
    assert result.status == target_status and result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == 0
    assert store.calls[result.call["call_identity"]]["actual"] == 0
