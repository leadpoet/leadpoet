"""The captured non-retryable TLS read failure stays local to its source."""

import base64
import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from lab_arena import broker as br, shim
from tests.lab_arena.score_compat_capability_test import firecrawl_catalog
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, DL_KEY, FakeTransport, deepline_history, deepline_history_entry,
    make_broker,
)
from tests.lab_arena.typed_investigator_fallback_test import _investigate


URL = "https://www.technimark.com/"
CAPTURED = json.loads((Path(__file__).parents[1] / "fixtures" /
                      "firecrawl_technimark_tls_error_20261109.json").read_text())
PARAMETERS = {"tool": "firecrawl_scrape", "payload": {"url": URL}}


def _response(document=CAPTURED, status=502, **kwargs):
    return br.ProviderResponse(status, {}, json.dumps(document).encode(), **kwargs)


def _routed_call(*, funding_source="miner_key", credits=0, document=None):
    document = copy.deepcopy(CAPTURED if document is None else document)
    entry = deepline_history_entry(
        document["request_id"], "firecrawl_scrape", credits, provider="firecrawl",
    )
    entry.update(status="error", outcome="error")
    broker, store, transport = make_broker(
        transport=FakeTransport([(502, document), (200, deepline_history(entry))]),
        credential_for=lambda *_: DL_KEY,
        funding_source_for=lambda *_: funding_source,
        **({"host_shadow_score_compat_round_id": "arena-2026-10-09",
            "host_shadow_score_compat_source_round_id": "arena-2026-10-09"}
           if funding_source == "host" else {}),
    )
    context = replace(CONTEXT, kind="score", round_id="arena-2026-10-09",
                      deepline_catalog=firecrawl_catalog())
    arguments = dict(operation_id="scrapingdog.scrape", parameters={"url": URL},
                     action_sequence=0, timeout_ms=30_000)
    return broker.execute(context, **arguments), broker, store, transport, context, arguments


@pytest.mark.parametrize("funding_source", ["miner_key", "host"])
@pytest.mark.parametrize("credits,amount", [(0, 0), (0.5, 50_000)])
def test_captured_tls_failure_keeps_source_local_settlement_and_replay(funding_source, credits, amount):
    result, broker, store, transport, context, arguments = _routed_call(
        funding_source=funding_source, credits=credits,
    )
    assert result.status == 403
    assert result.body == b'{"error":{"code":"provider_request_refused"}}'
    assert result.call["provider_status"] == 502
    assert result.call["error_code"] == "provider_request_refused"
    assert result.call["outcome"] == "settled"
    assert result.call["actual_microusd"] == amount
    assert store.log == ["reserve", "dispatch", "settle"]
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert terminal["call_succeeded"] is False
    assert "account_failure_evidence" not in terminal
    assert "skipTlsVerification" not in result.body.decode()
    receipt = terminal["upstream_response_receipt"]
    assert receipt["status"] == 502
    assert json.loads(base64.b64decode(receipt["body_b64"])) == CAPTURED
    assert "upstream_response_receipt" not in json.dumps(result.to_document())
    sent = len(transport.sent)
    replay = broker.execute(context, **arguments)
    assert replay.status == result.status and replay.body == result.body
    assert replay.call["actual_microusd"] == amount
    assert replay.call["idempotent"] is True
    assert len(transport.sent) == sent


def test_captured_tls_failure_reaches_existing_exact_source_fallback(monkeypatch):
    response, *_ = _routed_call()
    response = replace(response, headers={
        **response.headers, shim.CALL_IDENTITY_HEADER: response.call["call_identity"],
    })
    result, calls = _investigate(monkeypatch, response, source_url=URL)
    assert calls == {"fetch": 1, "exa": 1, "model": 2}
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert result["usage"]["fetch_calls"] == 2


def test_completed_source_absence_finishes_local_but_failed_provider_fallback_does_not(monkeypatch):
    response, *_ = _routed_call()
    response = replace(response, headers={
        **response.headers, shim.CALL_IDENTITY_HEADER: response.call["call_identity"],
    })
    completed, calls = _investigate(monkeypatch, response, source_url=URL, exa_absence=True)
    assert calls["exa"] == 1
    assert completed["_completed_submit"] is True
    assert completed["failure_reason"] == ""
    assert completed["claims"]["stage"]["status"] == "UNPROVEN"
    failed, calls = _investigate(monkeypatch, response, source_url=URL, exa_ok=False)
    assert calls["exa"] == 1
    assert failed["_completed_submit"] is False
    assert failed["failure_reason"] == "provider_error"


def test_captured_source_absence_finishes_real_company_scorer(monkeypatch):
    from lab_arena import scoring
    from qualification.scoring import company_evidence_investigator as investigator, lead_scorer
    from tests.test_company_evidence_investigator import _competition_company, _complete_verdict, _icp

    response, *_ = _routed_call()
    response = replace(response, headers={
        **response.headers, shim.CALL_IDENTITY_HEADER: response.call["call_identity"],
    })
    completed, _ = _investigate(monkeypatch, response, source_url=URL, exa_absence=True)
    attempts = []

    async def prechecks(*_, **__):
        return lead_scorer.company_fit_match("prechecks passed")

    async def homepage(*_, **__):
        return lead_scorer.company_fit_match("homepage identity verified")

    async def broad(**_):
        return _complete_verdict(
            observed_company_stage="Public", stage_matches=True,
            stage_evidence_url=URL,
            stage_evidence_quote="Acme launched its public product.",
        ), ""

    async def investigation(*, diagnostic, **_):
        attempts.append(True)
        diagnostic[investigator.VERIFIER_FAILURE_REASON_KEY] = completed["failure_reason"]
        return completed

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigation)
    scorer = scoring.lab_scorer(scoring.build_scorer_policy(
        scoring_adapter_version="qualification_contacts_v3",
    ))
    rows = scoring.score_work_item(
        {"scored_run_id": "captured-tls-local-finality"},
        icp=_icp(company_stage="Public", intent_signals=["Announced a funding event"]).model_dump(mode="json"),
        companies=[_competition_company()], scorer=scorer, max_retries=3,
    )
    assert len(rows) == 1 and rows[0]["final_score"] == 0
    assert attempts == [True]


@pytest.mark.parametrize("field,value", [
    ("code", "UPSTREAM_RATE_LIMIT"), ("provider", "generic_http"),
    ("operation", "another_operation"), ("failure_origin", "account"),
    ("error_category", "provider_account"), ("credential_owner", "workspace"),
    ("credential_source", "workspace"), ("upstream_status", 429),
    ("upstream_status", "500"), ("upstream_error_code", "SCRAPE_TIMEOUT"),
    ("upstream_error_code", "INTERNAL_SERVER_ERROR"), ("tool_error", {}),
    ("executionRecovery", {"state": "pending"}),
])
def test_provider_faults_and_ambiguous_native_failures_stay_infrastructure(field, value):
    body = copy.deepcopy(CAPTURED)
    body[field] = value
    assert not br._provider_request_refused("deepline", PARAMETERS, _response(body))


@pytest.mark.parametrize("field,value", [
    ("code", "UPSTREAM_TIMEOUT"), ("provider", "other"),
    ("operation", "other"), ("origin", "account"), ("category", "rate_limit"),
    ("statusCode", 429), ("statusCode", "502"), ("retryable", True),
    ("retryable", None), ("retryable", 0),
])
def test_inconsistent_tool_error_cannot_authorize_source_finality(field, value):
    body = copy.deepcopy(CAPTURED)
    body["tool_error"][field] = value
    assert not br._provider_request_refused("deepline", PARAMETERS, _response(body))


@pytest.mark.parametrize("status", [401, 402, 403, 408, 429, 500, 503, 504])
def test_other_outer_statuses_never_gain_local_authority(status):
    assert not br._provider_request_refused("deepline", PARAMETERS, _response(status=status))


@pytest.mark.parametrize("url", ["https://127.0.0.1/", "https://user@example.com/", "http://example.com/"])
def test_unvalidated_target_does_not_gain_source_local_authority(url):
    assert not br._provider_request_refused(
        "deepline", {**PARAMETERS, "payload": {"url": url}}, _response(),
    )


def test_wrong_provider_tool_and_internal_response_do_not_gain_local_authority():
    assert not br._provider_request_refused("openrouter", PARAMETERS, _response())
    assert not br._provider_request_refused("deepline", {**PARAMETERS, "tool": "generic_http_request"}, _response())
    assert not br._provider_request_refused("deepline", PARAMETERS, _response(internal_provenance="credential_echo"))
