"""Exercise broker responses through the real page reader and investigator."""

import asyncio
import json
from types import SimpleNamespace

import aiohttp
import pytest

from lab_arena import broker as br, operations, shim
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import company_verification
from tests.lab_arena.firecrawl_diagnostic_adapter_test import (
    DIAGNOSTIC, URL, _envelope, _score_context,
)
from tests.lab_arena.test_lab_arena_broker import DL_KEY, FakeTransport, make_broker


def _adapted_result(raw_html):
    broker, ledger, _ = make_broker(
        transport=FakeTransport([(200, _envelope(raw_html))]),
        credential_for=lambda *_: DL_KEY,
        funding_source_for=lambda *_: "miner_key",
    )
    # Keep this fixture focused on the primary response consumed by the
    # investigator's separate exact-source Exa fallback.
    result = broker._execute_once(
        _score_context(), operation_id="scrapingdog.scrape",
        parameters={"url": URL}, action_sequence=0, timeout_ms=30_000,
        provider_attempt=1, champion_credential_retry=False,
    )
    return result, ledger


def _investigate(monkeypatch, response, *, exa_ok=True, exa_absence=False, cap=None,
                 deadline=False, remove_exa=False, source_url=URL,
                 fetch_exception=None):
    calls = {"fetch": 0, "exa": 0, "model": 0}
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def get(self, url, **kwargs):
            calls["fetch"] += 1
            assert url == "https://api.scrapingdog.com/scrape"
            assert kwargs["params"]["url"] == source_url
            if deadline:
                clock.now = investigator.ADMISSION_DEADLINE_SECONDS
            if remove_exa:
                monkeypatch.delenv("EXA_API_KEY")
            if fetch_exception is not None:
                raise fetch_exception
            return shim._AiohttpResponse(
                request_url=url,
                response_url=response.headers.get(operations.TRUSTED_RESPONSE_URL_HEADER, ""),
                status=response.status, headers=response.headers, body=response.body,
            )

    async def post(_session, url, *, headers, payload):
        if url == "https://api.exa.ai/contents":
            calls["exa"] += 1
            assert payload["ids"] == [source_url]
            assert payload["maxAgeHours"] == 0
            if not exa_ok:
                return 503, {"error": "offline failure"}
            if exa_absence:
                return 200, {"results": [], "statuses": [{
                    "id": source_url, "status": "error",
                    "error": {"httpStatusCode": 404, "tag": "CRAWL_HTTP_404"},
                }]}
            return 200, {"results": [{"url": source_url,
                                      "text": "Acme supplies software products."}]}
        calls["model"] += 1
        if calls["model"] == 1:
            name, args = "fetch_page", {"url": source_url}
        else:
            name, args = "submit_findings", {"findings": [{
                "target": "stage", "status": "UNPROVEN", "observed_value": None,
                "observed_country": "", "observed_state": "",
                "observed_industry": "", "observed_subindustry": "",
                "activity_role": "unresolved", "evidence_url": "",
                "evidence_quote": "", "supporting_evidence": [],
                "old_name": "", "new_name": "", "old_domain": "", "new_domain": "",
                "shared_linkedin_slug": "", "reason": "No company-bound stage proof.",
            }]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": f"call-{calls['model']}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}}]}

    async def search(*_, **__):
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-test-key")
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "offline-test-key")
    monkeypatch.setattr("socket.socket.connect", lambda *_: pytest.fail("network forbidden"))
    monkeypatch.setattr(investigator.aiohttp, "ClientSession", lambda **_: Session())
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "time", clock)
    if cap is not None:
        monkeypatch.setattr(investigator, "MAX_FETCH_CALLS", cap)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://example.com"},
        targets=("stage",), requested_stage="Private Equity",
    ))
    return result, calls


def test_paid_adapter_diagnostic_uses_real_reader_and_exact_url_fallback(monkeypatch):
    response, ledger = _adapted_result(DIAGNOSTIC)
    assert response.status == 502
    assert response.call["actual_microusd"] == 2_000
    assert response.headers[operations.TRUSTED_RESPONSE_URL_HEADER] == URL
    result, calls = _investigate(monkeypatch, response)
    assert calls == {"fetch": 1, "exa": 1, "model": 2}
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert result["usage"]["fetch_calls"] == 2
    assert [x["ok"] for x in result["usage"]["fetch_outcomes"]] == [False, True]
    assert ledger.log.count("settle") == 1


@pytest.mark.parametrize("status", [404, 502, 503, 401, 402, 403, 429, 500, 504])
def test_only_typed_unbound_404_502_and_503_allow_exact_url_fallback(monkeypatch, status):
    response = br.BrokerResult(status, {}, b'{"error":{"code":"provider_unavailable"}}', {})
    result, calls = _investigate(monkeypatch, response)
    if status in (404, 502, 503):
        assert calls["exa"] == 1
        assert result["_completed_submit"] is True
    else:
        assert calls["exa"] == 0
        assert result["failure_reason"] == "provider_error"
        assert result.get("_completed_submit") is not True


def test_validated_target_404_is_not_a_broker_error_or_exa_fallback(monkeypatch):
    response = br.BrokerResult(
        404, {operations.TRUSTED_RESPONSE_URL_HEADER: URL}, b"", {},
    )
    result, calls = _investigate(monkeypatch, response)
    assert calls["fetch"] == 1 and calls["exa"] == 0
    assert result["usage"]["fetch_outcomes"][0]["error_class"] == "source_not_found"


def test_unbound_404_failed_exact_fallback_stays_incomplete(monkeypatch):
    response = br.BrokerResult(404, {}, b'{"message":"provider failure"}', {})
    result, calls = _investigate(monkeypatch, response, exa_ok=False)
    assert calls["fetch"] == 1 and calls["exa"] == 1
    assert result["failure_reason"] == "provider_error"
    assert result.get("_completed_submit") is not True


def test_unbound_broker_404_needs_exact_exa_absence_for_unproven(monkeypatch):
    response = br.BrokerResult(404, {}, b'{"message":"unbound provider failure"}', {})
    result, calls = _investigate(monkeypatch, response, exa_absence=True)
    assert calls["fetch"] == 1 and calls["exa"] == 1
    assert result["failure_reason"] == ""
    assert result["_completed_submit"] is True
    assert result["claims"]["stage"]["status"] == "UNPROVEN"
    assert [row["error_class"] for row in result["usage"]["fetch_outcomes"]] == [
        "fetch_failed", "source_not_found",
    ]
    assert all(row["loaded_text_length"] == 0 for row in result["usage"]["fetch_outcomes"])


@pytest.mark.parametrize("status", [None, "502", True, 0, 600])
def test_unknown_or_invalid_typed_status_does_not_enable_fallback(monkeypatch, status):
    error = company_verification._ArenaProviderFetchError("offline", broker_status=status)
    assert error.broker_status is None
    response = br.BrokerResult(502, {}, b"", {})
    result, calls = _investigate(monkeypatch, response, fetch_exception=error)
    assert calls["exa"] == 0
    assert result["failure_reason"] == "provider_error"
    assert result.get("_completed_submit") is not True


def test_untyped_client_error_does_not_enable_fallback(monkeypatch):
    response = br.BrokerResult(502, {}, b"", {})
    result, calls = _investigate(
        monkeypatch, response, fetch_exception=aiohttp.ClientError("offline"),
    )
    assert calls["exa"] == 0
    assert result["failure_reason"] == "provider_error"


def test_failed_fallback_stays_incomplete_infrastructure(monkeypatch):
    response, _ = _adapted_result(DIAGNOSTIC)
    result, calls = _investigate(monkeypatch, response, exa_ok=False)
    assert calls["exa"] == 1
    assert result["usage"]["fetch_calls"] == 2
    assert [x["ok"] for x in result["usage"]["fetch_outcomes"]] == [False, False]
    assert result["failure_reason"] == "provider_error"
    assert result["_completed_submit"] is False


@pytest.mark.parametrize("status", [404, 502])
def test_failed_fallback_cannot_finalize_an_arena_zero(monkeypatch, status):
    from lab_arena import scoring
    from qualification.scoring import lead_scorer
    from tests.test_company_evidence_investigator import (
        _competition_company, _complete_verdict, _icp,
    )

    response = (br.BrokerResult(404, {}, b'{"message":"unbound provider failure"}', {})
                if status == 404 else _adapted_result(DIAGNOSTIC)[0])
    failed, _ = _investigate(monkeypatch, response, exa_ok=False)
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
        diagnostic[investigator.VERIFIER_FAILURE_REASON_KEY] = failed["failure_reason"]
        return failed

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigation)
    scorer = scoring.lab_scorer(scoring.build_scorer_policy(
        scoring_adapter_version="qualification_contacts_v3",
    ))
    with pytest.raises(scoring.ScoringError, match="judge_exhausted") as error:
        scoring.score_work_item(
            {"scored_run_id": "failed-fallback-control"},
            icp=_icp(
                company_stage="Public", intent_signals=["Announced a funding event"],
            ).model_dump(mode="json"),
            companies=[_competition_company()], scorer=scorer, max_retries=3,
        )
    assert attempts == [True, True, True]
    assert error.value.failure_reason == "provider_error"


@pytest.mark.parametrize("status", [404, 502])
@pytest.mark.parametrize("limit", ["cap", "deadline", "no_exa"])
def test_existing_admission_and_fetch_limits_still_bound_fallback(monkeypatch, limit, status):
    response = (br.BrokerResult(404, {}, b'{"message":"unbound provider failure"}', {})
                if status == 404 else _adapted_result(DIAGNOSTIC)[0])
    result, calls = _investigate(
        monkeypatch, response, cap=1 if limit == "cap" else None,
        deadline=limit == "deadline", remove_exa=limit == "no_exa",
    )
    assert calls["fetch"] == 1
    assert calls["exa"] == 0
    assert result["usage"]["fetch_calls"] == 1
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] in {"provider_error", "admission_budget_interrupted"}


def test_normal_adapter_page_does_not_use_fallback(monkeypatch):
    response, _ = _adapted_result("<p>Acme supplies software products.</p>")
    assert response.status == 200
    result, calls = _investigate(monkeypatch, response)
    assert calls["fetch"] == 1 and calls["exa"] == 0
    assert result["usage"]["fetch_calls"] == 1
    assert result["_completed_submit"] is True


@pytest.mark.parametrize("url", ["not-a-url", "https://user:password@example.com/evidence"])
def test_malformed_source_never_reaches_transport_or_fallback(monkeypatch, url):
    response = br.BrokerResult(502, {}, b"", {})
    result, calls = _investigate(monkeypatch, response, source_url=url)
    assert calls["fetch"] == calls["exa"] == 0
    assert result["usage"]["fetch_calls"] <= 1
