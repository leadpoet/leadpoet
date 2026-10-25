"""Outer Exa faults cannot become final zero scores for missing source proof.

Keep the real public broker, response readers, investigator and final scorer.
Only transport, ledger, model replies and unrelated fit checks are simulated.
"""

import asyncio
import json

import pytest

from lab_arena import broker as br, operations, scoring, shim
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from tests.lab_arena.firecrawl_diagnostic_adapter_test import URL, _score_context
from tests.lab_arena.test_lab_arena_broker import DL_KEY, FakeTransport, make_broker
from tests.test_company_evidence_investigator import (
    _competition_company, _complete_verdict, _finding, _icp,
)


def _review(monkeypatch, *, primary=404, status=404, body=None, direct=False,
            trusted_primary=False):
    calls = {"fetch": 0, "exa": 0, "model": 0}
    broker_results = []

    def broker_response(operation, parameters, http_status, response_body):
        broker, ledger, transport = make_broker(
            transport=FakeTransport([(http_status, response_body)]),
            credential_for=lambda *_: DL_KEY,
            funding_source_for=lambda *_: "miner_key",
        )
        result = broker.execute(
            _score_context(), operation_id=operation, parameters=parameters,
            action_sequence=0, timeout_ms=30_000,
        )
        assert "dispatch" in ledger.log
        assert transport.sent
        broker_results.append((operation, result.status))
        return result

    def response(url, result):
        return shim._AiohttpResponse(
            request_url=url,
            response_url=result.headers.get(operations.TRUSTED_RESPONSE_URL_HEADER, ""),
            status=result.status, headers=result.headers, body=result.body,
        )

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def get(self, url, **kwargs):
            calls["fetch"] += 1
            assert url == "https://api.scrapingdog.com/scrape"
            assert kwargs["params"]["url"] == URL
            if trusted_primary:
                return response(url, br.BrokerResult(
                    primary, {operations.TRUSTED_RESPONSE_URL_HEADER: URL}, b"", {},
                ))
            return response(url, broker_response(
                "scrapingdog.scrape", {"url": URL}, primary,
                {"error": {"code": "not_found" if primary == 404 else "provider_unavailable"}},
            ))

        def post(self, url, **kwargs):
            payload = kwargs["json"]
            if url == "https://api.exa.ai/contents":
                calls["exa"] += 1
                assert payload["ids"] == [URL]
                assert payload["maxAgeHours"] == 0
                contents = body if body is not None else {"error": {"code": "not_found"}}
                if direct:
                    result = br.BrokerResult(status, {}, json.dumps(contents).encode(), {})
                else:
                    if status == 200:
                        contents = {"result": {"data": contents}, "billing": {"credits_charged": 0}}
                    result = broker_response("exa.contents", payload, status, contents)
                return response(url, result)
            calls["model"] += 1
            if calls["model"] == 1:
                name, arguments = "fetch_page", {"url": URL}
            else:
                name, arguments = "submit_findings", {"findings": [_finding(
                    "stage", status="UNPROVEN", observed_value=None,
                    evidence_url="", evidence_quote="", reason="No company-bound stage proof.",
                )]}
            model_body = {"choices": [{"message": {"tool_calls": [{
                "id": str(calls["model"]), "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }]}}]}
            return response(url, br.BrokerResult(200, {}, json.dumps(model_body).encode(), {}))

    async def search(*_, **__):
        return {"results": []}

    for key in ("OPENROUTER_API_KEY", "EXA_API_KEY", "SCRAPINGDOG_API_KEY"):
        monkeypatch.setenv(key, "offline-test-key")
    monkeypatch.setattr("socket.socket.connect", lambda *_: pytest.fail("network forbidden"))
    monkeypatch.setattr(investigator.aiohttp, "ClientSession", lambda **_: Session())
    monkeypatch.setattr(investigator, "_search_web", search)
    receipt = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://example.com"},
        targets=("stage",), requested_stage="Private Equity",
    ))
    assert calls == {"fetch": 1, "exa": 0 if trusted_primary else 1, "model": 2}
    assert receipt["usage"]["fetch_calls"] == (1 if trusted_primary else 2)
    if not direct and not trusted_primary:
        assert broker_results[-1][0] == "exa.contents"
    return receipt


def _final_score(monkeypatch, receipt, attempts):
    async def prechecks(*_, **__):
        return lead_scorer.company_fit_match("prechecks passed")

    async def homepage(*_, **__):
        return lead_scorer.company_fit_match("homepage identity verified")

    async def broad(**_):
        return _complete_verdict(
            observed_company_stage="Public", stage_matches=True,
            stage_evidence_url=URL, stage_evidence_quote="Acme launched its public product.",
        ), ""

    async def investigation(*, diagnostic, **_):
        # Replay this actual investigator receipt through the existing retry
        # policy; no further provider request is needed to prove finality.
        attempts.append(True)
        diagnostic[investigator.VERIFIER_FAILURE_REASON_KEY] = receipt.get("failure_reason", "")
        return receipt

    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(lead_scorer, "verify_company_exists", homepage)
    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", broad)
    monkeypatch.setattr(lead_scorer, "investigate_company_evidence", investigation)
    scorer = scoring.lab_scorer(scoring.build_scorer_policy(
        scoring_adapter_version="qualification_contacts_v3",
    ))
    return scoring.score_work_item(
        {"scored_run_id": "offline-exa-outer-status"},
        icp=_icp(company_stage="Public", intent_signals=["Announced a funding event"]).model_dump(mode="json"),
        companies=[_competition_company()], scorer=scorer, max_retries=3,
    )


@pytest.mark.parametrize("primary", [404, 502, 503])
@pytest.mark.parametrize("status", [404, 410, 403, 503])
@pytest.mark.parametrize("direct", [False, True], ids=["public-broker", "direct"])
def test_outer_contents_fault_cannot_finalize_zero(monkeypatch, primary, status, direct):
    receipt = _review(monkeypatch, primary=primary, status=status, direct=direct)
    # Assert the final observable outcome first: a provider fault exhausts
    # retries and must never yield a completed zero-score row.
    attempts = []
    with pytest.raises(scoring.ScoringError, match="judge_exhausted") as error:
        _final_score(monkeypatch, receipt, attempts)
    assert error.value.failure_reason == "provider_error"
    assert attempts == [True, True, True]
    assert receipt["failure_reason"] == "provider_error"
    assert receipt.get("_completed_submit") is not True
    assert receipt["usage"]["fetch_outcomes"][-1]["error_class"] not in {
        "source_not_found", "source_gone", "source_access_denied",
    }


@pytest.mark.parametrize("status,expected", [
    (404, "source_not_found"), (410, "source_gone"),
    (401, "source_access_denied"), (403, "source_access_denied"),
])
@pytest.mark.parametrize("direct", [False, True], ids=["public-broker", "direct"])
def test_successful_exact_target_status_can_finalize_unproven(monkeypatch, status, expected, direct):
    body = {"results": [], "statuses": [{
        "id": URL, "status": "error",
        "error": {"httpStatusCode": status, "tag": f"CRAWL_HTTP_{status}"},
    }]}
    receipt = _review(monkeypatch, status=200, body=body, direct=direct)
    attempts = []
    rows = _final_score(monkeypatch, receipt, attempts)
    assert len(rows) == 1 and rows[0]["final_score"] == 0.0
    assert attempts == [True]
    assert receipt["_completed_submit"] is True and receipt["failure_reason"] == ""
    assert receipt["claims"]["stage"]["status"] == "UNPROVEN"
    assert receipt["usage"]["fetch_outcomes"][-1]["error_class"] == expected


@pytest.mark.parametrize("status", [404, 410, 401, 403])
@pytest.mark.parametrize("direct", [False, True], ids=["public-broker", "direct"])
def test_wrong_target_status_cannot_finalize_zero(monkeypatch, status, direct):
    body = {"results": [], "statuses": [{
        "id": "https://other.example/about", "status": "error",
        "error": {"httpStatusCode": status, "tag": f"CRAWL_HTTP_{status}"},
    }]}
    receipt = _review(monkeypatch, status=200, body=body, direct=direct)
    attempts = []
    with pytest.raises(scoring.ScoringError, match="judge_exhausted") as error:
        _final_score(monkeypatch, receipt, attempts)
    assert error.value.failure_reason == "provider_error"
    assert attempts == [True, True, True]


@pytest.mark.parametrize("status", [404, 410])
@pytest.mark.parametrize("direct", [False, True], ids=["public-broker", "direct"])
def test_outer_fault_cannot_authenticate_even_an_exact_target_record(monkeypatch, status, direct):
    body = {"results": [], "statuses": [{
        "id": URL, "status": "error",
        "error": {"httpStatusCode": status, "tag": f"CRAWL_HTTP_{status}"},
    }]}
    receipt = _review(monkeypatch, status=status, body=body, direct=direct)
    attempts = []
    with pytest.raises(scoring.ScoringError, match="judge_exhausted") as error:
        _final_score(monkeypatch, receipt, attempts)
    assert error.value.failure_reason == "provider_error"
    assert attempts == [True, True, True]


@pytest.mark.parametrize("status,expected", [(404, "source_not_found"), (410, "source_gone")])
def test_trusted_primary_page_absence_keeps_existing_finality(monkeypatch, status, expected):
    receipt = _review(monkeypatch, primary=status, trusted_primary=True)
    attempts = []
    rows = _final_score(monkeypatch, receipt, attempts)
    assert len(rows) == 1 and rows[0]["final_score"] == 0.0
    assert attempts == [True]
    assert receipt["_completed_submit"] is True and receipt["failure_reason"] == ""
    assert receipt["usage"]["fetch_outcomes"][0]["error_class"] == expected
