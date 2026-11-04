"""Exact Flightpath source-access failures stay distinct from provider outages."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from tests.test_company_evidence_investigator import _finding
from tests.test_completed_cross_domain_negative import _identity, _receipt
from tests import test_completed_cross_domain_negative as fit_controls


URL = "https://flightpath.fm/template/contact-v1"
ALTERNATE = "https://flightpath.fm/about"
QUOTE = "Flightpath sells commercial workflow software to enterprise customers."
FLIGHTPATH_401 = {"results": [], "statuses": [{
    "id": URL, "status": "error",
    "error": {"httpStatusCode": 401, "tag": "CRAWL_HTTP_401"},
}]}


def _classify(monkeypatch, body, status=200):
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setattr(investigator, "_post_json", AsyncMock(return_value=(status, body)))
    return asyncio.run(investigator._fetch_exa_exact_page(object(), URL))


@pytest.mark.parametrize("status", [401, 403])
def test_exact_source_denial_is_typed(monkeypatch, status):
    body = copy.deepcopy(FLIGHTPATH_401)
    body["statuses"][0]["error"] = {"httpStatusCode": status, "tag": f"CRAWL_HTTP_{status}"}
    result = _classify(monkeypatch, body)
    assert result == {"ok": False, "error": "source_access_denied"}
    assert investigator._fetch_outcome(URL, result)["error_class"] == "source_access_denied"


@pytest.mark.parametrize("status,expected", [(404, "source_not_found"), (410, "source_gone")])
def test_exact_exa_crawl_absence_is_typed(monkeypatch, status, expected):
    body = copy.deepcopy(FLIGHTPATH_401)
    body["statuses"][0]["error"] = {"httpStatusCode": status, "tag": f"CRAWL_HTTP_{status}"}
    result = _classify(monkeypatch, body)
    assert result == {"ok": False, "error": expected}
    assert investigator._fetch_outcome(URL, result)["error_class"] == expected


def test_captured_exa_crawl_not_found_is_exact_source_absence(monkeypatch):
    body = {"results": [], "statuses": [{
        "id": URL, "status": "error",
        "error": {"httpStatusCode": 404, "tag": "CRAWL_NOT_FOUND"},
    }]}
    result = _classify(monkeypatch, body)
    assert result == {"ok": False, "error": "source_not_found"}
    assert investigator._fetch_outcome(URL, result)["error_class"] == "source_not_found"


@pytest.mark.parametrize("fault", [
    "wrong_url", "missing_id", "wrong_code", "wrong_tag", "string_code",
    "bool_code", "extra_status", "text_result", "missing_results",
    "top_error", "status_success", "api_403",
])
def test_captured_not_found_shape_requires_exact_bound_source(monkeypatch, fault):
    body = {"results": [], "statuses": [{
        "id": URL, "status": "error",
        "error": {"httpStatusCode": 404, "tag": "CRAWL_NOT_FOUND"},
    }]}
    source = body["statuses"][0]
    api_status = 200
    if fault == "wrong_url": source["id"] = ALTERNATE
    elif fault == "missing_id": source.pop("id")
    elif fault == "wrong_code": source["error"]["httpStatusCode"] = 410
    elif fault == "wrong_tag": source["error"]["tag"] = "OTHER_NOT_FOUND"
    elif fault == "string_code": source["error"]["httpStatusCode"] = "404"
    elif fault == "bool_code": source["error"]["httpStatusCode"] = True
    elif fault == "extra_status": body["statuses"].append(copy.deepcopy(source))
    elif fault == "text_result": body["results"] = [{"url": URL, "text": QUOTE}]
    elif fault == "missing_results": body.pop("results")
    elif fault == "top_error": body["error"] = {"message": "provider unavailable"}
    elif fault == "status_success": source["status"] = "success"
    elif fault == "api_403": api_status = 403
    assert _classify(monkeypatch, body, api_status).get("error") != "source_not_found"


def test_captured_not_found_receipt_can_complete_only_unproven(monkeypatch):
    body = {"results": [], "statuses": [{
        "id": URL, "status": "error",
        "error": {"httpStatusCode": 404, "tag": "CRAWL_NOT_FOUND"},
    }]}
    result, fetches, searches = _full_review(monkeypatch, response=(200, body))
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert fetches == [URL] and searches
    assert [row["error_class"] for row in result["usage"]["fetch_outcomes"]] == [
        "http_error", "source_not_found",
    ]


@pytest.mark.parametrize("fault", [
    "wrong_url", "missing_id", "wrong_tag", "string_status", "bool_status",
    "duplicate", "text_result", "missing_results", "top_error", "not_error",
])
def test_unbound_exa_404_cannot_claim_source_absence(monkeypatch, fault):
    body = copy.deepcopy(FLIGHTPATH_401)
    source = body["statuses"][0]
    source["error"] = {"httpStatusCode": 404, "tag": "CRAWL_HTTP_404"}
    if fault == "wrong_url": source["id"] = ALTERNATE
    elif fault == "missing_id": source.pop("id")
    elif fault == "wrong_tag": source["error"]["tag"] = "CRAWL_HTTP_410"
    elif fault == "string_status": source["error"]["httpStatusCode"] = "404"
    elif fault == "bool_status": source["error"]["httpStatusCode"] = True
    elif fault == "duplicate": body["statuses"].append(copy.deepcopy(source))
    elif fault == "text_result": body["results"] = [{"url": URL, "text": QUOTE}]
    elif fault == "missing_results": body.pop("results")
    elif fault == "top_error": body["error"] = {"message": "API unavailable"}
    elif fault == "not_error": source["status"] = "success"
    result = _classify(monkeypatch, body)
    assert result["error"] != "source_not_found"


@pytest.mark.parametrize("fault", [
    "wrong_url", "missing_id", "wrong_tag", "string_status", "bool_status",
    "rate_limit", "credit", "server", "success", "duplicate", "text_result",
    "missing_results", "top_error", "top_errors", "not_error",
])
def test_unbound_ambiguous_and_provider_failures_are_not_source_denial(monkeypatch, fault):
    body = copy.deepcopy(FLIGHTPATH_401)
    source = body["statuses"][0]
    if fault == "wrong_url": source["id"] = ALTERNATE
    elif fault == "missing_id": source.pop("id")
    elif fault == "wrong_tag": source["error"]["tag"] = "CRAWL_HTTP_403"
    elif fault == "string_status": source["error"]["httpStatusCode"] = "401"
    elif fault == "bool_status": source["error"]["httpStatusCode"] = True
    elif fault in {"rate_limit", "credit", "server"}:
        status = {"rate_limit": 429, "credit": 402, "server": 503}[fault]
        source["error"] = {"httpStatusCode": status, "tag": f"CRAWL_HTTP_{status}"}
    elif fault == "success": body["statuses"].append({"id": URL, "status": "success"})
    elif fault == "duplicate": body["statuses"].append(copy.deepcopy(source))
    elif fault == "text_result": body["results"] = [{"url": URL, "text": QUOTE}]
    elif fault == "missing_results": body.pop("results")
    elif fault == "top_error": body["error"] = {"message": "API unavailable"}
    elif fault == "top_errors": body["errors"] = ["API unavailable"]
    elif fault == "not_error": source["status"] = "success"
    result = _classify(monkeypatch, body)
    assert result["ok"] is False
    assert result["error"] != "source_access_denied"


@pytest.mark.parametrize("status", [401, 402, 403, 429, 500, 502, 503])
def test_api_http_errors_remain_provider_faults(monkeypatch, status):
    assert _classify(monkeypatch, FLIGHTPATH_401, status) == {
        "ok": False, "error": f"http_{status}",
    }


def _full_review(monkeypatch, *, response=(200, FLIGHTPATH_401), positive=False,
                 unrelated_fault=False, bad_quote=False, required_stage=False,
                 timeout=False):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    requests, fetches, searches = [], [], []
    native_fetch = investigator._fetch_page

    async def timed_out_transport(*args, **kwargs):
        raise asyncio.TimeoutError

    if timeout:
        monkeypatch.setattr(investigator, "_fetch_bounded_html", timed_out_transport)

    async def fetch(_session, url, **kwargs):
        fetches.append(url)
        if timeout and url == URL:
            return await native_fetch(_session, url, **kwargs)
        if url == URL or unrelated_fault:
            return {"ok": False, "error": "http_502"}
        return {"ok": True, "url": url, "final_url": url, "text": QUOTE}

    async def search(*args, **kwargs):
        searches.append(True)
        if required_stage:
            raise RuntimeError("test provider unavailable")
        return {"results": [{"url": ALTERNATE, "title": "Flightpath"}]}

    async def post(_session, endpoint, *, headers, payload):
        if endpoint == "https://api.exa.ai/contents":
            return response
        requests.append(payload)
        forced = payload.get("tool_choice")
        forced = forced.get("function", {}).get("name") if isinstance(forced, dict) else ""
        if len(requests) == 1:
            name, args = "fetch_page", {"url": URL}
        elif forced == "search_web":
            name, args = "search_web", {"query": "Flightpath software"}
        elif (positive or unrelated_fault) and ALTERNATE not in fetches:
            name, args = "fetch_page", {"url": ALTERNATE}
        else:
            target = "stage" if required_stage else "industry"
            name, args = "submit_findings", {"findings": [_finding(
                target, status="VERIFIED" if positive else "UNPROVEN",
                observed_industry="Software" if positive else "",
                activity_role="supplier_operator" if positive else "unresolved",
                evidence_url=ALTERNATE if positive else "",
                evidence_quote=("Fabricated proof" if bad_quote else QUOTE) if positive else "",
            )]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}}]}

    monkeypatch.setattr(investigator, "_fetch_page", fetch)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_post_json", post)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Flightpath", "website": "https://flightpath.fm"},
        targets=("stage",) if required_stage else ("industry",),
        requested_industry="Software", requested_stage="Public" if required_stage else "",
        verified_homepage_identity={"normalized_name": "flightpath",
                                    "registrable_dns_domain": "flightpath.fm"},
    ))
    return result, fetches, searches


def test_flightpath_502_then_precise_401_can_finish_bounded_unproven(monkeypatch):
    result, fetches, searches = _full_review(monkeypatch)
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert searches  # Existing targeted discovery remains mandatory.
    assert fetches == [URL]
    assert [row["error_class"] for row in result["usage"]["fetch_outcomes"]] == [
        "http_error", "source_access_denied"]
    assert result["usage"]["fetch_calls"] == 2


def test_denied_source_does_not_prevent_valid_alternate_positive_claim(monkeypatch):
    result, fetches, _ = _full_review(monkeypatch, positive=True)
    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert fetches == [URL, ALTERNATE]


def test_primary_timeout_can_settle_as_precise_same_source_denial(monkeypatch):
    result, fetches, searches = _full_review(monkeypatch, timeout=True)
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert fetches == [URL] and searches
    assert [row["error_class"] for row in result["usage"]["fetch_outcomes"]] == [
        "fetch_failed", "source_access_denied"]


@pytest.mark.parametrize("status", [401, 402, 403, 429, 503])
def test_primary_timeout_is_not_cleared_by_failed_api_fallback(monkeypatch, status):
    result, _, _ = _full_review(monkeypatch, timeout=True, response=(status, FLIGHTPATH_401))
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] == "provider_error"


def test_precise_denial_cannot_clear_unrelated_transport_fault(monkeypatch):
    result, _, _ = _full_review(monkeypatch, unrelated_fault=True)
    # Give the other URL its own unbound-status failure, not a precise denial.
    assert result["_completed_submit"] is False
    assert result["failure_reason"] == "provider_error"


@pytest.mark.parametrize("status", [401, 402, 403, 429, 503])
def test_api_failures_still_leave_full_investigation_incomplete(monkeypatch, status):
    result, _, _ = _full_review(monkeypatch, response=(status, FLIGHTPATH_401))
    assert result["_completed_submit"] is False
    assert result["failure_reason"] == "provider_error"


def test_required_discovery_failure_cannot_be_cleared_by_source_denial(monkeypatch):
    result, _, _ = _full_review(monkeypatch, required_stage=True)
    assert result["_completed_submit"] is False
    assert result["failure_reason"] == "provider_error"


def test_fabricated_alternate_quote_is_not_accepted(monkeypatch):
    result, _, _ = _full_review(monkeypatch, positive=True, bad_quote=True)
    assert result.get("claims", {}).get("industry", {}).get("status") != "VERIFIED"


def test_finality_accepts_same_source_disposition_but_keeps_other_faults():
    receipt = _receipt()
    receipt["usage"]["fetch_outcomes"].extend([
        {"url_sha256": "c" * 64, "ok": False, "error_class": "http_error"},
        {"url_sha256": "c" * 64, "ok": False, "error_class": "source_access_denied"},
    ])
    kwargs = {"identity_receipt": _identity(), "investigation_receipt": receipt}
    assert lead_scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("identity", "industry", "stage"), **kwargs)
    receipt["usage"]["fetch_outcomes"].append(
        {"url_sha256": "d" * 64, "ok": False, "error_class": "fetch_failed"})
    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("identity", "industry", "stage"), **kwargs)


@pytest.mark.parametrize("semantic", [False, True])
def test_exact_investigator_outcomes_reach_fit_and_arena_finality(monkeypatch, semantic):
    result, _, _ = _full_review(monkeypatch)
    assert result["_completed_submit"] is True
    original_receipt = fit_controls._receipt

    def receipt():
        value = original_receipt()
        value["usage"] = copy.deepcopy(result["usage"])
        return value

    monkeypatch.setattr(fit_controls, "_receipt", receipt)
    # Existing validated-negative fixture exercises actual fit-finality and
    # Arena score_work_item. Only external provider/identity responses are
    # substituted, as in its positive and genuine-outage controls.
    fit_controls.test_negative_completion_through_fit_and_arena(
        monkeypatch, semantic=semantic, homepage_fault=None,
        investigation_fault="", preserve_marker=True)
