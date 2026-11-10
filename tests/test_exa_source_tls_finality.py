"""Captured Exa target TLS errors remain local; API faults stay infrastructure."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests import test_investigator_source_access_denied as existing

CAPTURED = json.loads((Path(__file__).parent / "fixtures" /
    "exa_technimark_tls_error_20261109.json").read_text())
URL = CAPTURED["statuses"][0]["id"]


def classify(monkeypatch, body, status=200):
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setattr(investigator, "_post_json", AsyncMock(return_value=(status, body)))
    return asyncio.run(investigator._fetch_exa_exact_page(object(), URL))


def test_captured_exact_target_ssl_error_is_source_local(monkeypatch):
    result = classify(monkeypatch, CAPTURED)
    assert result == {"ok": False, "error": "source_access_denied"}
    assert investigator._fetch_outcome(URL, result)["error_class"] == "source_access_denied"


@pytest.mark.parametrize("positive", [False, True])
def test_target_ssl_keeps_alternate_discovery_and_bounded_finality(monkeypatch, positive):
    body = deepcopy(CAPTURED)
    body["statuses"][0]["id"] = existing.URL
    result, fetches, searches = existing._full_review(monkeypatch, response=(200, body), positive=positive)
    assert searches or positive
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert result["claims"]["industry"]["status"] == ("VERIFIED" if positive else "UNPROVEN")
    assert fetches == ([existing.URL, existing.ALTERNATE] if positive else [existing.URL])


@pytest.mark.parametrize("fault", ["wrong_url", "missing_id", "missing_results", "nonempty_results",
    "duplicate", "top_error", "top_errors", "wrong_status", "string_status", "bool_status",
    "wrong_tag", "not_error", "api_auth", "api_credits", "api_limit", "api_server"])
def test_target_ssl_requires_exact_bound_source_and_successful_api(monkeypatch, fault):
    body = deepcopy(CAPTURED)
    source = body["statuses"][0]
    status = 200
    if fault == "wrong_url": source["id"] = URL + "unrelated"
    elif fault == "missing_id": source.pop("id")
    elif fault == "missing_results": body.pop("results")
    elif fault == "nonempty_results": body["results"] = [{"url": URL, "text": "partial"}]
    elif fault == "duplicate": body["statuses"].append(deepcopy(source))
    elif fault == "top_error": body["error"] = {"message": "provider unavailable"}
    elif fault == "top_errors": body["errors"] = ["provider unavailable"]
    elif fault == "wrong_status": source["error"]["httpStatusCode"] = 503
    elif fault == "string_status": source["error"]["httpStatusCode"] = "500"
    elif fault == "bool_status": source["error"]["httpStatusCode"] = True
    elif fault == "wrong_tag": source["error"]["tag"] = "CRAWL_HTTP_500"
    elif fault == "not_error": source["status"] = "success"
    elif fault.startswith("api_"): status = {"api_auth": 401, "api_credits": 402, "api_limit": 429, "api_server": 500}[fault]
    assert classify(monkeypatch, body, status).get("error") != "source_access_denied"


@pytest.mark.parametrize("semantic", [False, True])
def test_target_ssl_reaches_real_fit_and_arena_finality(monkeypatch, semantic):
    body = deepcopy(CAPTURED)
    body["statuses"][0]["id"] = existing.URL
    result, _, _ = existing._full_review(monkeypatch, response=(200, body))
    assert result["_completed_submit"] is True
    original = existing.fit_controls._receipt
    def receipt():
        value = original()
        value["usage"] = deepcopy(result["usage"])
        return value
    monkeypatch.setattr(existing.fit_controls, "_receipt", receipt)
    existing.fit_controls.test_negative_completion_through_fit_and_arena(
        monkeypatch, semantic=semantic, homepage_fault=None,
        investigation_fault="", preserve_marker=True)


@pytest.mark.parametrize("status", [401, 402, 429, 500, 502, 503])
def test_target_tls_payload_cannot_hide_failed_api_or_complete_review(monkeypatch, status):
    body = deepcopy(CAPTURED)
    body["statuses"][0]["id"] = existing.URL
    result, _, _ = existing._full_review(monkeypatch, response=(status, body))
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] == "provider_error"


def test_target_tls_cannot_clear_another_sources_transport_fault(monkeypatch):
    body = deepcopy(CAPTURED)
    body["statuses"][0]["id"] = existing.URL
    result, _, _ = existing._full_review(monkeypatch, response=(200, body), unrelated_fault=True)
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] == "provider_error"
