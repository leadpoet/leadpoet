"""A native target HTTP500 permits existing discovery, never a made-up verdict."""
import asyncio
import json
from copy import deepcopy
from pathlib import Path

import pytest

from qualification.scoring import competition
from qualification.scoring import intent_verification_three_stage as intent
from tests.test_intent_failed_archive_same_event import install_exact_failures, new_failure
from tests.test_intent_unavailable_same_event_source import ORIGINAL, run_verifier
from tests.test_intent_wayback_empty_body import SOURCE


def test_retained_teleco_http_responses_reach_discovery_without_proving_absence(monkeypatch):
    from tests import test_intent_failed_archive_same_event as native
    case = json.loads((Path(__file__).parent / "fixtures/company_evidence/teleco_target_server_error.json").read_text())
    monkeypatch.setattr(native, "SOURCE", case["source_url"])
    monkeypatch.setattr(native, "SNAPSHOT", case["snapshot_url"])
    monkeypatch.setattr(native, "TIMESTAMP", "20260313143608")
    native.install_exact_failures(monkeypatch, final_snapshot=case["snapshot_url"],
                                 exa_documents=case["exa_documents"], archive=case["archive_html"])
    result = asyncio.run(intent._fetch_sd_then_exa([case["source_url"]]))
    assert intent._unusable_source_allows_same_event_search(result, case["source_url"])
    assert not intent._confirmed_source_absence(result, case["source_url"])
    assert not competition._intent_attempts_have_exact_target_crawl_exhaustion(
        intent._project_contents_for_prompt(result)["statuses"])


def native_target_error():
    # Exact Nov20 provider response shape. Outer HTTP is 200, not 500.
    return {"requestId": "retained-exa-target-failure", "results": [], "statuses": [{
        "id": SOURCE, "status": "error",
        "error": {"httpStatusCode": 500, "tag": "CRAWL_HTTP_500"},
    }]}


def discovery_failure():
    result = new_failure()
    for row in result["statuses"][0]["exa_target_crawl_failure"]["observations"]:
        row["error_tag"] = "CRAWL_HTTP_500"
    return result


def test_completed_target_server_error_reaches_existing_search_not_terminal_zero(monkeypatch):
    calls = install_exact_failures(monkeypatch, exa_documents=[native_target_error()] * 2)
    result = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    assert not result["results"]
    assert intent._unusable_source_allows_same_event_search(result, SOURCE)
    assert not intent._confirmed_source_absence(result, SOURCE)
    assert not competition._intent_attempts_have_exact_target_crawl_exhaustion(
        intent._project_contents_for_prompt(result)["statuses"])
    assert [method for method, _ in calls] == ["GET", "GET", "GET", "POST", "POST"]


@pytest.mark.parametrize("case", ["outer500", "outer401", "outer429", "wrong_target", "wrong_tag", "pending", "one_failure"])
def test_provider_or_unbound_failures_remain_unavailable(monkeypatch, case):
    docs = [native_target_error(), native_target_error()]
    status = 200
    if case.startswith("outer"):
        status = int(case[5:])
    elif case == "wrong_target":
        docs[1]["statuses"][0]["id"] = "https://other.example/"
    elif case == "wrong_tag":
        docs[1]["statuses"][0]["error"]["tag"] = "PROVIDER_ERROR"
    elif case == "pending":
        docs[1]["statuses"][0]["status"] = "pending"
    else:
        docs[1]["statuses"] = []
    install_exact_failures(monkeypatch, exa_documents=docs, exa_http_status=status)
    result = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    assert not intent._unusable_source_allows_same_event_search(result, SOURCE)
    assert not intent._confirmed_source_absence(result, SOURCE)


@pytest.mark.parametrize("search", [None, {"status": "unavailable", "reason": "search_provider_unavailable"},
                                     {"status": "unproven", "reason": "no_same_event_locator"}])
def test_existing_search_requires_verified_evidence_and_does_not_change_exhaustion(monkeypatch, search):
    result, fetch, discover, judge = asyncio.run(run_verifier(
        monkeypatch, original=discovery_failure(), search=search))
    assert discover.await_count == 1
    if search is None:
        assert result["client_ready"]
        assert fetch.await_count == 2 and judge.await_count == 1
    else:
        assert result["decision"] == "unavailable"
        assert fetch.await_count == 1 and judge.await_count == 0
        from tests.test_same_event_source_local_finality import _propagate_to_breakdown
        row = _propagate_to_breakdown(monkeypatch, result)
        assert competition.scorer_breakdown_has_retryable_infrastructure_failure(row, integrity_policy=True)
        assert not competition.scorer_breakdown_has_company_local_verification_failure(row)


@pytest.mark.parametrize("mutation", ["archive_missing", "archive_foreign", "one_attempt"])
def test_target_error_alone_does_not_expand_discovery(monkeypatch, mutation):
    original = deepcopy(discovery_failure())
    status = original["statuses"][0]
    if mutation == "archive_missing":
        status.pop("sd_archive_failure")
    elif mutation == "archive_foreign":
        status["sd_archive_failure"]["requested_url"] = "https://other.example/"
    else:
        status["exa_target_crawl_failure"]["confirmed_attempts"] = 1
    result, _, discover, judge = asyncio.run(run_verifier(monkeypatch, original=original))
    assert result["decision"] == "unavailable"
    discover.assert_not_awaited(); judge.assert_not_awaited()
    assert not intent._confirmed_source_absence(original, ORIGINAL)
