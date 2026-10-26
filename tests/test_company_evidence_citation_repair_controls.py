import asyncio
import json

import pytest
from qualification.scoring import company_evidence_investigator as inv
from tests.product_coverage_fixtures import native_product_post
from tests.test_company_evidence_judgment_reserve import (
    _Clock,
    _finding,
    _response,
    QUOTE,
    URL,
)

STAGE_URL = "https://acme.example/series-a"
STAGE_QUOTE = "Acme today announced a $9 million Series A round of funding led by Example Ventures."


def stage(status="VERIFIED", missing=False):
    f = dict(
        _finding(status=status),
        target="stage",
        observed_value="Series A" if status != "UNPROVEN" else None,
        old_name="Acme",
        new_name="Acme",
        old_domain="acme.example",
        new_domain="acme.example",
        shared_linkedin_slug="acme",
    )
    f["evidence_url"] = "" if missing or status == "UNPROVEN" else STAGE_URL
    f["evidence_quote"] = "" if missing or status == "UNPROVEN" else STAGE_QUOTE
    return f


def run(monkeypatch, post, search, targets, *, fetch=None):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("EXA_API_KEY", "test")
    monkeypatch.setattr(inv, "time", _Clock())
    monkeypatch.setattr(inv, "_post_json", native_product_post(post))
    monkeypatch.setattr(inv, "_search_web", search)

    async def no_fetch(*args, **kwargs):
        raise AssertionError("no fresh fetch needed")

    monkeypatch.setattr(inv, "_fetch_page", fetch or no_fetch)
    return asyncio.run(
        inv.investigate_company_evidence(
            company_locator={"name": "Acme", "website": "https://acme.example"},
            targets=targets,
            requested_stage="Series A",
            requested_industry="Education",
            requested_subindustry="Higher education services",
            requested_product_service="Enrollment software for universities",
            prior_observations={"submitted_source_urls": [URL, STAGE_URL]},
            verified_homepage_identity={
                "normalized_name": "acme",
                "registrable_dns_domain": "acme.example",
                "linkedin_company_slug": "acme",
            },
            prefetched_pages={
                URL: {"final_url": URL, "text": QUOTE},
                STAGE_URL: {"final_url": STAGE_URL, "text": STAGE_QUOTE},
            },
        )
    )


@pytest.mark.parametrize("search_fails", [False, True])
def test_actual_venture_stage_missing_citation_repair_keeps_required_discovery(
    monkeypatch, search_fails
):
    requests = []
    searches = []

    async def search(session, query, **kwargs):
        searches.append(query)
        if search_fails:
            raise RuntimeError("provider unavailable")
        return {"results": []}

    async def post(session, url, *, headers, payload):
        requests.append(payload)
        assert len(requests) <= 2
        if len(requests) == 1:
            return 200, _response(
                "submit_findings", {"findings": [stage(missing=True)]}
            )
        feedback = json.loads(
            next(
                m["content"]
                for m in reversed(payload["messages"])
                if m["role"] == "tool"
            )
        )
        assert feedback["rejected_findings"][0]["reason"] == inv.MISSING_CITATION_REASON
        assert "No passage was rejected for those findings" in feedback["instruction"]
        return 200, _response(
            "submit_findings",
            {"findings": [stage("UNPROVEN" if search_fails else "VERIFIED")]},
            call=2,
        )

    result = run(monkeypatch, post, search, ("stage",))
    assert len(searches) == 1  # mandatory current-stage discovery, no new repair search
    assert len(requests) == 2
    assert result["usage"]["fetch_calls"] == 0
    if search_fails:
        assert result["_completed_submit"] is False
        assert result["failure_reason"] == "provider_error"
    else:
        assert result["_completed_submit"] is True
        assert result["claims"]["stage"]["status"] == "VERIFIED"
        assert result["_validated_stage_finding"]["evidence_quote"] == STAGE_QUOTE


def test_mixed_missing_stage_and_fabricated_industry_keep_distinct_feedback(
    monkeypatch,
):
    requests = []
    searches = []

    async def search(session, query, **kwargs):
        searches.append(query)
        return {"results": [{"url": URL}]}

    async def post(session, url, *, headers, payload):
        requests.append(payload)
        assert len(requests) <= 4
        if len(requests) == 1:
            return 200, _response(
                "submit_findings",
                {
                    "findings": [
                        stage(missing=True),
                        _finding(quote="Acme manufactures passenger aircraft."),
                    ]
                },
            )
        if len(requests) == 3:
            return 200, _response(
                "submit_findings", {"findings": [stage(), _finding()]}, call=3,
            )
        feedback = json.loads(
            next(
                m["content"]
                for m in reversed(payload["messages"])
                if m["role"] == "tool"
            )
        )
        reasons = {x["target"]: x["reason"] for x in feedback["rejected_findings"]}
        assert reasons["stage"] == inv.MISSING_CITATION_REASON
        assert (
            reasons["industry"] == "submitted quote was not present in fetched source"
        )
        assert (
            "For findings with missing or invalid citation fields"
            in feedback["instruction"]
        )
        assert "No passage was rejected for those findings" in feedback["instruction"]
        assert "Never repeat a rejected quote." in feedback["instruction"]
        if len(requests) == 2:
            assert payload["tool_choice"] == "required"
            return 200, _response(
                "search_web", {"query": "Acme enrollment software for universities"}, call=2,
            )
        raise AssertionError("unexpected investigator turn")

    async def fetch(session, url):
        assert url == URL
        return {"ok": True, "url": URL, "final_url": URL, "text": QUOTE}

    result = run(monkeypatch, post, search, ("stage", "industry"), fetch=fetch)
    assert len(requests) == 3 and len(searches) == 2
    assert result["_completed_submit"] is True
    assert (
        result["claims"]["stage"]["status"]
        == result["claims"]["industry"]["status"]
        == "VERIFIED"
    )
