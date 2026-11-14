"""An admitted submission that settles late gets one bounded correction."""

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from tests.test_company_evidence_judgment_reserve import (
    QUOTE, _Clock, _finding, _response, _run_industry,
)


@pytest.mark.parametrize("settled_at", [110.0, 180.0])
@pytest.mark.parametrize("final", ["verified", "unproven", "bad_quote", "provider", "research"])
def test_late_ordinary_submission_gets_only_one_submit_correction(monkeypatch, settled_at, final):
    clock = _Clock()
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        if len(requests) == 1:
            clock.now = settled_at
            return 200, _response("submit_findings", {
                "findings": [_finding(url="", quote="")],
            })
        assert len(requests) == 2, "no third call or repeated correction"
        assert payload["tool_choice"]["function"]["name"] == "submit_findings"
        assert [t["function"]["name"] for t in payload["tools"]] == ["submit_findings"]
        if final == "provider":
            return 503, {"error": "provider unavailable"}
        if final == "research":
            return 200, _response("fetch_page", {"url": "https://acme.example/other"}, call=2)
        finding = (
            _finding(status="UNPROVEN") if final == "unproven"
            else _finding(quote="Acme manufactures aircraft.") if final == "bad_quote"
            else _finding()
        )
        return 200, _response("submit_findings", {"findings": [finding]}, call=2)

    async def no_research(*args, **kwargs):
        raise AssertionError("no new research after admission closes")

    monkeypatch.setattr(investigator, "_search_web", no_research)
    monkeypatch.setattr(investigator, "_fetch_page", no_research)
    result = _run_industry(monkeypatch, post, clock)
    assert len(requests) == 2
    assert (result.get("_completed_submit") is True) is (final in {"verified", "unproven"})
    assert result["failure_reason"] == (
        "" if final in {"verified", "unproven"}
        else "provider_error" if final == "provider" else "malformed_response"
    )
    if final == "verified":
        assert result["claims"]["industry"]["evidence_quote"] == QUOTE
    elif final == "bad_quote":
        assert result["claims"]["industry"]["status"] == "UNPROVEN"
        assert not result["claims"]["industry"]["evidence_quote"]


@pytest.mark.parametrize("status", ["VERIFIED", "UNPROVEN"])
def test_valid_late_submission_finishes_without_another_call(monkeypatch, status):
    clock = _Clock()
    calls = []

    async def post(_session, _url, *, headers, payload):
        calls.append(payload)
        assert len(calls) == 1
        clock.now = 180.0
        return 200, _response("submit_findings", {"findings": [_finding(status=status)]})

    result = _run_industry(monkeypatch, post, clock)
    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == status
    assert len(calls) == 1
