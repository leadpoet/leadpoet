"""Exact-profile source failures must not become shared provider failures."""

import asyncio
import hashlib

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer, linkedin_company_size
from tests.test_company_evidence_investigator import _finding
from tests.test_company_evidence_judgment_reserve import _response
from tests.test_current_linkedin_company_size import _install_exa_bodies


PROFILE = "https://www.linkedin.com/company/acme"
PAGES = ("https://acme.example/about", "https://acme.example/product")
TARGETS = ("headcount", "industry", "required_attribute")


def _run(monkeypatch, case):
    code, tag, profile = {
        "blocked409": (409, "CRAWL_NON_CANONICAL", PROFILE),
        "blocked504": (504, "CRAWL_LIVECRAWL_TIMEOUT", PROFILE),
        "crawl502": (502, "CRAWL_INFRASTRUCTURE_ERROR", PROFILE),
        "crawl504": (504, "CRAWL_OTHER", PROFILE),
        "unbound504": (504, "CRAWL_LIVECRAWL_TIMEOUT", PROFILE + "-other"),
    }.get(case, (409, "CRAWL_NON_CANONICAL", PROFILE))
    body = {"statuses": [{"id": profile, "status": "error",
                          "error": {"httpStatusCode": code, "tag": tag}}], "results": []}
    if case == "malformed":
        body = []
    elif case == "wall":
        body = {"statuses": [{"status": "success"}], "results": [{
            "url": PROFILE, "text": (
                "Cadastre-se | LinkedIn\n# Cadastre-se no LinkedIn\nE-mail\n"
                "Senha (+ de 6 caracteres)\n## Entrar\nE-mail ou telefone\nSenha\n"
            ),
        }]}
    profile_calls, pending_bodies = _install_exa_bodies(monkeypatch, body)
    if case in {"outer504", "outer_timeout"}:
        session_type = linkedin_company_size.aiohttp.ClientSession
        original_post = session_type.post

        def outer_failure(session, *args, **kwargs):
            response = original_post(session, *args, **kwargs)
            if case == "outer_timeout":
                raise asyncio.TimeoutError("controlled outer transport timeout")
            response.status = 504
            return response

        monkeypatch.setattr(session_type, "post", outer_failure)
    pending_urls = [PROFILE, *PAGES]
    search_calls = []
    reasoning_calls = []
    search_requested = False

    async def fetch(_session, url, **_kwargs):
        assert url in PAGES
        return 200, url, "Acme maintains a company information page."

    async def search(_session, query, **_kwargs):
        search_calls.append(query)
        if case == "required_search_failure":
            raise RuntimeError("controlled search provider failure")
        return {"results": [], "notice": "discovery_only_not_evidence"}

    targets = (*TARGETS, "stage") if case == "required_search_failure" else TARGETS

    async def post(_session, _url, *, headers, payload):
        nonlocal search_requested
        reasoning_calls.append(payload)
        forced = payload.get("tool_choice")
        if pending_urls:
            name, arguments = "fetch_page", {"url": pending_urls.pop(0)}
        elif not search_requested or (
            isinstance(forced, dict) and forced["function"]["name"] == "search_web"
        ):
            search_requested = True
            name, arguments = "search_web", {"query": "Acme company evidence"}
        else:
            findings = [_finding(
                target, status="UNPROVEN", observed_value=None,
                evidence_url="", evidence_quote="", reason="No company-bound proof found.",
            ) for target in targets]
            if case == "invalid_evidence":
                findings[1] = _finding(
                    "industry", observed_industry="Software", activity_role="supplier_operator",
                    evidence_url=PAGES[0], evidence_quote="Invented proof absent from that page.",
                )
            name, arguments = "submit_findings", {"findings": findings}
        return 200, _response(name, arguments, call=len(reasoning_calls))

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-openrouter-key")
    monkeypatch.setattr(investigator, "_fetch_bounded_html", fetch)
    monkeypatch.setattr(investigator, "_search_web", search)
    monkeypatch.setattr(investigator, "_post_json", post)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example", "linkedin": PROFILE},
        targets=targets, requested_industry="Software", requested_attribute="Supplies software.",
        requested_stage="Public" if case == "required_search_failure" else "",
        verified_homepage_identity={"normalized_name": "acme", "registrable_dns_domain": "acme.example"},
    ))
    assert not pending_urls and not pending_bodies
    assert len(profile_calls) == 1
    assert search_calls
    return result


@pytest.mark.parametrize("case", ["blocked409", "blocked504", "wall"])
def test_typed_linkedin_block_plus_successful_sources_can_complete_unproven(monkeypatch, case):
    result = _run(monkeypatch, case)
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    assert set(result["claims"]) == set(TARGETS)
    assert all(finding["status"] == "UNPROVEN" for finding in result["claims"].values())
    outcomes = result["usage"]["fetch_outcomes"]
    assert outcomes[0] == {
        "url_sha256": hashlib.sha256(PROFILE.encode()).hexdigest(),
        "ok": False, "error_class": "source_access_denied", "loaded_text_length": 0,
    }
    assert len(outcomes) == 3 and all(outcome["ok"] for outcome in outcomes[1:])
    # The existing fit finality consumer already recognizes this canonical
    # source-local class; no separate permissive exception is needed there.
    receipt = {"gate": "company_evidence_investigation", "targets": list(TARGETS),
               "claims": result["claims"], "usage": result["usage"],
               "completed_submitted_findings": True, "failure_reason": ""}
    assert lead_scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("employee_size", "industry"), investigation_receipt=receipt,
    )


@pytest.mark.parametrize("case", [
    "crawl502", "crawl504", "unbound504", "outer504", "outer_timeout",
    "malformed", "invalid_evidence", "required_search_failure",
])
def test_genuine_provider_and_incomplete_evidence_failures_remain_retryable(monkeypatch, case):
    result = _run(monkeypatch, case)
    assert result.get("_completed_submit") is not True
    assert result["failure_reason"] in {"provider_error", "malformed_response"}
    assert all(finding["status"] != "VERIFIED" for finding in result["claims"].values())
