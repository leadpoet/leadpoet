"""A completed evidence search can exhaust an unusable optional profile."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json

import pytest

from qualification.scoring import lead_scorer as s, linkedin_company_size as l
from qualification.scoring.company_fit_decision import company_fit_match
from tests.test_current_linkedin_company_size import _company, _icp, _install_exa_bodies, _verdict


def _install(monkeypatch, guard="complete"):
    company = _company(linkedin="")
    company.company_name = "ACM.US"
    company.company_website = "https://acm.us"
    icp = _icp()
    icp.company_stage = "Series A"
    icp.intent_signals = ["Announced a product launch"]
    icp.required_attribute = "Provides software and is actively growing."
    verdict = {k: v.replace("acme.example.com", "acm.us").replace("/company/acme", "/company/acmconsultinggroup") if isinstance(v, str) else v
               for k, v in _verdict().items()}
    verdict.update(observed_company_name="ACM", observed_company_website=company.company_website,
                   observed_company_linkedin="https://www.linkedin.com/company/acmconsultinggroup",
                   observed_company_stage="Private Equity", stage_matches=False, stage_evidence_url="https://acm.us/investment", stage_evidence_quote="ACM has private equity backing.",
                   attribute_satisfied=True, required_attribute_evidence_url="https://acm.us/product", required_attribute_evidence_quote="ACM provides workflow software.")
    claims = {target: {"target": target, "status": "UNPROVEN", "evidence_url": "", "evidence_quote": ""}
              for target in ("rebrand", "headcount", "stage", "required_attribute")}
    claims["industry"] = {"target": "industry", "status": "VERIFIED", "activity_role": "supplier_operator",
                         "observed_industry": "Software", "observed_subindustry": "SaaS",
                         "evidence_url": "https://acm.us/product",
                         "evidence_quote": "ACM provides workflow software for customers."}
    outcome = {"ok": True, "loaded_text_length": 100,
               "url_sha256": hashlib.sha256(b"https://acm.us/product").hexdigest()}
    review = {"claims": claims, "_completed_submit": True, "failure_reason": "",
              "usage": {"fetch_outcomes": [outcome]}}
    not_found = {"results": [], "statuses": [{"id": "https://www.linkedin.com/company/acmconsultinggroup",
        "status": "error", "error": {"httpStatusCode": 404, "tag": "CRAWL_NOT_FOUND"}}]}
    profile = {"status": "completed", "result": {"data": {"error": None, "status": 200, "element": None}}}
    if guard in {"403", "429", "502", "503"}:
        profile["result"]["data"]["status"] = int(guard)
    _install_exa_bodies(monkeypatch, not_found, not_found, profile)
    for key in ("DEEPLINE_API_KEY", "EXA_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(key, "offline-key")
    if guard == "incomplete": review["_completed_submit"] = False
    if guard == "failure": review["failure_reason"] = "provider_error"
    if guard == "missing_claim": del claims["headcount"]
    if guard == "wrong_claim": claims["headcount"]["target"] = "stage"
    if guard == "positive_missing_proof": claims["headcount"]["status"] = "VERIFIED"
    if guard == "cited_unproven": claims["headcount"]["evidence_url"] = "https://acm.us/unknown"
    if guard == "no_fetches": review["usage"]["fetch_outcomes"] = []
    if guard == "empty_text": outcome["loaded_text_length"] = 0
    if guard == "bool_length": outcome["loaded_text_length"] = True
    if guard == "unbound": outcome.pop("url_sha256")
    if guard == "bad_hash": outcome["url_sha256"] = "not-a-hash"
    if guard in {"transport", "unbound_transport"}:
        review["usage"]["fetch_outcomes"].append({"ok": False, "error_class": "http_error",
            **({"url_sha256": "f" * 64} if guard == "transport" else {})})

    async def homepage(*args, **kwargs):
        return company_fit_match(details={"identity": {
            "decision": "match", "evidence_source": "company_homepage",
            "submitted_name": "acmus", "observed_name": "acmus",
            "submitted_domain": "acm.us", "observed_domain": "acm.us",
            "submitted_linkedin_slug": "", "observed_linkedin_slug": "acmconsultinggroup",
        }, "verified_homepage_transport_domain": "acm.us"})

    async def grounding(*args, **kwargs): return 200, "https://acm.us/product", "ACM provides workflow software."
    async def broad(**kwargs): return copy.deepcopy(verdict), ""
    async def investigate(**kwargs): return copy.deepcopy(review)
    actual_refresh = s._refresh_linkedin_employee_size_observation

    async def refresh(*args, **kwargs):
        value = await actual_refresh(*args, **kwargs)
        cache = kwargs["invocation_cache"]
        if guard in {"timeout", "provider_fault"}:
            cache["evidence"] = None
            cache[s.VERIFIER_FAILURE_DETAIL_KEY] = "provider_error"
        if guard == "structured_flag": cache["structured_failure"] = False
        if guard == "structured_reason": cache["structured_failure_reason"] = "provider_error"
        if guard == "diagnostic_reason": cache[s.VERIFIER_FAILURE_DETAIL_KEY] = "provider_error"
        if guard == "current_outcome": cache["evidence"]["outcome"] = "unrecognized"
        return value

    monkeypatch.setattr(s, "_fetch_bounded_html", grounding)
    monkeypatch.setattr(s, "verify_company_exists", homepage)
    monkeypatch.setattr(s, "_request_company_reverify_json", broad)
    monkeypatch.setattr(s, "investigate_company_evidence", investigate)
    monkeypatch.setattr(s, "_refresh_linkedin_employee_size_observation", refresh)
    actual_targeted = s._run_targeted_company_evidence_investigation
    async def targeted(**kwargs):
        value = await actual_targeted(**kwargs)
        if guard == "wrong_gate": value[1].details["investigation_receipt"]["gate"] = "unverified"
        if guard == "missing_target": value[1].details["investigation_receipt"]["targets"].remove("headcount")
        if guard == "attribute_grounding": value[0][s._REQUIRED_ATTRIBUTE_GROUNDING]["status"] = "failed"
        return value
    monkeypatch.setattr(s, "_run_targeted_company_evidence_investigation", targeted)
    return company, icp


@pytest.mark.parametrize("guard", ["complete", "403", "429", "502", "503", "timeout", "provider_fault",
    "incomplete", "failure", "missing_claim", "wrong_claim", "positive_missing_proof", "cited_unproven",
    "no_fetches", "empty_text", "bool_length", "unbound", "bad_hash", "transport", "unbound_transport",
    "structured_flag", "structured_reason", "diagnostic_reason", "current_outcome", "wrong_gate", "missing_target", "attribute_grounding"])
def test_completed_search_and_every_failure_guard(monkeypatch, guard):
    company, icp = _install(monkeypatch, guard)
    result = asyncio.run(s._verify_company_fit(company, icp, 0, 0, set(),
        require_https_transport=True, evidence_investigator=True))
    assert result.decision == "unavailable"
    assert result.details.get("failure_class") == ("insufficient_fit_evidence" if guard == "complete"
        else "employee_size_verification_failed")
    if guard == "complete":
        assert next(r for r in result.details["supporting_receipts"] if r.get("gate") == "company_evidence_investigation")["prior_linkedin_refresh_failure_reason"] == "malformed_response"
    else:
        assert result.details.get(s.VERIFIER_FAILURE_DETAIL_KEY) in {"malformed_response", "provider_error"}


def test_terminal_negative_does_not_fail_group_or_replace_positive_sibling(monkeypatch):
    from lab_arena import scoring
    from tests.test_arena_score_integrity import _positive_breakdown, _public_company
    company, icp = _install(monkeypatch)
    calls = []
    actual = s.score_company_competition_intent

    async def prechecks(*args, **kwargs): return company_fit_match("Prechecks passed")
    async def score(**kwargs):
        calls.append(kwargs["company"].company_name)
        if kwargs["company"].company_name == "Sibling": return s.LeadScoreBreakdown(**_positive_breakdown("Sibling", "sibling.example"))
        return await actual(**kwargs)

    monkeypatch.setattr(s, "run_company_zero_checks", prechecks)
    monkeypatch.setattr(s, "score_company_competition_intent", score)
    policy = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2")
    rows = scoring.score_work_item({"scored_run_id": "completed-review"}, icp=icp.model_dump(mode="json"),
        companies=[{**_public_company("ACM.US", domain="acm.us"), "company_stage": "Series A", "employee_count": "11-50"}, {**_public_company("Sibling", domain="sibling.example"), "employee_count": "11-50"}],
        scorer=scoring.lab_scorer(policy), max_retries=3)
    assert calls == ["ACM.US", "Sibling"]
    assert len(rows) == 2 and rows[0]["final_score"] == 0 and rows[0]["company_qualified"] is False
    assert rows[1]["final_score"] > 0 and rows[1]["company_qualified"] is True


def test_valid_profile_still_supplies_bound_employee_size(monkeypatch):
    from tests.test_current_linkedin_company_size import _structured_company_payload
    _install_exa_bodies(monkeypatch, _structured_company_payload())
    monkeypatch.setenv("DEEPLINE_API_KEY", "offline-key")
    diagnostic = {}
    result = asyncio.run(l.fetch_structured_linkedin_company_size("acme.example.com",
        "https://www.linkedin.com/company/acme", diagnostic=diagnostic))
    assert result["employee_count"] == "11-50" and diagnostic == {}


def test_exact_captured_review_with_recorded_timing(monkeypatch):
    """Opt-in private packet replay; no customer bodies enter source control."""
    import base64
    import os
    from datetime import datetime
    from pathlib import Path
    from types import SimpleNamespace
    from qualification.scoring import company_evidence_investigator as investigator
    from qualification.scoring import company_verification
    from qualification.scoring.competition import _normalized_company, _normalized_icp
    from gateway.qualification.models import CompanyOutput, ICPPrompt

    capture_root = os.environ.get("ARENA_CAPTURE_REPLAY_ROOT")
    if not capture_root:
        pytest.skip("Private capture replay is opt-in")
    root = Path(capture_root)
    packets = json.loads((root / "recovery46c-target3-score1-bodies-private.json").read_text())
    events = json.loads((root / "recovery46c-target3-score1-events-private.json").read_text())
    requests = [event["content"] for event in events if event["event_kind"] == "provider.request"]
    responses = [event for event in events if event["event_kind"] == "provider.response"]
    bundle = json.loads((root / "recovery46c-validation-bundle-2226-private.json").read_text())
    benchmark = next(obj["document"] for obj in bundle["objects"] if obj["ref"].endswith("/benchmark.json"))
    source = json.loads((root / "target3-original-execute-private.json").read_text())[0]["document"]["companies"][0]
    company = CompanyOutput(**_normalized_company(source, integrity_policy=True))
    icp = ICPPrompt(**_normalized_icp(benchmark["icps"][2]))

    def raw(index): return base64.b64decode(packets[index]["body_b64"]).decode()
    def body(index): return json.loads(raw(index))
    def timestamp(value):
        date, fraction = value.split("+")[0].split(".")
        return datetime.strptime(date + "." + fraction.ljust(6, "0"), "%Y-%m-%d %H:%M:%S.%f").timestamp()

    start = timestamp(next(event["occurred_at"] for event in events
        if event["event_kind"] == "provider.request" and event["content"] is requests[45]))
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    def advance(index):
        if index >= 45: clock.now = timestamp(responses[index]["occurred_at"]) - start
    exa = [42, 43]
    model = [47, 49, 51, 53, 55, 56, 57]
    search = [46, 50]
    html = {requests[index]["parameters"]["url"]: index for index in (39, 41, 45, 48, 52, 54)}

    class Response:
        status = 200
        def __init__(self, value): self.value = value
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def json(self): return self.value

    class Session:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        def post(self, url, **kwargs):
            return Response(body(exa.pop(0) if url == "https://api.exa.ai/contents" else 44))

    async def captured_html(_session, url, **kwargs):
        target = kwargs.get("params", {}).get("url", url)
        index = html[target]
        advance(index)
        return 200, target, raw(index)

    async def captured_post(_session, url, *, headers, payload):
        index = (model if "openrouter" in url else search).pop(0)
        if index == 57:
            assert len(payload["tools"]) == 1
            assert "single final submit-only correction" in payload["messages"][-1]["content"]
        advance(index)
        return 200, body(index)

    async def broad(**kwargs): return json.loads(body(40)["choices"][0]["message"]["content"]), ""
    actual_investigation = investigator.investigate_company_evidence
    completed = []
    async def investigate(**kwargs):
        with monkeypatch.context() as context:
            context.setattr(investigator, "time", clock)
            result = await actual_investigation(**kwargs)
        assert result["_completed_submit"] is True and result["failure_reason"] == ""
        assert result["usage"]["fetch_calls"] == 4
        assert all(row["ok"] is True and row["loaded_text_length"] > 0
                   for row in result["usage"]["fetch_outcomes"])
        completed.append(result)
        return result

    for key in ("DEEPLINE_API_KEY", "EXA_API_KEY", "OPENROUTER_API_KEY", "SCRAPINGDOG_API_KEY"):
        monkeypatch.setenv(key, "offline-key")
    monkeypatch.setattr(l.aiohttp, "ClientSession", Session)
    for module in (s, company_verification, investigator):
        monkeypatch.setattr(module, "_fetch_bounded_html", captured_html)
    monkeypatch.setattr(investigator, "_post_json", captured_post)
    monkeypatch.setattr(s, "_request_company_reverify_json", broad)
    monkeypatch.setattr(s, "investigate_company_evidence", investigate)
    result = asyncio.run(s._verify_company_fit(company, icp, 0, 0, set(),
        require_https_transport=True, evidence_investigator=True))
    assert len(completed) == 1 and not exa and not model and not search
    assert result.decision == "unavailable"
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert result.details["required_attribute_decision"] == "unavailable"
    assert result.details["company_fit_dimensions"]["employee_size"] == "unavailable"


def test_independently_verified_company_still_matches(monkeypatch):
    from tests.test_current_linkedin_company_size import _homepage_anchor
    company, icp = _company(), _icp()
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-key")
    async def prechecks(*args, **kwargs): return company_fit_match("Prechecks passed")
    monkeypatch.setattr(s, "run_company_zero_checks", prechecks)
    async def homepage(*args, **kwargs): return _homepage_anchor()
    async def broad(**kwargs): return _verdict(observed_size="11-50", size_matches=True), ""
    async def current(url, **kwargs):
        return {"employee_count": "11-50", "url": url, "quote": "Company size 11-50 employees"}
    async def investigation(**kwargs):
        assert kwargs["targets"] == ("industry",)
        return {"claims": {"industry": {"target": "industry", "status": "VERIFIED",
            "activity_role": "supplier_operator", "observed_industry": "Software", "observed_subindustry": "SaaS",
            "evidence_url": "https://acme.example.com/product", "evidence_quote": "Acme provides workflow software."}},
            "_completed_submit": True, "failure_reason": ""}
    monkeypatch.setattr(s, "verify_company_exists", homepage)
    monkeypatch.setattr(s, "_request_company_reverify_json", broad)
    monkeypatch.setattr(s, "fetch_current_linkedin_company_size", current)
    monkeypatch.setattr(s, "investigate_company_evidence", investigation)
    result = asyncio.run(s._verify_company_fit(company, icp, 0, 0, set(),
        require_https_transport=True, evidence_investigator=True))
    assert result.decision == "match", (result.reason, result.details)
    assert not result.details.get("failure_class")
