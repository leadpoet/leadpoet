"""Final scoped correction normalizes unused negative citations only."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from qualification.scoring import company_evidence_investigator as c
from tests.test_company_evidence_investigator import _finding, _attribute_finding

URL = "https://acme.example/product"
INDUSTRY = "Acme supplies workflow software."
ATTRIBUTE = "Acme is actively hiring engineers."


@pytest.mark.parametrize("guard", ["negative", "positive", "ordinary", "verified_empty", "verified_unfetched",
    "contradicted_empty", "missing_target", "duplicate_target", "unexpected_target", "wrong_status",
    "nonstring_url", "nonstring_quote", "missing_submit", "malformed_submit", "deadline",
    "401", "403", "429", "502", "503", "timeout"])
def test_existing_final_correction_and_failure_guards(monkeypatch, guard):
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    calls, submitted = [], []
    fault = guard in {"401", "403", "429", "502", "503", "timeout"}
    turn = 0
    initial = [_finding("industry", observed_value=None, observed_industry="Software", observed_subindustry="SaaS",
        activity_role="supplier_operator", evidence_url=URL, evidence_quote="Acme supplies an unfetched claim."),
        _attribute_finding("VERIFIED", url=URL, quote=ATTRIBUTE)]

    async def post(_session, _url, *, headers, payload):
        nonlocal turn
        calls.append(payload)
        if len(calls) <= 2:
            name, args = "search_web", {"query": "Acme software"}
        elif fault and len(calls) == 3:
            name, args = "fetch_page", {"url": "https://acme.example/failed-source"}
        else:
            turn += 1
            name = "submit_findings"
            if turn < 3:
                args = {"findings": copy.deepcopy(initial)}
                clock.now = (20 if turn == 1 else 20 if guard == "ordinary" else 111 if guard == "deadline" else 85)
            elif guard == "ordinary":
                args = {"findings": copy.deepcopy(initial)}
                args["findings"][0].update(status="UNPROVEN", evidence_quote=INDUSTRY)
                if turn == 4: args["findings"][0].update(status="VERIFIED")
            else:
                assert len(payload["tools"]) == 1
                if turn == 3:
                    assert "Correction targets: industry" in payload["messages"][-1]["content"] or "correction_targets" in payload["messages"][-1]["content"]
                final = _finding("industry", status="UNPROVEN", observed_value=None, observed_industry="Software",
                    activity_role="supplier_operator", evidence_url=URL, evidence_quote=INDUSTRY,
                    reason="The previous quote did not prove the claim; no proof remains.")
                if guard == "positive": final.update(status="VERIFIED")
                if guard == "verified_empty": final.update(status="VERIFIED", evidence_url="", evidence_quote="")
                if guard == "verified_unfetched": final.update(status="VERIFIED", evidence_url="https://acme.example/not-loaded")
                if guard == "contradicted_empty": final.update(status="CONTRADICTED", evidence_url="", evidence_quote="")
                if guard == "wrong_status": final["status"] = "UNKNOWN"
                if guard == "nonstring_url": final["evidence_url"] = {"url": URL}
                if guard == "nonstring_quote": final["evidence_quote"] = 123
                args = {"findings": [final]}
                if guard == "missing_target": args["findings"] = []
                if guard == "duplicate_target": args["findings"] *= 2
                if guard == "unexpected_target": final["target"] = "stage"
                if guard == "malformed_submit": args = {"findings": None}
                clock.now = 111 if turn == 3 else 151
                if guard == "missing_submit": return 200, {"choices": [{"message": {"content": "No findings"}}]}
            submitted.append(copy.deepcopy(args))
        return 200, {"choices": [{"message": {"tool_calls": [{"id": "call-" + str(len(calls)),
            "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}]}

    async def search(*args, **kwargs): return {"results": [], "notice": "discovery_only_not_evidence"}
    async def fetch(_session, url, **kwargs):
        if guard == "timeout": raise asyncio.TimeoutError()
        return {"ok": False, "error": "http_" + guard}
    async def fallback(*args, **kwargs): return {"ok": False, "error": "contents_provider_unavailable"}
    for key in ("OPENROUTER_API_KEY", "EXA_API_KEY"):
        monkeypatch.setenv(key, "offline-key")
    monkeypatch.setattr(c, "_post_json", post)
    monkeypatch.setattr(c, "_search_web", search)
    monkeypatch.setattr(c, "_fetch_page", fetch)
    monkeypatch.setattr(c, "_fetch_exa_exact_page", fallback)
    monkeypatch.setattr(c, "time", clock)
    result = asyncio.run(c.investigate_company_evidence(company_locator={"name": "Acme", "website": "https://acme.example"},
        targets=("industry", "required_attribute"), requested_industry="Software", requested_attribute="Actively hiring",
        prior_observations={"submitted_source_urls": [URL]},
        prefetched_pages={URL: {"final_url": URL, "text": INDUSTRY + " " + ATTRIBUTE}},
        verified_homepage_identity={"normalized_name": "acme", "registrable_dns_domain": "acme.example", "linkedin_company_slug": "acme"},
        positive_semantic_review=True))
    completed = guard in {"negative", "positive", "ordinary"}
    assert (result.get("_completed_submit") is True) is completed, result
    if completed:
        assert result["failure_reason"] == ""
        assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
        assert result["claims"]["required_attribute"]["evidence_quote"] == ATTRIBUTE
        assert result["claims"]["industry"]["status"] == ("UNPROVEN" if guard == "negative" else "VERIFIED")
        if guard == "negative":
            assert result["claims"]["industry"]["evidence_url"] == result["claims"]["industry"]["evidence_quote"] == ""
            assert result["claims"]["industry"]["reason"] == submitted[-1]["findings"][0]["reason"]
    elif fault:
        assert result["failure_reason"] == "provider_error"
    else:
        assert result["failure_reason"] in {"malformed_response", "admission_budget_interrupted"}
    if guard == "ordinary":
        assert turn == 4  # Ordinary cited UNPROVEN still receives repair and can verify.
    elif not fault and guard != "deadline":
        assert turn == (4 if guard in {"missing_target", "duplicate_target", "unexpected_target", "wrong_status", "malformed_submit"} else 3)


def test_exact_final_packet_and_retained_rows(monkeypatch):
    """Opt-in exact private replay; no captured company data is committed."""
    import base64
    import os
    from datetime import datetime
    from pathlib import Path
    from lab_arena import scoring
    from qualification.scoring import lead_scorer as s, linkedin_company_size as l, company_verification as cv
    from qualification.scoring.competition import _normalized_company, _normalized_icp
    from gateway.qualification.models import CompanyOutput, ICPPrompt

    capture_root = os.environ.get("ARENA_CAPTURE_REPLAY_ROOT")
    if not capture_root: pytest.skip("Private exact-packet replay is opt-in")
    root = Path(capture_root)
    events = json.loads((root / "target0d-exact-events-private.json").read_text())
    requests = {v["content"]["action_sequence"]: v for v in events if v["event_kind"] == "provider.request"}
    responses = {v["content"]["action_sequence"]: v for v in events if v["event_kind"] == "provider.response"}
    heads = {v["call_identity"]: v for v in json.loads((root / "target0d-exact-heads-private.json").read_text())}
    packets = {i: heads[responses[i]["content"]["call"]["call_identity"]] for i in [58, *range(90, 109)]}
    source = json.loads((root / "recoveryd-target0-exact-input-private.json").read_text())[0]["document"]
    context = json.loads((root / "recoveryd-target0-exact-icp-policy-private.json").read_text())
    prior = json.loads((root / "recoveryd-target0-partial-private.json").read_text())[0]["document"]["terminal_breakdowns"]
    company = CompanyOutput(**_normalized_company(source["companies"][4], integrity_policy=True))
    icp = ICPPrompt(**_normalized_icp(context["icp"]))
    def raw(i): return base64.b64decode(packets[i]["body_b64"]).decode()
    def body(i): return json.loads(raw(i))
    def stamp(value):
        date, fraction = value.split("+")[0].split(".")
        return datetime.strptime(date + "." + fraction.ljust(6, "0"), "%Y-%m-%d %H:%M:%S.%f").timestamp()
    start = stamp(requests[95]["occurred_at"])
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    def advance(i):
        if i >= 95: clock.now = stamp(responses[i]["occurred_at"]) - start
    exa, model, searches = [92, 93], [97, 99, 101, 103, 105, 106, 107, 108], [96, 102]
    html = {requests[i]["content"]["parameters"]["url"]: i for i in [58, 90, 95, 98, 100, 104]}
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
            return Response(body(exa.pop(0) if url == "https://api.exa.ai/contents" else 94))
    async def fetched(_session, url, **kwargs):
        url = kwargs.get("params", {}).get("url", url)
        i = html[url]
        advance(i)
        return int(packets[i]["status"]), url, raw(i)
    async def post(_session, url, *, headers, payload):
        i = (model if "openrouter" in url else searches).pop(0)
        if i == 108:
            assert len(payload["tools"]) == 1
            assert "Correction targets: stage, headcount, geography, industry" in payload["messages"][-1]["content"]
        advance(i)
        return 200, body(i)
    async def broad(**kwargs): return json.loads(body(91)["choices"][0]["message"]["content"]), ""
    actual = c.investigate_company_evidence
    completed = []
    async def investigated(**kwargs):
        with monkeypatch.context() as scoped:
            scoped.setattr(c, "time", clock)
            result = await actual(**kwargs)
        assert result["_completed_submit"] is True and result["failure_reason"] == ""
        assert len(result["usage"]["fetch_outcomes"]) == 4
        assert all(v["ok"] and v["loaded_text_length"] > 0 for v in result["usage"]["fetch_outcomes"])
        final = json.loads(body(108)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        raw_industry = next(v for v in final["findings"] if v["target"] == "industry")
        assert raw_industry["status"] == "UNPROVEN" and raw_industry["evidence_url"] and raw_industry["evidence_quote"]
        normalized = result["claims"]["industry"]
        assert normalized["status"] == "UNPROVEN" and normalized["evidence_url"] == normalized["evidence_quote"] == ""
        assert normalized["reason"] == raw_industry["reason"][:300]
        prior_submit = json.loads(body(107)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        attribute = next(v for v in prior_submit["findings"] if v["target"] == "required_attribute")
        assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
        assert result["claims"]["required_attribute"]["evidence_quote"] == attribute["evidence_quote"]
        completed.append(result)
        return result
    for key in ("DEEPLINE_API_KEY", "EXA_API_KEY", "OPENROUTER_API_KEY", "SCRAPINGDOG_API_KEY"):
        monkeypatch.setenv(key, "offline-key")
    monkeypatch.setattr(l.aiohttp, "ClientSession", Session)
    for module in (s, cv, c): monkeypatch.setattr(module, "_fetch_bounded_html", fetched)
    monkeypatch.setattr(c, "_post_json", post)
    monkeypatch.setattr(s, "_request_company_reverify_json", broad)
    monkeypatch.setattr(s, "investigate_company_evidence", investigated)
    fit = asyncio.run(s._verify_company_fit(company, icp, 0, 0, set(), require_https_transport=True, evidence_investigator=True))
    assert len(completed) == 1 and not model and not exa and not searches
    assert fit.decision == "unavailable" and fit.details["failure_class"] == "insufficient_fit_evidence"
    async def completed_fit(*args, **kwargs): return fit
    monkeypatch.setattr(s, "_verify_company_fit", completed_fit)
    rows = scoring.score_work_item({"scored_run_id": "exact-final-correction"}, icp=context["icp"],
        companies=source["companies"], scorer=scoring.lab_scorer(context["scorer_policy"]),
        prior_terminal_breakdowns=prior, on_terminal_breakdowns=lambda rows: None)
    assert len(rows) == 5 and rows[:4] == prior
    assert rows[4]["company_qualified"] is False and rows[4]["final_score"] == 0
