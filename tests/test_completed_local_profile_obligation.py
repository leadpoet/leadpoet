"""A completed negative review keeps independently resolved source obligations."""
import copy
import hashlib

import pytest

from qualification.scoring import lead_scorer as scorer
from qualification.scoring.company_fit_decision import company_fit_unavailable
from tests.test_recovered_local_headcount_finality import fixture


def case(control=None):
    cache, finding, investigation, projected, icp = fixture()
    projected["observed_company_name"] = "Acme"
    claims = {
        "headcount": finding,
        "industry": {"target": "industry", "status": "UNPROVEN", "evidence_url": "", "evidence_quote": ""},
    }
    receipt = {
        "gate": "company_evidence_investigation", "completed_submitted_findings": True,
        "failure_reason": "", "targets": list(claims), "claims": claims,
        "usage": {"fetch_outcomes": [{"ok": True, "url_sha256": hashlib.sha256(finding["evidence_url"].encode()).hexdigest(), "loaded_text_length": 100}]},
    }
    result = company_fit_unavailable("unproven industry", details={
        "identity_decision": "match", "dimension_decisions": {"employee_size": "match", "industry": "unavailable", "geography": "match", "stage": "match"},
        "investigation_receipt": receipt,
    })
    receipt = result.details["investigation_receipt"]
    finding = receipt["claims"]["headcount"]
    if control in {"provider_error", "malformed_response"}:
        cache["structured_failure_reason"] = cache["failure_reason_code"] = control
        cache["evidence"] = None  # An actual initial transport/contract fault.
    elif control == "incomplete":
        receipt["completed_submitted_findings"] = False
    elif control == "review_failure":
        receipt["failure_reason"] = "provider_error"
    elif control == "mixed_positive":
        icp = icp.model_copy(update={"required_attribute": "workflow software"})
        result.details["required_attribute_decision"] = "unavailable"
        receipt["targets"].append("required_attribute")
        receipt["claims"]["required_attribute"] = {"target": "required_attribute", "status": "VERIFIED", "evidence_url": "", "evidence_quote": ""}
    elif control == "mixed_unproven":
        icp = icp.model_copy(update={"required_attribute": "workflow software"})
        result.details["required_attribute_decision"] = "unavailable"
        receipt["targets"].append("required_attribute")
        receipt["claims"]["required_attribute"] = {"target": "required_attribute", "status": "UNPROVEN", "evidence_url": "", "evidence_quote": ""}
        projected[scorer._REQUIRED_ATTRIBUTE_GROUNDING] = {"status": "unproven"}
    elif control == "positive_result":
        result.decision = "match"
    elif control == "no_pages_unfinished":
        receipt["usage"] = {"fetch_outcomes": [], "prefetched_pages": 0, "total_loaded_pages": 0}
        receipt["completed_submitted_findings"] = False
        receipt["failure_reason"] = "admission_budget_interrupted"
    elif control == "unbound_transport":
        receipt["usage"]["fetch_outcomes"].append({"ok": False, "error_class": "http_error"})
    elif control == "recovered_source":
        receipt["usage"]["fetch_outcomes"] = [{"ok": False, "url_sha256": "b" * 64, "error_class": "http_error"}, {"ok": True, "url_sha256": "b" * 64, "loaded_text_length": 100}]
    elif control == "fetch_failed":
        receipt["usage"]["fetch_outcomes"].append({"ok": False, "url_sha256": "b" * 64, "error_class": "http_error"})
    elif control == "missing_target":
        receipt["targets"].remove("industry")
    elif control == "missing_claim":
        receipt["claims"].pop("industry")
    elif control == "unresolved_positive":
        receipt["claims"]["industry"]["status"] = "VERIFIED"
    elif control == "cited_unproven":
        receipt["claims"]["industry"]["evidence_url"] = "https://acme.example.com/about"
    elif control == "prefetched_only":
        receipt["usage"] = {"fetch_outcomes": [], "prefetched_pages": 1, "total_loaded_pages": 1}
    elif control == "no_headcount_claim":
        receipt["claims"].pop("headcount")
        receipt["targets"].remove("headcount")
    elif control == "identity_unavailable":
        result.details["identity_decision"] = "unavailable"
    elif control == "headcount_unavailable":
        result.details["dimension_decisions"]["employee_size"] = "unavailable"
    elif control == "geography_unavailable":
        result.details["dimension_decisions"]["geography"] = "unavailable"
    return result, projected, icp, cache


@pytest.mark.parametrize("control", [
    None, "provider_error", "malformed_response",
    "incomplete", "review_failure", "fetch_failed", "identity_unavailable", "missing_target", "missing_claim", "unresolved_positive", "cited_unproven", "prefetched_only", "no_headcount_claim",
    "headcount_unavailable", "geography_unavailable", "mixed_positive", "mixed_unproven", "positive_result", "no_pages_unfinished", "unbound_transport", "recovered_source",
])
def test_source_local_profile_miss_cannot_reopen_a_complete_negative_review(control):
    result, projected, icp, cache = case(control)
    before = copy.deepcopy(cache)
    settled = scorer._settle_completed_company_investigation(
        result, projected, investigation_targets=tuple(result.details["investigation_receipt"]["targets"]), icp=icp,
        icp_attribute=str(icp.required_attribute or ""), icp_stage="", current_profile_cache=cache,
    )
    assert cache == before  # Negative finality does not clear a provider diagnostic.
    terminal = settled is not None and settled.details.get("failure_class") == "insufficient_fit_evidence"
    assert terminal is (control in {None, "prefetched_only", "no_headcount_claim", "mixed_unproven", "recovered_source"})
    if terminal:
        assert settled.decision == "unavailable"
        assert settled.details["dimension_decisions"] == result.details["dimension_decisions"]


def test_exact_saved_completed_profile_obligation_packets(monkeypatch):
    """Opt-in private packet replay; no customer evidence enters the repository."""
    import asyncio
    import base64
    from datetime import datetime
    import json
    import os
    from pathlib import Path
    import socket
    from types import SimpleNamespace
    from gateway.qualification.models import CompanyOutput, ICPPrompt
    from qualification.scoring import company_evidence_investigator as investigator, company_verification, linkedin_company_size
    from qualification.scoring.competition import _normalized_company, _normalized_icp

    root = os.environ.get("ARENA_PROFILE_OBLIGATION_REPLAY_ROOT")
    if not root:
        pytest.skip("Private saved packet replay is opt-in")
    packet = json.loads((Path(root) / "completed-local-profile-replay.private.json").read_text())
    plan = packet["plan"]
    responses = packet["responses"]
    queues = {name: list(plan[name]) for name in ("exa", "models", "search")}
    consumed = []
    clock = SimpleNamespace(now=0.0)
    clock.monotonic = lambda: clock.now
    start = datetime.fromisoformat(plan["investigation_started_at"]).timestamp()

    def raw(index):
        return base64.b64decode(responses[index]["terminal"]["body_b64"]).decode()

    def consume(index):
        consumed.append(index)
        if index >= plan["first_investigation_response"]:
            clock.now = datetime.fromisoformat(responses[index]["at"]).timestamp() - start
        return raw(index)

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
            if url == "https://api.exa.ai/contents":
                index = queues["exa"].pop(0)
            else:
                assert "deepline" in url
                index = plan["structured"]
            return Response(json.loads(consume(index)))

    async def html(_session, url, **kwargs):
        target = kwargs.get("params", {}).get("url", url)
        selected = [row for row in plan["html"] if row["url"] == target]
        assert len(selected) == 1
        index = selected[0]["response"]
        return responses[index]["terminal"]["status"], target, consume(index)

    async def post(_session, url, *, headers, payload):
        index = queues["models" if "openrouter" in url else "search"].pop(0)
        return responses[index]["terminal"]["status"], json.loads(consume(index))

    async def broad(**kwargs):
        return json.loads(json.loads(consume(plan["broad"]))["choices"][0]["message"]["content"]), ""

    def deny(*args, **kwargs):
        raise AssertionError("offline_network_forbidden")

    for key in ("OPENROUTER_API_KEY", "DEEPLINE_API_KEY", "EXA_API_KEY", "SCRAPINGDOG_API_KEY"):
        monkeypatch.setenv(key, "offline-key")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(linkedin_company_size.aiohttp, "ClientSession", Session)
    for module in (scorer, investigator, company_verification):
        monkeypatch.setattr(module, "_fetch_bounded_html", html)
    monkeypatch.setattr(investigator, "_post_json", post)
    monkeypatch.setattr(investigator, "time", clock)
    monkeypatch.setattr(scorer, "_request_company_reverify_json", broad)
    result = asyncio.run(scorer._verify_company_fit(
        CompanyOutput(**_normalized_company(packet["company"], integrity_policy=True)),
        ICPPrompt(**_normalized_icp(packet["icp"])), 0, 0, set(),
        require_https_transport=True, evidence_investigator=True,
    ))
    assert result.decision == "unavailable"
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert consumed == plan["expected_responses"]
    assert not any(queues.values())
