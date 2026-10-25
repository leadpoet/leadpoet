"""The final envelope retry may retain native-validated sibling findings."""
import asyncio
import json

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring.product_coverage import (
    ValidatedWholeProductCoverage, WHOLE_SCHEMA, criterion_hash,
)


TARGETS = ("headcount", "industry", "geography", "stage", "required_attribute")
PENDING = ("headcount", "geography")


def _finding(target, *, status="VERIFIED", reason="Company-bound proof passed"):
    return {
        "target": target, "status": status,
        "observed_value": "500" if target == "headcount" else "Software",
        "observed_country": "United States" if target == "geography" else "",
        "observed_state": "California" if target == "geography" else "",
        "observed_industry": "Software" if target == "industry" else "",
        "observed_subindustry": "Practice software" if target == "industry" else "",
        "activity_role": "supplier_operator" if target in {"industry", "required_attribute"}
                         else "unresolved",
        "evidence_url": "https://acme.example/about" if status == "VERIFIED" else "",
        "evidence_quote": "Acme supplies practice software." if status == "VERIFIED" else "",
        "supporting_evidence_url_1": "", "supporting_evidence_quote_1": "",
        "supporting_evidence_url_2": "", "supporting_evidence_quote_2": "",
        "product_coverage": "VERIFIED" if target == "industry" else None,
        "old_name": "", "new_name": "", "old_domain": "", "new_domain": "",
        "shared_linkedin_slug": "", "reason": reason,
    }


def _response(name, args, turn, *, truncated=False):
    return 200, {"choices": [{
        "finish_reason": "length" if truncated else "tool_calls",
        "message": {"tool_calls": [{
            "id": f"call-{turn}", "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
        }]},
    }]}


def _admitted(target, *, status="VERIFIED", reason="Company-bound proof passed"):
    finding = _finding(target, status=status, reason=reason)
    if target == "industry":
        finding["product_coverage"] = ValidatedWholeProductCoverage(
            schema=WHOLE_SCHEMA,
            criterion_sha256=criterion_hash("Practice software for clinics"),
            status=status, whole_criterion_status=status,
            evidence=([{"url": finding["evidence_url"],
                        "quote": finding["evidence_quote"]}]
                      if status == "VERIFIED" else []),
        )
    return finding


def test_serialization_retry_scopes_only_rejected_targets_after_research(monkeypatch):
    requests, validations = [], []

    async def post(_session, _url, *, headers, payload):
        from lab_arena.operations import validate_operation_request
        validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        turn = len(requests)
        if turn <= 2:
            return _response("search_web", {"query": f"Acme practice software {turn}"}, turn)
        if turn == 3:
            return _response("submit_findings", '{"findings":[', turn, truncated=True)
        if turn == 4:
            return _response("submit_findings", {"findings": [_finding(t) for t in TARGETS]}, turn)
        return _response("submit_findings", {"findings": [_finding(t) for t in PENDING]}, turn)

    async def search(_session, _query, *, key):
        return {"results": []}

    def validate(arguments, *, targets, **kwargs):
        validations.append(tuple(targets))
        if tuple(targets) == TARGETS:
            return {
                    target: (_admitted(target, status="UNPROVEN",
                                  reason="source quote did not identify the investigated company")
                         if target in PENDING else _admitted(target))
                for target in targets
            }
        assert tuple(targets) == PENDING
        return {target: _admitted(target) for target in targets}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "_search_web", search)
    monkeypatch.setattr(inv, "_validated_findings", validate)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=TARGETS, requested_industry="Software",
        requested_product_service="Practice software for clinics",
        requested_attribute="Supplies practice software to clinics",
        positive_semantic_review=True,
        verified_homepage_identity={"normalized_name": "Acme",
                                    "registrable_dns_domain": "acme.example"},
    ))
    assert len(requests) == 5, (result, validations)
    assert validations == [TARGETS, PENDING]
    correction_tool = requests[-1]["tools"][0]["function"]
    assert len(requests[-1]["tools"]) == 1
    schema = correction_tool["parameters"]["properties"]["findings"]
    assert schema["minItems"] == schema["maxItems"] == 2
    assert set(schema["items"]["properties"]["target"]["enum"]) == set(PENDING)
    assert requests[-1]["tool_choice"] == {"type": "function",
                                             "function": {"name": "submit_findings"}}
    assert result["_completed_submit"] is True
    assert set(result["claims"]) == set(TARGETS)
    assert all(finding["status"] == "VERIFIED" for finding in result["claims"].values())
    assert result["usage"]["search_calls"] == 2
    assert result["usage"]["fetch_calls"] == 0


def test_early_retry_does_not_scope_when_required_product_search_remains(monkeypatch):
    requests = []

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            return _response("submit_findings", '{"findings":[', turn, truncated=True)
        if turn == 2:
            return _response("submit_findings", {"findings": [
                _finding(t, status="UNPROVEN", reason="missing practice capability")
                if t == "industry" else _finding(t) for t in TARGETS
            ]}, turn)
        return _response("search_web", {"query": "Acme missing practice capability"}, turn)

    async def search(_session, _query, *, key):
        return {"results": []}

    def validate(arguments, *, targets, **kwargs):
        return {
            target: (_admitted(target, status="UNPROVEN", reason="unproven capability")
                     if target == "industry" else _admitted(target))
            for target in targets
        }

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "_search_web", search)
    monkeypatch.setattr(inv, "_validated_findings", validate)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=TARGETS, requested_industry="Software",
        requested_product_service="Practice software for clinics",
        requested_attribute="Supplies practice software to clinics",
        positive_semantic_review=True,
        verified_homepage_identity={"normalized_name": "Acme",
                                    "registrable_dns_domain": "acme.example"},
    ))
    assert len(requests) >= 3, result
    assert requests[2]["tool_choice"] == {"type": "function",
                                            "function": {"name": "search_web"}}
    assert result.get("_completed_submit") is not True


def test_ordinary_new_source_can_change_previous_industry_judgment(monkeypatch):
    requests = []
    targets = ("industry", "headcount")

    async def post(_session, _url, *, headers, payload):
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            return _response("search_web", {"query": "Acme practice software"}, turn)
        if turn == 2:
            return _response("submit_findings", {"findings": [_finding(t) for t in targets]}, turn)
        if turn == 3:
            return _response("fetch_page", {"url": "https://acme.example/new"}, turn)
        return _response("submit_findings", {"findings": [
            _finding("industry", status="UNPROVEN", reason="The new page creates unresolved scope."),
            _finding("headcount"),
        ]}, turn)

    async def search(_session, _query, *, key):
        return {"results": []}

    async def fetch(_session, url, **kwargs):
        return {"ok": True, "url": url, "final_url": url,
                "text": "Acme describes a different practice software scope."}

    def validate(arguments, *, targets, **kwargs):
        submitted = {finding["target"]: finding for finding in arguments["findings"]}
        if len(requests) == 2:
            return {"industry": _admitted("industry"),
                    "headcount": _admitted("headcount", status="UNPROVEN",
                                           reason="source quote did not identify the investigated company")}
        return {target: _admitted(target, status=submitted[target]["status"],
                                  reason=submitted[target]["reason"])
                for target in targets}

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "_search_web", search)
    monkeypatch.setattr(inv, "_fetch_page", fetch)
    monkeypatch.setattr(inv, "_validated_findings", validate)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme", "website": "https://acme.example/"},
        targets=targets, requested_industry="Software",
        requested_product_service="Practice software for clinics",
        positive_semantic_review=True,
        verified_homepage_identity={"normalized_name": "Acme",
                                    "registrable_dns_domain": "acme.example"},
    ))
    assert len(requests) == 4, result
    assert len(requests[2]["tools"]) > 1
    assert result["_completed_submit"] is True
    assert result["claims"]["industry"]["status"] == "UNPROVEN"
    assert result["claims"]["headcount"]["status"] == "VERIFIED"
