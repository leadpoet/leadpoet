"""Public investigator contract; runs unchanged on the old verifier too.

All HTTP is fake. No new planner/helper imports are needed for fail-before.
"""
import asyncio
import copy
import hashlib
import json as jsonlib

import pytest

from lab_arena.operations import validate_operation_request
from qualification.scoring import company_evidence_investigator as inv


@pytest.mark.parametrize("pressure", [False, True])
def test_public_investigator_compiles_then_accepts_claims_only_frozen_obligations(monkeypatch, pressure):
    criterion = ("Provides voice and workflow software for clinics. \n " if pressure else
                 "A platform providing voice and workflow software for clinics.")
    url = "https://contractco.example/products"
    quote = "ContractCo provides voice and workflow software for clinics."
    requests, evidence_payloads = [], []
    bounded = inv._bounded_message_json

    def bounded_observer(value, *, prefix=""):
        if prefix.startswith("Investigate this bounded request"):
            assert "product_requirement_plan" not in value
            assert "server_frozen_product_plan" not in value
            # Counterfactual allocation without any semantic-plan metadata.
            expected = bounded(copy.deepcopy(value), prefix=prefix)
            evidence_payloads.append(expected)
        return bounded(value, prefix=prefix)

    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            return self.body

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def post(self, request_url, *, headers, json):
            validate_operation_request("openrouter.chat", json)
            requests.append(copy.deepcopy(json))
            names = [t["function"]["name"] for t in json["tools"]]
            if "submit_product_plan" in names:
                assert json["model"] == inv.POSITIVE_SEMANTIC_REVIEW_MODEL
                planner_prompt = json["messages"][0]["content"]
                assert "only leaves carry obligations" in planner_prompt
                assert "never only in ALL/ANY text" in planner_prompt
                assert "separate ALL leaf outside ANY" in planner_prompt
                assert jsonlib.loads(json["messages"][1]["content"]) == {
                    "requested_product_service": criterion,
                }
                assert len(json["messages"]) == 2
                texts = (["Voice software supply", "Workflow software supply", "Clinic customer scope"]
                         if not pressure else [f"Semantic obligation {i:02d} " + "x" * 878 for i in range(23)])
                nodes = [{"id": "root", "op": "all", "text": "",
                          "children": [f"r{i}" for i in range(len(texts))]}]
                nodes += [{"id": f"r{i}", "op": "leaf", "text": text, "children": []}
                          for i, text in enumerate(texts)]
                arguments = {"plan_json": jsonlib.dumps({"root": "root", "nodes": nodes})}
                name = "submit_product_plan"
            else:
                frozen = None
                for message in json["messages"]:
                    if message.get("role") != "user":
                        continue
                    try:
                        document = jsonlib.loads(message.get("content", ""))
                    except ValueError:
                        continue
                    frozen = document.get("server_frozen_product_plan") or frozen
                # The old verifier has no host plan. It receives the same new
                # claims-only contract and must fail this test on its verdict,
                # rather than failing to import a new implementation symbol.
                leaves = ([n for n in frozen["nodes"] if n["op"] == "leaf"] if frozen else
                          [{"id": "p1"}, {"id": "p2"}, {"id": "p3"}])
                if frozen:
                    assert frozen["criterion_sha256"] == hashlib.sha256(criterion.encode()).hexdigest()
                    assert json["messages"][1]["content"] == evidence_payloads[-1]
                    if pressure:
                        assert len(leaves) == 23
                        assert all(len(n["text"]) == 901 for n in leaves)
                wire = {"plan_id": frozen["plan_id"] if frozen else "no-host-plan",
                        "claims": [{"id": n["id"], "evidence": [{"url": url, "quote": quote}],
                                    "activity_role": "supplier_operator", "status": "VERIFIED"} for n in leaves]}
                if frozen is None and not pressure:
                    # Old protocol, semantically complete, with only the
                    # harmless leading article omitted from its partition.
                    # This reproduces the saved literal-grammar failure.
                    wire = {"nodes": [
                        {"id": "p0", "span": criterion, "op": "all", "children": ["p1", "p2"]},
                        {"id": "p1", "span": "platform providing voice and workflow software", "op": "leaf", "children": []},
                        {"id": "p2", "span": "for clinics.", "op": "leaf", "children": []},
                    ], "claims": [{"id": key, "evidence": [{"url": url, "quote": quote}],
                                   "activity_role": "supplier_operator", "status": "VERIFIED"}
                                  for key in ("p1", "p2")]}
                finding = {"target": "industry", "status": "VERIFIED", "observed_industry": "Software",
                           "activity_role": "supplier_operator", "evidence_url": url, "evidence_quote": quote,
                           "reason": "The whole original criterion and all obligations are supported.",
                           "product_coverage": jsonlib.dumps(wire)}
                arguments, name = {"findings": [finding]}, "submit_findings"
            return Response({"choices": [{"message": {"tool_calls": [{"id": str(len(requests)),
                "type": "function", "function": {"name": name, "arguments": jsonlib.dumps(arguments)}}]}}]})

    async def forbidden_research(*args, **kwargs):
        pytest.fail("Loaded compound evidence should need no research in this contract control")

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv.aiohttp, "ClientSession", lambda **kwargs: Session())
    monkeypatch.setattr(inv, "_search_web", forbidden_research)
    monkeypatch.setattr(inv, "_fetch_page", forbidden_research)
    monkeypatch.setattr(inv, "_bounded_message_json", bounded_observer)
    urls = [url] + (["https://contractco.example/technical", "https://contractco.example/jobs"] if pressure else [])
    pages = {u: {"final_url": u, "text": (quote + " " + "e" * 23900) if pressure else quote} for u in urls}
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "ContractCo", "website": "https://contractco.example"},
        targets=("industry",), requested_industry="Software", requested_product_service=criterion,
        positive_semantic_review=True,
        verified_homepage_identity={"normalized_name": "ContractCo", "registrable_dns_domain": "contractco.example"},
        prior_observations={"submitted_source_urls": urls, **({"metadata": "m" * 8000} if pressure else {})},
        prefetched_pages=pages,
    ))
    if not pressure and not any("submit_product_plan" in [t["function"]["name"] for t in r["tools"]] for r in requests):
        errors = []
        for request in requests:
            for message in request["messages"]:
                if message.get("role") == "tool":
                    feedback = jsonlib.loads(message["content"])
                    errors.append(feedback.get("product_coverage", {}).get("error", ""))
        assert any("'A '" in error for error in errors), errors
    assert result.get("claims", {}).get("industry", {}).get("status") == "VERIFIED", (
        result, [(r["tool_choice"], r["messages"][-1].get("content", "")[:1100]) for r in requests],
    )
    assert result["_completed_submit"] is True
    assert len(requests) == 2
    assert result["usage"]["reasoning_turns"] == 1
    assert result["usage"]["product_plan"]["outcome"] == "accepted"
    assert result["usage"]["search_calls"] == result["usage"]["fetch_calls"] == 0
