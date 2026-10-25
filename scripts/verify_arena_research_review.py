#!/usr/bin/env python3
"""Benchmark the production full-source reviewer against causal synthetic code.

Offline mode executes only checked-in fixtures. Live mode sends exactly one
existing code_review.prepare_request(research_control=True) request per case.
It never uses a separate research reviewer or production store, and it does not
resolve gateway credentials. The caller supplies the authorized paid key in
this process and a shared budget state.
Only bounded classifications and counts enter the summary. Raw replies stay in
a newly created private directory for audit.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import broker, code_review, contracts, source_bundle
from scripts.verify_arena_live_capture import SharedBudget

FIXTURES = ROOT / "tests" / "fixtures" / "research_review"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
CASES = {case["id"]: case["expected"] for case in MANIFEST["cases"]}
ICP = {"industry": "software"}
COMPANIES = [{"company_name": "Example", "company_website": "https://example.com"}]
PRICE_SOURCE = "https://openrouter.ai/anthropic/claude-sonnet-5.5"
SHARED_BUDGET_USD = Decimal("8")
PRICE_TABLE = broker.validate_price_table({
    "schema_version": broker.PRICE_TABLE_SCHEMA_VERSION,
    "fetched_at": "2026-10-24T00:00:00Z", "source": broker.OPENROUTER_MODELS_URL,
    "models": {code_review.DEFAULT_REVIEW_MODEL: {
        "prompt": "0.000002", "completion": "0.000010",
        "request": "0", "image": "0", "web_search": "0",
        "internal_reasoning": "0",
    }},
})
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_CATALOG_BYTES = 4 * 1024 * 1024


class Model:
    def __init__(self, policy):
        self.policy = policy
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.policy(request)


class Tools:
    def __init__(self, finding="continue"):
        self.finding = finding
        self.calls = []

    def __call__(self, name, arguments):
        self.calls.append((name, arguments))
        return {"url": "https://example.com/evidence", "finding": self.finding}


def _module(name):
    path = FIXTURES / name / "harness.py"
    spec = importlib.util.spec_from_file_location("research_review_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _action(name, **arguments):
    return {"action": name, "arguments": arguments}


def _finish():
    return {"action": "finish", "companies": COMPANIES}


def _native_call(name, **arguments):
    return {"id": "call_1", "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


def _native_reply(name=None):
    return ({"role": "assistant", "content": json.dumps({"companies": COMPANIES})}
            if name is None else
            {"role": "assistant", "content": None, "tool_calls": [_native_call(name, query="software")]})


def verify_causality(name: str) -> dict[str, Any]:
    """Intervene on model choices and observations; inspect actual tool calls."""
    expected = CASES[name]
    runs = []
    if name in {"adaptive_json", "bounded_hybrid"}:
        for finding in ("continue", "sufficient"):
            module, tools = _module(name), Tools(finding)
            def policy(request):
                observations = request["observations"]
                if not observations:
                    return _action("search", query="software")
                if observations[-1]["result"]["finding"] == "sufficient" or len(observations) > 1:
                    return _finish()
                return _action("fetch", url="https://example.com/evidence")
            model = Model(policy)
            module.configure(model, tools)
            assert module.run_icp(ICP) == COMPANIES
            runs.append([item[0] for item in tools.calls])
        assert runs == [["search", "fetch"], ["search"]]
        choices = []
        for selected in ("fetch", "search"):
            module, tools = _module(name), Tools()
            def choose(request):
                count = len(request["observations"])
                if count == 0:
                    return _action("search", query="software")
                if count == 1:
                    return (_action("fetch", url="https://example.com/evidence")
                            if selected == "fetch" else _action("search", query="alternative"))
                return _finish()
            module.configure(Model(choose), tools)
            assert module.run_icp(ICP) == COMPANIES
            choices.append([item[0] for item in tools.calls])
        assert choices == [["search", "fetch"], ["search", "search"]]
    elif name == "native_tools":
        for finding in ("continue", "sufficient"):
            module, tools = _module(name), Tools(finding)
            def policy(request):
                observations = [json.loads(item["content"]) for item in request["messages"] if item["role"] == "tool"]
                if not observations:
                    return _native_reply("search")
                if observations[-1]["finding"] == "sufficient" or len(observations) > 1:
                    return _native_reply()
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_2", "type": "function", "function": {
                        "name": "fetch", "arguments": json.dumps({"url": observations[-1]["url"]})}}]}
            module.configure(Model(policy), tools)
            assert module.run_icp(ICP) == COMPANIES
            runs.append([item[0] for item in tools.calls])
        assert runs == [["search", "fetch"], ["search"]]
        choices = []
        for selected in ("fetch", "search"):
            module, tools = _module(name), Tools()
            def choose(request):
                observations = [item for item in request["messages"] if item["role"] == "tool"]
                if not observations:
                    return _native_reply("search")
                if len(observations) > 1:
                    return _native_reply()
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_2", "type": "function", "function": {
                        "name": selected, "arguments": json.dumps(
                            {"url": "https://example.com/evidence"} if selected == "fetch"
                            else {"query": "alternative"})}}]}
            module.configure(Model(choose), tools)
            assert module.run_icp(ICP) == COMPANIES
            choices.append([item[0] for item in tools.calls])
        assert choices == [["search", "fetch"], ["search", "search"]]
    elif name == "dynamic_helper":
        for helper in (lambda _icp, _model, _tools: COMPANIES,
                       lambda _icp, _model, tools: [tools("search", {"query": "external"})]):
            module, tools = _module(name), Tools()
            module.configure(Model(lambda _request: _finish()), tools, helper=helper)
            module.run_icp(ICP)
            runs.append([item[0] for item in tools.calls])
        assert runs == [[], ["search"]]
    elif name == "fixed_extractor":
        for company in ("A", "B"):
            module, tools = _module(name), Tools()
            module.configure(Model(lambda _request: {"companies": [{"name": company}]}), tools)
            module.run_icp(ICP)
            runs.append([item[0] for item in tools.calls])
        assert runs == [["search", "fetch"], ["search", "fetch"]]
    elif name == "fixed_query_plan":
        for finding in ("continue", "sufficient"):
            module, tools = _module(name), Tools(finding)
            module.configure(Model(lambda request: ({"queries": ["first", "second"]}
                                                 if request.get("stage") == "plan" else
                                                 {"companies": COMPANIES})), tools)
            module.run_icp(ICP)
            runs.append([item[0] for item in tools.calls])
        assert runs == [["search", "fetch", "search", "fetch"]] * 2
    elif name in {"ignored_model_decoy", "forced_tool_decoy", "unreachable_adaptive", "posthoc_explanation"}:
        for decision in (_finish(), _action("search", query="different")):
            module, tools = _module(name), Tools()
            reply = ({**decision, "role": "assistant", "tool_calls": [_native_call("search")]}
                     if name == "forced_tool_decoy" else decision)
            model = Model(lambda _request: reply)
            module.configure(model, tools)
            module.run_icp(ICP)
            runs.append([item[0] for item in tools.calls])
            if name == "unreachable_adaptive":
                assert model.requests == []
        assert runs == [["search", "fetch"]] * 2
    else:
        raise ValueError("unknown_fixture")
    return {"case": name, "expected": expected, "causal_check": "passed",
            "actual_tool_count_first": len(runs[0]),
            "actual_tool_count_second": len(runs[1])}


def prepared_case(name: str) -> code_review.PreparedCodeReview:
    if name not in CASES:
        raise ValueError("unknown_fixture")
    source_dir = FIXTURES / name
    source_bundle.validate_source_directory(source_dir, require_license=True)
    with tempfile.TemporaryDirectory(prefix="arena-research-review-") as temporary:
        target = Path(temporary) / "source.tar.gz"
        source_bundle.write_source_archive(source_dir, target)
        payload = target.read_bytes()
    source_bundle.validate_source_archive(payload, require_license=True)
    return code_review.prepare_request(payload, research_control=True)


def _private_write(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _price_reservation(prepared: code_review.PreparedCodeReview) -> int:
    return broker.max_openrouter_cost_microusd(
        PRICE_TABLE, prepared.parameters["model"], prepared.parameters,
        max_output_tokens=prepared.parameters["max_tokens"],
    )


def verify_catalog_pricing() -> None:
    """Fail before payment if the public model catalog exceeds reserved rates."""
    with urllib.request.urlopen(broker.OPENROUTER_MODELS_URL, timeout=15) as response:
        raw = response.read(MAX_CATALOG_BYTES + 1)
    if len(raw) > MAX_CATALOG_BYTES:
        raise ValueError("model_catalog_too_large")
    document = code_review._json_object_without_duplicates(raw.decode("utf-8"))
    models = document.get("data")
    if not isinstance(models, list):
        raise ValueError("model_catalog_invalid")
    matches = [item for item in models if isinstance(item, dict)
               and item.get("id") == code_review.DEFAULT_REVIEW_MODEL]
    if len(matches) != 1 or not isinstance(matches[0].get("pricing"), dict):
        raise ValueError("review_model_price_unavailable")
    pricing = matches[0]["pricing"]
    approved = PRICE_TABLE["models"][code_review.DEFAULT_REVIEW_MODEL]
    for component in ("prompt", "completion"):
        value = pricing.get(component)
        if not isinstance(value, str) or Decimal(value) < 0 or Decimal(value) > Decimal(approved[component]):
            raise ValueError("review_model_price_above_reservation")
    for component in ("request", "internal_reasoning"):
        value = pricing.get(component, "0")
        if Decimal(str(value)) > Decimal(approved[component]):
            raise ValueError("review_model_extra_price_above_reservation")
    if int(matches[0].get("context_length") or 0) < code_review.DEFAULT_CONTEXT_WINDOW_TOKENS:
        raise ValueError("review_model_context_below_policy")


def run_live_case(name: str, *, key: str, budget: SharedBudget,
                  private_dir: Path, transport=None,
                  catalog_pricing_verified: bool = False) -> dict[str, Any]:
    """One paid call through the existing reviewer request and strict parser."""
    if not isinstance(key, str) or not key:
        raise ValueError("review_key_unavailable")
    if not catalog_pricing_verified:
        verify_catalog_pricing()
    causal = verify_causality(name)
    prepared = prepared_case(name)
    reservation = _price_reservation(prepared)
    budget.reserve(Decimal(reservation) / Decimal(1_000_000))
    result = {**causal, "source_archive_sha256": prepared.source_archive_sha256,
              "policy_hash": code_review.RESEARCH_POLICY_HASH,
              "reservation_microusd": reservation, "status": "error",
              "classification": "unavailable", "cost_status": "held"}
    wire = transport or broker.HttpxProviderTransport(max_response_bytes=MAX_RESPONSE_BYTES)
    try:
        response = wire.send(
            method="POST", url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                     "X-OpenRouter-Metadata": "enabled"},
            body=contracts.canonical_json(prepared.parameters).encode("utf-8"),
            timeout_seconds=300,
        )
        result["provider_http_status"] = response.status
        if broker._response_contains_credential(response, key):
            result["error_code"] = "credential_echo"
            return result
        _private_write(private_dir / (name + ".response.json"), response.body)
        try:
            raw = code_review._json_object_without_duplicates(response.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            result["error_code"] = "invalid_provider_json"
            return result
        actual = broker.actual_openrouter_cost_microusd(PRICE_TABLE, prepared.parameters["model"], raw)
        if actual is not None:
            budget.settle(Decimal(reservation) / Decimal(1_000_000),
                          Decimal(actual) / Decimal(1_000_000))
            result.update(actual_cost_microusd=actual, cost_status="settled")
        if response.status != 200:
            result["error_code"] = "provider_http_failure"
            return result
        try:
            effective = broker._openrouter_effective_response(response)
            if effective.status != 200:
                result["error_code"] = "provider_effective_failure"
                return result
            review = code_review.parse_response(raw, prepared)
        except Exception:
            result["error_code"] = "review_response_invalid"
            return result
        research = review.research_control
        result.update(status="reviewed" if actual is not None else "unaccounted",
                      classification=research["classification"],
                      misconduct_verdict=review.verdict,
                      evidence_count=research["evidence_count"],
                      reviewed_file_count=len(review.reviewed_files),
                      expected_match=research["classification"] == CASES[name])
        return result
    except Exception:
        result["error_code"] = "transport_or_local_failure"
        return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=tuple(CASES),
                        help="repeat to select cases; default is all")
    parser.add_argument("--live", action="store_true", help="send paid reviewer calls")
    parser.add_argument("--key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--budget-state", type=Path)
    parser.add_argument("--budget-usd", type=Decimal, default=SHARED_BUDGET_USD)
    parser.add_argument("--private-dir", type=Path)
    args = parser.parse_args(argv)
    names = args.case or list(CASES)
    causal = [verify_causality(name) for name in names]
    prepared = [prepared_case(name) for name in names]
    total_reservation = sum(_price_reservation(item) for item in prepared)
    if not args.live:
        print(json.dumps({"mode": "offline", "scope": "synthetic_source_only",
                          "runtime": "injected_callable_no_arena_admission", "cases": causal,
                          "total_max_reservation_microusd": total_reservation,
                          "price_source": PRICE_SOURCE}, sort_keys=True))
        return 0
    if not args.budget_state or not args.private_dir:
        parser.error("live mode requires --budget-state and --private-dir")
    if args.budget_usd != SHARED_BUDGET_USD:
        parser.error("live benchmark must use the shared 8 USD budget cap")
    key = os.environ.get(args.key_env)
    if not key:
        parser.error("live reviewer key unavailable in selected environment variable")
    verify_catalog_pricing()
    args.private_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    budget = SharedBudget(args.budget_state, args.budget_usd)
    results = []
    for name in names:
        result = run_live_case(name, key=key, budget=budget, private_dir=args.private_dir,
                               catalog_pricing_verified=True)
        results.append(result)
        _private_write(args.private_dir / (name + ".summary.json"),
                       (json.dumps(result, sort_keys=True) + "\n").encode())
        print(json.dumps({"case": name, "status": result["status"],
                          "classification": result["classification"],
                          "expected_match": result.get("expected_match")}, sort_keys=True), flush=True)
    print(json.dumps({"mode": "live", "scope": "synthetic_source_only",
                      "runtime": "injected_callable_no_arena_admission",
                      "reviewed": sum(x["status"] == "reviewed" for x in results),
                      "matched": sum(x.get("expected_match") is True for x in results),
                      "cases": len(results)}, sort_keys=True))
    return 0 if all(x["status"] == "reviewed" and x.get("expected_match") for x in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
