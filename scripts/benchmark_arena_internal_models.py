#!/usr/bin/env python3
"""Full-source internal reviewer benchmark. Default/dry-run is offline and free.

Gold labels are fixed fixture data, never inferred from model replies. Malicious
source is packaged as text and NEVER imported or executed. Live calls require an
explicit flag, a private artifact directory and the shared <=20 USD budget file.
A JSON case may specify files, research_case, or source_archive (relative to its
JSON file), plus independently reviewed expected labels (null means unlabelled).
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from decimal import Decimal, InvalidOperation
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab_arena import broker, code_review, contracts
from scripts.verify_arena_live_capture import SharedBudget, _private_write, _json_bytes
from scripts import verify_arena_research_review as research

FIXTURES = ROOT / "tests/fixtures/internal_model_review"
CAP_USD = Decimal("20")
MAX_BYTES = 4 * 1024 * 1024
API = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {
    "anthropic/claude-sonnet-5.5": "google-vertex",
    "deepseek/deepseek-v4.1-flash-1017": "nebius/fp8",
    "z-ai/glm-5.3": "fireworks",
    "moonshotai/kimi-k3.1-pro": "moonshotai/mxfp4",
}


def decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid_money") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("invalid_money")
    return result


def load_cases(selectors: list[str] | None = None) -> list[dict]:
    def read(path):
        doc = json.loads(path.read_text())
        cases = doc.get("cases", [doc])
        return [{**case, "_base": path.parent} for case in cases]
    known = {}
    for path in sorted(FIXTURES.glob("*.json")):
        for case in read(path):
            if case["id"] in known:
                raise ValueError("duplicate_case_id")
            known[case["id"]] = case
    selected = []
    for selector in selectors or list(known):
        for token in selector.split(","):
            selected.extend([known[token]] if token in known else read(Path(token).resolve()))
    if len({case["id"] for case in selected}) != len(selected):
        raise ValueError("duplicate_case_id")
    for case in selected:
        if not isinstance(case["id"], str) or not case["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in case["id"]):
            raise ValueError("unsafe_case_id")
        expected = case.get("expected", {})
        if set(expected) - {"verdict", "research_classification", "categories"}:
            raise ValueError("unknown_expected_label")
        if expected.get("verdict") not in (None, "pass", "reject"):
            raise ValueError("invalid_expected_verdict")
        if expected.get("research_classification") not in (None, *code_review.RESEARCH_CLASSIFICATIONS):
            raise ValueError("invalid_expected_research")
        if expected.get("categories") is not None and (not isinstance(expected["categories"], list) or any(x not in code_review.REVIEW_CATEGORIES for x in expected["categories"])):
            raise ValueError("invalid_expected_categories")
    return selected


def prepare_case(case: dict, *, model: str, provider: str, mode: str,
                 route: dict | None = None) -> code_review.PreparedCodeReview:
    if "research_case" in case:
        prepared = research.prepared_case(case["research_case"])
    else:
        if "source_archive" in case:
            archive = (case["_base"] / case["source_archive"]).read_bytes()
        else:
            stream = io.BytesIO()
            # Deterministic archive; no extraction or fixture execution.
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive_stream:
                    for name, content in sorted(case["files"].items()):
                        raw = content.encode("utf-8")
                        info = tarfile.TarInfo(name)
                        info.size = len(raw)
                        archive_stream.addfile(info, io.BytesIO(raw))
            archive = stream.getvalue()
        prepared = code_review.prepare_request(archive, research_control=True,
                                               similarity_contexts=tuple(case.get("similarity_contexts", [])))
    parameters = copy.deepcopy(prepared.parameters)
    # Keep the production prompt, bindings, full source and strict original schema.
    parameters["model"] = model
    parameters["provider"]["only"] = [provider]
    parameters["reasoning"] = {"effort": mode, "exclude": False}
    if route:
        parameters["provider"]["max_price"] = {
            "prompt": float(decimal(route["pricing"]["prompt"]) * 1_000_000),
            "completion": float(decimal(route["pricing"]["completion"]) * 1_000_000),
            "request": float(decimal(route["pricing"].get("request", "0"))),
        }
    bound = code_review.REQUEST_TOKEN_OVERHEAD + len(contracts.canonical_json(parameters).encode())
    return replace(prepared, parameters=parameters, input_tokens_upper_bound=bound,
                   requires_context_integrity=bound + parameters["max_tokens"] > prepared.context_window_tokens)


def fetch_metadata(model: str) -> dict:
    url = "https://openrouter.ai/api/v1/models/" + model + "/endpoints"
    with urllib.request.urlopen(url, timeout=20) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("metadata_too_large")
    return code_review._json_object_without_duplicates(raw.decode())


def validate_route(document: dict, model: str, provider: str) -> dict:
    data = document.get("data", {})
    if data.get("id") != model:
        raise ValueError("metadata_model_mismatch")
    endpoints = [e for e in data.get("endpoints", []) if e.get("tag") == provider]
    # google-vertex is the existing approved aggregate alias. Bound its most
    # expensive eligible regional endpoint; candidate tags require exact match.
    if provider == "google-vertex" and not endpoints:
        endpoints = [e for e in data.get("endpoints", []) if str(e.get("tag", "")).startswith("google-vertex/")]
    if not endpoints:
        raise ValueError("route_unavailable")
    for endpoint in endpoints:
        if endpoint.get("status") != 0 or endpoint.get("uptime_last_30m") == 0:
            raise ValueError("route_unhealthy")
        if int(endpoint.get("context_length") or 0) < code_review.DEFAULT_CONTEXT_WINDOW_TOKENS:
            raise ValueError("route_context_below_policy")
        if int(endpoint.get("max_completion_tokens") or 0) < code_review.DEFAULT_MAX_OUTPUT_TOKENS:
            raise ValueError("route_output_below_policy")
        if not {"reasoning", "response_format", "max_tokens"}.issubset(endpoint.get("supported_parameters", [])):
            raise ValueError("route_parameters_unsupported")
    pricing = {key: str(max(decimal(e["pricing"].get(key, "0")) for e in endpoints))
               for key in ("prompt", "completion", "request")}
    if decimal(pricing["prompt"]) == 0 or decimal(pricing["completion"]) == 0:
        raise ValueError("route_price_unavailable")
    for endpoint in endpoints:
        if decimal(endpoint["pricing"].get("internal_reasoning", "0")):
            raise ValueError("unbounded_reasoning_price")
    canonicals = {e.get("canonical_slug") or str(e.get("name", "")).split(" | ")[-1] for e in endpoints}
    if len(canonicals) != 1 or "/" not in next(iter(canonicals)):
        raise ValueError("canonical_model_unavailable")
    return {"model": model, "provider": provider, "provider_name": endpoints[0]["provider_name"],
            "canonical_model": next(iter(canonicals)), "pricing": pricing,
            "context_length": min(e["context_length"] for e in endpoints),
            "max_completion_tokens": min(e["max_completion_tokens"] for e in endpoints)}


def reservation_usd(prepared, route):
    price = route["pricing"]
    return (Decimal(prepared.input_tokens_upper_bound) * decimal(price["prompt"])
            + Decimal(prepared.parameters["max_tokens"]) * decimal(price["completion"])
            + decimal(price.get("request", "0")))


def score_review(case, review):
    actual = {"verdict": review.verdict, "research_classification": review.research_control["classification"],
              "categories": sorted({finding["category"] for finding in review.findings})}
    expected = case.get("expected", {})
    matches = {key: (sorted(value) if key == "categories" else value) == actual[key]
               for key, value in expected.items() if value is not None}
    return {"actual": actual, "label_matches": matches,
            "accuracy_match": all(matches.values()) if matches else None,
            "reviewed_file_count": len(review.reviewed_files)}


def reasoning_observation(document):
    choices = document.get("choices") or []
    message = choices[0].get("message", {}) if len(choices) == 1 else {}
    counts = {}
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            counts[key] = len(value)
    for part in message.get("reasoning_details") or []:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str) and part["text"].strip():
            counts["reasoning_details.text"] = counts.get("reasoning_details.text", 0) + len(part["text"])
    return {"full_text_returned": bool(counts), "field_characters": counts}


def response_route_verified(document, prepared, route):
    metadata = document.get("openrouter_metadata", {})
    available = metadata.get("endpoints", {}).get("available", [])
    selected = [e for e in available if e.get("selected") is True]
    if (len(selected) != 1 or selected[0].get("provider") != route["provider_name"]
            or selected[0].get("model") not in {route["model"], route["canonical_model"]}
            or document.get("provider") not in (None, route["provider_name"])):
        return False
    # Production knows its existing model aliases. This benchmark also knows
    # candidate aliases from fresh endpoint metadata, without changing policy.
    normalized = copy.deepcopy(document)
    for endpoint in normalized["openrouter_metadata"]["endpoints"]["available"]:
        if endpoint.get("selected") is True:
            endpoint["model"] = prepared.parameters["model"]
    return code_review.context_integrity_failure(normalized, prepared) is None


def run_live_case(case, prepared, route, *, key, budget, private_dir, transport=None):
    """One dispatch only. Failed/uncertain attempts retain their reservation."""
    request = contracts.canonical_json(prepared.parameters).encode()
    if key.encode() in request:
        raise ValueError("credential_in_request")
    name = case["id"]
    _private_write(private_dir / (name + ".request.json"), request)
    amount = reservation_usd(prepared, route)
    result = {"case": name, "schema_valid": False, "transport_ok": False,
              "route_verified": False, "cost_status": "held", "reserved_usd": str(amount),
              "accuracy_match": None, "label_matches": {}}
    budget.reserve(amount)
    started = time.monotonic()
    wire = transport or broker.HttpxProviderTransport(max_response_bytes=MAX_BYTES)
    try:
        response = wire.send(method="POST", url=API,
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                      "X-OpenRouter-Metadata": "enabled"},
                             body=request, timeout_seconds=300)
        result["http_status"] = response.status
        if broker._response_contains_credential(response, key):
            result["error_code"] = "credential_echo"
            return result
        _private_write(private_dir / (name + ".response.json"), response.body)
        document = code_review._json_object_without_duplicates(response.body.decode())
        # Settle only the provider's explicit reported bill. Do not invent a
        # successful bill from token counts when actual provider cost is absent.
        actual = document.get("usage", {}).get("cost")
        if actual is not None:
            cost = decimal(actual)
            budget.settle(amount, cost)
            result.update(cost_status="settled", actual_cost_usd=str(cost))
        if response.status != 200 or broker._openrouter_effective_response(response).status != 200:
            result["error_code"] = "provider_failure"
            return result
        result["transport_ok"] = True
        result["reasoning"] = reasoning_observation(document)
        result["route_verified"] = response_route_verified(document, prepared, route)
        # Canonical model aliases are checked against fresh endpoint metadata;
        # only the validated envelope copy is normalized for the existing parser.
        if document.get("model") == route["canonical_model"]:
            document = {**document, "model": prepared.parameters["model"]}
        parse_prepared = prepared
        if prepared.requires_context_integrity and result["route_verified"]:
            parse_prepared = replace(prepared, requires_context_integrity=False)
        review = code_review.parse_response(document, parse_prepared)
        result.update(schema_valid=True, **score_review(case, review))
        return result
    except code_review.CodeReviewError as exc:
        result["error_code"] = exc.code
        result["schema_error"] = getattr(exc, "response_reason", None)
    except Exception:
        # Never expose provider error prose, raw source or exception text.
        result["error_code"] = "transport_or_accounting_or_parse_failure"
    finally:
        result["latency_seconds"] = round(time.monotonic() - started, 3)
    return result


def summarize(results):
    labels = [r for r in results if r.get("accuracy_match") is not None]
    return {"cases": len(results), "labelled_parsed_cases": len(labels),
            "accuracy_matched": sum(r["accuracy_match"] for r in labels),
            "schema_valid": sum(r["schema_valid"] for r in results),
            "transport_ok": sum(r["transport_ok"] for r in results),
            "route_verified": sum(r["route_verified"] for r in results),
            "full_reasoning_returned": sum(r.get("reasoning", {}).get("full_text_returned", False) for r in results),
            "cost_settled": sum(r["cost_status"] == "settled" for r in results)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--provider")
    parser.add_argument("--mode", choices=("low", "high"), default="low")
    parser.add_argument("--cases", nargs="+", help="fixture ids, comma-separated ids, or private case JSON paths")
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--budget-state", type=Path)
    parser.add_argument("--budget-usd", type=Decimal, default=CAP_USD)
    parser.add_argument("--key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--metadata-json", type=Path, help="offline endpoint snapshot; live always fetches fresh")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.budget_usd.is_finite() or not 0 < args.budget_usd <= CAP_USD:
        parser.error("budget must be positive and at most 20 USD")
    provider = args.provider or MODELS[args.model]
    cases = load_cases(args.cases)
    route = None
    metadata = None
    if args.live or args.metadata_json:
        metadata = fetch_metadata(args.model) if args.live else json.loads(args.metadata_json.read_text())
        route = validate_route(metadata, args.model, provider)
    prepared = [prepare_case(case, model=args.model, provider=provider, mode=args.mode, route=route) for case in cases]
    if not args.live:
        print(json.dumps({"mode": "offline", "model": args.model, "provider": provider,
                          "reasoning_mode": args.mode, "metadata_verified": bool(route),
                          "cases": [{"case": c["id"], "files": len(p.files), "source_bytes": p.reviewed_file_bytes,
                                     "expected": c.get("expected", {}),
                                     "max_reservation_usd": str(reservation_usd(p, route)) if route else None}
                                    for c, p in zip(cases, prepared)]}, sort_keys=True))
        return 0
    if not args.budget_state or not args.private_dir or not os.environ.get(args.key_env):
        parser.error("live requires --budget-state, new --private-dir and selected key environment")
    args.private_dir.mkdir(mode=0o700, exist_ok=False)
    if args.private_dir.stat().st_mode & 0o077:
        raise ValueError("private_directory_permissions")
    _private_write(args.private_dir / "endpoint-metadata.json", _json_bytes(metadata))
    _private_write(args.private_dir / "fixed-labels.json", _json_bytes([{k:v for k,v in c.items() if k != "_base"} for c in cases]))
    budget = SharedBudget(args.budget_state, args.budget_usd)
    results = []
    for case, request in zip(cases, prepared):
        result = run_live_case(case, request, route, key=os.environ[args.key_env], budget=budget, private_dir=args.private_dir)
        results.append(result)
        _private_write(args.private_dir / (case["id"] + ".summary.json"), _json_bytes(result))
        print(json.dumps(result, sort_keys=True), flush=True)
    summary = summarize(results)
    _private_write(args.private_dir / "summary.json", _json_bytes(summary))
    print(json.dumps(summary, sort_keys=True))
    return 0 if all(r["schema_valid"] and r["route_verified"] and r["cost_status"] == "settled" and r.get("accuracy_match") is not False for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
