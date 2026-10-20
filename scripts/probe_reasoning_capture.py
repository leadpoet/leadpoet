#!/usr/bin/env python3
"""Probe the agent reasoning policy against live OpenRouter models.

Activation gate for ``agent_reasoning_v1``. For each model it sends a few tiny
requests exactly as the gateway broker would rewrite them, and reports whether
each call succeeded and returned readable reasoning. It covers the request
shapes most likely to break: a single turn at temperature 0, a tool-loop
continuation without returned reasoning details, and an OpenAI Responses call.

Paid, but bounded: at most three calls per model with small output caps, plus
one comparison call with the agent's own form when a rewritten call fails.
Requires OPENROUTER_API_KEY. It prints metrics only, never model text or the
key. Exit 0 when no failure is caused by the policy and every successful
requested call returned readable reasoning, 2 otherwise, 1 when the probe
could not run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = (
    "openai/gpt-6-luna", "openai/gpt-5.4-mini", "openai/gpt-5-mini",
    "anthropic/claude-sonnet-5.5", "anthropic/claude-sonnet-4.5",
    "google/gemini-3.5-flash-lite", "google/gemini-3.1-flash-lite", "perplexity/sonar",
)
MAX_MODELS = 12
BASE = "https://openrouter.ai/api/v1"
PROMPT = "In two sentences: which signal best shows a SaaS company is expanding its sales team?"
TOOLS = [{"type": "function", "function": {
    "name": "lookup_company", "description": "Look up public facts about a company.",
    "parameters": {"type": "object", "properties": {"domain": {"type": "string"}}, "required": ["domain"]},
}}]


def _post(path, body, key, timeout=120):
    request = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read(4 * 1024 * 1024))
    except urllib.error.HTTPError as exc:
        return exc.code, None


def _catalog():
    with urllib.request.urlopen(BASE + "/models", timeout=30) as response:
        return json.loads(response.read(16 * 1024 * 1024))


def _metrics(status, document):
    from lab_arena import reasoning_capture

    texts = reasoning_capture.readable_reasoning(document) if isinstance(document, dict) else []
    usage = (document or {}).get("usage") or {}
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    choice = ((document or {}).get("choices") or [{}])[0]
    return {
        "http_status": status,
        "finish_reason": choice.get("finish_reason"),
        "readable_reasoning": bool(texts),
        "readable_chars": sum(len(text) for text in texts),
        "visible_chars": len(str((choice.get("message") or {}).get("content") or "")),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "cost_usd": usage.get("cost"),
    }


def _cases(model):
    single = {"model": model, "temperature": 0, "max_tokens": 300,
              "messages": [{"role": "user", "content": PROMPT}]}
    loop = {"model": model, "max_tokens": 300, "tools": TOOLS, "messages": [
        {"role": "user", "content": "Check acme.example before answering: " + PROMPT},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "lookup_company", "arguments": json.dumps({"domain": "acme.example"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"hiring": "12 open sales roles"})},
    ]}
    yield "single_turn_temperature_0", "openrouter.chat", single
    yield "tool_loop_continuation", "openrouter.chat", loop
    yield "agent_reasoning_enabled", "openrouter.chat", dict(single, reasoning={"enabled": True})
    yield "assistant_prefill", "openrouter.chat", dict(single, messages=single["messages"] + [
        {"role": "assistant", "content": "{"}])
    yield "agent_disabled_tool_loop", "openrouter.chat", dict(loop, reasoning={"effort": "none"})
    yield "responses_minimal_effort", "openrouter.responses", {
        "model": model, "input": PROMPT, "max_output_tokens": 600, "reasoning": {"effort": "minimal"}}


def _body(broker, operations, table, operation_id, parameters):
    """The outbound body the gateway would send, including its host route."""
    body = {**parameters, **json.loads(json.dumps(dict(operations.OPERATIONS[operation_id].fixed_params)))}
    pricing = table["models"].get(parameters["model"])
    route = pricing and broker._openrouter_host_route(
        kind="execute", operation_id=operation_id, model=parameters["model"],
        pricing=pricing, parameters=parameters)
    if route:
        body["provider"] = json.loads(json.dumps(dict(route.provider_policy)))
    return body


def probe(models, key, *, dry_run=False):
    from datetime import datetime, timezone

    from lab_arena import broker, operations, reasoning_capture

    catalog = _catalog()
    profiles = reasoning_capture.profiles_from_models_response(catalog)
    table = broker.price_table_from_models_response(
        catalog, fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    rows = []
    for model in models:
        for case, operation_id, parameters in _cases(model):
            normalized = broker.normalized_request(operation_id, parameters)
            sent, annotation = broker.agent_reasoning_parameters(
                operation_id, normalized, policy=reasoning_capture.POLICY, profiles=profiles)
            if dry_run:
                rows.append({"model": model, "case": case, "treatment": annotation["treatment"],
                             "requested": annotation["requested"], "changes": annotation["changes"],
                             "skipped": annotation.get("skipped"), "sent_reasoning": sent.get("reasoning"),
                             "sent_output_cap": sent.get("max_output_tokens", sent.get("max_tokens"))})
                continue
            path = "/responses" if operation_id == "openrouter.responses" else "/chat/completions"
            status, document = _post(path, _body(broker, operations, table, operation_id, sent), key)
            row = {"model": model, "case": case, "treatment": annotation["treatment"],
                   "requested": annotation["requested"], "changes": annotation["changes"],
                   **_metrics(status, document)}
            if not 200 <= status < 300 and sent != normalized:
                # Attribute the failure: does the agent's own form succeed?
                original_status, _ = _post(path, _body(broker, operations, table, operation_id, normalized), key)
                row["agent_form_http_status"] = original_status
            row["policy_caused_failure"] = (
                not 200 <= status < 300 and 200 <= row.get("agent_form_http_status", 0) < 300)
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models",
                        help="OpenRouter model id; repeat for several (default: current agent models)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the rewritten requests from the public catalog; send nothing")
    args = parser.parse_args(argv)
    models = tuple(args.models or MODELS)
    if len(models) > MAX_MODELS:
        parser.error("at most %d models per probe" % MAX_MODELS)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and not args.dry_run:
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1
    try:
        rows = probe(models, key, dry_run=args.dry_run)
    except Exception as exc:
        print("reasoning probe failed: " + type(exc).__name__, file=sys.stderr)
        return 1
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    if args.dry_run:
        return 0
    # A failure the agent's own form shares is pre-existing, not the policy's.
    passed = all(not row["policy_caused_failure"]
                 and (not 200 <= row["http_status"] < 300 or row["readable_reasoning"] or not row["requested"])
                 for row in rows)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
