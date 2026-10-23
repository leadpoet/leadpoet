#!/usr/bin/env python3
"""Small synthetic OpenRouter compatibility probe. Never activates a route.

Uses only synthetic evidence, pins one provider with ZDR and no fallback, and
retains exact request/response bodies for independent storage/export checks.
The API key comes from the environment and is never included in the report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"
SCHEMA = {"type": "object", "properties": {
    "company": {"type": "string"}, "qualified": {"type": "boolean"},
    "evidence_ids": {"type": "array", "items": {"type": "string"}},
}, "required": ["company", "qualified", "evidence_ids"], "additionalProperties": False}
TOOLS = [{"type": "function", "function": {
    "name": "lookup_evidence", "description": "Read synthetic company evidence by ID.",
    "parameters": {"type": "object", "properties": {"evidence_id": {"type": "string", "enum": ["profile", "headcount"]}},
                   "required": ["evidence_id"], "additionalProperties": False},
}}]
INITIAL = [{"role": "user", "content": (
    "Synthetic compatibility test. Qualify Acme Test only if it is a US B2B software "
    "company with 50-200 employees. First call lookup_evidence with evidence_id exactly 'profile', then "
    "call lookup_evidence with evidence_id exactly 'headcount'. Make one call per turn. After both "
    "results return only JSON with company, qualified, evidence_ids. Use only "
    "the supplied tool evidence. Do not infer missing facts.") }]
EVIDENCE = {
    "profile": {"evidence_id": "profile", "company": "Acme Test", "country": "US", "business": "B2B software"},
    "headcount": {"evidence_id": "headcount", "employees": 80},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def assistant_history(message):
    # Preserve protocol fields byte-for-value; omit transport-only refusal/logprobs.
    return {k: copy.deepcopy(v) for k, v in message.items() if k in {
        "role", "content", "tool_calls", "reasoning", "reasoning_content", "reasoning_details"}}


def thinking_fields(message):
    fields = {}
    for key in ("reasoning", "reasoning_content"):
        if isinstance(message.get(key), str) and message[key].strip():
            fields[key] = len(message[key])
    for part in message.get("reasoning_details") or []:
        if isinstance(part, dict):
            for key in ("text", "summary"):
                if isinstance(part.get(key), str) and part[key].strip():
                    label = "reasoning_details." + key
                    fields[label] = fields.get(label, 0) + len(part[key])
    return fields


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def request(body, key):
    started = time.monotonic()
    req = urllib.request.Request(URL, data=canonical(body), headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json",
        "X-OpenRouter-Metadata": "enabled"})
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=120) as r:
            raw = r.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("probe_response_too_large")
            response = json.loads(raw)
            return {"status": r.status, "seconds": round(time.monotonic()-started, 3),
                    "request": body, "response": response,
                    "request_sha256": hashlib.sha256(canonical(body)).hexdigest(),
                    "response_sha256": hashlib.sha256(canonical(response)).hexdigest()}
    except urllib.error.HTTPError as exc:
        # Synthetic request only. Keep bounded provider error for route diagnosis.
        raw = exc.read(16384).decode("utf-8", "replace").replace(key, "[redacted]")
        return {"status": exc.code, "seconds": round(time.monotonic()-started, 3),
                "request": body, "error": raw}


def probe(*, model, provider, effort="high", max_tokens=8192, max_price=None, key, request_fn=request):
    messages = copy.deepcopy(INITIAL)
    attempts = []
    checks = {"two_tool_continuations": False, "final_json": False,
              "returned_text": False, "no_truncation": False}
    for turn in range(3):
        body = {"model": model, "messages": copy.deepcopy(messages), "max_tokens": max_tokens,
                "reasoning": {"effort": effort, "exclude": False}, "stream": False,
                "provider": {"only": [provider], "allow_fallbacks": False,
                             "zdr": True, "data_collection": "deny", "require_parameters": True}}
        if max_price is not None:
            if (set(max_price) != {"prompt", "completion", "request"}
                    or any(type(v) not in (int, float) or not 0 <= v <= 1000 for v in max_price.values())):
                raise ValueError("invalid_probe_price_limit")
            body["provider"]["max_price"] = copy.deepcopy(max_price)
        if turn < 2:
            body.update(tools=TOOLS, tool_choice={"type": "function", "function": {"name": "lookup_evidence"}})
        else:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "qualification", "strict": True, "schema": SCHEMA}}
        item = request_fn(body, key)
        item["turn"] = turn
        attempts.append(item)
        if item["status"] != 200:
            break
        response = item["response"]
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        item["thinking_fields"] = thinking_fields(message)
        if turn < 2:
            calls = message.get("tool_calls") or []
            if len(calls) != 1:
                item["failure"] = "expected_one_tool_call"
                break
            call = calls[0]
            try:
                arguments = json.loads(call["function"]["arguments"])
                expected = ("profile", "headcount")[turn]
                if call["function"]["name"] != "lookup_evidence" or arguments != {"evidence_id": expected}:
                    raise ValueError()
            except (KeyError, ValueError, TypeError):
                item["failure"] = "tool_arguments_invalid"
                break
            messages.extend([assistant_history(message), {"role": "tool", "tool_call_id": call["id"],
                             "content": canonical(EVIDENCE[expected]).decode()}])
        else:
            checks["two_tool_continuations"] = True
            try:
                output = json.loads(message.get("content") or "")
                checks["final_json"] = (output == {"company": "Acme Test", "qualified": True,
                                                   "evidence_ids": ["profile", "headcount"]})
            except (TypeError, ValueError):
                pass
    checks["returned_text"] = any(a.get("thinking_fields") for a in attempts)
    checks["no_truncation"] = len(attempts) == 3 and all(
        a.get("response", {}).get("choices", [{}])[0].get("finish_reason") in ("stop", "tool_calls")
        for a in attempts)
    cost = sum(Decimal(str(a.get("response", {}).get("usage", {}).get("cost") or 0)) for a in attempts)
    return {"probe_id": "full-thinking-" + model.replace("/", "-") + "-" + provider.replace("/", "-"),
            "verified_at": datetime.now(timezone.utc).isoformat(), "model": model, "provider": provider,
            "api": "openrouter.chat", "zdr": True, "checks": checks, "attempts": attempts,
            "reported_cost_usd": str(cost), "approval_granted": False,
            "limitation": "Synthetic compatibility evidence only; requires semantic review, gateway/export tests, and ICP evaluation."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--provider", required=True)
    p.add_argument("--effort", default="high", choices=("low", "medium", "high"))
    p.add_argument("--max-tokens", type=int, default=8192, choices=(4096, 8192))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-price-json", type=json.loads,
                   help="OpenRouter prompt/completion prices per million tokens and request price")
    args = p.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        p.error("OPENROUTER_API_KEY is required; never pass it as an argument")
    result = probe(model=args.model, provider=args.provider, effort=args.effort,
                   max_tokens=args.max_tokens, max_price=args.max_price_json, key=key)
    with args.output.open("x") as f:
        os.chmod(args.output, 0o600)
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("model", "provider", "checks", "reported_cost_usd")}))


if __name__ == "__main__":
    main()
