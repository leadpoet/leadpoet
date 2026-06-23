#!/usr/bin/env python3
"""Probe OpenRouter usage accounting with a tiny request.

Set OPENROUTER_KEY or OPENROUTER_API_KEY before running. The key value is never
printed. This is intentionally separate from normal verification because it
uses live network and paid provider access.
"""

from __future__ import annotations

import json
import os
import re
import sys
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


def main() -> int:
    api_key = os.getenv("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("SKIP: set OPENROUTER_KEY or OPENROUTER_API_KEY to run the live usage probe")
        return 2

    model = os.getenv("OPENROUTER_USAGE_PROBE_MODEL", "openai/gpt-4.1-nano")
    max_tokens = int(os.getenv("OPENROUTER_USAGE_PROBE_MAX_TOKENS", "16"))
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with ok."}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urlrequest.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        error_body = re.sub(r"sk-or-[A-Za-z0-9_-]+", "[redacted-openrouter-key]", error_body)
        detail = f": {error_body[:500]}" if error_body else ""
        print(f"ERROR: OpenRouter usage probe failed: HTTP {exc.code}{detail}")
        return 1
    except (URLError, json.JSONDecodeError) as exc:
        print(f"ERROR: OpenRouter usage probe failed: {exc}")
        return 1

    usage = decoded.get("usage") if isinstance(decoded.get("usage"), dict) else {}
    errors = []
    if not decoded.get("id"):
        errors.append("missing response id")
    if int(usage.get("total_tokens") or 0) <= 0:
        errors.append("missing usage.total_tokens")
    if usage.get("cost") is None:
        errors.append("missing usage.cost")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(
        "OpenRouter usage accounting verified: "
        f"id present, total_tokens={int(usage['total_tokens'])}, cost={usage['cost']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
