"""One-company, live web-research Arena source.

The Arena worker socket is the only network boundary available to this
bundle.  The broker supplies the provider credential and fixed policy after
the request crosses that socket.
"""

from __future__ import annotations

import datetime as _datetime
import ipaddress
import json
import os
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx


MODEL = "perplexity/sonar-pro"
WORKER_SOCKET_ENV = "LAB_ARENA_WORKER_SOCKET"
OPENROUTER_URL = "http://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 120.0
PRIMARY_SIGNAL_INDEX = 0
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "companies": {
            "type": "array",
            "maxItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "company_name": {"type": "string"},
                    "company_website": {"type": "string"},
                    "company_linkedin": {"type": "string"},
                    "industry": {"type": "string"},
                    "employee_count": {"type": "string"},
                    "company_stage": {"type": "string"},
                    "country": {"type": "string"},
                    "state": {"type": "string"},
                    "fit_summary": {"type": "string"},
                    "fit_evidence_urls": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {"type": "string"},
                    },
                    "intent_signals": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "matched_icp_signal": {"type": "integer", "minimum": 0},
                                "description": {"type": "string"},
                                "date": {"type": "string"},
                                "why_now": {"type": "string"},
                                "url": {"type": "string"},
                                "snippet": {"type": "string"},
                            },
                            "required": [
                                "matched_icp_signal",
                                "description",
                                "date",
                                "why_now",
                                "url",
                                "snippet",
                            ],
                        },
                    },
                },
                "required": [
                    "company_name",
                    "company_website",
                    "company_linkedin",
                    "industry",
                    "employee_count",
                    "company_stage",
                    "country",
                    "state",
                    "fit_summary",
                    "fit_evidence_urls",
                    "intent_signals",
                ],
            },
        }
    },
    "required": ["companies"],
}


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or CONTROL_RE.search(value):
        return None
    return value


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str) or CONTROL_RE.search(value):
        return None
    return value.strip()


def _public_url(value: Any) -> str | None:
    value = _text(value)
    if value is None:
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
        or port is None and ":" in parsed.netloc.rsplit("@", 1)[-1]
    ):
        return None
    host = hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".internal", ".invalid", ".local", ".localhost", ".onion", ".test")):
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        if not address.is_global:
            return None
    else:
        labels = host.split(".")
        if len(labels) < 2 or not labels[-1].isalpha():
            return None
    return value


def _date(value: Any) -> str | None:
    value = _text(value)
    if value is None or DATE_RE.fullmatch(value) is None:
        return None
    try:
        _datetime.date.fromisoformat(value)
    except ValueError:
        return None
    return value


def _company(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    expected = {
        "company_name", "company_website", "company_linkedin", "industry",
        "employee_count", "company_stage", "country", "state", "fit_summary",
        "fit_evidence_urls", "intent_signals",
    }
    if set(value) != expected:
        return None
    result: dict[str, Any] = {}
    for key in ("company_name", "industry", "employee_count", "country", "fit_summary"):
        text = _text(value.get(key))
        if text is None:
            return None
        result[key] = text
    website = _public_url(value.get("company_website"))
    if website is None:
        return None
    result["company_website"] = website
    linkedin = _optional_text(value.get("company_linkedin"))
    if linkedin:
        linkedin = _public_url(linkedin)
        if linkedin is None:
            return None
    result["company_linkedin"] = linkedin or ""
    for key in ("company_stage", "state"):
        text = _optional_text(value.get(key))
        if text is None:
            return None
        result[key] = text
    evidence = value.get("fit_evidence_urls")
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 5:
        return None
    result["fit_evidence_urls"] = []
    for item in evidence:
        url = _public_url(item)
        if url is None:
            return None
        result["fit_evidence_urls"].append(url)
    signals = value.get("intent_signals")
    if not isinstance(signals, list) or len(signals) != 1:
        return None
    signal = signals[0]
    if not isinstance(signal, Mapping) or set(signal) != {
        "matched_icp_signal", "description", "date", "why_now", "url", "snippet"
    }:
        return None
    if type(signal.get("matched_icp_signal")) is not int or signal["matched_icp_signal"] != PRIMARY_SIGNAL_INDEX:
        return None
    clean_signal: dict[str, Any] = {"matched_icp_signal": PRIMARY_SIGNAL_INDEX}
    for key in ("description", "why_now", "snippet"):
        text = _text(signal.get(key))
        if text is None:
            return None
        clean_signal[key] = text
    signal_date = _date(signal.get("date"))
    signal_url = _public_url(signal.get("url"))
    if signal_date is None or signal_url is None:
        return None
    clean_signal["date"] = signal_date
    clean_signal["url"] = signal_url
    result["intent_signals"] = [clean_signal]
    return result


def _request(icp: Mapping[str, Any]) -> dict[str, Any]:
    icp_json = json.dumps(dict(icp), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    system = (
        "Use live web research to find at most one company matching the exact ICP. "
        "Every returned fact must be supported by the cited public URL. Never guess, "
        "invent, or substitute today's date for a missing event date; a real event "
        "dated today is valid. If no company and dated primary intent signal "
        "can be verified, return {\"companies\":[]}. The only valid matched_icp_signal "
        "for this task is zero."
    )
    user = (
        "Research this ICP now with your web search: " + icp_json + "\n"
        "Return only the requested JSON schema. A company requires one truthful dated "
        "intent signal, public HTTPS URLs, and all required fit fields."
    )
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 1200,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "single_company_research",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
    }


def run_icp(icp: dict[str, Any]) -> list[dict[str, Any]]:
    """Make exactly one brokered Sonar Pro request for this ICP."""

    if not isinstance(icp, dict):
        return []
    socket_path = str(os.environ.get(WORKER_SOCKET_ENV) or "").strip()
    if not socket_path.startswith("/"):
        return []
    request = _request(icp)
    transport = httpx.HTTPTransport(uds=socket_path, retries=0)
    with httpx.Client(transport=transport, timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.post(
            OPENROUTER_URL,
            headers={"content-type": "application/json"},
            json=request,
        )
    response.raise_for_status()
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError("unexpected provider status")
    try:
        document = response.json()
        content = document["choices"][0]["message"]["content"]
        decoded = json.loads(content) if isinstance(content, str) else None
        companies = decoded.get("companies") if isinstance(decoded, Mapping) else None
        if not isinstance(companies, list) or len(companies) > 1:
            return []
        if not companies:
            return []
        company = _company(companies[0])
        return [company] if company is not None else []
    except (IndexError, KeyError, TypeError, ValueError):
        return []
