"""Recorded-tape helpers for the Lab Arena benchmark tests.

The tape files under ``tests/lab_arena/fixtures/benchmark/`` hold generation
responses shaped exactly like OpenRouter chat completions from the Lab's
generator model. They were produced by ``write_default_tapes`` (deterministic,
no network) and are committed so the tests replay fixed bytes. Exclusion
prompts are answered by ``exclusion_response`` from the ICP's industry so the
tape stays small.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from gateway.tasks.icp_generator import INDUSTRY_DISTRIBUTION, INTERNATIONAL_GEOGRAPHIES, GEOGRAPHIES

from lab_arena.benchmark import ProviderFailure

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "benchmark"
INDUSTRIES = list(INDUSTRY_DISTRIBUTION)
STAGES = ["Seed", "Series A", "Series B", "Series C+", "Private Equity", "Public"]
BUCKETS = ["11-50", "51-200", "201-500", "501-1,000", "1,001-5,000"]
DOMESTIC = ["United States", "United States, West Coast", "United States, Northeast", "United States, California", "United States, Texas"]
CATEGORIES = ["FUNDING", "PRODUCT_LAUNCH", "MARKET_EXPANSION", "ACQUISITION", "LEADERSHIP_CHANGE", "PARTNERSHIP", "HIRING", "FACILITY_OPENING", "REGULATORY_CLEARANCE"]
SIGNAL_TEMPLATES = [
    "Announced a Series A or later funding round in the last 12 months, per a press release or news coverage ({industry} {n})",
    "Launched a new product or major platform capability in the last 12 months, per a press release or changelog ({industry} {n})",
    "Expanded into a new country or region in the last year, per a press release or company announcement ({industry} {n})",
    "Announced a strategic partnership in the last 12 months, per a press release ({industry} {n})",
    "Actively hiring for platform, integration, or revenue-operations roles, per current job postings ({industry} {n})",
]


def raw_icp(industry: str, ordinal: int, *, variant: int = 0) -> Dict[str, Any]:
    """One realistic raw generator ICP; ``ordinal`` makes content unique."""

    index = INDUSTRIES.index(industry)
    stage = STAGES[(index + ordinal) % len(STAGES)]
    if stage == "Seed":
        buckets = ["11-50", "51-200"]
    elif stage == "Public":
        buckets = ["1,001-5,000", "5,001-10,000"]
    else:
        start = (index + ordinal) % 3
        buckets = BUCKETS[start:start + 3]
    international = (index + ordinal) % 4 == 0
    geography = INTERNATIONAL_GEOGRAPHIES[(index + ordinal) % len(INTERNATIONAL_GEOGRAPHIES)] if international else DOMESTIC[(index + ordinal) % len(DOMESTIC)]
    country = geography.split(",")[0] if international else "United States"
    signal = SIGNAL_TEMPLATES[(index + ordinal + variant) % len(SIGNAL_TEMPLATES)].format(industry=industry, n=ordinal + variant)
    bonus = SIGNAL_TEMPLATES[(index + ordinal + 1 + variant) % len(SIGNAL_TEMPLATES)].format(industry=industry, n=ordinal + 7 + variant)
    return {
        "icp_id": "icp_%d_%03d" % (20260902, index + 1),
        "prompt": "I need %s companies in %s at the %s stage showing recent momentum (set %d)" % (industry.lower(), geography, stage, ordinal),
        "industry": industry,
        "sub_industry": "%s specialists" % industry,
        "geography": geography,
        "country": country,
        "employee_count": buckets,
        "company_stage": stage,
        "product_service": "A subscription platform that %s teams use to run their core workflows" % industry.lower(),
        "required_attribute": "Sells a %s product used by operating teams" % industry.lower(),
        "intent_signal": signal,
        "intent_category": CATEGORIES[(index + ordinal) % len(CATEGORIES)],
        "intent_max_age_days": 365,
        "intent_signals": [signal, bonus],
        "bonus_intents": [{"intent_signal": bonus, "intent_category": CATEGORIES[(index + ordinal + 1) % len(CATEGORIES)], "intent_max_age_days": 365}],
        "verified_example_company": "%s Example Co %d" % (industry, ordinal),
    }


def completion(icps: Sequence[Mapping[str, Any]], *, response_id: str, fenced: bool = False) -> Dict[str, Any]:
    content = json.dumps({"icps": list(icps)}, ensure_ascii=False)
    if fenced:
        content = "```json\n" + content + "\n```"
    return {
        "id": response_id,
        "object": "chat.completion",
        "model": "perplexity/sonar-pro",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3100, "completion_tokens": 2400, "total_tokens": 5500},
    }


def batch_icps(*, ordinal_base: int = 0):
    """The two fixed batches as raw ICP lists: b1 over the 20 industries, b2 over the first 10."""

    first = [raw_icp(industry, ordinal_base + 1) for industry in INDUSTRIES]
    second = [raw_icp(industry, ordinal_base + 2) for industry in INDUSTRIES[:10]]
    return first, second


def batch_responses(*, ordinal_base: int = 0) -> List[Dict[str, Any]]:
    first, second = batch_icps(ordinal_base=ordinal_base)
    return [
        completion(first, response_id="gen-b1"),
        completion(second, response_id="gen-b2", fenced=True),
    ]


def write_default_tapes() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "clean_run.json").write_text(json.dumps(batch_responses(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # A run needing one replacement: b1 carries two invalid ICPs (slot 3 lacks an
    # example company, slot 5 uses an unsupported geography) and b2 repeats b1's
    # Biotechnology ICP verbatim (slot 27 duplicate), so slots 3, 5 and 27 are replaced.
    first, second = batch_icps()
    first[3]["verified_example_company"] = ""
    first[5]["geography"] = "Europe"
    second[7] = raw_icp("Biotechnology", 1)
    replacement = [raw_icp("Hardware", 9), raw_icp("Privacy and Security", 9), raw_icp("Biotechnology", 9)]
    flawed = [
        completion(first, response_id="gen-b1-flawed"),
        completion(second, response_id="gen-b2-flawed", fenced=True),
        completion(replacement, response_id="gen-r1"),
    ]
    (FIXTURES / "replacement_run.json").write_text(json.dumps(flawed, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # Exhausted attempts: every Software ICP carries an unsupported stage.
    exhausted = []
    for batch, response_id in zip(batch_icps(), ("gen-b1-exhausted", "gen-b2-exhausted")):
        for icp in batch:
            if icp["industry"] == "Software":
                icp["company_stage"] = "Unicorn"
        exhausted.append(completion(batch, response_id=response_id))
    (FIXTURES / "exhausted_run.json").write_text(json.dumps(exhausted, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_tape(name: str) -> List[Dict[str, Any]]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


_EXCLUSION_RE = re.compile(r"- Industry: ([^(\n]+) \(")


def exclusion_response(prompt: str, *, count: int = 1) -> Dict[str, Any]:
    match = _EXCLUSION_RE.search(prompt)
    industry = match.group(1).strip() if match else "unknown"
    slug = re.sub(r"[^a-z0-9]+", "", industry.lower()) or "unknown"
    rows = [{"name": "%s Excluded %d" % (industry, i + 1), "domain": "%s-excluded-%d.example.com" % (slug, i + 1)} for i in range(count)]
    return {
        "id": "excl-" + slug,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(rows)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 60, "total_tokens": 260},
    }


class TapeProvider:
    """Replays generation responses in order and synthesizes exclusion answers.

    ``fail_at`` is a set of 1-based generation call ordinals that raise
    ``ProviderFailure`` (the unknown-outcome path); ``crash_after`` raises a
    ``RuntimeError`` right after the given call returns, modelling a service
    crash between the provider response and the next journal write.
    """

    def __init__(self, responses: Sequence[Mapping[str, Any]], *, fail_at: Optional[set] = None, crash_after: Optional[int] = None, exclusion_failures: Optional[set] = None) -> None:
        self._responses = list(responses)
        self._fail_at = set(fail_at or ())
        self._crash_after = crash_after
        self._exclusion_failures = set(exclusion_failures or ())
        self.generation_calls = 0
        self.exclusion_calls = 0
        self.requests: List[Dict[str, Any]] = []
        self.crashed = False

    def chat(self, *, messages, temperature, max_tokens, timeout_seconds):
        self.requests.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens, "timeout_seconds": timeout_seconds})
        prompt = messages[-1]["content"]
        if prompt.startswith("Name ") and "compan" in prompt:
            self.exclusion_calls += 1
            match = _EXCLUSION_RE.search(prompt)
            industry = match.group(1).strip() if match else ""
            if industry in self._exclusion_failures:
                raise ProviderFailure("exclusion provider unavailable")
            return exclusion_response(prompt)
        self.generation_calls += 1
        ordinal = self.generation_calls
        if ordinal in self._fail_at:
            raise ProviderFailure("generation provider timeout")
        if not self._responses:
            raise ProviderFailure("tape exhausted")
        response = self._responses.pop(0)
        if self._crash_after is not None and ordinal == self._crash_after:
            self.crashed = True
            raise RuntimeError("simulated service crash after provider response")
        return response


if __name__ == "__main__":
    write_default_tapes()
    print("tapes written to", FIXTURES)
