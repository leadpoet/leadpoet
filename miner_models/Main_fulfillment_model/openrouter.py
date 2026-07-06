"""
OpenRouter API client for LLM calls (Gemini, Perplexity).

Provides chat_completion and chat_completion_json with retry logic.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

import requests

from target_fit_model.config import (
    OPENROUTER_KEY,
    OPENROUTER_BASE_URL,
    LLM_TIMEOUT,
    PERPLEXITY_TIMEOUT,
    LLM_MAX_RETRIES,
    API_DELAY,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = LLM_MAX_RETRIES
_BACKOFF_BASE = 2

# ── Cost tracking ──
_MODEL_COSTS = {
    "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "perplexity/sonar-pro": {"input": 3.00, "output": 15.00},
    "perplexity/sonar-deep-research": {"input": 2.00, "output": 8.00},
    "perplexity/sonar-deep-research:online": {"input": 2.00, "output": 8.00},
    "openai/gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "openai/gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}

_cost_tracker = {
    "total_usd": 0.0,
    "calls": 0,
    "by_model": {},
    "scrapingdog_credits": 0,
}


def get_cost_tracker():
    return _cost_tracker


def reset_cost_tracker():
    _cost_tracker["total_usd"] = 0.0
    _cost_tracker["calls"] = 0
    _cost_tracker["by_model"] = {}
    _cost_tracker["scrapingdog_credits"] = 0


def track_scrapingdog(credits: int):
    _cost_tracker["scrapingdog_credits"] += credits
    _cost_tracker["total_usd"] += credits * 0.00005  # $0.00005 per credit


def _track_cost(model: str, tokens_in: int, tokens_out: int):
    costs = _MODEL_COSTS.get(model, {"input": 1.0, "output": 1.0})
    cost = (tokens_in * costs["input"] / 1_000_000) + (tokens_out * costs["output"] / 1_000_000)
    _cost_tracker["total_usd"] += cost
    _cost_tracker["calls"] += 1
    if model not in _cost_tracker["by_model"]:
        _cost_tracker["by_model"][model] = {"calls": 0, "cost": 0.0, "tokens_in": 0, "tokens_out": 0}
    _cost_tracker["by_model"][model]["calls"] += 1
    _cost_tracker["by_model"][model]["cost"] += cost
    _cost_tracker["by_model"][model]["tokens_in"] += tokens_in
    _cost_tracker["by_model"][model]["tokens_out"] += tokens_out
    return cost


def chat_completion(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    max_tokens: int = 4000,
    timeout: Optional[int] = None,
) -> Optional[str]:
    """
    Send a chat completion request to OpenRouter.
    Returns the assistant's response text, or None on failure.
    """
    api_key = OPENROUTER_KEY.strip()
    if not api_key:
        raise ValueError("OPENROUTER_KEY not set. Add it to your environment.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    effective_timeout = timeout or LLM_TIMEOUT
    if "deep-research" in model.lower():
        effective_timeout = max(effective_timeout, PERPLEXITY_TIMEOUT, 120)
    elif "perplexity" in model.lower():
        effective_timeout = max(effective_timeout, PERPLEXITY_TIMEOUT)

    start = time.time()

    try:
        resp = None
        for attempt in range(_MAX_RETRIES + 1):
            resp = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://leadpoet.ai",
                    "X-Title": "Leadpoet Intent Model",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=effective_timeout,
            )
            if resp.status_code == 429 and attempt < _MAX_RETRIES:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = min(int(retry_after), 60)
                    except (ValueError, TypeError):
                        wait = _BACKOFF_BASE * (3 ** attempt)
                else:
                    wait = _BACKOFF_BASE * (3 ** attempt)
                wait = min(wait, 60)
                logger.warning(
                    f"[OpenRouter] 429 rate limited, retry {attempt + 1}/{_MAX_RETRIES} in {wait}s"
                )
                time.sleep(wait)
                continue
            break

        assert resp is not None
        duration = time.time() - start

        if resp.status_code != 200:
            print(f"  [OpenRouter] HTTP {resp.status_code}: {resp.text[:200]}")
            logger.error(f"[OpenRouter] HTTP {resp.status_code}: {resp.text[:500]}")
            return None

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        call_cost = _track_cost(model, tokens_in, tokens_out)
        logger.info(
            f"[OpenRouter] {model} -> 200 | {duration:.1f}s | "
            f"{tokens_in}+{tokens_out} tokens | ${call_cost:.4f} | total: ${_cost_tracker['total_usd']:.4f}"
        )

        time.sleep(API_DELAY)
        return content

    except requests.Timeout:
        print(f"  [OpenRouter] TIMEOUT after {effective_timeout}s (model={model})")
        logger.error(f"[OpenRouter] Timed out after {effective_timeout}s (model={model})")
        return None
    except Exception as e:
        print(f"  [OpenRouter] ERROR: {e}")
        logger.error(f"[OpenRouter] Exception: {e}")
        return None


def chat_completion_json(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0,
    max_tokens: int = 4000,
    timeout: Optional[int] = None,
) -> Optional[dict | list]:
    """
    Chat completion that parses the response as JSON.
    Strips markdown code fences if present.
    """
    raw = chat_completion(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if raw is None:
        return None
    return parse_json_response(raw)


def parse_json_response(text: str) -> Optional[dict | list]:
    """Parse JSON from LLM response, stripping code fences if present."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # First attempt: parse as-is
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Second attempt: fix empty values like "key": , → "key": null,
    cleaned2 = re.sub(r':\s*,', ': null,', cleaned)
    cleaned2 = re.sub(r':\s*}', ': null}', cleaned2)
    try:
        return json.loads(cleaned2)
    except json.JSONDecodeError:
        pass

    # Third attempt: strip Perplexity citation markers "text"[1] (not valid JSON arrays like [7])
    cleaned3 = re.sub(r'"\s*\[\d+\]', '"', cleaned2)
    cleaned3 = re.sub(r'\[\d+\]\s*(?=[,}\]])', '', cleaned3)
    try:
        return json.loads(cleaned3)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object or array in the text
    for attempt in [cleaned, cleaned2, cleaned3]:
        for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
            match = re.search(pattern, attempt)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

    # Truncated array recovery
    if cleaned.lstrip().startswith("["):
        last_brace = cleaned.rfind("}")
        if last_brace > 0:
            candidate = cleaned[:last_brace + 1].rstrip().rstrip(",") + "\n]"
            try:
                result = json.loads(candidate)
                if isinstance(result, list) and result:
                    logger.info(f"[OpenRouter] Recovered {len(result)} items from truncated array")
                    return result
            except json.JSONDecodeError:
                pass

    logger.error(f"[OpenRouter] Failed to parse JSON: {cleaned[:200]}...")
    return None
