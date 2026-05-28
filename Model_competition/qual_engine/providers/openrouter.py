"""OpenRouter wrapper for all LLM calls (Claude, GPT, Sonar)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

import httpx

from qual_engine.config import CONFIG
from qual_engine.infra.cache import Cache
from qual_engine.infra.rate_limit import get_semaphore
from qual_engine.infra.retry import retry_async

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(text: str) -> Any:
    """Robust JSON extraction; tolerates code fences and trailing prose."""
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try to find first {...} or [...] block
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == opener:
                depth += 1
            elif cleaned[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


class OpenRouterClient:
    def __init__(self, client: httpx.AsyncClient, cache: Cache):
        self._client = client
        self._cache = cache
        self._api_key = CONFIG.OPENROUTER_API_KEY
        self._url = "https://openrouter.ai/api/v1/chat/completions"

    async def chat(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        label: str = "chat",
    ) -> dict:
        """Returns dict with keys: text, cost_usd, elapsed_s, model.
        Cache only for temperature==0 (deterministic)."""

        cache_key = None
        if use_cache and temperature == 0:
            cache_key = self._cache.make_key("openrouter", model, prompt, temperature)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return {**cached, "cached": True}

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async def do_call():
            async with get_semaphore("openrouter"):
                t0 = time.time()
                resp = await self._client.post(
                    self._url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=CONFIG.OPENROUTER_TIMEOUT,
                )
                if resp.status_code >= 400:
                    body = resp.text[:300]
                    raise httpx.HTTPStatusError(
                        f"OpenRouter {resp.status_code}: {body}",
                        request=resp.request,
                        response=resp,
                    )
                data = resp.json()
                return data, time.time() - t0

        try:
            data, elapsed = await retry_async(do_call, label=f"openrouter:{label}")
        except Exception as e:
            logger.error("OpenRouter call failed (%s): %s", label, e)
            raise

        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        cost_usd = (data.get("usage") or {}).get("cost", 0.0) or 0.0
        result = {
            "text": text,
            "cost_usd": float(cost_usd),
            "elapsed_s": round(elapsed, 3),
            "model": model,
            "cached": False,
        }
        if cache_key:
            self._cache.set(cache_key, result, CONFIG.LLM_TEMP0_TTL)
        return result

    async def json_call(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        use_cache: bool = True,
        label: str = "json",
    ) -> dict:
        """Returns {parsed: dict|list|None, raw: str, cost_usd, elapsed_s, model, cached}."""
        result = await self.chat(
            model, prompt, temperature=temperature, use_cache=use_cache, label=label
        )
        parsed = _extract_json(result["text"])
        return {**result, "parsed": parsed, "raw": result["text"]}
