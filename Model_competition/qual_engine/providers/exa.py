"""Exa client: /search (neural + keyword), /findSimilar, /contents."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Optional

import httpx

from qual_engine.config import CONFIG
from qual_engine.infra.cache import Cache
from qual_engine.infra.rate_limit import get_semaphore
from qual_engine.infra.retry import retry_async

logger = logging.getLogger(__name__)


# Exa pricing is roughly $0.005 per search call, $0.001 per contents call.
EXA_SEARCH_COST_USD = 0.005
EXA_CONTENTS_COST_USD = 0.001


class ExaClient:
    def __init__(self, client: httpx.AsyncClient, cache: Cache):
        self._client = client
        self._cache = cache
        self._api_key = CONFIG.EXA_API_KEY
        self._headers = {"x-api-key": self._api_key, "Content-Type": "application/json"}

    async def _post(self, path: str, body: dict, timeout: float, label: str) -> dict:
        async def do_call():
            async with get_semaphore("exa"):
                resp = await self._client.post(
                    f"https://api.exa.ai{path}", headers=self._headers, json=body, timeout=timeout
                )
                if resp.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"Exa {resp.status_code}: {resp.text[:300]}",
                        request=resp.request,
                        response=resp,
                    )
                return resp.json()

        return await retry_async(do_call, label=f"exa:{label}")

    async def search_neural(
        self,
        query: str,
        *,
        num_results: int = 10,
        days_back: Optional[int] = None,
        include_domains: Optional[list[str]] = None,
        use_autoprompt: bool = True,
    ) -> dict:
        body = {
            "query": query,
            "numResults": num_results,
            "type": "neural",
            "useAutoprompt": use_autoprompt,
        }
        if days_back:
            body["startPublishedDate"] = (date.today() - timedelta(days=days_back)).isoformat()
        if include_domains:
            body["includeDomains"] = include_domains
        cache_key = self._cache.make_key("exa", "search_neural", body)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}
        t0 = time.time()
        try:
            data = await self._post("/search", body, CONFIG.EXA_TIMEOUT, "search_neural")
        except Exception as e:
            logger.warning("Exa neural search failed for %r: %s", query, e)
            return {"results": [], "cached": False, "cost_usd": 0, "elapsed_s": 0, "error": str(e)[:200]}
        result = {
            "results": data.get("results", []),
            "cost_usd": EXA_SEARCH_COST_USD,
            "elapsed_s": round(time.time() - t0, 3),
            "cached": False,
        }
        self._cache.set(cache_key, result, CONFIG.EXA_SEARCH_TTL)
        return result

    async def search_keyword(
        self, query: str, *, num_results: int = 10, days_back: Optional[int] = None
    ) -> dict:
        body = {
            "query": query,
            "numResults": num_results,
            "type": "keyword",
        }
        if days_back:
            body["startPublishedDate"] = (date.today() - timedelta(days=days_back)).isoformat()
        cache_key = self._cache.make_key("exa", "search_keyword", body)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}
        t0 = time.time()
        try:
            data = await self._post("/search", body, CONFIG.EXA_TIMEOUT, "search_keyword")
        except Exception as e:
            logger.warning("Exa keyword search failed for %r: %s", query, e)
            return {"results": [], "cached": False, "cost_usd": 0, "elapsed_s": 0, "error": str(e)[:200]}
        result = {
            "results": data.get("results", []),
            "cost_usd": EXA_SEARCH_COST_USD,
            "elapsed_s": round(time.time() - t0, 3),
            "cached": False,
        }
        self._cache.set(cache_key, result, CONFIG.EXA_SEARCH_TTL)
        return result


    async def contents(self, url: str) -> dict:
        """Fetch clean markdown content for a URL."""
        cache_key = self._cache.make_key("exa", "contents", url)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}
        body = {"ids": [url], "text": True}
        t0 = time.time()
        try:
            data = await self._post("/contents", body, CONFIG.EXA_TIMEOUT, "contents")
        except Exception as e:
            logger.warning("Exa contents failed for %s: %s", url, e)
            return {"text": "", "cached": False, "cost_usd": 0, "elapsed_s": 0, "error": str(e)[:200]}
        results = data.get("results") or []
        text = results[0].get("text", "") if results else ""
        title = results[0].get("title", "") if results else ""
        result = {
            "text": text,
            "title": title,
            "url": url,
            "cost_usd": EXA_CONTENTS_COST_USD,
            "elapsed_s": round(time.time() - t0, 3),
            "cached": False,
        }
        # Only cache non-empty
        if text:
            self._cache.set(cache_key, result, CONFIG.EXA_CONTENTS_TTL)
        return result
