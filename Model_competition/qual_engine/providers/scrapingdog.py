"""ScrapingDog: LinkedIn company / jobs / Google Search."""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from qual_engine.config import CONFIG
from qual_engine.infra.cache import Cache
from qual_engine.infra.rate_limit import get_semaphore
from qual_engine.infra.retry import retry_async

logger = logging.getLogger(__name__)


# Approx costs (per call) — used for cost tracking only.
SD_LINKEDIN_COMPANY_COST = 0.0005     # ~10 credits at $0.0000476/credit
SD_LINKEDIN_JOBS_COST = 0.0005
SD_GOOGLE_COST = 0.0001               # ~2 credits


class ScrapingDogClient:
    def __init__(self, client: httpx.AsyncClient, cache: Cache):
        self._client = client
        self._cache = cache
        self._api_key = CONFIG.SCRAPINGDOG_API_KEY
        self._base = "https://api.scrapingdog.com"

    async def _get(self, path: str, params: dict, label: str) -> dict:
        params = {"api_key": self._api_key, **params}

        async def do_call():
            async with get_semaphore("scrapingdog"):
                resp = await self._client.get(
                    f"{self._base}{path}", params=params, timeout=CONFIG.SCRAPINGDOG_TIMEOUT
                )
                if resp.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"ScrapingDog {resp.status_code}: {resp.text[:300]}",
                        request=resp.request,
                        response=resp,
                    )
                return resp.json()

        return await retry_async(do_call, label=f"sd:{label}")

    # -------- LinkedIn Company --------
    async def linkedin_company(self, slug: str) -> dict:
        cache_key = self._cache.make_key("sd", "linkedin_company", slug)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}
        t0 = time.time()
        try:
            data = await self._get(
                "/linkedin", {"type": "company", "linkId": slug, "private": "false"}, "linkedin_company"
            )
        except Exception as e:
            logger.warning("SD linkedin_company failed for %r: %s", slug, e)
            return {"data": None, "cached": False, "cost_usd": 0, "elapsed_s": 0, "error": str(e)[:200]}
        # Response is either a dict or a list with one dict
        if isinstance(data, list):
            data = data[0] if data else {}
        result = {
            "data": data,
            "cost_usd": SD_LINKEDIN_COMPANY_COST,
            "elapsed_s": round(time.time() - t0, 3),
            "cached": False,
        }
        if data:
            self._cache.set(cache_key, result, CONFIG.SD_LINKEDIN_COMPANY_TTL)
        return result

    # -------- LinkedIn Jobs Search --------
    async def linkedin_jobs(
        self,
        field: str,
        *,
        geoid: str = "103644278",  # United States by default
        page: int = 1,
        sort_by: str = "day",
    ) -> dict:
        """Generic LinkedIn jobs keyword search.

        Returns: jobs list with keys job_position, job_link, job_id, company_name,
        company_profile, job_location, job_posting_date, company_logo_url.

        Note: results are NOT pre-filtered by company; use `linkedin_jobs_by_company`
        for company-targeted searches.
        """
        params = {"field": field, "page": str(page), "sort_by": sort_by}
        if geoid:
            params["geoid"] = geoid
        cache_key = self._cache.make_key("sd", "linkedin_jobs", params)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}
        t0 = time.time()
        try:
            data = await self._get("/linkedinjobs", params, "linkedin_jobs")
        except Exception as e:
            logger.warning("SD linkedin_jobs failed for %r: %s", field, e)
            return {"jobs": [], "cached": False, "cost_usd": 0, "elapsed_s": 0, "error": str(e)[:200]}
        jobs = data if isinstance(data, list) else data.get("jobs", [])
        result = {
            "jobs": jobs,
            "cost_usd": SD_LINKEDIN_JOBS_COST,
            "elapsed_s": round(time.time() - t0, 3),
            "cached": False,
        }
        if jobs:
            self._cache.set(cache_key, result, CONFIG.SD_LINKEDIN_JOBS_TTL)
        return result

    async def linkedin_jobs_by_company(
        self,
        company_name: str,
        *,
        role_keywords: str = "",
        max_pages: int = 2,
        geoid: str = "103644278",
    ) -> dict:
        """Fetch jobs and post-filter by exact-ish company name match.

        Returns jobs where company_name from the response contains company_name argument
        (case-insensitive, normalized).
        """
        import re

        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", (s or "").lower())

        target = norm(company_name)
        if not target:
            return {"jobs": [], "cached": False, "cost_usd": 0, "elapsed_s": 0}

        field = f"{role_keywords} {company_name}".strip() if role_keywords else company_name
        filtered = []
        total_cost = 0.0
        total_elapsed = 0.0

        for page in range(1, max_pages + 1):
            page_result = await self.linkedin_jobs(field, geoid=geoid, page=page)
            total_cost += page_result.get("cost_usd", 0)
            total_elapsed += page_result.get("elapsed_s", 0)
            for job in page_result.get("jobs", []):
                sc = norm(job.get("company_name", ""))
                if target in sc or sc in target:
                    filtered.append(job)
            if not page_result.get("jobs"):
                break  # no more results

        return {
            "jobs": filtered,
            "cost_usd": total_cost,
            "elapsed_s": round(total_elapsed, 3),
            "cached": False,
        }

    # -------- Google Search --------
    async def google(self, query: str, *, results: int = 10) -> dict:
        params = {"query": query, "results": str(results), "country": "us"}
        cache_key = self._cache.make_key("sd", "google", params)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}
        t0 = time.time()
        try:
            data = await self._get("/google", params, "google")
        except Exception as e:
            logger.warning("SD google failed for %r: %s", query, e)
            return {"results": [], "cached": False, "cost_usd": 0, "elapsed_s": 0, "error": str(e)[:200]}
        organic = (data or {}).get("organic_results") or data.get("organic_data") or []
        result = {
            "results": organic,
            "cost_usd": SD_GOOGLE_COST,
            "elapsed_s": round(time.time() - t0, 3),
            "cached": False,
        }
        if organic:
            self._cache.set(cache_key, result, CONFIG.SD_GOOGLE_TTL)
        return result
