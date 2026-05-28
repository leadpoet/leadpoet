"""Per-provider concurrency semaphores."""

from __future__ import annotations

import asyncio

from qual_engine.config import CONFIG


_semaphores: dict[str, asyncio.Semaphore] = {}


def get_semaphore(provider: str) -> asyncio.Semaphore:
    if provider not in _semaphores:
        limit = {
            "exa": CONFIG.EXA_CONCURRENCY,
            "scrapingdog": CONFIG.SCRAPINGDOG_CONCURRENCY,
            "openrouter": CONFIG.OPENROUTER_CONCURRENCY,
        }.get(provider, 10)
        _semaphores[provider] = asyncio.Semaphore(limit)
    return _semaphores[provider]
