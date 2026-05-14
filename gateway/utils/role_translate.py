"""Validator-side role translation client.

Asks the gateway to translate a role string to English.  The gateway owns
DeepL + Supabase + cache; this module is just a thin HTTP client.  Never
raises — returns None on any failure so callers can fall through to
existing English-only role-match logic.

Open-source-safe: requires only GATEWAY_URL + an optional shared secret.
Has no DeepL key, no Supabase credentials, no schema knowledge.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import unicodedata
from collections import OrderedDict
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

GATEWAY_URL = (os.environ.get("GATEWAY_URL", "") or "").rstrip("/")
INTERNAL_SECRET = os.environ.get("LEADPOET_INTERNAL_SECRET", "")
_ENDPOINT = f"{GATEWAY_URL}/fulfillment/translate-role" if GATEWAY_URL else ""

# Local LRU as a micro-optimization: same role within a batch in the same
# container skips even the gateway HTTP call.  Tiny memory footprint.
_LRU_MAX = 1000
_lru: "OrderedDict[str, str]" = OrderedDict()
_lru_lock = asyncio.Lock()

# Normalization MUST match what gateway/api/role_translate.py:_normalize does,
# otherwise an accent/whitespace variant misses our per-process LRU even
# though the gateway resolves it to the same cached row.
_WS_RE = re.compile(r"\s+")


def _normalize(role: str) -> str:
    if not isinstance(role, str):
        return ""
    s = role.strip().lower()
    if not s:
        return ""
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    s = _WS_RE.sub(" ", s).strip()
    return s


async def _lru_get(key: str) -> Optional[str]:
    async with _lru_lock:
        if key in _lru:
            _lru.move_to_end(key)
            return _lru[key]
    return None


async def _lru_put(key: str, val: str) -> None:
    async with _lru_lock:
        _lru[key] = val
        _lru.move_to_end(key)
        while len(_lru) > _LRU_MAX:
            _lru.popitem(last=False)


_TIMEOUT_SECS = 8           # tight — gateway L1/L2 hits are sub-50ms
_RETRY_BACKOFF_SECS = 0.5   # single retry on transient failure


async def _post_once(
    session: aiohttp.ClientSession, headers: dict, role: str,
) -> "tuple[Optional[str], bool]":
    """One HTTP attempt.  Returns (value, is_definitive).

      * (translation, True)  -> gateway gave us a non-empty string
      * ("",          True)  -> gateway returned null (DeepL unavailable for this role)
      * (None,        False) -> transient failure (5xx, timeout, network)
      * ("",          True)  -> 401/403/422 (treat as definitive — retry won't help)
    """
    try:
        async with session.post(_ENDPOINT, json={"role": role}, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                val = data.get("translated_en") or ""
                return (val, True)
            if resp.status in (401, 403, 422):
                logger.debug("translate-role HTTP %d — definitive, no retry", resp.status)
                return ("", True)
            # 5xx, 408, 429, anything else → transient
            logger.debug("translate-role HTTP %d — transient", resp.status)
            return (None, False)
    except asyncio.TimeoutError:
        return (None, False)
    except aiohttp.ClientError:
        return (None, False)


async def translate_to_english(role: str, country: Optional[str] = None) -> Optional[str]:
    """Return the English equivalent of a role, or None on any failure.

    Calls the gateway endpoint POST /fulfillment/translate-role.  The
    gateway handles cache lookup + DeepL + persistence.  Caller falls
    through to existing logic on None.

    `country` is accepted for API compatibility but ignored (DeepL
    auto-detects language).
    """
    if not isinstance(role, str) or not role.strip():
        return None
    if not _ENDPOINT:
        return None  # GATEWAY_URL unset — disabled (dev only)

    # Local LRU short-circuit (per-process)
    cache_key = _normalize(role)
    if not cache_key:
        return None
    cached = await _lru_get(cache_key)
    if cached is not None:
        return cached or None

    headers = {"Content-Type": "application/json"}
    if INTERNAL_SECRET:
        headers["X-Internal-Auth"] = INTERNAL_SECRET

    timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECS)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            val, definitive = await _post_once(session, headers, role)
            if not definitive:
                await asyncio.sleep(_RETRY_BACKOFF_SECS)
                val, definitive = await _post_once(session, headers, role)
            if not definitive:
                # Both attempts transient — do NOT cache.  Caller falls
                # through to existing logic and the next call will retry.
                return None
            if not val:
                # Gateway definitively said null (e.g. DeepL down for this
                # role).  Cache "" so identical roles in the same batch
                # skip the round-trip.
                await _lru_put(cache_key, "")
                return None
            await _lru_put(cache_key, val)
            return val
    except Exception as e:
        logger.debug("translate-role error: %s", e)
        return None


def is_configured() -> bool:
    """True iff the gateway endpoint is configured.  Callers may use this
    to skip translation attempts entirely when not in production."""
    return bool(_ENDPOINT)
