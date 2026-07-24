"""Role translation endpoint.

Validators ask the gateway to translate a role string (e.g. an Apify-
returned LinkedIn role in Portuguese) to its English equivalent.  The
gateway owns the DeepL key + Supabase access + cache state.  Validators
just make one HTTP call.

Architecture:
  Validator  --POST /fulfillment/translate-role-->  Gateway
                                                       │
                                                       │  1) L1 in-process LRU
                                                       │  2) L2 Supabase role_translations
                                                       │  3) DeepL Pro API (only on miss)
                                                       │
                                                       ▼
                                              {"translated_en": "<English>"}

Why gateway-only access to DeepL + Supabase:
  - Validators are open-source; any creds in their env leak to operators.
  - Single writer = no cache poisoning by rogue validators.
  - One place to rotate DeepL key, monitor quota, audit translations.

Failure modes return {"translated_en": null} — callers fall through to
existing LLM-based role comparison.
"""

import asyncio
import logging
import os
import re
import unicodedata
from collections import OrderedDict
from typing import Optional, Tuple

import aiohttp
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from gateway.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from gateway.db.client import create_http1_sync_client
from supabase import Client

logger = logging.getLogger(__name__)

DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY", "")
_DEEPL_BASE = (
    "https://api-free.deepl.com" if DEEPL_API_KEY.endswith(":fx")
    else "https://api.deepl.com"
)
_DEEPL_TRANSLATE_URL = f"{_DEEPL_BASE}/v2/translate"

INTERNAL_SECRET = os.environ.get("LEADPOET_INTERNAL_SECRET", "")
ALLOW_NO_AUTH = os.environ.get("LEADPOET_ALLOW_NO_AUTH", "").lower() in ("1", "true", "yes")
if not INTERNAL_SECRET and not ALLOW_NO_AUTH:
    logger.critical(
        "translate-role: LEADPOET_INTERNAL_SECRET is unset.  Endpoint will "
        "fail closed (503) until set, or set LEADPOET_ALLOW_NO_AUTH=true for dev."
    )

# Shared across threadpool workers — must stay HTTP/1-pinned; the default
# HTTP/2 HPACK encoder is not thread-safe.
_sb: Client = create_http1_sync_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

router = APIRouter(prefix="/fulfillment", tags=["Role Translation"])


# ---------------------------------------------------------------------------
# In-process LRU (shared across all incoming requests to this gateway)
# ---------------------------------------------------------------------------
_LRU_MAX = 5000
_lru: "OrderedDict[str, str]" = OrderedDict()
_lru_lock = asyncio.Lock()


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


# ---------------------------------------------------------------------------
# Normalization (must match what the validator client computes)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Supabase cache helpers
# ---------------------------------------------------------------------------

def _sb_lookup(key: str) -> Optional[str]:
    try:
        result = _sb.table("role_translations") \
            .select("translated_en") \
            .eq("role_normalized", key) \
            .eq("verified", True) \
            .limit(1) \
            .execute()
        rows = result.data or []
        return rows[0]["translated_en"] if rows else None
    except Exception as e:
        logger.warning("supabase role_translations lookup failed for %r: %s", key, e)
        return None


def _sb_upsert(key: str, translated: str, src_lang: Optional[str]) -> None:
    try:
        _sb.table("role_translations").upsert({
            "role_normalized": key,
            "translated_en":   translated,
            "src_lang":        src_lang or None,
            "verified":        True,
        }, on_conflict="role_normalized").execute()
    except Exception as e:
        logger.warning("supabase role_translations upsert failed for %r: %s", key, e)


# ---------------------------------------------------------------------------
# DeepL call
# ---------------------------------------------------------------------------

async def _deepl_translate(role: str) -> Optional[Tuple[str, str]]:
    if not DEEPL_API_KEY:
        return None
    payload = {"text": role, "target_lang": "EN-US"}
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(_DEEPL_TRANSLATE_URL, data=payload, headers=headers) as resp:
                if resp.status != 200:
                    body_text = await resp.text()
                    logger.warning("DeepL HTTP %d for role=%r: %s", resp.status, role[:60], body_text[:200])
                    return None
                data = await resp.json()
                translations = data.get("translations") or []
                if not translations:
                    return None
                t = translations[0]
                return (t.get("text", "").strip(), t.get("detected_source_language", ""))
    except asyncio.TimeoutError:
        logger.warning("DeepL timeout for role=%r", role[:60])
    except Exception as e:
        logger.warning("DeepL error for role=%r: %s", role[:60], e)
    return None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

class TranslateRoleRequest(BaseModel):
    role: str = Field(..., min_length=1, max_length=500)


class TranslateRoleResponse(BaseModel):
    translated_en: Optional[str] = None


def _check_auth(x_internal_auth: Optional[str]) -> None:
    """Shared-secret check.

    Three states:
      * INTERNAL_SECRET set                            -> require matching header
      * INTERNAL_SECRET unset + ALLOW_NO_AUTH=true     -> no auth (dev only)
      * INTERNAL_SECRET unset + ALLOW_NO_AUTH not set  -> fail closed (503)
    """
    if not INTERNAL_SECRET:
        if ALLOW_NO_AUTH:
            return
        raise HTTPException(
            status_code=503,
            detail="translate-role disabled: LEADPOET_INTERNAL_SECRET not configured",
        )
    if x_internal_auth != INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="invalid internal auth")


@router.post("/translate-role", response_model=TranslateRoleResponse)
async def translate_role(
    body: TranslateRoleRequest,
    x_internal_auth: Optional[str] = Header(None, alias="X-Internal-Auth"),
):
    """Translate a role string to English.  Caches every translation.

    Returns translated_en=null when DeepL is unavailable or returns nothing
    — callers should fall through to existing role-comparison logic.
    """
    _check_auth(x_internal_auth)

    key = _normalize(body.role)
    if not key:
        return TranslateRoleResponse(translated_en=None)

    # L1
    cached = await _lru_get(key)
    if cached is not None:
        return TranslateRoleResponse(translated_en=cached)

    # L2 — sync supabase-py wrapped in to_thread so we don't stall the loop
    cached = await asyncio.to_thread(_sb_lookup, key)
    if cached is not None:
        await _lru_put(key, cached)
        return TranslateRoleResponse(translated_en=cached)

    # L3 — DeepL
    result = await _deepl_translate(body.role)
    if not result:
        return TranslateRoleResponse(translated_en=None)
    translated, src_lang = result
    if not translated:
        return TranslateRoleResponse(translated_en=None)

    # Write back to both layers
    await _lru_put(key, translated)
    await asyncio.to_thread(_sb_upsert, key, translated, src_lang)
    return TranslateRoleResponse(translated_en=translated)
