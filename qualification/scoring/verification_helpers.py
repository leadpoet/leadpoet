"""Shared helpers used by the active Arena qualification scorer."""

import asyncio
import logging
import os
import re
from datetime import date
from html.parser import HTMLParser
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("QUALIFICATION_OPENROUTER_API_KEY", "")
LLM_TIMEOUT = 30.0

try:
    import trafilatura as _trafilatura
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    _TRAFILATURA_AVAILABLE = False


class _VisibleHTMLTextParser(HTMLParser):
    """Collect page-visible text while excluding executable or fallback markup."""

    _HIDDEN_ELEMENTS = frozenset({"script", "style", "template", "noscript"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self.parts = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag.lower() in self._HIDDEN_ELEMENTS:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._HIDDEN_ELEMENTS and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


def _visible_html_text(content: str) -> str:
    parser = _VisibleHTMLTextParser()
    try:
        parser.feed(content)
        parser.close()
    except Exception:
        return ""
    return " ".join(" ".join(parser.parts).split())


def extract_article_body(content: str, *, min_body_chars: int = 200) -> str:
    """Extract article body from raw HTML without admitting hidden page text.

    Returns the cleanest content available:
      - If trafilatura is installed AND input looks like HTML AND extraction
        succeeds with ≥ ``min_body_chars`` chars, returns the extracted body.
      - Otherwise HTML becomes visible plain text through the stdlib parser.
      - Non-HTML returns unchanged.

    Safe to call on any content — markdown, plain text, or HTML. The cheap HTML
    pre-check leaves non-HTML inputs unchanged.
    """
    if not content:
        return content
    # Cheap pre-check: only attempt extraction if the content looks like HTML.
    # trafilatura accepts non-HTML but spends real time parsing — skip it for
    # markdown/text inputs.
    if "<html" not in content[:2000].lower() and "<body" not in content[:2000].lower() and "<div" not in content[:5000].lower():
        return content
    if _TRAFILATURA_AVAILABLE:
        try:
            body = _trafilatura.extract(
                content,
                include_comments=False,
                include_links=True,
                include_tables=True,
                favor_recall=True,
                no_fallback=False,
            )
        except Exception:
            body = None
        if body and len(body) >= min_body_chars:
            return body
    return _visible_html_text(content)


GENERIC_INTENT_PATTERNS = [
    # Exact patterns from the cipher model's fallback
    r"is\s+actively\s+operating\s+in\s+\w+",
    r"visible\s+market\s+activity",
    r"market\s+activity\s+and\s+company\s+updates",
    r"business\s+operations\s+and\s+updates",
    # Generic patterns that apply to ANY company
    r"^.{0,50}\s+is\s+(?:actively\s+)?(?:operating|expanding|growing)",
    r"company\s+(?:updates|activities|operations)",
    r"market\s+(?:activity|presence|operations)",
]


SPECIFIC_INTENT_KEYWORDS = [
    "hiring", "recruit", "job", "position", "opening",  # Hiring intent
    "launch", "released", "announced", "introduced",    # Product launch
    "raised", "funding", "series", "investment",        # Funding
    "partnership", "partnered", "collaboration",        # Partnership
    "acquired", "acquisition", "merger",                # M&A
    "expansion", "opened", "new office", "new location", # Geographic expansion
    "migrating", "adopting", "implementing",            # Technology adoption
]


def is_generic_intent_description(description: str) -> Tuple[bool, str]:
    """
    Check if an intent description is generic/templated (gaming attempt).
    
    This runs BEFORE the LLM call to save costs on obvious fallbacks.
    
    Args:
        description: The intent signal description
        
    Returns:
        Tuple of (is_generic: bool, reason: str)
    """
    desc_lower = description.lower().strip()
    
    # Check for known generic patterns
    for pattern in GENERIC_INTENT_PATTERNS:
        if re.search(pattern, desc_lower, re.IGNORECASE):
            return True, f"Generic pattern detected: matches '{pattern[:30]}...'"
    
    # Check if description has ANY specific intent keywords
    has_specific_keyword = False
    for keyword in SPECIFIC_INTENT_KEYWORDS:
        if keyword in desc_lower:
            has_specific_keyword = True
            break
    
    # Very short descriptions with no specific keywords are likely generic
    if len(desc_lower) < 80 and not has_specific_keyword:
        return True, "Description too short and lacks specific intent keywords"
    
    # Check for templated structure: "{company} is {verb}ing" with no specifics
    templated_pattern = r"^\w+(?:\s+\w+){0,3}\s+is\s+\w+ing\s+(?:in\s+)?\w+\s*\.?$"
    if re.match(templated_pattern, desc_lower) and not has_specific_keyword:
        return True, "Templated structure with no specific details"
    
    return False, "Description appears specific"


def check_future_date(signal_date: Optional[str]) -> Optional[str]:
    """
    Reject dates set in the future — obviously fabricated.
    
    Returns an error message if the date is in the future, None if OK.
    """
    if not signal_date:
        return None
    try:
        parsed = date.fromisoformat(signal_date)
    except (ValueError, TypeError):
        return None
    from qualification.scoring.evaluation_clock import evaluation_date

    if parsed > evaluation_date():
        return f"Signal date {signal_date} is in the future — fabricated"
    return None


async def openrouter_chat(
    prompt: str,
    model: str,
    max_retries: int = 2,
    api_key: str = "",
    *,
    system_prompt: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
    max_tokens: int = 200,
) -> str:
    """Call OpenRouter LLM API with automatic retry on transient failures.

    Retries on 5xx, 429 (rate limit), and network errors.

    Keyword arguments:

      * ``system_prompt`` — when provided, the request is structured as a
        two-message chat ([system, user]) instead of a single user message.
        Anything in the user message — including untrusted miner-supplied
        text — is then evaluated against an authoritative system prompt
        that the model is much harder to override via prompt injection.
        Callers that interpolate miner text into ``prompt`` SHOULD always
        set ``system_prompt`` so the scoring/verification rules live
        outside the user content.

      * ``response_format`` — passes through to OpenRouter as the OpenAI
        ``response_format`` field. Pass a JSON schema or
        ``{"type": "json_object"}`` to force the model to emit only
        schema-conformant JSON. Critical
        defense layer: even if injection succeeds in steering reasoning,
        the output channel is locked.

      * ``max_tokens`` — bounds the response size. Default 200.
    """
    key = api_key or OPENROUTER_API_KEY
    if not key:
        raise ValueError("No OpenRouter API key configured (neither api_key param nor QUALIFICATION_OPENROUTER_API_KEY env var)")

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    last_error = None
    for attempt in range(1 + max_retries):
        try:
            async with httpx.AsyncClient() as client:
                payload: Dict[str, Any] = {
                    "model": model if "/" in model else f"openai/{model}",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                    "provider": {
                        "data_collection": "deny",
                        "zdr": True,
                    },
                }
                if response_format is not None:
                    payload["response_format"] = response_format

                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://leadpoet.ai",
                        "X-Title": "Leadpoet Qualification"
                    },
                    json=payload,
                    timeout=LLM_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            is_retryable = isinstance(e, (httpx.TimeoutException, httpx.ConnectError))
            if isinstance(e, httpx.HTTPStatusError):
                is_retryable = e.response.status_code in (429, 500, 502, 503, 504)
                # If response_format is rejected (some models or proxies
                # don't yet support strict json_schema), retry once with
                # the looser json_object form so the call doesn't hard-fail.
                # Detected via 400 with relevant text.
                if (
                    response_format
                    and response_format.get("type") == "json_schema"
                    and e.response.status_code == 400
                    and attempt < max_retries
                ):
                    body = e.response.text or ""
                    if "response_format" in body or "json_schema" in body:
                        logger.warning(
                            "openrouter_chat: strict json_schema rejected, "
                            "falling back to json_object for this attempt"
                        )
                        response_format = {"type": "json_object"}
                        continue

            if is_retryable and attempt < max_retries:
                wait = 1.5 * (attempt + 1)
                logger.warning(f"OpenRouter call failed (attempt {attempt+1}), retrying in {wait}s: {e}")
                await asyncio.sleep(wait)
                continue
            raise
    raise last_error
