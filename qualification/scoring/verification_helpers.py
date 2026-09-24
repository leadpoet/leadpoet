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

    _HIDDEN_ELEMENTS = frozenset({
        "aside", "footer", "nav", "noscript", "script", "style", "template",
    })
    _VOID_ELEMENTS = frozenset({
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr",
    })
    _NON_ARTICLE_PREFIXES = (
        "blog-index", "blog-posts-grid", "entry-related", "recommend-",
        "recommended", "related-", "related_",
    )

    def __init__(
        self,
        *,
        hidden_classes: frozenset[str] = frozenset(),
        hidden_ids: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_classes = hidden_classes
        self._hidden_ids = hidden_ids
        self._stack: list[tuple[str, bool]] = []
        self._hidden_depth = 0
        self._heading_depth: Optional[int] = None
        self._saw_heading = False
        self.parts: list[str] = []
        self.heading_parts: list[str] = []
        self.links: list[str] = []

    @staticmethod
    def _attribute_map(attrs: Any) -> Dict[str, str]:
        return {
            str(key or "").casefold(): str(value or "")
            for key, value in attrs
        }

    def _is_hidden(self, tag: str, attrs: Any) -> bool:
        values = self._attribute_map(attrs)
        classes = frozenset(values.get("class", "").casefold().split())
        element_id = values.get("id", "").casefold()
        style = re.sub(r"\s+", "", values.get("style", "").casefold())
        semantic_values = classes | ({element_id} if element_id else set())
        return bool(
            tag in self._HIDDEN_ELEMENTS
            or "hidden" in values
            or values.get("aria-hidden", "").strip().casefold() in {"true", "1"}
            or "display:none" in style
            or "visibility:hidden" in style
            or classes & self._hidden_classes
            or element_id in self._hidden_ids
            or any(
                value == "related"
                or any(
                    value.startswith(prefix)
                    for prefix in self._NON_ARTICLE_PREFIXES
                )
                for value in semantic_values
            )
        )

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        lowered = tag.casefold()
        if lowered in self._VOID_ELEMENTS:
            return
        hidden = self._is_hidden(lowered, attrs)
        self._stack.append((lowered, hidden))
        if hidden:
            self._hidden_depth += 1
        if lowered == "a" and not self._hidden_depth:
            href = self._attribute_map(attrs).get("href", "").strip()
            if href:
                self.links.append(href)
        if (
            lowered == "h1"
            and not self._hidden_depth
            and self._heading_depth is None
            and not self._saw_heading
        ):
            self._heading_depth = len(self._stack)
            self._saw_heading = True

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in self._VOID_ELEMENTS:
            return
        match = next(
            (
                index for index in range(len(self._stack) - 1, -1, -1)
                if self._stack[index][0] == lowered
            ),
            None,
        )
        if match is None:
            return
        popped = self._stack[match:]
        del self._stack[match:]
        self._hidden_depth -= sum(hidden for _tag, hidden in popped)
        self._hidden_depth = max(0, self._hidden_depth)
        if self._heading_depth is not None and len(self._stack) < self._heading_depth:
            self._heading_depth = None

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)
            if self._heading_depth is not None:
                self.heading_parts.append(data)


_CSS_HIDDEN_DECLARATION_RE = re.compile(
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden)",
    re.IGNORECASE,
)


def _css_hidden_selectors(content: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return simple class and ID selectors hidden by inline page CSS."""

    classes: set[str] = set()
    ids: set[str] = set()
    for style in re.findall(
        r"<style\b[^>]*>(.*?)</style\s*>", content, re.IGNORECASE | re.DOTALL
    ):
        depth = 0
        selector_start = 0
        declaration_start = 0
        for position, character in enumerate(style):
            if character == "{":
                if depth == 0:
                    selectors = style[selector_start:position]
                    declaration_start = position + 1
                depth += 1
                continue
            if character != "}" or depth == 0:
                continue
            depth -= 1
            if depth != 0:
                continue
            declarations = style[declaration_start:position]
            selector_start = position + 1
            # Conditional at-rules contain nested blocks. Their declarations
            # depend on a viewport or browser capability, so do not infer them.
            if "{" in declarations:
                continue
            if _CSS_HIDDEN_DECLARATION_RE.search(declarations) is None:
                continue
            for selector in selectors.split(","):
                class_match = re.fullmatch(
                    r"\s*\.([A-Za-z_][A-Za-z0-9_-]*)\s*", selector
                )
                if class_match is not None:
                    classes.add(class_match.group(1).casefold())
                    continue
                id_match = re.fullmatch(
                    r"\s*#([A-Za-z_][A-Za-z0-9_-]*)\s*", selector
                )
                if id_match is not None:
                    ids.add(id_match.group(1).casefold())
    return frozenset(classes), frozenset(ids)


def _visible_html_document(content: str) -> tuple[str, str]:
    """Return visible page text and its first visible H1, if present."""

    hidden_classes, hidden_ids = _css_hidden_selectors(content)
    parser = _VisibleHTMLTextParser(
        hidden_classes=hidden_classes,
        hidden_ids=hidden_ids,
    )
    try:
        parser.feed(content)
        parser.close()
    except Exception:
        return "", ""
    text = " ".join(" ".join(parser.parts).split())
    heading = " ".join(" ".join(parser.heading_parts).split())
    return text, heading


def _visible_html_text(content: str) -> str:
    return _visible_html_document(content)[0]


def visible_html_links(content: str) -> tuple[str, ...]:
    """Return link targets from visible HTML elements in document order."""

    if not content:
        return ()
    hidden_classes, hidden_ids = _css_hidden_selectors(content)
    parser = _VisibleHTMLTextParser(
        hidden_classes=hidden_classes,
        hidden_ids=hidden_ids,
    )
    try:
        parser.feed(content)
        parser.close()
    except Exception:
        return ()
    return tuple(dict.fromkeys(parser.links))


_HEADING_STOPWORDS = frozenset({
    "about", "after", "company", "from", "into", "news", "that", "the",
    "their", "this", "with",
})


def _extraction_matches_primary_heading(heading: str, body: str) -> bool:
    """Return whether extracted content still represents the page's main H1."""

    if heading and len(body) - len(heading) < 200:
        return False
    heading_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", heading.casefold())
        if len(token) >= 4 and token not in _HEADING_STOPWORDS
    }
    if len(heading_tokens) < 3:
        return True
    body_tokens = set(re.findall(r"[a-z0-9]+", body.casefold()))
    return len(heading_tokens & body_tokens) / len(heading_tokens) >= 0.5


def _normalized_visible_evidence(value: str) -> str:
    """Normalize extractor link markup and layout for visible-text binding."""

    without_link_targets = re.sub(
        r"\[([^\]]+)\]\([^\s)]+(?:\s+[^)]*)?\)",
        r"\1",
        str(value or ""),
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_link_targets.casefold()))


def _extraction_is_visible(visible_document: str, body: str) -> bool:
    """Bind every ordered extractor span to sanitized visible page text."""

    visible = _normalized_visible_evidence(visible_document)
    raw_segments = re.split(r"(?<=[.!?])\s+|[\r\n]+", str(body or ""))
    segments = [
        normalized
        for segment in raw_segments
        if (normalized := _normalized_visible_evidence(segment))
    ]
    if not visible or not segments:
        return False
    cursor = 0
    for segment in segments:
        position = visible.find(segment, cursor)
        if position < 0:
            return False
        cursor = position + len(segment)
    return True


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
    visible_document, primary_heading = _visible_html_document(content)
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
            if (
                _extraction_matches_primary_heading(primary_heading, body)
                and _extraction_is_visible(visible_document, body)
            ):
                return body
            if len(visible_document) >= min_body_chars:
                return visible_document
    return visible_document


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
