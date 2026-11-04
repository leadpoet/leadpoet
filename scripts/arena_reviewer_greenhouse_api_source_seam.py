"""Repair only exact Greenhouse API source transport in reviewed frozen code."""
from __future__ import annotations

import ast
import hashlib

from scripts.arena_reviewer_investigation_finality_seam import _named_bindings

CORRECTION_ID = "greenhouse_api_posting_source_v1"
PATCHES = {
    '_greenhouse_posting_identity': {
        'original_sha256': '96bad43f12add8e029c81ef17e4cfb3c2cd734a42972c74d989718a81eb31617',
        'repaired_sha256': '56bf0884a5ad870f694a32675595eb991c760282e720848da4aed8228b55ebcb',
        'anchor': '''def _greenhouse_posting_identity(source_url: str) -> tuple[str, str] | None:
    """Return one exact Greenhouse board/posting identity."""

    try:
        canonical = canonical_candidate_prompt_url(
            source_url,
            "intent_signal.url",
        )
        parsed = urlsplit(canonical)
    except (TypeError, ValueError):
        return None
    if (
        _GREENHOUSE_EXACT_POSTING_HOST_RE.fullmatch(
            (parsed.hostname or "").casefold()
        )
        is None
        or parsed.fragment
    ):
        return None
    match = _GREENHOUSE_EXACT_POSTING_PATH_RE.fullmatch(parsed.path)
    if match is None:
        return None
    posting = match.group("posting")
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if query and query != [("gh_jid", posting)]:
        return None
    return match.group("board"), posting
''',
        'replacement': '''def _greenhouse_posting_identity(source_url: str) -> tuple[str, str] | None:
    """Return one exact Greenhouse board/posting identity."""

    try:
        canonical = canonical_candidate_prompt_url(
            source_url,
            "intent_signal.url",
        )
        parsed = urlsplit(canonical)
    except (TypeError, ValueError):
        return None
    if parsed.fragment:
        return None
    api_source = (parsed.hostname or "").casefold() == "boards-api.greenhouse.io"
    if api_source:
        match = re.fullmatch(
            r"/v1/boards/(?P<board>[A-Za-z0-9_-]{1,100})/jobs/"
            r"(?P<posting>[0-9]{5,20})/?", parsed.path,
        )
    elif _GREENHOUSE_EXACT_POSTING_HOST_RE.fullmatch(
        (parsed.hostname or "").casefold()
    ):
        match = _GREENHOUSE_EXACT_POSTING_PATH_RE.fullmatch(parsed.path)
    else:
        return None
    if match is None:
        return None
    posting = match.group("posting")
    query = parse_qsl(parsed.query, keep_blank_values=True)
    allowed_query = [("content", "true")] if api_source else [("gh_jid", posting)]
    if query and query != allowed_query:
        return None
    return match.group("board"), posting
''',
    },
    '_ats_board_identity': {
        'original_sha256': 'df34a826a8e2e930fe5dbb8b5613e427224aba9a267aec9f2e69fd76232272da',
        'repaired_sha256': '7798ef4d6ecc38bbad7ced5db94d1e0ebdc9a7217bcf9cb1b84eca0786627fcd',
        'anchor': '''        if _GREENHOUSE_EXACT_POSTING_HOST_RE.fullmatch(host):
''',
        'replacement': '''        if host == "boards-api.greenhouse.io":
            provider = "greenhouse:"
        elif _GREENHOUSE_EXACT_POSTING_HOST_RE.fullmatch(host):
''',
    },
    '_scrape_exa': {
        'original_sha256': 'ca6b467d32adec52561a4b7fc4d5013c68906b2299e0de93beefd3a94fcc003e',
        'repaired_sha256': '8a28d2af799df646579a62835a5b666e93fa8a80347f2fabce75338913ba310b',
        'anchor': '''                            text = (result.get("text") or "")[:MAX_SCRAPED_CHARS]
''',
        'replacement': '''                            text = result.get("text")
                            if text is not None and not isinstance(text, str):
                                return {"ok": False, "stage": "exa_failed",
                                        "content": "", "error": "non_string_text"}
                            text = (text or "")[:MAX_SCRAPED_CHARS]
''',
    },
}


class GreenhouseApiSourceSeamError(ValueError):
    """The frozen source helpers differ from the reviewed versions."""


def convert_greenhouse_api_source_seam(source: bytes) -> bytes:
    try:
        text = source.decode("utf-8")
        ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise GreenhouseApiSourceSeamError("source must be UTF-8 Python") from exc
    if "\r" in text:
        raise GreenhouseApiSourceSeamError("source must use LF newlines")
    for name, patch in PATCHES.items():
        module = ast.parse(text)
        nodes = [node for node in module.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == name]
        if len(nodes) != 1 or _named_bindings(module, name) != nodes:
            raise GreenhouseApiSourceSeamError("source helper binding is missing or ambiguous")
        node = nodes[0]
        if node.decorator_list:
            raise GreenhouseApiSourceSeamError("source helper must not be decorated")
        lines = text.splitlines(keepends=True)
        body = "".join(lines[node.lineno - 1:node.end_lineno])
        digest = hashlib.sha256(body.encode()).hexdigest()
        if digest == patch["repaired_sha256"]:
            continue
        if digest != patch["original_sha256"] or body.count(patch["anchor"]) != 1:
            raise GreenhouseApiSourceSeamError("unrecognized source helper")
        repaired = body.replace(patch["anchor"], patch["replacement"], 1)
        if hashlib.sha256(repaired.encode()).hexdigest() != patch["repaired_sha256"]:
            raise GreenhouseApiSourceSeamError("repair differs from reviewed source")
        lines[node.lineno - 1:node.end_lineno] = [repaired]
        text = "".join(lines)
    return text.encode()
