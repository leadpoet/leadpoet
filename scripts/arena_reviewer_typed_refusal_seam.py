"""Apply only the authenticated Arena page-refusal transport correction.

The pinned grader remains the source of all scoring rules. Unknown or partly
corrected helper shapes fail closed; no current grader module is copied.
"""

from __future__ import annotations

import ast


CORRECTION_ID = "authenticated_page_refusal_transport_v1"


class TypedRefusalSeamError(ValueError):
    """Frozen helper bytes do not have the reviewed correction shape."""


_VERIFIER_CLASS_OLD = '''class _ArenaProviderFetchError(aiohttp.ClientError):
    """An Arena error response has no validated target response URL."""

    def __init__(self, message: str, *, broker_status: Optional[int] = None):
        super().__init__(message)
        self.broker_status = (
            broker_status
            if type(broker_status) is int and 100 <= broker_status <= 599
            else None
        )


'''
_VERIFIER_CLASS_NEW = '''class _ArenaProviderFetchError(aiohttp.ClientError):
    """An Arena error response has no validated target response URL."""

    def __init__(
        self, message: str, *, broker_status: Optional[int] = None,
        broker_error_code: str = "",
    ):
        super().__init__(message)
        self.broker_status = (
            broker_status
            if type(broker_status) is int and 100 <= broker_status <= 599
            else None
        )
        self.broker_error_code = (
            "provider_request_refused"
            if self.broker_status == 403
            and broker_error_code == "provider_request_refused"
            else ""
        )


_ARENA_PROVIDER_REQUEST_REFUSED_BODY = (
    b'{"error":{"code":"provider_request_refused"}}'
)


'''
_VERIFIER_FETCH_OLD = '''        if (
            resp.status >= 400
            and getattr(resp, "_lab_arena_response_url_validated", None) is False
        ):
            raise _ArenaProviderFetchError(
                "website fetch error: Arena provider response unavailable",
                broker_status=resp.status,
            )
'''
_VERIFIER_FETCH_NEW = '''        if (
            resp.status >= 400
            and getattr(resp, "_lab_arena_response_url_validated", None) is False
        ):
            broker_error_code = ""
            if (
                resp.status == 403
                and bool(getattr(resp, "headers", {}).get(
                    "x-leadpoet-call-identity"
                ))
            ):
                # The broker's fixed refusal has no target URL. Read at most
                # one byte beyond its exact envelope; never return this body
                # or the shim's request-URL fallback as page evidence.
                raw_error = bytearray()
                while len(raw_error) <= len(_ARENA_PROVIDER_REQUEST_REFUSED_BODY):
                    chunk = await resp.content.read(
                        len(_ARENA_PROVIDER_REQUEST_REFUSED_BODY) + 1
                        - len(raw_error)
                    )
                    if not chunk:
                        break
                    raw_error.extend(chunk)
                if bytes(raw_error) == _ARENA_PROVIDER_REQUEST_REFUSED_BODY:
                    broker_error_code = "provider_request_refused"
            raise _ArenaProviderFetchError(
                "website fetch error: Arena provider response unavailable",
                broker_status=resp.status,
                broker_error_code=broker_error_code,
            )
'''
_INVESTIGATOR_PREFIX_OLD = '''    safe_url = _safe_https_url(url)
    if not safe_url:
        return {"ok": False, "error": "invalid_url"}
    try:
'''
_INVESTIGATOR_PREFIX_NEW = '''    safe_url = _safe_https_url(url)
    if not safe_url:
        return {"ok": False, "error": "invalid_url"}
    api_key = ""
    try:
'''
_INVESTIGATOR_CATCH_OLD = '''        safe_final_url = public_http_url(safe_final_url)
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid_url"}
'''
_INVESTIGATOR_CATCH_NEW = '''        safe_final_url = public_http_url(safe_final_url)
    except _ArenaProviderFetchError as exc:
        if (
            api_key
            and exc.broker_status == 403
            and exc.broker_error_code == "provider_request_refused"
        ):
            return {"ok": False, "error": "provider_request_refused"}
        raise
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid_url"}
'''


def _convert(source: bytes, *, deltas: tuple) -> bytes:
    try:
        text = source.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise TypedRefusalSeamError("grading source must be valid UTF-8 Python") from exc
    if "\r" in text:
        raise TypedRefusalSeamError("grading source must use LF newlines")
    states = set()
    for name, old, new in deltas:
        module = ast.parse(text)
        found = [node for node in module.body
                 if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef))
                 and node.name == name]
        if len(found) != 1:
            raise TypedRefusalSeamError("correction helper is missing or ambiguous")
        lines = text.splitlines(keepends=True)
        start = sum(map(len, lines[:found[0].lineno - 1]))
        stop = sum(map(len, lines[:found[0].end_lineno]))
        counts = (text.count(old), text.count(new))
        if counts == (1, 0):
            states.add("original")
            if not start <= text.find(old) < stop:
                raise TypedRefusalSeamError("correction block moved outside helper")
            text = text.replace(old, new, 1)
        elif counts == (0, 1):
            states.add("corrected")
            if not start <= text.find(new) < stop:
                raise TypedRefusalSeamError("correction block moved outside helper")
        else:
            raise TypedRefusalSeamError("unrecognized or partial correction block")
    if len(states) != 1:
        raise TypedRefusalSeamError("partially corrected grading source")
    compile(text, "frozen_typed_refusal.py", "exec")
    return text.encode("utf-8")


def convert_verification_seam(source: bytes) -> bytes:
    return _convert(source, deltas=(
        ("_ArenaProviderFetchError", _VERIFIER_CLASS_OLD, _VERIFIER_CLASS_NEW),
        ("_fetch_bounded_html", _VERIFIER_FETCH_OLD, _VERIFIER_FETCH_NEW),
    ))


def convert_investigator_seam(source: bytes) -> bytes:
    return _convert(source, deltas=(
        ("_fetch_page", _INVESTIGATOR_PREFIX_OLD, _INVESTIGATOR_PREFIX_NEW),
        ("_fetch_page", _INVESTIGATOR_CATCH_OLD, _INVESTIGATOR_CATCH_NEW),
    ))
