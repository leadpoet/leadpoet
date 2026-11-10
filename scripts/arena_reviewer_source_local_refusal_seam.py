"""Exact frozen810 source-local refusal readers; no grader or evidence changes."""
from __future__ import annotations
import hashlib

CORRECTION_ID = "source_local_refusal_finality_v1"
ORIGINAL_SHA256 = {'qualification/scoring/company_verification.py': '06155e2987a21b4327885f1b7b49790144e8c3eb2b14da8db4d0db4d2bf45e7c', 'lab_arena/shim.py': '27a667f715effdb06852cd3fd84b8ca382ddad4051b8e0db145dd29fd84bc610'}
REPAIRED_SHA256 = {'qualification/scoring/company_verification.py': '9b9488b471727e03647a2bd11e637d6033c739d6fc3bb6c39c768beb04d7a042', 'lab_arena/shim.py': '5af4676353871beb73673934c4a9f31f1e0d8e8780632c47f2fb463478ccd6cb'}

_ANCHORS = {
    'qualification/scoring/company_verification.py': (
        (r'''_ARENA_PROVIDER_REQUEST_REFUSED_BODY = (
    b'{"error":{"code":"provider_request_refused"}}'
)


async def _fetch_bounded_html(
''', r'''_ARENA_PROVIDER_REQUEST_REFUSED_BODY = (
    b'{"error":{"code":"provider_request_refused"}}'
)
_ARENA_SOURCE_UNAVAILABLE_HEADER = "x-leadpoet-source-unavailable"
_ARENA_SOURCE_UNAVAILABLE_VALUE = "verified-source-v1"
_ARENA_SOURCE_UNAVAILABLE_REASON = "website source unavailable: verified provider refusal"


async def _fetch_bounded_html(
'''),
        (r'''            and getattr(resp, "_lab_arena_response_url_validated", None) is False
        ):
            broker_error_code = ""
            if (
                resp.status == 403
                and bool(getattr(resp, "headers", {}).get(
''', r'''            and getattr(resp, "_lab_arena_response_url_validated", None) is False
        ):
            broker_error_code = ""
            source_unavailable = False
            if (
                resp.status == 403
                and bool(getattr(resp, "headers", {}).get(
'''),
        (r'''                    raw_error.extend(chunk)
                if bytes(raw_error) == _ARENA_PROVIDER_REQUEST_REFUSED_BODY:
                    broker_error_code = "provider_request_refused"
            raise _ArenaProviderFetchError(
                "website fetch error: Arena provider response unavailable",
                broker_status=resp.status,
                broker_error_code=broker_error_code,
            )
''', r'''                    raw_error.extend(chunk)
                if bytes(raw_error) == _ARENA_PROVIDER_REQUEST_REFUSED_BODY:
                    broker_error_code = "provider_request_refused"
                    headers = getattr(resp, "headers", {})
                    proof = [value for name, value in headers.items()
                             if name.lower() == _ARENA_SOURCE_UNAVAILABLE_HEADER]
                    identities = [value for name, value in headers.items()
                                  if name.lower() == "x-leadpoet-call-identity"]
                    source_unavailable = bool(
                        proof == [_ARENA_SOURCE_UNAVAILABLE_VALUE]
                        and len(identities) == 1
                        and isinstance(identities[0], str)
                        and re.fullmatch(r"sha256:[0-9a-f]{64}", identities[0])
                    )
            raise _ArenaProviderFetchError(
                (_ARENA_SOURCE_UNAVAILABLE_REASON if source_unavailable
                 else "website fetch error: Arena provider response unavailable"),
                broker_status=resp.status,
                broker_error_code=broker_error_code,
            )
'''),
    ),
    'lab_arena/shim.py': (
        (r'''SOCKET_GRACE_SECONDS = 15.0
SETTLED_MICROUSD_HEADER = "x-leadpoet-settled-microusd"
CALL_IDENTITY_HEADER = "x-leadpoet-call-identity"
ERROR_PREFIX = "lab arena: "
SITECUSTOMIZE_SOURCE = "import lab_arena.shim as _lab_arena_shim\n_lab_arena_shim.install()\n"
SHIM_IMAGE_MODULES = (
''', r'''SOCKET_GRACE_SECONDS = 15.0
SETTLED_MICROUSD_HEADER = "x-leadpoet-settled-microusd"
CALL_IDENTITY_HEADER = "x-leadpoet-call-identity"
SOURCE_UNAVAILABLE_HEADER = "x-leadpoet-source-unavailable"
SOURCE_UNAVAILABLE_VALUE = "verified-source-v1"
ERROR_PREFIX = "lab arena: "
SITECUSTOMIZE_SOURCE = "import lab_arena.shim as _lab_arena_shim\n_lab_arena_shim.install()\n"
SHIM_IMAGE_MODULES = (
'''),
        (r'''        raise ShimTransportError("invalid_response")
    headers: Dict[str, str] = {}
    trusted_names = []
    for name, value in raw_headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ShimTransportError("invalid_response")
''', r'''        raise ShimTransportError("invalid_response")
    headers: Dict[str, str] = {}
    trusted_names = []
    source_marker_seen = False
    for name, value in raw_headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ShimTransportError("invalid_response")
'''),
        (r'''            if not valid:
                raise ShimTransportError("invalid_response")
            headers[normalized_name] = value
    if SETTLED_MICROUSD_HEADER in headers and CALL_IDENTITY_HEADER not in headers:
        raise ShimTransportError("invalid_response")
    if len(trusted_names) > 1:
''', r'''            if not valid:
                raise ShimTransportError("invalid_response")
            headers[normalized_name] = value
        elif normalized_name == SOURCE_UNAVAILABLE_HEADER:
            if source_marker_seen:
                raise ShimTransportError("invalid_response")
            source_marker_seen = True
            if value == SOURCE_UNAVAILABLE_VALUE:
                headers[normalized_name] = value
    if SETTLED_MICROUSD_HEADER in headers and CALL_IDENTITY_HEADER not in headers:
        raise ShimTransportError("invalid_response")
    if len(trusted_names) > 1:
'''),
        (r'''        body = base64.b64decode(str(payload["body_b64"]), validate=True)
    except (TypeError, ValueError) as exc:
        raise ShimTransportError("invalid_response") from exc
    response_url = ""
    if trusted_names:
        try:
''', r'''        body = base64.b64decode(str(payload["body_b64"]), validate=True)
    except (TypeError, ValueError) as exc:
        raise ShimTransportError("invalid_response") from exc
    if SOURCE_UNAVAILABLE_HEADER in headers and not (
        status == 403 and CALL_IDENTITY_HEADER in headers
        and body == b'{"error":{"code":"provider_request_refused"}}'
    ):
        headers.pop(SOURCE_UNAVAILABLE_HEADER)
    response_url = ""
    if trusted_names:
        try:
'''),
    ),
}


class SourceLocalRefusalSeamError(ValueError):
    """The source is not the exact reviewed frozen generation."""


def convert_source_local_refusal_seam(name: str, source: bytes) -> bytes:
    if name not in ORIGINAL_SHA256:
        raise SourceLocalRefusalSeamError("unexpected source-local refusal file")
    digest = hashlib.sha256(source).hexdigest()
    if digest == REPAIRED_SHA256[name]:
        return source
    if digest != ORIGINAL_SHA256[name]:
        raise SourceLocalRefusalSeamError("unrecognized or partial source-local refusal source")
    text = source.decode("utf-8")
    for before, after in _ANCHORS[name]:
        if text.count(before) != 1:
            raise SourceLocalRefusalSeamError("source-local refusal source anchor differs")
        text = text.replace(before, after, 1)
    result = text.encode("utf-8")
    compile(result, name, "exec")
    if hashlib.sha256(result).hexdigest() != REPAIRED_SHA256[name]:
        raise SourceLocalRefusalSeamError("source-local refusal correction differs")
    return result
