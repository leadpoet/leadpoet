"""Gamma tool-market routes (docs/gamma-tool-market.md). Off by default.

While ``LEADPOET_TOOL_MARKET_MODE`` is ``off`` (the default) every route
answers 404 and nothing is built. The routes touch no Arena or lead-order
state.

* Signed routes: sr25519 hotkey signature, 60 s timestamp window, one use
  per signature (``lab_arena.lead_orders.auth``), as the lead-order routes.
* Tab routes: the token a signed ``POST /tool-market/tabs`` returned, as a
  bearer token, so an agent can call tools without signing each call.
* Public: the order book of each enabled tool.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Optional, Tuple

import anyio
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from lab_arena.tool_market import gateway_support as support

router = APIRouter(tags=["tool-market"])

MAX_BODY_BYTES = 64 * 1024

_lock = threading.Lock()
_state: Optional[Tuple[Any, Any]] = None
_builder: Optional[Callable[[], Tuple[Any, Any]]] = None
_verifier: Optional[Callable[[str, str, str], bool]] = None


def configure(*, builder: Optional[Callable[[], Tuple[Any, Any]]] = None,
              verifier: Optional[Callable[[str, str, str], bool]] = None) -> None:
    """Test and deployment hook: replace how state is built and signatures are checked."""

    global _state, _builder, _verifier
    with _lock:
        _state = None
        _builder = builder
        _verifier = verifier


def _get_state() -> Tuple[Any, Any]:
    global _state
    with _lock:
        if _state is not None:
            return _state
        built = _builder() if _builder is not None else support.build_state()
        if built[1] is not None:
            _state = built  # off builds nothing and caches nothing
        return built


def _error(status: int, code: str, **detail: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content=dict(detail, error=code))


class _Refusal(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status = status
        self.code = code


def _unique(pairs):
    document = dict(pairs)
    if len(document) != len(pairs):
        raise ValueError("duplicate member")
    return document


def _json(body: bytes) -> Any:
    try:
        # A repeated member is refused: stored bodies must read one way only.
        document = json.loads(body.decode("utf-8"), object_pairs_hook=_unique) if body else {}
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise _Refusal(400, "invalid_json") from None
    if not isinstance(document, dict):
        raise _Refusal(400, "invalid_json")
    return document


def _bearer(request: Request) -> str:
    value = request.headers.get("authorization") or ""
    return value[len("Bearer "):] if value.startswith("Bearer ") else ""


async def _handle(request: Request, work: Callable[[Any, Any, bytes], Any], *, signed: bool) -> Response:
    """Shared route body: mode gate, signature, thread offload, errors."""

    try:
        config, market = await anyio.to_thread.run_sync(_get_state)
    except support.ToolMarketConfigError:
        return _error(503, "tool_market_misconfigured")
    except Exception:
        return _error(503, "tool_market_unavailable")
    if not config.enabled or market is None:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    try:
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            raise _Refusal(413, "request_too_large")
        identity = None
        if signed:
            headers = request.headers
            identity = await anyio.to_thread.run_sync(lambda: support.verify_buyer_request(
                method=request.method, host=config.public_host, path=request.url.path, body=body,
                hotkey=headers.get("x-hotkey"), timestamp=headers.get("x-timestamp"),
                signature=headers.get("x-signature"), verifier=_verifier or support.verify_hotkey_signature,
                use_nonce=market.use_nonce,
            ))
        result = await anyio.to_thread.run_sync(lambda: work(market, identity, body))
    except _Refusal as exc:
        return _error(exc.status, exc.code)
    except support.ToolMarketError as exc:
        return _error(exc.status, exc.code, **exc.detail)
    except support.AuthError as exc:
        return _error(exc.status, exc.code)
    return JSONResponse(status_code=200, content=result)


# -- public ---------------------------------------------------------------------

@router.get("/tool-market/catalog")
async def tool_catalog(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.tool_catalog(), signed=False)


@router.get("/tool-market/books/{tool}")
async def order_book(tool: str, request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.order_book(tool), signed=False)


# -- tab token ------------------------------------------------------------------

@router.post("/tool-market/call")
async def call(request: Request) -> Response:
    token = _bearer(request)
    return await _handle(request, lambda m, who, body: m.call(token, _json(body)), signed=False)


@router.get("/tool-market/tab")
async def tab(request: Request) -> Response:
    token = _bearer(request)
    return await _handle(request, lambda m, who, body: m.tab(token), signed=False)


@router.post("/tool-market/tab/close")
async def close_tab(request: Request) -> Response:
    token = _bearer(request)
    return await _handle(request, lambda m, who, body: m.close_tab(token), signed=False)


# -- signed: buyers and miners --------------------------------------------------

@router.post("/tool-market/terms")
async def accept_terms(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.accept_terms(who, _json(body)), signed=True)


@router.get("/tool-market/balance")
async def balance(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.balance(who.hotkey), signed=True)


@router.post("/tool-market/tabs")
async def open_tab(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.open_tab(who, _json(body)), signed=True)


@router.post("/tool-market/keys")
async def register_key(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.register_key(who, _json(body), body), signed=True)


@router.post("/tool-market/keys/{key_id}/check")
async def check_key(key_id: str, request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.check_key(who, key_id), signed=True)


@router.post("/tool-market/keys/{key_id}/revoke")
async def revoke_key(key_id: str, request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.revoke_key(who, key_id), signed=True)


@router.post("/tool-market/asks")
async def post_ask(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.post_ask(who, _json(body)), signed=True)


@router.get("/tool-market/usage")
async def usage(request: Request, after: str = "0", limit: str = "100") -> Response:
    # The query is not signed; it only pages the signer's own calls.
    return await _handle(request, lambda m, who, body: m.usage(who, _digits(after), _digits(limit)), signed=True)


def _digits(text: str) -> Any:
    return int(text) if text.isascii() and text.isdigit() and len(text) <= 19 else None


@router.get("/tool-market/statement")
async def statement(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.statement(who.hotkey), signed=True)


__all__ = ["configure", "router"]
