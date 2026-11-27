"""Subnet 71 tool-market routes (docs/gamma-tool-market.md). Off by default.

While ``LEADPOET_TOOL_MARKET_MODE`` is ``off`` (the default) every route
answers 404 and nothing is built. The routes touch no Arena or lead-order
state.

* Signed routes: btauth/1, Bittensor's signed requests, with the Gamma
  service hotkey as receiver (``lab_arena.tool_market.btauth``), or the
  lead-order scheme (``lab_arena.lead_orders.auth``). The terms and key
  registrations take only the lead-order scheme: their signature is stored,
  and its message (host, timestamp) can be rebuilt from what is stored.
* ``POST /tool-market/call``: btauth/1, charged to the hotkey's Gamma
  balance (the Gamma standard), or a tab's token as a bearer token.
* Public, with ``Access-Control-Allow-Origin: *``: the services document,
  the OpenAPI spec, the catalog, each tool's book and any hotkey's balance.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Optional, Tuple

import anyio
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from lab_arena.tool_market import gateway_support as support

router = APIRouter(tags=["tool-market"])

MAX_BODY_BYTES = 64 * 1024
# A failed build (the chain or the vault down) is not retried for this long.
# While a build runs, other requests answer 503 at once instead of holding a
# shared thread behind it (one build can wait minutes on a hung chain).
BUILD_RETRY_SECONDS = 30.0
# Calls wait on vendors, so they get their own threads: slow calls cannot
# queue the catalog, signed routes or the rest of the gateway behind them.
CALL_THREADS = 32
_call_threads = anyio.CapacityLimiter(CALL_THREADS)
# Key checks wait on the vault and a vendor (up to 45 s), so they share a
# few threads: a miner re-posting a key cannot hold up other routes.
CHECK_THREADS = 4
_check_threads = anyio.CapacityLimiter(CHECK_THREADS)
# The public balance reads the chain, so it gets a few threads of its own.
BALANCE_THREADS = 4
_balance_threads = anyio.CapacityLimiter(BALANCE_THREADS)
PUBLIC = {"access-control-allow-origin": "*"}

_lock = threading.Lock()
_state: Optional[Tuple[Any, Any]] = None
_builder: Optional[Callable[[], Tuple[Any, Any]]] = None
_verifier: Optional[Callable[[str, str, str], bool]] = None
_failed: Optional[Tuple[float, Exception]] = None


def configure(*, builder: Optional[Callable[[], Tuple[Any, Any]]] = None,
              verifier: Optional[Callable[[str, str, str], bool]] = None) -> None:
    """Test and deployment hook: replace how state is built and signatures are checked."""

    global _state, _builder, _verifier, _failed
    with _lock:
        _state = _failed = None
        _builder = builder
        _verifier = verifier


def _get_state() -> Tuple[Any, Any]:
    global _state, _failed
    if _state is not None:
        return _state
    if not _lock.acquire(blocking=False):
        raise RuntimeError("tool market state is being built")
    try:
        if _state is not None:
            return _state
        if _failed is not None and time.monotonic() - _failed[0] < BUILD_RETRY_SECONDS:
            raise _failed[1].with_traceback(None)
        try:
            built = _builder() if _builder is not None else support.build_state()
        except Exception as error:
            _failed = (time.monotonic(), error)
            raise
        _failed = None
        if built[1] is not None:
            _state = built  # off builds nothing and caches nothing
        return built
    finally:
        _lock.release()


def _error(status: int, code: str, **detail: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content=dict(detail, error=code))


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
        raise support.ToolMarketError(400, "invalid_json") from None
    if not isinstance(document, dict):
        raise support.ToolMarketError(400, "invalid_json")
    return document


def _bearer(request: Request) -> str:
    value = request.headers.get("authorization") or ""
    return value[len("Bearer "):] if value.startswith("Bearer ") else ""


def _target(request: Request) -> str:
    """The request target as sent (percent-encoded), as btauth/1 signs it."""

    target = request.scope.get("raw_path", request.url.path.encode()).decode("latin-1")
    query = request.scope.get("query_string", b"").decode("latin-1")
    return target + ("?" + query if query else "")


def _identity(request: Request, config: Any, market: Any, body: bytes, *, proof: bool) -> Any:
    headers = request.headers
    if support.btauth.VERSION_HEADER in headers and not proof:
        if "authorization" in headers:
            raise support.ToolMarketError(400, "btauth_ambiguous_credentials")
        return support.btauth.verify(
            headers, body, method=request.method, target=_target(request), receiver=config.gamma_service,
            verifier=_verifier or support.verify_hotkey_signature, use_nonce=market.use_nonce)
    return support.verify_buyer_request(
        method=request.method, host=config.public_host, path=request.url.path, body=body,
        hotkey=headers.get("x-hotkey"), timestamp=headers.get("x-timestamp"),
        signature=headers.get("x-signature"), verifier=_verifier or support.verify_hotkey_signature,
        use_nonce=market.use_nonce)


async def _handle(request: Request, work: Callable[[Any, Any, bytes], Any], *, signed: bool,
                  limiter: Optional[anyio.CapacityLimiter] = None, proof: bool = False,
                  public: bool = False) -> Response:
    """Shared route body: mode gate, signature, thread offload, errors.

    ``signed``: btauth/1 or the lead-order scheme (``proof``: that one only).
    ``public``: readable from any web page.
    """

    try:
        config, market = await anyio.to_thread.run_sync(_get_state)
    except support.ToolMarketConfigError:
        return _error(503, "tool_market_misconfigured")
    except Exception:
        return _error(503, "tool_market_unavailable")
    if not config.enabled or market is None:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    try:
        buf = bytearray()
        async for chunk in request.stream():
            if len(buf) + len(chunk) > MAX_BODY_BYTES:
                raise support.ToolMarketError(413, "request_too_large")
            buf.extend(chunk)
        body = bytes(buf)
        identity = None
        if signed:
            identity = await anyio.to_thread.run_sync(
                lambda: _identity(request, config, market, body, proof=proof))
        result = await anyio.to_thread.run_sync(lambda: work(market, identity, body), limiter=limiter)
    except support.ToolMarketError as exc:
        response = _error(exc.status, exc.code, **exc.detail)
    except support.AuthError as exc:
        response = _error(exc.status, exc.code)
        if exc.code.startswith("btauth_"):
            response.headers["www-authenticate"] = "btauth/1"
    except support.ArenaStoreError:
        response = _error(503, "store_unavailable")
    else:
        response = JSONResponse(status_code=200, content=result)
    if public:
        response.headers.update(PUBLIC)
    return response


# -- public ---------------------------------------------------------------------

@router.get("/tool-market/services.json")
async def services(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.services(), signed=False,
                         public=True)


@router.get("/tool-market/openapi.json")
async def openapi(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.openapi(), signed=False, public=True)


@router.get("/tool-market/catalog")
async def tool_catalog(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.tool_catalog(), signed=False, public=True)


@router.get("/tool-market/books/{tool}")
async def order_book(tool: str, request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.order_book(tool), signed=False, public=True)


@router.get("/tool-market/gamma/balance")
async def gamma_balance(request: Request, hotkey: str = "") -> Response:
    return await _handle(request, lambda m, who, body: m.gamma_balance(hotkey), signed=False, public=True,
                         limiter=_balance_threads)


# -- calls: btauth/1 or a tab token ----------------------------------------------

@router.post("/tool-market/call")
async def call(request: Request) -> Response:
    signed = support.btauth.VERSION_HEADER in request.headers
    token = "" if signed else _bearer(request)
    return await _handle(request, lambda m, who, body: m.call(token, _json(body), who), signed=signed,
                         limiter=_call_threads)


@router.post("/tool-market/api/v2/integrations/zerobounce_validate/execute")
async def lead_search_validation(request: Request) -> Response:
    token = _bearer(request)
    return await _handle(request, lambda market, who, body: market.lead_search_validation(
        token, _json(body)), signed=False, limiter=_call_threads)


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
    return await _handle(request, lambda m, who, body: m.accept_terms(who, _json(body)), signed=True,
                         proof=True)


@router.post("/tool-market/tabs")
async def open_tab(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.open_tab(who, _json(body)), signed=True)


@router.post("/tool-market/keys")
async def register_key(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.register_key(who, _json(body), body), signed=True,
                         limiter=_check_threads, proof=True)


@router.post("/tool-market/keys/{key_id}/check")
async def check_key(key_id: str, request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.check_key(who, key_id), signed=True,
                         limiter=_check_threads)


@router.post("/tool-market/keys/{key_id}/revoke")
async def revoke_key(key_id: str, request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.revoke_key(who, key_id), signed=True)


@router.post("/tool-market/asks")
async def post_ask(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.post_ask(who, _json(body)), signed=True)


@router.get("/tool-market/usage")
async def usage(request: Request, after: str = "0", limit: str = "100") -> Response:
    # Under the lead-order scheme the query is not signed; it only pages the signer's own calls.
    return await _handle(request, lambda m, who, body: m.usage(who, _digits(after), _digits(limit)), signed=True)


def _digits(text: str) -> Any:
    return int(text) if text.isascii() and text.isdigit() and len(text) <= 19 else None


@router.get("/tool-market/statement")
async def statement(request: Request) -> Response:
    return await _handle(request, lambda m, who, body: m.statement(who.hotkey), signed=True)


__all__ = ["configure", "router"]
