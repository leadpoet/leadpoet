"""Gamma lead-order routes (docs/gamma-lead-orders.md). Off by default.

While ``LEADPOET_LEAD_ORDERS_MODE`` is ``off`` (the default) every route
answers 404 and nothing is built: no store, chain, KMS or provider client.
The routes touch no Arena state, reward or weight.

* Buyer routes (``/lead-orders``): sr25519 hotkey signature, 60 s timestamp
  window, one use per signature (``lab_arena.lead_orders.auth``).
* Runner routes (``/internal/lead-orders``): bearer runner secret compared in
  constant time.
* Broker (``/lead-orders/broker/{provider}``): the one-order token issued at
  claim is the bearer API key, so provider SDKs work unchanged. It is bound
  to one order and stops working when that order finishes or its deadline
  passes.
* Public routes: email confirmation, signed receipts, epoch payouts.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Optional, Tuple

import anyio
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from lab_arena.lead_orders import gateway_support as support

router = APIRouter(tags=["lead-orders"])

MAX_BODY_BYTES = 1_100_000

_lock = threading.Lock()
_state: Optional[Tuple[Any, Any, Any]] = None
_builder: Optional[Callable[[], Tuple[Any, Any, Any]]] = None
_verifier: Optional[Callable[[str, str, str], bool]] = None


def configure(*, builder: Optional[Callable[[], Tuple[Any, Any, Any]]] = None,
              verifier: Optional[Callable[[str, str, str], bool]] = None) -> None:
    """Test and deployment hook: replace how state is built and signatures are checked."""

    global _state, _builder, _verifier
    with _lock:
        _state = None
        _builder = builder
        _verifier = verifier


def _get_state() -> Tuple[Any, Any, Any]:
    global _state
    with _lock:
        if _state is not None:
            return _state
        built = _builder() if _builder is not None else support.build_state()
        if built[1] is not None:
            _state = built  # off builds nothing and caches nothing
        return built


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code})


def _not_found() -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "Not Found"})


async def _body(request: Request) -> bytes:
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise _Refusal(413, "request_too_large")
    return body


class _Refusal(Exception):
    def __init__(self, status: int, code: str) -> None:
        self.status = status
        self.code = code


def _json(body: bytes) -> Any:
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise _Refusal(400, "invalid_json") from None


async def _handle(request: Request, work: Callable[[Any, Any, Any, bytes], Any], *, auth: str) -> Response:
    """Shared route body: mode gate, authentication, thread offload, errors."""

    try:
        config, service, broker = await anyio.to_thread.run_sync(_get_state)
    except support.LeadOrdersConfigError:
        return _error(503, "lead_orders_misconfigured")
    except Exception:
        return _error(503, "lead_orders_unavailable")
    if not config.enabled or service is None:
        return _not_found()
    try:
        body = await _body(request)
        identity = None
        if auth == "buyer":
            verifier = _verifier or support.verify_hotkey_signature
            headers = request.headers
            identity = await anyio.to_thread.run_sync(lambda: support.verify_buyer_request(
                method=request.method, path=request.url.path, body=body,
                hotkey=headers.get("x-hotkey"), timestamp=headers.get("x-timestamp"),
                signature=headers.get("x-signature"), verifier=verifier,
                use_nonce=lambda h, s, t: service._store.use_nonce(hotkey=h, signature_hash=s, signed_at=t),
            ))
        elif auth == "runner":
            support.verify_runner(config, request.headers.get("authorization"))
        result = await anyio.to_thread.run_sync(lambda: work(service, broker, identity, body))
    except _Refusal as exc:
        return _error(exc.status, exc.code)
    except support.REQUEST_ERRORS as exc:
        return _error(exc.status, exc.code)
    if isinstance(result, Response):
        return result
    return JSONResponse(status_code=200, content=result)


def _bearer(request: Request) -> str:
    value = request.headers.get("authorization") or ""
    return value[len("Bearer "):] if value.startswith("Bearer ") else ""


# Fixed paths are declared before ``/lead-orders/{order_id}``.

# -- public routes --------------------------------------------------------------

@router.get("/lead-orders/buyers/confirm")
async def confirm_buyer(request: Request, token: str = "") -> Response:
    return await _handle(request, lambda s, b, who, body: s.confirm_buyer(token), auth="none")


@router.get("/lead-orders/receipts/{order_id}")
async def receipt(order_id: str, request: Request) -> Response:
    return await _handle(request, lambda s, b, who, body: s.public_receipt(order_id), auth="none")


@router.get("/lead-orders/epochs/{epoch}")
async def epoch_payout(epoch: str, request: Request) -> Response:
    return await _handle(request, lambda s, b, who, body: s.public_epoch(epoch), auth="none")


# -- broker (one-order token as the API key) ------------------------------------

@router.api_route("/lead-orders/broker/{provider}/{path:path}", methods=["GET", "POST"])
async def broker(provider: str, path: str, request: Request) -> Response:
    token = _bearer(request)

    def work(s, b, who, body):
        response = b.call(token, support.BrokerRequest(
            provider=provider, method=request.method, path=path,
            query=dict(request.query_params), body=body, content_type=request.headers.get("content-type"),
        ))
        return Response(content=bytes(response.body), status_code=int(response.status),
                        media_type=response.headers.get("content-type"))
    return await _handle(request, work, auth="none")


# -- buyer routes -------------------------------------------------------------

@router.post("/lead-orders/buyers")
async def register_buyer(request: Request) -> Response:
    return await _handle(request, lambda s, b, who, body: s.register_buyer(who.hotkey, _json(body)), auth="buyer")


@router.get("/lead-orders/balance")
async def balance(request: Request) -> Response:
    return await _handle(request, lambda s, b, who, body: s.balance(who.hotkey), auth="buyer")


@router.post("/lead-orders/keys")
async def register_key(request: Request) -> Response:
    return await _handle(request, lambda s, b, who, body: s.register_key(who.hotkey, _json(body)), auth="buyer")


@router.post("/lead-orders/keys/{key_id}/revoke")
async def revoke_key(key_id: str, request: Request) -> Response:
    return await _handle(request, lambda s, b, who, body: s.revoke_key(who.hotkey, key_id), auth="buyer")


@router.post("/lead-orders")
async def place_order(request: Request) -> Response:
    return await _handle(request, lambda s, b, who, body: s.place_order(who.hotkey, _json(body)), auth="buyer")


@router.get("/lead-orders/{order_id}")
async def order_view(order_id: str, request: Request) -> Response:
    return await _handle(request, lambda s, b, who, body: s.order_view(who.hotkey, order_id), auth="buyer")


# -- runner routes --------------------------------------------------------------

@router.post("/internal/lead-orders/claim")
async def runner_claim(request: Request) -> Response:
    def work(s, b, who, body):
        claimed = s.claim()
        return claimed if claimed is not None else Response(status_code=204)
    return await _handle(request, work, auth="runner")


@router.post("/internal/lead-orders/{order_id}/heartbeat")
async def runner_heartbeat(order_id: str, request: Request) -> Response:
    return await _handle(request, lambda s, b, who, body: s.heartbeat(order_id), auth="runner")


@router.post("/internal/lead-orders/{order_id}/finish")
async def runner_finish(order_id: str, request: Request) -> Response:
    return await _handle(request, lambda s, b, who, body: s.finish(order_id, _json(body)), auth="runner")


__all__ = ["configure", "router"]
