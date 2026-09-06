"""Thin public proxy to the standalone Lab Arena sidecar.

The gateway owns no Arena state or competition logic. It only exposes the
sidecar on the existing public gateway address. Arena runners can use the
sidecar loopback/private address directly.
"""

from __future__ import annotations

import os
from typing import Mapping

import httpx
from fastapi import APIRouter, HTTPException, Request, Response


router = APIRouter(prefix="/arena", tags=["agent-competition"])
testnet_router = APIRouter(prefix="/testnet/arena", tags=["agent-competition-testnet"])

_MAX_REQUEST_BYTES = 1_100_000
# A completion can contain the judge's accepted 2 MiB scoring output plus the
# small signed-request wrapper. The sidecar performs the exact schema checks.
_MAX_COMPLETION_REQUEST_BYTES = (2 * 1_048_576) + 65_536
_SIDECAR_URL = "http://127.0.0.1:8792"
_TESTNET_SIDECAR_URL = "http://127.0.0.1:8793"
_FORWARDED_REQUEST_HEADERS = ("content-type", "x-lab-arena-lease")
_FORWARDED_RESPONSE_HEADERS = (
    "content-type",
    "cache-control",
    "x-content-type-options",
)


def _arena_enabled() -> bool:
    return os.environ.get("LAB_ARENA_MODE", "off").strip().lower() in {
        "shadow",
        "live",
    }


def _is_public_benchmark_path(arena_path: str) -> bool:
    parts = arena_path.split("/")
    return (
        len(parts) == 4
        and parts[:2] == ["v1", "rounds"]
        and parts[2] not in {"", ".", ".."}
        and parts[3] == "benchmark"
    )


def _is_public_results_path(arena_path: str) -> bool:
    parts = arena_path.split("/")
    return (
        len(parts) == 5
        and parts[:2] == ["v1", "rounds"]
        and parts[2] not in {"", ".", ".."}
        and parts[3] == "results"
        and parts[4] not in {"", ".", ".."}
    )


def _is_public_round_detail_path(arena_path: str) -> bool:
    parts = arena_path.split("/")
    return (
        len(parts) == 3
        and parts[:2] == ["v1", "rounds"]
        and parts[2] not in {"", ".", ".."}
    )


def _is_public_submissions_path(arena_path: str) -> bool:
    parts = arena_path.split("/")
    return (
        len(parts) == 4
        and parts[:2] == ["v1", "rounds"]
        and parts[2] not in {"", ".", ".."}
        and parts[3] == "submissions"
    )


def _is_public_source_code_path(arena_path: str) -> bool:
    parts = arena_path.split("/")
    return (
        len(parts) == 4
        and parts[:2] == ["v1", "submissions"]
        and parts[2] not in {"", ".", ".."}
        and parts[3] == "code"
    )


async def _bounded_body(request: Request, *, limit: int = _MAX_REQUEST_BYTES) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                raise HTTPException(status_code=413, detail="body too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="content-length invalid") from exc
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail="body too large")
        chunks.append(chunk)
    return b"".join(chunks)


async def _request_sidecar(
    method: str,
    path: str,
    *,
    query: str,
    body: bytes,
    headers: Mapping[str, str],
    testnet: bool = False,
) -> httpx.Response:
    timeout = httpx.Timeout(connect=3.0, read=150.0, write=30.0, pool=3.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        return await client.request(
            method,
            "%s/arena/%s" % (_TESTNET_SIDECAR_URL if testnet else _SIDECAR_URL, path),
            params=query,
            content=body,
            headers=dict(headers),
        )


# The Arena contract's own endpoints, registered individually so a request is
# labelled by the operation it performed instead of collapsing into the
# catch-all. Nothing about proxying changes: every one of these delegates to the
# same `_proxy_request` with the same derived path, the same guards, and the same
# body limits. They exist because the gateway exports the MATCHED ROUTE TEMPLATE
# as the span name, so an unlabelled catch-all makes a runner claiming work and a
# miner submitting an agent indistinguishable in telemetry — which is exactly the
# ambiguity that left the 2026-09-05 12:56-13:00 UTC submission refusals
# unattributable. The templates are low-cardinality and carry no client-controlled
# segment, so the telemetry boundary validator's route rules are unaffected.
#
# A path NOT listed here still matches the catch-all below and behaves exactly as
# it does today, so the sidecar can grow endpoints without this list blocking
# them; it only loses the named label until the entry is added.
# Registered with BOTH methods, exactly as the catch-all is, so method handling
# is unchanged: the sidecar still decides what it accepts, and a GET against a
# POST-only endpoint is still its 405, not the gateway's.
_CONTRACT_ROUTES: tuple[str, ...] = (
    "/v1/current",
    "/v1/signing-key",
    "/v1/reward-basis",
    "/v1/rounds/{round_id}",
    "/v1/rounds/{round_id}/benchmark",
    "/v1/rounds/{round_id}/results/{submission_id}",
    "/v1/submissions/presign",
    "/v1/submissions/{submission_id}",
    "/v1/submissions/{submission_id}/finalize",
    "/v1/runs/claim",
    "/v1/runs/{run_id}/provider",
    "/v1/runs/{run_id}/source",
    "/v1/runs/{run_id}/complete",
)
_CONTRACT_ROUTE_METHODS = ["GET", "POST"]


def _arena_path_of(request: Request, prefix: str) -> str:
    """The sidecar path this request names, taken from the routed path itself.

    ASGI has already decoded the path once, which is the same string the
    catch-all receives as its path parameter, so every downstream guard sees
    exactly what it sees today.
    """
    path = str(request.scope.get("path") or request.url.path)
    return path[len(prefix) + 1 :] if path.startswith(prefix + "/") else ""


async def _named_arena_request(request: Request) -> Response:
    if not _arena_enabled():
        raise HTTPException(status_code=404, detail="agent competition is disabled")
    return await _proxy_request(_arena_path_of(request, router.prefix), request)


async def _named_testnet_request(request: Request) -> Response:
    arena_path = _arena_path_of(request, testnet_router.prefix)
    return await _testnet_request(arena_path, request)


for _contract_path in _CONTRACT_ROUTES:
    router.add_api_route(
        _contract_path, _named_arena_request, methods=_CONTRACT_ROUTE_METHODS
    )
    testnet_router.add_api_route(
        _contract_path, _named_testnet_request, methods=_CONTRACT_ROUTE_METHODS
    )


@router.api_route("/{arena_path:path}", methods=("GET", "POST"))
async def proxy_arena_request(arena_path: str, request: Request) -> Response:
    if not _arena_enabled():
        raise HTTPException(status_code=404, detail="agent competition is disabled")
    return await _proxy_request(arena_path, request)


@testnet_router.api_route("/{arena_path:path}", methods=("GET", "POST"))
async def proxy_testnet_request(arena_path: str, request: Request) -> Response:
    return await _testnet_request(arena_path, request)


async def _testnet_request(arena_path: str, request: Request) -> Response:
    # An explicit operator switch and a fixed loopback destination keep testnet
    # requests out of the mainnet service. Never fall back if testnet is down.
    if os.environ.get("LAB_ARENA_TESTNET_ENABLED", "false").strip().lower() != "true":
        raise HTTPException(status_code=404, detail="testnet competition is disabled")
    # ASGI has decoded the path once. A residual percent could be decoded again
    # by an upstream client and must not bypass the exact benchmark guard.
    if "%" in arena_path:
        raise HTTPException(status_code=404, detail="arena path invalid")
    if request.method == "GET" and _is_public_benchmark_path(arena_path):
        raise HTTPException(status_code=403, detail="testnet benchmark is private")
    if request.method == "GET" and _is_public_results_path(arena_path):
        raise HTTPException(status_code=403, detail="testnet results are private")
    if request.method == "GET" and (
        _is_public_round_detail_path(arena_path)
        or _is_public_submissions_path(arena_path)
        or _is_public_source_code_path(arena_path)
    ):
        raise HTTPException(status_code=403, detail="testnet evaluation is private")
    # Temporary compatibility boundary: the current testnet worker predates
    # publication-gated results, source, and finalist projections. Remove these
    # GET guards together only after that worker is upgraded and verified.
    return await _proxy_request(arena_path, request, testnet=True)


async def _proxy_request(arena_path: str, request: Request, *, testnet: bool = False) -> Response:
    if not arena_path or any(part in {"", ".", ".."} for part in arena_path.split("/")):
        raise HTTPException(status_code=404, detail="arena path invalid")

    parts = arena_path.split("/")
    request_limit = (
        _MAX_COMPLETION_REQUEST_BYTES
        if request.method == "POST"
        and len(parts) == 4
        and parts[:2] == ["v1", "runs"]
        and parts[3] == "complete"
        else _MAX_REQUEST_BYTES
    )
    body = await _bounded_body(request, limit=request_limit)
    request_headers = {
        name: request.headers[name]
        for name in _FORWARDED_REQUEST_HEADERS
        if name in request.headers
    }
    try:
        upstream = await _request_sidecar(
            request.method,
            arena_path,
            query=request.url.query,
            body=body,
            headers=request_headers,
            **({"testnet": True} if testnet else {}),
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="agent competition is unavailable") from exc
    response_headers = {
        name: upstream.headers[name]
        for name in _FORWARDED_RESPONSE_HEADERS
        if name in upstream.headers
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
    )
