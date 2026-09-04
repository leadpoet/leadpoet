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

from lab_arena import contracts


router = APIRouter(prefix="/arena", tags=["agent-competition"])

_MAX_REQUEST_BYTES = 1_100_000
_SIDECAR_URL = "http://127.0.0.1:8792"
_FORWARDED_REQUEST_HEADERS = ("content-type", "x-lab-arena-lease")
_FORWARDED_RESPONSE_HEADERS = ("content-type", "cache-control")


def _arena_enabled() -> bool:
    return os.environ.get("LAB_ARENA_MODE", "off").strip().lower() in {
        "shadow",
        "live",
    }


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
) -> httpx.Response:
    timeout = httpx.Timeout(connect=3.0, read=150.0, write=30.0, pool=3.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        return await client.request(
            method,
            "%s/arena/%s" % (_SIDECAR_URL, path),
            params=query,
            content=body,
            headers=dict(headers),
        )


@router.api_route("/{arena_path:path}", methods=("GET", "POST"))
async def proxy_arena_request(arena_path: str, request: Request) -> Response:
    if not _arena_enabled():
        raise HTTPException(status_code=404, detail="agent competition is disabled")
    if not arena_path or any(part in {"", ".", ".."} for part in arena_path.split("/")):
        raise HTTPException(status_code=404, detail="arena path invalid")

    parts = arena_path.split("/")
    request_limit = (
        contracts.COMPLETION_REQUEST_LIMITS.max_total_bytes
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
