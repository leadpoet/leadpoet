"""``/arena/v1`` routes (labarena.md section 14): a thin FastAPI layer over
``ArenaService``. Every miner and runner request is a signed canonical Arena
document validated by the service; bodies are size-bounded here.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from lab_arena import contracts
from lab_arena.contracts import ArenaContractError
from lab_arena.service import ArenaService, ServiceError

MAX_JSON_BODY_BYTES = 1_048_576


def _refuse_declared_oversize(request: Request, limit: int) -> None:
    """Refuse a body whose declared length exceeds the limit before buffering it."""

    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                raise HTTPException(status_code=413, detail="body too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="content-length invalid")


async def _read_json(
    request: Request,
    *,
    limits: Optional[contracts.StrictLimits] = None,
) -> Any:
    limits = limits or contracts.StrictLimits(
        max_depth=12,
        max_list_items=2048,
        max_object_keys=256,
        max_string_bytes=524_288,
        max_total_bytes=MAX_JSON_BODY_BYTES,
    )
    limit = limits.max_total_bytes
    _refuse_declared_oversize(request, limit)
    chunks = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="body too large")
        chunks.append(chunk)
    raw = b"".join(chunks)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="body is not JSON")
    try:
        contracts.check_strict_document(document, limits)
    except ArenaContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)[:120])
    return document


def _lease_header(value: Optional[str]) -> str:
    if not value or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise HTTPException(status_code=401, detail="lease token invalid")
    return value


def create_app(service: ArenaService) -> FastAPI:
    app = FastAPI(title="Leadpoet Lab Arena", version=contracts.ARENA_CONTRACT_VERSION, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"status": "rejected", "code": exc.code})

    @app.exception_handler(ArenaContractError)
    async def _contract_error(request: Request, exc: ArenaContractError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"status": "rejected", "code": "contract:%s" % str(exc)[:100]})

    # -- public -----------------------------------------------------------

    @app.get("/arena/v1/current")
    async def current() -> Any:
        return await run_in_threadpool(service.public_current)

    @app.get("/arena/v1/signing-key")
    async def signing_key() -> Any:
        return await run_in_threadpool(service.signing_key_document)

    @app.get("/arena/v1/reward-basis")
    async def reward_basis(epoch: int) -> Any:
        basis = await run_in_threadpool(service.public_reward_basis, int(epoch))
        if basis is None:
            raise HTTPException(status_code=404, detail="no governing round")
        return basis

    @app.get("/arena/v1/rounds/{round_id}")
    async def round_view(round_id: str) -> Any:
        return await run_in_threadpool(service.public_round, round_id)

    @app.get("/arena/v1/rounds/{round_id}/benchmark")
    async def round_benchmark(round_id: str) -> Any:
        return await run_in_threadpool(service.public_benchmark, round_id)

    @app.get("/arena/v1/rounds/{round_id}/results/{submission_id}")
    async def round_results(round_id: str, submission_id: str) -> Any:
        return await run_in_threadpool(service.public_results, round_id, submission_id)

    # -- miner --------------------------------------------------------------

    @app.post("/arena/v1/submissions/presign")
    async def submission_presign(request: Request) -> Any:
        """Reserve one bounded private source upload."""

        envelope = await _read_json(request)
        return await run_in_threadpool(service.handle_submission_presign, envelope)

    @app.post("/arena/v1/submissions/{submission_id}/finalize")
    async def submission_finalize(submission_id: str, request: Request) -> Any:
        """Verify and accept the source bytes uploaded for this submission."""

        envelope = await _read_json(request)
        body = envelope.get("body") if isinstance(envelope, dict) else None
        if not isinstance(body, dict) or str(body.get("submission_id") or "") != submission_id:
            raise HTTPException(status_code=400, detail="path submission id does not match body")
        return await run_in_threadpool(
            service.handle_submission_finalize, submission_id, envelope
        )

    @app.get("/arena/v1/submissions/{submission_id}")
    async def submission_status(submission_id: str) -> Any:
        row = await run_in_threadpool(service.store.get_submission, submission_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown submission")
        return {
            "submission_id": submission_id, "status": row["status"], "rejection_rule": row.get("rejection_rule"),
        }

    # -- runner -------------------------------------------------------------

    @app.post("/arena/v1/runs/claim")
    async def claim(request: Request) -> Any:
        envelope = await _read_json(request)
        return await run_in_threadpool(service.handle_claim, envelope)

    @app.post("/arena/v1/runs/{run_id}/provider")
    async def provider(run_id: str, request: Request, x_lab_arena_lease: Optional[str] = Header(default=None)) -> Any:
        lease_token = _lease_header(x_lab_arena_lease)
        frame = await _read_json(request)
        return await run_in_threadpool(service.handle_provider, run_id, lease_token, frame)

    @app.get("/arena/v1/runs/{run_id}/source")
    async def source(run_id: str, x_lab_arena_lease: Optional[str] = Header(default=None)) -> Any:
        lease_token = _lease_header(x_lab_arena_lease)
        payload = await run_in_threadpool(service.handle_source, run_id, lease_token)
        return Response(content=payload, media_type="application/gzip")

    @app.post("/arena/v1/runs/{run_id}/complete")
    async def complete(run_id: str, request: Request) -> Any:
        envelope = await _read_json(
            request, limits=contracts.COMPLETION_REQUEST_LIMITS
        )
        body = envelope.get("body") if isinstance(envelope, dict) else None
        if not isinstance(body, dict) or str(body.get("run_id") or "") != run_id:
            raise HTTPException(status_code=400, detail="path run id does not match body")
        return await run_in_threadpool(service.handle_complete, envelope)

    return app
