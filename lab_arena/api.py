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
from lab_arena.store import ArenaStoreUnavailable

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

    async def no_store_public_call(call: Any, *args: Any) -> JSONResponse:
        headers = {
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        }
        try:
            content = await run_in_threadpool(call, *args)
        except ServiceError as exc:
            return JSONResponse(
                status_code=exc.status,
                content={"status": "rejected", "code": exc.code},
                headers=headers,
            )
        return JSONResponse(content=content, headers=headers)

    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError) -> JSONResponse:
        content = {"status": "rejected", "code": exc.code}
        if exc.code == "submission_rejected:source_contains_credentials" and exc.source_path:
            content["source_path"] = exc.source_path
        return JSONResponse(status_code=exc.status, content=content)

    @app.exception_handler(ArenaContractError)
    async def _contract_error(request: Request, exc: ArenaContractError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"status": "rejected", "code": "contract:%s" % str(exc)[:100]})

    @app.exception_handler(ArenaStoreUnavailable)
    async def _store_unavailable(request: Request, exc: ArenaStoreUnavailable) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "code": "arena_store_unavailable"},
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Retry-After": "1",
            },
        )

    # -- public -----------------------------------------------------------

    @app.get("/arena/v1/current")
    async def current() -> Any:
        return await run_in_threadpool(service.public_current)

    @app.get("/arena/v1/competition")
    async def competition() -> Any:
        return await run_in_threadpool(service.public_competition)

    @app.get("/arena/v1/signing-key")
    async def signing_key() -> Any:
        return await run_in_threadpool(service.signing_key_document)

    @app.get("/arena/v1/reward-basis")
    async def reward_basis(epoch: int) -> Any:
        basis = await run_in_threadpool(service.public_reward_basis, int(epoch))
        if basis is None:
            raise HTTPException(status_code=404, detail="no governing round")
        return basis

    @app.get("/arena/v1/weight-state")
    async def accepted_weight_state(epoch: int) -> Any:
        return await run_in_threadpool(service.public_weight_state, int(epoch))

    @app.post("/arena/v1/chain-outcomes")
    async def record_chain_outcome(request: Request) -> Any:
        document = await _read_json(request)
        return await run_in_threadpool(service.record_chain_outcome, document)

    @app.get("/arena/v1/chain-outcomes")
    async def chain_outcomes(epoch: int) -> Any:
        return await run_in_threadpool(service.public_chain_outcomes, int(epoch))

    @app.get("/arena/v1/rounds/{round_id}")
    async def round_view(round_id: str) -> Any:
        return await run_in_threadpool(service.public_round, round_id)

    @app.get("/arena/v1/rounds/{round_id}/benchmark-commitment")
    async def round_benchmark_commitment(round_id: str) -> JSONResponse:
        return await no_store_public_call(service.public_benchmark_commitment, round_id)

    @app.get("/arena/v1/rounds/{round_id}/benchmark")
    async def round_benchmark(round_id: str) -> JSONResponse:
        return await no_store_public_call(service.public_benchmark, round_id)

    @app.get("/arena/v1/rounds/{round_id}/submissions")
    async def round_submissions(round_id: str) -> Any:
        return await run_in_threadpool(service.public_submissions, round_id)

    @app.get("/arena/v1/rounds/{round_id}/results/{submission_id}")
    async def round_results(round_id: str, submission_id: str) -> JSONResponse:
        return await no_store_public_call(
            service.public_results, round_id, submission_id
        )

    @app.get("/arena/v1/submissions/{submission_id}/code")
    async def submission_code(submission_id: str) -> JSONResponse:
        return await no_store_public_call(
            service.public_submission_code, submission_id
        )

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
        return await run_in_threadpool(service.submission_status, submission_id)

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

    @app.get("/arena/v1/runs/{run_id}/image-access")
    async def scorer_image_access(
        run_id: str,
        x_lab_arena_lease: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        lease_token = _lease_header(x_lab_arena_lease)
        return await no_store_public_call(
            service.handle_scorer_image_access, run_id, lease_token
        )

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
