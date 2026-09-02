"""``/arena/v1`` routes (labarena.md section 14): a thin FastAPI layer over
``ArenaService``. Every miner and runner request is a signed canonical Arena
document validated by the service; bodies are size-bounded here.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from lab_arena import contracts, credentials
from lab_arena.contracts import ArenaContractError
from lab_arena.service import ArenaService, ServiceError

MAX_JSON_BODY_BYTES = 1_048_576
MAX_PACKAGE_BODY_BYTES = 64 * 1024 * 1024


async def _read_json(request: Request, *, limit: int = MAX_JSON_BODY_BYTES) -> Any:
    raw = await request.body()
    if len(raw) > limit:
        raise HTTPException(status_code=413, detail="body too large")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="body is not JSON")
    try:
        contracts.check_strict_document(document, contracts.PUBLICATION_LIMITS if limit > MAX_JSON_BODY_BYTES else contracts.StrictLimits(max_depth=12, max_list_items=2048, max_object_keys=256, max_string_bytes=524_288, max_total_bytes=limit))
    except ArenaContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)[:120])
    return document


def _lease_header(value: Optional[str]) -> str:
    if not value or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise HTTPException(status_code=401, detail="lease token invalid")
    return value


def create_app(service: ArenaService, *, recipient_document: Optional[Dict[str, Any]] = None, funding_confirm=None, credential_register=None) -> FastAPI:
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
        return service.public_current()

    @app.get("/arena/v1/recipient")
    async def recipient() -> Any:
        if recipient_document is None:
            raise HTTPException(status_code=503, detail="recipient key unavailable")
        return recipient_document

    @app.get("/arena/v1/signing-key")
    async def signing_key() -> Any:
        return service.signing_key_document()

    @app.get("/arena/v1/reward-basis")
    async def reward_basis(epoch: int) -> Any:
        basis = service.public_reward_basis(int(epoch))
        if basis is None:
            raise HTTPException(status_code=404, detail="no governing round")
        return basis

    @app.get("/arena/v1/rounds/{round_id}")
    async def round_view(round_id: str) -> Any:
        return service.public_round(round_id)

    @app.get("/arena/v1/rounds/{round_id}/benchmark")
    async def round_benchmark(round_id: str) -> Any:
        return service.public_benchmark(round_id)

    @app.get("/arena/v1/rounds/{round_id}/results/{submission_id}")
    async def round_results(round_id: str, submission_id: str) -> Any:
        return service.public_results(round_id, submission_id)

    # -- miner --------------------------------------------------------------

    @app.post("/arena/v1/submissions")
    async def submissions(request: Request, x_lab_arena_envelope: Optional[str] = Header(default=None)) -> Any:
        raw = await request.body()
        if len(raw) > MAX_PACKAGE_BODY_BYTES:
            raise HTTPException(status_code=413, detail="package too large")
        if not x_lab_arena_envelope:
            raise HTTPException(status_code=400, detail="signed envelope header required")
        try:
            envelope = json.loads(x_lab_arena_envelope)
        except ValueError:
            raise HTTPException(status_code=400, detail="envelope is not JSON")
        return service.handle_submission(envelope, raw)

    @app.get("/arena/v1/submissions/{submission_id}")
    async def submission_status(submission_id: str) -> Any:
        row = service.store.get_submission(submission_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown submission")
        return {"submission_id": submission_id, "status": row["status"], "rejection_rule": row.get("rejection_rule"), "image_digest": row.get("image_digest")}

    confirm_deposit = funding_confirm
    register_key = credential_register

    @app.post("/arena/v1/funding/confirm")
    async def funding_confirm_route(request: Request) -> Any:
        if confirm_deposit is None:
            raise HTTPException(status_code=503, detail="funding unavailable")
        return service.handle_funding(await _read_json(request), confirm=confirm_deposit)

    @app.post("/arena/v1/credentials/openrouter")
    async def credentials_openrouter_route(request: Request) -> Any:
        if register_key is None:
            raise HTTPException(status_code=503, detail="credential registration unavailable")
        return service.handle_credential(await _read_json(request), register=register_key)

    # -- runner -------------------------------------------------------------

    @app.post("/arena/v1/runs/claim")
    async def claim(request: Request) -> Any:
        return service.handle_claim(await _read_json(request))

    @app.post("/arena/v1/runs/{run_id}/provider")
    async def provider(run_id: str, request: Request, x_lab_arena_lease: Optional[str] = Header(default=None)) -> Any:
        return service.handle_provider(run_id, _lease_header(x_lab_arena_lease), await _read_json(request))

    @app.post("/arena/v1/runs/{run_id}/events")
    async def events(run_id: str, request: Request, x_lab_arena_lease: Optional[str] = Header(default=None)) -> Any:
        document = await _read_json(request)
        return service.handle_events(run_id, _lease_header(x_lab_arena_lease), document.get("events") if isinstance(document, dict) else None)

    @app.post("/arena/v1/runs/{run_id}/complete")
    async def complete(run_id: str, request: Request) -> Any:
        envelope = await _read_json(request)
        return service.handle_complete(envelope)

    return app
