#!/usr/bin/env python3
"""Serve the existing Arena gateway proxy without a coordinator dependency."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from leadpoet_observability.sentry_bootstrap import INGEST_ENVIRONMENT_KEYS


EDGE_ENVIRONMENT_KEYS = frozenset({
    "LAB_ARENA_MODE", "LAB_ARENA_TESTNET_ENABLED",
    "GATEWAY_OTEL_ENABLED", "GATEWAY_OTEL_ENDPOINT", "GATEWAY_OTEL_TOKEN",
    "GATEWAY_OTEL_METRICS_ENDPOINT", "GATEWAY_MAX_BODY_BYTES",
    "GATEWAY_UVICORN_LIMIT_CONCURRENCY", "GATEWAY_UVICORN_BACKLOG",
    "GATEWAY_UVICORN_KEEPALIVE_SECONDS",
    "MINER_MAX_CONCURRENT", "MINER_MAX_WAITING", "MINER_SLOT_WAIT_TIMEOUT_SECONDS",
    "VALIDATOR_MAX_CONCURRENT", "VALIDATOR_MAX_WAITING", "VALIDATOR_SLOT_WAIT_TIMEOUT_SECONDS",
    "OTHER_MAX_CONCURRENT", "OTHER_MAX_WAITING", "OTHER_SLOT_WAIT_TIMEOUT_SECONDS",
    *INGEST_ENVIRONMENT_KEYS,
})


def load_edge_environment(path: Path) -> None:
    """Read only routing and telemetry settings, never provider credentials."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Arena edge environment file is unavailable")
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("Arena edge environment file is too large")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError("Arena edge environment document is invalid")
        selected = {str(key): str(item) for key, item in value.items()
                    if str(key) in EDGE_ENVIRONMENT_KEYS}
    else:
        selected = {}
        for line in raw.decode("utf-8").replace("\0", "\n").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, separator, raw_value = line.partition("=")
            key = key.strip()
            if not separator or key not in EDGE_ENVIRONMENT_KEYS:
                continue
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError("Arena edge environment key is invalid")
            parts = shlex.split("VALUE=" + raw_value, comments=True, posix=True)
            if len(parts) != 1 or not parts[0].startswith("VALUE="):
                raise ValueError("Arena edge environment value is invalid")
            value = parts[0].split("=", 1)[1]
            if key in selected and selected[key] != value:
                raise ValueError("Arena edge environment key is duplicated")
            selected[key] = value
    for key, value in selected.items():
        os.environ.setdefault(key, value)


def create_app() -> FastAPI:
    # Import after the scoped environment is loaded: priority limits are
    # intentionally frozen when its module first loads in this process.
    from gateway.api.arena_proxy import router, testnet_router
    from gateway.middleware.body_size import BodySizeLimitMiddleware
    from gateway.middleware.priority import PriorityMiddleware

    app = FastAPI(
        title="LeadPoet Arena Proxy",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(PriorityMiddleware, max_concurrent_miners=75)
    app.include_router(router)
    app.include_router(testnet_router)
    try:
        from gateway.observability.otel_bootstrap import configure_gateway_otel

        configure_gateway_otel(app)
    except Exception as exc:
        print("arena edge telemetry unavailable", type(exc).__name__, file=sys.stderr)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, choices=(8800, 8801), required=True)
    parser.add_argument("--sidecar-port", type=int, choices=(8792, 8794), required=True)
    parser.add_argument("--environment-file", type=Path, required=True)
    args = parser.parse_args()

    source_sha = os.environ.get("LAB_ARENA_RELEASE_COMMIT", "")
    if (not re.fullmatch(r"[0-9a-f]{40}", source_sha)
            or os.environ.get("GITHUB_SHA") != source_sha
            or Path(os.environ.get("LAB_ARENA_RELEASE_ROOT", "")).resolve() != ROOT):
        raise ValueError("Arena edge source identity differs from its verified release")

    load_edge_environment(args.environment_file)
    os.environ["LAB_ARENA_PROXY_MAIN_PORT"] = str(args.sidecar_port)
    try:
        from leadpoet_observability import init_sentry, configure_sentry_context

        init_sentry(component="gateway")
        configure_sentry_context(
            component="gateway", physical_role="arena-edge",
            runtime_sha=os.environ.get("GITHUB_SHA", ""),
            restart_invocation_id=os.environ.get("LEADPOET_RESTART_INVOCATION_ID"),
        )
    except Exception as exc:
        print("arena edge sentry unavailable", type(exc).__name__, file=sys.stderr)

    import uvicorn

    print(f"ARENA_EDGE_LIFECYCLE event=startup source_sha={source_sha} sidecar_port={args.sidecar_port}")
    uvicorn.run(
        create_app(), host="127.0.0.1", port=args.port, log_level="info",
        proxy_headers=True, forwarded_allow_ips="127.0.0.1",
        limit_concurrency=int(os.getenv("GATEWAY_UVICORN_LIMIT_CONCURRENCY", "300")),
        backlog=int(os.getenv("GATEWAY_UVICORN_BACKLOG", "2048")),
        timeout_keep_alive=int(os.getenv("GATEWAY_UVICORN_KEEPALIVE_SECONDS", "5")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
