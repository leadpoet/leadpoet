#!/usr/bin/env python3
"""Run the full gateway and non-forwarding weight path on one Nitro host."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
    urlopen,
)

import boto3
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_canonical.production_parity import (  # noqa: E402
    ProductionParityError,
    sha256_json,
    validate_snapshot_manifest,
)
from leadpoet_canonical.production_parity_boundary_v2 import (  # noqa: E402
    validate_production_parity_boundary_document_v2,
)
from scripts.build_production_parity_contract import build_contract  # noqa: E402
from scripts.capture_production_parity_runtime_config import capture  # noqa: E402
from scripts.materialize_production_parity_secrets import (  # noqa: E402
    SecretMaterializationError,
    _parse_environment_document,
    create as create_gateway_secret,
    delete as delete_gateway_secret,
    is_process_control_environment_key,
    production_parity_scoring_cache_dir,
    production_parity_trace_prefixes,
)
from scripts.production_parity_snapshot import (  # noqa: E402
    capture_snapshot,
    restore_snapshot,
)
from scripts.run_production_parity_fast import _DockerDatabase  # noqa: E402
from qualification.competition_models import (  # noqa: E402
    public_http_url,
    validate_companies,
)


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
ARTIFACT_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
PINNED_IMAGE_RE = re.compile(r"^[A-Za-z0-9._/:@-]+@sha256:[0-9a-f]{64}$")
SCHEMA_VERSION = "leadpoet.production_parity_full.v3"
ARENA_REBENCHMARK_REQUEST_SCHEMA_VERSION = (
    "leadpoet.production_parity_arena_rebenchmark_request.v1"
)
ARENA_REBENCHMARK_EVIDENCE_SCHEMA_VERSION = (
    "leadpoet.production_parity_arena_rebenchmark_evidence.v1"
)
ARENA_BASELINE_SOURCE_URL = (
    "https://github.com/leadpoet/pydantic-harness/"
    "archive/refs/heads/main.tar.gz"
)
MINER_INTAKE_ENVIRONMENT_OVERRIDES = {
    "RESEARCH_LAB_GATEWAY_API_ENABLED": "true",
    "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED": "true",
    "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
    "RESEARCH_LAB_SOURCE_ADD_ENABLED": "true",
    "RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED": "false",
}
EARLY_BOOT_MARKER = Path(
    "/run/leadpoet-production-parity/early-boot-isolated"
)
FULL_WORK_ROOT = Path("/opt/leadpoet-production-parity")
ATTESTED_V2_RELEASE_BUCKET = "leadpoet-attested-v2-artifacts-493765492819"
ATTESTED_V2_RELEASE_PREFIX = "attested-v2/releases"
ATTESTED_V2_KMS_KEY_ID = (
    "arn:aws:kms:us-east-1:493765492819:"
    "key/c5412928-093e-4bf5-aafc-7b27c02f1445"
)
MAX_FULL_TIMEOUT_SECONDS = 72_000
CLONE_PREFIX_ADAPTER_MAX_BODY_BYTES = 64 * 1024 * 1024
CLONE_PREFIX_ADAPTER_TIMEOUT_SECONDS = 15
CHILD_REQUEST_MAX_BYTES = 64 * 1024
CLONE_PREFIX_ADAPTER_METHODS = frozenset(
    {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"}
)
FULL_FAILURE_STAGES = frozenset(
    {
        "initialization",
        "public-origin",
        "checkout-identity",
        "early-boot-isolation",
        "runtime-identity",
        "work-root",
        "runtime-config-capture",
        "parity-contract",
        "production-dsn",
        "snapshot-capture",
        "clone-start",
        "snapshot-restore",
        "clone-http-origin",
        "clone-secret",
        "gateway-restart",
        "gateway-health",
        "arena-rebenchmark",
        "weight-readiness",
        "allocation-handoff",
        "nonforwarding-weight-path",
        "miner-intake",
        "clone-shape",
        "clone-weight-history",
        "cleanup",
        "unknown",
    }
)
FULL_ERROR_TYPES = frozenset(
    {
        "BotoCoreError",
        "CalledProcessError",
        "ClientError",
        "CleanupError",
        "FullParityError",
        "HTTPError",
        "JSONDecodeError",
        "OSError",
        "ProductionParityError",
        "RuntimeError",
        "TimeoutError",
        "TimeoutExpired",
        "URLError",
        "ValueError",
    }
)
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _connection_nominated_headers(headers: Mapping[str, Any]) -> set[str]:
    value = str(headers.get("Connection") or "")
    return {
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    }


class FullParityError(RuntimeError):
    """The full disposable workflow did not reach every required stage."""


def _validated_public_origin(origin: str) -> str:
    parsed = urlsplit(str(origin or ""))
    try:
        port = parsed.port
    except ValueError as exc:
        raise FullParityError("clone public origin identity is invalid") from exc
    if (
        parsed.scheme != "https"
        or not str(parsed.hostname or "").endswith(".cloudfront.net")
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise FullParityError("clone public origin identity is invalid")
    return f"https://{parsed.hostname}"


def _failure_identity(stage: str, exc: BaseException) -> tuple[str, str]:
    bounded_stage = stage if stage in FULL_FAILURE_STAGES else "unknown"
    raw_type = type(exc).__name__
    bounded_type = raw_type if raw_type in FULL_ERROR_TYPES else "UnexpectedError"
    return bounded_stage, bounded_type


def _write_early_failure_evidence(
    *,
    output: Path,
    run_id: str,
    base_sha: str,
    candidate_sha: str,
    started_at: datetime,
    started_monotonic: float,
    error: BaseException,
) -> None:
    """Retain a bounded failure identity when run_full exits before its finally."""

    if (
        RUN_RE.fullmatch(run_id) is None
        or SHA_RE.fullmatch(base_sha) is None
        or SHA_RE.fullmatch(candidate_sha) is None
    ):
        return
    failure_stage, error_type = _failure_identity("initialization", error)
    finished_at = datetime.now(timezone.utc)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "base_sha": base_sha,
        "started_at": started_at.isoformat(),
        "status": "failed",
        "failure_stage": failure_stage,
        "error_type": error_type,
        "cleanup": {},
        "duration_seconds": round(
            max(0.0, time.monotonic() - started_monotonic),
            3,
        ),
        "finished_at": finished_at.isoformat(),
    }
    payload = (
        json.dumps(evidence, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(output, flags, 0o600)
    except FileExistsError:
        return
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("bounded evidence write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _RejectCloneRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise HTTPError(req.full_url, code, msg, headers, fp)


class _ClonePrefixServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        upstream_origin: str,
        public_origin: str,
    ) -> None:
        self.upstream_origin = upstream_origin
        self.public_origin = public_origin
        self.opener = build_opener(ProxyHandler({}), _RejectCloneRedirects())
        super().__init__(server_address, handler)


class _ClonePrefixHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return None

    def _reject(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _forward(self) -> None:
        if self.command not in CLONE_PREFIX_ADAPTER_METHODS:
            self._reject(405)
            return
        target = urlsplit(self.path)
        request_origin = str(self.headers.get("Origin") or "").rstrip("/")
        if (
            target.scheme
            or target.netloc
            or target.fragment
            or (
                target.path not in {"/rest/v1", "/rest/v1/"}
                and not target.path.startswith("/rest/v1/")
            )
            or (
                request_origin
                and request_origin
                != self.server.public_origin  # type: ignore[attr-defined]
            )
            or self.headers.get("Proxy-Authorization") is not None
            or self.headers.get("Proxy-Connection") is not None
            or self.headers.get("Transfer-Encoding") is not None
        ):
            self._reject(404)
            return
        raw_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_length)
        except ValueError:
            self._reject(400)
            return
        if not 0 <= content_length <= CLONE_PREFIX_ADAPTER_MAX_BODY_BYTES:
            self._reject(413)
            return
        self.connection.settimeout(CLONE_PREFIX_ADAPTER_TIMEOUT_SECONDS)
        try:
            body = self.rfile.read(content_length) if content_length else b""
        except (OSError, TimeoutError):
            self._reject(408)
            return
        if len(body) != content_length:
            self._reject(400)
            return

        upstream_path = target.path[len("/rest/v1") :] or "/"
        upstream = urlsplit(self.server.upstream_origin)  # type: ignore[attr-defined]
        upstream_url = urlunsplit(
            (upstream.scheme, upstream.netloc, upstream_path, target.query, "")
        )
        request_hop_headers = (
            set(_HOP_BY_HOP_HEADERS)
            | _connection_nominated_headers(self.headers)
        )
        headers = {
            str(name): str(value)
            for name, value in self.headers.items()
            if str(name).lower() not in request_hop_headers
        }
        request = Request(
            upstream_url,
            data=body if content_length else None,
            headers=headers,
            method=self.command,
        )
        response = None
        try:
            try:
                response = self.server.opener.open(  # type: ignore[attr-defined]
                    request, timeout=CLONE_PREFIX_ADAPTER_TIMEOUT_SECONDS
                )
            except HTTPError as exc:
                response = exc
            status = int(response.getcode())
            if 300 <= status < 400:
                self._reject(502)
                return
            response_body = response.read(CLONE_PREFIX_ADAPTER_MAX_BODY_BYTES + 1)
            if len(response_body) > CLONE_PREFIX_ADAPTER_MAX_BODY_BYTES:
                self._reject(502)
                return
            response_headers = list(response.headers.items())
            response_hop_headers = (
                set(_HOP_BY_HOP_HEADERS)
                | _connection_nominated_headers(response.headers)
            )
        except (HTTPException, OSError, TimeoutError, URLError):
            self._reject(502)
            return
        finally:
            if response is not None:
                response.close()

        self.send_response(status)
        for name, value in response_headers:
            if str(name).lower() not in response_hop_headers:
                self.send_header(str(name), str(value))
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(response_body)
        self.close_connection = True

    do_GET = _forward
    do_HEAD = _forward
    do_OPTIONS = _forward
    do_POST = _forward
    do_PUT = _forward
    do_PATCH = _forward
    do_DELETE = _forward


class _ClonePostgrestPrefixAdapter:
    """Expose only Supabase `/rest/v1` paths to one loopback PostgREST."""

    def __init__(
        self,
        *,
        upstream_origin: str,
        public_origin: str,
        listen_host: str = "0.0.0.0",
        listen_port: int = 3000,
    ) -> None:
        parsed = urlsplit(upstream_origin)
        try:
            upstream_port = parsed.port
        except ValueError as exc:
            raise FullParityError(
                "clone PostgREST prefix adapter identity is invalid"
            ) from exc
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or upstream_port in (None, 3000)
            or not 1 <= upstream_port <= 65535
            or listen_host not in {"0.0.0.0", "127.0.0.1"}
            or not isinstance(listen_port, int)
            or isinstance(listen_port, bool)
            or not 0 <= listen_port <= 65535
        ):
            raise FullParityError("clone PostgREST prefix adapter identity is invalid")
        self.upstream_origin = upstream_origin.rstrip("/")
        self.public_origin = _validated_public_origin(public_origin)
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.server: _ClonePrefixServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> dict[str, Any]:
        if self.server is not None or self.thread is not None:
            raise FullParityError("clone PostgREST prefix adapter already started")
        try:
            server = _ClonePrefixServer(
                (self.listen_host, self.listen_port),
                _ClonePrefixHandler,
                upstream_origin=self.upstream_origin,
                public_origin=self.public_origin,
            )
        except OSError as exc:
            raise FullParityError(
                "clone PostgREST prefix adapter could not bind"
            ) from exc
        thread = threading.Thread(
            target=server.serve_forever,
            name="production-parity-clone-prefix-adapter",
            daemon=True,
        )
        thread.start()
        if not thread.is_alive():
            server.server_close()
            raise FullParityError("clone PostgREST prefix adapter did not start")
        self.server = server
        self.thread = thread
        return {
            "listen_host": str(server.server_address[0]),
            "listen_port": int(server.server_address[1]),
            "upstream_host": "127.0.0.1",
            "path_prefix": "/rest/v1",
        }

    def cleanup(self) -> str:
        server = self.server
        thread = self.thread
        if server is None or thread is None:
            return "already_absent"
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        self.server = None
        self.thread = None
        if thread.is_alive():
            raise FullParityError(
                "clone PostgREST prefix adapter remained after cleanup"
            )
        return "removed"


def _full_deadline(*, started: float, timeout_seconds: int) -> float:
    if (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
        or timeout_seconds > MAX_FULL_TIMEOUT_SECONDS
    ):
        raise FullParityError("full parity timeout is invalid")
    return started + timeout_seconds


def _remaining_full_timeout(*, deadline: float, stage: str) -> int:
    remaining = math.ceil(deadline - time.monotonic())
    if remaining <= 0:
        raise FullParityError(f"full parity budget exhausted before {stage}")
    return min(remaining, MAX_FULL_TIMEOUT_SECONDS)


def _run(
    command: Sequence[str],
    *,
    timeout: int,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if log_path is None:
        return subprocess.run(
            list(command),
            cwd=ROOT,
            env=dict(env) if env is not None else None,
            text=True,
            capture_output=True,
            input=input_text,
            check=False,
            timeout=timeout,
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        return subprocess.run(
            list(command),
            cwd=ROOT,
            env=dict(env) if env is not None else None,
            text=True,
            input=input_text,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )


def _require(result: subprocess.CompletedProcess[str], *, stage: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-1200:]
        raise FullParityError(f"{stage} failed: {detail}")
    return result.stdout or ""


def _checkout_identity(candidate_sha: str) -> None:
    head = _require(
        _run(["git", "rev-parse", "HEAD"], timeout=20),
        stage="candidate source identity",
    ).strip()
    dirty = _require(
        _run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            timeout=20,
        ),
        stage="candidate source cleanliness",
    ).strip()
    if head != candidate_sha or dirty:
        raise FullParityError("full parity checkout differs from the exact candidate")


def _secret_value(client: Any, secret_id: str, *, field: str) -> str:
    value = client.get_secret_value(SecretId=secret_id).get("SecretString")
    if not isinstance(value, str) or not value:
        raise FullParityError(f"{field} is unavailable")
    return value


def _dsn_from_secret(raw: str) -> str:
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    if isinstance(value, Mapping):
        candidates = [
            value.get("dsn"),
            value.get("url"),
            value.get("readonly_dsn"),
        ]
        value = next((item for item in candidates if item), "")
    dsn = str(value or "").strip()
    parsed = urlparse(dsn)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise FullParityError("read-only production DSN secret is invalid")
    return dsn



def _wait_https_origin(origin: str, *, timeout_seconds: int = 300) -> None:
    origin = _validated_public_origin(origin)
    deadline = time.monotonic() + timeout_seconds
    last_error = "pending"
    opener = build_opener(ProxyHandler({}), _RejectCloneRedirects())
    while time.monotonic() < deadline:
        try:
            request = Request(
                origin.rstrip("/") + "/rest/v1/", method="GET"
            )
            with opener.open(request, timeout=10) as response:
                if int(response.status) == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - bounded readiness probe
            last_error = type(exc).__name__
        time.sleep(5)
    raise FullParityError(f"TLS clone origin did not become ready: {last_error}")


def _validated_clone_environment(
    gateway_env_file: Path,
    *,
    candidate_sha: str,
    run_id: str,
    supabase_origin: str,
    artifact_bucket: str,
) -> dict[str, str]:
    expected_path = FULL_WORK_ROOT / run_id / "runtime" / "gateway.env"
    try:
        metadata = gateway_env_file.lstat()
    except OSError as exc:
        raise FullParityError("clone gateway environment is unavailable") from exc
    if (
        gateway_env_file.resolve() != expected_path.resolve()
        or gateway_env_file.is_symlink()
        or not gateway_env_file.is_file()
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o777 != 0o600
    ):
        raise FullParityError("clone gateway environment ownership differs")
    values = _parse_gateway_environment_file(gateway_env_file)
    try:
        boundary = validate_production_parity_boundary_document_v2(
            values,
            network=str(values.get("BITTENSOR_NETWORK") or ""),
            netuid=int(values.get("BITTENSOR_NETUID") or 0),
        )
    except (TypeError, ValueError) as exc:
        raise FullParityError("clone gateway boundary is invalid") from exc
    normalized_origin = supabase_origin.rstrip("/")
    if not ARTIFACT_BUCKET_RE.fullmatch(artifact_bucket):
        raise FullParityError("clone gateway boundary identity differs")
    forbidden_aws_environment = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "BOTO_CONFIG",
    }
    forbidden_external_authority = {
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_ENDPOINT",
        "MINIO_BUCKET",
    }
    try:
        expected_trace_prefixes = production_parity_trace_prefixes(
            artifact_bucket=artifact_bucket,
            run_id=run_id,
        )
    except SecretMaterializationError as exc:
        raise FullParityError("clone trace boundary identity differs") from exc
    if (
        boundary.get("mode") != "production-parity"
        or boundary.get("run_id") != run_id
        or boundary.get("supabase_origin") != normalized_origin
        or values.get("LEADPOET_PARITY_CANDIDATE_SHA") != candidate_sha
        or str(values.get("SUPABASE_URL") or "").rstrip("/")
        != normalized_origin
        or not str(values.get("SUPABASE_ANON_KEY") or "").strip()
        or not str(values.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
        or str(values.get("DISABLE_BACKGROUND_TASKS") or "").lower()
        != "true"
        or str(values.get("LANGFUSE_ENABLED") or "").lower() != "false"
        or str(values.get("LAB_ARENA_MODE") or "").lower() != "off"
        or str(values.get("LAB_ARENA_SUPABASE_URL") or "").rstrip("/")
        != normalized_origin
        or not str(values.get("LAB_ARENA_SUPABASE_ANON_KEY") or "").strip()
        or str(values.get("LAB_ARENA_SERVICE_JWT") or "").count(".") != 2
        or values.get("LAB_ARENA_BUCKET") != artifact_bucket
        or values.get("AWS_S3_BUCKET") != artifact_bucket
        or values.get("RESEARCH_LAB_ATTESTED_V2_ARTIFACT_BUCKET")
        != artifact_bucket
        or str(values.get("RESEARCH_LAB_CORPUS_EXPORT_ENABLED") or "").lower()
        != "false"
        or bool(values.get("RESEARCH_LAB_CORPUS_EXPORT_S3_PREFIX"))
        or bool(values.get("RESEARCH_LAB_EVIDENCE_PROXY_URL"))
        or bool(values.get("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR"))
        or bool(values.get("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH"))
        or bool(values.get("RESEARCH_LAB_PROVIDER_OUTCOME_SIDECAR_PATH"))
        or bool(values.get("RESEARCH_LAB_SCORE_BUNDLE_SIGNATURE_URI_PREFIX"))
        or values.get("RESEARCH_LAB_SCORING_CACHE_DIR")
        != production_parity_scoring_cache_dir(run_id=run_id)
        or any(
            values.get(name) != expected
            for name, expected in expected_trace_prefixes.items()
        )
        or any(str(values.get(name) or "").strip() for name in forbidden_aws_environment)
        or any(name in values for name in forbidden_external_authority)
    ):
        raise FullParityError("clone gateway boundary identity differs")
    return values


def _write_run_owned_secret(path: Path, payload: bytes, *, field: str) -> str:
    if not payload:
        raise FullParityError(f"run-owned {field} payload is empty")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FullParityError(f"run-owned {field} write failed")
            view = view[written:]
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    metadata = path.lstat()
    if (
        path.is_symlink()
        or path.resolve(strict=True) != path
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != len(payload)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or (metadata.st_uid, metadata.st_gid) != (os.getuid(), os.getgid())
    ):
        raise FullParityError(f"run-owned {field} identity differs")
    return str(path)


def _b64url_uint(value: int) -> str:
    if not isinstance(value, int) or value < 0:
        raise FullParityError("run-owned RSA identity is invalid")
    width = max(1, (value.bit_length() + 7) // 8)
    return base64.urlsafe_b64encode(value.to_bytes(width, "big")).decode(
        "ascii"
    ).rstrip("=")


def _materialize_run_owned_runtime_identity(identity_dir: Path) -> dict[str, str]:
    """Create non-production signing material for the non-forwarding clone."""

    try:
        identity_dir.mkdir(mode=0o700)
        os.chown(
            identity_dir,
            -1,
            os.getgid(),
            follow_symlinks=False,
        )
        os.chmod(identity_dir, 0o700, follow_symlinks=False)
    except OSError as exc:
        raise FullParityError(
            "run-owned runtime identity directory is unavailable"
        ) from exc
    directory_metadata = identity_dir.lstat()
    if (
        identity_dir.is_symlink()
        or not stat.S_ISDIR(directory_metadata.st_mode)
        or stat.S_IMODE(directory_metadata.st_mode) != 0o700
        or (directory_metadata.st_uid, directory_metadata.st_gid)
        != (os.getuid(), os.getgid())
    ):
        raise FullParityError("run-owned runtime identity directory differs")

    gateway_key = Ed25519PrivateKey.generate()
    gateway_private_key = gateway_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    gateway_public_key = gateway_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    gateway_private_key_path = _write_run_owned_secret(
        identity_dir / "gateway_private_key.pem",
        gateway_private_key,
        field="gateway private-key",
    )

    arweave_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
    )
    numbers = arweave_key.private_numbers()
    public_numbers = numbers.public_numbers
    arweave_jwk = {
        "d": _b64url_uint(numbers.d),
        "dp": _b64url_uint(numbers.dmp1),
        "dq": _b64url_uint(numbers.dmq1),
        "e": _b64url_uint(public_numbers.e),
        "kty": "RSA",
        "n": _b64url_uint(public_numbers.n),
        "p": _b64url_uint(numbers.p),
        "q": _b64url_uint(numbers.q),
        "qi": _b64url_uint(numbers.iqmp),
    }
    arweave_payload = (
        json.dumps(arweave_jwk, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    arweave_keyfile_path = _write_run_owned_secret(
        identity_dir / "arweave_keyfile.json",
        arweave_payload,
        field="Arweave keyfile",
    )
    try:
        from arweave.arweave_lib import Wallet

        arweave_address = str(Wallet(arweave_keyfile_path).address or "")
    except Exception as exc:
        raise FullParityError("run-owned Arweave identity is invalid") from exc
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", arweave_address):
        raise FullParityError("run-owned Arweave identity is invalid")

    return {
        "gateway_private_key_path": gateway_private_key_path,
        "gateway_public_key": gateway_public_key,
        "gateway_public_key_hash": "sha256:"
        + hashlib.sha256(bytes.fromhex(gateway_public_key)).hexdigest(),
        "arweave_keyfile_path": arweave_keyfile_path,
        "arweave_address_hash": "sha256:"
        + hashlib.sha256(arweave_address.encode("ascii")).hexdigest(),
    }


@contextmanager
def _applied_clone_environment(
    values: Mapping[str, str],
    gateway_env_file: Path,
    *,
    overrides: Mapping[str, str] | None = None,
) -> Iterator[None]:
    explicit_overrides = dict(overrides or {})
    if any(
        is_process_control_environment_key(key)
        for key in explicit_overrides
    ):
        raise FullParityError("clone environment override is invalid")
    updates = {
        **{
            key: value
            for key, value in values.items()
            if not is_process_control_environment_key(key)
        },
        "GATEWAY_ENV_FILE": str(gateway_env_file),
        **explicit_overrides,
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _clone_child_environment(*, region: str | None = None) -> dict[str, str]:
    child = {
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "BOTO_CONFIG": "/dev/null",
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
    }
    if region is not None:
        child["AWS_REGION"] = region
        child["AWS_DEFAULT_REGION"] = region
    return child


def _clone_runtime_environment(
    values: Mapping[str, str],
    *,
    gateway_env_file: Path,
    region: str,
) -> dict[str, str]:
    if region != "us-east-1":
        raise FullParityError("clone runtime region is invalid")
    return {
        **_clone_child_environment(region=region),
        **{
            key: value
            for key, value in values.items()
            if not is_process_control_environment_key(key)
        },
        "GATEWAY_ENV_FILE": str(gateway_env_file),
        "AWS_REGION": region,
        "AWS_DEFAULT_REGION": region,
    }


def _full_restart_environment(
    *,
    region: str,
    updates: Mapping[str, str],
) -> dict[str, str]:
    if region != "us-east-1":
        raise FullParityError("gateway restart region is invalid")
    return {
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "BOTO_CONFIG": "/dev/null",
        "AWS_REGION": region,
        "AWS_DEFAULT_REGION": region,
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "LOGNAME": "root",
        "PATH": (
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "PYTHONNOUSERSITE": "1",
        "SHELL": "/bin/bash",
        "USER": "root",
        **dict(updates),
    }


def _read_child_request() -> dict[str, Any]:
    payload = sys.stdin.buffer.read(CHILD_REQUEST_MAX_BYTES + 1)
    if not payload or len(payload) > CHILD_REQUEST_MAX_BYTES:
        raise FullParityError("child request is invalid")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise FullParityError("child request is invalid") from exc
    if not isinstance(value, Mapping):
        raise FullParityError("child request is invalid")
    return dict(value)



def _gateway_json(path: str) -> dict[str, Any]:
    with urlopen("http://127.0.0.1:8000" + path, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise FullParityError(f"gateway response is invalid: {path}")
    return value


def _report_document(value: Mapping[str, Any]) -> Mapping[str, Any]:
    report = value.get("report_doc")
    return report if isinstance(report, Mapping) else value


def _parse_gateway_environment_file(path: Path) -> dict[str, str]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FullParityError("gateway environment file is unavailable") from exc
    values: dict[str, str] = {}
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise FullParityError("gateway environment file has an invalid row")
        if "\x00" in line:
            raise FullParityError("gateway environment file has an invalid row")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise FullParityError("gateway environment file has an invalid key")
        if key in values and values[key] != value:
            raise FullParityError(
                "gateway environment file has a conflicting duplicate"
            )
        values[key] = value
    if not values:
        raise FullParityError("gateway environment file is empty")
    return values


def _required_secret_from_environment(
    values: Mapping[str, str],
    references: Sequence[str],
    *,
    field: str,
) -> str:
    for reference in references:
        value = str(values.get(reference) or "").strip()
        if value:
            return value
    raise FullParityError(f"{field} is unavailable in the authorized environment")


def _clone_role_key(
    values: Mapping[str, str],
    *,
    environment_key: str,
    expected_role: str,
    role_label: str,
    candidate_sha: str,
    run_id: str,
    supabase_origin: str,
    jwt_secret: str,
) -> str:
    """Verify one run-scoped 48-hour clone token."""

    boundary = validate_production_parity_boundary_document_v2(
        values,
        network=str(values.get("BITTENSOR_NETWORK") or ""),
        netuid=int(values.get("BITTENSOR_NETUID") or 0),
    )
    normalized_origin = supabase_origin.rstrip("/")
    if (
        boundary.get("mode") != "production-parity"
        or boundary.get("run_id") != run_id
        or boundary.get("supabase_origin") != normalized_origin
        or values.get("LEADPOET_PARITY_CANDIDATE_SHA") != candidate_sha
        or str(values.get("SUPABASE_URL") or "").rstrip("/") != normalized_origin
        or not jwt_secret
    ):
        raise FullParityError(f"run-scoped clone {role_label} identity differs")
    token = _required_secret_from_environment(
        values,
        (environment_key,),
        field=f"run-scoped clone {role_label} credential",
    )
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
        padding = "=" * (-len(encoded_payload) % 4)
        header_padding = "=" * (-len(encoded_header) % 4)
        signature_padding = "=" * (-len(encoded_signature) % 4)
        header = json.loads(
            base64.b64decode(
                encoded_header + header_padding,
                altchars=b"-_",
                validate=True,
            )
        )
        payload = json.loads(
            base64.b64decode(
                encoded_payload + padding,
                altchars=b"-_",
                validate=True,
            )
        )
        signature = base64.b64decode(
            encoded_signature + signature_padding,
            altchars=b"-_",
            validate=True,
        )
        if not isinstance(header, Mapping) or not isinstance(payload, Mapping):
            raise ValueError("JWT documents must be objects")
        if isinstance(payload.get("iat"), bool) or isinstance(
            payload.get("exp"), bool
        ):
            raise ValueError("JWT timestamps must be integers")
        issued_at = int(payload.get("iat"))
        expires_at = int(payload.get("exp"))
    except (binascii.Error, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise FullParityError(
            f"run-scoped clone {role_label} credential is invalid"
        ) from exc
    expected_signature = hmac.new(
        jwt_secret.encode("ascii"),
        f"{encoded_header}.{encoded_payload}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    now = int(time.time())
    if (
        header != {"alg": "HS256", "typ": "JWT"}
        or payload.get("aud") != "authenticated"
        or payload.get("iss") != "leadpoet-production-parity"
        or payload.get("role") != expected_role
        or expires_at - issued_at != 172_805
        or issued_at > now
        or expires_at <= now
        or not hmac.compare_digest(signature, expected_signature)
    ):
        raise FullParityError(
            f"run-scoped clone {role_label} credential identity differs"
        )
    return token


def _clone_service_role_key(
    values: Mapping[str, str],
    *,
    candidate_sha: str,
    run_id: str,
    supabase_origin: str,
    jwt_secret: str,
) -> str:
    """Verify the run-scoped clone service-role token."""

    return _clone_role_key(
        values,
        environment_key="SUPABASE_SERVICE_ROLE_KEY",
        expected_role="service_role",
        role_label="service role",
        candidate_sha=candidate_sha,
        run_id=run_id,
        supabase_origin=supabase_origin,
        jwt_secret=jwt_secret,
    )


def _clone_arena_service_role_key(
    values: Mapping[str, str],
    *,
    candidate_sha: str,
    run_id: str,
    supabase_origin: str,
    jwt_secret: str,
) -> str:
    """Verify the run-scoped clone Arena least-privilege token."""

    return _clone_role_key(
        values,
        environment_key="LAB_ARENA_SERVICE_JWT",
        expected_role="lab_arena_service",
        role_label="Arena service role",
        candidate_sha=candidate_sha,
        run_id=run_id,
        supabase_origin=supabase_origin,
        jwt_secret=jwt_secret,
    )


def _builtwith_key_from_secret(raw: str) -> str:
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise FullParityError("miner-intake secret is invalid") from exc
    key = (
        str(value.get("builtwith_api_key") or "").strip()
        if isinstance(value, Mapping)
        else ""
    )
    if (
        not 8 <= len(key) <= 512
        or any(character.isspace() for character in key)
        or "\x00" in key
    ):
        raise FullParityError("miner-intake secret is invalid")
    return key


def _last_json_document(output: str, *, field: str) -> dict[str, Any]:
    for raw_line in reversed(output.splitlines()):
        try:
            value = json.loads(raw_line)
        except ValueError:
            continue
        if isinstance(value, Mapping):
            return dict(value)
    raise FullParityError(f"{field} did not return redacted JSON evidence")


def _arena_provider_keys(values: Mapping[str, str]) -> dict[str, str]:
    """Resolve the three organizer-held provider credentials without exposing them."""

    return {
        "openrouter": _required_secret_from_environment(
            values,
            (
                "LAB_ARENA_OPENROUTER_API_KEY",
                "RESEARCH_LAB_OPENROUTER_API_KEY",
                "RESEARCH_LAB_V2_OPENROUTER_API_KEY",
                "OPENROUTER_API_KEY",
                "OPENROUTER_KEY",
                "QUALIFICATION_OPENROUTER_API_KEY",
            ),
            field="Arena OpenRouter credential",
        ),
        "scrapingdog": _required_secret_from_environment(
            values,
            (
                "LAB_ARENA_SCRAPINGDOG_API_KEY",
                "RESEARCH_LAB_SCRAPINGDOG_API_KEY",
                "RESEARCH_LAB_V2_SCRAPINGDOG_API_KEY",
                "SCRAPINGDOG_API_KEY",
                "QUALIFICATION_SCRAPINGDOG_API_KEY",
            ),
            field="Arena ScrapingDog credential",
        ),
        "deepline": _required_secret_from_environment(
            values,
            (
                "LAB_ARENA_DEEPLINE_API_KEY",
                "RESEARCH_LAB_DEEPLINE_API_KEY",
                "RESEARCH_LAB_V2_DEEPLINE_API_KEY",
                "DEEPLINE_API_KEY",
            ),
            field="Arena Deepline credential",
        ),
    }


def _verified_arena_runsc_path(*, run_id: str) -> tuple[Path, dict[str, Any]]:
    """Return the exact verified runsc artifact prepared by gateway restart."""

    from gateway.tee.sandbox_runtime_artifact import (
        load_runsc_lock,
        verify_runsc_artifact,
    )

    if not RUN_RE.fullmatch(run_id):
        raise FullParityError("Arena runsc identity is invalid")
    lock_path = ROOT / "gateway" / "tee" / "runsc-runtime.lock.json"
    lock = load_runsc_lock(lock_path)
    artifact = (
        FULL_WORK_ROOT
        / run_id
        / "runtime"
        / "offline-artifacts"
        / str(lock["artifact_filename"])
    )
    try:
        metadata = artifact.lstat()
    except OSError as exc:
        raise FullParityError("Arena runsc artifact is unavailable") from exc
    if (
        artifact.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or not os.access(artifact, os.X_OK)
    ):
        raise FullParityError("Arena runsc artifact identity differs")
    facts = verify_runsc_artifact(
        lock_path=lock_path,
        artifact_path=artifact,
    )
    return artifact, facts


def _arena_https_evidence_urls(company: Mapping[str, Any]) -> set[str]:
    """Return normalized public HTTPS evidence URLs for one company."""

    values = list(company.get("fit_evidence_urls") or [])
    for signal in company.get("intent_signals") or []:
        if isinstance(signal, Mapping):
            values.append(signal.get("url"))
    required = company.get("required_attribute")
    if isinstance(required, Mapping):
        values.append(required.get("evidence_url"))
    urls: set[str] = set()
    for value in values:
        try:
            normalized = public_http_url(value)
        except ValueError:
            continue
        if urlsplit(normalized).scheme == "https":
            urls.add(normalized)
    return urls


def _successful_openrouter_settlement_count(
    settlements: Sequence[Mapping[str, Any]],
) -> int:
    """Count settled OpenRouter calls with one successful provider response."""

    successful = 0
    for settlement in settlements:
        terminal = settlement.get("terminal_response")
        status = terminal.get("status") if isinstance(terminal, Mapping) else None
        if (
            settlement.get("entry_kind") == "settlement"
            and settlement.get("provider") == "openrouter"
            and isinstance(status, int)
            and not isinstance(status, bool)
            and 200 <= status < 300
        ):
            successful += 1
    return successful


def _validate_arena_rebenchmark_evidence(
    value: Mapping[str, Any],
    *,
    candidate_sha: str,
    run_id: str,
    artifact_bucket: str,
) -> dict[str, Any]:
    """Require one complete, live, public-baseline Arena result."""

    from lab_arena import contracts as arena_contracts

    counts = value.get("counts")
    providers = value.get("providers")
    recovery = value.get("restart_recovery")
    runtime = value.get("runtime")
    if not all(
        isinstance(item, Mapping)
        for item in (counts, providers, recovery, runtime)
    ):
        raise FullParityError("Arena rebenchmark evidence is incomplete")
    assert isinstance(counts, Mapping)
    assert isinstance(providers, Mapping)
    assert isinstance(recovery, Mapping)
    assert isinstance(runtime, Mapping)
    configured = counts.get("configured_icp_count")
    stage_1_count = counts.get("stage_1_icp_count")
    stage_2_count = counts.get("stage_2_icp_count")
    numeric_counts = (
        configured,
        stage_1_count,
        stage_2_count,
        counts.get("accepted_execute_runs"),
        counts.get("accepted_score_runs"),
        counts.get("scored_icp_count"),
        counts.get("unique_icp_positions"),
        counts.get("company_count"),
        counts.get("evidence_url_count"),
        providers.get("settled_provider_call_count"),
        providers.get("execute_settled_provider_call_count"),
        providers.get("score_settled_provider_call_count"),
        providers.get("successful_openrouter_execute_call_count"),
        providers.get("successful_openrouter_score_settlement_count"),
    )
    provider_names = providers.get("names")
    evaluation_date = str(value.get("evaluation_date") or "")
    try:
        final_score = float(value.get("baseline_final_score"))
        set_id = int(value.get("daily_icp_set_id"))
    except (TypeError, ValueError) as exc:
        raise FullParityError("Arena rebenchmark evidence is incomplete") from exc
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in numeric_counts
    ):
        raise FullParityError("Arena rebenchmark evidence is incomplete")
    assert isinstance(configured, int)
    icp_results = value.get("icp_results")
    if not isinstance(icp_results, list) or len(icp_results) != configured:
        raise FullParityError("Arena rebenchmark evidence is incomplete")
    per_icp_count_fields = (
        "company_count",
        "valid_company_with_https_evidence_count",
        "https_evidence_url_count",
        "successful_openrouter_execute_call_count",
        "successful_openrouter_score_settlement_count",
    )
    positions: set[int] = set()
    for result in icp_results:
        if not isinstance(result, Mapping):
            raise FullParityError("Arena rebenchmark evidence is incomplete")
        position = result.get("icp_position")
        per_icp_counts = tuple(result.get(name) for name in per_icp_count_fields)
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or result.get("execute_accepted") is not True
            or result.get("score_accepted") is not True
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in per_icp_counts
            )
        ):
            raise FullParityError("Arena rebenchmark evidence is incomplete")
        positions.add(position)
    if positions != set(range(configured)):
        raise FullParityError("Arena rebenchmark evidence is incomplete")
    if (
        value.get("schema_version")
        != ARENA_REBENCHMARK_EVIDENCE_SCHEMA_VERSION
        or value.get("candidate_sha") != candidate_sha
        or value.get("run_id") != run_id
        or value.get("artifact_bucket") != artifact_bucket
        or value.get("status") != "passed"
        or value.get("mode") != "shadow"
        or value.get("baseline_source_url") != ARENA_BASELINE_SOURCE_URL
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", evaluation_date) is None
        or str(set_id) != evaluation_date.replace("-", "")
        or re.fullmatch(
            r"arena-\d{4}-\d{2}-\d{2}-[a-z0-9]{1,16}",
            str(value.get("round_id") or ""),
        )
        is None
        or stage_1_count != len(arena_contracts.stage_positions(1))
        or stage_2_count != len(arena_contracts.stage_positions(2))
        or configured != stage_1_count + stage_2_count
        or configured != arena_contracts.BENCHMARK_ICP_COUNT
        or any(counts.get(name) != configured for name in (
            "accepted_execute_runs",
            "accepted_score_runs",
            "scored_icp_count",
            "unique_icp_positions",
        ))
        or int(counts.get("company_count") or 0) < 1
        or int(counts.get("evidence_url_count") or 0) < 1
        or not math.isfinite(final_score)
        or not 0.0 <= final_score <= 100.0
        or not isinstance(provider_names, list)
        or any(not isinstance(name, str) for name in provider_names)
        or "openrouter" not in provider_names
        or int(providers.get("settled_provider_call_count") or 0) < 2
        or int(providers.get("execute_settled_provider_call_count") or 0) < 1
        or int(providers.get("score_settled_provider_call_count") or 0) < 1
        or int(providers.get("successful_openrouter_execute_call_count") or 0)
        < configured
        or int(
            providers.get("successful_openrouter_score_settlement_count") or 0
        )
        < configured
        or providers.get("transport") != "live-httpx"
        or runtime.get("runner") != "lab_arena.runner.Runner"
        or runtime.get("sandbox") != "gvisor-runsc"
        or runtime.get("api") != "lab_arena.api.loopback-http"
        or runtime.get("object_store") != "s3"
        or runtime.get("judge_image_materialization")
        != "exact-candidate-local-docker"
        or recovery.get("service_restarted") is not True
        or recovery.get("runner_restarted") is not True
        or recovery.get("resumed_round_status") != "stage1"
        or recovery.get("persisted_execute_runs") != stage_1_count
        or value.get("publication_visible") is not True
        or value.get("public_benchmark_visible") is not True
        or value.get("public_results_visible") is not True
        or value.get("production_database_mutated") is not False
        or value.get("production_chain_mutated") is not False
    ):
        raise FullParityError("Arena rebenchmark evidence is incomplete")
    return dict(value)


def _run_arena_rebenchmark_path(
    *,
    region: str,
    candidate_sha: str,
    run_id: str,
    supabase_origin: str,
    gateway_env_file: Path,
    artifact_bucket: str,
    jwt_secret: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run the public baseline through the real Arena on the disposable clone."""

    if region != "us-east-1" or timeout_seconds <= 0:
        raise FullParityError("Arena rebenchmark request is invalid")
    values = _validated_clone_environment(
        gateway_env_file,
        candidate_sha=candidate_sha,
        run_id=run_id,
        supabase_origin=supabase_origin,
        artifact_bucket=artifact_bucket,
    )
    _clone_arena_service_role_key(
        values,
        candidate_sha=candidate_sha,
        run_id=run_id,
        supabase_origin=supabase_origin,
        jwt_secret=jwt_secret,
    )
    runsc_path, runsc_facts = _verified_arena_runsc_path(run_id=run_id)
    request = {
        "schema_version": ARENA_REBENCHMARK_REQUEST_SCHEMA_VERSION,
        "candidate_sha": candidate_sha,
        "run_id": run_id,
        "artifact_bucket": artifact_bucket,
        "region": region,
        "supabase_origin": supabase_origin,
        "gateway_env_file": str(gateway_env_file),
        "runsc_path": str(runsc_path),
        "runsc_artifact_hash": runsc_facts["artifact_hash"],
        "timeout_seconds": min(timeout_seconds, MAX_FULL_TIMEOUT_SECONDS),
    }
    result = _run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--arena-rebenchmark-child",
        ],
        timeout=min(timeout_seconds, MAX_FULL_TIMEOUT_SECONDS),
        env=_clone_child_environment(region=region),
        input_text=json.dumps(request, separators=(",", ":")),
    )
    request.clear()
    if (
        result.returncode != 0
        or len(result.stdout or "") > 256 * 1024
        or len(result.stderr or "") > 64 * 1024
    ):
        raise FullParityError("Arena rebenchmark child failed closed")
    evidence = _last_json_document(
        result.stdout or "", field="Arena rebenchmark child"
    )
    return _validate_arena_rebenchmark_evidence(
        evidence,
        candidate_sha=candidate_sha,
        run_id=run_id,
        artifact_bucket=artifact_bucket,
    )


class _ParityArenaChainReads:
    """The shadow run needs runner ownership checks, not chain authority."""

    def __init__(self, runner_hotkey: str) -> None:
        self._runner_hotkey = runner_hotkey

    def finalized_head(self) -> Any:
        raise RuntimeError("chain reads are disabled in production parity")

    def metagraph(self, finalized: bool = True) -> Any:
        del finalized
        raise RuntimeError("chain reads are disabled in production parity")

    def current_settlement_epoch(self) -> int:
        raise RuntimeError("chain reads are disabled in production parity")

    def hotkeys_owned_by_same_coldkey(self, hotkey: str) -> list[str]:
        return [hotkey] if hotkey == self._runner_hotkey else []

    def uid_for_hotkey(self, hotkey: str) -> None:
        del hotkey
        return None


class _ArenaApiServer:
    """Run the candidate Arena FastAPI app on one loopback HTTP socket."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self._server: Any | None = None
        self._socket: Any | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    def start(self) -> str:
        import socket

        import uvicorn

        if self._server is not None:
            raise FullParityError("Arena API server already started")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(2048)
        port = int(listener.getsockname()[1])
        server = uvicorn.Server(
            uvicorn.Config(
                self._app,
                host="127.0.0.1",
                port=port,
                access_log=False,
                log_level="critical",
                lifespan="off",
            )
        )
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [listener]},
            name="production-parity-arena-api",
            daemon=True,
        )
        self._server = server
        self._socket = listener
        self._thread = thread
        self.base_url = f"http://127.0.0.1:{port}"
        thread.start()
        deadline = time.monotonic() + 30
        opener = build_opener(ProxyHandler({}))
        while time.monotonic() < deadline:
            if not thread.is_alive():
                break
            try:
                with opener.open(
                    self.base_url + "/arena/v1/current", timeout=2
                ) as response:
                    if int(response.status) == 200:
                        return self.base_url
            except (OSError, TimeoutError, URLError):
                time.sleep(0.1)
        self.stop()
        raise FullParityError("Arena API server did not become ready")

    def stop(self) -> None:
        server = self._server
        listener = self._socket
        thread = self._thread
        if server is None:
            return
        server.should_exit = True
        if thread is not None:
            thread.join(timeout=30)
        try:
            if listener is not None:
                listener.close()
        finally:
            self._server = None
            self._socket = None
            self._thread = None
        if thread is not None and thread.is_alive():
            raise FullParityError("Arena API server did not stop")


def _build_arena_judge_rootfs(
    *,
    candidate_sha: str,
    run_id: str,
    arena_root: Path,
) -> dict[str, Any]:
    """Build the exact candidate judge and export its merged root filesystem."""

    suffix = hashlib.sha256(run_id.encode("ascii")).hexdigest()[:12]
    image_tag = f"leadpoet-parity-judge:{candidate_sha[:12]}-{suffix}"
    container_name = f"leadpoet-parity-judge-{suffix}"
    build_log = arena_root / "judge-build.log"
    build = _run(
        [
            "docker",
            "build",
            "--pull",
            "--platform",
            "linux/amd64",
            "--file",
            str(ROOT / "lab_arena" / "judge" / "Dockerfile"),
            "--build-arg",
            f"LEADPOET_BUILD_COMMIT={candidate_sha}",
            "--tag",
            image_tag,
            str(ROOT),
        ],
        timeout=7200,
        log_path=build_log,
    )
    _require(build, stage="exact-candidate Arena judge build")
    inspect_result = _run(
        ["docker", "image", "inspect", "--format={{.Id}}", image_tag],
        timeout=60,
    )
    image_digest = _require(
        inspect_result, stage="exact-candidate Arena judge identity"
    ).strip()
    if HASH_RE.fullmatch(image_digest) is None:
        raise FullParityError("exact-candidate Arena judge identity differs")
    image_reference = (
        "localhost/leadpoet/production-parity-judge@" + image_digest
    )
    image_root = arena_root / "runner" / "images"
    cache_target = image_root / ("sha256-" + image_digest.removeprefix("sha256:"))
    rootfs = cache_target / "rootfs"
    archive = arena_root / "judge-rootfs.tar"
    image_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    cache_target.mkdir(mode=0o700)
    rootfs.mkdir(mode=0o700)
    created = False
    try:
        _require(
            _run(
                ["docker", "create", "--name", container_name, image_tag],
                timeout=60,
            ),
            stage="exact-candidate Arena judge container",
        )
        created = True
        _require(
            _run(
                [
                    "docker",
                    "export",
                    "--output",
                    str(archive),
                    container_name,
                ],
                timeout=1800,
            ),
            stage="exact-candidate Arena judge export",
        )
        _require(
            _run(
                [
                    "tar",
                    "--extract",
                    "--file",
                    str(archive),
                    "--directory",
                    str(rootfs),
                    "--numeric-owner",
                ],
                timeout=1800,
            ),
            stage="exact-candidate Arena judge rootfs",
        )
    finally:
        archive.unlink(missing_ok=True)
        if created:
            _run(
                ["docker", "container", "rm", "--force", container_name],
                timeout=60,
            )
    if (
        not (rootfs / "model" / "scorer_entrypoint.py").is_file()
        or not (rootfs / "usr" / "local" / "bin" / "python3").exists()
    ):
        raise FullParityError("exact-candidate Arena judge rootfs is incomplete")
    marker = cache_target / ".exported"
    marker.write_text(image_digest, encoding="ascii")
    marker.chmod(0o600)
    return {
        "image_digest": image_digest,
        "image_reference": image_reference,
        "image_tag": image_tag,
        "image_cache_root": image_root,
    }


def _remove_arena_judge_image(image_tag: str) -> None:
    if image_tag:
        _run(
            ["docker", "image", "rm", "--force", image_tag],
            timeout=120,
        )


def _build_parity_arena_service(
    *,
    values: Mapping[str, str],
    artifact_bucket: str,
    region: str,
    runner_hotkey: str,
    baseline_hotkey: str,
    scorer_image_digest: str,
    scorer_image_reference: str,
    provider_keys: Mapping[str, str],
    price_table: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Build the production Arena service against only disposable state."""

    from lab_arena import broker as broker_module
    from lab_arena import chain as chain_module
    from lab_arena.api import create_app
    from lab_arena.service import (
        ArenaService,
        RoundDefaults,
        S3ObjectStore,
        ServiceConfig,
    )
    from lab_arena.store import ArenaStore, PostgrestTransport
    from lab_arena.wiring import fetch_public_source_archive

    transport = PostgrestTransport(
        str(values["LAB_ARENA_SUPABASE_URL"]),
        anon_key=str(values["LAB_ARENA_SUPABASE_ANON_KEY"]),
        service_jwt=str(values["LAB_ARENA_SERVICE_JWT"]),
        timeout_seconds=30,
    )
    store = ArenaStore(transport)
    object_store = S3ObjectStore(artifact_bucket, region_name=region)

    def key_for(provider: str) -> str:
        secret = str(provider_keys.get(provider) or "")
        if not secret:
            raise broker_module.BrokerError("broker_unavailable")
        return secret

    def broker_factory(
        service: Any, round_row: Mapping[str, Any]
    ) -> Any:
        del round_row
        judge_models = sorted(
            {
                str(model)
                for model in (
                    service.scorer_policy.get("judge_models") or {}
                ).values()
                if model
            }
        )
        return broker_module.Broker(
            store=store,
            key_for=key_for,
            judge_models=judge_models,
            price_table=price_table,
            transport=broker_module.HttpxProviderTransport(),
        )

    def daily_icp_source(*, set_id: int, active_at: datetime) -> Mapping[str, Any]:
        del active_at
        return store.current_daily_icp_set(set_id)

    service = ArenaService(
        ServiceConfig(
            mode="shadow",
            store=store,
            object_store=object_store,
            signer=None,
            chain=_ParityArenaChainReads(runner_hotkey),
            verify_signature=chain_module.verify_hotkey_signature,
            daily_icp_source=daily_icp_source,
            banned_hotkeys_source=lambda: (),
            broker_factory=broker_factory,
            defaults=RoundDefaults(
                runner_hotkeys=(runner_hotkey,),
                baseline_hotkey=baseline_hotkey,
                baseline_source_url=ARENA_BASELINE_SOURCE_URL,
                scorer_image_digest=scorer_image_digest,
                scorer_image_reference=scorer_image_reference,
                daily_cutoff_hour_utc=None,
                rewards_enabled=False,
            ),
            baseline_source_fetcher=fetch_public_source_archive,
            reward_signer_factory=None,
        )
    )
    return service, create_app(service)


def _build_parity_arena_runner(
    *,
    api_base_url: str,
    round_id: str,
    evaluation_date: str,
    runner_keypair: Any,
    runsc_path: Path,
    runner_root: Path,
) -> tuple[Any, Any]:
    """Build the real signed Arena runner with the verified runsc runtime."""

    from lab_arena import runner as runner_module
    from lab_arena import runtime as runtime_module

    sandboxes = runner_root / "sandboxes"
    runs = runner_root / "runs"
    images = runner_root / "images"
    sources = runner_root / "sources"
    for directory in (runner_root, sandboxes, runs, images, sources):
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise FullParityError("Arena runner work directory is unsafe")
        directory.chmod(0o700)

    def refuse_image_export(
        image_reference: str, image_digest: str, target_dir: Path
    ) -> None:
        del image_reference, image_digest, target_dir
        raise runner_module.RunnerError(
            "exact-candidate Arena judge rootfs is not preloaded"
        )

    api = runner_module.HttpArenaApiClient(api_base_url)
    api.round(round_id)
    cache = runner_module.ImageCache(images, refuse_image_export)
    source_cache = runner_module.SourceCache(sources, api.source)
    sandbox_runtime = runtime_module.RunscRuntime(
        runtime_module.RuntimeConfig(
            runsc_path=runsc_path,
            work_dir=sandboxes,
        )
    )
    identity = runner_module.RunnerIdentity(
        hotkey=runner_keypair.ss58_address,
        sign=lambda message: runner_keypair.sign(
            message.encode("utf-8")
        ).hex(),
    )
    config = runner_module.RunnerConfig(
        round_id=round_id,
        identity=identity,
        api=api,
        sandbox_runtime=sandbox_runtime,
        image_cache=cache,
        source_cache=source_cache,
        work_dir=runs,
        max_parallel_runs=8,
        evaluation_date=evaluation_date,
        socket_root=Path("/tmp"),
    )
    return runner_module.Runner(config), api


def _drain_arena_assignments(
    *,
    service: Any,
    runner: Any,
    round_id: str,
    stage: int,
    kind: str,
    deadline: float,
) -> list[dict[str, Any]]:
    """Run until every configured position has one accepted real result."""

    from lab_arena import contracts as arena_contracts

    expected_positions = set(arena_contracts.stage_positions(stage))
    while True:
        rows = service.store.list_runs(round_id, stage=stage, kind=kind)
        if rows and all(
            row.get("status") in {"accepted", "failed"} for row in rows
        ):
            break
        if time.monotonic() >= deadline:
            raise FullParityError("Arena assignment deadline expired")
        service.store.expire_leases(round_id)
        taken = runner.run_once(max_claims=1000)
        if taken == 0:
            time.sleep(2)
    accepted = [row for row in rows if row.get("status") == "accepted"]
    accepted_positions = {int(row["icp_position"]) for row in accepted}
    if (
        accepted_positions != expected_positions
        or len(accepted) != len(expected_positions)
        or any(
            row.get("status") in {"pending", "leased", "submitted"}
            for row in rows
        )
    ):
        raise FullParityError(
            f"Arena {kind} did not accept every configured ICP"
        )
    return [dict(row) for row in accepted]


def _require_arena_round_status(
    service: Any, round_id: str, expected: str
) -> dict[str, Any]:
    row = service.store.get_round(round_id)
    if not isinstance(row, Mapping) or row.get("status") != expected:
        raise FullParityError(f"Arena round did not reach {expected}")
    return dict(row)


def _arena_public_json(base_url: str, path: str) -> dict[str, Any]:
    opener = build_opener(ProxyHandler({}), _RejectCloneRedirects())
    try:
        with opener.open(base_url.rstrip("/") + path, timeout=30) as response:
            if int(response.status) != 200:
                raise FullParityError("Arena public result is unavailable")
            value = json.load(response)
    except (HTTPError, OSError, TimeoutError, URLError, ValueError) as exc:
        raise FullParityError("Arena public result is unavailable") from exc
    if not isinstance(value, Mapping):
        raise FullParityError("Arena public result is invalid")
    return dict(value)


def _close_parity_arena_runtime(
    *,
    runner: Any | None,
    api_client: Any | None,
    api_server: _ArenaApiServer | None,
    service: Any | None,
) -> None:
    errors = []
    for resource, method in (
        (runner, "close"),
        (api_client, "close"),
        (api_server, "stop"),
    ):
        if resource is None:
            continue
        try:
            getattr(resource, method)()
        except Exception as exc:  # noqa: BLE001 - child reports a fixed error only
            errors.append(type(exc).__name__)
    if service is not None:
        try:
            service.store.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(type(exc).__name__)
    if errors:
        raise FullParityError("Arena runtime cleanup failed")


def _run_arena_rebenchmark_child(
    request: Mapping[str, Any]
) -> dict[str, Any]:
    """Execute one complete public-baseline daily Arena round on the clone."""

    if (
        request.get("schema_version")
        != ARENA_REBENCHMARK_REQUEST_SCHEMA_VERSION
        or not SHA_RE.fullmatch(str(request.get("candidate_sha") or ""))
        or not RUN_RE.fullmatch(str(request.get("run_id") or ""))
        or not ARTIFACT_BUCKET_RE.fullmatch(
            str(request.get("artifact_bucket") or "")
        )
        or request.get("region") != "us-east-1"
        or isinstance(request.get("timeout_seconds"), bool)
        or not isinstance(request.get("timeout_seconds"), int)
        or not 1 <= int(request.get("timeout_seconds") or 0) <= MAX_FULL_TIMEOUT_SECONDS
    ):
        raise FullParityError("Arena rebenchmark child request is invalid")
    candidate_sha = str(request["candidate_sha"])
    run_id = str(request["run_id"])
    artifact_bucket = str(request["artifact_bucket"])
    region = str(request["region"])
    supabase_origin = str(request.get("supabase_origin") or "")
    gateway_env_file = Path(str(request.get("gateway_env_file") or ""))
    arena_root = FULL_WORK_ROOT / run_id / "runtime" / "arena"
    if arena_root.exists() or arena_root.is_symlink():
        raise FullParityError("Arena work directory already exists")
    values = _validated_clone_environment(
        gateway_env_file,
        candidate_sha=candidate_sha,
        run_id=run_id,
        supabase_origin=supabase_origin,
        artifact_bucket=artifact_bucket,
    )
    runsc_path, runsc_facts = _verified_arena_runsc_path(run_id=run_id)
    if (
        str(request.get("runsc_path") or "") != str(runsc_path)
        or request.get("runsc_artifact_hash") != runsc_facts["artifact_hash"]
    ):
        raise FullParityError("Arena runsc child identity differs")
    provider_keys = _arena_provider_keys(values)
    arena_root.mkdir(mode=0o700)
    deadline = time.monotonic() + int(request["timeout_seconds"])
    image: dict[str, Any] = {}
    service = runner = api_client = api_server = None
    try:
        image = _build_arena_judge_rootfs(
            candidate_sha=candidate_sha,
            run_id=run_id,
            arena_root=arena_root,
        )
        from bittensor_wallet import Keypair
        from lab_arena import broker as broker_module
        from lab_arena import contracts as arena_contracts

        runner_keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        baseline_keypair = Keypair.create_from_mnemonic(
            Keypair.generate_mnemonic()
        )
        price_table = broker_module.fetch_openrouter_price_table()
        service, app = _build_parity_arena_service(
            values=values,
            artifact_bucket=artifact_bucket,
            region=region,
            runner_hotkey=runner_keypair.ss58_address,
            baseline_hotkey=baseline_keypair.ss58_address,
            scorer_image_digest=str(image["image_digest"]),
            scorer_image_reference=str(image["image_reference"]),
            provider_keys=provider_keys,
            price_table=price_table,
        )
        startup = service.startup_checks()
        now = datetime.now(timezone.utc)
        evaluation_date = now.date().isoformat()
        set_id = int(now.strftime("%Y%m%d"))
        daily = service.store.current_daily_icp_set(set_id)
        raw_icps = daily.get("icps") if isinstance(daily, Mapping) else None
        icps = (
            [dict(item) for item in raw_icps if isinstance(item, Mapping)]
            if isinstance(raw_icps, list)
            else []
        )
        icp_ids = [str(item.get("icp_id") or "") for item in icps]
        if (
            startup.get("database_identity", {}).get("current_user")
            != "lab_arena_service"
            or daily.get("status") != "ready"
            or int(daily.get("set_id") or 0) != set_id
            or len(icps) != arena_contracts.BENCHMARK_ICP_COUNT
            or len(icps) != len(raw_icps or [])
            or any(not item for item in icp_ids)
            or len(set(icp_ids)) != len(icp_ids)
            or datetime.now(timezone.utc).date().isoformat() != evaluation_date
        ):
            raise FullParityError("latest clone daily ICP set is not ready")
        suffix = hashlib.sha256(
            f"{run_id}:{candidate_sha}".encode("ascii")
        ).hexdigest()[:12]
        round_id = f"arena-{evaluation_date}-{suffix}"
        configuration = service.create_round(now, round_id=round_id)
        _require_arena_round_status(service, round_id, "open")
        stage_counts = {
            1: int(configuration["stage_1_icp_count"]),
            2: int(configuration["stage_2_icp_count"]),
        }
        configured_count = stage_counts[1] + stage_counts[2]
        expected_positions = {
            stage: set(arena_contracts.stage_positions(stage))
            for stage in (1, 2)
        }
        if (
            configured_count != arena_contracts.BENCHMARK_ICP_COUNT
            or any(
                len(expected_positions[stage]) != stage_counts[stage]
                for stage in (1, 2)
            )
        ):
            raise FullParityError("Arena round ICP configuration differs")
        committed = service.commit_benchmark(round_id)
        if int(committed.get("participants") or 0) != 1:
            raise FullParityError("Arena baseline was not the sole participant")
        round_row = _require_arena_round_status(service, round_id, "committed")
        participants = list(round_row.get("participants") or [])
        if len(participants) != 1 or participants[0].get("is_king") is not True:
            raise FullParityError("Arena baseline participant is invalid")
        baseline_submission_id = str(participants[0].get("submission_id") or "")

        opened_stage1 = service.open_stage(round_id, 1)
        if int(opened_stage1.get("assignments") or 0) != stage_counts[1]:
            raise FullParityError("Arena stage 1 assignments differ")
        _require_arena_round_status(service, round_id, "stage1")
        api_server = _ArenaApiServer(app)
        api_base_url = api_server.start()
        runner, api_client = _build_parity_arena_runner(
            api_base_url=api_base_url,
            round_id=round_id,
            evaluation_date=evaluation_date,
            runner_keypair=runner_keypair,
            runsc_path=runsc_path,
            runner_root=arena_root / "runner",
        )
        stage1_execute = _drain_arena_assignments(
            service=service,
            runner=runner,
            round_id=round_id,
            stage=1,
            kind="execute",
            deadline=deadline,
        )
        persisted_execute_runs = len(stage1_execute)

        _close_parity_arena_runtime(
            runner=runner,
            api_client=api_client,
            api_server=api_server,
            service=service,
        )
        service = runner = api_client = api_server = None

        service, app = _build_parity_arena_service(
            values=values,
            artifact_bucket=artifact_bucket,
            region=region,
            runner_hotkey=runner_keypair.ss58_address,
            baseline_hotkey=baseline_keypair.ss58_address,
            scorer_image_digest=str(image["image_digest"]),
            scorer_image_reference=str(image["image_reference"]),
            provider_keys=provider_keys,
            price_table=price_table,
        )
        service.startup_checks()
        _require_arena_round_status(service, round_id, "stage1")
        recovered = service.store.list_runs(
            round_id, stage=1, status="accepted", kind="execute"
        )
        if len(recovered) != persisted_execute_runs:
            raise FullParityError("Arena restart did not recover stage results")
        api_server = _ArenaApiServer(app)
        api_base_url = api_server.start()
        runner, api_client = _build_parity_arena_runner(
            api_base_url=api_base_url,
            round_id=round_id,
            evaluation_date=evaluation_date,
            runner_keypair=runner_keypair,
            runsc_path=runsc_path,
            runner_root=arena_root / "runner",
        )

        service.close_stage(round_id, 1)
        _require_arena_round_status(service, round_id, "stage1_closed")
        opened_scoring = service.open_scoring(round_id, 1)
        if int(opened_scoring.get("assignments") or 0) != stage_counts[1]:
            raise FullParityError("Arena stage 1 scoring assignments differ")
        _require_arena_round_status(service, round_id, "stage1_scoring")
        stage1_score = _drain_arena_assignments(
            service=service,
            runner=runner,
            round_id=round_id,
            stage=1,
            kind="score",
            deadline=deadline,
        )
        service.close_scoring(round_id, 1)
        _require_arena_round_status(service, round_id, "stage1_judged")
        service.score_stage(round_id, 1)
        _require_arena_round_status(service, round_id, "stage1_scored")

        opened_stage2 = service.open_stage(round_id, 2)
        if int(opened_stage2.get("assignments") or 0) != stage_counts[2]:
            raise FullParityError("Arena stage 2 assignments differ")
        _require_arena_round_status(service, round_id, "stage2")
        stage2_execute = _drain_arena_assignments(
            service=service,
            runner=runner,
            round_id=round_id,
            stage=2,
            kind="execute",
            deadline=deadline,
        )
        service.close_stage(round_id, 2)
        _require_arena_round_status(service, round_id, "stage2_closed")
        opened_scoring = service.open_scoring(round_id, 2)
        if int(opened_scoring.get("assignments") or 0) != stage_counts[2]:
            raise FullParityError("Arena stage 2 scoring assignments differ")
        _require_arena_round_status(service, round_id, "stage2_scoring")
        stage2_score = _drain_arena_assignments(
            service=service,
            runner=runner,
            round_id=round_id,
            stage=2,
            kind="score",
            deadline=deadline,
        )
        service.close_scoring(round_id, 2)
        _require_arena_round_status(service, round_id, "stage2_judged")
        service.score_stage(round_id, 2)
        _require_arena_round_status(service, round_id, "scored")
        service.publish(round_id)
        _require_arena_round_status(service, round_id, "published")

        round_view = _arena_public_json(
            api_base_url, f"/arena/v1/rounds/{round_id}"
        )
        benchmark_view = _arena_public_json(
            api_base_url, f"/arena/v1/rounds/{round_id}/benchmark"
        )
        results_view = _arena_public_json(
            api_base_url,
            f"/arena/v1/rounds/{round_id}/results/{baseline_submission_id}",
        )
        benchmark_icps = benchmark_view.get("icps")
        benchmark_ids = (
            [str(item.get("icp_id") or "") for item in benchmark_icps]
            if isinstance(benchmark_icps, list)
            and all(isinstance(item, Mapping) for item in benchmark_icps)
            else []
        )
        outputs = results_view.get("outputs")
        score_doc = results_view.get("scores")
        stage1_scores = (
            score_doc.get("stage_1") if isinstance(score_doc, Mapping) else None
        )
        stage2_scores = (
            score_doc.get("stage_2") if isinstance(score_doc, Mapping) else None
        )
        if (
            round_view.get("status") != "published"
            or not isinstance(round_view.get("publication"), Mapping)
            or benchmark_ids != icp_ids
            or not isinstance(outputs, Mapping)
            or len(outputs) != configured_count
            or not isinstance(stage1_scores, list)
            or not isinstance(stage2_scores, list)
            or len(stage1_scores) + len(stage2_scores) != configured_count
            or results_view.get("submission", {}).get("is_baseline") is not True
        ):
            raise FullParityError("Arena public daily result is incomplete")
        all_execute = stage1_execute + stage2_execute
        all_score = stage1_score + stage2_score
        positions = {int(row["icp_position"]) for row in all_execute}
        score_positions = {int(row["icp_position"]) for row in all_score}
        all_expected_positions = expected_positions[1] | expected_positions[2]
        if (
            positions != all_expected_positions
            or score_positions != all_expected_positions
        ):
            raise FullParityError("Arena did not cover every configured ICP")

        execute_by_position = {
            int(row["icp_position"]): row for row in all_execute
        }
        score_by_position = {
            int(row["icp_position"]): row for row in all_score
        }
        company_count = 0
        evidence_urls: set[str] = set()
        execute_settlements = 0
        score_settlements = 0
        successful_openrouter_execute_calls = 0
        successful_openrouter_score_settlements = 0
        provider_names: set[str] = set()
        icp_results = []
        for position in sorted(all_expected_positions):
            execute_row = execute_by_position[position]
            score_row = score_by_position[position]
            if score_row.get("scored_run_id") != execute_row.get("run_id"):
                raise FullParityError("Arena score does not match its execute run")
            output = outputs.get(str(execute_row["run_id"]))
            companies = (
                output.get("companies") if isinstance(output, Mapping) else None
            )
            if not isinstance(companies, list):
                raise FullParityError("Arena public company output is invalid")
            try:
                validated_companies = validate_companies(
                    companies,
                    max_companies=max(1, len(companies)),
                )
            except ValueError as exc:
                raise FullParityError(
                    "Arena public company output is invalid"
                ) from exc
            company_count += len(validated_companies)
            position_evidence_urls: set[str] = set()
            valid_companies_with_https_evidence = 0
            for company in validated_companies:
                valid_company_urls = _arena_https_evidence_urls(company)
                if valid_company_urls:
                    valid_companies_with_https_evidence += 1
                    position_evidence_urls.update(valid_company_urls)
            evidence_urls.update(position_evidence_urls)

            execute_ledger = service.store.list_ledger(
                run_id=str(execute_row["run_id"])
            )
            score_ledger = service.store.list_ledger(
                run_id=str(score_row["run_id"])
            )
            execute_position_settlements = [
                item
                for item in execute_ledger
                if item.get("entry_kind") == "settlement"
            ]
            score_position_settlements = [
                item
                for item in score_ledger
                if item.get("entry_kind") == "settlement"
            ]
            execute_settlements += len(execute_position_settlements)
            score_settlements += len(score_position_settlements)
            provider_names.update(
                str(item.get("provider") or "")
                for item in execute_position_settlements + score_position_settlements
                if item.get("provider")
            )

            execute_openrouter_successes = _successful_openrouter_settlement_count(
                execute_position_settlements
            )
            score_openrouter_successes = _successful_openrouter_settlement_count(
                score_position_settlements
            )
            successful_openrouter_execute_calls += execute_openrouter_successes
            successful_openrouter_score_settlements += score_openrouter_successes
            icp_results.append(
                {
                    "icp_position": position,
                    "execute_accepted": execute_row.get("status") == "accepted",
                    "score_accepted": score_row.get("status") == "accepted",
                    "company_count": len(validated_companies),
                    "valid_company_with_https_evidence_count": (
                        valid_companies_with_https_evidence
                    ),
                    "https_evidence_url_count": len(position_evidence_urls),
                    "successful_openrouter_execute_call_count": (
                        execute_openrouter_successes
                    ),
                    "successful_openrouter_score_settlement_count": (
                        score_openrouter_successes
                    ),
                }
            )
        final_score = (
            results_view.get("submission_scores", {}).get("final")
            if isinstance(results_view.get("submission_scores"), Mapping)
            else None
        )
        evidence = {
            "schema_version": ARENA_REBENCHMARK_EVIDENCE_SCHEMA_VERSION,
            "candidate_sha": candidate_sha,
            "run_id": run_id,
            "artifact_bucket": artifact_bucket,
            "status": "passed",
            "mode": "shadow",
            "round_id": round_id,
            "evaluation_date": evaluation_date,
            "daily_icp_set_id": set_id,
            "baseline_source_url": ARENA_BASELINE_SOURCE_URL,
            "baseline_final_score": final_score,
            "icp_results": icp_results,
            "counts": {
                "configured_icp_count": len(icps),
                "stage_1_icp_count": stage_counts[1],
                "stage_2_icp_count": stage_counts[2],
                "accepted_execute_runs": len(all_execute),
                "accepted_score_runs": len(all_score),
                "scored_icp_count": len(stage1_scores) + len(stage2_scores),
                "unique_icp_positions": len(positions),
                "company_count": company_count,
                "evidence_url_count": len(evidence_urls),
            },
            "providers": {
                "transport": "live-httpx",
                "names": sorted(provider_names),
                "settled_provider_call_count": (
                    execute_settlements + score_settlements
                ),
                "execute_settled_provider_call_count": execute_settlements,
                "score_settled_provider_call_count": score_settlements,
                "successful_openrouter_execute_call_count": (
                    successful_openrouter_execute_calls
                ),
                "successful_openrouter_score_settlement_count": (
                    successful_openrouter_score_settlements
                ),
            },
            "runtime": {
                "runner": "lab_arena.runner.Runner",
                "sandbox": "gvisor-runsc",
                "runsc_artifact_hash": runsc_facts["artifact_hash"],
                "api": "lab_arena.api.loopback-http",
                "object_store": "s3",
                "judge_image_materialization": "exact-candidate-local-docker",
            },
            "restart_recovery": {
                "service_restarted": True,
                "runner_restarted": True,
                "resumed_round_status": "stage1",
                "persisted_execute_runs": persisted_execute_runs,
            },
            "publication_visible": True,
            "public_benchmark_visible": True,
            "public_results_visible": True,
            "production_database_mutated": False,
            "production_chain_mutated": False,
        }
        return _validate_arena_rebenchmark_evidence(
            evidence,
            candidate_sha=candidate_sha,
            run_id=run_id,
            artifact_bucket=artifact_bucket,
        )
    finally:
        provider_keys = {}
        cleanup_error = None
        try:
            _close_parity_arena_runtime(
                runner=runner,
                api_client=api_client,
                api_server=api_server,
                service=service,
            )
        except Exception as exc:  # noqa: BLE001 - child prints only fixed text
            cleanup_error = exc
        _remove_arena_judge_image(str(image.get("image_tag") or ""))
        if cleanup_error is not None:
            raise cleanup_error


def _run_miner_intake_path(
    *,
    region: str,
    candidate_sha: str,
    run_id: str,
    supabase_origin: str,
    gateway_env_file: Path,
    artifact_bucket: str,
    miner_intake_secret: str,
) -> dict[str, Any]:
    builtwith_credential = _builtwith_key_from_secret(miner_intake_secret)
    _validated_clone_environment(
        gateway_env_file,
        candidate_sha=candidate_sha,
        run_id=run_id,
        supabase_origin=supabase_origin,
        artifact_bucket=artifact_bucket,
    )
    child_env = {
        **_clone_child_environment(region=region),
        **MINER_INTAKE_ENVIRONMENT_OVERRIDES,
    }
    request = {
        "schema_version": "leadpoet.production_parity_miner_intake_request.v1",
        "candidate_sha": candidate_sha,
        "run_id": run_id,
        "artifact_bucket": artifact_bucket,
        "region": region,
        "supabase_origin": supabase_origin,
        "gateway_env_file": str(gateway_env_file),
        "builtwith_credential": builtwith_credential,
    }
    result = _run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--miner-intake-child",
        ],
        timeout=360,
        env=child_env,
        input_text=json.dumps(request, separators=(",", ":")),
    )
    # Drop all parent references before interpreting child output. The child
    # emits only bounded booleans/counts; credentials never enter evidence.
    request.clear()
    builtwith_credential = ""
    if (
        result.returncode != 0
        or len(result.stdout or "") > 256 * 1024
        or len(result.stderr or "") > 64 * 1024
    ):
        raise FullParityError("exact miner-intake workflow failed closed")
    evidence = _last_json_document(
        result.stdout or "", field="miner-intake workflow"
    )
    if (
        evidence.get("schema_version")
        != "leadpoet.production_parity_miner_intake_evidence.v1"
        or evidence.get("candidate_sha") != candidate_sha
        or evidence.get("run_id") != run_id
        or evidence.get("artifact_bucket") != artifact_bucket
        or evidence.get("status") != "passed"
        or evidence.get("production_database_mutated") is not False
        or evidence.get("production_chain_mutated") is not False
        or evidence.get("chain_registration_boundary")
        != "strict-ephemeral-hotkey"
        or evidence.get("source_add", {}).get("admitted") is not True
        or evidence.get("source_add", {}).get(
            "global_miner_submissions_enabled"
        )
        is not False
        or evidence.get("source_add", {}).get("source_add_paused") is not False
    ):
        raise FullParityError("miner-intake evidence is incomplete")
    return evidence


def _verify_builtwith_credential_live(credential: str) -> dict[str, Any]:
    url = (
        "https://api.builtwith.com/v23/api.json?LOOKUP=builtwith.com"
        "&HIDETEXT=yes&NOMETA=yes&NOPII=yes&NOATTR=yes"
    )
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"API {credential}",
            "User-Agent": "Leadpoet-Production-Parity/1.0",
        },
    )
    try:
        with urlopen(request, timeout=45) as response:
            payload = response.read(4 * 1024 * 1024 + 1)
            status = int(response.status)
    except Exception as exc:
        raise FullParityError(
            "BuiltWith live credential verification failed"
        ) from exc
    finally:
        url = ""
    if status != 200 or not payload or len(payload) > 4 * 1024 * 1024:
        raise FullParityError("BuiltWith live credential verification failed")
    try:
        document = json.loads(payload)
    except ValueError as exc:
        raise FullParityError(
            "BuiltWith live credential verification returned invalid JSON"
        ) from exc
    if not isinstance(document, (Mapping, list)):
        raise FullParityError(
            "BuiltWith live credential verification returned invalid JSON"
        )
    if isinstance(document, Mapping):
        errors = document.get("Errors") or document.get("errors")
        if errors:
            raise FullParityError("BuiltWith live credential verification failed")
    return {
        "http_status": status,
        "json_verified": True,
        "response_bytes": len(payload),
    }


async def _run_miner_intake_child(request: Mapping[str, Any]) -> dict[str, Any]:
    if (
        request.get("schema_version")
        != "leadpoet.production_parity_miner_intake_request.v1"
        or not SHA_RE.fullmatch(str(request.get("candidate_sha") or ""))
        or not RUN_RE.fullmatch(str(request.get("run_id") or ""))
        or not ARTIFACT_BUCKET_RE.fullmatch(
            str(request.get("artifact_bucket") or "")
        )
        or request.get("region") != "us-east-1"
        or os.getenv("AWS_REGION") != "us-east-1"
        or os.getenv("AWS_DEFAULT_REGION") != "us-east-1"
    ):
        raise FullParityError("miner-intake child identity differs")
    candidate_sha = str(request["candidate_sha"])
    run_id = str(request["run_id"])
    artifact_bucket = str(request["artifact_bucket"])
    supabase_origin = _validated_public_origin(
        str(request.get("supabase_origin") or "")
    )
    gateway_env_file = Path(str(request.get("gateway_env_file") or ""))
    values = _validated_clone_environment(
        gateway_env_file,
        candidate_sha=candidate_sha,
        run_id=run_id,
        supabase_origin=supabase_origin,
        artifact_bucket=artifact_bucket,
    )
    with _applied_clone_environment(
        values,
        gateway_env_file,
        overrides=MINER_INTAKE_ENVIRONMENT_OVERRIDES,
    ):
        return await _run_miner_intake_child_validated(request)


async def _run_miner_intake_child_validated(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    from types import SimpleNamespace

    import httpx
    from bittensor_wallet import Keypair

    from gateway.main import app
    from gateway.research_lab import api as research_lab_api
    from gateway.research_lab.source_add_workflow import source_add_control_state
    from gateway.research_lab.store import call_rpc, select_many, select_one
    from neurons.miner import (
        _research_lab_signed_payload,
        _research_lab_source_add_signed_payload,
    )
    from research_lab.source_add_miner import build_source_add_submission_docs

    if (
        request.get("schema_version")
        != "leadpoet.production_parity_miner_intake_request.v1"
        or not SHA_RE.fullmatch(str(request.get("candidate_sha") or ""))
        or not RUN_RE.fullmatch(str(request.get("run_id") or ""))
        or not ARTIFACT_BUCKET_RE.fullmatch(
            str(request.get("artifact_bucket") or "")
        )
        or os.getenv("LEADPOET_PARITY_CANDIDATE_SHA")
        != request.get("candidate_sha")
    ):
        raise FullParityError("miner-intake child identity differs")
    candidate_sha = str(request["candidate_sha"])
    run_id = str(request["run_id"])
    builtwith_credential = str(request.get("builtwith_credential") or "").strip()
    if not builtwith_credential:
        raise FullParityError("miner-intake credentials are incomplete")

    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet = SimpleNamespace(hotkey=keypair)
    miner_hotkey = keypair.ss58_address
    observed_chain_checks: list[str] = []
    chain_check_milestones: list[int] = []
    original_chain_registration = research_lab_api.chain_is_hotkey_registered

    async def strict_chain_registration(hotkey: str):
        observed_chain_checks.append(str(hotkey))
        if str(hotkey) != miner_hotkey:
            return False, None
        return True, "miner"

    research_lab_api.chain_is_hotkey_registered = strict_chain_registration
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    source_controls: dict[str, Any] = {}
    try:
        if research_lab_api.ResearchLabGatewayConfig.from_env().miner_submissions_enabled:
            raise FullParityError(
                "miner-intake child did not start with non-SOURCE_ADD intake closed"
            )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://production-parity.invalid",
            timeout=httpx.Timeout(180.0),
        ) as client:
            builtwith_probe = _verify_builtwith_credential_live(
                builtwith_credential
            )
            source_state = await source_add_control_state()
            source_controls = {
                "source_add_paused": bool(source_state.get("paused", True)),
            }
            if source_controls["source_add_paused"]:
                await call_rpc(
                    "research_lab_source_add_set_paused",
                    {
                        "p_paused": False,
                        "p_reason": "production_parity_miner_intake",
                        "p_actor_ref": "system:production-parity",
                    },
                )
            source_only_source_state = await source_add_control_state()
            if source_only_source_state.get("paused") is not False:
                raise FullParityError("SOURCE_ADD maintenance control did not activate")
            manifest, source_brief, idempotency_key, source_metadata = (
                build_source_add_submission_docs(
                    miner_hotkey=miner_hotkey,
                    source_name=f"BuiltWith API parity {run_id}",
                    source_kind="tech_stack",
                    api_base_url="https://api.builtwith.com/v23",
                    documentation_url=(
                        "https://github.com/builtwith/builtwith-ai-sdk"
                    ),
                    auth_type="api_key_header",
                    endpoint_examples=(
                        {
                            "method": "GET",
                            "path": "/api.json",
                            "purpose": "Retrieve company technology usage",
                            "example_query": (
                                "LOOKUP=builtwith.com&HIDETEXT=yes&NOMETA=yes"
                                "&NOPII=yes&NOATTR=yes"
                            ),
                        },
                    ),
                    rate_limit_notes=(
                        "Provider account limits apply; one bounded read-only "
                        "lookup is used for admission validation."
                    ),
                    data_provenance_notes=(
                        "Technology observations returned by the BuiltWith API."
                    ),
                    third_party_refs=(
                        "https://api.builtwith.com/domain-api",
                    ),
                    credential_supplied=False,
                )
            )
            source_payload = _research_lab_source_add_signed_payload(
                wallet,
                {
                    "miner_hotkey": miner_hotkey,
                    "timestamp": int(time.time()),
                    "idempotency_key": idempotency_key,
                    "manifest": manifest,
                    "source_brief": source_brief,
                    "source_metadata": source_metadata,
                },
            )
            source_response = await client.post(
                "/research-lab/source-adapters", json=source_payload
            )
            if source_response.status_code != 200:
                raise FullParityError(
                    "SOURCE_ADD admission rejected the exact miner request"
                )
            chain_check_milestones.append(len(observed_chain_checks))
            source_result = source_response.json()
            submission_id = str(source_result.get("submission_id") or "")
            source_row = await select_one(
                "research_lab_source_add_submission_current",
                filters=(("submission_id", submission_id),),
            )
            work_rows = await select_many(
                "research_lab_source_add_work_items",
                filters=(("submission_id", submission_id),),
                limit=5,
            )
            source_persistence = json.dumps(
                {
                    "response": source_result,
                    "submission": source_row,
                    "work": work_rows,
                },
                sort_keys=True,
                default=str,
            )
            current_doc = (
                source_row.get("submission_doc")
                if isinstance(source_row, Mapping)
                and isinstance(source_row.get("submission_doc"), Mapping)
                else {}
            )
            if (
                not submission_id
                or source_result.get("stage") != "provenance_queued"
                or not isinstance(source_row, Mapping)
                or source_row.get("stage") != "provenance_queued"
                or len(work_rows) != 1
                or work_rows[0].get("work_kind") != "provenance"
                or work_rows[0].get("work_status") != "queued"
                or int(work_rows[0].get("attempt_count") or 0) != 0
                or current_doc.get("credential_envelope") not in ({}, None)
                or builtwith_credential in source_persistence
            ):
                raise FullParityError("SOURCE_ADD admission persistence is incomplete")
            if (
                (await source_add_control_state()).get("paused") is not False
                or research_lab_api.ResearchLabGatewayConfig.from_env().miner_submissions_enabled
            ):
                raise FullParityError("SOURCE_ADD admission changed its intake controls")

            retired_payload = _research_lab_signed_payload(
                wallet,
                {
                    "miner_hotkey": miner_hotkey,
                    "timestamp": int(time.time()),
                    "idempotency_key": f"source-add-recipient:{run_id}",
                    "adapter_id": str(source_result.get("adapter_id") or ""),
                },
            )
            retired_response = await client.post(
                "/research-lab/source-adapters/credential-recipient",
                json=retired_payload,
            )
            chain_check_milestones.append(len(observed_chain_checks))
            forbidden_payload = {
                **source_payload,
                "adapter_credential": "production-parity-forbidden-value",
            }
            forbidden_response = await client.post(
                "/research-lab/source-adapters", json=forbidden_payload
            )
            if (
                retired_response.status_code != 410
                or forbidden_response.status_code != 422
            ):
                raise FullParityError(
                    "SOURCE_ADD public credential boundary did not fail closed"
                )

        if (
            any(hotkey != miner_hotkey for hotkey in observed_chain_checks)
            or len(chain_check_milestones) != 2
            or any(
                current <= previous
                for previous, current in zip(
                    (0, *chain_check_milestones[:-1]),
                    chain_check_milestones,
                )
            )
        ):
            raise FullParityError(
                "miner intake did not traverse the strict registration boundary"
            )
        return {
            "schema_version": (
                "leadpoet.production_parity_miner_intake_evidence.v1"
            ),
            "status": "passed",
            "candidate_sha": candidate_sha,
            "run_id": run_id,
            "artifact_bucket": str(request["artifact_bucket"]),
            "chain_registration_boundary": "strict-ephemeral-hotkey",
            "production_database_mutated": False,
            "production_chain_mutated": False,
            "source_add": {
                "provider": "builtwith",
                "live_provider_credential_verified": (
                    builtwith_probe["http_status"] == 200
                    and builtwith_probe["json_verified"] is True
                ),
                "miner_signature_verified": True,
                "admitted": True,
                "stage": "provenance_queued",
                "queued_work_count": 1,
                "downstream_executed": False,
                "credential_transport": "operator-managed-production-contract",
                "public_credentials_forbidden": True,
                "plaintext_absent": True,
                "global_miner_submissions_enabled": False,
                "source_add_paused": False,
            },
        }
    finally:
        research_lab_api.chain_is_hotkey_registered = original_chain_registration
        try:
            await _restore_miner_intake_controls(source_controls, call_rpc=call_rpc)
        finally:
            request = {}
            builtwith_credential = ""


async def _restore_miner_intake_controls(
    source_controls: Mapping[str, Any],
    *,
    call_rpc: Any,
) -> None:
    """Restore the clone-local SOURCE_ADD control changed for intake."""

    if source_controls.get("source_add_paused"):
        await call_rpc(
            "research_lab_source_add_set_paused",
            {
                "p_paused": True,
                "p_reason": "production_parity_miner_intake_complete",
                "p_actor_ref": "system:production-parity",
            },
        )


def _current_epoch_from_readiness(output: str) -> int:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, Mapping):
            for key in ("epoch_id", "effective_epoch", "epoch"):
                try:
                    epoch = int(value.get(key))
                except (TypeError, ValueError):
                    continue
                if epoch > 0:
                    return epoch
    raise FullParityError("weight readiness did not report its effective epoch")


def _validate_real_handoff(
    *, epoch: int, candidate_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    from leadpoet_canonical.allocation_handoff_v2 import (
        validate_allocation_handoff_v2,
    )
    from research_lab.validator_integration import (
        ResearchLabValidatorFlags,
        build_research_lab_allocation_component,
        fetch_research_lab_attested_allocation_bundle,
        verify_research_lab_allocation_bundle,
    )

    handoff = fetch_research_lab_attested_allocation_bundle(
        "http://127.0.0.1:8000",
        epoch,
        timeout_seconds=360,
    )
    normalized = validate_allocation_handoff_v2(
        handoff,
        expected_epoch_id=epoch,
        expected_netuid=71,
    )
    flags = ResearchLabValidatorFlags.from_mapping(os.environ)
    verification = verify_research_lab_allocation_bundle(
        normalized["bundle"], flags=flags
    )
    if verification.get("passed") is not True:
        raise FullParityError("production validator rejected the real allocation handoff")
    component = build_research_lab_allocation_component(
        normalized["bundle"], flags=flags
    )
    public_evidence = {
        "epoch": epoch,
        "root_receipt_hash": normalized["root_receipt_hash"],
        "allocation_hash": component["allocation_hash"],
        "handoff_hash": sha256_json(normalized),
        "serialized_bytes": len(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "validator_verification_passed": True,
    }
    rehearsal_input = {
        "schema_version": "leadpoet.rehearsal_production_allocation.v1",
        "candidate_sha": candidate_sha,
        "source_epoch": epoch,
        "root_receipt_hash": normalized["root_receipt_hash"],
        "handoff_hash": public_evidence["handoff_hash"],
        "allocation_hash": component["allocation_hash"],
        "allocation_doc": component["allocation_doc"],
    }
    return public_evidence, rehearsal_input


def _run_clone_handoff_child(request: Mapping[str, Any]) -> dict[str, Any]:
    if (
        request.get("schema_version")
        != "leadpoet.production_parity_clone_handoff_request.v1"
        or not SHA_RE.fullmatch(str(request.get("candidate_sha") or ""))
        or not RUN_RE.fullmatch(str(request.get("run_id") or ""))
        or not ARTIFACT_BUCKET_RE.fullmatch(
            str(request.get("artifact_bucket") or "")
        )
        or not isinstance(request.get("epoch"), int)
        or isinstance(request.get("epoch"), bool)
        or int(request.get("epoch") or 0) <= 0
    ):
        raise FullParityError("clone allocation handoff request is invalid")
    candidate_sha = str(request["candidate_sha"])
    run_id = str(request["run_id"])
    artifact_bucket = str(request["artifact_bucket"])
    epoch = int(request["epoch"])
    supabase_origin = str(request.get("supabase_origin") or "")
    gateway_env_file = Path(str(request.get("gateway_env_file") or ""))
    values = _validated_clone_environment(
        gateway_env_file,
        candidate_sha=candidate_sha,
        run_id=run_id,
        supabase_origin=supabase_origin,
        artifact_bucket=artifact_bucket,
    )
    with _applied_clone_environment(values, gateway_env_file):
        handoff, allocation_input = _validate_real_handoff(
            epoch=epoch,
            candidate_sha=candidate_sha,
        )
    return {
        "schema_version": "leadpoet.production_parity_clone_handoff_evidence.v1",
        "candidate_sha": candidate_sha,
        "run_id": run_id,
        "artifact_bucket": artifact_bucket,
        "epoch": epoch,
        "handoff": handoff,
        "allocation_input": allocation_input,
    }


def _run_clone_handoff(
    *,
    epoch: int,
    candidate_sha: str,
    run_id: str,
    supabase_origin: str,
    gateway_env_file: Path,
    artifact_bucket: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validated_clone_environment(
        gateway_env_file,
        candidate_sha=candidate_sha,
        run_id=run_id,
        supabase_origin=supabase_origin,
        artifact_bucket=artifact_bucket,
    )
    request = {
        "schema_version": "leadpoet.production_parity_clone_handoff_request.v1",
        "candidate_sha": candidate_sha,
        "run_id": run_id,
        "artifact_bucket": artifact_bucket,
        "epoch": epoch,
        "supabase_origin": supabase_origin,
        "gateway_env_file": str(gateway_env_file),
    }
    result = _run(
        [sys.executable, str(Path(__file__).resolve()), "--clone-handoff-child"],
        timeout=600,
        env=_clone_child_environment(),
        input_text=json.dumps(request, separators=(",", ":")),
    )
    if result.returncode != 0 or len(result.stdout or "") > 4 * 1024 * 1024:
        raise FullParityError("clone allocation handoff child failed closed")
    evidence = _last_json_document(
        result.stdout or "", field="clone allocation handoff child"
    )
    handoff = evidence.get("handoff")
    allocation_input = evidence.get("allocation_input")
    if (
        evidence.get("schema_version")
        != "leadpoet.production_parity_clone_handoff_evidence.v1"
        or evidence.get("candidate_sha") != candidate_sha
        or evidence.get("run_id") != run_id
        or evidence.get("artifact_bucket") != artifact_bucket
        or evidence.get("epoch") != epoch
        or not isinstance(handoff, Mapping)
        or not isinstance(allocation_input, Mapping)
        or handoff.get("epoch") != epoch
        or allocation_input.get("candidate_sha") != candidate_sha
        or int(allocation_input.get("source_epoch") or 0) != epoch
        or handoff.get("allocation_hash") != allocation_input.get("allocation_hash")
    ):
        raise FullParityError("clone allocation handoff child evidence is incomplete")
    return dict(handoff), dict(allocation_input)


def _run_nonforwarding_weight_path(
    *,
    base_sha: str,
    candidate_sha: str,
    production_allocation: Path,
) -> dict[str, Any]:
    evidence = Path("/tmp") / f"leadpoet-restart-rehearsal-{candidate_sha}-prepush.json"
    evidence.unlink(missing_ok=True)
    result = _run(
        [
            sys.executable,
            "scripts/run_local_restart_rehearsal.py",
            "--from-sha",
            base_sha,
            "--candidate-sha",
            candidate_sha,
            "--transition",
            "forward",
            "--profile",
            "prepush",
            "--production-allocation",
            str(production_allocation),
        ],
        timeout=600,
    )
    _require(result, stage="primary/audit non-forwarding submission path")
    try:
        value = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FullParityError("weight-path evidence is unreadable") from exc
    if (
        value.get("status") != "passed"
        or value.get("release_sha") != candidate_sha
        or value.get("from_sha") != base_sha
    ):
        raise FullParityError("weight-path evidence identity differs")
    canonical = value.get("canonical_vector")
    auditor = value.get("auditor")
    if not isinstance(canonical, Mapping) or not isinstance(auditor, Mapping):
        raise FullParityError("primary/audit weight evidence is incomplete")
    allocation_input = json.loads(
        production_allocation.read_text(encoding="utf-8")
    )
    production_allocation_hash = sha256_json(
        allocation_input["allocation_doc"]
    )
    allocation_evidence = value.get("production_allocation")
    if (
        not isinstance(allocation_evidence, Mapping)
        or allocation_evidence.get("allocation_hash")
        != allocation_input["allocation_hash"]
        or allocation_evidence.get("handoff_hash")
        != allocation_input["handoff_hash"]
        or int(allocation_evidence.get("source_epoch") or -1)
        != int(allocation_input["source_epoch"])
    ):
        raise FullParityError(
            "primary/audit workflow did not consume the clone allocation"
        )
    return {
        "bundle_hash": value.get("bundle_hash"),
        "canonical_vector_hash": sha256_json(dict(canonical)),
        "primary_audit_equal": True,
        "sdk_signed": bool(value.get("signed_extrinsic")),
        "finalization_verified": bool(value.get("finalization")),
        "readback_verified": bool(value.get("reveal")),
        "chain_boundary": "strict-non-forwarding",
        "production_allocation_hash": allocation_input["allocation_hash"],
        "production_allocation_document_hash": production_allocation_hash,
        "production_allocation_bound": True,
    }


def run_full(
    *,
    region: str,
    run_id: str,
    base_sha: str,
    candidate_sha: str,
    production_gateway_secret_id: str,
    readonly_dsn_secret_id: str,
    miner_intake_secret_id: str,
    supabase_origin: str,
    artifact_bucket: str,
    postgres_image: str,
    postgrest_image: str,
    output: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    if (
        region != "us-east-1"
        or not RUN_RE.fullmatch(run_id)
        or not SHA_RE.fullmatch(base_sha)
        or not SHA_RE.fullmatch(candidate_sha)
        or base_sha == candidate_sha
        or not PINNED_IMAGE_RE.fullmatch(postgres_image)
        or not PINNED_IMAGE_RE.fullmatch(postgrest_image)
        or not ARTIFACT_BUCKET_RE.fullmatch(artifact_bucket)
    ):
        raise FullParityError("full parity inputs are invalid")
    started = time.monotonic()
    deadline = _full_deadline(
        started=started,
        timeout_seconds=timeout_seconds,
    )
    work = FULL_WORK_ROOT / run_id / "runtime"
    scoring_cache = work / "scoring-cache"
    runtime_config = work / "runtime-config.json"
    contract_path = work / "contract.json"
    archive_path = work / "production.dump"
    manifest_path = work / "snapshot-manifest.json"
    gateway_env_file = work / "gateway.env"
    gateway_log = work / "gateway-restart.log"
    allocation_override = work / "production-allocation.json"
    artifact_policy = work / "v2-config" / "encrypted-artifact-policy.json"
    secrets_client: Any | None = None
    database: _DockerDatabase | None = None
    prefix_adapter: _ClonePostgrestPrefixAdapter | None = None
    secret_created = False
    failure_stage = "initialization"
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "base_sha": base_sha,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        failure_stage = "public-origin"
        supabase_origin = _validated_public_origin(supabase_origin)
        failure_stage = "checkout-identity"
        _checkout_identity(candidate_sha)
        failure_stage = "early-boot-isolation"
        try:
            boot_state = EARLY_BOOT_MARKER.read_text(
                encoding="utf-8"
            ).strip()
        except OSError as exc:
            raise FullParityError(
                "transient host did not prove early production-service isolation"
            ) from exc
        if boot_state != "isolated":
            raise FullParityError(
                "transient host early production-service isolation differs"
            )
        failure_stage = "work-root"
        work.mkdir(parents=True, mode=0o700, exist_ok=False)
        scoring_cache.mkdir(mode=0o700)
        failure_stage = "runtime-identity"
        runtime_identity = _materialize_run_owned_runtime_identity(
            work / "runtime-identity"
        )
        gateway_private_key_path = runtime_identity["gateway_private_key_path"]
        arweave_keyfile_path = runtime_identity["arweave_keyfile_path"]
        evidence["runtime_identity"] = {
            "gateway": "run-scoped-ephemeral",
            "gateway_public_key_hash": runtime_identity[
                "gateway_public_key_hash"
            ],
            "arweave": "run-scoped-ephemeral-write-blocked",
            "arweave_address_hash": runtime_identity[
                "arweave_address_hash"
            ],
        }
        secrets_client = boto3.client("secretsmanager", region_name=region)
        database = _DockerDatabase(
            candidate_sha=candidate_sha,
            postgres_image=postgres_image,
            postgrest_image=postgrest_image,
            postgres_publish="127.0.0.1::5432",
            postgrest_publish="127.0.0.1::3000",
        )
        failure_stage = "runtime-config-capture"
        capture(
            client=secrets_client,
            secret_id=production_gateway_secret_id,
            output=runtime_config,
        )
        failure_stage = "parity-contract"
        contract = build_contract(
            root=ROOT,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
            runtime_config=runtime_config,
            require_runtime_config=True,
        )
        contract_path.write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        failure_stage = "production-dsn"
        dsn = _dsn_from_secret(
            _secret_value(
                secrets_client,
                readonly_dsn_secret_id,
                field="read-only production DSN",
            )
        )
        production_host = str(urlparse(dsn).hostname or "").lower()
        failure_stage = "snapshot-capture"
        manifest = capture_snapshot(
            contract_path=contract_path,
            archive_path=archive_path,
            manifest_path=manifest_path,
            dsn=dsn,
            expected_production_host=production_host,
            ttl_hours=24,
            source_sha=base_sha,
            timeout_seconds=_remaining_full_timeout(
                deadline=deadline,
                stage="full production snapshot capture",
            ),
            postgres_image=postgres_image,
        )
        if manifest["capture_mode"] != "full":
            raise FullParityError("authoritative parity requires a full production clone")
        failure_stage = "clone-start"
        database.start()
        prerequisites = database.prepare_snapshot_restore()
        failure_stage = "snapshot-restore"
        restore = restore_snapshot(
            root=ROOT,
            contract_path=contract_path,
            manifest_path=manifest_path,
            archive_path=archive_path,
            target_dsn=database.target_dsn,
            production_host=production_host,
            timeout_seconds=_remaining_full_timeout(
                deadline=deadline,
                stage="full production snapshot restore",
            ),
            postgres_image=postgres_image,
        )
        restore_contract = database.verify_snapshot_restore()
        restore = {
            **restore,
            "clone_prerequisites": prerequisites,
            "clone_restore_contract": restore_contract,
        }
        failure_stage = "clone-http-origin"
        local_url, _ = database.start_postgrest()
        prefix_adapter = _ClonePostgrestPrefixAdapter(
            upstream_origin=local_url,
            public_origin=supabase_origin,
        )
        prefix_adapter_evidence = prefix_adapter.start()
        if (
            prefix_adapter_evidence.get("listen_host") != "0.0.0.0"
            or prefix_adapter_evidence.get("listen_port") != 3000
            or prefix_adapter_evidence.get("path_prefix") != "/rest/v1"
        ):
            raise FullParityError("clone PostgREST prefix adapter bind differs")
        _wait_https_origin(supabase_origin)
        failure_stage = "clone-secret"
        secret_state = create_gateway_secret(
            client=secrets_client,
            source_secret_id=production_gateway_secret_id,
            run_id=run_id,
            candidate_sha=candidate_sha,
            gateway_public_key=runtime_identity["gateway_public_key"],
            supabase_origin=supabase_origin,
            artifact_bucket=artifact_bucket,
            benchmark_date=manifest["database"]["target_rebenchmark_date"],
            jwt_secret=database.jwt_secret,
        )
        secret_created = True
        artifact_policy.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        artifact_policy.write_text(
            json.dumps(
                {
                    "schema_version": "leadpoet.encrypted_artifact_policy.v2",
                    "bucket_host": (
                        f"{artifact_bucket}.s3.{region}.amazonaws.com"
                    ),
                    "key_prefix": "/encrypted-artifacts/",
                    "minimum_retention_days": 1,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact_policy.chmod(0o600)
        env = _full_restart_environment(
            region=region,
            updates={
                "LEADPOET_REPO_ROOT": str(ROOT),
                "GATEWAY_ROOT": str(ROOT / "gateway"),
                "LEADPOET_GATEWAY_ENV_SECRET_ID": secret_state["secret_id"],
                "GATEWAY_ENV_FILE": str(gateway_env_file),
                "GATEWAY_LOG_ROOT": str(work / "gateway"),
                "GATEWAY_LOG_FILE": str(work / "gateway" / "gateway.log"),
                "GATEWAY_PRIVATE_KEY_PATH": gateway_private_key_path,
                "ARWEAVE_KEYFILE_PATH": arweave_keyfile_path,
                "GATEWAY_RESTART_CONTROLLER_ROOT": str(work / "restart-controller"),
                "GATEWAY_DEPLOYMENT_DIR": str(work / "deployments"),
                "GATEWAY_HOST_RESTART_SCRIPT": str(ROOT / "gw_restart.sh"),
                "GATEWAY_TEE_EIF_ROOT": str(work / "tee"),
                "GATEWAY_V2_CONFIG_DIR": str(work / "v2-config"),
                "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT": str(work / "offline-artifacts"),
                "VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT": str(work / "offline-artifacts" / "validator-runtime"),
                "GATEWAY_RESTART_LOCK_FILE": str(work / "gateway-restart.lock"),
                "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(work / "docker-operation.lock"),
                "GATEWAY_ACTIVE_RELEASE_FALLBACK_CONTEXT": "full-parity",
                "GATEWAY_DEPLOY_COMMIT": candidate_sha,
                "GATEWAY_PYTHON_BIN": sys.executable,
                "GATEWAY_V2_RELEASE_BUCKET": ATTESTED_V2_RELEASE_BUCKET,
                "GATEWAY_V2_RELEASE_PREFIX": ATTESTED_V2_RELEASE_PREFIX,
                "GATEWAY_V2_KMS_KEY_ID": ATTESTED_V2_KMS_KEY_ID,
            },
        )
        failure_stage = "gateway-restart"
        restart = _run(
            ["bash", str(ROOT / "gw_restart.sh"), "--commit", candidate_sha],
            timeout=min(
                _remaining_full_timeout(
                    deadline=deadline,
                    stage="exact gateway restart",
                ),
                10800,
            ),
            env=env,
            log_path=gateway_log,
        )
        if restart.returncode != 0:
            raise FullParityError("exact gateway restart failed")
        failure_stage = "gateway-health"
        health = _gateway_json("/health/v2-authority")
        build = _gateway_json("/build-info")
        if (
            health.get("status") != "ready"
            or str(health.get("commit_sha") or "").lower() != candidate_sha
            or str(build.get("git_commit") or "").lower() != candidate_sha
        ):
            raise FullParityError("gateway V2 health is not exact-candidate ready")

        parsed_env = _validated_clone_environment(
            gateway_env_file,
            candidate_sha=candidate_sha,
            run_id=run_id,
            supabase_origin=supabase_origin,
            artifact_bucket=artifact_bucket,
        )
        failure_stage = "arena-rebenchmark"
        arena_rebenchmark = _run_arena_rebenchmark_path(
            region=region,
            candidate_sha=candidate_sha,
            run_id=run_id,
            supabase_origin=supabase_origin,
            gateway_env_file=gateway_env_file,
            artifact_bucket=artifact_bucket,
            jwt_secret=database.jwt_secret,
            timeout_seconds=_remaining_full_timeout(
                deadline=deadline,
                stage="live Arena baseline rebenchmark",
            ),
        )
        failure_stage = "weight-readiness"
        readiness = _run(
            [
                sys.executable,
                "-m",
                "gateway.tee.verify_weight_submission_ready_v2",
                "--gateway-url",
                "http://127.0.0.1:8000",
                "--http-timeout-seconds",
                "360",
            ],
            timeout=min(
                _remaining_full_timeout(
                    deadline=deadline,
                    stage="real gateway weight readiness",
                ),
                1800,
            ),
            env=_clone_runtime_environment(
                parsed_env,
                gateway_env_file=gateway_env_file,
                region=region,
            ),
        )
        readiness_output = _require(readiness, stage="real gateway weight readiness")
        epoch = _current_epoch_from_readiness(readiness_output)
        failure_stage = "allocation-handoff"
        handoff, allocation_input = _run_clone_handoff(
            epoch=epoch,
            candidate_sha=candidate_sha,
            run_id=run_id,
            supabase_origin=supabase_origin,
            gateway_env_file=gateway_env_file,
            artifact_bucket=artifact_bucket,
        )
        allocation_override.write_text(
            json.dumps(allocation_input, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        allocation_override.chmod(0o600)
        failure_stage = "nonforwarding-weight-path"
        weight_path = _run_nonforwarding_weight_path(
            base_sha=base_sha,
            candidate_sha=candidate_sha,
            production_allocation=allocation_override,
        )
        if (
            handoff["allocation_hash"]
            != weight_path["production_allocation_hash"]
        ):
            raise FullParityError(
                "gateway handoff and primary/audit allocation hashes differ"
            )
        failure_stage = "miner-intake"
        miner_intake = _run_miner_intake_path(
            region=region,
            candidate_sha=candidate_sha,
            run_id=run_id,
            supabase_origin=supabase_origin,
            gateway_env_file=gateway_env_file,
            artifact_bucket=artifact_bucket,
            miner_intake_secret=_secret_value(
                secrets_client,
                miner_intake_secret_id,
                field="miner-intake credential",
            ),
        )
        failure_stage = "clone-shape"
        service_role_key = _clone_service_role_key(
            parsed_env,
            candidate_sha=candidate_sha,
            run_id=run_id,
            supabase_origin=supabase_origin,
            jwt_secret=database.jwt_secret,
        )
        shape = database.shape_evidence(
            service_role_key=service_role_key,
            expected_shape=validate_snapshot_manifest(manifest)["database"],
            capture_mode="full",
        )
        failure_stage = "clone-weight-history"
        scale = database.weight_input_scale_evidence(
            service_role_key=service_role_key
        )
        evidence.update(
            {
                "status": "passed",
                "contract_hash": contract["contract_hash"],
                "snapshot_hash": manifest["manifest_hash"],
                "snapshot_restore": restore,
                "clone_http_adapter": prefix_adapter_evidence,
                "production_shape": shape,
                "production_weight_input_scale": scale,
                "gateway": {
                    "commit_sha": candidate_sha,
                    "pcr0": health.get("pcr0"),
                    "attestation_ready": True,
                },
                "external_write_boundaries": {
                    "arweave": "blocked-production-parity",
                },
                "arena_rebenchmark": arena_rebenchmark,
                "allocation_handoff": handoff,
                "weight_path": weight_path,
                "miner_intake": miner_intake,
            }
        )
    except Exception as exc:
        bounded_stage, bounded_type = _failure_identity(failure_stage, exc)
        evidence["status"] = "failed"
        evidence["failure_stage"] = bounded_stage
        evidence["error_type"] = bounded_type
        raise
    finally:
        cleanup: dict[str, Any] = {}
        if prefix_adapter is not None:
            try:
                cleanup["prefix_adapter"] = prefix_adapter.cleanup()
            except Exception as exc:  # noqa: BLE001 - bounded cleanup evidence
                cleanup["prefix_adapter_error"] = _failure_identity(
                    "cleanup", exc
                )[1]
        if database is not None:
            try:
                cleanup["database"] = database.cleanup()
            except Exception as exc:  # noqa: BLE001 - cleanup evidence must survive
                cleanup["database_error"] = _failure_identity("cleanup", exc)[1]
        if secret_created and secrets_client is not None:
            try:
                cleanup["secret"] = delete_gateway_secret(
                    client=secrets_client, run_id=run_id
                )
            except Exception as exc:  # noqa: BLE001
                cleanup["secret_error"] = _failure_identity("cleanup", exc)[1]
        try:
            shutil.rmtree(work)
            cleanup["work"] = "removed"
        except FileNotFoundError:
            cleanup["work"] = "already_absent"
        except OSError as exc:
            cleanup["work_error"] = _failure_identity("cleanup", exc)[1]
        evidence["cleanup"] = cleanup
        evidence["duration_seconds"] = round(time.monotonic() - started, 3)
        evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
        if evidence.get("status") == "passed" and (
            "prefix_adapter_error" in cleanup
            or "database_error" in cleanup
            or "secret_error" in cleanup
            or "work_error" in cleanup
        ):
            evidence["status"] = "failed"
            evidence["failure_stage"] = "cleanup"
            evidence["error_type"] = "CleanupError"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(evidence, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    if evidence.get("status") != "passed":
        raise FullParityError("full parity failed closed")
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv if argv is not None else sys.argv[1:])
    child_modes = {
        "--arena-rebenchmark-child",
        "--clone-handoff-child",
        "--miner-intake-child",
    }
    if len(raw_args) == 1 and raw_args[0] in child_modes:
        child_mode = raw_args[0]
        logging.disable(logging.CRITICAL)
        try:
            request = _read_child_request()
            if child_mode == "--arena-rebenchmark-child":
                result = _run_arena_rebenchmark_child(request)
            elif child_mode == "--clone-handoff-child":
                result = _run_clone_handoff_child(request)
            else:
                result = asyncio.run(_run_miner_intake_child(request))
        except Exception:  # noqa: BLE001 - never print secret-bearing detail
            print("ERROR: clone child failed closed", file=sys.stderr)
            return 1
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--production-gateway-secret-id", required=True)
    parser.add_argument("--readonly-dsn-secret-id", required=True)
    parser.add_argument("--miner-intake-secret-id", required=True)
    parser.add_argument("--supabase-origin", required=True)
    parser.add_argument("--artifact-bucket", required=True)
    parser.add_argument("--postgres-image", required=True)
    parser.add_argument("--postgrest-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timeout-seconds", type=int, default=MAX_FULL_TIMEOUT_SECONDS
    )
    args = parser.parse_args(argv)
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    base_sha = args.base_sha.lower()
    candidate_sha = args.candidate_sha.lower()
    try:
        result = run_full(
            region=args.region,
            run_id=args.run_id,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
            production_gateway_secret_id=args.production_gateway_secret_id,
            readonly_dsn_secret_id=args.readonly_dsn_secret_id,
            miner_intake_secret_id=args.miner_intake_secret_id,
            supabase_origin=args.supabase_origin,
            artifact_bucket=args.artifact_bucket,
            postgres_image=args.postgres_image,
            postgrest_image=args.postgrest_image,
            output=args.output,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - evidence exposes only fixed fields
        try:
            _write_early_failure_evidence(
                output=args.output,
                run_id=args.run_id,
                base_sha=base_sha,
                candidate_sha=candidate_sha,
                started_at=started_at,
                started_monotonic=started_monotonic,
                error=exc,
            )
        except Exception:  # noqa: BLE001 - retention must not expose raw detail
            pass
        print("ERROR: full parity failed closed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_sha": result["candidate_sha"],
                "duration_seconds": result["duration_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
