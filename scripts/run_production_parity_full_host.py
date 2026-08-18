#!/usr/bin/env python3
"""Run the full rebenchmark and non-forwarding weight path on one Nitro host."""

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
from gateway.tee.acceptance_corpus_v2 import (  # noqa: E402
    load_and_validate_acceptance_corpus_v2,
)
from gateway.tee.release_channel_v2 import (  # noqa: E402
    fetch_release_channel_v2,
)
from gateway.tee.release_manifest_v2 import (  # noqa: E402
    validate_release_manifest,
)


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
ARTIFACT_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
PINNED_IMAGE_RE = re.compile(r"^[A-Za-z0-9._/:@-]+@sha256:[0-9a-f]{64}$")
SCHEMA_VERSION = "leadpoet.production_parity_full.v3"
OPENROUTER_RUNTIME_CREDENTIAL_REFS = (
    "RESEARCH_LAB_OPENROUTER_API_KEY",
    "RESEARCH_LAB_V2_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
    "QUALIFICATION_OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
)
OPENROUTER_MANAGEMENT_CREDENTIAL_REFS = (
    "RESEARCH_LAB_OPENROUTER_MANAGEMENT_KEY",
    "OPENROUTER_MANAGEMENT_KEY",
    "OPENROUTER_API_MANAGEMENT_KEY",
    "OR_MANAGEMENT_KEY",
)
MINER_INTAKE_ENVIRONMENT_OVERRIDES = {
    "RESEARCH_LAB_GATEWAY_API_ENABLED": "true",
    "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED": "true",
    "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "true",
    "RESEARCH_LAB_SOURCE_ADD_ENABLED": "true",
    "RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED": "false",
    "RESEARCH_LAB_PAID_LOOPS_ENABLED": "false",
}
EARLY_BOOT_MARKER = Path(
    "/run/leadpoet-production-parity/early-boot-isolated"
)
FULL_WORK_ROOT = Path("/opt/leadpoet-production-parity")
PRODUCTION_V2_CONFIG_DIR = Path("/home/ec2-user/.config/leadpoet/v2")
PRODUCTION_GATEWAY_PRIVATE_KEY_PATH = Path(
    "/home/ec2-user/gateway/secrets/gateway_private_key.pem"
)
PRODUCTION_GATEWAY_PRIVATE_KEY_OWNER = (1000, 1000)
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
        "runtime-config-capture",
        "parity-contract",
        "production-dsn",
        "snapshot-capture",
        "clone-start",
        "snapshot-restore",
        "clone-http-origin",
        "clone-secret",
        "acceptance-corpus",
        "gateway-restart",
        "gateway-health",
        "clone-controls",
        "rebenchmark-publication",
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
        "AcceptanceCorpusV2Error",
        "CalledProcessError",
        "ClientError",
        "CleanupError",
        "FullParityError",
        "HTTPError",
        "JSONDecodeError",
        "OSError",
        "ProductionParityError",
        "ReleaseChannelV2Error",
        "ReleaseManifestV2Error",
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


def _acceptance_corpus_tree(
    root: Path,
    *,
    owner_uid: int,
    owner_gid: int,
) -> tuple[list[Path], list[Path]]:
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise FullParityError("signed acceptance corpus is unavailable") from exc
    if (
        root.is_symlink()
        or not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
        or root_metadata.st_uid != owner_uid
        or root_metadata.st_gid != owner_gid
    ):
        raise FullParityError("signed acceptance corpus ownership differs")

    directories: list[Path] = []
    files: list[Path] = []
    for current, names, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(names):
            path = current_path / name
            metadata = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or metadata.st_uid != owner_uid
                or metadata.st_gid != owner_gid
            ):
                raise FullParityError("signed acceptance corpus ownership differs")
            directories.append(path)
        for name in sorted(filenames):
            path = current_path / name
            metadata = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_uid != owner_uid
                or metadata.st_gid != owner_gid
            ):
                raise FullParityError("signed acceptance corpus ownership differs")
            files.append(path)
    return directories, files


def _copy_acceptance_file(source: Path, destination: Path) -> None:
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_descriptor = os.open(source, source_flags)
    try:
        source_metadata = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(source_metadata.st_mode)
            or stat.S_IMODE(source_metadata.st_mode) != 0o600
        ):
            raise FullParityError("signed acceptance corpus ownership differs")
        destination_descriptor = os.open(
            destination,
            destination_flags,
            0o600,
        )
        try:
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_descriptor, view)
                    if written <= 0:
                        raise FullParityError(
                            "signed acceptance corpus copy failed"
                        )
                    view = view[written:]
            os.fchmod(destination_descriptor, 0o600)
            os.fsync(destination_descriptor)
        finally:
            os.close(destination_descriptor)
    finally:
        os.close(source_descriptor)


def _materialize_acceptance_corpus(
    *,
    source_config_dir: Path,
    destination_config_dir: Path,
    candidate_sha: str,
    candidate_release_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    release = validate_release_manifest(candidate_release_manifest)
    signer_hash = str(release.get("acceptance_signer_pubkey_hash") or "")
    if (
        release.get("commit_sha") != candidate_sha
        or not HASH_RE.fullmatch(signer_hash)
        or not HASH_RE.fullmatch(str(release.get("release_hash") or ""))
    ):
        raise FullParityError("candidate acceptance signer identity differs")

    source_config = Path(source_config_dir)
    source_manifest = source_config / "acceptance-corpus-v2.json"
    source_root = source_config / "acceptance-corpus-v2"
    try:
        source_config_metadata = source_config.lstat()
        manifest_metadata = source_manifest.lstat()
    except OSError as exc:
        raise FullParityError("signed acceptance corpus is unavailable") from exc
    if (
        source_config.is_symlink()
        or not stat.S_ISDIR(source_config_metadata.st_mode)
        or stat.S_IMODE(source_config_metadata.st_mode) != 0o700
        or source_manifest.is_symlink()
        or not stat.S_ISREG(manifest_metadata.st_mode)
        or stat.S_IMODE(manifest_metadata.st_mode) != 0o600
        or manifest_metadata.st_uid != source_config_metadata.st_uid
        or manifest_metadata.st_gid != source_config_metadata.st_gid
    ):
        raise FullParityError("signed acceptance corpus ownership differs")
    directories, files = _acceptance_corpus_tree(
        source_root,
        owner_uid=manifest_metadata.st_uid,
        owner_gid=manifest_metadata.st_gid,
    )
    try:
        source_value = load_and_validate_acceptance_corpus_v2(
            source_manifest,
            corpus_root=source_root,
            expected_signing_pubkey_hash=signer_hash,
        )
    except Exception as exc:
        raise FullParityError(
            "signed acceptance corpus does not match candidate release"
        ) from exc
    listed_files = {
        Path(str(item.get("artifact_path") or "")).as_posix()
        for item in source_value.get("fixtures") or ()
        if isinstance(item, Mapping)
    }
    discovered_files = {
        path.relative_to(source_root).as_posix() for path in files
    }
    expected_directories = {
        parent.as_posix()
        for listed_file in listed_files
        for parent in Path(listed_file).parents
        if parent != Path(".")
    }
    discovered_directories = {
        path.relative_to(source_root).as_posix() for path in directories
    }
    if (
        not listed_files
        or listed_files != discovered_files
        or expected_directories != discovered_directories
        or len(listed_files) != len(source_value.get("fixtures") or ())
    ):
        raise FullParityError("signed acceptance corpus file set differs")

    destination_config = Path(destination_config_dir)
    destination_manifest = destination_config / "acceptance-corpus-v2.json"
    destination_root = destination_config / "acceptance-corpus-v2"
    try:
        destination_metadata = destination_config.lstat()
    except OSError as exc:
        raise FullParityError(
            "run-owned acceptance corpus destination is unavailable"
        ) from exc
    if (
        destination_config.is_symlink()
        or not stat.S_ISDIR(destination_metadata.st_mode)
        or stat.S_IMODE(destination_metadata.st_mode) != 0o700
        or destination_metadata.st_uid != os.getuid()
        or destination_manifest.exists()
        or destination_root.exists()
    ):
        raise FullParityError("run-owned acceptance corpus destination differs")

    try:
        destination_root.mkdir(mode=0o700)
        for source_directory in sorted(
            directories,
            key=lambda path: (len(path.relative_to(source_root).parts), str(path)),
        ):
            destination_directory = destination_root / source_directory.relative_to(
                source_root
            )
            destination_directory.mkdir(mode=0o700)
            destination_directory.chmod(0o700)
        for source_file in sorted(files):
            _copy_acceptance_file(
                source_file,
                destination_root / source_file.relative_to(source_root),
            )
        _copy_acceptance_file(source_manifest, destination_manifest)
        destination_value = load_and_validate_acceptance_corpus_v2(
            destination_manifest,
            corpus_root=destination_root,
            expected_signing_pubkey_hash=signer_hash,
        )
        destination_directories, destination_files = _acceptance_corpus_tree(
            destination_root,
            owner_uid=os.getuid(),
            owner_gid=os.getgid(),
        )
        destination_manifest_metadata = destination_manifest.lstat()
        if (
            source_value != destination_value
            or len(destination_directories) != len(directories)
            or len(destination_files) != len(files)
            or destination_manifest_metadata.st_uid != os.getuid()
            or destination_manifest_metadata.st_gid != os.getgid()
            or stat.S_IMODE(destination_manifest_metadata.st_mode) != 0o600
        ):
            raise FullParityError("copied acceptance corpus identity differs")
    except Exception:
        shutil.rmtree(destination_root, ignore_errors=True)
        destination_manifest.unlink(missing_ok=True)
        raise
    return {
        "candidate_sha": candidate_sha,
        "release_hash": release["release_hash"],
        "manifest_hash": source_value["manifest_hash"],
        "fixture_count": len(files),
        "copied_exact": True,
    }


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


def _validated_baked_gateway_private_key_path() -> str:
    """Return the AMI-baked gateway key path without reading its contents."""

    path = PRODUCTION_GATEWAY_PRIVATE_KEY_PATH
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FullParityError("baked gateway private-key path is unavailable") from exc
    if (
        resolved != path
        or path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or (metadata.st_uid, metadata.st_gid)
        != PRODUCTION_GATEWAY_PRIVATE_KEY_OWNER
    ):
        raise FullParityError("baked gateway private-key path identity differs")
    return str(path)


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


async def _run_clone_controls_child(request: Mapping[str, Any]) -> dict[str, Any]:
    if (
        request.get("schema_version")
        != "leadpoet.production_parity_clone_controls_request.v1"
        or not SHA_RE.fullmatch(str(request.get("candidate_sha") or ""))
        or not RUN_RE.fullmatch(str(request.get("run_id") or ""))
        or not ARTIFACT_BUCKET_RE.fullmatch(
            str(request.get("artifact_bucket") or "")
        )
    ):
        raise FullParityError("clone controls request is invalid")
    candidate_sha = str(request["candidate_sha"])
    run_id = str(request["run_id"])
    artifact_bucket = str(request["artifact_bucket"])
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
        from gateway.research_lab.maintenance import (
            set_autoresearch_maintenance_paused,
            set_scoring_maintenance_paused,
        )

        scoring = await set_scoring_maintenance_paused(
            paused=False,
            reason="production_parity_full_rebenchmark",
            actor_ref="system:production-parity",
            event_doc={"production_parity": True},
        )
        autoresearch = await set_autoresearch_maintenance_paused(
            paused=True,
            reason="production_parity_no_miner_or_candidate_activity",
            actor_ref="system:production-parity",
            event_doc={"production_parity": True},
        )
    scoring_event_id = str(scoring.get("event_id") or "")
    autoresearch_event_id = str(autoresearch.get("event_id") or "")
    if (
        not 1 <= len(scoring_event_id) <= 128
        or not 1 <= len(autoresearch_event_id) <= 128
    ):
        raise FullParityError("clone controls did not persist bounded events")
    return {
        "schema_version": "leadpoet.production_parity_clone_controls_evidence.v1",
        "candidate_sha": candidate_sha,
        "run_id": run_id,
        "artifact_bucket": artifact_bucket,
        "scoring_event_id": scoring_event_id,
        "autoresearch_event_id": autoresearch_event_id,
        "scoring_paused": False,
        "autoresearch_paused": True,
    }


def _run_clone_controls(
    *,
    candidate_sha: str,
    run_id: str,
    supabase_origin: str,
    gateway_env_file: Path,
    artifact_bucket: str,
) -> dict[str, Any]:
    _validated_clone_environment(
        gateway_env_file,
        candidate_sha=candidate_sha,
        run_id=run_id,
        supabase_origin=supabase_origin,
        artifact_bucket=artifact_bucket,
    )
    request = {
        "schema_version": "leadpoet.production_parity_clone_controls_request.v1",
        "candidate_sha": candidate_sha,
        "run_id": run_id,
        "artifact_bucket": artifact_bucket,
        "supabase_origin": supabase_origin,
        "gateway_env_file": str(gateway_env_file),
    }
    result = _run(
        [sys.executable, str(Path(__file__).resolve()), "--clone-controls-child"],
        timeout=180,
        env=_clone_child_environment(),
        input_text=json.dumps(request, separators=(",", ":")),
    )
    if result.returncode != 0 or len(result.stdout or "") > 64 * 1024:
        raise FullParityError("clone controls child failed closed")
    evidence = _last_json_document(
        result.stdout or "", field="clone controls child"
    )
    if (
        evidence.get("schema_version")
        != "leadpoet.production_parity_clone_controls_evidence.v1"
        or evidence.get("candidate_sha") != candidate_sha
        or evidence.get("run_id") != run_id
        or evidence.get("artifact_bucket") != artifact_bucket
        or evidence.get("scoring_paused") is not False
        or evidence.get("autoresearch_paused") is not True
        or not str(evidence.get("scoring_event_id") or "")
        or not str(evidence.get("autoresearch_event_id") or "")
    ):
        raise FullParityError("clone controls child evidence is incomplete")
    return evidence


def _gateway_json(path: str) -> dict[str, Any]:
    with urlopen("http://127.0.0.1:8000" + path, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise FullParityError(f"gateway response is invalid: {path}")
    return value


def _report_document(value: Mapping[str, Any]) -> Mapping[str, Any]:
    report = value.get("report_doc")
    return report if isinstance(report, Mapping) else value


def _rebenchmark_identity(
    value: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and identify one published result using candidate policy."""

    report = _report_document(value)
    expected_counts = {
        "public": int(policy.get("public_total_icps") or 0),
        "private": int(policy.get("private_total_icps") or 0),
        "conditional": int(policy.get("conditional_total_icps") or 0),
    }
    expected_total = sum(expected_counts.values())
    public_weak = int(policy.get("public_weak_total") or 0)
    private_weak = int(policy.get("private_weak_total") or 0)
    if (
        any(count <= 0 for count in expected_counts.values())
        or public_weak < 0
        or public_weak > expected_counts["public"]
        or private_weak < 0
        or private_weak > expected_counts["private"]
    ):
        raise FullParityError("candidate ICP policy is invalid")
    split = report.get("visibility_split")
    if not isinstance(split, Mapping):
        raise FullParityError("published rebenchmark assignment is missing")
    observed_counts = {
        "public": int(split.get("public_count") or 0),
        "private": int(split.get("private_count") or 0),
        "conditional": int(split.get("conditional_count") or 0),
    }
    try:
        aggregate_score = float(
            value.get("aggregate_score")
            if value.get("aggregate_score") is not None
            else report.get("aggregate_score")
        )
    except (TypeError, ValueError) as exc:
        raise FullParityError("published rebenchmark score is invalid") from exc
    public_strength = split.get("public_strength_counts")
    private_strength = split.get("private_strength_counts")
    expected_public_strength = {
        "strong": expected_counts["public"] - public_weak,
        "weak": public_weak,
    }
    expected_private_strength = {
        "strong": expected_counts["private"] - private_weak,
        "weak": private_weak,
    }
    expected_public_strength = {
        key: count
        for key, count in expected_public_strength.items()
        if count > 0
    }
    expected_private_strength = {
        key: count
        for key, count in expected_private_strength.items()
        if count > 0
    }
    report_hash = str(report.get("report_public_hash") or "")
    report_without_hash = {
        key: item for key, item in report.items() if key != "report_public_hash"
    }
    if (
        value.get("current_report_status") != "published"
        or value.get("benchmark_quality") != "passed"
        or report.get("report_type") != "research_lab_public_daily_benchmark"
        or int(report.get("item_count") or 0) != expected_total
        or observed_counts != expected_counts
        or dict(public_strength or {}) != expected_public_strength
        or dict(private_strength or {}) != expected_private_strength
        or str(split.get("split_policy") or "")
        != str(policy.get("selection_policy") or "")
        or str(split.get("rolling_window_hash") or "")
        != str(report.get("rolling_window_hash") or "")
        or not 0.0 <= aggregate_score <= 100.0
        or not HASH_RE.fullmatch(report_hash)
        or report_hash != sha256_json(report_without_hash)
    ):
        raise FullParityError(
            "published rebenchmark score or assignment differs from candidate policy"
        )
    identity = {
        "report_id": str(value.get("report_id") or ""),
        "benchmark_bundle_id": str(value.get("benchmark_bundle_id") or ""),
        "benchmark_date": str(value.get("benchmark_date") or ""),
        "rolling_window_hash": str(value.get("rolling_window_hash") or ""),
        "private_model_artifact_hash": str(
            value.get("private_model_artifact_hash") or ""
        ),
        "private_model_manifest_hash": str(
            value.get("private_model_manifest_hash") or ""
        ),
        "aggregate_score": aggregate_score,
        "item_count": expected_total,
        "report_public_hash": report_hash,
        "category_counts": observed_counts,
        "public_strength_counts": dict(public_strength or {}),
        "private_strength_counts": dict(private_strength or {}),
    }
    if (
        not identity["report_id"]
        or not identity["benchmark_bundle_id"]
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", identity["benchmark_date"])
        or not HASH_RE.fullmatch(identity["rolling_window_hash"])
        or not HASH_RE.fullmatch(identity["private_model_artifact_hash"])
        or not HASH_RE.fullmatch(identity["private_model_manifest_hash"])
    ):
        raise FullParityError("published rebenchmark identity is incomplete")
    return identity


def _contains_dashboard_identity(
    value: Any,
    identity: Mapping[str, Any],
) -> bool:
    """Match the public subnet-dashboard projection to a durable result."""

    if not isinstance(value, Mapping) or value.get("success") is not True:
        return False
    data = value.get("data")
    benchmark = data.get("benchmark") if isinstance(data, Mapping) else None
    if not isinstance(benchmark, Mapping):
        return False
    try:
        score_matches = float(benchmark.get("aggregateScore")) == float(
            identity["aggregate_score"]
        )
        count_matches = int(benchmark.get("itemCount")) == int(
            identity["item_count"]
        )
    except (TypeError, ValueError):
        return False
    return (
        str(benchmark.get("reportId") or "") == identity["report_id"]
        and str(benchmark.get("benchmarkDate") or "")
        == identity["benchmark_date"]
        and str(benchmark.get("rollingWindowHash") or "")
        == identity["rolling_window_hash"]
        and score_matches
        and count_matches
    )


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


def _clone_service_role_key(
    values: Mapping[str, str],
    *,
    candidate_sha: str,
    run_id: str,
    supabase_origin: str,
    jwt_secret: str,
) -> str:
    """Verify the run-scoped 48-hour clone token used after rebenchmarking."""

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
        raise FullParityError("run-scoped clone service role identity differs")
    token = _required_secret_from_environment(
        values,
        ("SUPABASE_SERVICE_ROLE_KEY",),
        field="run-scoped clone service role credential",
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
            "run-scoped clone service role credential is invalid"
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
        or payload.get("role") != "service_role"
        or expires_at - issued_at != 172_805
        or issued_at > now
        or expires_at <= now
        or not hmac.compare_digest(signature, expected_signature)
    ):
        raise FullParityError(
            "run-scoped clone service role credential identity differs"
        )
    return token


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


def _run_miner_intake_path(
    *,
    region: str,
    candidate_sha: str,
    run_id: str,
    supabase_origin: str,
    gateway_env_file: Path,
    artifact_bucket: str,
    production_gateway_environment: Mapping[str, str],
    miner_intake_secret: str,
) -> dict[str, Any]:
    runtime_credential = _required_secret_from_environment(
        production_gateway_environment,
        OPENROUTER_RUNTIME_CREDENTIAL_REFS,
        field="production OpenRouter runtime credential",
    )
    management_credential = _required_secret_from_environment(
        production_gateway_environment,
        OPENROUTER_MANAGEMENT_CREDENTIAL_REFS,
        field="production OpenRouter management credential",
    )
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
        "openrouter_runtime_credential": runtime_credential,
        "openrouter_management_credential": management_credential,
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
    runtime_credential = management_credential = builtwith_credential = ""
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
        or evidence.get("openrouter", {}).get("admitted") is not True
        or evidence.get("source_add", {}).get("admitted") is not True
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
    from gateway.research_lab.maintenance import (
        get_autoresearch_maintenance_state,
        get_scoring_maintenance_state,
        set_autoresearch_maintenance_paused,
        set_scoring_maintenance_paused,
    )
    from gateway.research_lab.source_add_workflow import source_add_control_state
    from gateway.research_lab.store import call_rpc, select_many, select_one
    from leadpoet_canonical.credential_recipient_v2 import (
        verify_and_encrypt_openrouter_credential_v2,
        verify_openrouter_credential_release_v2,
    )
    from neurons.miner import (
        _research_lab_openrouter_key_signed_payload,
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
    runtime_credential = str(request.get("openrouter_runtime_credential") or "").strip()
    management_credential = str(
        request.get("openrouter_management_credential") or ""
    ).strip()
    builtwith_credential = str(request.get("builtwith_credential") or "").strip()
    if not runtime_credential or not management_credential or not builtwith_credential:
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
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://production-parity.invalid",
            timeout=httpx.Timeout(180.0),
        ) as client:
            timestamp = int(time.time())
            key_fingerprint = hashlib.sha256(
                f"{runtime_credential}:{management_credential}".encode("utf-8")
            ).hexdigest()[:24]
            recipient_payload = _research_lab_signed_payload(
                wallet,
                {
                    "miner_hotkey": miner_hotkey,
                    "timestamp": timestamp,
                    "idempotency_key": (
                        f"research-openrouter-recipient:{miner_hotkey}:"
                        f"{key_fingerprint}"
                    ),
                },
            )
            recipient_response = await client.post(
                "/research-lab/openrouter-keys/credential-recipient",
                json=recipient_payload,
            )
            if recipient_response.status_code != 200:
                raise FullParityError(
                    "OpenRouter credential recipient rejected the exact miner request"
                )
            chain_check_milestones.append(len(observed_chain_checks))
            recipients = recipient_response.json()
            verified_coordinator_boot = verify_openrouter_credential_release_v2(
                recipients["release_evidence"]
            )
            encrypted_runtime = verify_and_encrypt_openrouter_credential_v2(
                recipients["runtime"],
                runtime_credential,
                miner_hotkey=miner_hotkey,
                credential_kind="runtime",
                verified_coordinator_boot_identity=verified_coordinator_boot,
            )
            encrypted_management = verify_and_encrypt_openrouter_credential_v2(
                recipients["management"],
                management_credential,
                miner_hotkey=miner_hotkey,
                credential_kind="management",
                verified_coordinator_boot_identity=verified_coordinator_boot,
            )
            register_payload = _research_lab_openrouter_key_signed_payload(
                wallet,
                {
                    "miner_hotkey": miner_hotkey,
                    "timestamp": int(time.time()),
                    "idempotency_key": (
                        f"research-openrouter-key:{miner_hotkey}:"
                        f"{key_fingerprint}"
                    ),
                    "openrouter_api_key_v2": encrypted_runtime,
                    "openrouter_management_key_v2": encrypted_management,
                    "key_label": "production-parity-miner-intake",
                },
            )
            register_response = await client.post(
                "/research-lab/openrouter-keys", json=register_payload
            )
            if register_response.status_code != 200:
                raise FullParityError(
                    "OpenRouter registration rejected the exact miner request"
                )
            chain_check_milestones.append(len(observed_chain_checks))
            register_result = register_response.json()
            key_ref = str(register_result.get("key_ref") or "")
            key_row = await select_one(
                "research_lab_openrouter_key_refs",
                filters=(("key_ref", key_ref),),
            )
            envelope_rows = await select_many(
                "research_lab_provider_credential_envelopes_v2",
                filters=(("key_ref", key_ref),),
                limit=3,
            )
            openrouter_persistence = json.dumps(
                {
                    "response": register_result,
                    "key_ref": key_row,
                    "envelopes": envelope_rows,
                },
                sort_keys=True,
                default=str,
            )
            if (
                not key_ref
                or not isinstance(key_row, Mapping)
                or key_row.get("preflight_status") != "passed"
                or len(envelope_rows) != 2
                or {str(row.get("credential_kind") or "") for row in envelope_rows}
                != {"runtime", "management"}
                or runtime_credential in openrouter_persistence
                or management_credential in openrouter_persistence
            ):
                raise FullParityError(
                    "OpenRouter registration persistence is incomplete"
                )

            builtwith_probe = _verify_builtwith_credential_live(
                builtwith_credential
            )
            autoresearch_state = await get_autoresearch_maintenance_state()
            scoring_state = await get_scoring_maintenance_state()
            source_state = await source_add_control_state()
            source_controls = {
                "autoresearch_paused": bool(autoresearch_state.get("paused")),
                "scoring_paused": bool(scoring_state.get("paused")),
                "source_add_paused": bool(source_state.get("paused", True)),
            }
            if source_controls["autoresearch_paused"]:
                await set_autoresearch_maintenance_paused(
                    paused=False,
                    reason="production_parity_miner_intake",
                    actor_ref="system:production-parity",
                    event_doc={"production_parity": True},
                )
            if source_controls["scoring_paused"]:
                await set_scoring_maintenance_paused(
                    paused=False,
                    reason="production_parity_miner_intake",
                    actor_ref="system:production-parity",
                    event_doc={"production_parity": True},
                )
            if source_controls["source_add_paused"]:
                await call_rpc(
                    "research_lab_source_add_set_paused",
                    {
                        "p_paused": False,
                        "p_reason": "production_parity_miner_intake",
                        "p_actor_ref": "system:production-parity",
                    },
                )
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
                or runtime_credential in source_persistence
                or management_credential in source_persistence
            ):
                raise FullParityError("SOURCE_ADD admission persistence is incomplete")

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
            or len(chain_check_milestones) != 4
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
            "provider_security_write": "exact-idempotent-logging-disable",
            "openrouter": {
                "real_production_credentials": True,
                "recipient_attestation_verified": True,
                "miner_signature_verified": True,
                "measured_provider_preflight_passed": True,
                "admitted": True,
                "key_ref_persisted": True,
                "credential_envelope_count": 2,
                "plaintext_absent": True,
            },
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
            },
        }
    finally:
        research_lab_api.chain_is_hotkey_registered = original_chain_registration
        try:
            await _restore_miner_intake_controls(
                source_controls,
                call_rpc=call_rpc,
                set_autoresearch_maintenance_paused=(
                    set_autoresearch_maintenance_paused
                ),
                set_scoring_maintenance_paused=set_scoring_maintenance_paused,
            )
        finally:
            request = {}
            runtime_credential = management_credential = builtwith_credential = ""


async def _restore_miner_intake_controls(
    source_controls: Mapping[str, Any],
    *,
    call_rpc: Any,
    set_autoresearch_maintenance_paused: Any,
    set_scoring_maintenance_paused: Any,
) -> None:
    """Restore every clone-local maintenance state changed for miner intake."""

    if source_controls.get("source_add_paused"):
        await call_rpc(
            "research_lab_source_add_set_paused",
            {
                "p_paused": True,
                "p_reason": "production_parity_miner_intake_complete",
                "p_actor_ref": "system:production-parity",
            },
        )
    if source_controls.get("autoresearch_paused"):
        await set_autoresearch_maintenance_paused(
            paused=True,
            reason="production_parity_miner_intake_complete",
            actor_ref="system:production-parity",
            event_doc={"production_parity": True},
        )
    if source_controls.get("scoring_paused"):
        await set_scoring_maintenance_paused(
            paused=True,
            reason="production_parity_miner_intake_complete",
            actor_ref="system:production-parity",
            event_doc={"production_parity": True},
        )


def _wait_rebenchmark(
    *,
    candidate_sha: str,
    secret_id: str,
    region: str,
    run_id: str,
    supabase_origin: str,
    artifact_bucket: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        remaining = max(1, math.ceil(deadline - time.monotonic()))
        try:
            result = _run(
                [
                    sys.executable,
                    str(ROOT / "scripts/check_production_parity_rebenchmark.py"),
                    "--root",
                    str(ROOT),
                    "--candidate-sha",
                    candidate_sha,
                    "--secret-id",
                    secret_id,
                    "--region",
                    region,
                    "--run-id",
                    run_id,
                    "--supabase-origin",
                    supabase_origin,
                    "--artifact-bucket",
                    artifact_bucket,
                ],
                timeout=min(remaining, 300),
                env=_clone_child_environment(region=region),
            )
        except subprocess.TimeoutExpired:
            last = {"reason": "clone_readiness_child_timeout"}
            sleep_seconds = min(30, max(0, deadline - time.monotonic()))
            if sleep_seconds:
                time.sleep(sleep_seconds)
            continue
        if len(result.stdout or "") > 256 * 1024:
            raise FullParityError("clone readiness child output is unbounded")
        if result.returncode not in {0, 2}:
            raise FullParityError("clone readiness child failed closed")
        last = _last_json_document(
            result.stdout or "", field="clone readiness child"
        )
        if (
            last.get("schema_version")
            != "leadpoet.production_parity_rebenchmark_readiness.v1"
            or last.get("candidate_sha") != candidate_sha
            or last.get("artifact_bucket") != artifact_bucket
        ):
            raise FullParityError("clone readiness child identity differs")
        if result.returncode == 0 and last.get("available") is True:
            return last
        if result.returncode != 2 or last.get("available") is not False:
            raise FullParityError("clone readiness child status differs")
        sleep_seconds = min(30, max(0, deadline - time.monotonic()))
        if sleep_seconds:
            time.sleep(sleep_seconds)
    raise FullParityError(
        "full rebenchmark did not publish before timeout: "
        + str(last.get("reason") or "unknown")
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
    supabase_origin = _validated_public_origin(supabase_origin)
    _checkout_identity(candidate_sha)
    try:
        boot_state = EARLY_BOOT_MARKER.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise FullParityError(
            "transient host did not prove early production-service isolation"
        ) from exc
    if boot_state != "isolated":
        raise FullParityError(
            "transient host early production-service isolation differs"
        )
    gateway_private_key_path = _validated_baked_gateway_private_key_path()
    started = time.monotonic()
    deadline = _full_deadline(
        started=started,
        timeout_seconds=timeout_seconds,
    )
    work = FULL_WORK_ROOT / run_id / "runtime"
    work.mkdir(parents=True, mode=0o700, exist_ok=False)
    scoring_cache = work / "scoring-cache"
    scoring_cache.mkdir(mode=0o700)
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
        failure_stage = "acceptance-corpus"
        release_channel = fetch_release_channel_v2(
            bucket=ATTESTED_V2_RELEASE_BUCKET,
            commit_sha=candidate_sha,
            prefix=ATTESTED_V2_RELEASE_PREFIX,
            s3_client=boto3.client("s3", region_name=region),
        )
        acceptance_corpus = _materialize_acceptance_corpus(
            source_config_dir=PRODUCTION_V2_CONFIG_DIR,
            destination_config_dir=artifact_policy.parent,
            candidate_sha=candidate_sha,
            candidate_release_manifest=release_channel[
                "gateway_release_manifest"
            ],
        )
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
                "GATEWAY_RESTART_CONTROLLER_ROOT": str(work / "restart-controller"),
                "GATEWAY_DEPLOYMENT_DIR": str(work / "deployments"),
                "GATEWAY_HOST_RESTART_SCRIPT": str(ROOT / "gw_restart.sh"),
                "GATEWAY_TEE_EIF_ROOT": str(work / "tee"),
                "GATEWAY_V2_CONFIG_DIR": str(work / "v2-config"),
                "GATEWAY_V2_ACCEPTANCE_CORPUS_MANIFEST": str(
                    work / "v2-config" / "acceptance-corpus-v2.json"
                ),
                "GATEWAY_V2_ACCEPTANCE_CORPUS_ROOT": str(
                    work / "v2-config" / "acceptance-corpus-v2"
                ),
                "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT": str(work / "offline-artifacts"),
                "VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT": str(work / "offline-artifacts" / "validator-runtime"),
                "GATEWAY_RESTART_LOCK_FILE": str(work / "gateway-restart.lock"),
                "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(work / "docker-operation.lock"),
                "GATEWAY_DEPLOY_COMMIT": candidate_sha,
                "GATEWAY_PYTHON_BIN": "/home/ec2-user/venv311/bin/python3",
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

        failure_stage = "clone-controls"
        controls = _run_clone_controls(
            candidate_sha=candidate_sha,
            run_id=run_id,
            supabase_origin=supabase_origin,
            gateway_env_file=gateway_env_file,
            artifact_bucket=artifact_bucket,
        )
        failure_stage = "rebenchmark-publication"
        rebenchmark = _wait_rebenchmark(
            candidate_sha=candidate_sha,
            secret_id=secret_state["secret_id"],
            region=region,
            run_id=run_id,
            supabase_origin=supabase_origin,
            artifact_bucket=artifact_bucket,
            timeout_seconds=_remaining_full_timeout(
                deadline=deadline,
                stage="full rebenchmark publication",
            ),
        )

        parsed_env = _validated_clone_environment(
            gateway_env_file,
            candidate_sha=candidate_sha,
            run_id=run_id,
            supabase_origin=supabase_origin,
            artifact_bucket=artifact_bucket,
        )
        failure_stage = "weight-readiness"
        readiness = _run(
            [
                "/home/ec2-user/venv311/bin/python3",
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
        production_gateway_environment = _parse_environment_document(
            _secret_value(
                secrets_client,
                production_gateway_secret_id,
                field="production gateway environment",
            ),
            field="production gateway environment",
        )
        failure_stage = "miner-intake"
        miner_intake = _run_miner_intake_path(
            region=region,
            candidate_sha=candidate_sha,
            run_id=run_id,
            supabase_origin=supabase_origin,
            gateway_env_file=gateway_env_file,
            artifact_bucket=artifact_bucket,
            production_gateway_environment=production_gateway_environment,
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
                "controls": controls,
                "acceptance_corpus": acceptance_corpus,
                "external_write_boundaries": {
                    "arweave": "blocked-production-parity",
                },
                "rebenchmark": rebenchmark,
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
        "--clone-controls-child",
        "--clone-handoff-child",
        "--miner-intake-child",
    }
    if len(raw_args) == 1 and raw_args[0] in child_modes:
        child_mode = raw_args[0]
        logging.disable(logging.CRITICAL)
        try:
            request = _read_child_request()
            if child_mode == "--clone-controls-child":
                result = asyncio.run(_run_clone_controls_child(request))
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
    try:
        result = run_full(
            region=args.region,
            run_id=args.run_id,
            base_sha=args.base_sha.lower(),
            candidate_sha=args.candidate_sha.lower(),
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
    except (OSError, ValueError, ProductionParityError, FullParityError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
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
