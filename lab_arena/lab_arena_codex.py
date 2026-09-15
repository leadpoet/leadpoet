"""Codex inside Arena: private HTTP -> existing paid-provider Unix socket.

This module is mounted beside lab_arena_checkpoint for submitted harnesses.
It uses only the standard library and never accepts a provider credential.
The gateway validates, reserves and settles every request before this bridge
replays the completed response as Codex's expected SSE events.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
import signal
import socket
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

CODEX_VERSION = "0.154.0"
CODEX_BINARY = "/usr/local/bin/codex"
MAX_REQUEST_BYTES = 1_000_000
MAX_RESPONSE_BYTES = 4 * 1_048_576
MAX_LOG_BYTES = 64 * 1024
SOCKET_TIMEOUT_SECONDS = 185


class CodexRuntimeError(RuntimeError):
    """A bounded runtime failure, with no provider or account detail."""

    def __init__(self, message: str, *, diagnostics: str = "") -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _receive(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = connection.recv(size - len(chunks))
        if not part:
            raise CodexRuntimeError("worker response truncated")
        chunks.extend(part)
    return bytes(chunks)


def _dispatch(socket_path: str, parameters: dict[str, Any]) -> tuple[int, bytes]:
    payload = json.dumps({
        "schema_version": "leadpoet.lab_arena.operation_frame.v1",
        "operation_id": "openrouter.responses",
        "parameters": parameters,
        "timeout_ms": 120_000,
    }, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > 1_048_576:
        raise CodexRuntimeError("request too large")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(SOCKET_TIMEOUT_SECONDS)
        connection.connect(socket_path)
        connection.sendall(len(payload).to_bytes(4, "big") + payload)
        size = int.from_bytes(_receive(connection, 4), "big")
        if not 0 < size <= MAX_RESPONSE_BYTES:
            raise CodexRuntimeError("invalid worker response")
        response = json.loads(_receive(connection, size))
    if not isinstance(response, dict) or "error" in response:
        raise CodexRuntimeError("broker refused request")
    status = response.get("status")
    if type(status) is not int or not 100 <= status <= 599:
        raise CodexRuntimeError("invalid worker status")
    return status, base64.b64decode(response["body_b64"], validate=True)


def response_events(document: dict[str, Any]) -> Iterator[bytes]:
    """Replay terminal Responses items without translating their tool protocol."""

    if (document.get("object") != "response"
            or document.get("status") not in ("completed", "incomplete", "failed")
            or not isinstance(document.get("output"), list)):
        raise CodexRuntimeError("invalid Responses reply")
    sequence = 0

    def event(kind: str, **fields: Any) -> bytes:
        nonlocal sequence
        data = dict(type=kind, sequence_number=sequence, **fields)
        sequence += 1
        return ("event: " + kind + "\ndata: " + json.dumps(data, separators=(",", ":")) + "\n\n").encode()

    yield event("response.created", response=dict(document, status="in_progress", output=[]))
    for index, item in enumerate(document["output"]):
        if not isinstance(item, dict):
            raise CodexRuntimeError("invalid Responses item")
        yield event("response.output_item.added", output_index=index, item=item)
        if item.get("type") == "message":
            if not isinstance(item.get("content"), list):
                raise CodexRuntimeError("invalid Responses message")
            for part_index, part in enumerate(item.get("content", [])):
                if not isinstance(part, dict):
                    raise CodexRuntimeError("invalid Responses content")
                if part.get("type") == "output_text":
                    fields = dict(item_id=item.get("id", ""), output_index=index, content_index=part_index)
                    yield event("response.content_part.added", **fields, part=dict(part, text=""))
                    yield event("response.output_text.delta", **fields, delta=part["text"])
                    yield event("response.output_text.done", **fields, text=part["text"])
                    yield event("response.content_part.done", **fields, part=part)
        yield event("response.output_item.done", output_index=index, item=item)
    yield event("response." + document["status"], response=document)


class ResponsesBridge:
    """Attempt-local listener. The only upstream is the bound worker socket."""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        self.token = secrets.token_urlsafe(32)
        self._active = threading.BoundedSemaphore(1)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def reply(self, status: int, body: bytes, content_type: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                if self.path != "/v1/responses":
                    self.reply(404, b'{"error":{"message":"unsupported endpoint"}}')
                    return
                if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + owner.token):
                    self.reply(401, b'{"error":{"message":"invalid bridge token"}}')
                    return
                if not owner._active.acquire(blocking=False):
                    self.reply(429, b'{"error":{"message":"request already in progress"}}')
                    return
                try:
                    self.connection.settimeout(SOCKET_TIMEOUT_SECONDS)
                    lengths = self.headers.get_all("Content-Length", [])
                    if len(lengths) != 1 or self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding"):
                        raise ValueError("unsupported body encoding")
                    size = int(lengths[0])
                    if not 0 < size <= MAX_REQUEST_BYTES:
                        raise ValueError("invalid body size")
                    raw = self.rfile.read(size)
                    if len(raw) != size:
                        raise ValueError("truncated body")
                    body = json.loads(raw)
                    if not isinstance(body, dict):
                        raise ValueError("invalid body")
                    streaming = body.pop("stream", False)
                    if type(streaming) is not bool:
                        raise ValueError("invalid stream")
                    # Codex sends these explicit stateless settings. No server
                    # history, background task, extra routing or headers cross.
                    if body.pop("store", False) is not False or body.pop("previous_response_id", None) is not None:
                        raise ValueError("stateful request")
                    # Codex client telemetry is not part of OpenRouter's API.
                    body.pop("client_metadata", None)
                    status, response = _dispatch(owner.socket_path, body)
                    if 200 <= status < 300 and streaming:
                        response = b"".join(response_events(json.loads(response)))
                        self.reply(status, response, "text/event-stream")
                    else:
                        self.reply(status, response)
                except (ValueError, TypeError, KeyError):
                    self.reply(400, b'{"error":{"message":"invalid Responses request or reply"}}')
                except (OSError, CodexRuntimeError):
                    self.reply(502, b'{"error":{"message":"Arena broker unavailable"}}')
                finally:
                    owner._active.release()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:%d/v1" % self.server.server_port

    def __enter__(self) -> "ResponsesBridge":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@contextmanager
def session(*, model: str, reasoning_effort: str = "medium") -> Iterator[dict[str, str]]:
    """Yield an isolated child environment for a Codex CLI or SDK launcher.

    Call only inside an Arena execute sandbox with private loopback enabled.
    Provider keys and the caller's Codex login/configuration are not inherited.
    """

    socket_path = os.environ.get("LAB_ARENA_WORKER_SOCKET")
    if not socket_path or not os.environ.get("LAB_ARENA_WEB_EGRESS_SOCKET"):
        raise CodexRuntimeError("Codex requires an Arena execute sandbox with private loopback")
    if not model or "/" not in model or reasoning_effort not in ("none", "minimal", "low", "medium", "high", "xhigh"):
        raise CodexRuntimeError("invalid Codex model or reasoning effort")
    with tempfile.TemporaryDirectory(prefix="arena-codex-") as directory, ResponsesBridge(socket_path) as bridge:
        home = Path(directory)
        config = '\n'.join([
            "model = " + json.dumps(model),
            'model_provider = "arena"',
            "model_reasoning_effort = " + json.dumps(reasoning_effort),
            'approval_policy = "never"',
            # gVisor owns isolation. Nested platform sandboxes cannot run here.
            'sandbox_mode = "danger-full-access"',
            'web_search = "disabled"',
            'check_for_update_on_startup = false',
            '[features]',
            'enable_request_compression = false',
            'shell_snapshot = false',
            'multi_agent = false',
            'apps = false',
            '[model_providers.arena]',
            'name = "Arena"',
            'base_url = ' + json.dumps(bridge.base_url),
            'env_key = "LAB_ARENA_CODEX_TOKEN"',
            'wire_api = "responses"',
            'requires_openai_auth = false',
            'supports_websockets = false',
            'request_max_retries = 0',
            'stream_max_retries = 0',
            'stream_idle_timeout_ms = 200000',
            '',
        ])
        (home / "config.toml").write_text(config, encoding="utf-8")
        allowed = {"PATH", "LANG", "TZ", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"}
        environment = {key: value for key, value in os.environ.items() if key in allowed or key.startswith("LAB_ARENA_")}
        # operations.SCRAPINGDOG_RUNTIME_HANDLE is a public, non-secret SDK
        # placeholder already installed in the sandbox by runtime.py. Retain
        # only that exact literal; an actual provider key is never inherited.
        if os.environ.get("SCRAPINGDOG_API_KEY") == "lab-arena-brokered-scrapingdog":
            environment["SCRAPINGDOG_API_KEY"] = "lab-arena-brokered-scrapingdog"
        environment.update(CODEX_HOME=directory, HOME=directory, LAB_ARENA_CODEX_TOKEN=bridge.token, LAB_ARENA_CODEX_BASE_URL=bridge.base_url)
        # Shell-launched Python tools need the same admitted imports as the
        # parent harness, without loading the trusted scorer's sitecustomize.
        agent_dir = Path(__file__).resolve().parent
        environment.update(
            PYTHONPATH=os.pathsep.join(str(agent_dir / name) for name in (".", "source", "deps")),
            PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        )
        for name in ("NO_PROXY", "no_proxy"):
            environment[name] = ",".join(filter(None, [environment.get(name, ""), "127.0.0.1", "localhost"]))
        yield environment


def run(prompt: str, *, model: str, cwd: str | Path, reasoning_effort: str = "medium", timeout_seconds: float = 2700) -> str:
    """Run Codex once and return its final message; the harness owns JSON output."""

    if not isinstance(prompt, str) or not prompt or len(prompt.encode()) > MAX_REQUEST_BYTES:
        raise CodexRuntimeError("invalid Codex prompt")
    if not 0 < timeout_seconds <= 2700:
        raise CodexRuntimeError("invalid Codex timeout")
    with session(model=model, reasoning_effort=reasoning_effort) as environment:
        final_path = Path(environment["CODEX_HOME"]) / "final.txt"
        with tempfile.TemporaryFile() as prompt_file:
            prompt_file.write(prompt.encode())
            prompt_file.seek(0)
            process = subprocess.Popen(
                [CODEX_BINARY, "exec", "--skip-git-repo-check", "--ephemeral", "--color", "never", "-C", str(cwd), "-o", str(final_path), "-"],
                stdin=prompt_file, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=environment, start_new_session=True,
            )
            tail = bytearray()

            def drain() -> None:
                assert process.stdout is not None
                with process.stdout:
                    for block in iter(lambda: process.stdout.read(8192), b""):
                        tail.extend(block)
                        del tail[:-MAX_LOG_BYTES]

            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            try:
                process.wait(timeout=timeout_seconds)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
            finally:
                # A completed CLI must not leave shell descendants holding the
                # pipe or using the attempt's provider socket after return.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                reader.join(timeout=5)
            if process.returncode:
                raise CodexRuntimeError("Codex exited with status %d" % process.returncode,
                                        diagnostics=tail.decode("utf-8", errors="replace"))
        if not final_path.is_file() or final_path.stat().st_size > MAX_LOG_BYTES:
            raise CodexRuntimeError("Codex did not produce a bounded final message")
        return final_path.read_text(encoding="utf-8")
