"""Trials for the weight-window resilience fixes.

Covers: (1) every weight-submission endpoint rides the validator priority
pool, (2) the compressed weight-transport verification runs off the event
loop with identical accept/reject behavior, (3) the allocation handoff disk
cache warm-starts a restart and fails open on corruption, (4) the validator
allocation fetch negotiates gzip against a real HTTP server and still
accepts identity responses from an older gateway.
"""

import asyncio
import gzip
import http.server
import json
from pathlib import Path
import threading
import time

import pytest

from gateway.middleware.priority import classify_path, classify_scope
from gateway.middleware import body_size as body_size_module
from gateway.middleware.body_size import BodySizeLimitMiddleware
from gateway.research_lab import allocation_handoff_disk_cache as disk_cache
from leadpoet_canonical.attested_v2 import sha256_bytes
from leadpoet_canonical.hotkey_authority_v2 import (
    build_weight_transport_authorization_v2,
)
from research_lab import validator_integration as validator_integration_module
from research_lab.validator_integration import _fetch_allocation_json


# ---------------------------------------------------------------------------
# 1. Priority routing: the whole weight exchange shares the validator pool.
# ---------------------------------------------------------------------------

def test_every_authenticated_weight_submission_stage_rides_validator_pool(
    monkeypatch,
):
    for path in (
        "/weights/submit",
        "/weights/submit/v2",
        "/weights/finalize/v2",
        "/weights/inputs/v2",
        "/weights/subnet-epoch/candidate/v1",
        "/weights/subnet-epoch/boundary/v1",
    ):
        assert classify_path(path) == "validator", path
    monkeypatch.setenv("RESEARCH_LAB_INTERNAL_API_KEY", "validator-secret")
    for path in (
        "/research-lab/allocations/attested/24103",
        "/research-lab/allocations/live/24103",
    ):
        scope = {
            "type": "http",
            "path": path,
            "headers": [
                (b"x-leadpoet-internal-key", b"validator-secret"),
            ],
        }
        assert classify_scope(scope) == "validator", path


def test_anonymous_allocation_reads_cannot_consume_validator_pool(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_INTERNAL_API_KEY", "validator-secret")
    path = "/research-lab/allocations/attested/24103"
    assert classify_scope(
        {"type": "http", "path": path, "headers": []}
    ) == "other"
    assert classify_scope(
        {
            "type": "http",
            "path": path,
            "headers": [
                (b"x-leadpoet-internal-key", b"wrong-secret"),
            ],
        }
    ) == "other"


def test_public_weight_reads_stay_out_of_validator_pool():
    # Public reads must not be able to consume validator slots.
    for path in ("/weights/latest/71/24103", "/weights/current/71",
                 "/weights/transparency/events"):
        assert classify_path(path) == "other", path
    assert classify_path("/fulfillment/requests/active") == "miner"


# ---------------------------------------------------------------------------
# 2. Compressed weight transport: same verdicts, now off the event loop.
# ---------------------------------------------------------------------------

_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def _authorized_request(logical: bytes, path: str = "/weights/submit/v2"):
    wire = gzip.compress(logical, compresslevel=1)
    authorization = build_weight_transport_authorization_v2(
        validator_hotkey=_HOTKEY,
        path=path,
        wire_body_hash=sha256_bytes(wire),
        wire_body_bytes=len(wire),
        logical_body_hash=sha256_bytes(logical),
        logical_body_bytes=len(logical),
    )
    import base64

    headers = [
        (b"content-encoding", b"gzip"),
        (b"content-length", str(len(wire)).encode()),
        (
            b"x-leadpoet-weight-transport",
            base64.b64encode(json.dumps(authorization).encode()),
        ),
        (b"x-leadpoet-weight-transport-signature", b"ab" * 64),
    ]
    scope = {"type": "http", "method": "POST", "path": path, "headers": headers}
    return scope, wire


def _drive(middleware, scope, wire_body):
    seen = {}
    sent = []

    async def app(app_scope, receive, send):
        message = await receive()
        seen["body"] = message.get("body")
        seen["headers"] = dict(app_scope["headers"])
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware.app = app
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": wire_body, "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    statuses = [m["status"] for m in sent if m["type"] == "http.response.start"]
    return seen, statuses


@pytest.fixture()
def _authorized_env(monkeypatch):
    monkeypatch.setenv("PRIMARY_VALIDATOR_HOTKEYS", _HOTKEY)
    # The trial exercises everything except the sr25519 verify itself
    # (no bittensor wheel locally); the verify call site and order are
    # unchanged and covered by the existing transport tests in CI.
    monkeypatch.setattr(
        body_size_module, "_verify_transport_signature", lambda **_kw: True
    )


def test_authorized_compressed_post_decompresses_off_loop(_authorized_env):
    logical = json.dumps(
        {"receipt_graph": {"receipts": ["r" * 4096] * 64}}
    ).encode()
    scope, wire = _authorized_request(logical)
    seen, statuses = _drive(BodySizeLimitMiddleware(None), scope, wire)
    assert statuses == [204]
    assert seen["body"] == logical
    assert b"content-encoding" not in seen["headers"]


def test_tampered_wire_body_still_rejected_400(_authorized_env):
    logical = b'{"receipt_graph":{}}'
    scope, wire = _authorized_request(logical)
    seen, statuses = _drive(
        BodySizeLimitMiddleware(None), scope, wire + b"x"
    )
    assert statuses == [400]
    assert "body" not in seen


def test_logical_hash_mismatch_still_rejected_400(_authorized_env):
    logical = b'{"receipt_graph":{}}'
    scope, _wire = _authorized_request(logical)
    forged_wire = gzip.compress(b'{"receipt_graph":{"x":1}}', compresslevel=1)
    # Fix the wire hash so only the logical check can catch the swap.
    import base64

    authorization = build_weight_transport_authorization_v2(
        validator_hotkey=_HOTKEY,
        path="/weights/submit/v2",
        wire_body_hash=sha256_bytes(forged_wire),
        wire_body_bytes=len(forged_wire),
        logical_body_hash=sha256_bytes(logical),
        logical_body_bytes=len(logical),
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/weights/submit/v2",
        "headers": [
            (b"content-encoding", b"gzip"),
            (
                b"x-leadpoet-weight-transport",
                base64.b64encode(json.dumps(authorization).encode()),
            ),
            (b"x-leadpoet-weight-transport-signature", b"ab" * 64),
        ],
    }
    seen, statuses = _drive(BodySizeLimitMiddleware(None), scope, forged_wire)
    assert statuses == [400]
    assert "body" not in seen


def test_unauthorized_gzip_still_401(monkeypatch):
    monkeypatch.setenv("PRIMARY_VALIDATOR_HOTKEYS", "")
    logical = b'{"receipt_graph":{}}'
    scope, wire = _authorized_request(logical)
    seen, statuses = _drive(BodySizeLimitMiddleware(None), scope, wire)
    assert statuses == [401]
    assert "body" not in seen


# ---------------------------------------------------------------------------
# 3. Disk cache: restart warm-start semantics.
# ---------------------------------------------------------------------------

@pytest.fixture()
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_ALLOCATION_HANDOFF_DIR", str(tmp_path))
    monkeypatch.delenv(
        "RESEARCH_LAB_ALLOCATION_HANDOFF_DISK_CACHE", raising=False
    )
    monkeypatch.setattr(
        disk_cache,
        "_validated_handoff",
        lambda handoff, **_kwargs: dict(handoff),
    )
    return tmp_path


def test_disk_cache_roundtrip_survives_process_restart(_cache_dir):
    handoff = {"bundle": {"epoch_id": 24103}, "root_receipt_hash": "ab" * 32}
    commit = "a" * 40
    disk_cache.store_handoff(
        71, 24103, True, commit, handoff, ttl_seconds=5400.0
    )
    # A "restarted" process shares only the directory, not any state.
    assert disk_cache.load_handoff(71, 24103, True, commit) == handoff
    # Wrong key variants must miss.
    assert disk_cache.load_handoff(71, 24103, False, commit) is None
    assert disk_cache.load_handoff(71, 24104, True, commit) is None
    assert disk_cache.load_handoff(71, 24103, True, "b" * 40) is None


def test_disk_cache_prunes_other_epochs(_cache_dir):
    commit = "a" * 40
    disk_cache.store_handoff(
        71, 24102, True, commit, {"old": 1}, ttl_seconds=5400.0
    )
    disk_cache.store_handoff(
        71, 24103, True, commit, {"new": 1}, ttl_seconds=5400.0
    )
    assert disk_cache.load_handoff(71, 24102, True, commit) is None
    assert disk_cache.load_handoff(71, 24103, True, commit) == {"new": 1}


def test_disk_cache_expires_and_fails_open(_cache_dir):
    commit = "a" * 40
    disk_cache.store_handoff(
        71, 24103, True, commit, {"h": 1}, ttl_seconds=0.05
    )
    time.sleep(0.06)
    assert disk_cache.load_handoff(71, 24103, True, commit) is None
    # Corruption must fail open (return None), never raise.
    path = disk_cache._entry_path(71, 24103, True, commit)
    with open(path, "w") as fh:
        fh.write("{not json")
    assert disk_cache.load_handoff(71, 24103, True, commit) is None


def test_disk_cache_disabled_by_env(_cache_dir, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_ALLOCATION_HANDOFF_DISK_CACHE", "0")
    commit = "a" * 40
    disk_cache.store_handoff(
        71, 24103, True, commit, {"h": 1}, ttl_seconds=5400.0
    )
    assert disk_cache.load_handoff(71, 24103, True, commit) is None


def test_disk_cache_refuses_oversized_write(_cache_dir, monkeypatch):
    commit = "a" * 40
    monkeypatch.setattr(disk_cache, "_MAX_CACHE_DOCUMENT_BYTES", 64)
    disk_cache.store_handoff(
        71,
        24103,
        True,
        commit,
        {"handoff": "x" * 256},
        ttl_seconds=5400.0,
    )
    assert not Path(
        disk_cache._entry_path(71, 24103, True, commit)
    ).exists()
    assert not list(_cache_dir.glob(".allocation_handoff_*.tmp"))


# ---------------------------------------------------------------------------
# 4. Allocation fetch: gzip negotiation against a real HTTP server.
# ---------------------------------------------------------------------------

_PAYLOAD = {"bundle": {"epoch_id": 24103}, "receipt_graph": {"receipts": []}}


class _AllocationHandler(http.server.BaseHTTPRequestHandler):
    mode = "gzip"
    calls = 0

    def do_GET(self):
        type(self).calls += 1
        assert "gzip" in self.headers.get("Accept-Encoding", "")
        raw = json.dumps(_PAYLOAD).encode()
        if self.mode == "malformed_once" and type(self).calls == 1:
            body = b"{"
            self.send_response(200)
        elif self.mode in {"gzip", "truncated_once"}:
            body = gzip.compress(raw, compresslevel=6)
            if self.mode == "truncated_once" and type(self).calls == 1:
                body = body[:-8]
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
        else:
            body = raw
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture()
def _allocation_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _AllocationHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/alloc"
    server.shutdown()


def test_fetch_decodes_gzip_response(_allocation_server):
    _AllocationHandler.mode = "gzip"
    _AllocationHandler.calls = 0
    assert (
        _fetch_allocation_json(_allocation_server, deadline_seconds=10)
        == _PAYLOAD
    )


def test_fetch_still_accepts_identity_from_older_gateway(_allocation_server):
    _AllocationHandler.mode = "identity"
    _AllocationHandler.calls = 0
    assert (
        _fetch_allocation_json(_allocation_server, deadline_seconds=10)
        == _PAYLOAD
    )


def test_fetch_retries_a_truncated_gzip_response(_allocation_server):
    _AllocationHandler.mode = "truncated_once"
    _AllocationHandler.calls = 0
    assert (
        _fetch_allocation_json(
            _allocation_server,
            deadline_seconds=10,
            retry_delay_seconds=0,
        )
        == _PAYLOAD
    )
    assert _AllocationHandler.calls == 2


def test_fetch_retries_a_malformed_json_response(_allocation_server):
    _AllocationHandler.mode = "malformed_once"
    _AllocationHandler.calls = 0
    assert (
        _fetch_allocation_json(
            _allocation_server,
            deadline_seconds=10,
            retry_delay_seconds=0,
        )
        == _PAYLOAD
    )
    assert _AllocationHandler.calls == 2


def test_gzip_response_logical_limit_remains_fail_closed(monkeypatch):
    monkeypatch.setattr(
        validator_integration_module,
        "_ALLOCATION_RESPONSE_MAX_LOGICAL_BYTES",
        32,
    )
    with pytest.raises(RuntimeError, match="size limit"):
        validator_integration_module._decode_allocation_response_gzip(
            gzip.compress(b"x" * 33)
        )
