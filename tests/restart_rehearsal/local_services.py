#!/usr/bin/env python3.11
"""Strict local implementations of privileged rehearsal boundaries.

This module is intentionally incapable of contacting production.  It exposes
loopback HTTP plus SQLite state for the two stateful services needed by the V2
workflow: PostgREST-shaped durable storage and the stateful subnet chain.
Unknown operations, malformed requests, duplicate extrinsics, and unconsumed
fault injections fail the rehearsal.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


class LocalServiceError(RuntimeError):
    pass


class _State:
    def __init__(self, *, database_path: Path, fixture: Mapping[str, Any]) -> None:
        self.lock = threading.RLock()
        self.fixture = dict(fixture)
        self.database_path = database_path
        self.faults: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.chain: dict[int, dict[str, Any]] = {}
        self.connection = sqlite3.connect(
            str(database_path), check_same_thread=False
        )
        self.connection.execute(
            """
            CREATE TABLE durable_evidence (
                evidence_hash TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                epoch_id INTEGER NOT NULL,
                body_json TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def inject(self, fault: str) -> None:
        with self.lock:
            self.faults.append(str(fault))

    def consume_fault(self) -> str | None:
        with self.lock:
            if not self.faults:
                return None
            return self.faults.pop(0)

    def record(self, operation: str, body: Mapping[str, Any]) -> None:
        with self.lock:
            self.events.append(
                {
                    "ordinal": len(self.events) + 1,
                    "operation": operation,
                    "request_hash": _sha256(body),
                }
            )


class _Handler(BaseHTTPRequestHandler):
    server_version = "LeadpoetLocalBoundary/1"

    @property
    def state(self) -> _State:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _reply(self, status: int, value: Mapping[str, Any]) -> None:
        body = _canonical(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        try:
            length = int(self.headers.get("Content-Length", ""))
            if length <= 0 or length > 16 * 1024 * 1024:
                raise LocalServiceError("request body is outside the allowed range")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise LocalServiceError("request body must be an object")
            operation = self.path.removeprefix("/")
            fault = self.state.consume_fault()
            if fault:
                self.state.record(f"fault:{fault}", value)
                status = {
                    "http_400": 400,
                    "http_403": 403,
                    "http_429": 429,
                    "http_500": 500,
                    "duplicate_response": 409,
                    "malformed_json": 502,
                    "partial_body": 502,
                    "unexpected_eof": 502,
                    "timeout": 504,
                }.get(fault, 503)
                self._reply(status, {"status": "injected_failure", "fault": fault})
                return
            handlers = {
                "chain/submit_extrinsic": self._submit_extrinsic,
                "chain/finalize": self._finalize,
                "chain/reveal": self._reveal,
                "database/insert": self._database_insert,
            }
            handler = handlers.get(operation)
            if handler is None:
                raise LocalServiceError(f"unknown local boundary operation: {operation}")
            self.state.record(operation, value)
            self._reply(200, handler(value))
        except (ValueError, json.JSONDecodeError, LocalServiceError) as exc:
            self._reply(400, {"status": "rejected", "reason": str(exc)})

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        try:
            parts = self.path.strip("/").split("/")
            if parts[:2] == ["chain", "epoch"] and len(parts) == 4:
                epoch = int(parts[2])
                field = parts[3]
                if field not in {"finalization", "last_update", "reveal"}:
                    raise LocalServiceError("unknown chain read")
                row = self.state.chain.get(epoch)
                if row is None:
                    raise LocalServiceError("epoch is absent")
                value = row.get(field)
                if value is None:
                    raise LocalServiceError(f"{field} is absent")
                self.state.record(f"chain/{field}", {"epoch_id": epoch})
                self._reply(200, {"status": "ok", field: value})
                return
            raise LocalServiceError("unknown local boundary read")
        except (ValueError, LocalServiceError) as exc:
            self._reply(404, {"status": "rejected", "reason": str(exc)})

    @staticmethod
    def _required(value: Mapping[str, Any], fields: set[str]) -> None:
        if set(value) != fields:
            raise LocalServiceError(
                "boundary fields differ: expected %s got %s"
                % (sorted(fields), sorted(value))
            )

    def _submit_extrinsic(self, value: Mapping[str, Any]) -> dict[str, Any]:
        self._required(
            value,
            {
                "epoch_id",
                "extrinsic_hash",
                "extrinsic_hex",
                "bundle_hash",
                "weights_hash",
                "uids",
                "weights_u16",
            },
        )
        epoch = int(value["epoch_id"])
        extrinsic_hex = str(value["extrinsic_hex"])
        if not extrinsic_hex or any(
            character not in "0123456789abcdef" for character in extrinsic_hex
        ):
            raise LocalServiceError("signed extrinsic is not lowercase hex")
        expected_hash = "0x" + hashlib.blake2b(
            bytes.fromhex(extrinsic_hex), digest_size=32
        ).hexdigest()
        if value["extrinsic_hash"] != expected_hash:
            raise LocalServiceError("signed extrinsic hash differs")
        if epoch in self.state.chain:
            raise LocalServiceError("epoch extrinsic is duplicated")
        self.state.chain[epoch] = {
            "submission": dict(value),
            "finalization": None,
            "last_update": None,
            "reveal": None,
        }
        return {"status": "accepted", "extrinsic_hash": expected_hash}

    def _finalize(self, value: Mapping[str, Any]) -> dict[str, Any]:
        self._required(
            value, {"epoch_id", "extrinsic_hash", "finalized_block"}
        )
        epoch = int(value["epoch_id"])
        row = self.state.chain.get(epoch)
        if row is None:
            raise LocalServiceError("cannot finalize an absent extrinsic")
        if row["submission"]["extrinsic_hash"] != value["extrinsic_hash"]:
            raise LocalServiceError("finalization extrinsic differs")
        finalization = {
            "extrinsic_hash": value["extrinsic_hash"],
            "finalized_block": int(value["finalized_block"]),
            "finalized_block_hash": hashlib.sha256(
                f"finalized:{epoch}".encode("ascii")
            ).hexdigest(),
            "state_transition_hash": _sha256(
                {"epoch_id": epoch, "state": "committed"}
            ),
        }
        row["finalization"] = finalization
        row["last_update"] = int(value["finalized_block"])
        return {"status": "finalized", **finalization}

    def _reveal(self, value: Mapping[str, Any]) -> dict[str, Any]:
        self._required(value, {"epoch_id", "uids", "weights_u16"})
        epoch = int(value["epoch_id"])
        row = self.state.chain.get(epoch)
        if row is None or row["finalization"] is None:
            raise LocalServiceError("cannot reveal before finalization")
        submission = row["submission"]
        if (
            list(value["uids"]) != list(submission["uids"])
            or list(value["weights_u16"]) != list(submission["weights_u16"])
        ):
            raise LocalServiceError("revealed vector differs from submission")
        row["reveal"] = {
            "uids": list(value["uids"]),
            "weights_u16": list(value["weights_u16"]),
            "vector_hash": _sha256(
                {
                    "uids": list(value["uids"]),
                    "weights_u16": list(value["weights_u16"]),
                }
            ),
        }
        return {"status": "revealed", **row["reveal"]}

    def _database_insert(self, value: Mapping[str, Any]) -> dict[str, Any]:
        self._required(value, {"kind", "epoch_id", "body"})
        body = value["body"]
        if not isinstance(body, dict):
            raise LocalServiceError("durable evidence body must be an object")
        evidence_hash = _sha256(body)
        try:
            with self.state.lock:
                self.state.connection.execute(
                    """
                    INSERT INTO durable_evidence
                        (evidence_hash, kind, epoch_id, body_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        evidence_hash,
                        str(value["kind"]),
                        int(value["epoch_id"]),
                        _canonical(body).decode("ascii"),
                    ),
                )
                self.state.connection.commit()
        except sqlite3.IntegrityError as exc:
            raise LocalServiceError("durable evidence is duplicated") from exc
        return {"status": "persisted", "evidence_hash": evidence_hash}


class LocalBoundaryServices(AbstractContextManager["LocalBoundaryServices"]):
    def __init__(self, *, root: Path, fixture: Mapping[str, Any]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.state = _State(
            database_path=root / "production-shaped.sqlite3",
            fixture=fixture,
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.state = self.state  # type: ignore[attr-defined]
        self.opener = build_opener(ProxyHandler({}))
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name="leadpoet-local-boundary",
            daemon=True,
        )

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "LocalBoundaryServices":
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.state.close()
        if self.thread.is_alive():
            raise LocalServiceError("local boundary service did not stop")
        if self.state.faults:
            raise LocalServiceError(
                f"unconsumed fault injections: {self.state.faults!r}"
            )

    def inject(self, fault: str) -> None:
        self.state.inject(fault)

    def request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        *,
        expected_status: int = 200,
    ) -> dict[str, Any]:
        payload = None if body is None else _canonical(body)
        request = Request(
            self.url + path,
            data=payload,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=5) as response:
                status = response.status
                value = json.loads(response.read())
        except HTTPError as exc:
            status = exc.code
            value = json.loads(exc.read())
        if status != expected_status:
            raise LocalServiceError(
                f"local boundary status differs: {status} != {expected_status}: {value}"
            )
        if not isinstance(value, dict):
            raise LocalServiceError("local boundary response is not an object")
        return value


SDK_HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
SDK_PUBLIC_KEY = (
    "a6bfe69c29bf9e4db65c63ac6f6d1e23"
    "c252ca871744afb6edc5623d9bc39004"
)
SDK_FINALIZED_HASH = "0x" + "aa" * 32


class LocalEnclaveSigningBoundary:
    """Strict Nitro signing boundary consumed by the production SDK bridge."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def get_hotkey_state_v2(self) -> dict[str, Any]:
        return {
            "validator_hotkey": SDK_HOTKEY,
            "hotkey_public_key": SDK_PUBLIC_KEY,
            "provisioned": True,
        }

    def prepare_weight_commit_v2(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        required = {
            "weight_authorization_id",
            "weight_submission_event_hash",
            "uids",
            "weights_u16",
            "version_key",
            "last_epoch_block",
            "pending_epoch_at",
            "subnet_epoch_index",
            "tempo",
            "blocks_since_last_step",
            "current_block",
            "subnet_reveal_period_epochs",
            "block_time",
            "hotkey_public_key_hex",
        }
        if set(request) != required:
            raise LocalServiceError("SDK commit request fields differ")
        if request["hotkey_public_key_hex"] != SDK_PUBLIC_KEY:
            raise LocalServiceError("SDK commit hotkey differs")
        self.requests.append(("commit", dict(request)))
        return {
            "commit_authorization_id": _sha256(
                {"sdk_commit": dict(request)}
            ),
            "commitment_hex": hashlib.sha512(
                _canonical(dict(request))
            ).hexdigest(),
            "reveal_round": int(request["subnet_epoch_index"]) + 1,
        }

    def sign_weight_extrinsic_v2(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        required = {
            "commit_authorization_id",
            "runtime_block_hash",
            "era_current",
            "nonce",
            "block_hash",
            "signature_payload_hex",
        }
        if set(request) != required:
            raise LocalServiceError("SDK extrinsic signing fields differ")
        self.requests.append(("extrinsic", dict(request)))
        signature = hashlib.sha512(_canonical(dict(request))).hexdigest()
        return {
            "signature": signature,
            "receipt": {"receipt_hash": _sha256({"sdk_signature": signature})},
            "authorization": {
                "authorization_hash": _sha256(
                    {"sdk_authorization": dict(request)}
                )
            },
            "extrinsic_hash": "0x" + hashlib.blake2b(
                b"local-sdk-extrinsic", digest_size=32
            ).hexdigest(),
        }


class _LocalEra:
    def encode(self, era: Mapping[str, Any]) -> None:
        self.era = dict(era)

    def birth(self, current: int) -> int:
        return int(current) - (int(current) % int(self.era["period"]))


class _LocalRuntimeConfig:
    def create_scale_object(self, name: str) -> _LocalEra:
        if name != "Era":
            raise LocalServiceError("SDK requested an unknown SCALE object")
        return _LocalEra()


class LocalSDKSubstrateBoundary:
    """Exact substrate methods exercised by Bittensor's production SDK path."""

    runtime_config = _LocalRuntimeConfig()

    def __init__(self) -> None:
        self.original_calls: list[dict[str, Any]] = []
        self.create_signed_extrinsic = self.original_create_signed_extrinsic

    def init_runtime(self, block_hash: str | None = None) -> None:
        if block_hash != SDK_FINALIZED_HASH:
            raise LocalServiceError("SDK runtime initialization hash differs")

    def get_account_nonce(self, address: str) -> int:
        if address != SDK_HOTKEY:
            raise LocalServiceError("SDK account nonce hotkey differs")
        return 7

    def get_chain_finalised_head(self) -> str:
        return SDK_FINALIZED_HASH

    def get_block_number(self, head: str) -> int:
        if head != SDK_FINALIZED_HASH:
            raise LocalServiceError("SDK finalized head differs")
        return 123

    def get_block_hash(self, block_id: int) -> str:
        if int(block_id) != 120:
            raise LocalServiceError("SDK era birth block differs")
        return "0x" + "bb" * 32

    def generate_signature_payload(self, **kwargs: Any) -> Any:
        if kwargs.get("era") != {"period": 8, "current": 123}:
            raise LocalServiceError("SDK signing era differs")
        if kwargs.get("nonce") != 7:
            raise LocalServiceError("SDK signing nonce differs")
        return SimpleNamespace(data=b"canonical-scale-payload")

    def original_create_signed_extrinsic(self, **kwargs: Any) -> Any:
        self.original_calls.append(dict(kwargs))
        return SimpleNamespace(
            data=SimpleNamespace(data=b"local-sdk-extrinsic"),
            signature=kwargs["signature"],
        )


def local_enclave_backed_wallet(client: LocalEnclaveSigningBoundary) -> Any:
    from validator_tee.host.enclave_hotkey_v2 import (
        build_enclave_backed_wallet_v2,
    )

    return build_enclave_backed_wallet_v2(
        name="validator_72",
        hotkey_name="default",
        path="/sanitized-public-wallet",
        client=client,
    )
