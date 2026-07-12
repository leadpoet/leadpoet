"""Signed fail-closed host-operation channel for measured enclave jobs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import secrets
import threading
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from leadpoet_canonical.attested_v2 import (
    build_host_operation_request_body,
    build_host_operation_terminal_body,
    canonical_json,
    create_signed_host_operation_request,
    create_signed_host_operation_terminal,
    sha256_bytes,
    validate_boot_identity,
)


MAX_HOST_OPERATION_BYTES = 64 * 1024 * 1024
MAX_HOST_OPERATIONS_PER_JOB = 4096
DELIVERY_LEASE_SECONDS = 5.0


class HostOperationV2Error(RuntimeError):
    """A host operation was unauthorized, malformed, timed out, or failed."""

    def __init__(self, message: str, *, terminal: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.terminal = dict(terminal or {})


def _utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class HostOperationChannelV2:
    """Block enclave execution until a signed host command has one terminal."""

    def __init__(
        self,
        *,
        job_id: str,
        purpose: str,
        boot_identity: Mapping[str, Any],
        sign_digest: Callable[[bytes], Any],
        allowed_operations: Iterable[str],
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        validate_boot_identity(boot_identity)
        if boot_identity.get("role") != "gateway_autoresearch":
            raise HostOperationV2Error("host operation channel requires autoresearch boot")
        self.job_id = str(job_id)
        self.purpose = str(purpose)
        self.boot_identity = dict(boot_identity)
        self._sign_digest = sign_digest
        self._allowed = frozenset(str(item) for item in allowed_operations)
        if not self._allowed:
            raise HostOperationV2Error("host operation allowlist is empty")
        self._clock = clock
        self._monotonic = monotonic
        self._records = []  # type: list[Dict[str, Any]]
        self._by_hash = {}  # type: Dict[str, Dict[str, Any]]
        self._closed = False
        self._condition = threading.Condition(threading.RLock())

    def execute(
        self,
        *,
        operation: str,
        payload: Mapping[str, Any],
        expected_state_hash: str,
        timeout_seconds: int,
        response_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> Dict[str, Any]:
        normalized_operation = str(operation or "")
        if normalized_operation not in self._allowed:
            raise HostOperationV2Error("host operation is not allowlisted")
        payload_bytes = canonical_json(dict(payload)).encode("utf-8")
        if len(payload_bytes) > MAX_HOST_OPERATION_BYTES:
            raise HostOperationV2Error("host operation payload exceeds limit")
        timeout = max(1, int(timeout_seconds))
        now = self._clock()
        with self._condition:
            if self._closed:
                raise HostOperationV2Error("host operation channel is closed")
            if len(self._records) >= MAX_HOST_OPERATIONS_PER_JOB:
                raise HostOperationV2Error("host operation ledger is full")
            sequence = len(self._records)
            request = create_signed_host_operation_request(
                body=build_host_operation_request_body(
                    job_id=self.job_id,
                    purpose=self.purpose,
                    operation=normalized_operation,
                    sequence=sequence,
                    payload_hash=sha256_bytes(payload_bytes),
                    expected_state_hash=expected_state_hash,
                    boot_identity_hash=self.boot_identity["boot_identity_hash"],
                    request_nonce=secrets.token_hex(16),
                    issued_at=_utc(now),
                    expires_at=_utc(now + timedelta(seconds=timeout)),
                ),
                enclave_pubkey=self.boot_identity["signing_pubkey"],
                sign_digest=self._sign_digest,
            )
            record = {
                "request": request,
                "payload": dict(payload),
                "payload_bytes": payload_bytes,
                "response": None,
                "response_validator": response_validator,
                "terminal": None,
                "delivered_at": None,
            }
            self._records.append(record)
            self._by_hash[request["request_hash"]] = record
            self._condition.notify_all()
            deadline = self._monotonic() + timeout
            while record["terminal"] is None:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    record["terminal"] = self._terminal(
                        request,
                        terminal_status="failed",
                        response_hash=None,
                        failure_code="host_operation_timeout",
                    )
                    self._condition.notify_all()
                    break
                self._condition.wait(timeout=remaining)
            terminal = dict(record["terminal"])
            response = record["response"]
        if terminal["terminal_status"] != "succeeded":
            raise HostOperationV2Error(
                "host operation failed: %s" % terminal.get("failure_code"),
                terminal=terminal,
            )
        if not isinstance(response, Mapping):
            raise HostOperationV2Error("successful host operation response is missing")
        return dict(response)

    def next_command(self, *, wait_ms: int = 0) -> Optional[Dict[str, Any]]:
        deadline = self._monotonic() + max(0, int(wait_ms)) / 1000.0
        with self._condition:
            while True:
                now = self._monotonic()
                for record in self._records:
                    if record["terminal"] is not None:
                        continue
                    delivered_at = record["delivered_at"]
                    if delivered_at is None or now - delivered_at >= DELIVERY_LEASE_SECONDS:
                        record["delivered_at"] = now
                        return {
                            "request": dict(record["request"]),
                            "payload": dict(record["payload"]),
                        }
                remaining = deadline - now
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def complete(
        self,
        *,
        request_hash: str,
        terminal_status: str,
        response: Optional[Mapping[str, Any]],
        failure_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._condition:
            record = self._by_hash.get(str(request_hash or ""))
            if record is None:
                raise HostOperationV2Error("host operation request was not found")
            if record["terminal"] is not None:
                raise HostOperationV2Error("host operation already has a terminal")
            normalized_response = dict(response) if isinstance(response, Mapping) else None
            response_hash = (
                sha256_bytes(canonical_json(normalized_response).encode("utf-8"))
                if normalized_response is not None
                else None
            )
            normalized_terminal_status = str(terminal_status)
            normalized_failure_code = failure_code
            if normalized_terminal_status == "succeeded":
                try:
                    validated_response = dict(
                        record["response_validator"](normalized_response or {})
                    )
                    if canonical_json(validated_response) != canonical_json(
                        normalized_response
                    ):
                        raise ValueError("host response is not canonical for operation")
                    normalized_response = validated_response
                    response_hash = sha256_bytes(
                        canonical_json(normalized_response).encode("utf-8")
                    )
                except Exception:
                    normalized_terminal_status = "failed"
                    normalized_failure_code = "response_validation_failed"
            terminal = self._terminal(
                record["request"],
                terminal_status=normalized_terminal_status,
                response_hash=response_hash,
                failure_code=normalized_failure_code,
            )
            record["response"] = normalized_response
            record["terminal"] = terminal
            self._condition.notify_all()
            return dict(terminal)

    def ledger(self) -> tuple[Dict[str, Any], ...]:
        with self._condition:
            return tuple(
                {
                    "request": dict(record["request"]),
                    "terminal": (
                        dict(record["terminal"])
                        if record["terminal"] is not None
                        else None
                    ),
                }
                for record in self._records
            )

    def terminal_hashes(self) -> tuple[str, ...]:
        with self._condition:
            if any(record["terminal"] is None for record in self._records):
                raise HostOperationV2Error("host operation ledger has missing terminals")
            return tuple(record["terminal"]["terminal_hash"] for record in self._records)

    def complete_ledger(self) -> tuple[Dict[str, Any], ...]:
        with self._condition:
            if any(record["terminal"] is None for record in self._records):
                raise HostOperationV2Error("host operation ledger has missing terminals")
            return tuple(
                {
                    "request": dict(record["request"]),
                    "terminal": dict(record["terminal"]),
                }
                for record in self._records
            )

    def close(self, *, failure_code: str = "cancelled") -> None:
        """Close the channel and terminally fail every outstanding request."""

        with self._condition:
            self._closed = True
            for record in self._records:
                if record["terminal"] is None:
                    record["terminal"] = self._terminal(
                        record["request"],
                        terminal_status="failed",
                        response_hash=None,
                        failure_code=failure_code,
                    )
            self._condition.notify_all()

    def _terminal(
        self,
        request: Mapping[str, Any],
        *,
        terminal_status: str,
        response_hash: Optional[str],
        failure_code: Optional[str],
    ) -> Dict[str, Any]:
        return create_signed_host_operation_terminal(
            body=build_host_operation_terminal_body(
                request_hash=request["request_hash"],
                job_id=request["job_id"],
                purpose=request["purpose"],
                operation=request["operation"],
                sequence=request["sequence"],
                terminal_status=terminal_status,
                response_hash=response_hash,
                failure_code=failure_code,
                completed_at=_utc(self._clock()),
            ),
            enclave_pubkey=self.boot_identity["signing_pubkey"],
            sign_digest=self._sign_digest,
        )
