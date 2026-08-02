"""Authenticated client for measured gateway-owned V2 weight inputs."""

from __future__ import annotations

import asyncio
import json
import logging
import zlib
import os
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional
from urllib.parse import urlparse

from leadpoet_canonical.attested_v2 import (
    sha256_json,
    validate_boot_identity,
    validate_host_operation_record,
    validate_signed_execution_receipt,
    validate_transport_attempt,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    build_weight_inputs_request_v2,
    weight_inputs_request_message_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    WEIGHT_INPUT_PURPOSES,
)
from validator_tee.host.vsock_client import ValidatorEnclaveClient
from leadpoet_observability import (
    capture_failure,
    failure_code_for_exception,
    hash_identifier,
    record_retry,
    record_stage,
    weight_correlation_id,
)


logger = logging.getLogger(__name__)

# 500 included: the inputs fetch is a read/compute request the gateway
# serves idempotently (attested executions replay), and a transient
# app-level 500 (e.g. a DB blip surfacing through the handler) used to
# abort the whole attempt sequence inside the submission window.
_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_MAX_WEIGHT_INPUT_RESPONSE_FRAME_BYTES = 16 * 1024 * 1024
_MAX_WEIGHT_INPUT_RESPONSE_BYTES = 128 * 1024 * 1024
# Retry delays are capped so a timed-out attempt re-attaches to the gateway's
# in-flight singleflight build (which caches its exact success) within seconds
# of completion instead of drifting into minute-long exponential gaps.
_MAX_RETRY_DELAY_SECONDS = 8.0


class GatewayWeightInputsV2Error(RuntimeError):
    """The gateway input request or returned ancestry is not authoritative."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _decode_response_body(
    wire_body: bytes,
    *,
    content_encoding: str,
    logical_limit: int = _MAX_WEIGHT_INPUT_RESPONSE_BYTES,
) -> bytes:
    encoding = str(content_encoding or "").lower()
    if encoding == "gzip":
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            body = decompressor.decompress(wire_body, logical_limit + 1)
        except zlib.error as exc:
            raise GatewayWeightInputsV2Error(
                "gateway V2 weight input response gzip is invalid"
            ) from exc
        if (
            len(body) > logical_limit
            or not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            raise GatewayWeightInputsV2Error(
                "gateway V2 weight input response gzip failed validation"
            )
        return body
    if encoding in {"", "identity"}:
        if len(wire_body) > logical_limit:
            raise GatewayWeightInputsV2Error(
                "gateway V2 weight input response exceeds size limit"
            )
        return wire_body
    raise GatewayWeightInputsV2Error(
        "gateway V2 weight input response encoding is unsupported"
    )


def _is_retryable_gateway_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return (
        isinstance(exc, GatewayWeightInputsV2Error)
        and exc.status_code in _RETRYABLE_HTTP_STATUSES
    )


def _gateway_endpoint(value: str) -> str:
    normalized = str(value or "").rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme == "https" and parsed.netloc:
        return normalized
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        return normalized
    raise GatewayWeightInputsV2Error(
        "authoritative V2 gateway communication requires HTTPS"
    )


async def _post_json(
    url: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
    async with aiohttp.ClientSession(
        timeout=timeout,
        trust_env=False,
        auto_decompress=False,
        headers={"Accept-Encoding": "gzip"},
    ) as session:
        async with session.post(url, json=dict(payload)) as response:
            encoding = str(response.headers.get("Content-Encoding") or "").lower()
            wire_limit = (
                _MAX_WEIGHT_INPUT_RESPONSE_FRAME_BYTES
                if encoding == "gzip"
                else _MAX_WEIGHT_INPUT_RESPONSE_BYTES
            )
            wire_body = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                wire_body.extend(chunk)
                if len(wire_body) > wire_limit:
                    raise GatewayWeightInputsV2Error(
                        "gateway V2 weight input response frame exceeds size limit "
                        "(encoding=%s limit=%d)" % (encoding or "identity", wire_limit)
                    )
            if response.status != 200:
                raise GatewayWeightInputsV2Error(
                    "gateway V2 weight input request failed with HTTP %d: %s"
                    % (
                        response.status,
                        bytes(wire_body[:300]).decode("utf-8", errors="replace"),
                    ),
                    status_code=int(response.status),
                )
            body = _decode_response_body(
                bytes(wire_body),
                content_encoding=encoding,
            )
            try:
                value = json.loads(body.decode("utf-8"))
            except Exception as exc:
                raise GatewayWeightInputsV2Error(
                    "gateway V2 weight input response is not JSON"
                ) from exc
    if not isinstance(value, Mapping):
        raise GatewayWeightInputsV2Error(
            "gateway V2 weight input response is not an object"
        )
    return value


def _validate_receipt_set(
    *,
    value: Mapping[str, Any],
    input_receipt_hashes: Mapping[str, Any],
    epoch_id: int,
) -> Dict[str, list[dict[str, Any]]]:
    fields = {
        "boot_identities",
        "receipts",
        "transport_attempts",
        "host_operations",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise GatewayWeightInputsV2Error("gateway V2 receipt set fields are invalid")
    normalized = {field: [] for field in fields}
    for identity in value["boot_identities"]:
        validate_boot_identity(identity)
        normalized["boot_identities"].append(dict(identity))
    receipt_by_hash = {}
    for receipt in value["receipts"]:
        validate_signed_execution_receipt(receipt)
        receipt_hash = str(receipt["receipt_hash"])
        if receipt_hash in receipt_by_hash:
            raise GatewayWeightInputsV2Error("gateway V2 receipt is duplicated")
        receipt_by_hash[receipt_hash] = dict(receipt)
        normalized["receipts"].append(dict(receipt))
    for attempt in value["transport_attempts"]:
        validate_transport_attempt(attempt)
        normalized["transport_attempts"].append(dict(attempt))
    for record in value["host_operations"]:
        validate_host_operation_record(record)
        normalized["host_operations"].append(dict(record))
    known_receipts = set(receipt_by_hash)
    for receipt in receipt_by_hash.values():
        if not set(receipt["parent_receipt_hashes"]).issubset(known_receipts):
            raise GatewayWeightInputsV2Error(
                "gateway V2 receipt ancestry is incomplete"
            )
    for category, receipt_hash in input_receipt_hashes.items():
        receipt = receipt_by_hash.get(str(receipt_hash))
        expected_role, expected_purpose = WEIGHT_INPUT_PURPOSES[category]
        if (
            receipt is None
            or receipt.get("role") != expected_role
            or receipt.get("purpose") != expected_purpose
            or int(receipt.get("epoch_id", -1)) != int(epoch_id)
        ):
            raise GatewayWeightInputsV2Error(
                "gateway V2 input receipt differs for %s" % category
            )
    return normalized


def _validate_compact_ancestry(
    *,
    proofs_value: Mapping[str, Any],
    attempts_value: Any,
    input_receipt_hashes: Mapping[str, Any],
    gateway_authority_event_hash: str,
    epoch_id: int,
) -> Dict[str, Any]:
    """Perform host-side shape checks before enclave verification.

    The host is not an ancestry trust boundary: Nitro/release verification and
    detached-certificate signature checks happen inside the validator enclave.
    These checks bound the request, reject ambiguous duplicates, and guarantee
    every semantic input receipt is disclosed before crossing vsock.
    """

    if not isinstance(proofs_value, Mapping) or set(proofs_value) != set(
        GATEWAY_WEIGHT_INPUT_CATEGORIES
    ):
        raise GatewayWeightInputsV2Error(
            "gateway V2 compact ancestry categories are incomplete"
        )
    if not isinstance(attempts_value, list):
        raise GatewayWeightInputsV2Error(
            "gateway V2 compact transport attempts are invalid"
        )
    proofs: dict[str, dict[str, Any]] = {}
    direct_scopes = set()
    seen_receipts: dict[str, dict[str, Any]] = {}
    seen_proofs = set()
    allocation_authority_observed = False
    for category in sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES):
        proof = proofs_value.get(category)
        if not isinstance(proof, Mapping) or set(proof) != {
            "schema_version",
            "certificate",
            "disclosed_receipts",
            "disclosed_boot_identities",
            "proof_hash",
        }:
            raise GatewayWeightInputsV2Error(
                "gateway V2 compact ancestry proof is invalid for %s" % category
            )
        proof_hash = str(proof.get("proof_hash") or "")
        if proof_hash in seen_proofs:
            raise GatewayWeightInputsV2Error(
                "gateway V2 compact ancestry proof is duplicated"
            )
        seen_proofs.add(proof_hash)
        disclosed_boots = proof.get("disclosed_boot_identities")
        disclosed_receipts = proof.get("disclosed_receipts")
        if not isinstance(disclosed_boots, list) or not isinstance(
            disclosed_receipts, list
        ):
            raise GatewayWeightInputsV2Error(
                "gateway V2 compact ancestry disclosure is invalid"
            )
        for identity in disclosed_boots:
            validate_boot_identity(identity)
        direct_hash = str(input_receipt_hashes[category])
        direct_receipt = None
        for receipt in disclosed_receipts:
            validate_signed_execution_receipt(receipt)
            receipt_hash = str(receipt["receipt_hash"])
            normalized = dict(receipt)
            if receipt_hash in seen_receipts and seen_receipts[receipt_hash] != normalized:
                raise GatewayWeightInputsV2Error(
                    "gateway V2 compact receipt hash conflicts"
                )
            seen_receipts[receipt_hash] = normalized
            if receipt_hash == direct_hash:
                direct_receipt = normalized
        expected_role, expected_purpose = WEIGHT_INPUT_PURPOSES[category]
        if (
            direct_receipt is None
            or direct_receipt.get("role") != expected_role
            or direct_receipt.get("purpose") != expected_purpose
            or int(direct_receipt.get("epoch_id", -1)) != int(epoch_id)
        ):
            raise GatewayWeightInputsV2Error(
                "gateway V2 compact input receipt differs for %s" % category
            )
        direct_scopes.add(
            (str(direct_receipt["job_id"]), str(direct_receipt["purpose"]))
        )
        certificate = proof.get("certificate")
        claim = certificate.get("claim") if isinstance(certificate, Mapping) else None
        parent_authorities = (
            claim.get("parent_authorities") if isinstance(claim, Mapping) else None
        )
        if not isinstance(parent_authorities, list):
            raise GatewayWeightInputsV2Error(
                "gateway V2 compact parent authorities are invalid"
            )
        if category == "research_lab_allocation":
            allocation_authority_observed = any(
                isinstance(item, Mapping)
                and item.get("parent_receipt_hash")
                == gateway_authority_event_hash
                for item in parent_authorities
            )
        proofs[category] = dict(proof)
    if not allocation_authority_observed:
        raise GatewayWeightInputsV2Error(
            "gateway allocation authority is absent from compact ancestry"
        )

    attempts: dict[str, dict[str, Any]] = {}
    for attempt in attempts_value:
        validate_transport_attempt(attempt)
        scope = (str(attempt["job_id"]), str(attempt["purpose"]))
        if scope not in direct_scopes:
            raise GatewayWeightInputsV2Error(
                "gateway V2 compact transport attempt is out of scope"
            )
        attempt_hash = str(attempt["attempt_hash"])
        normalized = dict(attempt)
        if attempt_hash in attempts and attempts[attempt_hash] != normalized:
            raise GatewayWeightInputsV2Error(
                "gateway V2 compact transport attempt hash conflicts"
            )
        attempts[attempt_hash] = normalized
    return {
        "upstream_ancestry_proofs": proofs,
        "upstream_transport_attempts": [
            attempts[attempt_hash] for attempt_hash in sorted(attempts)
        ],
    }


async def fetch_gateway_weight_inputs_v2(
    *,
    gateway_url: str,
    calculation_snapshot: Mapping[str, Any],
    validator_hotkey: str,
    allocation_hash: str,
    leaderboard_window_start: str,
    leaderboard_window_end: str,
    client: Optional[ValidatorEnclaveClient] = None,
    post_json: Callable[
        [str, Mapping[str, Any], float], Awaitable[Mapping[str, Any]]
    ] = _post_json,
    timeout_seconds: float = 90.0,
    # The gateway coalesces concurrent input requests into one singleflight
    # build and caches the exact success, so a timed-out attempt re-attaches
    # cheaply rather than triggering a rebuild. The attempt budget must cover
    # the slowest observed production build (~700s, 2026-07-31 epochs
    # 24265-24266): exhausting it makes the caller restart the whole flow,
    # which does force a rebuild and compounds the delay.
    max_attempts: int = 10,
    retry_delay_seconds: float = 2.0,
) -> Dict[str, Any]:
    """Request one exact, enclave-signed gateway input set and verify its shape."""

    calculation = dict(calculation_snapshot)
    runtime_sha = str(
        os.environ.get("GITHUB_SHA")
        or os.environ.get("GIT_COMMIT")
        or ""
    ).lower()
    telemetry = {
        "runtime_sha": runtime_sha,
        "netuid": calculation.get("netuid"),
        "epoch_id": calculation.get("epoch_id"),
        "validator_role": "primary",
        "validator_id_hash": hash_identifier(validator_hotkey),
        "restart_invocation_id": os.environ.get(
            "LEADPOET_RESTART_INVOCATION_ID"
        ),
        "weight_correlation_id": weight_correlation_id(
            runtime_sha=runtime_sha,
            netuid=calculation.get("netuid"),
            epoch_id=calculation.get("epoch_id"),
        ),
    }
    try:
        request = build_weight_inputs_request_v2(
            validator_hotkey=validator_hotkey,
            netuid=calculation["netuid"],
            epoch_id=calculation["epoch_id"],
            block=calculation["block"],
            calculation_snapshot_hash=sha256_json(calculation),
            allocation_hash=allocation_hash,
            leaderboard_window_start=leaderboard_window_start,
            leaderboard_window_end=leaderboard_window_end,
        )
    except Exception as exc:
        raise GatewayWeightInputsV2Error(
            "authoritative V2 weight input request is invalid"
        ) from exc
    endpoint = (
        _gateway_endpoint(gateway_url)
        + "/weights/inputs/v2?ancestry=compact-v2"
    )
    enclave_client = client or ValidatorEnclaveClient()
    message = weight_inputs_request_message_v2(request).encode("utf-8")
    signature_result = await asyncio.to_thread(
        enclave_client.sign_application_message_v2,
        message,
    )
    if (
        not isinstance(signature_result, Mapping)
        or signature_result.get("purpose")
        != "validator.gateway_weight_inputs.v2"
        or signature_result.get("validator_hotkey") != validator_hotkey
        or not isinstance(signature_result.get("receipt"), Mapping)
    ):
        raise GatewayWeightInputsV2Error(
            "validator enclave did not authorize the gateway input request"
        )
    if max_attempts <= 0:
        raise GatewayWeightInputsV2Error(
            "gateway V2 weight input attempts must be positive"
        )
    payload = {
        "request": request,
        "calculation_snapshot": calculation,
        "validator_hotkey_signature": str(signature_result["signature"]),
    }
    response = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await post_json(
                endpoint,
                payload,
                float(timeout_seconds),
            )
            break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            retryable = _is_retryable_gateway_error(exc)
            if attempt >= max_attempts or not retryable:
                code = failure_code_for_exception(
                    exc,
                    default=(
                        "weight.gateway_endpoint_unavailable"
                        if retryable
                        else "weight.bundle_divergence"
                    ),
                )
                capture_failure(
                    code,
                    component="validator",
                    stage="gateway_weight_input_fetch",
                    exception=exc,
                    terminal=True,
                    retryable=retryable,
                    fail_closed=True,
                    attempt=attempt,
                    attempts=max_attempts,
                    http_status=getattr(exc, "status_code", None),
                    **telemetry,
                )
                raise
            record_retry(
                "weight.gateway_endpoint_unavailable",
                component="validator",
                stage="gateway_weight_input_fetch",
                attempt=attempt,
                attempts=max_attempts,
                http_status=getattr(exc, "status_code", None),
                **telemetry,
            )
            delay = min(
                float(retry_delay_seconds) * (2 ** (attempt - 1)),
                _MAX_RETRY_DELAY_SECONDS,
            )
            logger.warning(
                "gateway_weight_inputs_v2_retry attempt=%d/%d "
                "delay_seconds=%.1f type=%s error=%s",
                attempt,
                max_attempts,
                delay,
                type(exc).__name__,
                str(exc)[:200],
            )
            await asyncio.sleep(delay)
    if response is None:
        raise GatewayWeightInputsV2Error(
            "gateway V2 weight input request exhausted without a response"
        )
    full_fields = {
        "request_hash",
        "calculation_snapshot_hash",
        "input_receipt_hashes",
        "gateway_authority_event_hash",
        "upstream_receipt_set",
    }
    compact_fields = {
        "request_hash",
        "calculation_snapshot_hash",
        "input_receipt_hashes",
        "gateway_authority_event_hash",
        "upstream_ancestry_proofs",
        "upstream_transport_attempts",
    }
    response_fields = set(response) if isinstance(response, Mapping) else set()
    if (
        not isinstance(response, Mapping)
        or response_fields not in (full_fields, compact_fields)
    ):
        raise GatewayWeightInputsV2Error("gateway V2 weight input response fields are invalid")
    if (
        response.get("request_hash") != request["request_hash"]
        or response.get("calculation_snapshot_hash")
        != request["calculation_snapshot_hash"]
    ):
        raise GatewayWeightInputsV2Error(
            "gateway V2 response does not bind the authorized request"
        )
    input_hashes = response.get("input_receipt_hashes")
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != set(
        GATEWAY_WEIGHT_INPUT_CATEGORIES
    ):
        raise GatewayWeightInputsV2Error(
            "gateway V2 input receipt categories are incomplete"
        )
    authority_hash = str(response.get("gateway_authority_event_hash") or "")
    result = {
        "input_receipt_hashes": {
            category: str(input_hashes[category])
            for category in sorted(input_hashes)
        },
        "gateway_authority_event_hash": authority_hash,
        "request_authorization": dict(signature_result),
    }
    if response_fields == full_fields:
        receipt_set = _validate_receipt_set(
            value=response["upstream_receipt_set"],
            input_receipt_hashes=input_hashes,
            epoch_id=int(calculation["epoch_id"]),
        )
        receipt_hashes = {
            str(receipt["receipt_hash"]) for receipt in receipt_set["receipts"]
        }
        if authority_hash not in receipt_hashes:
            raise GatewayWeightInputsV2Error(
                "gateway allocation authority receipt is absent from the response"
            )
        result["upstream_receipt_set"] = receipt_set
        receipts = receipt_set.get("receipts")
        record_stage(
            component="validator",
            stage="gateway_weight_input_verified",
            status="passed",
            request_hash=request["request_hash"],
            root_receipt_hash=authority_hash,
            receipt_count=len(receipts) if isinstance(receipts, list) else None,
            **telemetry,
        )
        return result
    compact = _validate_compact_ancestry(
        proofs_value=response["upstream_ancestry_proofs"],
        attempts_value=response["upstream_transport_attempts"],
        input_receipt_hashes=input_hashes,
        gateway_authority_event_hash=authority_hash,
        epoch_id=int(calculation["epoch_id"]),
    )
    result.update(compact)
    input_receipt_hashes = result.get("input_receipt_hashes")
    record_stage(
        component="validator",
        stage="gateway_weight_input_verified",
        status="passed",
        request_hash=request["request_hash"],
        root_receipt_hash=authority_hash,
        receipt_count=(
            len(input_receipt_hashes)
            if isinstance(input_receipt_hashes, Mapping)
            else None
        ),
        **telemetry,
    )
    return result
