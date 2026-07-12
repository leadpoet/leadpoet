"""Authenticated client for measured gateway-owned V2 weight inputs."""

from __future__ import annotations

import asyncio
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


class GatewayWeightInputsV2Error(RuntimeError):
    """The gateway input request or returned ancestry is not authoritative."""


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
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        async with session.post(url, json=dict(payload)) as response:
            body = await response.text()
            if response.status != 200:
                raise GatewayWeightInputsV2Error(
                    "gateway V2 weight input request failed with HTTP %d: %s"
                    % (response.status, body[:300])
                )
            try:
                value = await response.json()
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
    timeout_seconds: float = 1800.0,
) -> Dict[str, Any]:
    """Request one exact, enclave-signed gateway input set and verify its shape."""

    calculation = dict(calculation_snapshot)
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
    endpoint = _gateway_endpoint(gateway_url) + "/weights/inputs/v2"
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
    response = await post_json(
        endpoint,
        {
            "request": request,
            "calculation_snapshot": calculation,
            "validator_hotkey_signature": str(signature_result["signature"]),
        },
        float(timeout_seconds),
    )
    expected_fields = {
        "request_hash",
        "calculation_snapshot_hash",
        "input_receipt_hashes",
        "gateway_authority_event_hash",
        "upstream_receipt_set",
    }
    if not isinstance(response, Mapping) or set(response) != expected_fields:
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
    receipt_set = _validate_receipt_set(
        value=response["upstream_receipt_set"],
        input_receipt_hashes=input_hashes,
        epoch_id=int(calculation["epoch_id"]),
    )
    authority_hash = str(response.get("gateway_authority_event_hash") or "")
    receipt_hashes = {
        str(receipt["receipt_hash"]) for receipt in receipt_set["receipts"]
    }
    if authority_hash not in receipt_hashes:
        raise GatewayWeightInputsV2Error(
            "gateway allocation authority receipt is absent from the response"
        )
    return {
        "input_receipt_hashes": {
            category: str(input_hashes[category])
            for category in sorted(input_hashes)
        },
        "gateway_authority_event_hash": authority_hash,
        "upstream_receipt_set": receipt_set,
        "request_authorization": dict(signature_result),
    }
