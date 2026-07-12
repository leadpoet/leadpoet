"""Authenticated replacement for qualification's shared block-file read."""

from __future__ import annotations

import base64
from typing import Any

from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ENDPOINT_HOST,
    CHAIN_ENDPOINT_PATH,
    CHAIN_FINALIZATION_EPOCH_BLOCKS,
    CHAIN_RPC_TIMEOUT_MS,
    json_rpc_request,
    parse_finalized_header,
    parse_json_rpc_response,
)


class QualificationEpochGuardV2Error(RuntimeError):
    """Current chain epoch could not be authenticated inside the job."""


class QualificationEpochGuardV2:
    """Match the legacy ``file_epoch > current_epoch`` decision using TLS RPC."""

    def __init__(self, transport: BrokeredProviderTransportV2) -> None:
        self._transport = transport

    def __call__(self, current_epoch: int, container_id: int = 0) -> bool:
        if (
            not isinstance(current_epoch, int)
            or isinstance(current_epoch, bool)
            or current_epoch < 0
            or not isinstance(container_id, int)
            or isinstance(container_id, bool)
            or container_id < 0
        ):
            raise QualificationEpochGuardV2Error("qualification epoch scope is invalid")
        request_id = 1
        body = json_rpc_request("chain_getHeader", (), request_id)
        result = self._transport.execute_http(
            method="POST",
            url="https://%s%s" % (CHAIN_ENDPOINT_HOST, CHAIN_ENDPOINT_PATH),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body=body,
            timeout_ms=CHAIN_RPC_TIMEOUT_MS,
        )
        if result.get("terminal_status") != "authenticated_response":
            raise QualificationEpochGuardV2Error(
                "chain epoch transport failed: %s" % result.get("failure_code")
            )
        if int(result.get("http_status", 0)) != 200:
            raise QualificationEpochGuardV2Error(
                "chain epoch RPC returned authenticated HTTP %s"
                % result.get("http_status")
            )
        try:
            response_body = base64.b64decode(
                str(result.get("body_b64") or ""), validate=True
            )
            header = parse_finalized_header(
                parse_json_rpc_response(response_body, request_id)
            )
        except Exception as exc:
            raise QualificationEpochGuardV2Error(
                "chain epoch RPC response is malformed"
            ) from exc
        observed_epoch = int(header["block"]) // CHAIN_FINALIZATION_EPOCH_BLOCKS
        if observed_epoch > current_epoch:
            print(
                "   WARNING: Container %s: authenticated epoch changed %s -> %s "
                "during batch processing!"
                % (container_id, current_epoch, observed_epoch)
            )
            return True
        return False
