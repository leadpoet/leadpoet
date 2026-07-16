"""Authenticated chain and price inputs for coordinator allocation authority."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ARCHIVE_ENDPOINT_HOST,
    CHAIN_FINALIZATION_EPOCH_BLOCKS,
    CHAIN_ENDPOINT_HOST,
    CHAIN_RPC_METHOD,
    CHAIN_RPC_RETRY_BACKOFF_SECONDS,
    CHAIN_RPC_TIMEOUT_MS,
    ChainSourceV2Error,
    decode_weights_storage,
    decode_selective_metagraph_result,
    encode_selective_metagraph_params,
    json_rpc_request,
    normalize_raw_hash,
    parse_finalized_header,
    parse_json_rpc_response,
    weights_storage_key,
)


CHAIN_ENDPOINT_URL = "https://%s/" % CHAIN_ENDPOINT_HOST
CHAIN_ARCHIVE_ENDPOINT_URL = "https://%s/" % CHAIN_ARCHIVE_ENDPOINT_HOST
COINGECKO_TAO_USD_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bittensor&vs_currencies=usd"
)
ALPHA_PRICE_RUNTIME_METHOD = "SwapRuntimeApi_current_alpha_price"
ALPHA_PRICE_TIMEOUT_MS = 8_000
ALPHA_PRICE_MAX_ATTEMPTS = 3
ALPHA_PRICE_RETRY_BACKOFF_SECONDS = (0.25, 0.5)


class CoordinatorChainSourceV2Error(RuntimeError):
    """An authenticated chain or price source could not be validated."""


def _utc_now_iso(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class CoordinatorChainSourceV2:
    """Read finalized Bittensor state and TAO/USD through the measured broker."""

    def __init__(
        self,
        *,
        execute_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        retry_policy_hashes: Mapping[str, str],
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._execute_provider = execute_provider
        self._retry_policy_hashes = {
            str(key): str(value) for key, value in retry_policy_hashes.items()
        }
        self._sleep = sleep
        self._clock = clock
        for provider_id in ("bittensor_chain", "coingecko"):
            if not self._retry_policy_hashes.get(provider_id):
                raise CoordinatorChainSourceV2Error(
                    "%s retry policy is unavailable" % provider_id
                )

    def read_finalized_metagraph(
        self,
        *,
        netuid: int,
        context: Any,
        attempt_number: int = 0,
    ) -> Dict[str, Any]:
        finalized = self._chain_call(
            method="chain_getFinalizedHead",
            params=(),
            request_id=1,
            logical_operation_id="%s:allocation:finalized-head" % context.job_id,
            attempt_number=attempt_number,
            context=context,
        )
        finalized_hash = normalize_raw_hash(finalized, "finalized head")
        header_value = self._chain_call(
            method="chain_getHeader",
            params=("0x" + finalized_hash,),
            request_id=2,
            logical_operation_id="%s:allocation:finalized-header" % context.job_id,
            attempt_number=attempt_number,
            context=context,
        )
        header = parse_finalized_header(header_value)
        metagraph_value = self._chain_call(
            method="state_call",
            params=(
                CHAIN_RPC_METHOD,
                encode_selective_metagraph_params(netuid=int(netuid)),
                "0x" + finalized_hash,
            ),
            request_id=3,
            logical_operation_id="%s:allocation:metagraph" % context.job_id,
            attempt_number=attempt_number,
            context=context,
        )
        metagraph = decode_selective_metagraph_result(metagraph_value)
        if int(metagraph["netuid"]) != int(netuid):
            raise CoordinatorChainSourceV2Error("allocation metagraph netuid differs")
        if int(metagraph["block"]) != int(header["block"]):
            raise CoordinatorChainSourceV2Error(
                "allocation metagraph and finalized header differ"
            )
        return {
            "finalized_block_hash": finalized_hash,
            "header": header,
            "metagraph": metagraph,
        }

    def read_tao_per_alpha(
        self,
        *,
        netuid: int,
        finalized_block_hash: str,
        context: Any,
        attempt_number: int,
    ) -> float:
        result = self._chain_call(
            method="state_call",
            params=(
                ALPHA_PRICE_RUNTIME_METHOD,
                "0x" + int(netuid).to_bytes(2, "little").hex(),
                "0x" + normalize_raw_hash(
                    finalized_block_hash, "alpha price finalized block"
                ),
            ),
            request_id=4,
            logical_operation_id="%s:allocation:alpha-price" % context.job_id,
            attempt_number=attempt_number,
            context=context,
        )
        text = str(result or "").lower()
        if not text.startswith("0x") or len(text) != 18:
            raise CoordinatorChainSourceV2Error("alpha price response is invalid")
        try:
            raw = bytes.fromhex(text[2:])
        except ValueError as exc:
            raise CoordinatorChainSourceV2Error(
                "alpha price response is invalid hex"
            ) from exc
        return int.from_bytes(raw, "little") / 1_000_000_000.0

    def read_tao_usd(self, *, context: Any, attempt_number: int) -> float:
        result = self._provider_call(
            provider_id="coingecko",
            logical_operation_id="%s:allocation:tao-usd" % context.job_id,
            attempt_number=attempt_number,
            method="GET",
            url=COINGECKO_TAO_USD_URL,
            headers={"accept": "application/json"},
            body=b"",
            timeout_ms=ALPHA_PRICE_TIMEOUT_MS,
            context=context,
        )
        try:
            parsed = json.loads(result["body"].decode("utf-8"))
            value = float(parsed["bittensor"]["usd"])
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise CoordinatorChainSourceV2Error(
                "TAO/USD response is malformed"
            ) from exc
        if value < 0:
            raise CoordinatorChainSourceV2Error("TAO/USD response is negative")
        return value

    def resolve_live_prices(
        self,
        *,
        netuid: int,
        context: Any,
    ) -> Dict[str, Any]:
        last_error: Optional[BaseException] = None
        for attempt_number in range(ALPHA_PRICE_MAX_ATTEMPTS):
            try:
                snapshot = self.read_finalized_metagraph(
                    netuid=netuid,
                    context=context,
                    attempt_number=attempt_number,
                )
                tao_per_alpha = self.read_tao_per_alpha(
                    netuid=netuid,
                    finalized_block_hash=snapshot["finalized_block_hash"],
                    context=context,
                    attempt_number=attempt_number,
                )
                tao_usd = self.read_tao_usd(
                    context=context,
                    attempt_number=attempt_number,
                )
                return {
                    **snapshot,
                    "tao_per_alpha": tao_per_alpha,
                    "tao_usd": tao_usd,
                    "fetched_at": _utc_now_iso(self._clock),
                }
            except Exception as exc:  # every failed attempt has terminal records
                last_error = exc
                if attempt_number < len(ALPHA_PRICE_RETRY_BACKOFF_SECONDS):
                    self._sleep(ALPHA_PRICE_RETRY_BACKOFF_SECONDS[attempt_number])
        raise CoordinatorChainSourceV2Error(
            "live allocation price exhausted measured retries"
        ) from last_error

    def read_historical_finalized_weights(
        self,
        *,
        netuid: int,
        epoch_id: int,
        validator_hotkey: str,
        context: Any,
    ) -> Dict[str, Any]:
        """Read one epoch-end weight vector from the canonical archive node."""

        if not self._retry_policy_hashes.get("bittensor_archive"):
            raise CoordinatorChainSourceV2Error(
                "bittensor_archive retry policy is unavailable"
            )
        normalized_epoch = int(epoch_id)
        if normalized_epoch < 0 or int(netuid) <= 0 or not str(validator_hotkey):
            raise CoordinatorChainSourceV2Error("historical weight request is invalid")
        target_block = (
            (normalized_epoch + 1) * CHAIN_FINALIZATION_EPOCH_BLOCKS - 1
        )
        finalized_hash = normalize_raw_hash(
            self._archive_call(
                method="chain_getFinalizedHead",
                params=(),
                request_id=101,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:finalized-head"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            ),
            "archive finalized head",
        )
        finalized_header = parse_finalized_header(
            self._archive_call(
                method="chain_getHeader",
                params=("0x" + finalized_hash,),
                request_id=102,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:finalized-header"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            )
        )
        if int(finalized_header["block"]) < target_block:
            raise CoordinatorChainSourceV2Error(
                "historical settlement block is not finalized"
            )
        target_hash = normalize_raw_hash(
            self._archive_call(
                method="chain_getBlockHash",
                params=(target_block,),
                request_id=103,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:block-hash"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            ),
            "historical settlement block hash",
        )
        target_header = parse_finalized_header(
            self._archive_call(
                method="chain_getHeader",
                params=("0x" + target_hash,),
                request_id=104,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:block-header"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            )
        )
        if int(target_header["block"]) != target_block:
            raise CoordinatorChainSourceV2Error(
                "historical settlement header differs from target"
            )
        metagraph = decode_selective_metagraph_result(
            self._archive_call(
                method="state_call",
                params=(
                    CHAIN_RPC_METHOD,
                    encode_selective_metagraph_params(netuid=int(netuid)),
                    "0x" + target_hash,
                ),
                request_id=105,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:metagraph"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            )
        )
        if (
            int(metagraph["netuid"]) != int(netuid)
            or int(metagraph["block"]) != target_block
        ):
            raise CoordinatorChainSourceV2Error(
                "historical settlement metagraph differs from target"
            )
        matching_uids = [
            uid
            for uid, hotkey in enumerate(metagraph["hotkeys"])
            if hotkey == str(validator_hotkey)
        ]
        if len(matching_uids) != 1:
            raise CoordinatorChainSourceV2Error(
                "historical validator UID is absent or ambiguous"
            )
        validator_uid = matching_uids[0]
        storage_key = weights_storage_key(
            netuid=int(netuid), validator_uid=validator_uid
        )
        weights = decode_weights_storage(
            self._archive_call(
                method="state_getStorage",
                params=(storage_key, "0x" + target_hash),
                request_id=106,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:weights"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            )
        )
        return {
            "epoch_id": normalized_epoch,
            "netuid": int(netuid),
            "target_block": target_block,
            "target_block_hash": target_hash,
            "target_header": target_header,
            "finalized_head_block": int(finalized_header["block"]),
            "finalized_head_hash": finalized_hash,
            "validator_hotkey": str(validator_hotkey),
            "validator_uid": validator_uid,
            "weights_storage_key": storage_key,
            "weights": [[int(uid), int(weight)] for uid, weight in weights],
        }

    def _chain_call(
        self,
        *,
        method: str,
        params: Sequence[Any],
        request_id: int,
        logical_operation_id: str,
        attempt_number: int,
        context: Any,
    ) -> Any:
        body = json_rpc_request(method, params, request_id)
        result = self._provider_call(
            provider_id="bittensor_chain",
            logical_operation_id=logical_operation_id,
            attempt_number=attempt_number,
            method="POST",
            url=CHAIN_ENDPOINT_URL,
            headers={"accept": "application/json", "content-type": "application/json"},
            body=body,
            timeout_ms=CHAIN_RPC_TIMEOUT_MS,
            context=context,
        )
        try:
            return parse_json_rpc_response(result["body"], request_id)
        except ChainSourceV2Error as exc:
            raise CoordinatorChainSourceV2Error(
                "authenticated chain response is invalid"
            ) from exc

    def _archive_call(
        self,
        *,
        method: str,
        params: Sequence[Any],
        request_id: int,
        logical_operation_id: str,
        context: Any,
    ) -> Any:
        body = json_rpc_request(method, params, request_id)
        last_error: Optional[BaseException] = None
        for attempt_number in range(len(CHAIN_RPC_RETRY_BACKOFF_SECONDS) + 1):
            try:
                result = self._provider_call(
                    provider_id="bittensor_archive",
                    logical_operation_id=logical_operation_id,
                    attempt_number=attempt_number,
                    method="POST",
                    url=CHAIN_ARCHIVE_ENDPOINT_URL,
                    headers={
                        "accept": "application/json",
                        "content-type": "application/json",
                    },
                    body=body,
                    timeout_ms=CHAIN_RPC_TIMEOUT_MS,
                    context=context,
                )
                return parse_json_rpc_response(result["body"], request_id)
            except (CoordinatorChainSourceV2Error, ChainSourceV2Error) as exc:
                last_error = exc
                if attempt_number < len(CHAIN_RPC_RETRY_BACKOFF_SECONDS):
                    self._sleep(CHAIN_RPC_RETRY_BACKOFF_SECONDS[attempt_number])
        raise CoordinatorChainSourceV2Error(
            "authenticated archive request exhausted measured retries"
        ) from last_error

    def _provider_call(
        self,
        *,
        provider_id: str,
        logical_operation_id: str,
        attempt_number: int,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_ms: int,
        context: Any,
    ) -> Dict[str, Any]:
        result = dict(
            self._execute_provider(
                {
                    "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                    "logical_operation_id": logical_operation_id,
                    "job_id": context.job_id,
                    "purpose": context.purpose,
                    "provider_id": provider_id,
                    "attempt_number": int(attempt_number),
                    "method": method,
                    "url": url,
                    "headers": dict(headers),
                    "body_b64": base64.b64encode(body).decode("ascii"),
                    "timeout_ms": int(timeout_ms),
                    "retry_policy_hash": self._retry_policy_hashes[provider_id],
                }
            )
        )
        attempt = result.get("transport_attempt")
        if not isinstance(attempt, Mapping):
            raise CoordinatorChainSourceV2Error("provider terminal attempt is missing")
        context.record_transport(attempt)
        context.record_artifact(str(attempt["request_artifact_hash"]))
        if attempt.get("terminal_status") == "authenticated_response":
            context.record_artifact(str(attempt["response_artifact_hash"]))
        if (
            result.get("terminal_status") != "authenticated_response"
            or not 200 <= int(result.get("http_status") or 0) < 300
        ):
            raise CoordinatorChainSourceV2Error(
                "%s request failed: %s"
                % (
                    provider_id,
                    result.get("failure_code")
                    or "http_%s" % result.get("http_status"),
                )
            )
        try:
            response_body = base64.b64decode(
                str(result.get("body_b64") or ""), validate=True
            )
        except Exception as exc:
            raise CoordinatorChainSourceV2Error(
                "%s response encoding is invalid" % provider_id
            ) from exc
        if sha256_bytes(response_body) != attempt.get("response_hash"):
            raise CoordinatorChainSourceV2Error(
                "%s response differs from terminal record" % provider_id
            )
        return {"body": response_body, "attempt": dict(attempt)}
