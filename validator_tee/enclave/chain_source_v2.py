"""Finalized Bittensor state acquired with TLS inside the validator enclave."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import http.client
import json
import os
import re
import socket
import ssl
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    canonical_json,
    sha256_bytes,
    sha256_json,
    validate_transport_attempt,
)
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ARCHIVE_ENDPOINT_HOST,
    CHAIN_ENDPOINT_HOST,
    CHAIN_ENDPOINT_PATH,
    CHAIN_ENDPOINT_PORT,
    CHAIN_MAX_FINALIZATION_SCAN_BLOCKS,
    CHAIN_MAX_RPC_RESPONSE_BYTES,
    CHAIN_RPC_METHOD,
    CHAIN_RPC_RETRY_BACKOFF_SECONDS,
    CHAIN_RPC_TIMEOUT_MS,
    CHAIN_SUBTENSOR_MAX_TEMPO,
    ChainSourceV2Error,
    chain_source_policy_hash,
    decode_timestamp_now_storage,
    decode_timelocked_weight_commits,
    decode_selective_metagraph_result,
    encode_selective_metagraph_params,
    json_rpc_request,
    normalize_raw_hash,
    parse_finalized_block_extrinsics,
    parse_finalized_header,
    parse_json_rpc_response,
    decode_subnet_epoch_storage,
    subnet_epoch_storage_key,
    timestamp_now_storage_key,
    timelocked_weight_commits_storage_key,
)
from leadpoet_canonical.hotkey_authority_v2 import signed_extrinsic_hash_v2


AF_VSOCK = 40
PARENT_CID = 3
CHAIN_RELAY_VSOCK_PORT = 5002
MAX_CONTROL_BYTES = 16 * 1024
DEFAULT_CA_BUNDLE = "/etc/pki/tls/certs/ca-bundle.crt"
FINALIZATION_RPC_PACING_SECONDS = 1.05


class ValidatorChainSourceV2Error(RuntimeError):
    """The validator enclave could not authenticate a complete chain snapshot."""


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chain_timestamp(timestamp_ms: int) -> str:
    value = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _recv_exact(connection: Any, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = connection.recv(min(64 * 1024, size - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


def _failure_code(exc: BaseException) -> str:
    if isinstance(exc, socket.timeout):
        return "timeout"
    if isinstance(exc, ssl.SSLCertVerificationError):
        return "certificate_invalid"
    if isinstance(exc, ssl.SSLError):
        return "tls_failure"
    if isinstance(exc, ConnectionRefusedError):
        return "connection_refused"
    if isinstance(exc, ConnectionResetError):
        return "connection_reset"
    if isinstance(exc, EOFError):
        return "unexpected_eof"
    if isinstance(
        exc,
        (
            json.JSONDecodeError,
            UnicodeDecodeError,
            ChainSourceV2Error,
            http.client.HTTPException,
        ),
    ):
        return "malformed_reply"
    return "proxy_failure"


class EnclaveChainRpcTransportV2:
    def __init__(
        self,
        *,
        socket_factory: Callable[..., Any] = socket.socket,
        ssl_context_factory: Optional[Callable[[], Any]] = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep: Callable[[float], None] = time.sleep,
        ca_bundle_path: str = DEFAULT_CA_BUNDLE,
        destination_host: str = CHAIN_ENDPOINT_HOST,
    ) -> None:
        self._socket_factory = socket_factory
        self._clock = clock
        self._sleep = sleep
        self._ca_bundle_path = str(ca_bundle_path)
        self._ssl_context_factory = ssl_context_factory or self._default_context
        if destination_host == CHAIN_ENDPOINT_HOST:
            self._provider_id = "bittensor_chain"
        elif destination_host == CHAIN_ARCHIVE_ENDPOINT_HOST:
            self._provider_id = "bittensor_archive"
        else:
            raise ValidatorChainSourceV2Error(
                "chain RPC destination is outside measured policy"
            )
        self._destination_host = str(destination_host)

    def _default_context(self) -> Any:
        if not os.path.isfile(self._ca_bundle_path):
            raise ValidatorChainSourceV2Error("measured CA bundle is unavailable")
        context = ssl.create_default_context(cafile=self._ca_bundle_path)
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context

    def _tls_socket(self) -> Any:
        parent = self._socket_factory(AF_VSOCK, socket.SOCK_STREAM)
        parent.settimeout(CHAIN_RPC_TIMEOUT_MS / 1000.0)
        try:
            parent.connect((PARENT_CID, CHAIN_RELAY_VSOCK_PORT))
            request = canonical_json(
                {
                    "schema_version": "leadpoet.validator_chain_relay.v2",
                    "host": self._destination_host,
                    "port": CHAIN_ENDPOINT_PORT,
                    "policy_hash": chain_source_policy_hash(),
                }
            ).encode("ascii")
            parent.sendall(len(request).to_bytes(4, "big") + request)
            prefix = _recv_exact(parent, 4)
            if len(prefix) != 4:
                raise ValidatorChainSourceV2Error("chain relay response is incomplete")
            size = int.from_bytes(prefix, "big")
            if size < 2 or size > MAX_CONTROL_BYTES:
                raise ValidatorChainSourceV2Error("chain relay response size is invalid")
            body = _recv_exact(parent, size)
            response = json.loads(body.decode("ascii"))
            if response != {
                "status": "connected",
                "policy_hash": chain_source_policy_hash(),
            }:
                raise ValidatorChainSourceV2Error("chain relay refused measured policy")
            return self._ssl_context_factory().wrap_socket(
                parent,
                server_hostname=self._destination_host,
            )
        except Exception:
            parent.close()
            raise

    def _http_post(self, body: bytes) -> Dict[str, Any]:
        tls_socket = self._tls_socket()
        try:
            request = b"".join(
                (
                    b"POST / HTTP/1.1\r\n",
                    ("Host: %s\r\n" % self._destination_host).encode("ascii"),
                    b"Content-Type: application/json\r\n",
                    b"Accept: application/json\r\n",
                    b"Accept-Encoding: identity\r\n",
                    b"Connection: close\r\n",
                    ("Content-Length: %d\r\n\r\n" % len(body)).encode("ascii"),
                    body,
                )
            )
            tls_socket.sendall(request)
            response = http.client.HTTPResponse(tls_socket)
            response.begin()
            response_body = response.read(CHAIN_MAX_RPC_RESPONSE_BYTES + 1)
            if len(response_body) > CHAIN_MAX_RPC_RESPONSE_BYTES:
                raise ValidatorChainSourceV2Error("chain RPC response exceeds limit")
            peer_certificate = tls_socket.getpeercert(binary_form=True)
            if not peer_certificate:
                raise ValidatorChainSourceV2Error("chain TLS peer certificate is absent")
            return {
                "status": int(response.status),
                "body": response_body,
                "tls_peer_chain_hash": sha256_json(
                    [sha256_bytes(peer_certificate)]
                ),
                "tls_protocol": str(tls_socket.version() or "unknown"),
            }
        finally:
            tls_socket.close()

    def call(
        self,
        *,
        method: str,
        params: Sequence[Any],
        request_id: int,
        job_id: str,
        purpose: str,
        logical_operation_id: str,
    ) -> Dict[str, Any]:
        body = json_rpc_request(method, params, request_id)
        attempts = []
        artifacts = []
        last_error = None
        for attempt_number in range(len(CHAIN_RPC_RETRY_BACKOFF_SECONDS) + 1):
            started_at = _timestamp(self._clock)
            request_artifact_hash = sha256_bytes(body)
            artifacts.append(
                {
                    "artifact_hash": request_artifact_hash,
                    "kind": "chain_rpc_request",
                    "body_b64": base64.b64encode(body).decode("ascii"),
                }
            )
            transport = None
            terminal_recorded = False
            try:
                transport = self._http_post(body)
                response_body = bytes(transport["body"])
                response_artifact_hash = sha256_bytes(response_body)
                artifacts.append(
                    {
                        "artifact_hash": response_artifact_hash,
                        "kind": "chain_rpc_response",
                        "body_b64": base64.b64encode(response_body).decode("ascii"),
                    }
                )
                attempt = build_transport_attempt(
                    request_id=os.urandom(16).hex(),
                    logical_operation_id=logical_operation_id,
                    job_id=job_id,
                    purpose=purpose,
                    provider_id=self._provider_id,
                    attempt_number=attempt_number,
                    method="POST",
                    destination_host=self._destination_host,
                    destination_port=CHAIN_ENDPOINT_PORT,
                    path_hash=sha256_json({"path": CHAIN_ENDPOINT_PATH}),
                    nonsecret_headers_hash=sha256_json(
                        {
                            "accept": "application/json",
                            "content-type": "application/json",
                            "host": self._destination_host,
                        }
                    ),
                    body_hash=sha256_bytes(body),
                    credential_ref_hash=sha256_json({"credential": "none"}),
                    retry_policy_hash=chain_source_policy_hash(),
                    timeout_ms=CHAIN_RPC_TIMEOUT_MS,
                    started_at=started_at,
                    terminal_status="authenticated_response",
                    http_status=int(transport["status"]),
                    response_hash=sha256_bytes(response_body),
                    request_artifact_hash=request_artifact_hash,
                    response_artifact_hash=response_artifact_hash,
                    tls_peer_chain_hash=str(transport["tls_peer_chain_hash"]),
                    tls_protocol=str(transport["tls_protocol"]),
                    failure_code=None,
                    completed_at=_timestamp(self._clock),
                )
                validate_transport_attempt(attempt)
                attempts.append(attempt)
                terminal_recorded = True
                if 200 <= int(transport["status"]) < 300:
                    result = parse_json_rpc_response(response_body, request_id)
                    return {
                        "result": result,
                        "attempts": attempts,
                        "artifacts": artifacts,
                    }
                last_error = ValidatorChainSourceV2Error(
                    "chain endpoint returned authenticated HTTP %d"
                    % int(transport["status"])
                )
            except Exception as exc:
                last_error = exc
                if terminal_recorded:
                    if attempt_number < len(CHAIN_RPC_RETRY_BACKOFF_SECONDS):
                        self._sleep(CHAIN_RPC_RETRY_BACKOFF_SECONDS[attempt_number])
                    continue
                attempt = build_transport_attempt(
                    request_id=os.urandom(16).hex(),
                    logical_operation_id=logical_operation_id,
                    job_id=job_id,
                    purpose=purpose,
                    provider_id=self._provider_id,
                    attempt_number=attempt_number,
                    method="POST",
                    destination_host=self._destination_host,
                    destination_port=CHAIN_ENDPOINT_PORT,
                    path_hash=sha256_json({"path": CHAIN_ENDPOINT_PATH}),
                    nonsecret_headers_hash=sha256_json(
                        {
                            "accept": "application/json",
                            "content-type": "application/json",
                            "host": self._destination_host,
                        }
                    ),
                    body_hash=sha256_bytes(body),
                    credential_ref_hash=sha256_json({"credential": "none"}),
                    retry_policy_hash=chain_source_policy_hash(),
                    timeout_ms=CHAIN_RPC_TIMEOUT_MS,
                    started_at=started_at,
                    terminal_status="transport_failure",
                    http_status=None,
                    response_hash=None,
                    request_artifact_hash=request_artifact_hash,
                    response_artifact_hash=None,
                    tls_peer_chain_hash=(
                        str(transport["tls_peer_chain_hash"])
                        if transport is not None
                        else None
                    ),
                    tls_protocol=(
                        str(transport["tls_protocol"])
                        if transport is not None
                        else None
                    ),
                    failure_code=_failure_code(exc),
                    completed_at=_timestamp(self._clock),
                )
                validate_transport_attempt(attempt)
                attempts.append(attempt)
            if attempt_number < len(CHAIN_RPC_RETRY_BACKOFF_SECONDS):
                self._sleep(CHAIN_RPC_RETRY_BACKOFF_SECONDS[attempt_number])
        raise ValidatorChainSourceV2Error("chain RPC exhausted measured retries") from last_error


class ValidatorChainSourceV2:
    def __init__(
        self,
        *,
        rpc_call: Optional[Callable[..., Mapping[str, Any]]] = None,
        archive_rpc_call: Optional[Callable[..., Mapping[str, Any]]] = None,
        finalization_sleep: Callable[[float], None] = time.sleep,
        epoch_authority_supplier: Optional[
            Callable[[], Optional[Mapping[str, Any]]]
        ] = None,
    ) -> None:
        self._rpc_call = rpc_call or EnclaveChainRpcTransportV2().call
        if archive_rpc_call is not None:
            self._archive_rpc_call = archive_rpc_call
        elif rpc_call is None:
            self._archive_rpc_call = EnclaveChainRpcTransportV2(
                destination_host=CHAIN_ARCHIVE_ENDPOINT_HOST,
            ).call
        else:
            self._archive_rpc_call = None
        self._finalization_sleep = finalization_sleep
        if epoch_authority_supplier is None:
            raise ValidatorChainSourceV2Error(
                "official SN71 epoch authority supplier is unavailable"
            )
        self._epoch_authority_supplier = epoch_authority_supplier

    def _read_stateful_epoch_authority(
        self,
        *,
        configuration: Mapping[str, Any],
        finalized_hash: str,
        header: Mapping[str, Any],
        netuid: int,
        settlement_epoch_id: int,
        chain_job: str,
        request_id_start: int,
        snapshot_job_override: Optional[str] = None,
        boundary_job_override: Optional[str] = None,
        historical_snapshot: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(configuration, Mapping) or set(configuration) != {
            "mode",
            "cutover_manifest",
        }:
            raise ValidatorChainSourceV2Error(
                "measured epoch authority configuration is invalid"
            )
        if configuration.get("mode") != "stateful_v1":
            raise ValidatorChainSourceV2Error("measured epoch mode is invalid")
        cutover = configuration.get("cutover_manifest")
        cutover_fields = {
            "schema_version",
            "epoch_scheme",
            "network_genesis_hash",
            "netuid",
            "cutover_block",
            "cutover_block_hash",
            "first_subnet_epoch_index",
            "first_settlement_epoch_id",
            "last_legacy_epoch_id",
            "mapping_hash",
        }
        if not isinstance(cutover, Mapping) or set(cutover) != cutover_fields:
            raise ValidatorChainSourceV2Error("measured epoch cutover is absent")
        if (
            cutover.get("schema_version") != "leadpoet.subnet_epoch_cutover.v1"
            or cutover.get("epoch_scheme") != "bittensor.subnet_epoch_index.v1"
        ):
            raise ValidatorChainSourceV2Error("measured epoch cutover schema is invalid")
        for field in (
            "netuid",
            "cutover_block",
            "first_subnet_epoch_index",
            "first_settlement_epoch_id",
            "last_legacy_epoch_id",
        ):
            if (
                not isinstance(cutover.get(field), int)
                or isinstance(cutover.get(field), bool)
                or int(cutover[field]) < 0
            ):
                raise ValidatorChainSourceV2Error(
                    "measured epoch cutover %s is invalid" % field
                )
        try:
            normalize_raw_hash(cutover["network_genesis_hash"], "network genesis")
            normalize_raw_hash(cutover["cutover_block_hash"], "cutover block hash")
        except ChainSourceV2Error as exc:
            raise ValidatorChainSourceV2Error(
                "measured epoch cutover hash is invalid"
            ) from exc
        cutover_body = {
            field: cutover[field]
            for field in (
                "schema_version",
                "epoch_scheme",
                "network_genesis_hash",
                "netuid",
                "cutover_block",
                "cutover_block_hash",
                "first_subnet_epoch_index",
                "first_settlement_epoch_id",
                "last_legacy_epoch_id",
            )
        }
        if (
            int(cutover["first_settlement_epoch_id"])
            != int(cutover["last_legacy_epoch_id"]) + 1
            or str(cutover.get("mapping_hash") or "").lower()
            != sha256_json(cutover_body)
        ):
            raise ValidatorChainSourceV2Error("measured epoch cutover mapping is invalid")
        if int(cutover.get("netuid", -1)) != int(netuid):
            raise ValidatorChainSourceV2Error("measured epoch cutover netuid differs")

        attempts = []
        artifacts = []
        request_id = int(request_id_start)
        snapshot_job = snapshot_job_override or (
            "subnet-epoch-snapshot:%d:%d"
            % (int(settlement_epoch_id), int(header["block"]))
        )

        def invoke(
            adapter: Callable[..., Dict[str, Any]],
            method: str,
            params: Sequence[Any],
            operation: str,
        ) -> Any:
            nonlocal request_id
            result = adapter(
                method=method,
                params=params,
                request_id=request_id,
                job_id=snapshot_job,
                purpose="validator.subnet_epoch_snapshot.v2",
                logical_operation_id=snapshot_job + ":" + operation,
            )
            request_id += 1
            attempts.extend(result["attempts"])
            artifacts.extend(result["artifacts"])
            return result["result"]

        def live_call(method: str, params: Sequence[Any], operation: str) -> Any:
            return invoke(self._call, method, params, operation)

        def archive_call(
            method: str,
            params: Sequence[Any],
            operation: str,
        ) -> Any:
            return invoke(self._archive_call, method, params, operation)

        genesis_hash = normalize_raw_hash(
            archive_call("chain_getBlockHash", [0], "genesis-hash"),
            "genesis block hash",
        )
        cutover_block = int(cutover["cutover_block"])
        observed_cutover_hash = normalize_raw_hash(
            archive_call(
                "chain_getBlockHash",
                [cutover_block],
                "cutover-block-hash",
            ),
            "cutover block hash",
        )
        if "0x" + genesis_hash != str(cutover["network_genesis_hash"]):
            raise ValidatorChainSourceV2Error("chain genesis differs from cutover")
        if "0x" + observed_cutover_hash != str(cutover["cutover_block_hash"]):
            raise ValidatorChainSourceV2Error("chain cutover block hash differs")
        if cutover_block <= 0:
            raise ValidatorChainSourceV2Error(
                "measured cutover block has no predecessor"
            )
        cutover_header = parse_finalized_header(
            archive_call(
                "chain_getHeader",
                ["0x" + observed_cutover_hash],
                "cutover-header",
            )
        )
        if int(cutover_header["block"]) != cutover_block:
            raise ValidatorChainSourceV2Error(
                "chain cutover header block differs"
            )
        cutover_index_key = subnet_epoch_storage_key(
            storage_name="SubnetEpochIndex",
            netuid=int(netuid),
        )
        cutover_index = decode_subnet_epoch_storage(
            archive_call(
                "state_getStorage",
                [cutover_index_key, "0x" + observed_cutover_hash],
                "cutover-subnet-epoch-index",
            ),
            storage_name="SubnetEpochIndex",
        )
        predecessor_hash = normalize_raw_hash(
            archive_call(
                "chain_getBlockHash",
                [cutover_block - 1],
                "cutover-predecessor-hash",
            ),
            "cutover predecessor block hash",
        )
        predecessor_index = decode_subnet_epoch_storage(
            archive_call(
                "state_getStorage",
                [cutover_index_key, "0x" + predecessor_hash],
                "cutover-predecessor-subnet-epoch-index",
            ),
            storage_name="SubnetEpochIndex",
        )
        if (
            cutover_index != int(cutover["first_subnet_epoch_index"])
            or predecessor_index + 1 != cutover_index
        ):
            raise ValidatorChainSourceV2Error(
                "measured cutover is not an official epoch transition"
            )

        snapshot_call = archive_call if historical_snapshot else live_call
        decoded = {}
        for storage_name in (
            "Tempo",
            "LastEpochBlock",
            "PendingEpochAt",
            "SubnetEpochIndex",
            "BlocksSinceLastStep",
        ):
            storage_value = snapshot_call(
                "state_getStorage",
                [
                    subnet_epoch_storage_key(
                        storage_name=storage_name,
                        netuid=int(netuid),
                    ),
                    "0x" + finalized_hash,
                ],
                "epoch-storage-" + storage_name.lower(),
            )
            decoded[storage_name] = decode_subnet_epoch_storage(
                storage_value,
                storage_name=storage_name,
            )
        current_observed_at = _chain_timestamp(
            decode_timestamp_now_storage(
                snapshot_call(
                    "state_getStorage",
                    [timestamp_now_storage_key(), "0x" + finalized_hash],
                    "epoch-storage-timestamp-now",
                )
            )
        )

        current_block = int(header["block"])
        tempo = int(decoded["Tempo"])
        last_epoch_block = int(decoded["LastEpochBlock"])
        pending_epoch_at = int(decoded["PendingEpochAt"])
        subnet_epoch_index = int(decoded["SubnetEpochIndex"])
        blocks_since_last_step = int(decoded["BlocksSinceLastStep"])
        if tempo <= 0 or last_epoch_block > current_block:
            raise ValidatorChainSourceV2Error("finalized epoch schedule is invalid")
        first_index = int(cutover["first_subnet_epoch_index"])
        if subnet_epoch_index < first_index:
            raise ValidatorChainSourceV2Error(
                "finalized epoch state predates measured cutover"
            )
        mapped_epoch = int(cutover["first_settlement_epoch_id"]) + (
            subnet_epoch_index - first_index
        )
        if mapped_epoch != int(settlement_epoch_id):
            raise ValidatorChainSourceV2Error(
                "finalized subnet epoch differs from requested settlement epoch"
            )
        if current_block < cutover_block:
            raise ValidatorChainSourceV2Error("finalized block predates cutover")
        automatic_next = last_epoch_block + tempo
        if blocks_since_last_step > CHAIN_SUBTENSOR_MAX_TEMPO:
            next_epoch_block = current_block
        else:
            safety_next = current_block + (
                CHAIN_SUBTENSOR_MAX_TEMPO + 1 - blocks_since_last_step
            )
            next_epoch_block = (
                min(automatic_next, pending_epoch_at, safety_next)
                if pending_epoch_at > 0
                else min(automatic_next, safety_next)
            )
        epoch_ref = sha256_json(
            {
                "epoch_scheme": cutover["epoch_scheme"],
                "network_genesis_hash": cutover["network_genesis_hash"],
                "netuid": int(netuid),
                "subnet_epoch_index": subnet_epoch_index,
            }
        )
        boundary_job = boundary_job_override or (
            "subnet-epoch-boundary:%d" % int(settlement_epoch_id)
        )

        def boundary_call(
            method: str,
            params: Sequence[Any],
            operation: str,
        ) -> Any:
            nonlocal request_id
            result = self._archive_call(
                method=method,
                params=params,
                request_id=request_id,
                job_id=boundary_job,
                purpose="validator.subnet_epoch_snapshot.v2",
                logical_operation_id=boundary_job + ":" + operation,
            )
            request_id += 1
            attempts.extend(result["attempts"])
            artifacts.extend(result["artifacts"])
            return result["result"]

        index_key = subnet_epoch_storage_key(
            storage_name="SubnetEpochIndex",
            netuid=int(netuid),
        )
        probed = {}

        def probe_index(
            block_number: int,
            known_hash: Optional[str] = None,
            *,
            live: bool = False,
        ) -> Any:
            normalized_block = int(block_number)
            cached = probed.get(normalized_block)
            if cached is not None:
                return cached
            block_hash = known_hash or normalize_raw_hash(
                boundary_call(
                    "chain_getBlockHash",
                    [normalized_block],
                    "search-block-hash-%d" % normalized_block,
                ),
                "subnet epoch search block hash",
            )
            index_call = live_call if live else boundary_call
            observed_index = decode_subnet_epoch_storage(
                index_call(
                    "state_getStorage",
                    [index_key, "0x" + block_hash],
                    "search-index-%d" % normalized_block,
                ),
                storage_name="SubnetEpochIndex",
            )
            probed[normalized_block] = (block_hash, observed_index)
            return probed[normalized_block]

        _current_hash, current_index_for_search = probe_index(
            current_block,
            finalized_hash,
            live=not historical_snapshot,
        )
        if current_index_for_search != subnet_epoch_index:
            raise ValidatorChainSourceV2Error(
                "finalized subnet epoch changed during boundary search"
            )
        if subnet_epoch_index == first_index:
            boundary_block = cutover_block
            boundary_hash, boundary_index = probe_index(
                boundary_block,
                observed_cutover_hash,
            )
            if boundary_index != subnet_epoch_index:
                raise ValidatorChainSourceV2Error(
                    "cutover boundary subnet epoch index differs"
                )
        else:
            high = current_block
            step = 1
            low = None
            while low is None:
                candidate = max(cutover_block, current_block - step)
                _candidate_hash, candidate_index = probe_index(candidate)
                if candidate_index > subnet_epoch_index:
                    raise ValidatorChainSourceV2Error(
                        "historical subnet epoch index is not monotonic"
                    )
                if candidate_index < subnet_epoch_index:
                    low = candidate
                    break
                high = candidate
                if candidate == cutover_block:
                    raise ValidatorChainSourceV2Error(
                        "subnet epoch boundary does not follow cutover"
                    )
                step *= 2
            while high - low > 1:
                midpoint = low + (high - low) // 2
                _midpoint_hash, midpoint_index = probe_index(midpoint)
                if midpoint_index > subnet_epoch_index:
                    raise ValidatorChainSourceV2Error(
                        "historical subnet epoch index is not monotonic"
                    )
                if midpoint_index < subnet_epoch_index:
                    low = midpoint
                else:
                    high = midpoint
            boundary_block = high
            boundary_hash, boundary_index = probe_index(boundary_block)
            _predecessor_hash, predecessor_index = probe_index(boundary_block - 1)
            if (
                boundary_index != subnet_epoch_index
                or predecessor_index + 1 != subnet_epoch_index
            ):
                raise ValidatorChainSourceV2Error(
                    "subnet epoch boundary transition is invalid"
                )
        boundary_header = parse_finalized_header(
            boundary_call(
                "chain_getHeader",
                ["0x" + boundary_hash],
                "boundary-header",
            )
        )
        if int(boundary_header["block"]) != boundary_block:
            raise ValidatorChainSourceV2Error(
                "subnet epoch boundary header block differs"
            )
        boundary_decoded = {}
        for storage_name in (
            "Tempo",
            "LastEpochBlock",
            "PendingEpochAt",
            "SubnetEpochIndex",
            "BlocksSinceLastStep",
        ):
            boundary_decoded[storage_name] = decode_subnet_epoch_storage(
                boundary_call(
                    "state_getStorage",
                    [
                        subnet_epoch_storage_key(
                            storage_name=storage_name,
                            netuid=int(netuid),
                        ),
                        "0x" + boundary_hash,
                    ],
                    "boundary-storage-" + storage_name.lower(),
                ),
                storage_name=storage_name,
            )
        boundary_observed_at = _chain_timestamp(
            decode_timestamp_now_storage(
                boundary_call(
                    "state_getStorage",
                    [timestamp_now_storage_key(), "0x" + boundary_hash],
                    "boundary-storage-timestamp-now",
                )
            )
        )
        boundary_tempo = int(boundary_decoded["Tempo"])
        boundary_last_epoch_block = int(boundary_decoded["LastEpochBlock"])
        boundary_pending_epoch_at = int(boundary_decoded["PendingEpochAt"])
        boundary_subnet_epoch_index = int(boundary_decoded["SubnetEpochIndex"])
        boundary_blocks_since_last_step = int(
            boundary_decoded["BlocksSinceLastStep"]
        )
        if (
            boundary_tempo <= 0
            or boundary_last_epoch_block != boundary_block
            or boundary_subnet_epoch_index != subnet_epoch_index
        ):
            raise ValidatorChainSourceV2Error(
                "historical subnet epoch boundary state differs"
            )
        boundary_next_epoch_block = boundary_last_epoch_block + boundary_tempo
        if boundary_blocks_since_last_step > CHAIN_SUBTENSOR_MAX_TEMPO:
            boundary_next_epoch_block = boundary_block
        else:
            boundary_safety_next = boundary_block + (
                CHAIN_SUBTENSOR_MAX_TEMPO
                + 1
                - boundary_blocks_since_last_step
            )
            boundary_next_epoch_block = (
                min(
                    boundary_next_epoch_block,
                    boundary_pending_epoch_at,
                    boundary_safety_next,
                )
                if boundary_pending_epoch_at > 0
                else min(boundary_next_epoch_block, boundary_safety_next)
            )
        boundary_snapshot = {
            "schema_version": "leadpoet.subnet_epoch_snapshot.v1",
            "epoch_scheme": cutover["epoch_scheme"],
            "network_genesis_hash": cutover["network_genesis_hash"],
            "netuid": int(netuid),
            "head_kind": "finalized",
            "block_hash": "0x" + boundary_hash,
            "current_block": boundary_last_epoch_block,
            "last_epoch_block": boundary_last_epoch_block,
            "pending_epoch_at": boundary_pending_epoch_at,
            "subnet_epoch_index": boundary_subnet_epoch_index,
            "tempo": boundary_tempo,
            "blocks_since_last_step": boundary_blocks_since_last_step,
            "observed_at": boundary_observed_at,
            "epoch_id": boundary_subnet_epoch_index,
            "epoch_ref": epoch_ref,
            "epoch_block": 0,
            "next_epoch_block": boundary_next_epoch_block,
            "blocks_remaining": max(
                0,
                boundary_next_epoch_block - boundary_last_epoch_block,
            ),
            "settlement_epoch_id": mapped_epoch,
            "cutover_mapping_hash": cutover["mapping_hash"],
        }
        return {
            "authority": {
                "schema_version": "leadpoet.subnet_epoch_snapshot.v1",
                "epoch_scheme": cutover["epoch_scheme"],
                "network_genesis_hash": cutover["network_genesis_hash"],
                "netuid": int(netuid),
                "head_kind": "finalized",
                "block_hash": "0x" + finalized_hash,
                "current_block": current_block,
                "last_epoch_block": last_epoch_block,
                "pending_epoch_at": pending_epoch_at,
                "subnet_epoch_index": subnet_epoch_index,
                "tempo": tempo,
                "blocks_since_last_step": blocks_since_last_step,
                "observed_at": current_observed_at,
                "epoch_id": subnet_epoch_index,
                "epoch_block": current_block - last_epoch_block,
                "next_epoch_block": next_epoch_block,
                "blocks_remaining": max(0, next_epoch_block - current_block),
                "epoch_ref": epoch_ref,
                "settlement_epoch_id": mapped_epoch,
                "cutover_mapping_hash": cutover["mapping_hash"],
            },
            "boundary_snapshot": boundary_snapshot,
            "snapshot_job": snapshot_job,
            "boundary_job": boundary_job,
            "attempts": attempts,
            "artifacts": artifacts,
            "next_request_id": request_id,
        }

    def capture_stateful_epoch_boundary(
        self,
        *,
        cutover_manifest: Mapping[str, Any],
        settlement_epoch_id: int,
        capture_scope: str,
    ) -> Dict[str, Any]:
        """Capture a proposed stateful cutover without activating stateful mode.

        This explicit shadow path is usable while the validator's measured
        runtime remains in legacy mode.  The proposed manifest is not trusted:
        ``_read_stateful_epoch_authority`` authenticates its genesis, cutover
        block, official index, and predecessor transition against exact chain
        state before returning either signed-document candidate.  A proposed
        boundary may be historical, but it must already be at or below the
        authenticated finalized head and its exact height must still resolve
        to the manifest's canonical block hash.
        """

        if (
            not isinstance(settlement_epoch_id, int)
            or isinstance(settlement_epoch_id, bool)
            or settlement_epoch_id < 0
        ):
            raise ValidatorChainSourceV2Error(
                "candidate settlement epoch is invalid"
            )
        if not isinstance(capture_scope, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", capture_scope
        ):
            raise ValidatorChainSourceV2Error(
                "candidate capture scope is invalid"
            )
        configuration = {
            "mode": "stateful_v1",
            "cutover_manifest": dict(cutover_manifest),
        }
        cutover_block = cutover_manifest.get("cutover_block")
        if (
            not isinstance(cutover_block, int)
            or isinstance(cutover_block, bool)
            or cutover_block <= 0
        ):
            raise ValidatorChainSourceV2Error(
                "candidate cutover block is invalid"
            )
        current_job = "subnet-epoch-capture-current:%d:%s" % (
            settlement_epoch_id,
            capture_scope[len("sha256:") :],
        )
        attempts = []
        artifacts = []

        finalized = self._call(
            method="chain_getFinalizedHead",
            params=[],
            request_id=1,
            job_id=current_job,
            purpose="validator.subnet_epoch_snapshot.v2",
            logical_operation_id=current_job + ":finalized-head",
        )
        attempts.extend(finalized["attempts"])
        artifacts.extend(finalized["artifacts"])
        finalized_hash = normalize_raw_hash(
            finalized["result"], "candidate finalized head"
        )
        header_result = self._call(
            method="chain_getHeader",
            params=["0x" + finalized_hash],
            request_id=2,
            job_id=current_job,
            purpose="validator.subnet_epoch_snapshot.v2",
            logical_operation_id=current_job + ":finalized-header",
        )
        attempts.extend(header_result["attempts"])
        artifacts.extend(header_result["artifacts"])
        finalized_header = parse_finalized_header(header_result["result"])
        if int(finalized_header["block"]) < cutover_block:
            raise ValidatorChainSourceV2Error(
                "candidate cutover block is not finalized"
            )
        canonical_cutover = self._archive_call(
            method="chain_getBlockHash",
            params=[cutover_block],
            request_id=3,
            job_id=current_job,
            purpose="validator.subnet_epoch_snapshot.v2",
            logical_operation_id=current_job + ":canonical-cutover-hash",
        )
        attempts.extend(canonical_cutover["attempts"])
        artifacts.extend(canonical_cutover["artifacts"])
        canonical_cutover_hash = normalize_raw_hash(
            canonical_cutover["result"], "candidate canonical cutover block"
        )
        try:
            manifest_cutover_hash = normalize_raw_hash(
                cutover_manifest.get("cutover_block_hash"),
                "candidate manifest cutover block",
            )
        except ChainSourceV2Error as exc:
            raise ValidatorChainSourceV2Error(
                "candidate cutover block hash is invalid"
            ) from exc
        if canonical_cutover_hash != manifest_cutover_hash:
            raise ValidatorChainSourceV2Error(
                "candidate cutover block is not canonical"
            )
        cutover_header_result = self._archive_call(
            method="chain_getHeader",
            params=["0x" + canonical_cutover_hash],
            request_id=4,
            job_id=current_job,
            purpose="validator.subnet_epoch_snapshot.v2",
            logical_operation_id=current_job + ":canonical-cutover-header",
        )
        attempts.extend(cutover_header_result["attempts"])
        artifacts.extend(cutover_header_result["artifacts"])
        cutover_header = parse_finalized_header(cutover_header_result["result"])
        if int(cutover_header["block"]) != cutover_block:
            raise ValidatorChainSourceV2Error(
                "candidate canonical cutover header differs"
            )
        stateful = self._read_stateful_epoch_authority(
            configuration=configuration,
            finalized_hash=canonical_cutover_hash,
            header=cutover_header,
            netuid=int(cutover_manifest.get("netuid", -1)),
            settlement_epoch_id=settlement_epoch_id,
            chain_job=current_job,
            request_id_start=5,
            snapshot_job_override=current_job,
            boundary_job_override=current_job,
            historical_snapshot=True,
        )
        attempts.extend(stateful["attempts"])
        artifacts.extend(stateful["artifacts"])
        if stateful["authority"] != stateful["boundary_snapshot"]:
            raise ValidatorChainSourceV2Error(
                "candidate capture did not resolve the proposed boundary"
            )
        return {
            "finalized_block_hash": finalized_hash,
            "header": finalized_header,
            "epoch_authority": stateful["authority"],
            "epoch_boundary": stateful["boundary_snapshot"],
            "attempts": attempts,
            "artifacts": artifacts,
            "jobs": {
                "subnet_epoch_snapshot": current_job,
                "subnet_epoch_boundary": stateful["boundary_job"],
            },
        }

    def read_finalized_snapshot(self, *, netuid: int, epoch_id: int) -> Dict[str, Any]:
        chain_job = "chain-state:%d" % int(epoch_id)
        metagraph_job = "metagraph-state:%d" % int(epoch_id)
        all_attempts = []
        all_artifacts = []

        finalized = self._call(
            method="chain_getFinalizedHead",
            params=[],
            request_id=1,
            job_id=chain_job,
            purpose="validator.chain_state.v2",
            logical_operation_id=chain_job + ":finalized-head",
        )
        all_attempts.extend(finalized["attempts"])
        all_artifacts.extend(finalized["artifacts"])
        finalized_hash = normalize_raw_hash(finalized["result"], "finalized head")

        header_result = self._call(
            method="chain_getHeader",
            params=["0x" + finalized_hash],
            request_id=2,
            job_id=chain_job,
            purpose="validator.chain_state.v2",
            logical_operation_id=chain_job + ":header",
        )
        all_attempts.extend(header_result["attempts"])
        all_artifacts.extend(header_result["artifacts"])
        header = parse_finalized_header(header_result["result"])
        epoch_configuration = self._epoch_authority_supplier()
        epoch_authority = None
        epoch_boundary = None
        epoch_snapshot_job = None
        epoch_boundary_job = None
        selector_job = None
        next_request_id = 3
        historical_snapshot = False
        metagraph_call = self._call
        if epoch_configuration is None:
            raise ValidatorChainSourceV2Error(
                "official SN71 epoch authority is unavailable"
            )

        # Weight publication starts near the end of an epoch. The finalized
        # head can move into the next official epoch while gateway evidence is
        # being assembled. Select the just-finished epoch's final block from
        # authenticated state instead of binding the old settlement to a new
        # epoch head.
        cutover = epoch_configuration.get("cutover_manifest")
        if (
            isinstance(epoch_configuration, Mapping)
            and epoch_configuration.get("mode") == "stateful_v1"
            and isinstance(cutover, Mapping)
        ):
            cutover_body_fields = (
                "schema_version",
                "epoch_scheme",
                "network_genesis_hash",
                "netuid",
                "cutover_block",
                "cutover_block_hash",
                "first_subnet_epoch_index",
                "first_settlement_epoch_id",
                "last_legacy_epoch_id",
            )
            try:
                cutover_body = {
                    field: cutover[field] for field in cutover_body_fields
                }
                mapping_valid = (
                    set(cutover) == set(cutover_body_fields) | {"mapping_hash"}
                    and cutover.get("schema_version")
                    == "leadpoet.subnet_epoch_cutover.v1"
                    and cutover.get("epoch_scheme")
                    == "bittensor.subnet_epoch_index.v1"
                    and int(cutover["netuid"]) == int(netuid)
                    and str(cutover.get("mapping_hash") or "").lower()
                    == sha256_json(cutover_body)
                )
            except (KeyError, TypeError, ValueError):
                mapping_valid = False
            if mapping_valid:
                if self._archive_rpc_call is None:
                    raise ValidatorChainSourceV2Error(
                        "archive RPC adapter is required for historical epoch state"
                    )
                # Selection and the resulting exact state reads are one
                # measured operation. Keeping them in one receipt scope makes
                # every authenticated selector attempt part of the snapshot
                # receipt instead of leaving valid transport evidence
                # unclaimed in the final authority graph.
                selector_job = "subnet-epoch-selector:%d" % int(epoch_id)

                def selector_call(
                    storage_name: str,
                    request_id: int,
                ) -> int:
                    result = self._call(
                        method="state_getStorage",
                        params=[
                            subnet_epoch_storage_key(
                                storage_name=storage_name,
                                netuid=int(netuid),
                            ),
                            "0x" + finalized_hash,
                        ],
                        request_id=request_id,
                        job_id=selector_job,
                        purpose="validator.subnet_epoch_snapshot.v2",
                        logical_operation_id=(
                            selector_job + ":" + storage_name.lower()
                        ),
                    )
                    all_attempts.extend(result["attempts"])
                    all_artifacts.extend(result["artifacts"])
                    return decode_subnet_epoch_storage(
                        result["result"], storage_name=storage_name
                    )

                finalized_index = selector_call("SubnetEpochIndex", 3)
                finalized_last_epoch_block = selector_call("LastEpochBlock", 4)
                finalized_settlement_epoch = int(
                    cutover["first_settlement_epoch_id"]
                ) + (
                    finalized_index - int(cutover["first_subnet_epoch_index"])
                )
                next_request_id = 5
                if finalized_settlement_epoch == int(epoch_id) + 1:
                    target_block = int(finalized_last_epoch_block) - 1
                    if target_block < int(cutover["cutover_block"]):
                        raise ValidatorChainSourceV2Error(
                            "requested settlement epoch has no finalized predecessor"
                        )
                    target_hash_result = self._archive_call(
                        method="chain_getBlockHash",
                        params=[target_block],
                        request_id=5,
                        job_id=selector_job,
                        purpose="validator.subnet_epoch_snapshot.v2",
                        logical_operation_id=selector_job + ":predecessor-hash",
                    )
                    all_attempts.extend(target_hash_result["attempts"])
                    all_artifacts.extend(target_hash_result["artifacts"])
                    finalized_hash = normalize_raw_hash(
                        target_hash_result["result"],
                        "settlement predecessor block hash",
                    )
                    target_header_result = self._archive_call(
                        method="chain_getHeader",
                        params=["0x" + finalized_hash],
                        request_id=6,
                        job_id=selector_job,
                        purpose="validator.subnet_epoch_snapshot.v2",
                        logical_operation_id=selector_job + ":predecessor-header",
                    )
                    all_attempts.extend(target_header_result["attempts"])
                    all_artifacts.extend(target_header_result["artifacts"])
                    header = parse_finalized_header(target_header_result["result"])
                    if int(header["block"]) != target_block:
                        raise ValidatorChainSourceV2Error(
                            "settlement predecessor header differs"
                        )
                    historical_snapshot = True
                    metagraph_call = self._archive_call
                    next_request_id = 7
                elif finalized_settlement_epoch != int(epoch_id):
                    raise ValidatorChainSourceV2Error(
                        "requested settlement epoch is not current or just finalized"
                    )
        stateful = self._read_stateful_epoch_authority(
            configuration=epoch_configuration,
            finalized_hash=finalized_hash,
            header=header,
            netuid=int(netuid),
            settlement_epoch_id=int(epoch_id),
            chain_job=chain_job,
            request_id_start=next_request_id,
            historical_snapshot=historical_snapshot,
            snapshot_job_override=selector_job,
        )
        epoch_authority = stateful["authority"]
        epoch_boundary = stateful["boundary_snapshot"]
        epoch_snapshot_job = stateful["snapshot_job"]
        epoch_boundary_job = stateful["boundary_job"]
        all_attempts.extend(stateful["attempts"])
        all_artifacts.extend(stateful["artifacts"])
        next_request_id = int(stateful["next_request_id"])

        metagraph_result = metagraph_call(
            method="state_call",
            params=[
                CHAIN_RPC_METHOD,
                encode_selective_metagraph_params(netuid=int(netuid)),
                "0x" + finalized_hash,
            ],
            request_id=next_request_id,
            job_id=metagraph_job,
            purpose="validator.metagraph_state.v2",
            logical_operation_id=metagraph_job + ":selective-metagraph",
        )
        all_attempts.extend(metagraph_result["attempts"])
        all_artifacts.extend(metagraph_result["artifacts"])
        metagraph = decode_selective_metagraph_result(metagraph_result["result"])
        if int(metagraph["netuid"]) != int(netuid):
            raise ValidatorChainSourceV2Error("chain metagraph netuid differs")
        if int(metagraph["block"]) != int(header["block"]):
            raise ValidatorChainSourceV2Error("metagraph and finalized block differ")
        return {
            "finalized_block_hash": finalized_hash,
            "header": header,
            "metagraph": metagraph,
            "epoch_authority": epoch_authority,
            "epoch_boundary": epoch_boundary,
            "attempts": all_attempts,
            "artifacts": all_artifacts,
            "jobs": {
                "chain_state": chain_job,
                "metagraph_state": metagraph_job,
                "subnet_epoch_snapshot": epoch_snapshot_job,
                "subnet_epoch_boundary": epoch_boundary_job,
            },
        }

    def find_finalized_extrinsic_inclusion(
        self,
        *,
        expected_extrinsics: Mapping[str, str],
        expected_commitments: Mapping[str, Mapping[str, Any]],
        minimum_block: int,
        maximum_block: int,
        epoch_id: int,
    ) -> Dict[str, Any]:
        """Find one exact enclave-built extrinsic in authenticated finalized blocks."""

        if not isinstance(expected_extrinsics, Mapping) or not expected_extrinsics:
            raise ValidatorChainSourceV2Error("expected extrinsic set is empty")
        if (
            not isinstance(expected_commitments, Mapping)
            or set(expected_commitments) != set(expected_extrinsics)
        ):
            raise ValidatorChainSourceV2Error(
                "expected commitment set differs from extrinsics"
            )
        normalized = {}
        for expected_hash, encoded in expected_extrinsics.items():
            hash_text = str(expected_hash or "").lower()
            encoded_text = str(encoded or "").lower()
            if hash_text.startswith("0x") and len(hash_text) == 66:
                pass
            else:
                raise ValidatorChainSourceV2Error("expected extrinsic hash is invalid")
            try:
                raw = bytes.fromhex(encoded_text)
            except ValueError as exc:
                raise ValidatorChainSourceV2Error(
                    "expected extrinsic bytes are invalid"
                ) from exc
            if signed_extrinsic_hash_v2(raw) != hash_text:
                raise ValidatorChainSourceV2Error(
                    "expected extrinsic hash differs from bytes"
                )
            normalized[hash_text] = raw.hex()

        start = int(minimum_block)
        requested_end = int(maximum_block)
        if start < 0 or requested_end < start:
            raise ValidatorChainSourceV2Error("finalization scan range is invalid")
        job_id = "weight-finalization:%d" % int(epoch_id)
        purpose = "validator.weights.finalized.v2"
        attempts = []
        artifacts = []
        rpc_started = False

        def finalization_call(**kwargs: Any) -> Dict[str, Any]:
            nonlocal rpc_started
            if rpc_started:
                self._finalization_sleep(FINALIZATION_RPC_PACING_SECONDS)
            result = self._call(**kwargs)
            rpc_started = True
            return result

        finalized = finalization_call(
            method="chain_getFinalizedHead",
            params=[],
            request_id=1,
            job_id=job_id,
            purpose=purpose,
            logical_operation_id=job_id + ":finalized-head",
        )
        attempts.extend(finalized["attempts"])
        artifacts.extend(finalized["artifacts"])
        finalized_hash = normalize_raw_hash(finalized["result"], "finalized head")
        header_result = finalization_call(
            method="chain_getHeader",
            params=["0x" + finalized_hash],
            request_id=2,
            job_id=job_id,
            purpose=purpose,
            logical_operation_id=job_id + ":finalized-header",
        )
        attempts.extend(header_result["attempts"])
        artifacts.extend(header_result["artifacts"])
        finalized_header = parse_finalized_header(header_result["result"])
        end = min(requested_end, int(finalized_header["block"]))
        if end < start:
            raise ValidatorChainSourceV2Error(
                "authorized extrinsic range is not finalized"
            )
        if end - start + 1 > CHAIN_MAX_FINALIZATION_SCAN_BLOCKS:
            raise ValidatorChainSourceV2Error(
                "finalization scan exceeds measured policy"
            )

        request_id = 3
        for block_number in range(start, end + 1):
            block_hash_result = finalization_call(
                method="chain_getBlockHash",
                params=[block_number],
                request_id=request_id,
                job_id=job_id,
                purpose=purpose,
                logical_operation_id=job_id + ":block-hash:%d" % block_number,
            )
            request_id += 1
            attempts.extend(block_hash_result["attempts"])
            artifacts.extend(block_hash_result["artifacts"])
            block_hash = normalize_raw_hash(
                block_hash_result["result"], "finalized block hash"
            )
            block_result = finalization_call(
                method="chain_getBlock",
                params=["0x" + block_hash],
                request_id=request_id,
                job_id=job_id,
                purpose=purpose,
                logical_operation_id=job_id + ":block:%d" % block_number,
            )
            request_id += 1
            attempts.extend(block_result["attempts"])
            artifacts.extend(block_result["artifacts"])
            block = parse_finalized_block_extrinsics(
                block_result["result"], expected_block=block_number
            )
            for extrinsic_hex in block["extrinsics"]:
                extrinsic_hash = signed_extrinsic_hash_v2(
                    bytes.fromhex(extrinsic_hex)
                )
                if normalized.get(extrinsic_hash) == extrinsic_hex:
                    expected_commitment = expected_commitments.get(extrinsic_hash)
                    if not isinstance(expected_commitment, Mapping) or set(
                        expected_commitment
                    ) != {
                        "netuid",
                        "subnet_epoch_index",
                        "hotkey_public_key",
                        "commitment_hex",
                        "reveal_round",
                    }:
                        raise ValidatorChainSourceV2Error(
                            "expected commitment fields are invalid"
                        )
                    storage_result = finalization_call(
                        method="state_getStorage",
                        params=[
                            timelocked_weight_commits_storage_key(
                                netuid=int(expected_commitment["netuid"]),
                                subnet_epoch_index=int(
                                    expected_commitment[
                                        "subnet_epoch_index"
                                    ]
                                ),
                            ),
                            "0x" + block_hash,
                        ],
                        request_id=request_id,
                        job_id=job_id,
                        purpose=purpose,
                        logical_operation_id=(
                            job_id + ":timelocked-weight-state:%d" % block_number
                        ),
                    )
                    attempts.extend(storage_result["attempts"])
                    artifacts.extend(storage_result["artifacts"])
                    commits = decode_timelocked_weight_commits(
                        storage_result["result"]
                    )
                    matched_commits = [
                        item
                        for item in commits
                        if item["hotkey_public_key"]
                        == str(expected_commitment["hotkey_public_key"])
                        and item["commitment_hex"]
                        == str(expected_commitment["commitment_hex"])
                        and int(item["reveal_round"])
                        == int(expected_commitment["reveal_round"])
                        and int(item["submitted_at"]) <= block_number
                    ]
                    if len(matched_commits) != 1:
                        raise ValidatorChainSourceV2Error(
                            "finalized extrinsic did not produce the expected chain state"
                        )
                    return {
                        "extrinsic_hash": extrinsic_hash,
                        "extrinsic_hex": extrinsic_hex,
                        "finalized_block": block_number,
                        "finalized_block_hash": block_hash,
                        "finalized_head": finalized_header,
                        "state_transition_hash": sha256_json(matched_commits[0]),
                        "attempts": attempts,
                        "artifacts": artifacts,
                        "job_id": job_id,
                    }
        raise ValidatorChainSourceV2Error(
            "no authorized weight extrinsic was found in finalized blocks"
        )

    def _call(self, **kwargs: Any) -> Dict[str, Any]:
        return self._adapter_call(
            self._rpc_call,
            provider_id="bittensor_chain",
            destination_host=CHAIN_ENDPOINT_HOST,
            **kwargs,
        )

    def _archive_call(self, **kwargs: Any) -> Dict[str, Any]:
        if self._archive_rpc_call is None:
            raise ValidatorChainSourceV2Error(
                "archive RPC adapter is required for historical epoch state"
            )
        result = self._adapter_call(
            self._archive_rpc_call,
            provider_id="bittensor_archive",
            destination_host=CHAIN_ARCHIVE_ENDPOINT_HOST,
            **kwargs,
        )
        if not result["attempts"]:
            raise ValidatorChainSourceV2Error(
                "archive RPC adapter omitted measured transport evidence"
            )
        return result

    @staticmethod
    def _adapter_call(
        adapter: Callable[..., Mapping[str, Any]],
        *,
        provider_id: str,
        destination_host: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        result = adapter(**kwargs)
        if not isinstance(result, Mapping) or set(result) != {
            "result",
            "attempts",
            "artifacts",
        }:
            raise ValidatorChainSourceV2Error("chain RPC adapter result is invalid")
        for attempt in result["attempts"]:
            validate_transport_attempt(attempt)
            if (
                attempt.get("provider_id") != provider_id
                or attempt.get("destination_host") != destination_host
                or attempt.get("destination_port") != CHAIN_ENDPOINT_PORT
            ):
                raise ValidatorChainSourceV2Error(
                    "chain RPC adapter used the wrong measured endpoint"
                )
        return {
            "result": result["result"],
            "attempts": [dict(item) for item in result["attempts"]],
            "artifacts": [dict(item) for item in result["artifacts"]],
        }
