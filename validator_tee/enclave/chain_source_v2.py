"""Finalized Bittensor state acquired with TLS inside the validator enclave."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import http.client
import json
import os
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
    CHAIN_ENDPOINT_HOST,
    CHAIN_ENDPOINT_PATH,
    CHAIN_ENDPOINT_PORT,
    CHAIN_FINALIZATION_EPOCH_BLOCKS,
    CHAIN_MAX_FINALIZATION_SCAN_BLOCKS,
    CHAIN_MAX_RPC_RESPONSE_BYTES,
    CHAIN_RPC_METHOD,
    CHAIN_RPC_RETRY_BACKOFF_SECONDS,
    CHAIN_RPC_TIMEOUT_MS,
    ChainSourceV2Error,
    chain_source_policy_hash,
    decode_timelocked_weight_commits,
    decode_selective_metagraph_result,
    encode_selective_metagraph_params,
    json_rpc_request,
    normalize_raw_hash,
    parse_finalized_block_extrinsics,
    parse_finalized_header,
    parse_json_rpc_response,
    timelocked_weight_commits_storage_key,
)
from leadpoet_canonical.hotkey_authority_v2 import signed_extrinsic_hash_v2


AF_VSOCK = 40
PARENT_CID = 3
CHAIN_RELAY_VSOCK_PORT = 5002
MAX_CONTROL_BYTES = 16 * 1024
DEFAULT_CA_BUNDLE = "/etc/pki/tls/certs/ca-bundle.crt"


class ValidatorChainSourceV2Error(RuntimeError):
    """The validator enclave could not authenticate a complete chain snapshot."""


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    ) -> None:
        self._socket_factory = socket_factory
        self._clock = clock
        self._sleep = sleep
        self._ca_bundle_path = str(ca_bundle_path)
        self._ssl_context_factory = ssl_context_factory or self._default_context

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
                    "host": CHAIN_ENDPOINT_HOST,
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
                server_hostname=CHAIN_ENDPOINT_HOST,
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
                    ("Host: %s\r\n" % CHAIN_ENDPOINT_HOST).encode("ascii"),
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
                    provider_id="bittensor_chain",
                    attempt_number=attempt_number,
                    method="POST",
                    destination_host=CHAIN_ENDPOINT_HOST,
                    destination_port=CHAIN_ENDPOINT_PORT,
                    path_hash=sha256_json({"path": CHAIN_ENDPOINT_PATH}),
                    nonsecret_headers_hash=sha256_json(
                        {
                            "accept": "application/json",
                            "content-type": "application/json",
                            "host": CHAIN_ENDPOINT_HOST,
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
                    provider_id="bittensor_chain",
                    attempt_number=attempt_number,
                    method="POST",
                    destination_host=CHAIN_ENDPOINT_HOST,
                    destination_port=CHAIN_ENDPOINT_PORT,
                    path_hash=sha256_json({"path": CHAIN_ENDPOINT_PATH}),
                    nonsecret_headers_hash=sha256_json(
                        {
                            "accept": "application/json",
                            "content-type": "application/json",
                            "host": CHAIN_ENDPOINT_HOST,
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
    ) -> None:
        self._rpc_call = rpc_call or EnclaveChainRpcTransportV2().call

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
        if int(header["block"]) // CHAIN_FINALIZATION_EPOCH_BLOCKS != int(epoch_id):
            raise ValidatorChainSourceV2Error(
                "finalized chain block differs from requested epoch"
            )

        metagraph_result = self._call(
            method="state_call",
            params=[
                CHAIN_RPC_METHOD,
                encode_selective_metagraph_params(netuid=int(netuid)),
                "0x" + finalized_hash,
            ],
            request_id=3,
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
            "attempts": all_attempts,
            "artifacts": all_artifacts,
            "jobs": {
                "chain_state": chain_job,
                "metagraph_state": metagraph_job,
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

        finalized = self._call(
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
        header_result = self._call(
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
            block_hash_result = self._call(
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
            block_result = self._call(
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
                        "mechid",
                        "hotkey_public_key",
                        "commitment_hex",
                        "reveal_round",
                    }:
                        raise ValidatorChainSourceV2Error(
                            "expected commitment fields are invalid"
                        )
                    storage_result = self._call(
                        method="state_getStorage",
                        params=[
                            timelocked_weight_commits_storage_key(
                                netuid=int(expected_commitment["netuid"]),
                                mechid=int(expected_commitment["mechid"]),
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
        result = self._rpc_call(**kwargs)
        if not isinstance(result, Mapping) or set(result) != {
            "result",
            "attempts",
            "artifacts",
        }:
            raise ValidatorChainSourceV2Error("chain RPC adapter result is invalid")
        for attempt in result["attempts"]:
            validate_transport_attempt(attempt)
        return {
            "result": result["result"],
            "attempts": [dict(item) for item in result["attempts"]],
            "artifacts": [dict(item) for item in result["artifacts"]],
        }
