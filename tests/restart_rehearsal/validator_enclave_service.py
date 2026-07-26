#!/usr/bin/env python3.11
"""Persistent local process for the real validator enclave RPC handler.

Production keeps one Nitro enclave alive across all host RPC clients.  The
rehearsal's AF_VSOCK adapter therefore forwards requests to this Unix socket
service instead of importing the enclave handler independently in every host
process.  This preserves the real in-memory release, hotkey, authorization,
and signing state while replacing only the privileged AF_VSOCK/NSM boundary.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

from sitecustomize import (
    _external_event,
    _local_provider_transport,
    _local_verify_nitro_attestation_full,
)


SOCKET_PATH = Path(
    os.environ.get(
        "REHEARSAL_VALIDATOR_ENCLAVE_SOCKET",
        "/rehearsal-state/validator-enclave.sock",
    )
)
MAX_FRAME_BYTES = 16 * 1024 * 1024
MEASURED_DRAND_PATH = Path(
    "/app/validator_tee/enclave/libbittensor_drand_v2.so"
)
MEASURED_DRAND_HASH_PATH = Path(
    "/source/validator_tee/enclave/libbittensor_drand_v2.sha256"
)


class _CFunction:
    def __init__(self, callback: Any) -> None:
        self._callback = callback
        self.argtypes: Any = None
        self.restype: Any = None

    def __call__(self, *args: Any) -> Any:
        return self._callback(*args)


def _ctypes_integer(value: Any) -> int:
    return int(getattr(value, "value", value))


class _MeasuredDrandLibrary:
    """Strict local implementation of the physical x86_64 C-ABI boundary."""

    def __init__(
        self,
        *,
        library_path: str,
        expected_sha256: str,
        buffer_type: Any,
    ) -> None:
        path = Path(library_path)
        if path != MEASURED_DRAND_PATH:
            raise ValueError("measured drand C ABI path differs")
        observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError("measured drand C ABI hash differs")
        self._buffer_type = buffer_type
        self._buffers: dict[int, Any] = {}
        self.cr_generate_commit_v2 = _CFunction(self._generate)
        self.cr_free = _CFunction(self._free)
        self.cr_free_str = _CFunction(self._free_string)
        _external_event(
            "nitro_enclaves",
            "measured_drand_cabi",
            physical_role="validator_weights",
            canonical_path=str(path),
            library_sha256=observed_sha256,
            physical_runtime="amazonlinux2-x86_64-python3.7",
            local_adapter_runtime="developer-capped-python3.11",
        )

    def _generate(
        self,
        uids: Any,
        uids_len: Any,
        weights: Any,
        weights_len: Any,
        version_key: Any,
        last_epoch_block: Any,
        pending_epoch_at: Any,
        subnet_epoch_index: Any,
        tempo: Any,
        blocks_since_last_step: Any,
        current_block: Any,
        reveal_epochs: Any,
        block_time: Any,
        hotkey: Any,
        hotkey_len: Any,
        round_out: Any,
        error_out: Any,
    ) -> Any:
        uid_count = _ctypes_integer(uids_len)
        weight_count = _ctypes_integer(weights_len)
        public_key_length = _ctypes_integer(hotkey_len)
        if uid_count <= 0 or uid_count != weight_count:
            raise ValueError("measured drand C ABI vector lengths differ")
        if public_key_length != 32:
            raise ValueError("measured drand C ABI public key length differs")
        payload = {
            "uids": [int(uids[index]) for index in range(uid_count)],
            "weights_u16": [
                int(weights[index]) for index in range(weight_count)
            ],
            "version_key": _ctypes_integer(version_key),
            "last_epoch_block": _ctypes_integer(last_epoch_block),
            "pending_epoch_at": _ctypes_integer(pending_epoch_at),
            "subnet_epoch_index": _ctypes_integer(subnet_epoch_index),
            "tempo": _ctypes_integer(tempo),
            "blocks_since_last_step": _ctypes_integer(
                blocks_since_last_step
            ),
            "current_block": _ctypes_integer(current_block),
            "subnet_reveal_period_epochs": _ctypes_integer(reveal_epochs),
            "block_time": float(getattr(block_time, "value", block_time)),
            "hotkey_public_key": bytes(
                hotkey[index] for index in range(public_key_length)
            ).hex(),
        }
        commitment = hashlib.sha512(
            b"leadpoet-local-measured-drand-v2\x00"
            + json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).digest()
        storage = (ctypes.c_uint8 * len(commitment))(*commitment)
        pointer = ctypes.cast(storage, ctypes.POINTER(ctypes.c_uint8))
        address = int(ctypes.cast(pointer, ctypes.c_void_p).value or 0)
        if address <= 0 or address in self._buffers:
            raise ValueError("measured drand C ABI buffer identity differs")
        self._buffers[address] = storage
        reveal_round = payload["subnet_epoch_index"] + 1
        if reveal_round <= 0:
            raise ValueError("measured drand C ABI reveal round differs")
        ctypes.cast(
            round_out, ctypes.POINTER(ctypes.c_uint64)
        )[0] = reveal_round
        ctypes.cast(
            error_out, ctypes.POINTER(ctypes.c_char_p)
        )[0] = None
        _external_event(
            "nitro_enclaves",
            "measured_drand_commit",
            physical_role="validator_weights",
            request_sha256=hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
            destination_count=uid_count,
            commitment_sha256=hashlib.sha256(commitment).hexdigest(),
            reveal_round=reveal_round,
        )
        return self._buffer_type(
            pointer,
            len(commitment),
            len(commitment),
        )

    def _free(self, buffer: Any) -> None:
        address = int(
            ctypes.cast(buffer.ptr, ctypes.c_void_p).value or 0
        )
        if address not in self._buffers:
            raise ValueError("measured drand C ABI freed an unknown buffer")
        del self._buffers[address]

    @staticmethod
    def _free_string(pointer: Any) -> None:
        if bool(pointer):
            raise ValueError("measured drand C ABI returned an unknown error")


def _install_measured_drand_boundary() -> None:
    from validator_tee.enclave import drand_v2
    from validator_tee.enclave.hotkey_authority_v2 import (
        MEASURED_DRAND_LIBRARY_PATH,
    )

    expected_sha256 = MEASURED_DRAND_HASH_PATH.read_text(
        encoding="ascii"
    ).strip()
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or MEASURED_DRAND_LIBRARY_PATH != str(MEASURED_DRAND_PATH)
        or hashlib.sha256(MEASURED_DRAND_PATH.read_bytes()).hexdigest()
        != expected_sha256
    ):
        raise ValueError("measured drand release input differs")
    dockerfile = Path(
        "/source/validator_tee/Dockerfile.enclave"
    ).read_text(encoding="utf-8")
    copy_contract = (
        "COPY .validator-tee-artifacts/libbittensor_drand_v2.so "
        "/app/validator_tee/enclave/libbittensor_drand_v2.so"
    )
    if dockerfile.count(copy_contract) != 1:
        raise ValueError("measured drand Docker copy contract differs")

    defaults = drand_v2.CtypesDrandCommitBackendV2.__init__.__kwdefaults__
    if (
        not isinstance(defaults, dict)
        or defaults.get("library_loader") is not ctypes.CDLL
    ):
        raise ValueError("candidate drand loader contract differs")
    defaults["library_loader"] = lambda path: _MeasuredDrandLibrary(
        library_path=path,
        expected_sha256=expected_sha256,
        buffer_type=drand_v2._CRByteBuffer,
    )


def _recv_exact(connection: socket.socket, length: int) -> bytes:
    output = bytearray()
    while len(output) < length:
        chunk = connection.recv(min(64 * 1024, length - len(output)))
        if not chunk:
            break
        output.extend(chunk)
    if len(output) != length:
        raise ValueError("persistent validator enclave frame is incomplete")
    return bytes(output)


def _local_chain_http_post(transport: Any, body: bytes) -> dict[str, Any]:
    """Replace only the enclave's privileged vsock/TLS network boundary."""

    from leadpoet_canonical.chain_source_v2 import (
        CHAIN_MAX_RPC_RESPONSE_BYTES,
        CHAIN_RPC_TIMEOUT_MS,
    )

    result = _local_provider_transport(
        method="POST",
        url="https://%s/" % str(transport._destination_host),
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Content-Type": "application/json",
            "Host": str(transport._destination_host),
        },
        body=bytes(body),
        timeout_ms=CHAIN_RPC_TIMEOUT_MS,
        max_response_bytes=CHAIN_MAX_RPC_RESPONSE_BYTES,
    )
    if set(result) != {
        "http_status",
        "headers",
        "body",
        "tls_peer_chain_hash",
        "tls_protocol",
    }:
        raise ValueError("local validator chain TLS response fields differ")
    return {
        "status": int(result["http_status"]),
        "body": bytes(result["body"]),
        "tls_peer_chain_hash": str(result["tls_peer_chain_hash"]),
        "tls_protocol": str(result["tls_protocol"]),
    }


def _install_local_chain_tls_boundary() -> None:
    from validator_tee.enclave.chain_source_v2 import (
        EnclaveChainRpcTransportV2,
    )
    from validator_tee.enclave.hotkey_authority_v2 import (
        ValidatorHotkeyAuthorityV2,
    )

    EnclaveChainRpcTransportV2._http_post = _local_chain_http_post
    original_profile_for_runtime_block = (
        ValidatorHotkeyAuthorityV2._profile_for_runtime_block
    )

    def traced_profile_for_runtime_block(
        authority: Any, runtime_block_hash: str
    ) -> dict[str, Any]:
        try:
            return original_profile_for_runtime_block(
                authority, runtime_block_hash
            )
        except Exception as exc:
            cause = exc.__cause__
            _external_event(
                "nitro_enclaves",
                "chain_signing_runtime_rejected",
                physical_role="validator_weights",
                runtime_block_hash=runtime_block_hash,
                error_type=type(exc).__name__,
                error=str(exc),
                cause_type=(
                    type(cause).__name__ if cause is not None else None
                ),
                cause=(str(cause) if cause is not None else None),
            )
            raise

    ValidatorHotkeyAuthorityV2._profile_for_runtime_block = (
        traced_profile_for_runtime_block
    )


def _install_local_nsm() -> Any:
    import leadpoet_canonical.nitro as leadpoet_nitro
    from validator_tee.enclave import nsm_lib

    _install_measured_drand_boundary()
    _install_local_chain_tls_boundary()
    from validator_tee.enclave import tee_service

    leadpoet_nitro.verify_nitro_attestation_full = (
        _local_verify_nitro_attestation_full
    )

    def local_attestation_document(
        user_data: bytes | None = None,
        nonce: bytes | None = None,
        public_key: bytes | None = None,
    ) -> dict[str, Any]:
        document = json.dumps(
            {
                "schema_version": "leadpoet.local_validator_nitro.v1",
                "pcr0": hashlib.sha384(
                    b"leadpoet-local-pcr0:"
                    + os.environ["REHEARSAL_CANDIDATE_SHA"].encode("ascii")
                ).hexdigest(),
                "user_data_b64": base64.b64encode(user_data or b"").decode(
                    "ascii"
                ),
                "nonce_b64": base64.b64encode(nonce or b"").decode("ascii"),
                "public_key_b64": base64.b64encode(
                    public_key or b""
                ).decode("ascii"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        _external_event(
            "nitro_enclaves",
            "verify_attestation",
            physical_role="validator_weights",
            document_bytes=len(document),
        )
        return {"Attestation": {"document": document}}

    nsm_lib.get_attestation_document = local_attestation_document
    nsm_lib.get_pcr_measurements = lambda: {
        "PCR0": hashlib.sha384(
            b"leadpoet-local-pcr0:"
            + os.environ["REHEARSAL_CANDIDATE_SHA"].encode("ascii")
        ).hexdigest()
    }
    return tee_service


def _handle(tee_service: Any, body: bytes) -> bytes:
    decoded = tee_service._decode_rpc_payload(
        body,
        logical_limit=tee_service.MAX_RPC_REQUEST_BYTES,
    )
    request = json.loads(decoded)
    if not isinstance(request, dict):
        raise ValueError("validator enclave request must be an object")
    method = str(request.get("command") or "")
    try:
        response = tee_service.handle_request(request)
        status = "ok" if response.get("status") != "error" else "rejected"
    except Exception as exc:
        response = {"status": "error", "error": str(exc)}
        status = "rejected"
    encoded = json.dumps(
        response, sort_keys=True, separators=(",", ":")
    ).encode()
    frame = tee_service._encode_rpc_payload(
        encoded,
        logical_limit=tee_service.MAX_RPC_RESPONSE_BYTES,
        frame_limit=tee_service.MAX_RPC_RESPONSE_FRAME_BYTES,
    )
    _external_event(
        "nitro_enclaves",
        "enclave_rpc",
        physical_role="validator_weights",
        method=method,
        request_bytes=len(body),
        response_bytes=len(frame),
        status=status,
    )
    return frame


def main() -> int:
    if os.environ.get("REHEARSAL_COMPONENT") != "validator":
        raise SystemExit("persistent validator enclave requires validator scope")
    tee_service = _install_local_nsm()
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(SOCKET_PATH))
        SOCKET_PATH.chmod(0o600)
        server.listen(16)
        (SOCKET_PATH.parent / "validator-enclave.ready").write_text(
            "ready\n", encoding="ascii"
        )
        while True:
            connection, _address = server.accept()
            with connection:
                try:
                    prefix = _recv_exact(connection, 4)
                    size = int.from_bytes(prefix, "big")
                    if size < 2 or size > MAX_FRAME_BYTES:
                        raise ValueError(
                            "persistent validator enclave request size differs"
                        )
                    body = _recv_exact(connection, size)
                    response = _handle(tee_service, body)
                except Exception as exc:
                    response = json.dumps(
                        {"status": "error", "error": str(exc)},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                connection.sendall(
                    len(response).to_bytes(4, "big") + response
                )
    finally:
        server.close()
        SOCKET_PATH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
