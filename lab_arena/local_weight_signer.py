"""Local-wallet Arena weight signing without a Nitro or vsock dependency.

The authorization and finalized-chain proof kernel remains
``ArenaWeightSigner``.  This module supplies only host-process boundaries:
the validator hotkey, the published Python drand SDK, and bounded HTTPS
JSON-RPC transports.  It deliberately does not reproduce the chain proof
implementation or add a second weight algorithm.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from Leadpoet.utils.subnet_epoch import (
    OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT,
    SubnetEpochCutover,
)
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_MAX_RPC_RESPONSE_BYTES,
    CHAIN_RPC_RETRY_BACKOFF_SECONDS,
    ChainSourceV2Error,
    json_rpc_request,
    parse_json_rpc_response,
    ss58_encode_account_id,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    select_chain_signing_profile,
    validate_chain_signing_profile,
)
from leadpoet_canonical.lab_arena_rewards import signing_key_from_document
from validator_tee.enclave.arena_weight_signer import ArenaWeightSigner
from validator_tee.enclave.chain_source_v2 import (
    ValidatorChainSourceV2,
    ValidatorChainSourceV2Error,
)


_PROFILE_ROOT = Path(__file__).resolve().parents[1] / "validator_tee" / "enclave"
_PUBLIC_PROFILES = {
    "finney": _PROFILE_ROOT / "chain_signing_profile_v2.json",
    "test": _PROFILE_ROOT / "chain_signing_profile_test_v2.json",
}
_TRANSIENT_RPC_HTTP_STATUSES = frozenset({429, 502, 503, 504})


class LocalWeightSignerError(RuntimeError):
    """Local signer configuration or an external host boundary is invalid."""


def _http_rpc_endpoint(endpoint: str) -> str:
    """Convert one credential-free WebSocket/HTTP origin to HTTP JSON-RPC."""

    value = str(endpoint or "").strip().rstrip("/")
    parsed = urlsplit(value)
    schemes = {"wss": "https", "ws": "http", "https": "https", "http": "http"}
    if (
        parsed.scheme not in schemes
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise LocalWeightSignerError("chain RPC endpoint must be a credential-free origin")
    if schemes[parsed.scheme] == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise LocalWeightSignerError("plaintext chain RPC is allowed only on loopback")
    return urlunsplit((schemes[parsed.scheme], parsed.netloc, "/", "", ""))


class HttpsJsonRpcTransport:
    """One stateless, bounded JSON-RPC transport for host chain reads."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: int,
        opener: Callable[..., Any] = urllib.request.urlopen,
        max_response_bytes: int = CHAIN_MAX_RPC_RESPONSE_BYTES,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not 1 <= int(timeout_seconds) <= 600:
            raise LocalWeightSignerError("chain RPC timeout is invalid")
        if isinstance(max_response_bytes, bool) or not 1 <= int(max_response_bytes) <= CHAIN_MAX_RPC_RESPONSE_BYTES:
            raise LocalWeightSignerError("chain RPC response limit is invalid")
        self.endpoint = _http_rpc_endpoint(endpoint)
        self.timeout_seconds = int(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self._opener = opener
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._closed = False
        self._lock = threading.Lock()

    def _require_open(self) -> None:
        with self._lock:
            if self._closed:
                raise ValidatorChainSourceV2Error("host chain RPC transport is closed")

    def _retry_after_seconds(self, headers: Any) -> Optional[float]:
        raw = headers.get("Retry-After") if headers is not None else None
        value = str(raw or "").strip()
        if not value:
            return None
        if value.isdecimal():
            if len(value) > 3:
                return float("inf")
            try:
                seconds = int(value)
            except (OverflowError, ValueError):
                return float("inf")
            return float(seconds) if seconds <= self.timeout_seconds else float("inf")
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                return None
            return max(0.0, retry_at.timestamp() - float(self._wall_clock()))
        except (OverflowError, TypeError, ValueError):
            return None

    def call(
        self,
        *,
        method: str,
        params: Sequence[Any],
        request_id: int,
        **_context: Any,
    ) -> Any:
        try:
            body = json_rpc_request(method, params, int(request_id))
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            deadline = float(self._clock()) + self.timeout_seconds
            last_error = None
            for attempt_number in range(len(CHAIN_RPC_RETRY_BACKOFF_SECONDS) + 1):
                self._require_open()
                remaining = deadline - float(self._clock())
                if remaining <= 0:
                    raise ValidatorChainSourceV2Error(
                        "host chain RPC retry deadline exhausted"
                    ) from last_error
                request_timeout = min(float(self.timeout_seconds), remaining)
                retry_after = None
                try:
                    with self._opener(request, timeout=request_timeout) as response:
                        status = int(response.getcode())
                        if status != 200:
                            if status not in _TRANSIENT_RPC_HTTP_STATUSES:
                                raise ValidatorChainSourceV2Error(
                                    "host chain RPC returned an HTTP error"
                                )
                            retry_after = self._retry_after_seconds(response.headers)
                            last_error = ValidatorChainSourceV2Error(
                                "host chain RPC returned a transient HTTP error"
                            )
                        else:
                            declared = response.headers.get("Content-Length")
                            if declared is not None and int(declared) > self.max_response_bytes:
                                raise ValidatorChainSourceV2Error(
                                    "host chain RPC response exceeds limit"
                                )
                            response_body = response.read(self.max_response_bytes + 1)
                    if status == 200:
                        if len(response_body) > self.max_response_bytes:
                            raise ValidatorChainSourceV2Error(
                                "host chain RPC response exceeds limit"
                            )
                        return parse_json_rpc_response(response_body, int(request_id))
                except urllib.error.HTTPError as exc:
                    try:
                        if int(exc.code) not in _TRANSIENT_RPC_HTTP_STATUSES:
                            raise ValidatorChainSourceV2Error(
                                "host chain RPC returned an HTTP error"
                            ) from exc
                        retry_after = self._retry_after_seconds(exc.headers)
                        last_error = exc
                    finally:
                        try:
                            exc.close()
                        except Exception as close_exc:
                            raise ValidatorChainSourceV2Error(
                                "host chain RPC HTTP response cleanup failed"
                            ) from close_exc
                if attempt_number == len(CHAIN_RPC_RETRY_BACKOFF_SECONDS):
                    raise ValidatorChainSourceV2Error(
                        "host chain RPC exhausted transient HTTP retries"
                    ) from last_error
                delay = max(
                    float(CHAIN_RPC_RETRY_BACKOFF_SECONDS[attempt_number]),
                    float(retry_after or 0.0),
                )
                if delay >= deadline - float(self._clock()):
                    raise ValidatorChainSourceV2Error(
                        "host chain RPC retry deadline exhausted"
                    ) from last_error
                self._sleep(delay)
        except ValidatorChainSourceV2Error:
            raise
        except (ChainSourceV2Error, urllib.error.URLError, OSError, ValueError) as exc:
            raise ValidatorChainSourceV2Error("host chain RPC request failed") from exc

    def close(self) -> None:
        with self._lock:
            self._closed = True


class HostValidatorChainSource(ValidatorChainSourceV2):
    """Reuse finalized chain proofs with ordinary host HTTPS transports."""

    def __init__(
        self,
        *,
        live_transport: Any,
        archive_transport: Any,
        cutover_manifest: Mapping[str, Any],
        finalization_sleep: Callable[[float], None],
    ) -> None:
        self._live_transport = live_transport
        self._host_archive_transport = archive_transport
        authority = {
            "mode": "stateful_v1",
            "cutover_manifest": dict(cutover_manifest),
        }
        super().__init__(
            rpc_call=live_transport.call,
            archive_rpc_call=archive_transport.call,
            finalization_sleep=finalization_sleep,
            epoch_authority_supplier=lambda: authority,
        )

    @staticmethod
    def _result(value: Any) -> Dict[str, Any]:
        # The shared proof code consumes these two collections only for the
        # old measured transport receipt graph.  Host validation has no such
        # receipt boundary, so the collections are intentionally empty.
        return {"result": value, "attempts": [], "artifacts": []}

    def _call(self, **kwargs: Any) -> Dict[str, Any]:
        return self._result(self._live_transport.call(**kwargs))

    def _archive_call(self, **kwargs: Any) -> Dict[str, Any]:
        return self._result(self._host_archive_transport.call(**kwargs))

    def close(self) -> None:
        transports = (self._live_transport, self._host_archive_transport)
        for index, transport in enumerate(transports):
            if index == 1 and transport is transports[0]:
                continue
            close = getattr(transport, "close", None)
            if callable(close):
                close()


class BittensorDrandBackend:
    """Adapter for bittensor-drand's stateful commit/reveal API."""

    def __init__(self, generate_commit: Optional[Callable[..., Any]] = None) -> None:
        if generate_commit is None:
            try:
                from bittensor_drand import get_encrypted_commit_v2
            except ImportError as exc:
                raise LocalWeightSignerError(
                    "bittensor-drand 2.x stateful API required; "
                    "install repository requirements"
                ) from exc
            generate_commit = get_encrypted_commit_v2
        self._generate_commit = generate_commit

    def generate_commit(
        self,
        *,
        uids: Sequence[int],
        weights_u16: Sequence[int],
        version_key: int,
        last_epoch_block: int,
        pending_epoch_at: int,
        subnet_epoch_index: int,
        tempo: int,
        blocks_since_last_step: int,
        current_block: int,
        subnet_reveal_period_epochs: int,
        block_time: float,
        hotkey_public_key: bytes,
    ) -> Tuple[bytes, int]:
        try:
            commitment, reveal_round = self._generate_commit(
                list(uids),
                list(weights_u16),
                int(version_key),
                int(last_epoch_block),
                int(pending_epoch_at),
                int(subnet_epoch_index),
                int(tempo),
                int(blocks_since_last_step),
                int(current_block),
                int(subnet_reveal_period_epochs),
                float(block_time),
                bytes(hotkey_public_key),
            )
        except Exception as exc:
            raise LocalWeightSignerError("bittensor-drand rejected the Arena commitment") from exc
        raw = bytes(commitment)
        round_number = int(reveal_round)
        if not raw or len(raw) > 1024 * 1024 or round_number <= 0:
            raise LocalWeightSignerError("bittensor-drand returned an invalid commitment")
        return raw, round_number


class LocalArenaWeightSigner:
    """In-process client compatible with ``ArenaSignerClient``."""

    def __init__(
        self,
        signer: ArenaWeightSigner,
        *,
        chain_source: HostValidatorChainSource,
        extrinsic_period: int,
        validator_hotkey: str,
        network: str,
        netuid: int,
        chain_profile: Mapping[str, Any],
    ) -> None:
        self._signer = signer
        self._chain_source = chain_source
        self.extrinsic_period = int(extrinsic_period)
        self.validator_hotkey = str(validator_hotkey)
        self.network = str(network)
        self.netuid = int(netuid)
        self._chain_profile = dict(chain_profile)
        self._closed = False
        self._lock = threading.Lock()

    def _require_open(self) -> None:
        with self._lock:
            if self._closed:
                raise LocalWeightSignerError("local Arena weight signer is closed")

    def prepare(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        self._require_open()
        return self._signer.prepare(request)

    def confirm(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        self._require_open()
        return self._signer.confirm(request)

    def recover(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        self._require_open()
        return self._signer.recover(request)

    def sign_chain_outcome(self, document: Mapping[str, Any]) -> Dict[str, Any]:
        self._require_open()
        return self._signer.sign_chain_outcome(document)

    def prepare_arena_weight_extrinsic_v1(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self.prepare(request)

    def confirm_arena_weight_extrinsic_v1(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self.confirm(request)

    def recover_arena_weight_extrinsic_v1(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self.recover(request)

    def sign_arena_chain_outcome_v1(self, document: Dict[str, Any]) -> Dict[str, Any]:
        return self.sign_chain_outcome(document)

    def readiness(self, epoch: int) -> Dict[str, Any]:
        """Prove the local signer can authenticate its current chain authority.

        This performs finalized reads only.  It does not generate a drand
        commitment, sign bytes, or submit an extrinsic.
        """

        self._require_open()
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise LocalWeightSignerError("Arena readiness epoch is invalid")
        snapshot = self._chain_source.read_finalized_snapshot(
            netuid=self.netuid, epoch_id=epoch
        )
        hotkeys = list(snapshot["metagraph"]["hotkeys"])
        matching_uids = [
            uid for uid, hotkey in enumerate(hotkeys) if hotkey == self.validator_hotkey
        ]
        if len(matching_uids) != 1:
            raise LocalWeightSignerError(
                "local validator hotkey has no unique finalized UID"
            )
        runtime = self._chain_source.read_chain_signing_runtime(
            runtime_block_hash=str(snapshot["finalized_block_hash"]),
            max_block_drift=int(self._chain_profile["max_snapshot_block_drift"]),
        )
        select_chain_signing_profile(
            self._chain_profile,
            runtime_version={
                "specVersion": runtime["spec_version"],
                "transactionVersion": runtime["transaction_version"],
            },
            genesis_hash=runtime["genesis_hash"],
        )
        authority = snapshot["epoch_authority"]
        if int(authority["settlement_epoch_id"]) != epoch:
            raise LocalWeightSignerError("finalized chain epoch differs from readiness epoch")
        return {
            "schema_version": "leadpoet.arena.local_weight_signer_readiness.v1",
            "network": self.network,
            "netuid": self.netuid,
            "epoch": epoch,
            "validator_hotkey": self.validator_hotkey,
            "validator_uid": matching_uids[0],
            "finalized_block": int(snapshot["header"]["block"]),
            "finalized_block_hash": str(snapshot["finalized_block_hash"]),
            "runtime_spec_version": int(runtime["spec_version"]),
            "transaction_version": int(runtime["transaction_version"]),
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._chain_source.close()


def load_public_chain_signing_profile(
    network: str,
    *,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load the repository's public runtime profile for one network."""

    selected = Path(path) if path is not None else _PUBLIC_PROFILES.get(str(network))
    if selected is None:
        raise LocalWeightSignerError("no public chain signing profile exists for this network")
    try:
        document = json.loads(selected.read_text(encoding="utf-8"))
        return validate_chain_signing_profile(document)
    except (OSError, ValueError, TypeError) as exc:
        raise LocalWeightSignerError("public chain signing profile is invalid") from exc


def build_local_weight_signer(
    keypair: Any,
    chain_config: Any,
    signing_key_document: Mapping[str, Any],
    expected_signing_key_hash: str,
    cutover: Any,
    *,
    burn_hotkey: str,
    chain_profile_path: Optional[Path] = None,
    archive_endpoint: Optional[str] = None,
    live_transport: Optional[Any] = None,
    archive_transport: Optional[Any] = None,
    drand_backend: Optional[Any] = None,
    finalization_sleep: Callable[[float], None] = time.sleep,
) -> LocalArenaWeightSigner:
    """Build one local-wallet signer from explicit public policy and chain inputs."""

    network = str(getattr(chain_config, "network_name", "") or "")
    endpoint = str(getattr(chain_config, "endpoint", "") or "")
    netuid = getattr(chain_config, "netuid", None)
    timeout = getattr(chain_config, "request_timeout_seconds", None)
    if not network or not endpoint or isinstance(netuid, bool) or not isinstance(netuid, int):
        raise LocalWeightSignerError("Arena chain configuration is invalid")
    if not isinstance(burn_hotkey, str) or not burn_hotkey.strip():
        raise LocalWeightSignerError("Arena burn hotkey policy is required")

    profile = load_public_chain_signing_profile(network, path=chain_profile_path)
    if profile["network"] != network:
        raise LocalWeightSignerError("public chain signing profile differs from chain network")
    # The profile authenticates chain and signing policy; the selected live
    # RPC is only a transport. Validate it even when a transport is injected.
    _http_rpc_endpoint(endpoint)
    selected_archive_endpoint = archive_endpoint or (
        OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT if network == "finney" else endpoint
    )
    # An operator-selected archive is also only a transport. Its origin must
    # remain safe even when tests or callers inject the transport itself.
    _http_rpc_endpoint(selected_archive_endpoint)

    if isinstance(cutover, SubnetEpochCutover):
        normalized_cutover = cutover.to_dict()
    elif isinstance(cutover, Mapping):
        try:
            normalized_cutover = SubnetEpochCutover.from_mapping(cutover).to_dict()
        except Exception as exc:
            raise LocalWeightSignerError("subnet epoch cutover is invalid") from exc
    else:
        raise LocalWeightSignerError("subnet epoch cutover is invalid")
    if (
        int(normalized_cutover["netuid"]) != netuid
        or normalized_cutover["network_genesis_hash"]
        != "0x" + profile["genesis_hash"]
    ):
        raise LocalWeightSignerError("subnet epoch cutover differs from chain configuration")

    try:
        arena_public_key_der = signing_key_from_document(
            signing_key_document, str(expected_signing_key_hash)
        )
        public_key = bytes(keypair.public_key)
        validator_hotkey = str(keypair.ss58_address)
    except Exception as exc:
        raise LocalWeightSignerError("local validator key or Arena signing key is invalid") from exc
    if len(public_key) != 32 or ss58_encode_account_id(public_key) != validator_hotkey:
        raise LocalWeightSignerError("local validator hotkey does not match its public key")
    if not callable(getattr(keypair, "sign", None)) or not callable(getattr(keypair, "verify", None)):
        raise LocalWeightSignerError("local validator hotkey cannot sign and verify")

    live = live_transport or HttpsJsonRpcTransport(
        endpoint, timeout_seconds=int(timeout)
    )
    if archive_transport is not None:
        archive = archive_transport
    else:
        archive = HttpsJsonRpcTransport(
            selected_archive_endpoint, timeout_seconds=int(timeout)
        )
    source = HostValidatorChainSource(
        live_transport=live,
        archive_transport=archive,
        cutover_manifest=normalized_cutover,
        finalization_sleep=finalization_sleep,
    )

    def sign(payload: bytes) -> bytes:
        return bytes(keypair.sign(bytes(payload)))

    def verify(signature: bytes, payload: bytes) -> bool:
        return bool(keypair.verify(bytes(payload), bytes(signature)))

    signer = ArenaWeightSigner(
        validator_hotkey=validator_hotkey,
        hotkey_public_key_hex=public_key.hex(),
        chain_profile=profile,
        chain_source=source,
        drand_backend=drand_backend or BittensorDrandBackend(),
        sign_sr25519=sign,
        arena_public_key_der=arena_public_key_der,
        verify_sr25519=verify,
        arena_public_key_hash=str(expected_signing_key_hash),
        network=network,
        netuid=netuid,
        burn_hotkey=burn_hotkey.strip(),
    )
    return LocalArenaWeightSigner(
        signer,
        chain_source=source,
        extrinsic_period=int(profile["extrinsic_period"]),
        validator_hotkey=validator_hotkey,
        network=network,
        netuid=netuid,
        chain_profile=profile,
    )


__all__ = [
    "BittensorDrandBackend",
    "HostValidatorChainSource",
    "HttpsJsonRpcTransport",
    "LocalArenaWeightSigner",
    "LocalWeightSignerError",
    "build_local_weight_signer",
    "load_public_chain_signing_profile",
]
