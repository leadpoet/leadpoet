"""Install strict local equivalents for privileged rehearsal boundaries.

Candidate repository modules still execute normally.  This module replaces
only clients whose real implementation would contact production infrastructure
from the network-isolated rehearsal container.
"""

from __future__ import annotations

import asyncio
import base64
import builtins
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional
from urllib.parse import parse_qsl, urlparse, urlsplit, urlunsplit

try:
    from fixture_contract import (
        load_rehearsal_current_settlement_epoch_id,
        load_rehearsal_metagraph_account_ids,
        load_rehearsal_metagraph_hotkeys,
    )
except ModuleNotFoundError as exc:
    if exc.name != "fixture_contract":
        raise
    from tests.restart_rehearsal.fixture_contract import (
        load_rehearsal_current_settlement_epoch_id,
        load_rehearsal_metagraph_account_ids,
        load_rehearsal_metagraph_hotkeys,
    )


STATE_ROOT = Path(os.environ.get("REHEARSAL_STATE_ROOT", "/rehearsal-state"))
SOURCE_ROOT = Path(os.environ.get("REHEARSAL_SOURCE_ROOT", "/source"))
FROM_FIXTURE_SEED_ROOT = Path("/rehearsal-from-fixture-seed")
DURABLE_SCHEMA_SEED_ROOT = Path("/rehearsal-durable-schema-seed")
EVENT_PATH = STATE_ROOT / "events.jsonl"
CURRENT_BLOCK = 8_700_040
LAST_EPOCH_BLOCK = 8_700_000
TEMPO = 360
SUBNET_EPOCH_INDEX = 24_166
CUTOVER_BLOCK = 8_670_636
CUTOVER_BLOCK_HASH = (
    "0x25c2109c70fb3502a9c20fd3b04c1db3f0a18d968e73b42da9f1a47770a5e106"
)
CUTOVER_EPOCH_INDEX = 24_020
GENESIS_HASH = (
    "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
)
EXTERNAL_ARTIFACT_ROOT = Path("/opt/leadpoet/external-artifacts")
VALIDATOR_RUNTIME_LOCK_PATH = Path(
    "/opt/leadpoet/runtime-artifacts-v2.lock.json"
)
_GATEWAY_RUNTIME_OBJECTS: dict[str, dict[str, Any]] = {}
_GATEWAY_RUNTIME_OBJECTS_LOCK = threading.Lock()
_PRIVATE_MODEL_BUCKET = "leadpoet-private-model-artifacts-493765492819"
_PRIVATE_MODEL_PREFIX = "research-lab/sourcing-model/"
_PRIVATE_MODEL_POINTER_KEY = (
    _PRIVATE_MODEL_PREFIX + "branches/leadpoet-lab/current.json"
)
_PRIVATE_MODEL_SIGNING_KEY_ID = (
    "alias/leadpoet-research-lab-artifact-signing"
)
_PRIVATE_MODEL_OBJECTS: dict[tuple[str, str], bytes] = {}
_PRIVATE_MODEL_OBJECTS_LOCK = threading.Lock()
_REAL_SUBTENSOR_CLASS: Any = None
_ORIGINAL_SOCKET = socket.socket
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_RESTART_EPOCH_TRANSIENT_HEAD_CALLS = 0
_BLOCK_NUMBERS_BY_HASH: dict[str, int] = {}


def _block_hash(block: int) -> str:
    normalized_block = int(block)
    value = "0x" + hashlib.sha256(
        f"leadpoet-local-block:{normalized_block}".encode()
    ).hexdigest()
    _BLOCK_NUMBERS_BY_HASH[value] = normalized_block
    return value


def _block_number(block_hash: str) -> int:
    value = str(block_hash)
    if value == CUTOVER_BLOCK_HASH:
        return CUTOVER_BLOCK
    if value in _BLOCK_NUMBERS_BY_HASH:
        return _BLOCK_NUMBERS_BY_HASH[value]
    raise ValueError("local chain block hash is unknown")


def _subnet_epoch_index_at(block: int) -> int:
    normalized_block = int(block)
    if normalized_block < CUTOVER_BLOCK:
        return CUTOVER_EPOCH_INDEX - 1
    if normalized_block >= LAST_EPOCH_BLOCK:
        return SUBNET_EPOCH_INDEX
    span = LAST_EPOCH_BLOCK - CUTOVER_BLOCK
    transitions = SUBNET_EPOCH_INDEX - CUTOVER_EPOCH_INDEX
    if span <= 0 or transitions <= 0:
        raise ValueError("local chain epoch fixture is invalid")
    return CUTOVER_EPOCH_INDEX + (
        ((normalized_block - CUTOVER_BLOCK) * transitions) // span
    )


def _current_settlement_epoch_id() -> int:
    return load_rehearsal_current_settlement_epoch_id(
        SOURCE_ROOT
    )


def _external_event(
    boundary: str,
    operation: str,
    **details: Any,
) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    row = {
        "at_ns": time.time_ns(),
        "kind": "local-chain-boundary",
        "status": "ok",
        "boundary": boundary,
        "operation": operation,
        "implementation": "external_boundary",
        "fixture_authenticity": "production_shaped_sanitized",
        "reject_unknown": True,
        **details,
    }
    descriptor = os.open(
        EVENT_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(
            descriptor,
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
    finally:
        os.close(descriptor)


def _event(operation: str, **details: Any) -> None:
    _external_event("stateful_subnet_chain", operation, **details)


class _ScaleValue:
    def __init__(self, value: int):
        self.value = value


class _LocalEra:
    def encode(self, era: Mapping[str, Any]) -> None:
        if set(era) != {"period", "current"}:
            raise ValueError("local chain signing era differs")
        self.era = dict(era)

    def birth(self, current: int) -> int:
        period = int(self.era["period"])
        if period != 8 or int(current) != int(self.era["current"]):
            raise ValueError("local chain signing era values differ")
        return int(current) - (int(current) % period)


class _LocalRuntimeConfig:
    def create_scale_object(self, name: str) -> _LocalEra:
        if name != "Era":
            raise ValueError("local runtime requested an unknown SCALE object")
        return _LocalEra()


class _LocalCall:
    def __init__(self, *, call_args: Mapping[str, Any], call_data: bytes):
        self.value = {
            "call_module": "SubtensorModule",
            "call_function": "serve_axon",
            "call_args": dict(call_args),
        }
        self.data = SimpleNamespace(data=bytes(call_data))


def _local_chain_signing_profile() -> dict[str, Any]:
    from leadpoet_canonical.hotkey_authority_v2 import (
        select_chain_signing_profile,
    )

    measured = json.loads(
        (
            SOURCE_ROOT
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).read_text(encoding="utf-8")
    )
    return select_chain_signing_profile(
        measured,
        runtime_version={
            "specVersion": 440,
            "transactionVersion": 1,
        },
        genesis_hash=GENESIS_HASH.removeprefix("0x"),
    )


class _LocalSubstrate:
    url = "ws://127.0.0.1:9944"
    runtime_config = _LocalRuntimeConfig()
    _SUBTENSOR_MODULE_INDEX = 7
    _CALL_METADATA = {
        "commit_timelocked_mechanism_weights": {
            "index": 118,
            "fields": [
                {"name": "netuid", "typeName": "NetUid"},
                {"name": "mecid", "typeName": "MechId"},
                {
                    "name": "commit",
                    "typeName": (
                        "BoundedVec<u8, ConstU32<MAX_CRV3_COMMIT_SIZE_BYTES>>"
                    ),
                },
                {"name": "reveal_round", "typeName": "u64"},
                {"name": "commit_reveal_version", "typeName": "u16"},
            ],
        },
        "serve_axon": {
            "index": 4,
            "fields": [
                {"name": "netuid", "typeName": "NetUid"},
                {"name": "version", "typeName": "u32"},
                {"name": "ip", "typeName": "u128"},
                {"name": "port", "typeName": "u16"},
                {"name": "ip_type", "typeName": "u8"},
                {"name": "protocol", "typeName": "u8"},
                {"name": "placeholder1", "typeName": "u8"},
                {"name": "placeholder2", "typeName": "u8"},
            ],
        },
    }

    def init_runtime(self, block_hash: Optional[str] = None) -> None:
        if block_hash != _block_hash(CURRENT_BLOCK):
            raise ValueError("local runtime initialization hash differs")
        _event("runtime_version", method="init_runtime", block_hash=block_hash)

    def get_account_nonce(self, address: str) -> int:
        if not str(address):
            raise ValueError("local account nonce address is missing")
        self._active_signer_address = str(address)
        _event(
            "epoch_snapshot",
            method="get_account_nonce",
            address_hash=_sha256(str(address)),
        )
        return 7

    def compose_call(
        self,
        *,
        call_module: str,
        call_function: str,
        call_params: Mapping[str, Any],
        **kwargs: Any,
    ) -> _LocalCall:
        if (
            call_module != "SubtensorModule"
            or call_function != "serve_axon"
            or kwargs
        ):
            raise ValueError("local chain compose_call contract differs")
        from leadpoet_canonical.hotkey_authority_v2 import (
            encode_serve_axon_call,
        )

        profile = _local_chain_signing_profile()
        call_data = encode_serve_axon_call(
            profile=profile,
            netuid=int(call_params["netuid"]),
            version=int(call_params["version"]),
            ip=int(call_params["ip"]),
            port=int(call_params["port"]),
            ip_type=int(call_params["ip_type"]),
            protocol=int(call_params["protocol"]),
            placeholder1=int(call_params["placeholder1"]),
            placeholder2=int(call_params["placeholder2"]),
        )
        _event(
            "submit_extrinsic",
            method="compose_call",
            call_module=call_module,
            call_function=call_function,
        )
        return _LocalCall(call_args=call_params, call_data=call_data)

    def generate_signature_payload(self, **kwargs: Any) -> Any:
        call = kwargs.get("call")
        era = kwargs.get("era")
        nonce = kwargs.get("nonce")
        if (
            not isinstance(call, _LocalCall)
            or not isinstance(era, Mapping)
            or set(kwargs)
            != {"call", "era", "nonce", "tip", "tip_asset_id"}
            or kwargs.get("tip") != 0
            or kwargs.get("tip_asset_id") is not None
        ):
            raise ValueError("local signature payload contract differs")
        from leadpoet_canonical.hotkey_authority_v2 import (
            build_serve_axon_extrinsic_authorization_v2,
        )

        profile = _local_chain_signing_profile()
        args = call.value["call_args"]
        validator_hotkey = getattr(self, "_active_signer_address", "")
        if not validator_hotkey:
            raise ValueError(
                "local signature payload has no SDK signer identity"
            )
        era_object = _LocalEra()
        era_object.encode(era)
        birth_block = era_object.birth(int(era["current"]))
        authorization = build_serve_axon_extrinsic_authorization_v2(
            profile=profile,
            validator_hotkey=validator_hotkey,
            hotkey_public_key_hex="00" * 32,
            netuid=int(args["netuid"]),
            version=int(args["version"]),
            ip=int(args["ip"]),
            port=int(args["port"]),
            ip_type=int(args["ip_type"]),
            protocol=int(args["protocol"]),
            placeholder1=int(args["placeholder1"]),
            placeholder2=int(args["placeholder2"]),
            era_current=int(era["current"]),
            nonce=int(nonce),
            block_hash=str(
                self.get_block_hash(block_id=birth_block)
            ).removeprefix("0x"),
        )
        signed = bytes.fromhex(authorization["signed_message_hex"])
        _event(
            "submit_extrinsic",
            method="generate_signature_payload",
            payload_hash="sha256:" + hashlib.sha256(signed).hexdigest(),
        )
        return SimpleNamespace(data=signed)

    def create_signed_extrinsic(self, **kwargs: Any) -> Any:
        call = kwargs.get("call")
        signature = kwargs.get("signature")
        if (
            not isinstance(call, _LocalCall)
            or not isinstance(signature, (bytes, bytearray))
            or len(signature) != 64
        ):
            raise ValueError("local signed extrinsic contract differs")
        data = (
            b"leadpoet-local-signed-extrinsic-v1:"
            + bytes(call.data.data)
            + bytes(signature)
        )
        _event(
            "submit_extrinsic",
            method="create_signed_extrinsic",
            extrinsic_hash=(
                "0x" + hashlib.blake2b(data, digest_size=32).hexdigest()
            ),
        )
        return SimpleNamespace(data=SimpleNamespace(data=data))

    def submit_extrinsic(
        self,
        *,
        extrinsic: Any,
        wait_for_inclusion: bool,
        wait_for_finalization: bool,
    ) -> Any:
        data = bytes(extrinsic.data.data)
        if not data.startswith(b"leadpoet-local-signed-extrinsic-v1:"):
            raise ValueError("local submitted extrinsic bytes differ")
        _event(
            "finalization",
            method="submit_extrinsic",
            wait_for_inclusion=bool(wait_for_inclusion),
            wait_for_finalization=bool(wait_for_finalization),
            extrinsic_hash=(
                "0x" + hashlib.blake2b(data, digest_size=32).hexdigest()
            ),
        )
        return SimpleNamespace(
            is_success=True,
            total_fee_amount=0,
            error_message=None,
        )

    def get_block_hash(
        self,
        block: Optional[int] = None,
        *,
        block_id: Optional[int] = None,
    ) -> str:
        if block is not None and block_id is not None:
            raise ValueError("local chain block hash request is ambiguous")
        normalized = int(block if block is not None else block_id)
        if normalized == 0:
            result = GENESIS_HASH
        elif normalized == CUTOVER_BLOCK:
            result = CUTOVER_BLOCK_HASH
        else:
            result = _block_hash(normalized)
        _event("epoch_snapshot", method="get_block_hash", block=normalized)
        return result

    def get_chain_head(self) -> str:
        global _RESTART_EPOCH_TRANSIENT_HEAD_CALLS
        configured_failures = int(
            os.environ.get(
                "LEADPOET_REHEARSAL_RESTART_EPOCH_TRANSIENT_FAILURES",
                "0",
            )
        )
        _RESTART_EPOCH_TRANSIENT_HEAD_CALLS += 1
        injected_failure = (
            _RESTART_EPOCH_TRANSIENT_HEAD_CALLS <= configured_failures
        )
        _event(
            "epoch_snapshot",
            method="get_chain_head",
            injected_failure=injected_failure,
            attempt=_RESTART_EPOCH_TRANSIENT_HEAD_CALLS,
        )
        if injected_failure:
            return "malformed-transient-head"
        return _block_hash(CURRENT_BLOCK)

    def get_chain_finalised_head(self) -> str:
        _event("finalized_head", method="get_chain_finalised_head")
        return _block_hash(CURRENT_BLOCK)

    def get_block_number(self, block_hash: str) -> int:
        if block_hash == GENESIS_HASH:
            return 0
        if block_hash == CUTOVER_BLOCK_HASH:
            return CUTOVER_BLOCK
        if block_hash == _block_hash(CUTOVER_BLOCK - 1):
            return CUTOVER_BLOCK - 1
        for block in range(CURRENT_BLOCK - TEMPO, CURRENT_BLOCK + 1):
            if _block_hash(block) == block_hash:
                _event(
                    "epoch_snapshot",
                    method="get_block_number",
                    block=block,
                )
                return block
        raise ValueError("local chain received an unknown block hash")

    def get_metadata_module(
        self,
        module_name: str,
        *,
        block_hash: str,
    ) -> Any:
        if (
            module_name != "SubtensorModule"
            or block_hash != _block_hash(CURRENT_BLOCK)
        ):
            raise ValueError("local chain metadata module contract differs")
        _event(
            "runtime_version",
            method="get_metadata_module",
            module=module_name,
            block_hash=block_hash,
        )
        return SimpleNamespace(
            value={
                "name": module_name,
                "index": self._SUBTENSOR_MODULE_INDEX,
            }
        )

    def get_metadata_call_function(
        self,
        module_name: str,
        function_name: str,
        *,
        block_hash: str,
    ) -> Any:
        if (
            module_name != "SubtensorModule"
            or block_hash != _block_hash(CURRENT_BLOCK)
            or function_name not in self._CALL_METADATA
        ):
            raise ValueError("local chain metadata call contract differs")
        _event(
            "runtime_version",
            method="get_metadata_call_function",
            module=module_name,
            function=function_name,
            block_hash=block_hash,
        )
        return SimpleNamespace(
            value=dict(self._CALL_METADATA[function_name])
        )

    def rpc_request(self, method: str, params: list[Any]) -> dict[str, Any]:
        finalized_hash = _block_hash(CURRENT_BLOCK)
        if method == "chain_getFinalizedHead" and params == []:
            result: Any = finalized_hash
        elif (
            method == "state_getRuntimeVersion"
            and params == [finalized_hash]
        ):
            result = {
                "specName": "node-subtensor",
                "implName": "node-subtensor",
                "authoringVersion": 1,
                "specVersion": 440,
                "implVersion": 0,
                "apis": [],
                "transactionVersion": 1,
                "stateVersion": 1,
            }
        elif method == "chain_getHeader" and params == [finalized_hash]:
            result = {
                "parentHash": _block_hash(CURRENT_BLOCK - 1),
                "number": hex(CURRENT_BLOCK),
                "stateRoot": _block_hash(CURRENT_BLOCK + 1),
                "extrinsicsRoot": _block_hash(CURRENT_BLOCK + 2),
                "digest": {"logs": []},
            }
        elif method == "chain_getBlockHash" and params == [0]:
            result = GENESIS_HASH
        else:
            raise ValueError(
                f"local chain received an unknown RPC: {method} {params!r}"
            )
        boundary_operation = {
            "state_getRuntimeVersion": "runtime_version",
            "chain_getFinalizedHead": "finalized_head",
            "chain_getHeader": "finalized_head",
            "chain_getBlockHash": "epoch_snapshot",
        }[method]
        _event(boundary_operation, method=method, params=params)
        return {"jsonrpc": "2.0", "id": 1, "result": result}

    async def subscribe_block_headers(
        self,
        *,
        subscription_handler: Callable[[dict[str, Any]], Any],
        finalized_only: bool,
    ) -> None:
        if not callable(subscription_handler) or finalized_only is not True:
            raise ValueError("local block subscription contract differs")
        _event(
            "finalized_head",
            method="subscribe_block_headers",
            finalized_only=True,
        )
        while True:
            should_stop = await subscription_handler(
                {"header": {"number": CURRENT_BLOCK}}
            )
            if should_stop is True:
                return
            await asyncio.sleep(3600)

    def query(
        self,
        *,
        module: str,
        storage_function: str,
        params: list[Any],
        block_hash: str,
    ) -> _ScaleValue:
        exact_blocks = {
            _block_hash(CURRENT_BLOCK): CURRENT_BLOCK,
            CUTOVER_BLOCK_HASH: CUTOVER_BLOCK,
            _block_hash(CUTOVER_BLOCK - 1): CUTOVER_BLOCK - 1,
        }
        if block_hash not in exact_blocks:
            raise ValueError(
                "local chain query was not pinned to a supported exact hash"
            )
        exact_block = exact_blocks[block_hash]
        if module == "Timestamp" and storage_function == "Now" and params == []:
            value = int(
                datetime(2026, 7, 25, tzinfo=timezone.utc).timestamp() * 1000
            )
        elif module == "SubtensorModule" and params == [71]:
            if exact_block == CURRENT_BLOCK:
                values = {
                    "Tempo": TEMPO,
                    "LastEpochBlock": LAST_EPOCH_BLOCK,
                    "PendingEpochAt": LAST_EPOCH_BLOCK + TEMPO,
                    "SubnetEpochIndex": SUBNET_EPOCH_INDEX,
                    "BlocksSinceLastStep": CURRENT_BLOCK - LAST_EPOCH_BLOCK,
                }
            elif exact_block == CUTOVER_BLOCK:
                values = {
                    "Tempo": TEMPO,
                    "LastEpochBlock": CUTOVER_BLOCK,
                    "PendingEpochAt": CUTOVER_BLOCK + TEMPO,
                    "SubnetEpochIndex": CUTOVER_EPOCH_INDEX,
                    "BlocksSinceLastStep": 0,
                }
            else:
                values = {
                    "Tempo": TEMPO,
                    "LastEpochBlock": CUTOVER_BLOCK - TEMPO,
                    "PendingEpochAt": CUTOVER_BLOCK,
                    "SubnetEpochIndex": CUTOVER_EPOCH_INDEX - 1,
                    "BlocksSinceLastStep": TEMPO - 1,
                }
            if storage_function not in values:
                raise ValueError("local chain received an unknown storage field")
            value = values[storage_function]
        else:
            raise ValueError("local chain received an unknown storage query")
        _event(
            "epoch_snapshot",
            method="query",
            module=module,
            storage_function=storage_function,
        )
        return _ScaleValue(value)


class _LocalSubtensor:
    def __init__(self, *args: Any, network: str = "", **kwargs: Any):
        del args
        config = kwargs.pop("config", None)
        if kwargs:
            raise ValueError(
                f"local Subtensor received unknown options: {sorted(kwargs)}"
            )
        configured_network = str(
            getattr(getattr(config, "subtensor", None), "network", "") or ""
        )
        self.network = str(network or configured_network)
        self.chain_endpoint = (
            self.network
            if self.network.startswith(("ws://", "wss://"))
            else "ws://127.0.0.1:9944"
        )
        self.substrate = _LocalSubstrate()
        self.substrate.url = self.chain_endpoint
        _event("epoch_snapshot", method="connect", network=self.network)

    @property
    def block(self) -> int:
        return CURRENT_BLOCK

    def get_current_block(self) -> int:
        _event("epoch_snapshot", method="get_current_block")
        return CURRENT_BLOCK

    def get_uid_for_hotkey_on_subnet(
        self,
        *,
        hotkey_ss58: str,
        netuid: int,
    ) -> Optional[int]:
        if int(netuid) != 71:
            raise ValueError("local metagraph received the wrong netuid")
        _event(
            "epoch_snapshot",
            method="get_uid_for_hotkey_on_subnet",
            hotkey_hash=_sha256(str(hotkey_ss58)),
        )
        return 0

    def serve_axon(self, *, netuid: int, axon: Any, period: int = 0) -> Any:
        if int(netuid) != 71 or int(period) not in {0, 8} or axon is None:
            raise ValueError("local serve_axon contract differs")
        if _REAL_SUBTENSOR_CLASS is None:
            raise RuntimeError("real Bittensor Subtensor class is unavailable")
        _event(
            "submit_extrinsic",
            method="serve_axon_sdk",
            period=int(period),
        )
        return _REAL_SUBTENSOR_CLASS.serve_axon(
            self,
            netuid=netuid,
            axon=axon,
            period=period,
        )

    def get_neuron_for_pubkey_and_subnet(
        self,
        hotkey_ss58: str,
        *,
        netuid: int,
    ) -> Any:
        if int(netuid) != 71 or not str(hotkey_ss58):
            raise ValueError("local neuron lookup contract differs")
        _event(
            "epoch_snapshot",
            method="get_neuron_for_pubkey_and_subnet",
            hotkey_hash=_sha256(str(hotkey_ss58)),
        )
        return SimpleNamespace(is_null=True)

    def compose_call(
        self,
        *,
        call_module: str,
        call_function: str,
        call_params: Mapping[str, Any],
        block: Optional[int] = None,
    ) -> _LocalCall:
        if block is not None:
            raise ValueError("local compose_call block override is forbidden")
        return self.substrate.compose_call(
            call_module=call_module,
            call_function=call_function,
            call_params=call_params,
        )

    def sign_and_send_extrinsic(self, **kwargs: Any) -> Any:
        if _REAL_SUBTENSOR_CLASS is None:
            raise RuntimeError("real Bittensor Subtensor class is unavailable")
        return _REAL_SUBTENSOR_CLASS.sign_and_send_extrinsic(self, **kwargs)

    def close(self) -> None:
        _event("epoch_snapshot", method="close")


class _LocalAsyncSubtensor:
    def __init__(self, *args: Any, network: str = "", **kwargs: Any):
        del args, kwargs
        self.network = network
        self.chain_endpoint = "ws://127.0.0.1:9944"
        self.substrate = _LocalSubstrate()

    async def __aenter__(self) -> "_LocalAsyncSubtensor":
        _event("epoch_snapshot", method="async_connect", network=self.network)
        return self

    async def __aexit__(self, *args: Any) -> None:
        del args
        _event("epoch_snapshot", method="async_close")

    async def close(self) -> None:
        _event("epoch_snapshot", method="async_close")


class _LocalAxonInfo:
    def __init__(self, uid: int):
        self.is_serving = True
        self.ip = f"192.0.2.{uid + 1}"
        self.port = 8091 + uid


def _local_metagraph_hotkeys() -> tuple[str, ...]:
    fixture_hotkeys = load_rehearsal_metagraph_hotkeys(SOURCE_ROOT)
    return (
        "5CUxhqZ2ewLA61PtdKYzdnLXq1jyFxsvjMg8mRsim4Ni8T3p",
        "5DLocalFulfillment111111111111111111111111111111",
        *fixture_hotkeys[2:],
    )


class _LocalMetagraph:
    def __init__(self, *, netuid: int, subtensor: Any):
        import numpy as np

        if int(netuid) != 71 or not isinstance(subtensor, _LocalSubtensor):
            raise ValueError("local metagraph contract differs")
        self.netuid = int(netuid)
        self.hotkeys = list(_local_metagraph_hotkeys())
        self.n = len(self.hotkeys)
        self.uids = np.arange(self.n, dtype=np.int64)
        self.validator_trust = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self.S = np.asarray([1000.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.active = np.asarray([True] * self.n, dtype=bool)
        self.validator_permit = np.asarray(
            [True, False, False, False], dtype=bool
        )
        self.axons = [_LocalAxonInfo(uid) for uid in range(self.n)]
        _event("epoch_snapshot", method="metagraph", netuid=self.netuid)

    def sync(self, *, subtensor: Any) -> None:
        if not isinstance(subtensor, _LocalSubtensor):
            raise ValueError("local metagraph sync source differs")
        _event("epoch_snapshot", method="metagraph_sync", netuid=self.netuid)


def _sha256(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _gateway_release_input() -> dict[str, Any]:
    value = json.loads(
        (STATE_ROOT / "release-build-input.json").read_text(encoding="utf-8")
    )
    if value.get("commit_sha") != os.environ.get(
        "REHEARSAL_CANDIDATE_SHA"
    ):
        raise ValueError("local enclave release input commit differs")
    roles = value.get("gateway_roles")
    if not isinstance(roles, dict):
        raise ValueError("local enclave release roles are unavailable")
    return value


def _gateway_enclave_state(
    mutation: Optional[Callable[[dict[str, Any]], Any]] = None,
) -> tuple[dict[str, Any], Any]:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = STATE_ROOT / "gateway-enclave-state.lock"
    state_path = STATE_ROOT / "gateway-enclave-state.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            state = {"roles": {}, "provisioned_slots": []}
        result = mutation(state) if mutation is not None else None
        if mutation is not None:
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(state, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, state_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return state, result


def _gateway_role_for_cid(cid: int) -> str:
    from gateway.tee.topology import ROLE_SPECS

    matches = [
        role
        for role, spec in ROLE_SPECS.items()
        if int(spec["cid"]) == int(cid)
    ]
    if len(matches) != 1:
        raise ValueError("local enclave RPC received an unknown CID")
    return matches[0]


def _gateway_enclave_socket_path(role: str) -> Path:
    from gateway.tee.topology import ROLE_SPECS

    if role not in ROLE_SPECS:
        raise ValueError("local gateway enclave role is unknown")
    root = Path(
        os.environ.get(
            "REHEARSAL_GATEWAY_ENCLAVE_SOCKET_ROOT",
            "/rehearsal-state",
        )
    )
    return root / ("gateway-enclave-%s.sock" % role)


def _call_persistent_gateway_enclave(
    role: str,
    method: str,
    params: Mapping[str, Any],
) -> Any:
    """Cross one strict process boundary into a persistent role enclave."""

    if method not in {
        "rehearsal_inter_enclave_artifact_call",
        "rehearsal_inter_enclave_provider_execute",
        "rehearsal_inter_enclave_provider_probe_resolve",
    }:
        raise ValueError("persistent inter-enclave method is not authorized")
    if not isinstance(params, Mapping):
        raise ValueError("persistent inter-enclave params are invalid")
    socket_path = _gateway_enclave_socket_path(role)
    if not socket_path.is_socket():
        raise ValueError("persistent target enclave service is unavailable")
    body = json.dumps(
        {"method": method, "params": dict(params)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    connection = _ORIGINAL_SOCKET(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.settimeout(120)
        connection.connect(str(socket_path))
        connection.sendall(len(body).to_bytes(4, "big") + body)

        prefix = bytearray()
        while len(prefix) < 4:
            chunk = connection.recv(4 - len(prefix))
            if not chunk:
                break
            prefix.extend(chunk)
        if len(prefix) != 4:
            raise ValueError("persistent inter-enclave response is incomplete")
        response_size = int.from_bytes(prefix, "big")
        if response_size < 2 or response_size > 128 * 1024 * 1024:
            raise ValueError("persistent inter-enclave response size differs")
        frame = bytearray()
        while len(frame) < response_size:
            chunk = connection.recv(
                min(64 * 1024, response_size - len(frame))
            )
            if not chunk:
                break
            frame.extend(chunk)
        if len(frame) != response_size:
            raise ValueError(
                "persistent inter-enclave response body is incomplete"
            )
    finally:
        connection.close()

    outer = json.loads(frame)
    if (
        not isinstance(outer, dict)
        or set(outer) != {"diagnostic", "response"}
        or not isinstance(outer["diagnostic"], dict)
        or not isinstance(outer["response"], dict)
    ):
        raise ValueError("persistent inter-enclave response envelope differs")
    response = outer["response"]
    diagnostic = outer["diagnostic"]
    if set(diagnostic) != {"error_hash", "error_type", "status"}:
        raise ValueError("persistent inter-enclave diagnostic fields differ")
    if response.get("status") == "error":
        if (
            set(response) != {"status", "error"}
            or diagnostic.get("status") != "rejected"
        ):
            raise ValueError("persistent inter-enclave error envelope differs")
        raise ValueError(
            "persistent coordinator RPC failed: %s" % response["error"]
        )
    if (
        set(response) != {"status", "result"}
        or response.get("status") != "success"
        or diagnostic
        != {"status": "ok", "error_type": "", "error_hash": ""}
    ):
        raise ValueError("persistent inter-enclave success envelope differs")
    return response["result"]


class _PersistentInterEnclaveArtifactClient:
    """Route production artifact chunks into the persistent coordinator role."""

    def __init__(self, *, peer_role: str) -> None:
        self._peer_role = str(peer_role)

    def call(
        self,
        *,
        target_physical_role: str,
        method: str,
        params: Mapping[str, Any],
        channel_id: str,
    ) -> dict[str, Any]:
        if (
            target_physical_role != "gateway_coordinator"
            or method
            not in {
                "artifact_seal_begin",
                "artifact_seal_chunk",
                "artifact_seal_finish",
                "artifact_seal_cancel",
            }
            or not isinstance(params, Mapping)
            or re.fullmatch(r"[0-9a-f]{32}", str(channel_id or "")) is None
        ):
            raise ValueError("persistent artifact channel differs")
        result = _call_persistent_gateway_enclave(
            "gateway_coordinator",
            "rehearsal_inter_enclave_artifact_call",
            {
                "peer_role": self._peer_role,
                "method": method,
                "params": dict(params),
                "channel_id": str(channel_id),
            },
        )
        if not isinstance(result, Mapping):
            raise ValueError("persistent artifact response differs")
        return dict(result)


def _local_transport_certificate(role: str) -> bytes:
    body = base64.b64encode(
        hashlib.sha256(
            ("leadpoet-local-enclave-certificate:" + role).encode()
        ).digest()
    )
    return (
        b"-----BEGIN CERTIFICATE-----\n"
        + body
        + b"\n-----END CERTIFICATE-----\n"
    )


def _local_signing_private_key(role: str) -> Any:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    seed = hashlib.sha256(
        ("leadpoet-local-signing:" + role).encode("ascii")
    ).digest()
    return Ed25519PrivateKey.from_private_bytes(seed)


def _local_signing_public_key(role: str) -> str:
    from cryptography.hazmat.primitives import serialization

    return _local_signing_private_key(role).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def _local_boot_identity(role: str, config_hash: str) -> dict[str, Any]:
    from gateway.tee.topology import ROLE_SPECS
    from leadpoet_canonical.attested_v2 import (
        build_boot_attestation_user_data,
        build_boot_identity_body,
        canonical_json,
        create_boot_identity,
        sha256_json,
    )

    role_release = _gateway_release_input()["gateway_roles"][role]
    signing_pubkey = _local_signing_public_key(role)
    transport_pubkey = hashlib.sha256(
        ("leadpoet-local-transport:" + role).encode()
    ).hexdigest()
    certificate_hash = _sha256(
        _local_transport_certificate(role).decode("ascii")
    )
    provisional = {
        "role": str(ROLE_SPECS[role]["service_role"]),
        "physical_role": role,
        "commit_sha": str(role_release["commit_sha"]),
        "pcr0": str(role_release["pcr0"]),
        "build_manifest_hash": str(
            role_release["execution_manifest_hash"]
        ),
        "dependency_lock_hash": str(
            role_release["dependency_lock_hash"]
        ),
        "config_hash": config_hash,
        "boot_nonce": hashlib.sha256(
            ("leadpoet-local-boot:" + role).encode()
        ).hexdigest()[:32],
        "signing_pubkey": signing_pubkey,
        "transport_pubkey": transport_pubkey,
        "transport_certificate_hash": certificate_hash,
        "issued_at": "2026-07-25T00:00:00Z",
    }
    user_data = build_boot_attestation_user_data(provisional)
    body = build_boot_identity_body(
        **provisional,
        attestation_user_data_hash=sha256_json(user_data),
    )
    attestation = canonical_json(
        {
            "schema_version": "leadpoet.local_nitro_attestation.v1",
            "pcr0": provisional["pcr0"],
            "enclave_pubkey": signing_pubkey,
            "user_data": user_data,
        }
    ).encode("utf-8")
    return create_boot_identity(
        body=body,
        attestation_document_b64=base64.b64encode(attestation).decode(
            "ascii"
        ),
    )


def _local_verify_nitro_attestation_full(
    *,
    attestation_b64: str,
    expected_pcr0: str,
    expected_pubkey: Optional[str],
    expected_purpose: str,
    role: str,
    certificate_validity_at_attestation_time: bool = False,
) -> tuple[bool, dict[str, Any]]:
    del certificate_validity_at_attestation_time
    try:
        document = json.loads(
            base64.b64decode(attestation_b64, validate=True)
        )
    except Exception as exc:
        return False, {"error": f"invalid local Nitro document: {exc}"}
    schema = document.get("schema_version")
    if schema == "leadpoet.local_validator_nitro.v1":
        if set(document) != {
            "schema_version",
            "pcr0",
            "public_key_b64",
            "user_data_b64",
            "nonce_b64",
        }:
            return False, {"error": "local validator Nitro fields differ"}
        try:
            user_data = json.loads(
                base64.b64decode(
                    str(document["user_data_b64"]), validate=True
                )
            )
            enclave_pubkey = base64.b64decode(
                str(document["public_key_b64"]), validate=True
            ).hex()
            nonce = base64.b64decode(
                str(document["nonce_b64"]), validate=True
            )
        except Exception as exc:
            return False, {
                "error": f"local validator Nitro encoding differs: {exc}"
            }
        expected_schema = "leadpoet.local_validator_nitro.v1"
        expected_role = "validator"
        extra_valid = nonce == b""
    elif schema == "leadpoet.local_nitro_document.v1":
        if set(document) != {
            "schema_version",
            "pcr0",
            "public_key_b64",
            "user_data_b64",
            "nonce_b64",
        }:
            return False, {"error": "local gateway Nitro fields differ"}
        try:
            user_data = json.loads(
                base64.b64decode(
                    str(document["user_data_b64"]), validate=True
                )
            )
            enclave_pubkey = base64.b64decode(
                str(document["public_key_b64"]), validate=True
            ).hex()
            nonce = base64.b64decode(
                str(document["nonce_b64"]), validate=True
            )
        except Exception as exc:
            return False, {
                "error": f"local gateway Nitro encoding differs: {exc}"
            }
        expected_schema = "leadpoet.local_nitro_document.v1"
        expected_role = "gateway"
        extra_valid = nonce == b""
    elif schema == "leadpoet.local_nitro_attestation.v1":
        if set(document) != {
            "schema_version",
            "pcr0",
            "enclave_pubkey",
            "user_data",
        }:
            return False, {"error": "local Nitro document fields differ"}
        user_data = document.get("user_data")
        enclave_pubkey = document.get("enclave_pubkey")
        expected_schema = "leadpoet.local_nitro_attestation.v1"
        expected_role = "gateway"
        extra_valid = True
    else:
        return False, {"error": "local Nitro document schema differs"}
    valid = (
        document.get("schema_version") == expected_schema
        and document.get("pcr0") == expected_pcr0
        and (
            expected_pubkey is None
            or enclave_pubkey == expected_pubkey
        )
        and isinstance(user_data, dict)
        and user_data.get("purpose") == expected_purpose
        and role == expected_role
        and extra_valid
    )
    _external_event(
        "nitro_enclaves",
        "verify_attestation",
        expected_pcr0=expected_pcr0,
        role=role,
        verified=valid,
    )
    if not valid:
        return False, {"error": "local Nitro claim differs"}
    result = {
        "pcr0": document["pcr0"],
        "enclave_pubkey": enclave_pubkey,
        "user_data": user_data,
    }
    if schema == "leadpoet.local_nitro_document.v1":
        result["attestation_public_key"] = enclave_pubkey
    return True, result


def _configured_credential_ref(
    role_state: Mapping[str, Any],
    slot: str,
) -> str:
    configuration = role_state.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("local enclave runtime configuration is unavailable")
    if slot == "artifact_master_key":
        reference = configuration.get("artifact_master_key_ref_hash")
    else:
        provider_refs = configuration.get("provider_ref_hashes")
        if not isinstance(provider_refs, Mapping) or slot not in provider_refs:
            raise ValueError(
                "local enclave RPC credential slot is not boot-authorized"
            )
        reference = provider_refs[slot]
    normalized = str(reference or "").lower()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized):
        raise ValueError(
            "local enclave RPC credential reference is invalid"
        )
    return normalized


def _gateway_enclave_runtime_modules() -> tuple[Any, Any]:
    """Import enclave modules with the exact image-level module layout."""

    gateway_root = Path(os.environ.get("GATEWAY_ROOT", "")).resolve()
    enclave_root = gateway_root / "tee"
    merkle_path = enclave_root / "merkle.py"
    if not merkle_path.is_file():
        raise ValueError("local gateway enclave source root is unavailable")
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(enclave_root))
        merkle = importlib.import_module("merkle")
        nsm_lib = importlib.import_module("nsm_lib")
        tee_service = importlib.import_module("gateway.tee.tee_service")
    finally:
        sys.path[:] = original_path
    if Path(str(getattr(merkle, "__file__", ""))).resolve() != merkle_path:
        raise ValueError("local gateway enclave merkle source differs")
    if not Path(str(getattr(tee_service, "__file__", ""))).resolve().is_relative_to(
        gateway_root
    ):
        raise ValueError("local gateway enclave service source differs")
    return nsm_lib, tee_service


def _install_gateway_nsm_attestation(
    supplier: Callable[..., dict[str, Any]],
) -> None:
    """Install one local Nitro boundary in both production import namespaces."""

    gateway_root = Path(os.environ.get("GATEWAY_ROOT", "")).resolve()
    expected_path = (gateway_root / "tee/nsm_lib.py").resolve()
    modules = [
        importlib.import_module("nsm_lib"),
        importlib.import_module("gateway.tee.nsm_lib"),
    ]
    for module in modules:
        if Path(str(getattr(module, "__file__", ""))).resolve() != expected_path:
            raise ValueError("local gateway NSM source differs")
        module.get_attestation_document = supplier

    runtime_identity = importlib.import_module(
        "gateway.tee.runtime_identity_v2"
    )

    def local_nsm_attestation_document(
        *,
        user_data: bytes,
        signing_pubkey: bytes,
    ) -> bytes:
        response = supplier(
            user_data=bytes(user_data),
            public_key=bytes(signing_pubkey),
        )
        if (
            not isinstance(response, Mapping)
            or set(response) != {"Attestation"}
            or not isinstance(response["Attestation"], Mapping)
            or set(response["Attestation"]) != {"document"}
            or not isinstance(
                response["Attestation"]["document"],
                (bytes, bytearray),
            )
            or not response["Attestation"]["document"]
        ):
            raise ValueError("local gateway NSM response differs")
        return bytes(response["Attestation"]["document"])

    runtime_identity.nsm_attestation_document = (
        local_nsm_attestation_document
    )


class _LocalRuntimeIdentity:
    """Candidate-backed immutable runtime identity with deterministic local boot."""

    def __init__(
        self,
        *,
        role: str,
        role_state: Mapping[str, Any],
    ) -> None:
        from gateway.tee.build_identity import load_identity
        from gateway.tee import runtime_identity_v2
        from gateway.tee.topology import role_spec
        from leadpoet_canonical.attested_v2 import canonical_json, sha256_json

        configuration = role_state.get("configuration")
        config_hash = str(role_state.get("config_hash") or "")
        if not isinstance(configuration, Mapping):
            raise ValueError("local runtime configuration is unavailable")
        gateway_root = Path(os.environ.get("GATEWAY_ROOT", "")).resolve()
        build_identity = load_identity(
            gateway_root=gateway_root,
            expected_role=role,
        )
        normalized = runtime_identity_v2._validate_public_configuration(
            configuration
        )
        runtime_identity_v2._validate_release_configuration(
            normalized,
            physical_role=role,
            build_identity=build_identity,
        )
        topology = role_spec(role)
        document = {
            "schema_version": runtime_identity_v2.RUNTIME_CONFIG_SCHEMA_VERSION,
            "physical_role": role,
            "service_role": topology["service_role"],
            "configuration": normalized,
        }
        if sha256_json(document) != config_hash:
            raise ValueError("local candidate runtime configuration hash differs")
        self._role = role
        self._service_role = str(topology["service_role"])
        self._document = json.loads(canonical_json(document))
        self._boot = _local_boot_identity(role, config_hash)

    def runtime_configuration(self) -> dict[str, Any]:
        from leadpoet_canonical.attested_v2 import canonical_json

        return json.loads(canonical_json(self._document))

    def boot_identity(self) -> dict[str, Any]:
        return dict(self._boot)

    def transport_certificate_pem(self) -> bytes:
        return _local_transport_certificate(self._role)

    def public_status(self) -> dict[str, Any]:
        return {
            "schema_version": "leadpoet.enclave_runtime_config.v2",
            "status": "ready",
            "physical_role": self._role,
            "service_role": self._service_role,
            "commit_sha": self._boot["commit_sha"],
            "pcr0": self._boot["pcr0"],
            "config_hash": self._boot["config_hash"],
            "boot_identity_hash": self._boot["boot_identity_hash"],
            "transport_certificate_hash": self._boot[
                "transport_certificate_hash"
            ],
        }

    def release_role_expectation(self, physical_role: str) -> dict[str, str]:
        releases = self._document["configuration"]["release_roles"]
        value = releases.get(str(physical_role or ""))
        if not isinstance(value, Mapping):
            raise ValueError("local release role expectation is unavailable")
        return {str(name): str(item) for name, item in value.items()}

    def peer_release_expectation(self, physical_role: str) -> dict[str, str]:
        releases = self._document["configuration"]["peer_releases"]
        value = releases.get(str(physical_role or ""))
        if not isinstance(value, Mapping):
            raise ValueError("local peer release expectation is unavailable")
        return {str(name): str(item) for name, item in value.items()}

    def expected_peer_roles(self) -> tuple[str, ...]:
        return tuple(
            sorted(self._document["configuration"]["peer_releases"])
        )

    def verify_release_lineage_boot(
        self, identity: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from gateway.tee.release_lineage_v2 import (
            build_compact_release_lineage_boot_verifier_v2,
        )

        verifier = build_compact_release_lineage_boot_verifier_v2(
            self._document["configuration"]["gateway_release_lineage"]
        )
        return verifier(identity)

    def research_lab_config(self) -> Any:
        from gateway.tee.research_lab_runtime_config_v2 import (
            research_lab_config_from_document,
        )

        return research_lab_config_from_document(
            self._document["configuration"][
                "research_lab_execution_config"
            ]
        )

    def apply_research_lab_behavior_environment(self) -> None:
        from gateway.tee.research_lab_runtime_config_v2 import (
            apply_behavior_environment,
        )

        apply_behavior_environment(
            self._document["configuration"][
                "research_lab_execution_config"
            ]
        )


def _selective_metagraph_fixture(
    block: int,
    *,
    last_field: int | None = None,
) -> str:
    from leadpoet_canonical.chain_source_v2 import (
        CHAIN_SELECTIVE_RESULT_LAST_FIELDS,
    )

    supported_last_fields = tuple(CHAIN_SELECTIVE_RESULT_LAST_FIELDS)
    selected_last_field = (
        max(supported_last_fields)
        if last_field is None
        else int(last_field)
    )
    if selected_last_field not in supported_last_fields:
        raise ValueError(
            "local selective metagraph layout is outside candidate policy"
        )
    account_ids = load_rehearsal_metagraph_account_ids(SOURCE_ROOT)
    owner = account_ids[0]
    encoded = bytearray((1, 0x1D, 0x01))
    encoded.extend(b"\x00" * 4)
    encoded.extend(b"\x01" + owner)
    encoded.extend(b"\x00")
    encoded.extend(b"\x01" + ((int(block) << 2) | 2).to_bytes(4, "little"))
    encoded.extend(b"\x00" * 44)
    encoded.extend(b"\x01" + bytes((len(account_ids) << 2,)))
    encoded.extend(b"".join(account_ids))
    encoded.extend(b"\x00" * (selected_last_field - 52))
    return "0x" + bytes(encoded).hex()


def _local_chain_rpc(body: bytes, *, archive: bool) -> bytes:
    from leadpoet_canonical.chain_source_v2 import (
        last_update_storage_key,
        subnet_epoch_storage_key,
        weights_storage_key,
    )

    try:
        request = json.loads(body)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("local chain request is not valid JSON") from exc
    if set(request) != {"jsonrpc", "id", "method", "params"}:
        raise ValueError("local chain JSON-RPC fields differ")
    if request.get("jsonrpc") != "2.0" or not isinstance(
        request.get("params"), list
    ):
        raise ValueError("local chain JSON-RPC envelope differs")
    method = str(request["method"])
    params = request["params"]
    current_hash = _block_hash(CURRENT_BLOCK)
    predecessor_hash = _block_hash(CUTOVER_BLOCK - 1)

    if method == "chain_getFinalizedHead" and params == []:
        result: Any = current_hash
    elif method == "chain_getBlockHash" and len(params) == 1:
        block = int(params[0])
        if block == 0:
            result = GENESIS_HASH
        elif block == CUTOVER_BLOCK:
            result = CUTOVER_BLOCK_HASH
        else:
            result = _block_hash(block)
    elif method == "chain_getHeader" and len(params) == 1:
        at_hash = str(params[0])
        block = _block_number(at_hash)
        result = {
            "number": hex(block),
            "stateRoot": _block_hash(block + 1),
            "parentHash": (
                predecessor_hash
                if block == CUTOVER_BLOCK
                else _block_hash(block - 1)
            ),
            "extrinsicsRoot": _block_hash(block + 2),
            "digest": {"logs": []},
        }
    elif method == "state_getRuntimeVersion" and params == [current_hash]:
        result = {
            "specName": "node-subtensor",
            "implName": "node-subtensor",
            "authoringVersion": 1,
            "specVersion": 440,
            "implVersion": 0,
            "apis": [],
            "transactionVersion": 1,
            "stateVersion": 1,
        }
    elif method == "state_call" and len(params) == 3:
        runtime_method = str(params[0])
        at_hash = str(params[2])
        block = _block_number(at_hash)
        if runtime_method == "SubnetInfoRuntimeApi_get_selective_mechagraph":
            from leadpoet_canonical.chain_source_v2 import (
                CHAIN_SELECTIVE_RESULT_LAST_FIELDS,
            )

            result = _selective_metagraph_fixture(
                block,
                last_field=(
                    min(CHAIN_SELECTIVE_RESULT_LAST_FIELDS)
                    if archive
                    else max(CHAIN_SELECTIVE_RESULT_LAST_FIELDS)
                ),
            )
        elif runtime_method == "SwapRuntimeApi_current_alpha_price":
            if at_hash != current_hash:
                raise ValueError(
                    "local price runtime call is not pinned to finalized head"
                )
            result = "0x9a0f4f0000000000"
        else:
            raise ValueError("local runtime method is unknown")
    elif method == "state_getStorage" and len(params) == 2:
        storage_key, at_hash = map(str, params)
        block = _block_number(at_hash)
        subnet_epoch_key = subnet_epoch_storage_key(
            storage_name="SubnetEpochIndex",
            netuid=71,
        )
        weight_key = weights_storage_key(netuid=71, validator_uid=0)
        update_key = last_update_storage_key(netuid=71)
        if storage_key == subnet_epoch_key:
            result = "0x" + _subnet_epoch_index_at(block).to_bytes(
                8, "little"
            ).hex()
        elif storage_key == weight_key:
            result = "0x" + (
                b"\x08"
                + (0).to_bytes(2, "little")
                + (65_535).to_bytes(2, "little")
                + (1).to_bytes(2, "little")
                + (16_384).to_bytes(2, "little")
            ).hex()
        elif storage_key == update_key:
            last_update = min(block, LAST_EPOCH_BLOCK - 1)
            result = "0x" + (
                b"\x08"
                + last_update.to_bytes(8, "little")
                + last_update.to_bytes(8, "little")
            ).hex()
        else:
            names = (
                "Tempo",
                "LastEpochBlock",
                "PendingEpochAt",
                "BlocksSinceLastStep",
            )
            matching = [
                name
                for name in names
                if storage_key
                == subnet_epoch_storage_key(storage_name=name, netuid=71)
            ]
            if len(matching) != 1:
                raise ValueError("local chain storage key is unknown")
            name = matching[0]
            if at_hash == CUTOVER_BLOCK_HASH:
                values = {
                    "LastEpochBlock": CUTOVER_BLOCK,
                }
            elif at_hash == current_hash:
                values = {
                    "Tempo": TEMPO,
                    "LastEpochBlock": LAST_EPOCH_BLOCK,
                    "PendingEpochAt": 0,
                    "BlocksSinceLastStep": CURRENT_BLOCK - LAST_EPOCH_BLOCK,
                }
            else:
                raise ValueError("local chain storage hash is unknown")
            if name not in values:
                raise ValueError("local chain storage value is unavailable")
            width = 2 if name == "Tempo" else 8
            result = "0x" + int(values[name]).to_bytes(width, "little").hex()
    else:
        raise ValueError(
            "local chain received an unknown RPC: %s %r" % (method, params)
        )

    _external_event(
        "stateful_subnet_chain",
        "archive_rpc" if archive else "rpc",
        method=method,
        exact_block_hash=(
            str(params[-1])
            if params
            and isinstance(params[-1], str)
            and str(params[-1]).startswith("0x")
            else ""
        ),
    )
    return json.dumps(
        {"jsonrpc": "2.0", "id": request["id"], "result": result},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _local_provider_transport(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout_ms: int,
    upstream_proxy_url: Optional[str] = None,
    max_response_bytes: int = 8 * 1024 * 1024,
    allow_http2: bool = True,
    connection_scope: str = "",
) -> dict[str, Any]:
    if not isinstance(allow_http2, bool):
        raise ValueError("local provider HTTP/2 policy is invalid")
    normalized_connection_scope = str(connection_scope or "")
    if upstream_proxy_url:
        if not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            normalized_connection_scope,
        ):
            raise ValueError(
                "local paid-provider request lacks its measured connection scope"
            )
    elif normalized_connection_scope:
        raise ValueError(
            "local direct-provider request supplied an unexpected connection scope"
        )
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.port not in {None, 443}:
        raise ValueError("local provider boundary requires production HTTPS")
    host = str(parsed.hostname or "").lower()
    normalized_method = str(method).upper()
    response_headers: dict[str, str] = {"content-type": "application/json"}
    if host == "qplwoislplkcegvdmbim.supabase.co":
        provider_broker = __import__(
            "gateway.tee.provider_broker_v2",
            fromlist=["BUILTIN_PROVIDER_ROUTES"],
        )
        expected_http2 = bool(
            provider_broker.BUILTIN_PROVIDER_ROUTES["supabase"].allow_http2
        )
        if allow_http2 is not expected_http2:
            raise ValueError("local Supabase provider route protocol differs")
        local_url = urlunsplit(
            ("http", "127.0.0.1:54321", parsed.path, parsed.query, "")
        )
        request = __import__("urllib.request", fromlist=["Request"]).Request(
            local_url,
            data=bytes(body) if normalized_method != "GET" else None,
            headers={str(name): str(value) for name, value in headers.items()},
            method=normalized_method,
        )
        with _real_urlopen(
            request,
            timeout=max(0.001, int(timeout_ms) / 1000.0),
        ) as response:
            response_body = response.read(max_response_bytes + 1)
            status = int(getattr(response, "status", 200))
            response_headers = {
                str(name).lower(): str(value)
                for name, value in response.headers.items()
            }
        boundary = "supabase_postgrest"
    elif host in {
        "entrypoint-finney.opentensor.ai",
        "archive.chain.opentensor.ai",
    }:
        if normalized_method != "POST" or parsed.path != "/":
            raise ValueError("local chain provider route differs")
        response_body = _local_chain_rpc(
            bytes(body),
            archive=host == "archive.chain.opentensor.ai",
        )
        status = 200
        boundary = "stateful_subnet_chain"
    elif host == "api.coingecko.com":
        if (
            normalized_method != "GET"
            or parsed.path != "/api/v3/simple/price"
        ):
            raise ValueError("local price provider route differs")
        response_body = b'{"bittensor":{"usd":201.25}}'
        status = 200
        boundary = "coingecko"
    elif host == "api.exa.ai":
        if (
            normalized_method != "POST"
            or parsed.path != "/search"
            or parsed.query
            or headers.get("x-api-key") != "rehearsal-exa"
        ):
            raise ValueError("local Exa provider route differs")
        payload = json.loads(body)
        if (
            not isinstance(payload, dict)
            or payload.get("query") != "provider preflight"
            or payload.get("numResults") != 1
        ):
            raise ValueError("local Exa provider preflight payload differs")
        response_body = b'{"results":[]}'
        status = 200
        boundary = "exa"
    elif host == "api.scrapingdog.com":
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if (
            normalized_method != "GET"
            or parsed.path != "/account"
            or bytes(body)
            or query != {"api_key": "rehearsal-scrapingdog"}
        ):
            raise ValueError("local ScrapingDog provider route differs")
        response_body = b'{"status":"active"}'
        status = 200
        boundary = "scrapingdog"
    else:
        raise ValueError("local provider boundary rejected unknown host")
    if boundary in {"exa", "scrapingdog"}:
        permitted_proxies = {
            (
                "https://rehearsal-auto:rehearsal-auto-password@"
                "93.184.216.34:443"
            ),
            (
                "https://rehearsal-scoring:rehearsal-scoring-password@"
                "93.184.216.34:443"
            ),
        }
        if upstream_proxy_url not in permitted_proxies:
            raise ValueError(
                "local paid-provider request lacks its job-scoped TLS proxy"
            )
    if len(response_body) > int(max_response_bytes):
        raise ValueError("local provider response exceeds requested ceiling")
    _external_event(
        boundary,
        "provider_transport",
        method=normalized_method,
        host=host,
        path=parsed.path,
        http2_allowed=allow_http2,
        connection_scoped=bool(normalized_connection_scope),
        status=status,
        response_bytes=len(response_body),
    )
    return {
        "http_status": status,
        "headers": response_headers,
        "body": response_body,
        "tls_peer_chain_hash": _sha256("leadpoet-local-tls:" + host),
        "tls_protocol": "TLSv1.3",
    }


def _artifact_record_path(bucket: str, key: str) -> Path:
    digest = hashlib.sha256(
        (str(bucket) + "\0" + str(key)).encode("utf-8")
    ).hexdigest()
    return STATE_ROOT / "s3-artifacts" / (digest + ".json")


def _private_model_signing_key() -> Any:
    """Return a deterministic local P-256 equivalent of the production key."""

    from cryptography.hazmat.primitives.asymmetric import ec

    order = int(
        "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551",
        16,
    )
    seed = hashlib.sha256(
        b"leadpoet-local-private-model-signing-key-v1"
    ).digest()
    scalar = (int.from_bytes(seed, "big") % (order - 1)) + 1
    return ec.derive_private_key(scalar, ec.SECP256R1())


def _sign_private_model_manifest_hash(manifest_hash: str) -> bytes:
    """Sign one canonical manifest hash through the strict local boundary."""

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    normalized = str(manifest_hash)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized):
        raise ValueError("local private model manifest hash is invalid")
    return _private_model_signing_key().sign(
        normalized.encode("utf-8"),
        ec.ECDSA(hashes.SHA256()),
    )


def _install_private_model_s3_object(
    *,
    bucket: str,
    key: str,
    body: bytes,
) -> None:
    """Install an immutable artifact or the one mutable branch pointer."""

    normalized_bucket = str(bucket)
    normalized_key = str(key)
    payload = bytes(body)
    if (
        normalized_bucket != _PRIVATE_MODEL_BUCKET
        or not normalized_key.startswith(_PRIVATE_MODEL_PREFIX)
        or not payload
    ):
        raise ValueError("local private model S3 fixture contract differs")
    identity = (normalized_bucket, normalized_key)
    with _PRIVATE_MODEL_OBJECTS_LOCK:
        existing = _PRIVATE_MODEL_OBJECTS.get(identity)
        if (
            existing is not None
            and existing != payload
            and normalized_key != _PRIVATE_MODEL_POINTER_KEY
        ):
            raise ValueError("local private model immutable object differs")
        _PRIVATE_MODEL_OBJECTS[identity] = payload


def _clear_private_model_s3_objects() -> None:
    """Reset only test-owned private-model fixtures between scenarios."""

    with _PRIVATE_MODEL_OBJECTS_LOCK:
        _PRIVATE_MODEL_OBJECTS.clear()


def _local_artifact_transport(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout_ms: int,
    max_response_bytes: int = 128 * 1024 * 1024,
) -> dict[str, Any]:
    del headers, body, timeout_ms
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.port not in {None, 443}:
        raise ValueError("local artifact verifier requires production HTTPS")
    state, _ = _gateway_enclave_state()
    role_state = state.get("roles", {}).get("gateway_coordinator")
    configuration = (role_state or {}).get("configuration") or {}
    policy = configuration.get("encrypted_artifact_policy") or {}
    if parsed.hostname != policy.get("bucket_host"):
        raise ValueError("local artifact bucket host differs")
    key = parsed.path.lstrip("/")
    bucket = str(parsed.hostname).split(".s3", 1)[0]
    record_path = _artifact_record_path(bucket, key)
    if not record_path.is_file():
        raise FileNotFoundError("local persisted artifact is unavailable")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if str(method).upper() == "GET":
        response_body = base64.b64decode(record["body_b64"], validate=True)
    elif str(method).upper() == "HEAD":
        response_body = b""
    else:
        raise ValueError("local artifact method differs")
    if len(response_body) > int(max_response_bytes):
        raise ValueError("local artifact response exceeds requested ceiling")
    response_headers = {
        "content-type": record["content_type"],
        "content-length": str(record["content_length"]),
        "x-amz-object-lock-mode": record["object_lock_mode"],
        "x-amz-object-lock-retain-until-date": record["retain_until"],
        "x-amz-version-id": record["version_id"],
    }
    _external_event(
        "aws_s3_object_lock",
        "verify_" + str(method).lower(),
        bucket=bucket,
        key=key,
        object_lock_mode=record["object_lock_mode"],
    )
    return {
        "http_status": 200,
        "headers": response_headers,
        "body": response_body,
        "tls_peer_chain_hash": _sha256(
            "leadpoet-local-tls:" + str(parsed.hostname)
        ),
        "tls_protocol": "TLSv1.3",
    }


def _gateway_runtime_objects(
    role: str,
    role_state: Mapping[str, Any],
) -> dict[str, Any]:
    config_hash = str(role_state.get("config_hash") or "")
    cache_key = role + ":" + config_hash
    with _GATEWAY_RUNTIME_OBJECTS_LOCK:
        existing = _GATEWAY_RUNTIME_OBJECTS.get(cache_key)
        if existing is not None:
            return existing

        from gateway.tee.artifact_persistence_v2 import (
            ArtifactPersistenceVerifierV2,
        )
        from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
        from gateway.tee.provider_broker_v2 import (
            ProviderBrokerV2,
            credential_reference_hash,
        )

        nsm_lib, tee_service = _gateway_enclave_runtime_modules()
        runtime = _LocalRuntimeIdentity(role=role, role_state=role_state)
        configuration = runtime.runtime_configuration()["configuration"]
        fixed_datetime = datetime(2026, 7, 25, tzinfo=timezone.utc)

        def local_attestation_document(
            *,
            user_data: bytes,
            public_key: bytes,
            nonce: bytes = b"",
        ) -> dict[str, Any]:
            document = json.dumps(
                {
                    "schema_version": "leadpoet.local_nitro_document.v1",
                    "pcr0": str(
                        _gateway_release_input()["gateway_roles"][role]["pcr0"]
                    ),
                    "public_key_b64": base64.b64encode(public_key).decode("ascii"),
                    "user_data_b64": base64.b64encode(user_data).decode("ascii"),
                    "nonce_b64": base64.b64encode(nonce).decode("ascii"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            _external_event(
                "nitro_enclaves",
                "verify_attestation",
                physical_role=role,
                purpose="gateway_kms_recipient",
                document_bytes=len(document),
            )
            return {"Attestation": {"document": document}}

        _install_gateway_nsm_attestation(local_attestation_document)
        vault = EncryptedArtifactVaultV2(
            master_key=hashlib.sha256(
                b"leadpoet-local-artifact-master-key"
            ).digest(),
            boot_identity_hash=runtime.boot_identity()[
                "boot_identity_hash"
            ],
            retention_days=int(
                configuration["encrypted_artifact_policy"][
                    "minimum_retention_days"
                ]
            ),
            clock=lambda: fixed_datetime,
        )
        broker = ProviderBrokerV2(
            credential_ref_hashes=configuration["provider_ref_hashes"],
            retry_policy_hashes=configuration[
                "provider_retry_policy_hashes"
            ],
            job_credential_slot_ref_hashes=configuration[
                "job_lease_slot_ref_hashes"
            ],
            transport=_local_provider_transport,
            artifact_sink=vault.seal,
            clock=lambda: "2026-07-25T00:00:00Z",
        )
        rehearsal_credentials = {
            "supabase_service_role": "rehearsal-secret",
            "openrouter": "rehearsal-openrouter",
            "exa": "rehearsal-exa",
            "scrapingdog": "rehearsal-scrapingdog",
            "deepline": "rehearsal-deepline",
            "truelist": "rehearsal-truelist",
        }
        if {
            slot: credential_reference_hash(value)
            for slot, value in rehearsal_credentials.items()
        } == configuration["provider_ref_hashes"]:
            broker.provision_credentials(rehearsal_credentials)
        persistence = ArtifactPersistenceVerifierV2(
            vault=vault,
            policy=configuration["encrypted_artifact_policy"],
            transport=_local_artifact_transport,
            clock=lambda: "2026-07-25T00:00:00Z",
            sleeper=lambda _seconds: None,
        )

        os.environ["LEADPOET_ENCLAVE_ROLE"] = role
        tee_service.v2_runtime_identity = runtime
        tee_service.v2_artifact_vault = vault
        tee_service.v2_provider_broker = broker
        tee_service.v2_artifact_persistence_verifier = persistence
        tee_service.v2_scoring_job_manager = None
        tee_service.v2_coordinator_job_manager = None
        tee_service.v2_autoresearch_job_manager = None
        tee_service.v2_provider_semantics_authority = None
        tee_service.v2_kms_recipient = None
        tee_service.v2_inter_enclave_client = None
        tee_service.sign_data = lambda data: _local_signing_private_key(
            role
        ).sign(bytes(data))
        # Candidate job managers still execute unchanged. Scoring and
        # autoresearch cross the same coordinator-owned provider authority
        # boundary as production; the coordinator is the only process that
        # receives the measured job credential lease.
        if role == "gateway_coordinator":
            tee_service.execute_v2_provider_request = broker.execute
            tee_service.execute_v2_provider_probe_request = broker.execute
        else:
            tee_service.execute_v2_provider_request = (
                lambda request: _call_persistent_gateway_enclave(
                    "gateway_coordinator",
                    "rehearsal_inter_enclave_provider_execute",
                    {"peer_role": role, "request": dict(request)},
                )
            )
            tee_service.execute_v2_provider_probe_request = (
                lambda request: _call_persistent_gateway_enclave(
                    "gateway_coordinator",
                    "rehearsal_inter_enclave_provider_probe_resolve",
                    {"peer_role": role, "request": dict(request)},
                )
            )
        if role == "gateway_coordinator":
            tee_service.seal_v2_inter_enclave_artifact = (
                lambda *, plaintext, job_id, purpose, artifact_kind: vault.seal(
                    bytes(plaintext),
                    job_id=str(job_id),
                    purpose=str(purpose),
                    artifact_kind=str(artifact_kind),
                )
            )
        else:
            from gateway.tee.inter_enclave_artifact_v2 import (
                seal_artifact_over_attested_tls_v2,
            )

            artifact_client = _PersistentInterEnclaveArtifactClient(
                peer_role=role
            )
            tee_service.seal_v2_inter_enclave_artifact = (
                lambda *, plaintext, job_id, purpose, artifact_kind: (
                    seal_artifact_over_attested_tls_v2(
                        client=artifact_client,
                        plaintext=bytes(plaintext),
                        job_id=str(job_id),
                        purpose=str(purpose),
                        artifact_kind=str(artifact_kind),
                    )
                )
            )
        objects = {
            "runtime": runtime,
            "vault": vault,
            "broker": broker,
            "persistence": persistence,
            "tee_service": tee_service,
        }
        _GATEWAY_RUNTIME_OBJECTS[cache_key] = objects
        _external_event(
            "nitro_enclaves",
            "candidate_role_runtime",
            physical_role=role,
            config_hash=config_hash,
            candidate_source=str(
                Path(str(tee_service.__file__)).resolve()
            ),
        )
        return objects


def _unwrap_candidate_rpc(response: Mapping[str, Any]) -> Any:
    if response.get("status") == "error" or "error" in response:
        raise ValueError(
            "candidate enclave RPC failed: %s" % response.get("error")
        )
    if set(response) != {"result"}:
        raise ValueError("candidate enclave RPC envelope differs")
    return response["result"]


def _install_allocation_resolver_diagnostic() -> None:
    from gateway.tee.coordinator_allocation_source_v2 import (
        CoordinatorAllocationSourceV2,
    )

    original = CoordinatorAllocationSourceV2.resolve
    if getattr(original, "_rehearsal_diagnostic", False):
        return

    def resolve_with_diagnostic(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original(self, *args, **kwargs)
        except Exception as exc:
            _external_event(
                "candidate_gateway",
                "allocation_resolver_exception",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

    resolve_with_diagnostic._rehearsal_diagnostic = True  # type: ignore[attr-defined]
    CoordinatorAllocationSourceV2.resolve = resolve_with_diagnostic


def _handle_gateway_enclave_rpc(
    role: str,
    method: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    import leadpoet_canonical.nitro as leadpoet_nitro
    from gateway.tee.provider_broker_v2 import (
        expected_job_credential_slot_ref_hashes,
        expected_provider_credential_slots,
        provider_registry_hash,
    )
    from gateway.tee.topology import ROLE_SPECS, topology_hash
    from leadpoet_canonical.attested_v2 import (
        sha256_json,
        validate_boot_identity,
    )

    leadpoet_nitro.verify_nitro_attestation_full = (
        _local_verify_nitro_attestation_full
    )
    if role not in ROLE_SPECS:
        raise ValueError("local enclave RPC role is unknown")
    release_role = _gateway_release_input()["gateway_roles"][role]
    empty_params = not params
    if role == "gateway_coordinator" and method in {
        "initialize_event_signer",
        "get_event_signing_identity",
    }:
        nsm_lib, tee_service = _gateway_enclave_runtime_modules()

        def local_attestation_document(
            *,
            user_data: bytes,
            public_key: bytes,
            nonce: bytes = b"",
        ) -> dict[str, Any]:
            document = json.dumps(
                {
                    "schema_version": "leadpoet.local_nitro_document.v1",
                    "pcr0": str(
                        _gateway_release_input()["gateway_roles"][role]["pcr0"]
                    ),
                    "public_key_b64": base64.b64encode(public_key).decode("ascii"),
                    "user_data_b64": base64.b64encode(user_data).decode("ascii"),
                    "nonce_b64": base64.b64encode(nonce).decode("ascii"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            _external_event(
                "nitro_enclaves",
                "verify_attestation",
                physical_role=role,
                purpose="gateway_event_signing",
                document_bytes=len(document),
            )
            return {"Attestation": {"document": document}}

        _install_gateway_nsm_attestation(local_attestation_document)
        if method == "initialize_event_signer":
            if set(params) != {"prev_log_tip_hash"}:
                raise ValueError("local event signer initialization fields differ")
            return tee_service.initialize_event_signer(
                params.get("prev_log_tip_hash")
            )
        if not empty_params:
            raise ValueError("local event signer identity params differ")
        return tee_service._event_signing_identity()
    if role == "gateway_coordinator" and method in {
        "append_event",
        "sign_transparency_event",
        "get_buffer",
        "clear_buffer",
        "acknowledge_checkpoint",
        "get_buffer_size",
        "get_buffer_stats",
        "build_checkpoint",
    }:
        _, tee_service = _gateway_enclave_runtime_modules()
        if method == "append_event":
            if set(params) != {"event"} or not isinstance(
                params.get("event"), Mapping
            ):
                raise ValueError("local event append fields differ")
            return tee_service.append_event(dict(params["event"]))
        if method == "sign_transparency_event":
            if set(params) != {
                "event_type",
                "payload",
                "payload_hash",
            } or not isinstance(params.get("payload"), Mapping):
                raise ValueError("local transparency event fields differ")
            return tee_service.sign_transparency_event(
                event_type=params["event_type"],
                payload=dict(params["payload"]),
                payload_hash=params["payload_hash"],
            )
        if method == "acknowledge_checkpoint":
            if set(params) != {
                "checkpoint_number",
                "merkle_root",
                "sequence_range",
            } or not isinstance(params.get("sequence_range"), Mapping):
                raise ValueError(
                    "local checkpoint acknowledgement fields differ"
                )
            return tee_service.acknowledge_checkpoint(
                checkpoint_number=params["checkpoint_number"],
                merkle_root=params["merkle_root"],
                sequence_range=dict(params["sequence_range"]),
            )
        if not empty_params:
            raise ValueError("local event buffer RPC params differ")
        return {
            "get_buffer": tee_service.get_buffer,
            "clear_buffer": tee_service.clear_buffer,
            "get_buffer_size": tee_service.get_buffer_size,
            "get_buffer_stats": tee_service.get_buffer_stats,
            "build_checkpoint": tee_service.build_checkpoint,
        }[method]()
    if method == "role_health" and empty_params:
        return {
            "status": "healthy",
            "role": role,
            "service_role": ROLE_SPECS[role]["service_role"],
            "commit_sha": release_role["commit_sha"],
            "build_identity_hash": release_role["build_identity_hash"],
            "execution_manifest_hash": release_role[
                "execution_manifest_hash"
            ],
            "dependency_lock_hash": release_role[
                "dependency_lock_hash"
            ],
            "topology_hash": topology_hash(),
            "public_key": _local_signing_public_key(role),
            "pcr0": release_role["pcr0"],
            "v2_runtime": {
                "schema_version": "leadpoet.enclave_runtime_config.v2",
                "status": "not_configured",
                "physical_role": role,
                "service_role": ROLE_SPECS[role]["service_role"],
            },
            "parent_rpc_transport": {
                "schema_version": (
                    "leadpoet.gateway_vsock_rpc_transport_health.v2"
                ),
                "status": "healthy",
            },
            "inter_enclave_transport": {
                "schema_version": (
                    "leadpoet.inter_enclave_role_transport_health.v2"
                ),
                "status": "error",
                "server": {"status": "unavailable"},
                "client": {"status": "unavailable"},
            },
        }
    if method == "get_attestation":
        if role != "gateway_coordinator" or not empty_params:
            raise ValueError(
                "local gateway attestation RPC contract differs"
            )
        attestation_payload = {
            "schema_version": "leadpoet.local_gateway_attestation.v1",
            "commit_sha": release_role["commit_sha"],
            "physical_role": role,
            "build_identity_hash": release_role["build_identity_hash"],
            "pcr0": release_role["pcr0"],
        }
        attestation_document = json.dumps(
            attestation_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        _external_event(
            "nitro_enclaves",
            "attestation_document",
            physical_role=role,
            commit_sha=release_role["commit_sha"],
            pcr0=release_role["pcr0"],
            document_bytes=len(attestation_document),
            status="ready",
        )
        return {
            "attestation_document": attestation_document.hex(),
            "public_key": _local_signing_public_key(role),
            "code_hash": hashlib.sha256(
                (
                    release_role["commit_sha"]
                    + ":"
                    + release_role["build_identity_hash"]
                ).encode()
            ).hexdigest(),
            "pcr0": release_role["pcr0"],
            "pcr1": hashlib.sha384(b"leadpoet-local-pcr1").hexdigest(),
            "pcr2": hashlib.sha384(b"leadpoet-local-pcr2").hexdigest(),
        }
    if method == "v2_configure_runtime":
        if set(params) != {
            "schema_version",
            "configuration",
            "configuration_hash",
        } or params.get("schema_version") != (
            "leadpoet.enclave_runtime_config.v2"
        ):
            raise ValueError("local enclave runtime configuration differs")
        configuration = params.get("configuration")
        config_hash = str(params.get("configuration_hash") or "")
        document = {
            "schema_version": "leadpoet.enclave_runtime_config.v2",
            "physical_role": role,
            "service_role": ROLE_SPECS[role]["service_role"],
            "configuration": configuration,
        }
        if not isinstance(configuration, dict) or sha256_json(
            document
        ) != config_hash:
            raise ValueError("local enclave runtime configuration hash differs")

        def configure(state: dict[str, Any]) -> None:
            current = state.setdefault("roles", {}).get(role)
            if current is not None and current.get("config_hash") != config_hash:
                raise ValueError(
                    "local enclave runtime configuration changed in one boot"
                )
            state["roles"][role] = {
                "config_hash": config_hash,
                "configuration": configuration,
            }

        _gateway_enclave_state(configure)
        return {
            "schema_version": "leadpoet.enclave_runtime_config.v2",
            "status": "ready",
            "physical_role": role,
            "service_role": ROLE_SPECS[role]["service_role"],
            "commit_sha": release_role["commit_sha"],
            "pcr0": release_role["pcr0"],
            "config_hash": config_hash,
            "boot_identity_hash": _local_boot_identity(
                role, config_hash
            )["boot_identity_hash"],
            "transport_certificate_hash": _sha256(
                _local_transport_certificate(role).decode("ascii")
            ),
        }
    state, _ = _gateway_enclave_state()
    role_state = state.get("roles", {}).get(role)
    if method.startswith("v2_") and role_state is None:
        raise ValueError("local enclave runtime is not configured")
    config_hash = str((role_state or {}).get("config_hash") or "")
    if role == "gateway_coordinator" and method == (
        "rehearsal_inter_enclave_artifact_call"
    ):
        if set(params) != {"peer_role", "method", "params", "channel_id"}:
            raise ValueError("local inter-enclave artifact fields differ")
        peer_role = str(params.get("peer_role") or "")
        target_method = str(params.get("method") or "")
        target_params = params.get("params")
        channel_id = str(params.get("channel_id") or "")
        peer_state = state.get("roles", {}).get(peer_role)
        if (
            peer_role not in {"gateway_scoring", "gateway_autoresearch"}
            or target_method
            not in {
                "artifact_seal_begin",
                "artifact_seal_chunk",
                "artifact_seal_finish",
                "artifact_seal_cancel",
            }
            or not isinstance(target_params, Mapping)
            or re.fullmatch(r"[0-9a-f]{32}", channel_id) is None
            or not isinstance(peer_state, Mapping)
        ):
            raise ValueError("local inter-enclave artifact peer differs")
        peer_config_hash = str(peer_state.get("config_hash") or "")
        objects = _gateway_runtime_objects(role, role_state)
        return objects["tee_service"].handle_inter_enclave_rpc(
            target_method,
            dict(target_params),
            {
                "physical_role": peer_role,
                "service_role": ROLE_SPECS[peer_role]["service_role"],
                "boot_identity": _local_boot_identity(
                    peer_role, peer_config_hash
                ),
            },
        )
    if role == "gateway_coordinator" and method in {
        "rehearsal_inter_enclave_provider_execute",
        "rehearsal_inter_enclave_provider_probe_resolve",
    }:
        if set(params) != {"peer_role", "request"}:
            raise ValueError("local inter-enclave provider fields differ")
        peer_role = str(params.get("peer_role") or "")
        request = params.get("request")
        target_method = {
            "rehearsal_inter_enclave_provider_execute": "provider_execute",
            "rehearsal_inter_enclave_provider_probe_resolve": (
                "provider_probe_resolve"
            ),
        }[method]
        allowed_peers = (
            {"gateway_scoring", "gateway_autoresearch"}
            if target_method == "provider_execute"
            else {"gateway_autoresearch"}
        )
        if peer_role not in allowed_peers or not isinstance(request, Mapping):
            raise ValueError("local inter-enclave provider peer differs")
        objects = _gateway_runtime_objects(role, role_state)
        return objects["tee_service"].handle_inter_enclave_rpc(
            target_method,
            dict(request),
            {"physical_role": peer_role},
        )
    execution_prefix = {
        "gateway_coordinator": "coordinator_v2_",
        "gateway_scoring": "scoring_v2_",
        "gateway_autoresearch": "autoresearch_v2_",
    }[role]
    if method.startswith(execution_prefix):
        if role == "gateway_coordinator":
            _install_allocation_resolver_diagnostic()
        objects = _gateway_runtime_objects(role, role_state)
        return _unwrap_candidate_rpc(
            objects["tee_service"].handle_v2_execution_rpc(
                method, dict(params)
            )
        )
    if role == "gateway_coordinator" and method in {
        "v2_list_encrypted_artifacts",
        "v2_export_encrypted_artifact",
        "v2_verify_encrypted_artifact_persistence",
        "v2_get_job_kms_recipient",
        "v2_provision_job_encrypted_secret",
        "v2_provision_job_sealed_source_add_secret",
        "v2_provision_job_sealed_openrouter_secret",
        "v2_release_job_credentials",
        "v2_get_source_add_ingress_recipient",
        "v2_seal_source_add_ingress_credential",
        "v2_get_openrouter_ingress_recipient",
        "v2_seal_openrouter_ingress_credential",
    }:
        objects = _gateway_runtime_objects(role, role_state)
        return _unwrap_candidate_rpc(
            objects["tee_service"].handle_v2_runtime_rpc(
                method, dict(params)
            )
        )
    if method == "v2_get_boot_identity" and empty_params:
        return _local_boot_identity(role, config_hash)
    if method == "v2_get_transport_certificate" and empty_params:
        return {
            "certificate_pem_b64": base64.b64encode(
                _local_transport_certificate(role)
            ).decode("ascii"),
            "status": {"status": "ready", "physical_role": role},
        }
    if method == "v2_register_peer":
        if set(params) != {"boot_identity", "certificate_pem_b64"}:
            raise ValueError("local enclave peer registration fields differ")
        boot = params.get("boot_identity")
        if not isinstance(boot, dict):
            raise ValueError("local enclave peer boot identity is invalid")
        validate_boot_identity(boot)
        peer = str(boot.get("physical_role") or "")
        expected_peers = (
            set(ROLE_SPECS) - {role}
            if role == "gateway_coordinator"
            else {"gateway_coordinator"}
        )
        if peer not in expected_peers:
            raise ValueError("local enclave peer topology differs")
        certificate = base64.b64decode(
            str(params.get("certificate_pem_b64") or ""),
            validate=True,
        )
        if certificate != _local_transport_certificate(peer):
            raise ValueError("local enclave peer certificate differs")
        return {"physical_role": peer, "status": "registered"}
    if method == "v2_start_tls_service" and empty_params:
        return {"status": "running"}
    if method == "v2_peer_status" and empty_params:
        return {"registered_roles": []}
    if method == "v2_call_peer_health":
        if set(params) != {"physical_role"}:
            raise ValueError("local enclave peer health fields differ")
        peer = str(params["physical_role"])
        allowed = (
            set(ROLE_SPECS) - {role}
            if role == "gateway_coordinator"
            else {"gateway_coordinator"}
        )
        if peer not in allowed:
            raise ValueError("local enclave peer health target differs")
        return {"status": "healthy", "physical_role": peer}
    if method == "v2_get_kms_recipient":
        if role != "gateway_coordinator" or set(params) != {
            "credential_slot"
        }:
            raise ValueError("local enclave KMS recipient request differs")
        slot = str(params["credential_slot"])
        credential_ref = _configured_credential_ref(role_state, slot)
        if slot != "artifact_master_key":
            objects = _gateway_runtime_objects(role, role_state)
            return _unwrap_candidate_rpc(
                objects["tee_service"].handle_v2_runtime_rpc(
                    method, dict(params)
                )
            )
        attestation = json.dumps(
            {
                "schema_version": "leadpoet.local_kms_recipient.v1",
                "credential_slot": slot,
                "credential_ref_hash": credential_ref,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return {
            "schema_version": "leadpoet.kms_recipient.v2",
            "purpose": "leadpoet.kms_recipient.v2",
            "boot_identity_hash": _local_boot_identity(
                role, config_hash
            )["boot_identity_hash"],
            "credential_slot": slot,
            "credential_ref_hash": credential_ref,
            "recipient_public_key_hash": _sha256(
                "leadpoet-local-kms-recipient"
            ),
            "request_nonce": hashlib.sha256(
                ("leadpoet-local-kms-request:" + slot).encode()
            ).hexdigest()[:32],
            "recipient_public_key_der_b64": base64.b64encode(
                b"leadpoet-local-kms-public-key"
            ).decode("ascii"),
            "attestation_document_b64": base64.b64encode(
                attestation
            ).decode("ascii"),
            "key_encryption_algorithm": "RSAES_OAEP_SHA_256",
        }
    if method == "v2_provision_encrypted_secret":
        if role != "gateway_coordinator" or set(params) != {
            "credential_slot",
            "ciphertext_for_recipient_b64",
        }:
            raise ValueError("local enclave KMS provision request differs")
        slot = str(params["credential_slot"])
        _configured_credential_ref(role_state, slot)
        if slot != "artifact_master_key":
            objects = _gateway_runtime_objects(role, role_state)
            result = _unwrap_candidate_rpc(
                objects["tee_service"].handle_v2_runtime_rpc(
                    method, dict(params)
                )
            )

            def provision_provider(state_value: dict[str, Any]) -> None:
                slots = set(state_value.get("provisioned_slots") or [])
                slots.add(slot)
                state_value["provisioned_slots"] = sorted(slots)

            _gateway_enclave_state(provision_provider)
            return result
        ciphertext = base64.b64decode(
            str(params["ciphertext_for_recipient_b64"]), validate=True
        )
        if not ciphertext.startswith(b"local-kms-recipient:"):
            raise ValueError("local enclave KMS ciphertext differs")

        def provision(state_value: dict[str, Any]) -> None:
            slots = set(state_value.get("provisioned_slots") or [])
            slots.add(slot)
            state_value["provisioned_slots"] = sorted(slots)

        state, _ = _gateway_enclave_state(provision)
        configured = list(state["provisioned_slots"])
        required = {
            "artifact_master_key",
            *expected_provider_credential_slots(),
        }
        missing = sorted(required - set(configured))
        return {
            "status": "ready" if not missing else "provisioning",
            "credential_slots": configured,
            "missing_credential_slots": missing,
        }
    if method == "v2_provider_broker_health" and empty_params:
        slots = sorted(expected_provider_credential_slots())
        return {
            "schema_version": "leadpoet.provider_broker.v2",
            "status": "ready",
            "credential_slots": slots,
            "missing_credential_slots": [],
            "inflight_count": 0,
            "terminal_count": 0,
            "job_credential_lease_count": 0,
            "registry_hash": provider_registry_hash(),
            "job_credential_slot_ref_hashes": (
                expected_job_credential_slot_ref_hashes()
            ),
            "egress_proxy": {"status": "ready"},
        }
    if method == "v2_provider_semantics_health" and empty_params:
        return {
            "schema_version": "leadpoet.provider_semantics.v2",
            "status": "ready",
            "broker_registry_hash": provider_registry_hash(),
            "cache_day": "",
            "memory_cache_entry_count": 0,
            "inflight_count": 0,
            "cost_scope_count": 0,
        }
    raise ValueError(
        f"local enclave RPC rejected unknown method for {role}: {method}"
    )


class _LocalVsock:
    def __init__(self, family: int, socket_type: int):
        if family != 40 or socket_type != socket.SOCK_STREAM:
            raise ValueError("local enclave RPC socket contract differs")
        self._role = ""
        self._validator = False
        self._response = b""
        self._offset = 0
        self._listener = False
        self._closed = False

    def settimeout(self, timeout: float) -> None:
        if float(timeout) not in {30.0, 120.0}:
            raise ValueError("local enclave RPC timeout differs")

    def connect(self, address: tuple[int, int]) -> None:
        if not isinstance(address, tuple) or len(address) != 2:
            raise ValueError("local enclave RPC address differs")
        port = int(address[1])
        if (
            os.environ.get("REHEARSAL_COMPONENT") == "validator"
            and port == 5001
        ):
            if int(address[0]) <= 0:
                raise ValueError("local validator enclave CID differs")
            self._validator = True
            return
        if port != 5000:
            raise ValueError("local enclave RPC port differs")
        self._role = _gateway_role_for_cid(int(address[0]))

    def bind(self, address: tuple[int, int]) -> None:
        if (
            not isinstance(address, tuple)
            or len(address) != 2
            or int(address[0]) != 0xFFFFFFFF
            or int(address[1]) not in {5001, 5002}
        ):
            raise ValueError("local enclave listener address differs")
        self._listener = True
        self._listener_port = int(address[1])
        _external_event(
            "nitro_enclaves",
            "enclave_listener",
            port=self._listener_port,
            status="bound",
        )

    def listen(self, backlog: int) -> None:
        expected_backlog = (
            8
            if os.environ.get("REHEARSAL_COMPONENT") == "validator"
            and self._listener_port == 5002
            else 64
        )
        if not self._listener or int(backlog) != expected_backlog:
            raise ValueError("local enclave listener backlog differs")

    def accept(self) -> tuple[Any, tuple[int, int]]:
        if not self._listener:
            raise ValueError("local enclave socket is not a listener")
        while not self._closed:
            time.sleep(1)
        raise OSError("local enclave listener closed")

    def sendall(self, payload: bytes) -> None:
        if (not self._role and not self._validator) or len(payload) < 6:
            raise ValueError("local enclave RPC request framing is invalid")
        size = int.from_bytes(payload[:4], "big")
        body = payload[4:]
        if size != len(body):
            raise ValueError("local enclave RPC request length differs")
        if self._validator:
            import leadpoet_canonical.nitro as leadpoet_nitro

            # The host verifies the enclave's returned boot document after
            # this RPC completes.  Install the strict local Nitro verifier in
            # this host process as well as in the persistent enclave process.
            leadpoet_nitro.verify_nitro_attestation_full = (
                _local_verify_nitro_attestation_full
            )
            enclave_path = Path(
                os.environ.get(
                    "REHEARSAL_VALIDATOR_ENCLAVE_SOCKET",
                    "/rehearsal-state/validator-enclave.sock",
                )
            )
            if not enclave_path.is_socket():
                raise ValueError(
                    "persistent validator enclave service is unavailable"
                )
            request = json.loads(
                __import__("validator_tee.enclave.tee_service", fromlist=["x"])
                ._decode_rpc_payload(
                    body,
                    logical_limit=64 * 1024 * 1024,
                )
            )
            method = str(request.get("command") or "")
            connection = _ORIGINAL_SOCKET(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                connection.settimeout(120)
                connection.connect(str(enclave_path))
                connection.sendall(
                    len(body).to_bytes(4, "big") + body
                )
                prefix = bytearray()
                while len(prefix) < 4:
                    chunk = connection.recv(4 - len(prefix))
                    if not chunk:
                        break
                    prefix.extend(chunk)
                if len(prefix) != 4:
                    raise ValueError(
                        "persistent validator enclave response is incomplete"
                    )
                response_size = int.from_bytes(prefix, "big")
                if response_size < 2 or response_size > 16 * 1024 * 1024:
                    raise ValueError(
                        "persistent validator enclave response size differs"
                    )
                response_frame = bytearray()
                while len(response_frame) < response_size:
                    chunk = connection.recv(
                        min(64 * 1024, response_size - len(response_frame))
                    )
                    if not chunk:
                        break
                    response_frame.extend(chunk)
                if len(response_frame) != response_size:
                    raise ValueError(
                        "persistent validator enclave response body is incomplete"
                    )
            finally:
                connection.close()
            self._response = (
                len(response_frame).to_bytes(4, "big")
                + bytes(response_frame)
            )
            self._offset = 0
            return
        else:
            request = json.loads(body)
            if set(request) != {"method", "params"}:
                raise ValueError("local enclave RPC request fields differ")
            method = str(request["method"])
            params = request["params"]
            if not isinstance(params, dict):
                raise ValueError("local enclave RPC params are not an object")
            import leadpoet_canonical.nitro as leadpoet_nitro

            leadpoet_nitro.verify_nitro_attestation_full = (
                _local_verify_nitro_attestation_full
            )
            enclave_path = _gateway_enclave_socket_path(self._role)
            if not enclave_path.is_socket():
                raise ValueError(
                    "persistent gateway enclave service is unavailable"
                )
            connection = _ORIGINAL_SOCKET(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                connection.settimeout(120)
                connection.connect(str(enclave_path))
                connection.sendall(len(body).to_bytes(4, "big") + body)
                prefix = bytearray()
                while len(prefix) < 4:
                    chunk = connection.recv(4 - len(prefix))
                    if not chunk:
                        break
                    prefix.extend(chunk)
                if len(prefix) != 4:
                    raise ValueError(
                        "persistent gateway enclave response is incomplete"
                    )
                response_size = int.from_bytes(prefix, "big")
                if response_size < 2 or response_size > 128 * 1024 * 1024:
                    raise ValueError(
                        "persistent gateway enclave response size differs"
                    )
                response_frame = bytearray()
                while len(response_frame) < response_size:
                    chunk = connection.recv(
                        min(64 * 1024, response_size - len(response_frame))
                    )
                    if not chunk:
                        break
                    response_frame.extend(chunk)
                if len(response_frame) != response_size:
                    raise ValueError(
                        "persistent gateway enclave response body is incomplete"
                    )
            finally:
                connection.close()
            service_response = json.loads(response_frame)
            if (
                not isinstance(service_response, dict)
                or set(service_response) != {"diagnostic", "response"}
                or not isinstance(service_response["response"], dict)
                or not isinstance(service_response["diagnostic"], dict)
            ):
                raise ValueError(
                    "persistent gateway enclave response envelope differs"
                )
            response = service_response["response"]
            diagnostic = service_response["diagnostic"]
            if response.get("status") == "success":
                if set(response) != {"status", "result"}:
                    raise ValueError(
                        "persistent gateway enclave success response differs"
                    )
                status = "ok"
            elif response.get("status") == "error":
                if set(response) != {"status", "error"}:
                    raise ValueError(
                        "persistent gateway enclave error response differs"
                    )
                status = "rejected"
            else:
                raise ValueError(
                    "persistent gateway enclave response status differs"
                )
            if set(diagnostic) != {"error_hash", "error_type", "status"}:
                raise ValueError(
                    "persistent gateway enclave diagnostic fields differ"
                )
            if diagnostic["status"] != status:
                raise ValueError(
                    "persistent gateway enclave diagnostic status differs"
                )
            error_type = str(diagnostic["error_type"])
            error_hash = str(diagnostic["error_hash"])
            error_message = (
                str(response.get("error") or "")
                if status == "rejected"
                else ""
            )
            if status == "ok" and (error_type or error_hash):
                raise ValueError(
                    "persistent gateway enclave success diagnostic differs"
                )
            if status == "rejected" and (
                not error_type
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", error_hash)
                or not error_message
                or _sha256(error_message) != error_hash
            ):
                raise ValueError(
                    "persistent gateway enclave failure diagnostic differs"
                )
        encoded = json.dumps(
            response, sort_keys=True, separators=(",", ":")
        ).encode()
        self._response = len(encoded).to_bytes(4, "big") + encoded
        self._offset = 0
        _external_event(
            "nitro_enclaves",
            "enclave_rpc",
            physical_role=self._role,
            method=method,
            request_bytes=len(body),
            response_bytes=len(encoded),
            status=status,
            error_type=error_type,
            error_hash=error_hash,
            error_message=error_message,
        )

    def recv(self, amount: int) -> bytes:
        result = self._response[self._offset : self._offset + amount]
        self._offset += len(result)
        return result

    def close(self) -> None:
        self._closed = True
        self._response = b""
        self._offset = 0


def _release_build_input_for_commit(commit: str) -> dict[str, Any]:
    requested = str(commit)
    if not re.fullmatch(r"[0-9a-f]{40}", requested):
        raise ValueError("local release build input commit is invalid")
    documents: list[dict[str, Any]] = []
    for path in (
        STATE_ROOT / "release-build-input.json",
        FROM_FIXTURE_SEED_ROOT / "release-build-input.json",
        DURABLE_SCHEMA_SEED_ROOT / "release-build-input.json",
    ):
        if not path.is_file():
            continue
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or not re.fullmatch(
            r"[0-9a-f]{40}",
            str(loaded.get("commit_sha") or ""),
        ):
            raise ValueError("local release build input document is invalid")
        if loaded["commit_sha"] == requested:
            documents.append(loaded)
    if not documents:
        raise ValueError("local release build input commit is unavailable")
    if any(document != documents[0] for document in documents[1:]):
        raise ValueError("local release build inputs conflict")
    return documents[0]


def _release_channel(commit: str) -> dict[str, Any]:
    from artifact_identity import (
        eif_hash,
        normalized_image_id,
        pcr0 as artifact_pcr0,
    )
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from gateway.tee.release_channel_v2 import build_release_channel_v2
    from gateway.tee.release_manifest_v2 import (
        BUILD_EVIDENCE_SCHEMA_VERSION,
        build_release_manifest,
    )
    from gateway.tee.topology import ROLE_SPECS
    from validator_tee.host.release_v2 import (
        build_validator_build_evidence,
        build_validator_release,
        build_validator_release_manifest,
    )

    pcr0 = artifact_pcr0(commit)
    validator_root = SOURCE_ROOT / "validator_tee"
    dockerfile_hash = "sha256:" + hashlib.sha256(
        (validator_root / "Dockerfile.enclave").read_bytes()
    ).hexdigest()
    base_dockerfile_hash = "sha256:" + hashlib.sha256(
        (validator_root / "Dockerfile.base").read_bytes()
    ).hexdigest()
    release_build_input = _release_build_input_for_commit(commit)
    expected_roles = release_build_input.get("gateway_roles")
    if not isinstance(expected_roles, dict):
        raise ValueError("local release build input roles are unavailable")
    gateway_rows = []
    for role, spec in sorted(ROLE_SPECS.items()):
        expected = expected_roles.get(role)
        if not isinstance(expected, dict):
            raise ValueError(
                f"local release build input is missing role: {role}"
            )
        deterministic = {
            name: expected[name]
            for name in (
                "build_identity_hash",
                "commit_sha",
                "dependency_lock_hash",
                "dockerfile_hash",
                "eif_hash",
                "execution_manifest_hash",
                "normalized_image_hash",
                "pcr0",
                "source_manifest_hash",
                "topology_hash",
            )
        }
        for domain in ("gateway", "validator"):
            for ordinal in (1, 2, 3):
                gateway_rows.append(
                    {
                        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
                        "builder_domain": domain,
                        "builder_id": f"local-{domain}-parent",
                        "build_ordinal": ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **deterministic,
                    }
                )
    acceptance_key = Ed25519PrivateKey.from_private_bytes(
        hashlib.sha256(b"leadpoet-local-acceptance-signer").digest()
    )
    acceptance_public_key = acceptance_key.public_key().public_bytes_raw()
    gateway_manifest = build_release_manifest(
        gateway_rows,
        acceptance_signer_pubkey_hash=(
            "sha256:" + hashlib.sha256(acceptance_public_key).hexdigest()
        ),
    )
    validator_release = build_validator_release(
        commit_sha=commit,
        pcr0=pcr0,
        app_manifest_hash=str(
            release_build_input["validator_app_manifest_hash"]
        ),
        dependency_lock_hash=str(
            release_build_input["validator_dependency_lock_hash"]
        ),
        normalized_image_hash=normalized_image_id(commit, "validator_weights"),
        eif_hash=eif_hash(commit, "validator_weights"),
        dockerfile_hash=dockerfile_hash,
        base_dockerfile_hash=base_dockerfile_hash,
    )
    validator_evidence = [
        build_validator_build_evidence(
            validator_release,
            builder_domain=domain,
            builder_id=f"local-{domain}-parent",
            build_ordinal=ordinal,
        )
        for domain in ("gateway", "validator")
        for ordinal in (1, 2, 3)
    ]
    return build_release_channel_v2(
        gateway_release_manifest=gateway_manifest,
        validator_release_manifest=build_validator_release_manifest(
            validator_evidence
        ),
    )


class _LocalS3:
    def _channel(self, key: str) -> tuple[str, bytes]:
        match = __import__("re").search(
            r"/([0-9a-f]{40})/release-channel-v2\.json$", key
        )
        if match is None:
            raise KeyError("local S3 received an unknown object key")
        commit = match.group(1)
        body = json.dumps(
            _release_channel(commit),
            sort_keys=True,
            separators=(",", ":"),
        ).encode() + b"\n"
        return commit, body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        private_model_identity = (str(Bucket), str(Key))
        with _PRIVATE_MODEL_OBJECTS_LOCK:
            private_model_body = _PRIVATE_MODEL_OBJECTS.get(
                private_model_identity
            )
        if private_model_body is not None:
            _external_event(
                "aws_s3_object_lock",
                "get_object",
                service="s3",
                bucket=Bucket,
                key=Key,
                private_model_artifact=True,
            )
            return {
                "Body": io.BytesIO(private_model_body),
                "ContentLength": len(private_model_body),
                "VersionId": "local-private-model-"
                + hashlib.sha256(private_model_body).hexdigest()[:24],
            }
        artifact_path = _artifact_record_path(Bucket, Key)
        if artifact_path.is_file():
            record = json.loads(artifact_path.read_text(encoding="utf-8"))
            body = base64.b64decode(record["body_b64"], validate=True)
            _external_event(
                "aws_s3_object_lock",
                "get_object",
                service="s3",
                bucket=Bucket,
                key=Key,
                artifact=True,
            )
            return {
                "Body": io.BytesIO(body),
                "ContentLength": len(body),
                "ObjectLockMode": record["object_lock_mode"],
                "ObjectLockRetainUntilDate": record["retain_until"],
                "VersionId": record["version_id"],
            }
        commit, body = self._channel(Key)
        _external_event(
            "aws_s3_object_lock",
            "get_object",
            service="s3",
            bucket=Bucket,
            key=Key,
            commit_sha=commit,
        )
        return {"Body": io.BytesIO(body)}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        private_model_identity = (str(Bucket), str(Key))
        with _PRIVATE_MODEL_OBJECTS_LOCK:
            private_model_body = _PRIVATE_MODEL_OBJECTS.get(
                private_model_identity
            )
        if private_model_body is not None:
            _external_event(
                "aws_s3_object_lock",
                "head_object",
                service="s3",
                bucket=Bucket,
                key=Key,
                private_model_artifact=True,
            )
            return {
                "ContentLength": len(private_model_body),
                "VersionId": "local-private-model-"
                + hashlib.sha256(private_model_body).hexdigest()[:24],
            }
        artifact_path = _artifact_record_path(Bucket, Key)
        if artifact_path.is_file():
            record = json.loads(artifact_path.read_text(encoding="utf-8"))
            _external_event(
                "aws_s3_object_lock",
                "head_object",
                service="s3",
                bucket=Bucket,
                key=Key,
                artifact=True,
            )
            return {
                "ContentLength": int(record["content_length"]),
                "ContentType": record["content_type"],
                "ObjectLockMode": record["object_lock_mode"],
                "ObjectLockRetainUntilDate": record["retain_until"],
                "VersionId": record["version_id"],
            }
        commit, body = self._channel(Key)
        _external_event(
            "aws_s3_object_lock",
            "head_object",
            service="s3",
            bucket=Bucket,
            key=Key,
            commit_sha=commit,
        )
        return {
            "ContentLength": len(body),
            "ObjectLockMode": "COMPLIANCE",
            "VersionId": "local-version",
        }

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        ObjectLockMode: str,
        ObjectLockRetainUntilDate: datetime,
    ) -> dict[str, Any]:
        payload = bytes(Body)
        if (
            not Bucket
            or not Key.startswith("encrypted-artifacts/")
            or ContentType != "application/json"
            or ObjectLockMode != "COMPLIANCE"
            or not isinstance(ObjectLockRetainUntilDate, datetime)
            or ObjectLockRetainUntilDate.tzinfo is None
            or not payload
        ):
            raise ValueError("local S3 artifact upload contract differs")
        retain_until = (
            ObjectLockRetainUntilDate.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        version_id = "local-" + hashlib.sha256(payload).hexdigest()[:24]
        record = {
            "bucket": Bucket,
            "key": Key,
            "body_b64": base64.b64encode(payload).decode("ascii"),
            "content_type": ContentType,
            "content_length": len(payload),
            "object_lock_mode": ObjectLockMode,
            "retain_until": retain_until,
            "version_id": version_id,
        }
        path = _artifact_record_path(Bucket, Key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != record:
                raise ValueError("local S3 immutable artifact differs")
        else:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(record, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        _external_event(
            "aws_s3_object_lock",
            "put_object",
            service="s3",
            bucket=Bucket,
            key=Key,
            content_length=len(payload),
            object_lock_mode=ObjectLockMode,
            retain_until=retain_until,
        )
        return {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "VersionId": version_id,
        }

    def generate_presigned_url(
        self,
        ClientMethod: str,
        *,
        Params: Mapping[str, str],
        ExpiresIn: int,
        HttpMethod: str,
    ) -> str:
        expected_http_method = {
            "get_object": "GET",
            "head_object": "HEAD",
        }.get(str(ClientMethod))
        if (
            expected_http_method is None
            or str(HttpMethod) != expected_http_method
            or set(Params) != {"Bucket", "Key"}
            or not 60 <= int(ExpiresIn) <= 900
        ):
            raise ValueError("local S3 presign contract differs")
        bucket = str(Params["Bucket"])
        key = str(Params["Key"])
        if not _artifact_record_path(bucket, key).is_file():
            raise ValueError("local S3 cannot presign an absent artifact")
        host = "%s.s3.us-east-1.amazonaws.com" % bucket
        query = (
            "X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=local-rehearsal"
            "%%2F20260725%%2Fus-east-1%%2Fs3%%2Faws4_request"
            "&X-Amz-Date=20260725T000000Z"
            "&X-Amz-Expires=%d"
            "&X-Amz-SignedHeaders=host"
            "&X-Amz-Signature=%s"
            % (
                int(ExpiresIn),
                hashlib.sha256(
                    (ClientMethod + "\0" + bucket + "\0" + key).encode()
                ).hexdigest(),
            )
        )
        _external_event(
            "aws_s3_object_lock",
            "presign_" + ClientMethod,
            bucket=bucket,
            key=key,
            expires_seconds=int(ExpiresIn),
        )
        return "https://%s/%s?%s" % (host, key, query)

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str,
        MaxKeys: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        commit = os.environ["REHEARSAL_CANDIDATE_SHA"]
        key = f"{Prefix.rstrip('/')}/{commit}/release-channel-v2.json"
        if MaxKeys != 1000:
            raise ValueError("local S3 received an unexpected pagination limit")
        _external_event(
            "aws_s3_object_lock",
            "get_object",
            service="s3",
            bucket=Bucket,
            prefix=Prefix,
            commit_sha=commit,
        )
        return {"Contents": [{"Key": key}], "IsTruncated": False}

    def get_bucket_versioning(self, *, Bucket: str) -> dict[str, str]:
        _external_event(
            "aws_s3_object_lock",
            "head_object",
            bucket=Bucket,
            target="bucket_versioning",
        )
        return {"Status": "Enabled"}

    def get_object_lock_configuration(
        self,
        *,
        Bucket: str,
    ) -> dict[str, Any]:
        _external_event(
            "aws_s3_object_lock",
            "head_object",
            bucket=Bucket,
            target="object_lock_configuration",
        )
        return {
            "ObjectLockConfiguration": {
                "ObjectLockEnabled": "Enabled",
                "Rule": {
                    "DefaultRetention": {
                        "Mode": "COMPLIANCE",
                        "Days": 365,
                    }
                },
            }
        }


class _LocalKMS:
    def verify(
        self,
        *,
        KeyId: str,
        Message: bytes,
        MessageType: str,
        Signature: bytes,
        SigningAlgorithm: str,
    ) -> dict[str, Any]:
        """Verify the production private-model ECDSA request contract."""

        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        if (
            str(KeyId) != _PRIVATE_MODEL_SIGNING_KEY_ID
            or str(MessageType) != "RAW"
            or str(SigningAlgorithm) != "ECDSA_SHA_256"
            or not isinstance(Message, (bytes, bytearray))
            or not re.fullmatch(
                rb"sha256:[0-9a-f]{64}", bytes(Message)
            )
            or not isinstance(Signature, (bytes, bytearray))
            or not Signature
        ):
            raise ValueError("local KMS verify contract differs")
        try:
            _private_model_signing_key().public_key().verify(
                bytes(Signature),
                bytes(Message),
                ec.ECDSA(hashes.SHA256()),
            )
        except (InvalidSignature, ValueError):
            valid = False
        else:
            valid = True
        _external_event(
            "aws_kms",
            "verify",
            key_id_hash=_sha256(str(KeyId)),
            message_hash=(
                "sha256:" + hashlib.sha256(bytes(Message)).hexdigest()
            ),
            signature_hash=(
                "sha256:" + hashlib.sha256(bytes(Signature)).hexdigest()
            ),
            signature_valid=valid,
            signing_algorithm=str(SigningAlgorithm),
        )
        return {
            "KeyId": str(KeyId),
            "SignatureValid": valid,
            "SigningAlgorithm": str(SigningAlgorithm),
        }

    def encrypt(
        self,
        *,
        KeyId: str,
        Plaintext: bytes,
        EncryptionContext: dict[str, str],
    ) -> dict[str, Any]:
        if not KeyId or not Plaintext or not EncryptionContext:
            raise ValueError("local KMS encrypt contract is incomplete")
        context = json.dumps(
            EncryptionContext,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if os.environ.get("REHEARSAL_COMPONENT") == "validator":
            ciphertext = (
                b"local-validator-kms-v2\0"
                + KeyId.encode("utf-8")
                + b"\0"
                + bytes(Plaintext)
            )
        else:
            ciphertext = (
                b"local-kms-v2\0"
                + KeyId.encode("utf-8")
                + b"\0"
                + bytes(Plaintext)
            )
        _external_event(
            "aws_kms",
            "encrypt",
            key_id_hash=_sha256(KeyId),
            encryption_context=EncryptionContext,
            plaintext_size=len(Plaintext),
        )
        return {"CiphertextBlob": ciphertext, "KeyId": KeyId}

    def decrypt(
        self,
        *,
        CiphertextBlob: bytes,
        EncryptionContext: dict[str, str],
        Recipient: dict[str, Any],
    ) -> dict[str, Any]:
        if os.environ.get("REHEARSAL_COMPONENT") == "validator":
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding

            prefix = b"local-validator-kms-v2\0"
            if (
                not isinstance(CiphertextBlob, (bytes, bytearray))
                or not bytes(CiphertextBlob).startswith(prefix)
                or not EncryptionContext
                or set(Recipient)
                != {"KeyEncryptionAlgorithm", "AttestationDocument"}
                or Recipient.get("KeyEncryptionAlgorithm")
                != "RSAES_OAEP_SHA_256"
            ):
                raise ValueError("local validator KMS decrypt contract differs")
            remainder = bytes(CiphertextBlob)[len(prefix) :]
            key_id_bytes, separator, plaintext = remainder.partition(b"\0")
            if not separator or len(plaintext) != 32:
                raise ValueError("local validator KMS ciphertext differs")
            try:
                attestation = json.loads(
                    bytes(Recipient["AttestationDocument"])
                )
                public_der = base64.b64decode(
                    attestation["public_key_b64"], validate=True
                )
                recipient_key = serialization.load_der_public_key(public_der)
                ciphertext_for_recipient = recipient_key.encrypt(
                    plaintext,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )
            except Exception as exc:
                raise ValueError(
                    "local validator recipient attestation is invalid"
                ) from exc
            key_id = key_id_bytes.decode("utf-8")
            _external_event(
                "aws_kms",
                "decrypt",
                key_id_hash=_sha256(key_id),
                encryption_context=EncryptionContext,
                recipient="validator_weights",
            )
            return {
                "CiphertextForRecipient": ciphertext_for_recipient,
                "KeyId": key_id,
            }
        gateway_prefix = b"local-kms-v2\0"
        if (
            not isinstance(CiphertextBlob, (bytes, bytearray))
            or not bytes(CiphertextBlob).startswith(gateway_prefix)
            or not EncryptionContext
            or set(Recipient) != {
                "KeyEncryptionAlgorithm",
                "AttestationDocument",
            }
            or Recipient.get("KeyEncryptionAlgorithm")
            != "RSAES_OAEP_SHA_256"
        ):
            raise ValueError("local KMS recipient decrypt contract differs")
        remainder = bytes(CiphertextBlob)[len(gateway_prefix) :]
        key_id_bytes, separator, plaintext = remainder.partition(b"\0")
        if not separator or not key_id_bytes or not plaintext:
            raise ValueError("local gateway KMS ciphertext differs")
        try:
            attestation = json.loads(
                bytes(Recipient["AttestationDocument"])
            )
        except Exception as exc:
            raise ValueError(
                "local KMS recipient attestation is invalid"
            ) from exc
        key_id = key_id_bytes.decode("utf-8")
        if attestation.get("schema_version") == (
            "leadpoet.local_nitro_document.v1"
        ):
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding

            expected_pcr0 = _gateway_release_input()["gateway_roles"][
                "gateway_coordinator"
            ]["pcr0"]
            if (
                set(attestation)
                != {
                    "schema_version",
                    "pcr0",
                    "public_key_b64",
                    "user_data_b64",
                    "nonce_b64",
                }
                or attestation.get("pcr0") != expected_pcr0
            ):
                raise ValueError("local job KMS recipient attestation differs")
            try:
                user_data = json.loads(
                    base64.b64decode(
                        attestation["user_data_b64"], validate=True
                    )
                )
                public_der = base64.b64decode(
                    attestation["public_key_b64"], validate=True
                )
                recipient_key = serialization.load_der_public_key(public_der)
                ciphertext_for_recipient = recipient_key.encrypt(
                    plaintext,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )
            except Exception as exc:
                raise ValueError(
                    "local job KMS recipient document is invalid"
                ) from exc
            expected_purposes = {
                "leadpoet.kms_recipient.v2": (
                    "leadpoet.provider_credential_unseal.v2"
                ),
                "leadpoet.kms_job_recipient.v2": (
                    "leadpoet.job_provider_credential_unseal.v2"
                ),
            }
            if (
                set(user_data) != {
                    "schema_version",
                    "purpose",
                    "claim_hash",
                }
                or expected_purposes.get(user_data.get("schema_version"))
                != user_data.get("purpose")
                or not re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(user_data.get("claim_hash") or ""),
                )
            ):
                raise ValueError("local KMS recipient claim differs")
            _external_event(
                "aws_kms",
                "decrypt",
                key_id_hash=_sha256(key_id),
                encryption_context=EncryptionContext,
                recipient="gateway_job_credential",
            )
            return {
                "CiphertextForRecipient": ciphertext_for_recipient,
                "KeyId": key_id,
            }
        if set(attestation) != {
            "schema_version",
            "credential_ref_hash",
            "credential_slot",
        } or attestation.get("schema_version") != (
            "leadpoet.local_kms_recipient.v1"
        ):
            raise ValueError("local KMS recipient attestation fields differ")
        ciphertext = b"local-kms-recipient:" + hashlib.sha256(
            bytes(CiphertextBlob)
            + b"\0"
            + json.dumps(
                EncryptionContext,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\0"
            + bytes(Recipient["AttestationDocument"])
        ).digest()
        _external_event(
            "aws_kms",
            "decrypt",
            key_id_hash=_sha256(key_id),
            encryption_context=EncryptionContext,
            credential_slot=attestation["credential_slot"],
        )
        return {
            "CiphertextForRecipient": ciphertext,
            "KeyId": key_id,
        }


class _LocalHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body
        self._offset = 0

    def __enter__(self) -> "_LocalHTTPResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        del args

    def getcode(self) -> int:
        return 200

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            result = self._body[self._offset :]
            self._offset = len(self._body)
            return result
        result = self._body[self._offset : self._offset + amount]
        self._offset += len(result)
        return result


def _local_urlopen(
    request: Any,
    *,
    timeout: Optional[float] = None,
) -> _LocalHTTPResponse:
    url = str(getattr(request, "full_url", request))
    parsed = urlparse(url)
    if url == "https://ident.me":
        if timeout is not None:
            raise ValueError("local external-IP timeout differs")
        _external_event(
            "http_service",
            "external_ip",
            method="GET",
            hostname="ident.me",
        )
        return _LocalHTTPResponse(b"203.0.113.10")
    if parsed.hostname == "169.254.169.254":
        if parsed.path == "/latest/api/token" and request.get_method() == "PUT":
            body = b"local-imds-token"
        elif (
            parsed.path == "/latest/meta-data/instance-type"
            and request.get_header("X-aws-ec2-metadata-token")
            == "local-imds-token"
        ):
            body = b"r7i.4xlarge"
        else:
            raise ValueError("local IMDS received an unknown request")
        _external_event(
            "host_kernel",
            "instance_metadata",
            method=request.get_method(),
            path=parsed.path,
        )
        return _LocalHTTPResponse(body)
    if VALIDATOR_RUNTIME_LOCK_PATH.is_file():
        runtime_lock = json.loads(
            VALIDATOR_RUNTIME_LOCK_PATH.read_text(encoding="utf-8")
        )
        artifacts = runtime_lock.get("artifacts", {})
    else:
        artifacts = {}
    for name, artifact in artifacts.items():
        if url != artifact.get("url"):
            continue
        path = EXTERNAL_ARTIFACT_ROOT / str(artifact.get("filename") or "")
        if not path.is_file():
            raise ValueError(
                f"local validator runtime artifact is unavailable: {name}"
            )
        body = path.read_bytes()
        observed_sha256 = hashlib.sha256(body).hexdigest()
        if observed_sha256 != artifact.get("sha256"):
            raise ValueError(
                f"local validator runtime artifact hash differs: {name}"
            )
        if timeout != 120:
            raise ValueError("local validator runtime artifact timeout differs")
        _external_event(
            "http_service",
            "download",
            artifact=name,
            artifact_sha256="sha256:" + observed_sha256,
        )
        return _LocalHTTPResponse(body)
    if parsed.hostname in {"127.0.0.1", "localhost"}:
        if parsed.port == 54321:
            return _real_urlopen(request, timeout=timeout)
        if parsed.port == 8000:
            method = str(
                getattr(request, "method", None) or request.get_method()
            ).upper()
            headers = {
                str(name).lower(): str(value)
                for name, value in request.header_items()
            }
            if (
                method != "GET"
                or not re.fullmatch(
                    r"/research-lab/allocations/attested/[0-9]+",
                    parsed.path,
                )
                or headers.get("x-leadpoet-internal-key")
                != "rehearsal-internal"
                or timeout is None
                or not 0 < float(timeout) <= 360
            ):
                raise ValueError(
                    "local gateway allocation handoff contract differs"
                )
            _external_event(
                "http_service",
                "gateway_request",
                method=method,
                path=parsed.path,
                authenticated=True,
            )
            return _real_urlopen(request, timeout=timeout)
    if parsed.hostname == "gateway.invalid" and parsed.scheme == "http":
        if timeout != 30:
            raise ValueError("local gateway authority timeout differs")
        if parsed.path == "/health/v2-authority":
            body = json.dumps(
                {
                    "schema_version": (
                        "leadpoet.gateway_v2_authority_health.v2"
                    ),
                    "status": "ready",
                    "commit_sha": os.environ["REHEARSAL_CANDIDATE_SHA"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        elif parsed.path == "/build-info":
            body = json.dumps(
                {"git_commit": os.environ["REHEARSAL_CANDIDATE_SHA"]},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        else:
            raise ValueError("local gateway received an unknown path")
        _external_event(
            "http_service",
            "gateway_request",
            method=str(getattr(request, "method", None) or "GET"),
            path=parsed.path,
        )
        return _LocalHTTPResponse(body)
    if parsed.scheme != "https" or parsed.hostname != "example.invalid":
        raise ValueError("local PostgREST received an unknown URL")
    headers = {
        str(name).lower(): str(value)
        for name, value in request.header_items()
    }
    if (
        headers.get("authorization") != "Bearer rehearsal-secret"
        or headers.get("apikey") != "rehearsal-secret"
        or timeout != 10.0
    ):
        raise ValueError("local PostgREST authentication contract differs")
    if parsed.path == "/rest/v1/":
        from gateway.tee.supabase_schema_preflight_v2 import (
            REQUIRED_SUPABASE_V2_RPCS,
        )

        body = json.dumps(
            {
                "paths": {
                    f"/rpc/{name}": {}
                    for _migration, name in REQUIRED_SUPABASE_V2_RPCS
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        operation = "rpc"
    elif (
        parsed.path
        == "/rest/v1/rpc/research_lab_compact_weight_settlement_contract_v1"
    ):
        if str(getattr(request, "method", None) or "GET").upper() != "POST":
            raise ValueError(
                "compact weight settlement contract method differs"
            )
        request_body = getattr(request, "data", None)
        if request_body not in {b"{}", None}:
            raise ValueError(
                "compact weight settlement contract body differs"
            )
        body = json.dumps(
            {
                "schema_version": (
                    "leadpoet.research_lab_compact_weight_settlement_contract.v1"
                ),
                "max_authority_bytes": 8_388_608,
                "size_constraint_valid": True,
                "append_only_trigger_enabled": True,
                "identity_unique_constraint_enabled": True,
                "row_level_security_enabled": True,
                "finalized_stage_supported": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        operation = "rpc"
    elif (
        parsed.path
        == "/rest/v1/research_lab_chain_realized_settlement_activation_v1"
        and "limit=2" in parsed.query
    ):
        activation_epoch = _current_settlement_epoch_id() - 1
        body = json.dumps(
            [
                {
                    "netuid": 71,
                    "schema_version": (
                        "leadpoet.research_lab_chain_realized_settlement_activation.v1"
                    ),
                    "first_epoch_id": activation_epoch,
                    "source_bundle_hash": "sha256:" + "a" * 64,
                    "source_bundle_epoch_id": activation_epoch,
                    "source_finalized_block": CURRENT_BLOCK - 1,
                }
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        operation = "select"
    elif parsed.path.startswith("/rest/v1/") and "limit=0" in parsed.query:
        body = b"[]"
        operation = "select"
    else:
        raise ValueError("local PostgREST received an unknown schema probe")
    _external_event(
        "supabase_postgrest",
        operation,
        method=str(getattr(request, "method", None) or "GET"),
        path=parsed.path,
        query=parsed.query,
    )
    return _LocalHTTPResponse(body)


if os.environ.get("REHEARSAL_SCOPE") == "exact":
    _real_open = builtins.open
    _real_path_read_text = Path.read_text
    try:
        _rehearsal_topology_module_path = (
            SOURCE_ROOT / "gateway/tee/topology.py"
        )
        if not _rehearsal_topology_module_path.is_file():
            raise FileNotFoundError("candidate topology module is unavailable")
        _rehearsal_topology_spec = importlib.util.spec_from_file_location(
            "_leadpoet_rehearsal_candidate_topology",
            _rehearsal_topology_module_path,
        )
        if (
            _rehearsal_topology_spec is None
            or _rehearsal_topology_spec.loader is None
        ):
            raise RuntimeError("candidate topology module cannot be loaded")
        _rehearsal_topology_module = importlib.util.module_from_spec(
            _rehearsal_topology_spec
        )
        _rehearsal_topology_spec.loader.exec_module(
            _rehearsal_topology_module
        )
        _validate_topology_manifest = getattr(
            _rehearsal_topology_module,
            "validate_manifest",
            None,
        )
        if not callable(_validate_topology_manifest):
            raise RuntimeError("candidate topology validator is unavailable")
        _rehearsal_topology_path = SOURCE_ROOT / "gateway/tee/topology.json"
        _rehearsal_topology = _validate_topology_manifest(
            json.loads(
                _real_path_read_text(
                    _rehearsal_topology_path,
                    encoding="utf-8",
                )
            )
        )
        _rehearsal_host_reserved_memory_mib = int(
            _rehearsal_topology["host_reserved_memory_mib"]
        )
        if _rehearsal_host_reserved_memory_mib <= 0:
            raise ValueError(
                "candidate topology host memory reservation is invalid"
            )
    except Exception as exc:
        raise SystemExit(
            "exact rehearsal candidate topology bootstrap failed"
        ) from exc

    import boto3
    import bittensor
    import urllib.request
    from bittensor.core import subtensor as bittensor_subtensor

    _REAL_SUBTENSOR_CLASS = bittensor.Subtensor
    bittensor.Subtensor = _LocalSubtensor
    bittensor.AsyncSubtensor = _LocalAsyncSubtensor
    bittensor.Metagraph = _LocalMetagraph
    bittensor_subtensor.Subtensor = _LocalSubtensor
    _real_boto3_client = boto3.client
    _real_socket = _ORIGINAL_SOCKET
    _real_getaddrinfo = _ORIGINAL_GETADDRINFO
    _real_sysconf = os.sysconf
    _real_urlopen = urllib.request.urlopen
    _rehearsal_proxy_hosts = {
        "autoresearch-proxy.example.com",
        "scoring-proxy.example.com",
    }
    _rehearsal_proxy_address = "93.184.216.34"
    _rehearsal_proxy_port = 18443

    def _local_boto3_client(service_name: str, *args: Any, **kwargs: Any) -> Any:
        if service_name == "s3":
            return _LocalS3()
        if service_name == "kms":
            return _LocalKMS()
        return _real_boto3_client(service_name, *args, **kwargs)

    boto3.client = _local_boto3_client
    urllib.request.urlopen = _local_urlopen

    class _RehearsalSocket(_real_socket):
        """Type-compatible socket factory with a strict AF_VSOCK boundary."""

        def __new__(
            cls,
            family: int = -1,
            type: int = -1,
            proto: int = -1,
            fileno: Any = None,
        ) -> Any:
            if family == 40:
                if proto not in (0, -1) or fileno is not None:
                    raise ValueError(
                        "local enclave RPC socket options differ"
                    )
                return _LocalVsock(family, type)
            return super().__new__(cls, family, type, proto, fileno)

        def connect(self, address: Any) -> Any:
            if (
                isinstance(address, tuple)
                and len(address) >= 2
                and str(address[0]) == _rehearsal_proxy_address
                and int(address[1]) == 443
            ):
                _external_event(
                    "http_service",
                    "proxy_tls_connect",
                    destination_port=443,
                )
                return super().connect(("127.0.0.1", _rehearsal_proxy_port))
            return super().connect(address)

    def _local_getaddrinfo(
        host: Any,
        port: Any,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> Any:
        normalized_host = str(host or "").strip().lower().rstrip(".")
        if normalized_host in _rehearsal_proxy_hosts and int(port) == 443:
            _external_event(
                "http_service",
                "proxy_dns",
                destination_port=443,
            )
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    (_rehearsal_proxy_address, 443),
                )
            ]
        return _real_getaddrinfo(host, port, family, type, proto, flags)

    def _local_sysconf(name: Any) -> int:
        if name in {"SC_NPROCESSORS_CONF", os.sysconf_names["SC_NPROCESSORS_CONF"]}:
            _external_event("host_kernel", "cpu_capacity", configured_cpus=16)
            return 16
        return _real_sysconf(name)

    def _local_meminfo_text(*, read_interface: str) -> str:
        memory_kib = _rehearsal_host_reserved_memory_mib * 1024
        _external_event(
            "host_kernel",
            "memory_capacity",
            read_interface=read_interface,
            memory_mib=_rehearsal_host_reserved_memory_mib,
            available_memory_mib=_rehearsal_host_reserved_memory_mib,
            capacity_source="candidate_topology.host_reserved_memory_mib",
            topology_hash=str(_rehearsal_topology["topology_hash"]),
            topology_path="gateway/tee/topology.json",
        )
        return (
            f"MemTotal:       {memory_kib} kB\n"
            f"MemAvailable:   {memory_kib} kB\n"
        )

    def _local_open(
        file: Any,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            path = os.fspath(file)
        except TypeError:
            return _real_open(file, mode, *args, **kwargs)
        if path != "/proc/meminfo":
            return _real_open(file, mode, *args, **kwargs)
        if mode not in {"r", "rt", "rb"}:
            raise ValueError("local /proc/meminfo boundary is read-only")
        value = _local_meminfo_text(read_interface="builtins.open")
        if mode == "rb":
            if any(name in kwargs for name in ("encoding", "errors", "newline")):
                raise ValueError("binary mode doesn't take an encoding argument")
            return io.BytesIO(value.encode("utf-8"))
        return io.StringIO(value, newline=kwargs.get("newline"))

    def _local_path_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if str(path) == "/proc/meminfo":
            return _local_meminfo_text(read_interface="pathlib.Path.read_text")
        return _real_path_read_text(path, *args, **kwargs)

    os.sysconf = _local_sysconf
    builtins.open = _local_open
    Path.read_text = _local_path_read_text
    socket.getaddrinfo = _local_getaddrinfo
    socket.socket = _RehearsalSocket
