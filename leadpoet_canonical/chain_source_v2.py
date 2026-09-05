"""Dependency-free canonical Bittensor chain-source helpers for V2.

Only the finalized block header and the selective metagraph fields consumed by
the unchanged weight formula are decoded.  Keeping this module in the shared
canonical package lets the validator enclave and offline auditors validate the
same bytes without importing Bittensor, substrate-interface, or a SCALE codec.
"""

from __future__ import annotations

import hashlib
import base64
import gzip
import json
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json
from leadpoet_canonical.production_parity_boundary_v2 import (
    PRODUCTION_CHAIN_ARCHIVE_HOST,
    PRODUCTION_CHAIN_HOST,
    configured_chain_source_boundary_v2,
)


CHAIN_SOURCE_SCHEMA_VERSION = "leadpoet.bittensor_chain_source.v2"
_CONFIGURED_CHAIN_BOUNDARY = configured_chain_source_boundary_v2()
CHAIN_ENDPOINT_HOST = _CONFIGURED_CHAIN_BOUNDARY["chain_host"]
CHAIN_ARCHIVE_ENDPOINT_HOST = _CONFIGURED_CHAIN_BOUNDARY["chain_archive_host"]
CHAIN_ENDPOINT_PORT = 443
CHAIN_ENDPOINT_PATH = "/"
CHAIN_RPC_METHOD = "SubnetInfoRuntimeApi_get_selective_mechagraph"
CHAIN_SELECTIVE_FIELDS = (0, 5, 7, 52)
CHAIN_SELECTIVE_RESULT_LAST_FIELDS = (73, 76)
CHAIN_SS58_FORMAT = 42
CHAIN_FINALIZATION_EPOCH_BLOCKS = 360
CHAIN_SUBTENSOR_MAX_TEMPO = 50_400
CHAIN_MAX_HOTKEYS = 4096
CHAIN_MAX_RPC_RESPONSE_BYTES = 8 * 1024 * 1024
CHAIN_MAX_RUNTIME_METADATA_BYTES = 1024 * 1024
CHAIN_RPC_TIMEOUT_MS = 30_000
CHAIN_RPC_RETRY_BACKOFF_SECONDS = (1.0, 3.0)
CHAIN_MAX_FINALIZATION_SCAN_BLOCKS = 96
CHAIN_MAX_BLOCK_EXTRINSICS = 8192
CHAIN_MAX_CHECKPOINT_EVENTS = 15_000
CHAIN_MAX_CHECKPOINT_COMPRESSED_BYTES = 64 * 1024 * 1024
CHAIN_MAX_CHECKPOINT_DECOMPRESSED_BYTES = 128 * 1024 * 1024

# ``RevealPeriodEpochs`` is a ValueQuery. A missing map entry therefore means
# the SCALE metadata default, not an absent chain value. Keep that default
# bound to the exact authenticated runtime metadata that was reviewed. An
# unknown runtime fails closed until its metadata is added here.
_REVEAL_PERIOD_METADATA_DEFAULTS_V2 = {
    (
        "2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03",
        452,
        1,
        "sha256:79fc9235a87651a0cd5b93856d4b5696ffb8a0bd26c6f30a1f1402ac8aaad195",
    ): 1,
    (
        "2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03",
        453,
        1,
        "sha256:99380e7d01eccc41ffa1304e782658c86b38ba9986acefa371e79ad367f76658",
    ): 1,
    (
        "2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03",
        454,
        1,
        "sha256:b592bafacd0f3cce1340a91f237f82a531968bd833cbd27339328c80ce92b1cf",
    ): 1,
}

_RAW_HASH_RE = re.compile(r"^(?:0x)?[0-9a-f]{64}$")
_BASE58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


class ChainSourceV2Error(ValueError):
    """A finalized chain response is malformed or outside the measured policy."""


def configure_chain_source_boundary_v2(
    *, chain_host: str, chain_archive_host: str
) -> Dict[str, str]:
    """Configure the enclave boundary once before importing chain consumers."""

    global CHAIN_ENDPOINT_HOST, CHAIN_ARCHIVE_ENDPOINT_HOST
    requested = {
        "chain_host": str(chain_host or "").strip().lower(),
        "chain_archive_host": str(chain_archive_host or "").strip().lower(),
    }
    allowed = {
        (PRODUCTION_CHAIN_HOST, PRODUCTION_CHAIN_ARCHIVE_HOST),
        ("test.finney.opentensor.ai", "test.finney.opentensor.ai"),
    }
    if (requested["chain_host"], requested["chain_archive_host"]) not in allowed:
        raise ChainSourceV2Error("chain source boundary is outside measured policy")
    current = (CHAIN_ENDPOINT_HOST, CHAIN_ARCHIVE_ENDPOINT_HOST)
    target = (requested["chain_host"], requested["chain_archive_host"])
    if current != (PRODUCTION_CHAIN_HOST, PRODUCTION_CHAIN_ARCHIVE_HOST) and current != target:
        raise ChainSourceV2Error("chain source boundary is already configured")
    CHAIN_ENDPOINT_HOST, CHAIN_ARCHIVE_ENDPOINT_HOST = target
    return dict(requested)


def chain_source_policy_document(
    *, chain_host: Optional[str] = None, chain_archive_host: Optional[str] = None
) -> Dict[str, Any]:
    live_host = str(chain_host or CHAIN_ENDPOINT_HOST)
    archive_host = str(chain_archive_host or CHAIN_ARCHIVE_ENDPOINT_HOST)
    return {
        "schema_version": CHAIN_SOURCE_SCHEMA_VERSION,
        "host": live_host,
        "archive_host": archive_host,
        "port": CHAIN_ENDPOINT_PORT,
        "path": CHAIN_ENDPOINT_PATH,
        "tls_terminates_in_enclave": True,
        "plaintext_allowed": False,
        "rpc_methods": [
            "chain_getFinalizedHead",
            "chain_getBlockHash",
            "chain_getBlock",
            "chain_getHeader",
            "state_getRuntimeVersion",
            "state_getMetadata",
            "state_getStorage",
            "state_getStorageHash",
            "state_call",
        ],
        "runtime_method": CHAIN_RPC_METHOD,
        "selective_fields": list(CHAIN_SELECTIVE_FIELDS),
        "selective_result_last_fields": list(
            CHAIN_SELECTIVE_RESULT_LAST_FIELDS
        ),
        "ss58_format": CHAIN_SS58_FORMAT,
        "max_hotkeys": CHAIN_MAX_HOTKEYS,
        "max_response_bytes": CHAIN_MAX_RPC_RESPONSE_BYTES,
        "max_finalization_scan_blocks": CHAIN_MAX_FINALIZATION_SCAN_BLOCKS,
        "max_block_extrinsics": CHAIN_MAX_BLOCK_EXTRINSICS,
        "timeout_ms": CHAIN_RPC_TIMEOUT_MS,
        "retry_backoff_seconds": list(CHAIN_RPC_RETRY_BACKOFF_SECONDS),
    }


def chain_source_policy_hash(
    *, chain_host: Optional[str] = None, chain_archive_host: Optional[str] = None
) -> str:
    return sha256_json(
        chain_source_policy_document(
            chain_host=chain_host,
            chain_archive_host=chain_archive_host,
        )
    )


def chain_source_boundary_for_profile_v2(
    profile: Mapping[str, Any],
) -> Dict[str, str]:
    """Bind one measured signing profile to its reviewed live/archive hosts."""

    from leadpoet_canonical.hotkey_authority_v2 import (
        validate_chain_signing_profile,
    )

    normalized = validate_chain_signing_profile(profile)
    endpoint = urlsplit(str(normalized["chain_endpoint"]))
    host = str(endpoint.hostname or "").lower()
    network = str(normalized["network"])
    if endpoint.scheme != "wss" or endpoint.port not in (None, 443):
        raise ChainSourceV2Error("chain signing endpoint is outside measured policy")
    if network == "finney" and host == PRODUCTION_CHAIN_HOST:
        archive_host = PRODUCTION_CHAIN_ARCHIVE_HOST
    elif network == "test" and host == "test.finney.opentensor.ai":
        archive_host = host
    else:
        raise ChainSourceV2Error("chain signing profile has no measured source boundary")
    return {
        "chain_host": host,
        "chain_archive_host": archive_host,
        "chain_source_policy_hash": chain_source_policy_hash(
            chain_host=host,
            chain_archive_host=archive_host,
        ),
        "chain_signing_profile_hash": sha256_json(normalized),
    }


def _compact_encode(value: int) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChainSourceV2Error("SCALE compact value is invalid")
    if value < 1 << 6:
        return bytes((value << 2,))
    if value < 1 << 14:
        return ((value << 2) | 1).to_bytes(2, "little")
    if value < 1 << 30:
        return ((value << 2) | 2).to_bytes(4, "little")
    length = max(4, (value.bit_length() + 7) // 8)
    if length > 67:
        raise ChainSourceV2Error("SCALE compact value is too large")
    return bytes((((length - 4) << 2) | 3,)) + value.to_bytes(length, "little")


def _compact_decode(data: bytes, offset: int) -> Tuple[int, int]:
    if offset < 0 or offset >= len(data):
        raise ChainSourceV2Error("SCALE compact value is truncated")
    first = data[offset]
    mode = first & 3
    if mode == 0:
        return first >> 2, offset + 1
    if mode == 1:
        end = offset + 2
        if end > len(data):
            raise ChainSourceV2Error("SCALE compact value is truncated")
        return int.from_bytes(data[offset:end], "little") >> 2, end
    if mode == 2:
        end = offset + 4
        if end > len(data):
            raise ChainSourceV2Error("SCALE compact value is truncated")
        return int.from_bytes(data[offset:end], "little") >> 2, end
    length = (first >> 2) + 4
    end = offset + 1 + length
    if end > len(data):
        raise ChainSourceV2Error("SCALE compact value is truncated")
    return int.from_bytes(data[offset + 1 : end], "little"), end


def _rotate_left_64(value: int, bits: int) -> int:
    mask = (1 << 64) - 1
    return ((value << bits) | (value >> (64 - bits))) & mask


def xxhash64(value: bytes, *, seed: int = 0) -> int:
    """Dependency-free xxHash64 used by Substrate Twox storage hashers."""

    data = bytes(value)
    mask = (1 << 64) - 1
    p1 = 11400714785074694791
    p2 = 14029467366897019727
    p3 = 1609587929392839161
    p4 = 9650029242287828579
    p5 = 2870177450012600261

    def round_value(accumulator: int, lane: int) -> int:
        accumulator = (accumulator + lane * p2) & mask
        accumulator = _rotate_left_64(accumulator, 31)
        return (accumulator * p1) & mask

    offset = 0
    if len(data) >= 32:
        v1 = (seed + p1 + p2) & mask
        v2 = (seed + p2) & mask
        v3 = seed & mask
        v4 = (seed - p1) & mask
        while offset <= len(data) - 32:
            v1 = round_value(v1, int.from_bytes(data[offset : offset + 8], "little"))
            v2 = round_value(v2, int.from_bytes(data[offset + 8 : offset + 16], "little"))
            v3 = round_value(v3, int.from_bytes(data[offset + 16 : offset + 24], "little"))
            v4 = round_value(v4, int.from_bytes(data[offset + 24 : offset + 32], "little"))
            offset += 32
        result = (
            _rotate_left_64(v1, 1)
            + _rotate_left_64(v2, 7)
            + _rotate_left_64(v3, 12)
            + _rotate_left_64(v4, 18)
        ) & mask
        for lane in (v1, v2, v3, v4):
            result ^= round_value(0, lane)
            result = (result * p1 + p4) & mask
    else:
        result = (seed + p5) & mask
    result = (result + len(data)) & mask
    while offset <= len(data) - 8:
        lane = round_value(0, int.from_bytes(data[offset : offset + 8], "little"))
        result ^= lane
        result &= mask
        result = (_rotate_left_64(result, 27) * p1 + p4) & mask
        offset += 8
    if offset <= len(data) - 4:
        result ^= (int.from_bytes(data[offset : offset + 4], "little") * p1) & mask
        result &= mask
        result = (_rotate_left_64(result, 23) * p2 + p3) & mask
        offset += 4
    while offset < len(data):
        result ^= data[offset] * p5
        result &= mask
        result = (_rotate_left_64(result, 11) * p1) & mask
        offset += 1
    result ^= result >> 33
    result = (result * p2) & mask
    result ^= result >> 29
    result = (result * p3) & mask
    result ^= result >> 32
    return result & mask


def _twox128(value: bytes) -> bytes:
    raw = bytes(value)
    return xxhash64(raw, seed=0).to_bytes(8, "little") + xxhash64(
        raw, seed=1
    ).to_bytes(8, "little")


def _twox64_concat(value: bytes) -> bytes:
    raw = bytes(value)
    return xxhash64(raw, seed=0).to_bytes(8, "little") + raw


_SUBNET_EPOCH_STORAGE_WIDTHS = {
    "Tempo": 2,
    "LastEpochBlock": 8,
    "PendingEpochAt": 8,
    "SubnetEpochIndex": 8,
    "BlocksSinceLastStep": 8,
}


def subnet_epoch_storage_key(*, storage_name: str, netuid: int) -> str:
    """Build an exact Identity-map key for Subtensor epoch scheduler state."""

    normalized_name = str(storage_name or "")
    normalized_netuid = int(netuid)
    if normalized_name not in _SUBNET_EPOCH_STORAGE_WIDTHS:
        raise ChainSourceV2Error("subnet epoch storage name is invalid")
    if not 0 <= normalized_netuid <= 0xFFFF:
        raise ChainSourceV2Error("subnet epoch storage netuid is invalid")
    key = b"".join(
        (
            _twox128(b"SubtensorModule"),
            _twox128(normalized_name.encode("ascii")),
            normalized_netuid.to_bytes(2, "little"),
        )
    )
    return "0x" + key.hex()


def reveal_period_epochs_storage_key(*, netuid: int) -> str:
    """Build the exact Twox64Concat key for subnet reveal-period state."""

    normalized_netuid = int(netuid)
    if not 0 <= normalized_netuid <= 0xFFFF:
        raise ChainSourceV2Error("subnet reveal-period netuid is invalid")
    key = b"".join(
        (
            _twox128(b"SubtensorModule"),
            _twox128(b"RevealPeriodEpochs"),
            _twox64_concat(normalized_netuid.to_bytes(2, "little")),
        )
    )
    return "0x" + key.hex()


def decode_reveal_period_epochs_storage(value: Any) -> Optional[int]:
    """Decode an optional SCALE u64 reveal-period storage override."""

    if value is None:
        return None
    text = str(value or "")
    if not text.startswith("0x"):
        raise ChainSourceV2Error("subnet reveal-period storage is invalid")
    try:
        raw = bytes.fromhex(text[2:])
    except ValueError as exc:
        raise ChainSourceV2Error(
            "subnet reveal-period storage is invalid"
        ) from exc
    if len(raw) != 8:
        raise ChainSourceV2Error("subnet reveal-period storage width is invalid")
    result = int.from_bytes(raw, "little")
    if result <= 0:
        raise ChainSourceV2Error("subnet reveal-period storage is invalid")
    return result


def decode_runtime_metadata_commitment(value: Any) -> Dict[str, Any]:
    """Hash one bounded SCALE runtime-metadata response without retaining it."""

    text = str(value or "")
    if not text.startswith("0x") or len(text) % 2:
        raise ChainSourceV2Error("runtime metadata is invalid")
    try:
        raw = bytes.fromhex(text[2:])
    except ValueError as exc:
        raise ChainSourceV2Error("runtime metadata is invalid hex") from exc
    if (
        len(raw) < 5
        or len(raw) > CHAIN_MAX_RUNTIME_METADATA_BYTES
        or raw[:4] != b"meta"
        or raw[4] not in (14, 15)
    ):
        raise ChainSourceV2Error("runtime metadata is outside measured policy")
    return {
        "metadata_hash": sha256_bytes(raw),
        "metadata_version": int(raw[4]),
        "metadata_bytes": len(raw),
    }


def resolve_reveal_period_metadata_default_v2(
    *,
    genesis_hash: Any,
    runtime_spec_version: Any,
    runtime_transaction_version: Any,
    metadata_hash: Any,
) -> int:
    """Resolve the reviewed ValueQuery default for one exact runtime."""

    genesis = str(genesis_hash or "").lower()
    if genesis.startswith("0x"):
        genesis = genesis[2:]
    digest = str(metadata_hash or "").lower()
    if (
        not _RAW_HASH_RE.fullmatch(genesis)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        or isinstance(runtime_spec_version, bool)
        or isinstance(runtime_transaction_version, bool)
    ):
        raise ChainSourceV2Error(
            "reveal-period metadata authority is invalid"
        )
    try:
        spec_version = int(runtime_spec_version)
        transaction_version = int(runtime_transaction_version)
    except (TypeError, ValueError) as exc:
        raise ChainSourceV2Error(
            "reveal-period metadata authority is invalid"
        ) from exc
    result = _REVEAL_PERIOD_METADATA_DEFAULTS_V2.get(
        (genesis, spec_version, transaction_version, digest)
    )
    if result is None:
        raise ChainSourceV2Error(
            "reveal-period metadata authority is not reviewed"
        )
    return int(result)


def decode_subnet_epoch_storage(value: Any, *, storage_name: str) -> int:
    """Decode one fixed-width SCALE scheduler storage value."""

    normalized_name = str(storage_name or "")
    width = _SUBNET_EPOCH_STORAGE_WIDTHS.get(normalized_name)
    text = str(value or "")
    if width is None or not text.startswith("0x"):
        raise ChainSourceV2Error("subnet epoch storage value is invalid")
    try:
        raw = bytes.fromhex(text[2:])
    except ValueError as exc:
        raise ChainSourceV2Error("subnet epoch storage value is invalid") from exc
    if len(raw) != width:
        raise ChainSourceV2Error("subnet epoch storage width is invalid")
    return int.from_bytes(raw, "little")


def timestamp_now_storage_key() -> str:
    """Build the exact plain storage key for ``Timestamp.Now``."""

    return "0x" + (_twox128(b"Timestamp") + _twox128(b"Now")).hex()


def system_events_storage_key() -> str:
    """Build the exact plain storage key for ``System.Events``."""

    return "0x" + (_twox128(b"System") + _twox128(b"Events")).hex()


def system_event_count_storage_key() -> str:
    """Build the exact plain storage key for ``System.EventCount``."""

    return "0x" + (_twox128(b"System") + _twox128(b"EventCount")).hex()


def decode_timestamp_now_storage(value: Any) -> int:
    """Decode ``Timestamp.Now`` as milliseconds since the Unix epoch."""

    text = str(value or "")
    if not text.startswith("0x"):
        raise ChainSourceV2Error("timestamp storage value is invalid")
    try:
        raw = bytes.fromhex(text[2:])
    except ValueError as exc:
        raise ChainSourceV2Error("timestamp storage value is invalid") from exc
    if len(raw) != 8:
        raise ChainSourceV2Error("timestamp storage width is invalid")
    return int.from_bytes(raw, "little")


def parse_runtime_version(value: Any) -> Dict[str, int]:
    """Validate the signing-relevant fields from state_getRuntimeVersion."""

    if not isinstance(value, Mapping):
        raise ChainSourceV2Error("runtime version response is invalid")
    result = {}
    for source, target in (
        ("specVersion", "spec_version"),
        ("transactionVersion", "transaction_version"),
    ):
        observed = value.get(source)
        if (
            not isinstance(observed, int)
            or isinstance(observed, bool)
            or not 0 <= observed < (1 << 32)
        ):
            raise ChainSourceV2Error(
                "runtime version %s is invalid" % source
            )
        result[target] = int(observed)
    return result


def timelocked_weight_commits_storage_key(
    *, netuid: int, subnet_epoch_index: int
) -> str:
    normalized_netuid = int(netuid)
    normalized_subnet_epoch_index = int(subnet_epoch_index)
    if (
        not 0 <= normalized_netuid <= 0xFFFF
        or not 0 <= normalized_subnet_epoch_index < 1 << 64
    ):
        raise ChainSourceV2Error("timelocked weight storage key input is invalid")
    key = b"".join(
        (
            _twox128(b"SubtensorModule"),
            _twox128(b"TimelockedWeightCommits"),
            _twox64_concat(normalized_netuid.to_bytes(2, "little")),
            _twox64_concat(normalized_subnet_epoch_index.to_bytes(8, "little")),
        )
    )
    return "0x" + key.hex()


def weights_storage_key(*, netuid: int, validator_uid: int) -> str:
    """Build the exact ``SubtensorModule.Weights`` double-map key.

    Live Finney metadata defines both map hashers as ``Identity`` and both keys
    as ``u16``.  Keeping this codec dependency-free lets the measured
    coordinator verify historical weight state without importing Bittensor.
    """

    normalized_netuid = int(netuid)
    normalized_uid = int(validator_uid)
    if not 0 <= normalized_netuid <= 0xFFFF or not 0 <= normalized_uid <= 0xFFFF:
        raise ChainSourceV2Error("weight storage key input is invalid")
    key = b"".join(
        (
            _twox128(b"SubtensorModule"),
            _twox128(b"Weights"),
            normalized_netuid.to_bytes(2, "little"),
            normalized_uid.to_bytes(2, "little"),
        )
    )
    return "0x" + key.hex()


def last_update_storage_key(*, netuid: int) -> str:
    """Build the exact ``SubtensorModule.LastUpdate`` map key."""

    normalized_netuid = int(netuid)
    if not 0 <= normalized_netuid <= 0xFFFF:
        raise ChainSourceV2Error("last-update storage netuid is invalid")
    key = b"".join(
        (
            _twox128(b"SubtensorModule"),
            _twox128(b"LastUpdate"),
            normalized_netuid.to_bytes(2, "little"),
        )
    )
    return "0x" + key.hex()


def decode_last_update_storage(value: Any) -> Sequence[int]:
    """Decode a SCALE ``Vec<u64>`` from exact historical chain state."""

    text = str(value or "")
    if not text.startswith("0x"):
        raise ChainSourceV2Error("last-update storage value is invalid")
    try:
        data = bytes.fromhex(text[2:])
    except ValueError as exc:
        raise ChainSourceV2Error(
            "last-update storage value is invalid hex"
        ) from exc
    if not data:
        raise ChainSourceV2Error("last-update storage value is empty")
    count, offset = _compact_decode(data, 0)
    if count > CHAIN_MAX_HOTKEYS:
        raise ChainSourceV2Error("last-update storage count exceeds policy")
    expected_end = offset + count * 8
    if expected_end != len(data):
        message = "truncated" if expected_end > len(data) else "trailing bytes"
        raise ChainSourceV2Error("last-update storage has %s" % message)
    result = []
    for _index in range(count):
        result.append(int.from_bytes(data[offset : offset + 8], "little"))
        offset += 8
    return result


def decode_weights_storage(value: Any) -> Sequence[Tuple[int, int]]:
    """Decode a SCALE ``Vec<(u16, u16)>`` from historical chain state."""

    text = str(value or "")
    if not text.startswith("0x"):
        raise ChainSourceV2Error("weight storage value is invalid")
    try:
        data = bytes.fromhex(text[2:])
    except ValueError as exc:
        raise ChainSourceV2Error("weight storage value is invalid hex") from exc
    if not data:
        raise ChainSourceV2Error("weight storage value is empty")
    count, offset = _compact_decode(data, 0)
    if count > CHAIN_MAX_HOTKEYS:
        raise ChainSourceV2Error("weight storage count exceeds policy")
    expected_end = offset + count * 4
    if expected_end != len(data):
        message = "truncated" if expected_end > len(data) else "trailing bytes"
        raise ChainSourceV2Error("weight storage has %s" % message)
    result = []
    seen = set()
    for _index in range(count):
        uid = int.from_bytes(data[offset : offset + 2], "little")
        weight = int.from_bytes(data[offset + 2 : offset + 4], "little")
        offset += 4
        if uid in seen:
            raise ChainSourceV2Error("weight storage contains duplicate UIDs")
        seen.add(uid)
        if weight > 0:
            result.append((uid, weight))
    return tuple(sorted(result))


def _checkpoint_merkle(events: Sequence[Mapping[str, Any]]) -> Tuple[str, list[list[str]]]:
    if not events or len(events) > CHAIN_MAX_CHECKPOINT_EVENTS:
        raise ChainSourceV2Error("Arweave checkpoint event count is invalid")
    level = [
        hashlib.sha256(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
        for event in events
    ]
    levels = [[item.hex() for item in level]]
    while len(level) > 1:
        next_level = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else left
            next_level.append(hashlib.sha256(left + right).digest())
        level = next_level
        levels.append([item.hex() for item in level])
    return level[0].hex(), levels


def validate_arweave_checkpoint_event(
    checkpoint: Mapping[str, Any],
    *,
    expected_event_hash: str,
    expected_signed_log_entry: Mapping[str, Any],
    expected_sequence: int,
    expected_merkle_root: str,
) -> Dict[str, Any]:
    """Verify one exact signed event is immutable in an Arweave checkpoint."""

    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "header",
        "signature",
        "events_compressed",
        "tree_levels",
    }:
        raise ChainSourceV2Error("Arweave checkpoint fields are invalid")
    header = checkpoint.get("header")
    if not isinstance(header, Mapping):
        raise ChainSourceV2Error("Arweave checkpoint header is invalid")
    root = normalize_raw_hash(header.get("merkle_root"), "checkpoint merkle root")
    expected_root = normalize_raw_hash(expected_merkle_root, "expected merkle root")
    if root != expected_root:
        raise ChainSourceV2Error("Arweave checkpoint root differs from durable anchor")
    encoded = str(checkpoint.get("events_compressed") or "")
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ChainSourceV2Error("Arweave checkpoint events are invalid base64") from exc
    if not compressed or len(compressed) > CHAIN_MAX_CHECKPOINT_COMPRESSED_BYTES:
        raise ChainSourceV2Error("Arweave checkpoint compressed size exceeds policy")
    try:
        decompressed = gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise ChainSourceV2Error("Arweave checkpoint events are invalid gzip") from exc
    if len(decompressed) > CHAIN_MAX_CHECKPOINT_DECOMPRESSED_BYTES:
        raise ChainSourceV2Error("Arweave checkpoint expanded size exceeds policy")
    try:
        events = json.loads(decompressed.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChainSourceV2Error("Arweave checkpoint events are invalid JSON") from exc
    if not isinstance(events, list) or any(not isinstance(item, Mapping) for item in events):
        raise ChainSourceV2Error("Arweave checkpoint event collection is invalid")
    if int(header.get("event_count", -1)) != len(events):
        raise ChainSourceV2Error("Arweave checkpoint event count differs")
    sequence_range = header.get("sequence_range")
    sequences = [int(item.get("sequence", -1)) for item in events]
    if (
        not isinstance(sequence_range, Mapping)
        or not sequences
        or sequence_range.get("first") != sequences[0]
        or sequence_range.get("last") != sequences[-1]
        or sequences != sorted(sequences)
        or len(sequences) != len(set(sequences))
    ):
        raise ChainSourceV2Error("Arweave checkpoint sequence commitment differs")
    computed_root, computed_levels = _checkpoint_merkle(events)
    if computed_root != root or checkpoint.get("tree_levels") != computed_levels:
        raise ChainSourceV2Error("Arweave checkpoint Merkle tree differs")
    matches = [
        item
        for item in events
        if item.get("event_hash") == expected_event_hash
        and item.get("signed_log_entry") == dict(expected_signed_log_entry)
    ]
    if len(matches) != 1 or int(matches[0].get("sequence", -1)) != int(expected_sequence):
        raise ChainSourceV2Error("signed audit event is absent or duplicated in checkpoint")
    return {
        "checkpoint_number": int(header.get("checkpoint_number", -1)),
        "event_count": len(events),
        "merkle_root": root,
        "event_sequence": int(expected_sequence),
    }


def decode_timelocked_weight_commits(value: Any) -> Sequence[Dict[str, Any]]:
    if value is None:
        return []
    text = str(value or "")
    if not text.startswith("0x"):
        raise ChainSourceV2Error("timelocked weight storage value is invalid")
    try:
        data = bytes.fromhex(text[2:])
    except ValueError as exc:
        raise ChainSourceV2Error(
            "timelocked weight storage value is invalid hex"
        ) from exc
    if not data:
        raise ChainSourceV2Error("timelocked weight storage value is empty")
    count, offset = _compact_decode(data, 0)
    if count > CHAIN_MAX_HOTKEYS:
        raise ChainSourceV2Error("timelocked weight commit count exceeds policy")
    result = []
    for _index in range(count):
        account = data[offset : offset + 32]
        if len(account) != 32:
            raise ChainSourceV2Error("timelocked weight account is truncated")
        offset += 32
        if offset + 8 > len(data):
            raise ChainSourceV2Error("timelocked weight block is truncated")
        submitted_at = int.from_bytes(data[offset : offset + 8], "little")
        offset += 8
        commitment_size, offset = _compact_decode(data, offset)
        if commitment_size <= 0 or commitment_size > 1 << 20:
            raise ChainSourceV2Error("timelocked commitment size is invalid")
        commitment = data[offset : offset + commitment_size]
        if len(commitment) != commitment_size:
            raise ChainSourceV2Error("timelocked commitment is truncated")
        offset += commitment_size
        if offset + 8 > len(data):
            raise ChainSourceV2Error("timelocked reveal round is truncated")
        reveal_round = int.from_bytes(data[offset : offset + 8], "little")
        offset += 8
        result.append(
            {
                "hotkey_public_key": account.hex(),
                "submitted_at": submitted_at,
                "commitment_hex": commitment.hex(),
                "reveal_round": reveal_round,
            }
        )
    if offset != len(data):
        raise ChainSourceV2Error("timelocked weight storage has trailing bytes")
    return result


def encode_selective_metagraph_params(
    *, netuid: int, mechid: int = 0, fields: Sequence[int] = CHAIN_SELECTIVE_FIELDS
) -> str:
    if not isinstance(netuid, int) or isinstance(netuid, bool) or not 0 <= netuid <= 0xFFFF:
        raise ChainSourceV2Error("netuid is outside u16")
    if not isinstance(mechid, int) or isinstance(mechid, bool) or not 0 <= mechid <= 0xFF:
        raise ChainSourceV2Error("mechid is outside u8")
    normalized = tuple(int(field) for field in fields)
    if normalized != CHAIN_SELECTIVE_FIELDS:
        raise ChainSourceV2Error("selective metagraph fields differ from policy")
    encoded = bytearray(netuid.to_bytes(2, "little"))
    encoded.extend(mechid.to_bytes(1, "little"))
    encoded.extend(_compact_encode(len(normalized)))
    for field in normalized:
        encoded.extend(field.to_bytes(2, "little"))
    return "0x" + bytes(encoded).hex()


def _base58_encode(value: bytes) -> str:
    leading_zeroes = len(value) - len(value.lstrip(b"\x00"))
    number = int.from_bytes(value, "big")
    encoded = bytearray()
    while number:
        number, remainder = divmod(number, 58)
        encoded.append(_BASE58_ALPHABET[remainder])
    encoded.extend(_BASE58_ALPHABET[0:1] * leading_zeroes)
    encoded.reverse()
    return bytes(encoded or _BASE58_ALPHABET[0:1]).decode("ascii")


def ss58_encode_account_id(account_id: bytes, ss58_format: int = CHAIN_SS58_FORMAT) -> str:
    raw = bytes(account_id)
    if len(raw) != 32:
        raise ChainSourceV2Error("account id must be 32 bytes")
    if not isinstance(ss58_format, int) or not 0 <= ss58_format <= 63:
        raise ChainSourceV2Error("only one-byte SS58 formats are supported")
    payload = bytes((ss58_format,)) + raw
    checksum = hashlib.blake2b(b"SS58PRE" + payload, digest_size=64).digest()[:2]
    return _base58_encode(payload + checksum)


def _require_unselected(data: bytes, offset: int, start: int, end: int) -> int:
    for field in range(start, end + 1):
        if offset >= len(data) or data[offset] != 0:
            raise ChainSourceV2Error(
                "unexpected selective metagraph field %d" % field
            )
        offset += 1
    return offset


def decode_selective_metagraph_result(encoded: Any) -> Dict[str, Any]:
    if isinstance(encoded, str):
        raw_hex = encoded[2:] if encoded.startswith("0x") else encoded
        try:
            data = bytes.fromhex(raw_hex)
        except ValueError as exc:
            raise ChainSourceV2Error("selective metagraph result is invalid hex") from exc
    elif isinstance(encoded, (bytes, bytearray)):
        data = bytes(encoded)
    else:
        raise ChainSourceV2Error("selective metagraph result is invalid")
    if not data or data[0] != 1:
        raise ChainSourceV2Error("selective metagraph result is absent")
    offset = 1
    netuid, offset = _compact_decode(data, offset)
    offset = _require_unselected(data, offset, 1, 4)
    if offset >= len(data) or data[offset] != 1:
        raise ChainSourceV2Error("owner hotkey is absent")
    offset += 1
    owner_account = data[offset : offset + 32]
    if len(owner_account) != 32:
        raise ChainSourceV2Error("owner hotkey is truncated")
    offset += 32
    offset = _require_unselected(data, offset, 6, 6)
    if offset >= len(data) or data[offset] != 1:
        raise ChainSourceV2Error("metagraph block is absent")
    offset += 1
    block, offset = _compact_decode(data, offset)
    offset = _require_unselected(data, offset, 8, 51)
    if offset >= len(data) or data[offset] != 1:
        raise ChainSourceV2Error("metagraph hotkeys are absent")
    offset += 1
    count, offset = _compact_decode(data, offset)
    if count <= 0 or count > CHAIN_MAX_HOTKEYS:
        raise ChainSourceV2Error("metagraph hotkey count is outside policy")
    byte_count = count * 32
    end = offset + byte_count
    if end > len(data):
        raise ChainSourceV2Error("metagraph hotkeys are truncated")
    hotkeys = [
        ss58_encode_account_id(data[index : index + 32])
        for index in range(offset, end, 32)
    ]
    offset = end
    tail_field_count = len(data) - offset
    max_tail_field_count = max(CHAIN_SELECTIVE_RESULT_LAST_FIELDS) - 52
    if tail_field_count > max_tail_field_count:
        raise ChainSourceV2Error(
            "selective metagraph result has trailing bytes"
        )
    last_field = 52 + tail_field_count
    if last_field not in CHAIN_SELECTIVE_RESULT_LAST_FIELDS:
        raise ChainSourceV2Error(
            "selective metagraph result layout is unsupported"
        )
    offset = _require_unselected(data, offset, 53, last_field)
    if offset != len(data):
        raise ChainSourceV2Error("selective metagraph result has trailing bytes")
    return {
        "netuid": netuid,
        "block": block,
        "owner_hotkey": ss58_encode_account_id(owner_account),
        "hotkeys": hotkeys,
    }


def normalize_raw_hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _RAW_HASH_RE.fullmatch(normalized):
        raise ChainSourceV2Error("%s is not a 32-byte hash" % field)
    return normalized[2:] if normalized.startswith("0x") else normalized


def parse_finalized_header(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ChainSourceV2Error("finalized header is invalid")
    try:
        number_text = str(value["number"])
        number = int(number_text, 16)
    except (KeyError, TypeError, ValueError) as exc:
        raise ChainSourceV2Error("finalized header number is invalid") from exc
    if number < 0:
        raise ChainSourceV2Error("finalized header number is negative")
    state_root = normalize_raw_hash(value.get("stateRoot"), "stateRoot")
    return {
        "block": number,
        "state_root": state_root,
        "state_root_commitment": sha256_bytes(bytes.fromhex(state_root)),
        "parent_hash": normalize_raw_hash(value.get("parentHash"), "parentHash"),
        "extrinsics_root": normalize_raw_hash(
            value.get("extrinsicsRoot"), "extrinsicsRoot"
        ),
    }


def parse_finalized_block_extrinsics(
    value: Mapping[str, Any], *, expected_block: int
) -> Dict[str, Any]:
    """Validate one authenticated ``chain_getBlock`` result.

    The caller obtains the block hash from the finalized chain and requests the
    block by that exact hash.  Only the header number and canonical extrinsic
    byte strings are needed to prove inclusion of an enclave-built extrinsic.
    """

    if not isinstance(value, Mapping) or set(value) - {"block", "justifications"}:
        raise ChainSourceV2Error("finalized block response fields are invalid")
    block = value.get("block")
    if not isinstance(block, Mapping) or set(block) != {"header", "extrinsics"}:
        raise ChainSourceV2Error("finalized block body is invalid")
    header = parse_finalized_header(block.get("header"))
    if header["block"] != int(expected_block):
        raise ChainSourceV2Error("finalized block number differs from request")
    extrinsics = block.get("extrinsics")
    if (
        not isinstance(extrinsics, list)
        or len(extrinsics) > CHAIN_MAX_BLOCK_EXTRINSICS
    ):
        raise ChainSourceV2Error("finalized block extrinsic set is invalid")
    normalized = []
    total_bytes = 0
    for item in extrinsics:
        text = str(item or "").lower()
        if not text.startswith("0x") or len(text) <= 2 or len(text[2:]) % 2:
            raise ChainSourceV2Error("finalized block extrinsic is invalid hex")
        try:
            raw = bytes.fromhex(text[2:])
        except ValueError as exc:
            raise ChainSourceV2Error(
                "finalized block extrinsic is invalid hex"
            ) from exc
        total_bytes += len(raw)
        if total_bytes > CHAIN_MAX_RPC_RESPONSE_BYTES:
            raise ChainSourceV2Error("finalized block extrinsics exceed policy")
        normalized.append(raw.hex())
    return {"header": header, "extrinsics": normalized}


def json_rpc_request(method: str, params: Sequence[Any], request_id: int) -> bytes:
    allowed = set(chain_source_policy_document()["rpc_methods"])
    if method not in allowed:
        raise ChainSourceV2Error("chain RPC method is outside policy")
    if not isinstance(request_id, int) or isinstance(request_id, bool) or request_id < 1:
        raise ChainSourceV2Error("chain RPC request id is invalid")
    return canonical_json(
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": list(params)}
    ).encode("ascii")


def parse_json_rpc_response(body: bytes, request_id: int) -> Any:
    try:
        value = json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ChainSourceV2Error("chain RPC response is invalid JSON") from exc
    if not isinstance(value, Mapping) or set(value) - {"jsonrpc", "id", "result", "error"}:
        raise ChainSourceV2Error("chain RPC response fields are invalid")
    if value.get("jsonrpc") != "2.0" or value.get("id") != request_id:
        raise ChainSourceV2Error("chain RPC response binding is invalid")
    if value.get("error") is not None:
        raise ChainSourceV2Error("chain RPC returned an authenticated error")
    if "result" not in value:
        raise ChainSourceV2Error("chain RPC result is missing")
    return value["result"]
