"""Canonical constraints for the V2 validator hotkey authority.

Dependency-free, Python 3.7-compatible SCALE and runtime-profile helpers
used by the protected Arena signer. This module does not hold key material.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from leadpoet_canonical.attested_v2 import sha256_json


CHAIN_SIGNING_PROFILE_SCHEMA_VERSION = "leadpoet.chain_signing_profile.v2"
_RAW_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_RE = re.compile(r"^[0-9a-f]*$")
_PROFILE_FIELDS = {
    "schema_version",
    "network",
    "chain_endpoint",
    "genesis_hash",
    "spec_version",
    "transaction_version",
    "version_key",
    "commit_call_index",
    "serve_axon_call_index",
    "commit_reveal_version",
    "mechid",
    "tempo",
    "subnet_reveal_period_epochs",
    "block_time_millis",
    "max_snapshot_block_drift",
    "extrinsic_period",
    "signed_extensions",
}
_PROFILE_OPTIONAL_FIELDS = {
    "supported_spec_versions",
    "runtime_upgrade_policy",
}

_EXPECTED_SIGNED_EXTENSIONS = (
    "CheckMortality",
    "CheckNonce",
    "ChargeTransactionPayment",
    "CheckMetadataHash",
    "CheckSpecVersion",
    "CheckTxVersion",
    "CheckGenesis",
    "CheckMortalityAdditionalSigned",
    "CheckMetadataHashAdditionalSigned",
)


class HotkeyAuthorityV2Error(ValueError):
    """A hotkey request is not in an explicitly authorized V2 domain."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HotkeyAuthorityV2Error(message)


def _integer(value: Any, field: str, *, minimum: int = 0, maximum: int) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        "%s must be an integer" % field,
    )
    _require(minimum <= value <= maximum, "%s is outside its range" % field)
    return int(value)


def _raw_hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    _require(bool(_RAW_HASH_RE.fullmatch(normalized)), "%s is invalid" % field)
    return normalized


def _hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    _require(bool(re.fullmatch(r"sha256:[0-9a-f]{64}", normalized)), "%s is invalid" % field)
    return normalized


def _hex_bytes(value: Any, field: str, *, exact_bytes: Optional[int] = None) -> bytes:
    normalized = str(value or "").strip().lower()
    _require(
        len(normalized) % 2 == 0 and bool(_HEX_RE.fullmatch(normalized)),
        "%s is invalid hex" % field,
    )
    result = bytes.fromhex(normalized)
    if exact_bytes is not None:
        _require(len(result) == exact_bytes, "%s has the wrong length" % field)
    return result


def compact_scale_uint(value: int) -> bytes:
    """Encode a non-negative integer using Substrate SCALE compact encoding."""

    normalized = _integer(value, "compact integer", maximum=(1 << 536) - 1)
    if normalized < 1 << 6:
        return bytes((normalized << 2,))
    if normalized < 1 << 14:
        return ((normalized << 2) | 0x01).to_bytes(2, "little")
    if normalized < 1 << 30:
        return ((normalized << 2) | 0x02).to_bytes(4, "little")
    length = max(4, (normalized.bit_length() + 7) // 8)
    _require(length <= 67, "compact integer is too large")
    return bytes((((length - 4) << 2) | 0x03,)) + normalized.to_bytes(
        length, "little"
    )


def encode_mortal_era(*, period: int, current: int) -> bytes:
    """Match Substrate's two-byte mortal-era encoding."""

    normalized_period = _integer(period, "era period", minimum=4, maximum=65536)
    _require(
        normalized_period & (normalized_period - 1) == 0,
        "era period must be a power of two",
    )
    normalized_current = _integer(current, "era current", maximum=(1 << 64) - 1)
    phase = normalized_current % normalized_period
    quantize_factor = max(normalized_period >> 12, 1)
    quantized_phase = (phase // quantize_factor) * quantize_factor
    trailing_zeros = normalized_period.bit_length() - 1
    encoded = min(15, max(1, trailing_zeros - 1)) | (
        (quantized_phase // quantize_factor) << 4
    )
    return int(encoded).to_bytes(2, "little")


def mortal_era_bounds(*, period: int, current: int) -> Tuple[int, int]:
    """Return the inclusive birth and exclusive death blocks for a mortal era."""

    normalized_period = _integer(period, "era period", minimum=4, maximum=65536)
    _require(
        normalized_period & (normalized_period - 1) == 0,
        "era period must be a power of two",
    )
    normalized_current = _integer(current, "era current", maximum=(1 << 64) - 1)
    phase = normalized_current % normalized_period
    quantize_factor = max(normalized_period >> 12, 1)
    quantized_phase = (phase // quantize_factor) * quantize_factor
    birth = normalized_current - (
        (normalized_current - quantized_phase) % normalized_period
    )
    return birth, birth + normalized_period


def validate_chain_signing_profile(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = set(value) if isinstance(value, Mapping) else set()
    _require(
        isinstance(value, Mapping)
        and _PROFILE_FIELDS.issubset(fields)
        and fields.issubset(_PROFILE_FIELDS | _PROFILE_OPTIONAL_FIELDS),
        "chain signing profile fields are invalid",
    )
    _require(
        value.get("schema_version") == CHAIN_SIGNING_PROFILE_SCHEMA_VERSION,
        "chain signing profile schema is invalid",
    )
    network = str(value.get("network") or "")
    endpoint = str(value.get("chain_endpoint") or "")
    _require(bool(re.fullmatch(r"[a-z][a-z0-9_-]{1,31}", network)), "network is invalid")
    _require(endpoint.startswith("wss://") and len(endpoint) <= 512, "chain endpoint is invalid")
    genesis_hash = _raw_hash(value.get("genesis_hash"), "genesis_hash")
    call_index = _hex_bytes(value.get("commit_call_index"), "commit_call_index", exact_bytes=2)
    serve_axon_call_index = _hex_bytes(
        value.get("serve_axon_call_index"),
        "serve_axon_call_index",
        exact_bytes=2,
    )
    signed_extensions = value.get("signed_extensions")
    _require(
        isinstance(signed_extensions, list)
        and tuple(signed_extensions) == _EXPECTED_SIGNED_EXTENSIONS,
        "signed extension profile differs from the measured SDK contract",
    )
    spec_version = _integer(
        value.get("spec_version"), "spec_version", maximum=(1 << 32) - 1
    )
    normalized = {
        "schema_version": CHAIN_SIGNING_PROFILE_SCHEMA_VERSION,
        "network": network,
        "chain_endpoint": endpoint,
        "genesis_hash": genesis_hash,
        "spec_version": spec_version,
        "transaction_version": _integer(
            value.get("transaction_version"),
            "transaction_version",
            maximum=(1 << 32) - 1,
        ),
        "version_key": _integer(
            value.get("version_key"), "version_key", maximum=(1 << 64) - 1
        ),
        "commit_call_index": call_index.hex(),
        "serve_axon_call_index": serve_axon_call_index.hex(),
        "commit_reveal_version": _integer(
            value.get("commit_reveal_version"),
            "commit_reveal_version",
            maximum=(1 << 16) - 1,
        ),
        "mechid": _integer(value.get("mechid"), "mechid", maximum=(1 << 8) - 1),
        "tempo": _integer(
            value.get("tempo"), "tempo", minimum=1, maximum=(1 << 32) - 1
        ),
        "subnet_reveal_period_epochs": _integer(
            value.get("subnet_reveal_period_epochs"),
            "subnet_reveal_period_epochs",
            minimum=1,
            maximum=(1 << 16) - 1,
        ),
        "block_time_millis": _integer(
            value.get("block_time_millis"),
            "block_time_millis",
            minimum=1,
            maximum=120000,
        ),
        "max_snapshot_block_drift": _integer(
            value.get("max_snapshot_block_drift"),
            "max_snapshot_block_drift",
            minimum=1,
            maximum=1024,
        ),
        "extrinsic_period": _integer(
            value.get("extrinsic_period"),
            "extrinsic_period",
            minimum=4,
            maximum=65536,
        ),
        "signed_extensions": list(_EXPECTED_SIGNED_EXTENSIONS),
    }
    if "supported_spec_versions" in value:
        supported = value.get("supported_spec_versions")
        _require(
            isinstance(supported, list)
            and 0 < len(supported) <= 16
            and all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and 0 <= item < (1 << 32)
                for item in supported
            ),
            "supported spec versions are invalid",
        )
        normalized_supported = sorted(set(int(item) for item in supported))
        _require(
            normalized_supported == supported
            and spec_version in normalized_supported,
            "supported spec versions are not canonical",
        )
        normalized["supported_spec_versions"] = normalized_supported
    if "runtime_upgrade_policy" in value:
        policy = value.get("runtime_upgrade_policy")
        _require(
            isinstance(policy, Mapping)
            and set(policy)
            == {"mode", "minimum_spec_version"},
            "runtime upgrade policy is invalid",
        )
        minimum_spec_version = _integer(
            policy.get("minimum_spec_version"),
            "minimum runtime spec version",
            maximum=(1 << 32) - 1,
        )
        _require(
            policy.get("mode") == "exact_payload_invariants_v1"
            and minimum_spec_version <= spec_version,
            "runtime upgrade policy is invalid",
        )
        normalized["runtime_upgrade_policy"] = {
            "mode": "exact_payload_invariants_v1",
            "minimum_spec_version": minimum_spec_version,
        }
    return normalized


def chain_signing_profiles(
    value: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    """Expand one measured manifest into exact, hashable runtime profiles."""

    manifest = validate_chain_signing_profile(value)
    versions = manifest.get(
        "supported_spec_versions", [manifest["spec_version"]]
    )
    base = {
        key: item
        for key, item in manifest.items()
        if key not in {"supported_spec_versions", "runtime_upgrade_policy"}
    }
    return tuple(
        {**base, "spec_version": int(spec_version)}
        for spec_version in versions
    )


def select_chain_signing_profile(
    value: Mapping[str, Any],
    *,
    runtime_version: Mapping[str, Any],
    genesis_hash: str,
) -> Dict[str, Any]:
    """Select one explicitly measured profile for an authenticated runtime."""

    _require(
        isinstance(runtime_version, Mapping),
        "chain runtime version is invalid",
    )
    spec_version = _integer(
        runtime_version.get("specVersion"),
        "runtime specVersion",
        maximum=(1 << 32) - 1,
    )
    transaction_version = _integer(
        runtime_version.get("transactionVersion"),
        "runtime transactionVersion",
        maximum=(1 << 32) - 1,
    )
    observed_genesis = str(genesis_hash or "").lower()
    if observed_genesis.startswith("0x"):
        observed_genesis = observed_genesis[2:]
    candidates = {
        int(profile["spec_version"]): profile
        for profile in chain_signing_profiles(value)
    }
    selected = candidates.get(spec_version)
    if selected is None:
        policy = validate_chain_signing_profile(value).get(
            "runtime_upgrade_policy"
        )
        _require(
            isinstance(policy, Mapping)
            and policy.get("mode") == "exact_payload_invariants_v1"
            and spec_version >= int(policy["minimum_spec_version"]),
            "runtime specVersion is not explicitly supported",
        )
        base = {
            key: item
            for key, item in validate_chain_signing_profile(value).items()
            if key
            not in {"supported_spec_versions", "runtime_upgrade_policy"}
        }
        selected = {**base, "spec_version": spec_version}
    _require(
        transaction_version == int(selected["transaction_version"]),
        "runtime transactionVersion differs from the measured profile",
    )
    _require(
        observed_genesis == str(selected["genesis_hash"]),
        "runtime genesis differs from the measured profile",
    )
    return selected


def resolve_chain_signing_profile_hash(
    value: Mapping[str, Any],
    profile_hash: Any,
    *,
    runtime_spec_version: Any = None,
) -> Dict[str, Any]:
    """Resolve an authorization to one exact member of a measured manifest."""

    expected_hash = _hash(profile_hash, "chain_signing_profile_hash")
    candidates = list(chain_signing_profiles(value))
    if runtime_spec_version is not None:
        manifest = validate_chain_signing_profile(value)
        candidates.append(
            select_chain_signing_profile(
                manifest,
                runtime_version={
                    "specVersion": runtime_spec_version,
                    "transactionVersion": manifest["transaction_version"],
                },
                genesis_hash=manifest["genesis_hash"],
            )
        )
    matches_by_hash = {
        sha256_json(profile): profile for profile in candidates
    }
    matches = [
        profile
        for digest, profile in matches_by_hash.items()
        if digest == expected_hash
    ]
    _require(
        len(matches) == 1,
        "chain signing profile hash is not explicitly supported",
    )
    return matches[0]


def chain_signing_profile_hash(value: Mapping[str, Any]) -> str:
    return sha256_json(validate_chain_signing_profile(value))


def encode_commit_timelocked_call(
    *,
    profile: Mapping[str, Any],
    netuid: int,
    commitment: bytes,
    reveal_round: int,
) -> bytes:
    normalized = validate_chain_signing_profile(profile)
    commitment_bytes = bytes(commitment)
    _require(0 < len(commitment_bytes) <= 1 << 20, "commitment size is invalid")
    return b"".join(
        (
            bytes.fromhex(normalized["commit_call_index"]),
            _integer(netuid, "netuid", maximum=(1 << 16) - 1).to_bytes(2, "little"),
            normalized["mechid"].to_bytes(1, "little"),
            compact_scale_uint(len(commitment_bytes)),
            commitment_bytes,
            _integer(
                reveal_round, "reveal_round", maximum=(1 << 64) - 1
            ).to_bytes(8, "little"),
            normalized["commit_reveal_version"].to_bytes(2, "little"),
        )
    )


def encode_weight_signature_payload(
    *,
    profile: Mapping[str, Any],
    call_bytes: bytes,
    era_current: int,
    nonce: int,
    block_hash: str,
) -> Tuple[bytes, bytes]:
    """Return the SCALE preimage and the exact bytes sr25519 must sign.

    ``substrate-interface`` hashes an extrinsic payload with BLAKE2b-256 when
    it exceeds 256 bytes.  Timelocked commitments normally take that path.
    """

    normalized = validate_chain_signing_profile(profile)
    call = bytes(call_bytes)
    _require(2 <= len(call) <= (1 << 20) + 32, "call bytes are outside limit")
    block_hash_bytes = _hex_bytes(block_hash, "block_hash", exact_bytes=32)
    preimage = b"".join(
        (
            call,
            encode_mortal_era(
                period=normalized["extrinsic_period"], current=era_current
            ),
            compact_scale_uint(_integer(nonce, "nonce", maximum=(1 << 64) - 1)),
            compact_scale_uint(0),  # ChargeTransactionPayment tip is fixed to zero.
            b"\x00",  # CheckMetadataHash mode = Disabled.
            normalized["spec_version"].to_bytes(4, "little"),
            normalized["transaction_version"].to_bytes(4, "little"),
            bytes.fromhex(normalized["genesis_hash"]),
            block_hash_bytes,
            b"\x00",  # CheckMetadataHash additional value = None.
        )
    )
    signed = (
        hashlib.blake2b(preimage, digest_size=32).digest()
        if len(preimage) > 256
        else preimage
    )
    return preimage, signed


def encode_signed_extrinsic_v2(
    *,
    hotkey_public_key_hex: str,
    signature_hex: str,
    era_period: int,
    era_current: int,
    nonce: int,
    call_data_hex: str,
) -> bytes:
    """Encode the exact signed Substrate V4 extrinsic emitted by the SDK.

    The production signing profile fixes zero tip and disabled metadata-hash
    mode.  Reconstructing the complete bytes inside Nitro lets the enclave
    later prove that this exact extrinsic appeared in a finalized block.
    """

    public_key = _hex_bytes(
        hotkey_public_key_hex, "hotkey_public_key", exact_bytes=32
    )
    signature = _hex_bytes(signature_hex, "signature", exact_bytes=64)
    call_data = _hex_bytes(call_data_hex, "call_data")
    _require(2 <= len(call_data) <= (1 << 20) + 32, "call data is outside limit")
    body = b"".join(
        (
            b"\x84",  # signed extrinsic, format version 4
            b"\x00",  # MultiAddress::Id
            public_key,
            b"\x01",  # MultiSignature::Sr25519
            signature,
            encode_mortal_era(period=era_period, current=era_current),
            compact_scale_uint(_integer(nonce, "nonce", maximum=(1 << 64) - 1)),
            compact_scale_uint(0),  # ChargeTransactionPayment tip
            b"\x00",  # CheckMetadataHash mode = Disabled
            call_data,
        )
    )
    return compact_scale_uint(len(body)) + body


def signed_extrinsic_hash_v2(extrinsic: bytes) -> str:
    raw = bytes(extrinsic)
    _require(bool(raw), "signed extrinsic is empty")
    return "0x" + hashlib.blake2b(raw, digest_size=32).hexdigest()
