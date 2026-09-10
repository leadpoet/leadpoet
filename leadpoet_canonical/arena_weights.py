"""Small canonical contract for normal-validator Arena weights.

The module deliberately has no Research Lab receipt, release, or ancestry
dependency.  A signed accepted state plus a finalized hotkey ordering is all
that is needed to derive the exact sparse vector which may be submitted.
"""

from __future__ import annotations

import base64
import json
import re
import struct
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Sequence

from leadpoet_canonical.lab_arena_rewards import (
    LabArenaRewardError,
    SIGNING_ALGORITHM,
    champion_values,
    public_key_hash,
    sha256_json,
    validate_reward_basis,
    verify_reward_basis_signature,
)
from leadpoet_canonical.weights import compare_weights_hash


ACCEPTED_WEIGHT_STATE_SCHEMA_VERSION = "leadpoet.arena.accepted_weight_state.v1"
WEIGHT_STATE_SIGNATURE_PREFIX = "arena_weight_state_hash:"
PARTS_PER_BILLION = 1_000_000_000
U16_MAX = 65_535

_BODY_FIELDS = (
    "schema_version", "network", "genesis_hash", "netuid", "epoch",
    "valid_from_block", "valid_until_block", "reward_basis",
    "burn_hotkey", "issued_at",
)
_HOTKEY_CHARS = frozenset("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$")


class ArenaWeightError(ValueError):
    """An accepted state or derived Arena vector is invalid."""


def accepted_weight_state_body(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the exact body covered by ``state_hash``."""

    return {field: value[field] for field in _BODY_FIELDS}


def classify_arena_signed_request_message(message: bytes, *, validator_hotkey: str) -> str:
    """Recognize only canonical Arena claim/completion request messages.

    This is used before the protected hotkey signs.  The caller supplies no
    signature, so contract validation is performed with a disposable shape
    signature and a verifier that can only approve that exact parsed message.
    """

    try:
        text = bytes(message).decode("utf-8")
        document = json.loads(text)
        from lab_arena import contracts
        _require(isinstance(document, Mapping), "Arena signing message must be an object")
        _require(contracts.signed_request_message(document) == text, "Arena signing message is not canonical")
        scope = document.get("scope")
        _require(scope in (contracts.SCOPE_CLAIM, contracts.SCOPE_COMPLETE), "Arena signing scope is not authorized")
        envelope = dict(document)
        envelope["signature"] = "0x" + ("0" * 128)
        contracts.validate_signed_request(
            envelope, expected_scope=scope, now=int(document.get("timestamp")),
            verify_signature=lambda _hotkey, _signature, candidate: candidate == text,
        )
        _require(document.get("hotkey") == validator_hotkey, "Arena signing hotkey differs")
        body = document.get("body")
        if scope == contracts.SCOPE_CLAIM:
            _require(isinstance(body, Mapping) and set(body) == {"declared_parallelism"}, "Arena claim body is invalid")
            parallelism = body["declared_parallelism"]
            _require(isinstance(parallelism, int) and not isinstance(parallelism, bool) and 1 <= parallelism <= 1024, "Arena claim parallelism is invalid")
        else:
            _require(isinstance(body, Mapping) and set(body) == {"run_id", "result", "output", "lease_token"}, "Arena completion body is invalid")
            contracts.validate_run_result(body["result"])
            _require(isinstance(body["run_id"], str) and body["run_id"], "Arena completion run_id is invalid")
            _require(isinstance(body["lease_token"], str) and body["lease_token"], "Arena completion lease token is invalid")
        return "validator.arena_%s.v1" % ("claim" if scope == contracts.SCOPE_CLAIM else "complete")
    except ArenaWeightError:
        raise
    except Exception as exc:
        raise ArenaWeightError("Arena signing message is invalid") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArenaWeightError(message)


def _integer(value: Any, name: str, maximum: int = (1 << 64) - 1) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), "%s must be an integer" % name)
    _require(0 <= value <= maximum, "%s is outside its range" % name)
    return int(value)


def _hotkey(value: Any, name: str) -> str:
    text = str(value or "")
    _require(40 <= len(text) <= 64 and all(char in _HOTKEY_CHARS for char in text), "%s is invalid" % name)
    return text


def validate_accepted_weight_state(value: Any) -> Dict[str, Any]:
    _require(isinstance(value, Mapping), "accepted weight state must be an object")
    _require(set(value) == set(_BODY_FIELDS) | {"state_hash", "signature"}, "accepted weight state fields are invalid")
    _require(value.get("schema_version") == ACCEPTED_WEIGHT_STATE_SCHEMA_VERSION, "accepted weight state schema is invalid")
    network = str(value.get("network") or "")
    _require(network and len(network) <= 64, "network is invalid")
    genesis_hash = str(value.get("genesis_hash") or "").lower()
    _require(len(genesis_hash) == 64 and all(c in "0123456789abcdef" for c in genesis_hash), "genesis_hash is invalid")
    netuid = _integer(value.get("netuid"), "netuid", (1 << 16) - 1)
    epoch = _integer(value.get("epoch"), "epoch")
    valid_from = _integer(value.get("valid_from_block"), "valid_from_block")
    valid_until = _integer(value.get("valid_until_block"), "valid_until_block")
    _require(valid_from <= valid_until, "accepted weight state validity window is invalid")
    _require(isinstance(value.get("issued_at"), str) and bool(_TIMESTAMP_RE.fullmatch(value["issued_at"])), "issued_at is invalid")
    reward_basis = validate_reward_basis(value.get("reward_basis"))
    _require(int(reward_basis["effective_reward_epoch"]) <= epoch, "reward basis is not effective for epoch")
    _hotkey(value.get("burn_hotkey"), "burn_hotkey")
    body = accepted_weight_state_body(value)
    expected_hash = sha256_json(body)
    _require(value.get("state_hash") == expected_hash, "state_hash does not match accepted weight state")
    signature = value.get("signature")
    _require(isinstance(signature, Mapping) and set(signature) == {"algorithm", "public_key_hash", "signature_b64"}, "accepted weight state signature fields are invalid")
    _require(signature.get("algorithm") == SIGNING_ALGORITHM, "accepted weight state signature algorithm is invalid")
    return dict(value)


def verify_accepted_weight_state_signature(value: Any, *, public_key_der: bytes, expected_public_key_hash: str) -> str:
    state = validate_accepted_weight_state(value)
    _require(public_key_hash(public_key_der) == expected_public_key_hash, "public key does not match pinned Arena key")
    signature = state["signature"]
    _require(signature.get("public_key_hash") == expected_public_key_hash, "accepted state was signed by a different key")
    try:
        raw_signature = base64.b64decode(str(signature.get("signature_b64") or ""), validate=True)
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        key = serialization.load_der_public_key(bytes(public_key_der))
        key.verify(raw_signature, (WEIGHT_STATE_SIGNATURE_PREFIX + state["state_hash"]).encode("utf-8"), ec.ECDSA(hashes.SHA256()))
    except Exception as exc:
        raise ArenaWeightError("accepted weight state signature is invalid") from exc
    try:
        verify_reward_basis_signature(state["reward_basis"], public_key_der=public_key_der, expected_public_key_hash=expected_public_key_hash)
    except LabArenaRewardError as exc:
        raise ArenaWeightError("reward basis signature is invalid") from exc
    return str(state["state_hash"])


def derive_arena_weights(value: Any, metagraph_hotkeys: Sequence[str]) -> Dict[str, Any]:
    """Derive exact emit weights from state and finalized UID ownership."""

    state = validate_accepted_weight_state(value)
    _require(not isinstance(metagraph_hotkeys, (str, bytes)) and isinstance(metagraph_hotkeys, Sequence), "metagraph_hotkeys must be a sequence")
    hotkeys = [_hotkey(item, "metagraph_hotkeys[]") for item in metagraph_hotkeys]
    _require(len(set(hotkeys)) == len(hotkeys), "finalized metagraph contains duplicate hotkeys")
    by_hotkey = {}  # type: Dict[str, Fraction]
    registered_hotkeys = set(hotkeys)
    burn_hotkey = str(state["burn_hotkey"])
    _require(burn_hotkey in registered_hotkeys, "burn_hotkey is not registered at finalized state")
    champion = champion_values(state["reward_basis"], int(state["epoch"]), hotkeys)
    champion_share = Fraction(str(champion["champion_share"]))
    _require(champion_share <= 1, "Arena allocation exceeds total emissions")
    if champion["champion_uid"] is not None and champion_share:
        champion_hotkey = hotkeys[int(champion["champion_uid"])]
        by_hotkey[champion_hotkey] = by_hotkey.get(champion_hotkey, Fraction(0)) + champion_share
    else:
        champion_share = Fraction(0)
    unused = Fraction(1) - champion_share
    if unused:
        by_hotkey[burn_hotkey] = by_hotkey.get(burn_hotkey, Fraction(0)) + unused
    registered = []
    for uid, hotkey in enumerate(hotkeys):
        if by_hotkey.get(hotkey, Fraction(0)) > 0:
            registered.append((uid, by_hotkey[hotkey]))
    _require(registered, "accepted state allocates no weight to finalized UIDs")
    # Bittensor converts the host's float32 vector, divides by its float max,
    # then applies Python's round(). Reproduce those operations exactly while
    # keeping all allocation arithmetic above rational and deterministic.
    float32 = lambda item: struct.unpack("!f", struct.pack("!f", float(item)))[0]
    float_weights = [(uid, float32(weight)) for uid, weight in registered]
    maximum = float(max(weight for _, weight in float_weights))
    sparse = [(uid, round((float(weight) / maximum) * U16_MAX)) for uid, weight in float_weights]
    sparse = [(uid, weight) for uid, weight in sparse if weight > 0]
    uids = [uid for uid, _ in sparse]
    weights = [weight for _, weight in sparse]
    return {
        "state_hash": state["state_hash"], "netuid": state["netuid"], "epoch": state["epoch"],
        "sparse_uids": uids, "sparse_weights_u16": weights,
        "weights_hash": compare_weights_hash(int(state["netuid"]), int(state["epoch"]), sparse),
        "champion_share_ppb": int(champion_share * PARTS_PER_BILLION),
        "burned_residual_ppb": int(unused * PARTS_PER_BILLION),
    }
