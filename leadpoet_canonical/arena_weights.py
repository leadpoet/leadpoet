"""Small canonical contract for normal-validator Arena weights.

The module deliberately has no Research Lab receipt, release, or ancestry
dependency.  A signed accepted state plus a finalized hotkey ordering is all
that is needed to derive the exact sparse vector which may be submitted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import struct
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Sequence

from leadpoet_canonical.lab_arena_rewards import (
    LabArenaRewardError,
    SIGNING_ALGORITHM,
    REWARD_BASIS_V2_SCHEMA_VERSION,
    champion_values,
    slot_allocations,
    public_key_hash,
    sha256_json,
    validate_reward_basis,
    validate_weekly_slot_decay,
    verify_reward_basis_signature,
)


ACCEPTED_WEIGHT_STATE_SCHEMA_VERSION = "leadpoet.arena.accepted_weight_state.v1"
COLLECTION_BASIS_SCHEMA = "leadpoet.collection.fixed_run_quotes.v1"
WEIGHT_STATE_SIGNATURE_PREFIX = "arena_weight_state_hash:"
PARTS_PER_BILLION = 1_000_000_000
U16_MAX = 65_535
SLOT_ELIGIBILITY_UNTIL_REPLACED = "until_replaced"

_BODY_FIELDS = (
    "schema_version", "network", "genesis_hash", "netuid", "epoch",
    "valid_from_block", "valid_until_block", "reward_basis",
    "burn_hotkey", "issued_at",
)
_SLOT_FIELDS = ("slot_eligibility", "weekly_slot_decay")
_OPTIONAL_BODY_FIELDS = _SLOT_FIELDS + ("tool_market",)
TOOL_MARKET_SCHEMA = "leadpoet.tool_market.supplier_allocations.v1"
_HOTKEY_CHARS = frozenset("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$")


class ArenaWeightError(ValueError):
    """An accepted state or derived Arena vector is invalid."""


def compare_weights_hash(netuid: int, epoch_id: int, weights: Sequence[tuple[int, int]]) -> str:
    """Hash the exact sparse Arena vector for finalized-chain comparison."""
    sorted_weights = sorted(weights, key=lambda x: x[0])

    payload = {
        "netuid": netuid,
        "epoch_id": epoch_id,
        "weights": [[uid, w_u16] for uid, w_u16 in sorted_weights]
    }

    # Canonical JSON: sorted keys, no whitespace
    canonical_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def accepted_weight_state_body(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the exact body covered by ``state_hash``."""

    return {field: value[field] for field in _BODY_FIELDS + _OPTIONAL_BODY_FIELDS if field in value}


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


def _collection_basis(state: Mapping[str, Any]) -> bool:
    return isinstance(state.get("reward_basis"), Mapping) and state["reward_basis"].get("schema_version") == COLLECTION_BASIS_SCHEMA


def validate_collection_basis(basis: Mapping[str, Any], epoch: int, start_block: int) -> None:
    """All monetary inputs are whole micro-USD or explicitly named chain base units."""
    _require(set(basis) == {"schema_version", "source_epoch", "job_count", "job_set_sha256",
                            "quotes_microusd", "valuation"}, "collection basis fields are invalid")
    _require(_integer(basis["source_epoch"], "source_epoch") + 1 == epoch, "collection source epoch differs")
    count = _integer(basis["job_count"], "job_count")
    _require(isinstance(basis["job_set_sha256"], str) and
             re.fullmatch(r"[0-9a-f]{64}", basis["job_set_sha256"]), "collection job hash is invalid")
    quotes = basis["quotes_microusd"]
    _require(type(quotes) is dict and len(quotes) <= min(65536, count), "collection beneficiaries are invalid")
    _require(bool(quotes) == bool(count), "collection job count differs")
    for key, amount in quotes.items():
        _hotkey(key, "beneficiary")
        _require(_integer(amount, "quote total") > 0, "quote total must be positive")
    valuation = basis["valuation"]
    if not count:
        _require(valuation is None, "empty collection has a valuation")
        return
    _validate_emission_valuation(valuation, start_block)


def _validate_emission_valuation(valuation: Any, start_block: int) -> None:
    _require(isinstance(valuation, dict) and set(valuation) == {
        "block", "block_hash", "miner_alpha_rao", "tao_rao_per_alpha", "tao_price_microusd",
        "price_updated_at", "observed_at", "emission_microusd"}, "collection valuation fields are invalid")
    _require(_integer(valuation["block"], "valuation block") == start_block, "collection valuation block differs")
    _require(isinstance(valuation["block_hash"], str) and
             re.fullmatch(r"0x[0-9a-f]{64}", valuation["block_hash"]), "valuation block hash is invalid")
    _require(0 <= _integer(valuation["observed_at"], "observed_at") -
             _integer(valuation["price_updated_at"], "price_updated_at") <= 900, "stale TAO/USD price")
    alpha = _integer(valuation["miner_alpha_rao"], "miner_alpha_rao")
    price = _integer(valuation["tao_rao_per_alpha"], "tao_rao_per_alpha")
    usd = _integer(valuation["tao_price_microusd"], "tao_price_microusd")
    budget = _integer(valuation["emission_microusd"], "emission_microusd")
    _require(price > 0 and usd > 0 and budget == alpha * price * usd // 10**18,
             "collection emission valuation differs")


def validate_tool_market(market: Any, basis: Mapping[str, Any], *, epoch: int, start_block: int, burn: str) -> None:
    """Authorize only registered supplier targets within the unquoted collection residual."""
    fields = {"schema_version", "epoch", "emission_microusd", "allocations_microusd"}
    if not basis["job_count"]:
        fields.add("valuation")
    _require(isinstance(market, dict) and set(market) == fields, "tool market allocation fields differ")
    _require(market["schema_version"] == TOOL_MARKET_SCHEMA and _integer(market["epoch"], "market epoch") == epoch,
             "tool market epoch or schema differs")
    if basis["job_count"]:
        valuation = basis["valuation"]  # verified with the unchanged collection quote set
    else:
        valuation = market["valuation"]
        _validate_emission_valuation(valuation, start_block)
    emission = _integer(market["emission_microusd"], "tool market emission")
    _require(emission > 0 and emission == valuation["emission_microusd"], "tool market emission differs")
    targets = market["allocations_microusd"]
    _require(isinstance(targets, dict) and 0 < len(targets) <= 65536, "invalid tool market beneficiaries")
    for key, amount in targets.items():
        _require(_hotkey(key, "supplier") != burn and _integer(amount, "supplier amount") * U16_MAX >= emission,
                 "invalid tool market target")
    # Reserve every collection quote even if a beneficiary left the metagraph.
    _require(sum(targets.values()) <= emission - sum(basis["quotes_microusd"].values()),
             "tool market exceeds residual after collection quotes")


def validate_accepted_weight_state(value: Any) -> Dict[str, Any]:
    _require(isinstance(value, Mapping), "accepted weight state must be an object")
    required = set(_BODY_FIELDS) | {"state_hash", "signature"}
    _require(required <= set(value) and not set(value) - required - set(_OPTIONAL_BODY_FIELDS), "accepted weight state fields are invalid")
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
    if _collection_basis(value):
        _require(not (set(value) & set(_SLOT_FIELDS)), "collection cannot pay champion slots")
        _require((network, netuid) == ("finney", 71), "collection subnet differs")
        reward_basis = value["reward_basis"]
        validate_collection_basis(reward_basis, epoch, valid_from)
        _require(value.get("burn_hotkey") not in reward_basis["quotes_microusd"], "burn cannot fund a job")
        if "tool_market" in value:
            validate_tool_market(value["tool_market"], reward_basis, epoch=epoch, start_block=valid_from,
                                 burn=value["burn_hotkey"])
    else:
        _require("tool_market" not in value, "tool market requires a collection reward state")
        reward_basis = validate_reward_basis(value.get("reward_basis"))
        _require(int(reward_basis["effective_reward_epoch"]) <= epoch, "reward basis is not effective for epoch")
    if "slot_eligibility" in value:
        _require(
            reward_basis["schema_version"] == REWARD_BASIS_V2_SCHEMA_VERSION
            and type(value["slot_eligibility"]) is str
            and value["slot_eligibility"] == SLOT_ELIGIBILITY_UNTIL_REPLACED,
            "slot_eligibility is invalid",
        )
    if "weekly_slot_decay" in value:
        try:
            validate_weekly_slot_decay(
                value["weekly_slot_decay"], reward_basis,
                epoch_id=epoch,
            )
        except LabArenaRewardError as exc:
            raise ArenaWeightError(str(exc)) from exc
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
    if _collection_basis(state):
        # Collection facts are frozen once in the signed state itself.
        return str(state["state_hash"])
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
    collection_share = Fraction(0)
    if _collection_basis(state):
        basis = state["reward_basis"]
        quotes = basis["quotes_microusd"]
        budget = (basis["valuation"] or {}).get("emission_microusd", 0)
        if quotes:
            divisor = max(budget, sum(quotes.values()))
            by_hotkey = {key: Fraction(amount, divisor) for key, amount in quotes.items()
                         if key in registered_hotkeys}
            collection_share = sum(by_hotkey.values(), Fraction(0))
        champion_share = Fraction(0)
    elif state["reward_basis"]["schema_version"] == REWARD_BASIS_V2_SCHEMA_VERSION:
        by_hotkey = slot_allocations(
            state["reward_basis"], int(state["epoch"]), hotkeys,
            until_replaced=state.get("slot_eligibility") == SLOT_ELIGIBILITY_UNTIL_REPLACED,
            weekly_slot_decay=state.get("weekly_slot_decay"),
        )
        champion_share = sum(by_hotkey.values(), Fraction(0))
    else:
        champion = champion_values(state["reward_basis"], int(state["epoch"]), hotkeys)
        champion_share = Fraction(str(champion["champion_share"]))
        if champion["champion_uid"] is not None and champion_share:
            champion_hotkey = hotkeys[int(champion["champion_uid"])]
            by_hotkey[champion_hotkey] = by_hotkey.get(champion_hotkey, Fraction(0)) + champion_share
        else:
            champion_share = Fraction(0)
    _require(champion_share <= 1, "Arena allocation exceeds total emissions")
    unused = Fraction(1) - champion_share - collection_share
    if "tool_market" in state:
        market = state["tool_market"]
        for key, amount in market["allocations_microusd"].items():
            if key in registered_hotkeys:
                share = Fraction(amount, market["emission_microusd"])
                by_hotkey[key] = by_hotkey.get(key, Fraction(0)) + share
                unused -= share
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
        **({"collection_share_ppb": int(collection_share * PARTS_PER_BILLION)} if _collection_basis(state) else {}),
    }
