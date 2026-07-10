"""Dependency-free canonical final-weight computation for validator enclaves.

The function mirrors the current validator allocation order and float
arithmetic. It accepts an immutable JSON snapshot, derives the float vector,
performs Bittensor's pinned float32/u16 conversion, and computes the canonical
bundle hash. It is intentionally Python 3.7 compatible.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from leadpoet_canonical.weights import bundle_weights_hash


WEIGHT_SNAPSHOT_SCHEMA_VERSION = "leadpoet.weight_computation_request.v1"
WEIGHT_RESULT_SCHEMA_VERSION = "leadpoet.weight_computation_result.v1"
U16_MAX = 65535

_SNAPSHOT_FIELDS = {
    "schema_version",
    "netuid",
    "epoch_id",
    "block",
    "commit_sha",
    "config_hash",
    "parent_receipt_hashes",
    "research_lab_allocation_receipt_hash",
    "burn_target_uid",
    "expected_burn_target_hotkey",
    "metagraph_hotkeys",
    "banned_hotkeys",
    "banned_lookup_ok",
    "ff_enabled",
    "base_burn_share",
    "champion_share",
    "champion_uid",
    "effective_champion_share",
    "research_lab_fallback_share",
    "research_lab_allocation_doc",
    "leaderboard_bonus_share",
    "leaderboard_rank_shares",
    "leaderboard_entries",
    "leaderboard_fetch_ok",
    "fulfillment_share",
    "fulfillment_rows",
    "fulfillment_fetch_ok",
    "rolling_lead_count",
    "rolling_scores",
    "sourcing_floor_threshold",
    "min_total_rep_for_distribution",
}

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class WeightComputationError(ValueError):
    """Raised when a weight snapshot is incomplete, unsafe, or inconsistent."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise WeightComputationError("weight value is not canonical JSON") from exc


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def weight_config_document(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the exact behavior configuration committed by a weight request."""

    return {
        "netuid": snapshot.get("netuid"),
        "burn_target_uid": snapshot.get("burn_target_uid"),
        "expected_burn_target_hotkey": snapshot.get("expected_burn_target_hotkey"),
        "ff_enabled": snapshot.get("ff_enabled"),
        "base_burn_share": snapshot.get("base_burn_share"),
        "champion_share": snapshot.get("champion_share"),
        "research_lab_fallback_share": snapshot.get("research_lab_fallback_share"),
        "leaderboard_bonus_share": snapshot.get("leaderboard_bonus_share"),
        "leaderboard_rank_shares": snapshot.get("leaderboard_rank_shares"),
        "sourcing_floor_threshold": snapshot.get("sourcing_floor_threshold"),
        "min_total_rep_for_distribution": snapshot.get("min_total_rep_for_distribution"),
    }


def weight_config_hash(snapshot: Mapping[str, Any]) -> str:
    return sha256_json(weight_config_document(snapshot))


def _float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise WeightComputationError("%s must be numeric" % field) from exc
    if not math.isfinite(result):
        raise WeightComputationError("%s must be finite" % field)
    return result


def _non_negative_float(value: Any, field: str) -> float:
    result = _float(value, field)
    if result < 0:
        raise WeightComputationError("%s must be non-negative" % field)
    return result


def _int(value: Any, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise WeightComputationError("%s must be an integer" % field)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise WeightComputationError("%s must be an integer" % field) from exc
    if result < minimum:
        raise WeightComputationError("%s is below its minimum" % field)
    return result


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def normalize_to_u16_with_uids_pure(uids: Sequence[int], weights: Sequence[float]) -> Tuple[List[int], List[int]]:
    """Match Bittensor 9.12.2 ``convert_weights_and_uids_for_emit`` exactly.

    Bittensor first creates a NumPy float32 array. ``_float32`` reproduces that
    quantization before the implementation converts values back to Python
    floats, max-upscales them, rounds with Python's round(), and removes zeros.
    """

    if len(uids) != len(weights):
        raise WeightComputationError("uids and weights must have the same length")
    if not weights:
        return [], []
    normalized_uids = [_int(uid, "uid") for uid in uids]
    normalized_weights = [_float32(_float(weight, "weight")) for weight in weights]
    if min(normalized_weights) < 0:
        raise WeightComputationError("negative weight cannot be emitted")
    if sum(normalized_weights) == 0:
        return [], []
    max_weight = float(max(normalized_weights))
    scaled = [float(value) / max_weight for value in normalized_weights]
    sparse_uids = []
    sparse_weights = []
    for uid, weight in zip(normalized_uids, scaled):
        value = round(float(weight) * U16_MAX)
        if value != 0:
            sparse_uids.append(uid)
            sparse_weights.append(value)
    return sparse_uids, sparse_weights


def research_lab_uid_weights_from_allocation(
    allocation_doc: Any,
    *,
    metagraph_hotkeys: Sequence[str],
    reserved_share: float,
) -> Tuple[Dict[int, float], float, Dict[str, float]]:
    """Dependency-free form of the validator's existing Research Lab helper."""

    reserved_share = float(reserved_share)
    if not isinstance(allocation_doc, dict) or not allocation_doc:
        return {}, reserved_share, {
            "paid": 0.0,
            "burn": reserved_share,
            "unallocated": reserved_share,
            "deregistered": 0.0,
        }
    lab_cap_share = max(
        0.0,
        min(reserved_share, float(allocation_doc.get("lab_cap_percent") or 0.0) / 100.0),
    )
    unallocated_share = min(
        lab_cap_share,
        max(0.0, float(allocation_doc.get("unallocated_percent") or 0.0) / 100.0),
    )
    uid_weights = {}  # type: Dict[int, float]
    paid_share = 0.0
    deregistered_share = 0.0
    for section in (
        "reimbursement_allocations",
        "champion_allocations",
        "queued_champion_allocations",
    ):
        for row in allocation_doc.get(section) or []:
            if not isinstance(row, dict):
                continue
            pct = max(0.0, float(row.get("paid_alpha_percent") or 0.0) / 100.0)
            if pct <= 0:
                continue
            paid_share += pct
            try:
                uid = int(row.get("uid"))
                expected_hotkey = str(row.get("miner_hotkey") or "")
                actual_hotkey = metagraph_hotkeys[uid]
            except Exception:
                deregistered_share += pct
                continue
            if expected_hotkey and actual_hotkey != expected_hotkey:
                deregistered_share += pct
                continue
            uid_weights[uid] = uid_weights.get(uid, 0.0) + pct
    payable_cap = max(0.0, lab_cap_share - unallocated_share)
    if paid_share > payable_cap and paid_share > 0:
        scale = payable_cap / paid_share
        uid_weights = {uid: weight * scale for uid, weight in uid_weights.items()}
        deregistered_share *= scale
        paid_share = payable_cap
    reported_total = paid_share + unallocated_share
    rounding_gap = max(0.0, lab_cap_share - reported_total)
    reserved_gap = max(0.0, reserved_share - lab_cap_share)
    burn_share = unallocated_share + deregistered_share + rounding_gap + reserved_gap
    return uid_weights, burn_share, {
        "paid": paid_share,
        "burn": burn_share,
        "unallocated": unallocated_share + rounding_gap + reserved_gap,
        "deregistered": deregistered_share,
    }


def _doc_percent_share(doc: Any, key: str, fallback_share: float) -> float:
    if isinstance(doc, dict) and doc.get(key) not in (None, ""):
        try:
            return max(0.0, min(1.0, float(doc.get(key)) / 100.0))
        except (TypeError, ValueError):
            return fallback_share
    return fallback_share


def _add_weight(uid_weights: Dict[int, float], uid: int, weight: float) -> None:
    if uid not in uid_weights:
        uid_weights[uid] = 0.0
    uid_weights[uid] += weight


def compute_final_weights(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(snapshot, Mapping) or set(snapshot) != _SNAPSHOT_FIELDS:
        raise WeightComputationError("weight snapshot fields do not match the canonical schema")
    if snapshot.get("schema_version") != WEIGHT_SNAPSHOT_SCHEMA_VERSION:
        raise WeightComputationError("unsupported weight snapshot schema")

    netuid = _int(snapshot["netuid"], "netuid")
    epoch_id = _int(snapshot["epoch_id"], "epoch_id")
    block = _int(snapshot["block"], "block")
    if not _COMMIT_RE.fullmatch(str(snapshot["commit_sha"] or "")):
        raise WeightComputationError("commit_sha must be a full Git object id")
    if not _HASH_RE.fullmatch(str(snapshot["config_hash"] or "")):
        raise WeightComputationError("config_hash must be a canonical SHA256 hash")
    if snapshot["config_hash"] != weight_config_hash(snapshot):
        raise WeightComputationError("config_hash does not match weight behavior configuration")
    parent_receipts = snapshot["parent_receipt_hashes"]
    if not isinstance(parent_receipts, list) or any(
        not _HASH_RE.fullmatch(str(item or "")) for item in parent_receipts
    ):
        raise WeightComputationError("parent_receipt_hashes are invalid")
    allocation_receipt_hash = str(snapshot["research_lab_allocation_receipt_hash"] or "")
    if allocation_receipt_hash and not _HASH_RE.fullmatch(allocation_receipt_hash):
        raise WeightComputationError("research_lab_allocation_receipt_hash is invalid")
    burn_uid = _int(snapshot["burn_target_uid"], "burn_target_uid")
    metagraph_hotkeys = snapshot["metagraph_hotkeys"]
    if not isinstance(metagraph_hotkeys, list) or any(not isinstance(item, str) for item in metagraph_hotkeys):
        raise WeightComputationError("metagraph_hotkeys must be a list of strings")
    if burn_uid >= len(metagraph_hotkeys):
        raise WeightComputationError("burn target UID is not registered")
    expected_burn_hotkey = str(snapshot.get("expected_burn_target_hotkey") or "")
    if expected_burn_hotkey and metagraph_hotkeys[burn_uid] != expected_burn_hotkey:
        raise WeightComputationError("burn target hotkey ownership mismatch")

    allocation_doc = snapshot["research_lab_allocation_doc"]
    fallback_lab_share = _non_negative_float(snapshot["research_lab_fallback_share"], "research_lab_fallback_share")
    research_lab_share = _doc_percent_share(allocation_doc, "lab_cap_percent", fallback_lab_share)
    base_burn_share = _non_negative_float(snapshot["base_burn_share"], "base_burn_share")
    champion_share = _non_negative_float(snapshot["champion_share"], "champion_share")
    effective_champion_share = _non_negative_float(
        snapshot["effective_champion_share"], "effective_champion_share"
    )
    champion_uid_value = snapshot.get("champion_uid")
    champion_uid = None if champion_uid_value is None else _int(champion_uid_value, "champion_uid")
    leaderboard_share = _non_negative_float(snapshot["leaderboard_bonus_share"], "leaderboard_bonus_share")
    fulfillment_pool_share = max(0.0, 1.0 - research_lab_share - champion_share - leaderboard_share)
    max_sourcing_share = (
        1.0
        - research_lab_share
        - champion_share
        - fulfillment_pool_share
        - leaderboard_share
    )

    rolling_scores_value = snapshot["rolling_scores"]
    if not isinstance(rolling_scores_value, list):
        raise WeightComputationError("rolling_scores must be a list")
    rolling_scores = []  # type: List[Tuple[str, float]]
    seen_hotkeys = set()
    banned = {str(item) for item in snapshot.get("banned_hotkeys") or []}
    for row in rolling_scores_value:
        if not isinstance(row, Mapping) or set(row) != {"hotkey", "score"}:
            raise WeightComputationError("rolling score row is invalid")
        hotkey = str(row["hotkey"])
        if hotkey in seen_hotkeys:
            raise WeightComputationError("rolling score hotkey is duplicated")
        seen_hotkeys.add(hotkey)
        score = _float(row["score"], "rolling score")
        if hotkey in banned and score > 0:
            raise WeightComputationError("banned hotkey has a positive sourcing score")
        rolling_scores.append((hotkey, score))

    hotkey_to_uid = {hotkey: index for index, hotkey in enumerate(metagraph_hotkeys)}
    all_rolling_total = sum(score for _, score in rolling_scores if score > 0)
    registered_rows = [
        (hotkey, score, hotkey_to_uid[hotkey])
        for hotkey, score in rolling_scores
        if hotkey in hotkey_to_uid
    ]
    registered_rolling_total = sum(score for _, score, _ in registered_rows if score > 0)
    deregistered_points = all_rolling_total - registered_rolling_total

    rolling_lead_count = _int(snapshot["rolling_lead_count"], "rolling_lead_count")
    sourcing_floor = _int(snapshot["sourcing_floor_threshold"], "sourcing_floor_threshold", minimum=1)
    if rolling_lead_count >= sourcing_floor:
        effective_sourcing_share = max_sourcing_share
    else:
        effective_sourcing_share = (rolling_lead_count / sourcing_floor) * max_sourcing_share
    dereg_burn = 0.0
    if all_rolling_total > 0 and deregistered_points > 0:
        dereg_burn = effective_sourcing_share * (deregistered_points / all_rolling_total)
    effective_sourcing_to_miners = effective_sourcing_share - dereg_burn

    ff_enabled = bool(snapshot["ff_enabled"])
    fulfillment_share = _non_negative_float(snapshot["fulfillment_share"], "fulfillment_share")
    fulfillment_rows = snapshot["fulfillment_rows"]
    if not isinstance(fulfillment_rows, list):
        raise WeightComputationError("fulfillment_rows must be a list")
    if not snapshot["fulfillment_fetch_ok"]:
        fulfillment_share = 0.0
        fulfillment_rows = []
    if not ff_enabled:
        fulfillment_share = 0.0
        fulfillment_rows = []
    unused_fulfillment = fulfillment_pool_share - fulfillment_share

    rank_shares = snapshot["leaderboard_rank_shares"]
    leaderboard_entries = snapshot["leaderboard_entries"]
    if not isinstance(rank_shares, list) or not isinstance(leaderboard_entries, list):
        raise WeightComputationError("leaderboard inputs must be lists")
    rank_shares = [_non_negative_float(value, "leaderboard rank share") for value in rank_shares]
    leaderboard_weights = {}  # type: Dict[int, float]
    leaderboard_burn = 0.0 if ff_enabled else leaderboard_share
    if ff_enabled:
        if not snapshot["leaderboard_fetch_ok"]:
            leaderboard_burn = leaderboard_share
        else:
            for index, rank_share in enumerate(rank_shares):
                if index >= len(leaderboard_entries):
                    leaderboard_burn += rank_share
                    continue
                entry = leaderboard_entries[index]
                if not isinstance(entry, Mapping):
                    raise WeightComputationError("leaderboard entry is invalid")
                hotkey = str(entry.get("miner_hotkey") or "")
                if hotkey in hotkey_to_uid:
                    _add_weight(leaderboard_weights, hotkey_to_uid[hotkey], rank_share)
                else:
                    leaderboard_burn += rank_share

    research_lab_weights, research_lab_burn, research_lab_breakdown = research_lab_uid_weights_from_allocation(
        allocation_doc,
        metagraph_hotkeys=metagraph_hotkeys,
        reserved_share=research_lab_share,
    )
    unused_sourcing_share = max_sourcing_share - effective_sourcing_share
    unused_champion = champion_share - effective_champion_share
    total_burn_share = (
        base_burn_share
        + unused_sourcing_share
        + unused_champion
        + dereg_burn
        + unused_fulfillment
        + leaderboard_burn
        + research_lab_burn
    )

    uid_weights = {burn_uid: total_burn_share}
    if effective_champion_share > 0 and champion_uid is not None:
        _add_weight(uid_weights, champion_uid, effective_champion_share)
    for row in fulfillment_rows:
        if not isinstance(row, Mapping) or set(row) != {"hotkey", "share"}:
            raise WeightComputationError("fulfillment row is invalid")
        hotkey = str(row["hotkey"])
        share = _non_negative_float(row["share"], "fulfillment row share")
        if hotkey in hotkey_to_uid:
            _add_weight(uid_weights, hotkey_to_uid[hotkey], share)
        else:
            uid_weights[burn_uid] = uid_weights.get(burn_uid, 0.0) + share
    for uid, share in leaderboard_weights.items():
        _add_weight(uid_weights, uid, share)
    for uid, share in research_lab_weights.items():
        _add_weight(uid_weights, uid, share)

    minimum_rep = _non_negative_float(
        snapshot["min_total_rep_for_distribution"], "min_total_rep_for_distribution"
    )
    if registered_rolling_total < minimum_rep:
        uid_weights[burn_uid] += effective_sourcing_to_miners
    else:
        for _, score, uid in registered_rows:
            if score <= 0:
                continue
            miner_proportion = score / registered_rolling_total
            _add_weight(uid_weights, uid, effective_sourcing_to_miners * miner_proportion)

    uids = list(uid_weights.keys())
    unnormalized_weights = list(uid_weights.values())
    weight_sum = sum(unnormalized_weights)
    if not (0.999 <= weight_sum <= 1.001):
        raise WeightComputationError("weights do not sum to one before normalization")
    weights = [max(0.0, float(weight)) for weight in unnormalized_weights]
    normalized_total = sum(weights)
    if normalized_total <= 0:
        raise WeightComputationError("sanitized weights sum to zero")
    weights = [weight / normalized_total for weight in weights]
    if any(weight < 0 for weight in weights):
        raise WeightComputationError("negative weight remained after sanitization")

    non_zero_pairs = sorted(
        ((uid, weight) for uid, weight in zip(uids, weights) if weight > 0),
        key=lambda pair: pair[0],
    )
    sparse_uids, sparse_weights_u16 = normalize_to_u16_with_uids_pure(
        [pair[0] for pair in non_zero_pairs],
        [pair[1] for pair in non_zero_pairs],
    )
    weights_hash = bundle_weights_hash(
        netuid,
        epoch_id,
        block,
        list(zip(sparse_uids, sparse_weights_u16)),
    )
    return {
        "schema_version": WEIGHT_RESULT_SCHEMA_VERSION,
        "snapshot_hash": sha256_json(dict(snapshot)),
        "netuid": netuid,
        "epoch_id": epoch_id,
        "block": block,
        "uids": uids,
        "weights": weights,
        "weight_float_bits": [struct.pack("!d", weight).hex() for weight in weights],
        "sparse_uids": sparse_uids,
        "sparse_weights_u16": sparse_weights_u16,
        "weights_hash": weights_hash,
        "components": {
            "research_lab_share": research_lab_share,
            "research_lab_burn": research_lab_burn,
            "research_lab_paid": research_lab_breakdown["paid"],
            "fulfillment_pool_share": fulfillment_pool_share,
            "fulfillment_share": fulfillment_share,
            "leaderboard_burn": leaderboard_burn,
            "max_sourcing_share": max_sourcing_share,
            "effective_sourcing_share": effective_sourcing_share,
            "deregistered_sourcing_burn": dereg_burn,
            "total_burn_share_before_deregistered_fulfillment": total_burn_share,
        },
    }
