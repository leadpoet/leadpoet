import random
import struct

from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    compute_final_weights,
    weight_config_hash,
)


def _bits(values):
    return [struct.pack("!d", float(value)).hex() for value in values]


def _legacy_lab_allocation(doc, hotkeys, reserved_share):
    if not isinstance(doc, dict) or not doc:
        return {}, reserved_share
    cap = max(0.0, min(reserved_share, float(doc.get("lab_cap_percent") or 0.0) / 100.0))
    unallocated = min(cap, max(0.0, float(doc.get("unallocated_percent") or 0.0) / 100.0))
    uid_weights = {}
    paid = 0.0
    deregistered = 0.0
    for section in ("reimbursement_allocations", "champion_allocations", "queued_champion_allocations"):
        for row in doc.get(section) or []:
            if not isinstance(row, dict):
                continue
            share = max(0.0, float(row.get("paid_alpha_percent") or 0.0) / 100.0)
            if share <= 0:
                continue
            paid += share
            try:
                uid = int(row.get("uid"))
                expected_hotkey = str(row.get("miner_hotkey") or "")
                actual_hotkey = hotkeys[uid]
            except Exception:
                deregistered += share
                continue
            if expected_hotkey and actual_hotkey != expected_hotkey:
                deregistered += share
                continue
            uid_weights[uid] = uid_weights.get(uid, 0.0) + share
    payable_cap = max(0.0, cap - unallocated)
    if paid > payable_cap and paid > 0:
        scale = payable_cap / paid
        uid_weights = {uid: weight * scale for uid, weight in uid_weights.items()}
        deregistered *= scale
        paid = payable_cap
    reported_total = paid + unallocated
    rounding_gap = max(0.0, cap - reported_total)
    reserved_gap = max(0.0, reserved_share - cap)
    return uid_weights, unallocated + deregistered + rounding_gap + reserved_gap


def _legacy_current_host_formula(snapshot):
    hotkeys = snapshot["metagraph_hotkeys"]
    burn_uid = snapshot["burn_target_uid"]
    lab_doc = snapshot["research_lab_allocation_doc"]
    if isinstance(lab_doc, dict) and lab_doc.get("lab_cap_percent") not in (None, ""):
        lab_share = max(0.0, min(1.0, float(lab_doc["lab_cap_percent"]) / 100.0))
    else:
        lab_share = snapshot["research_lab_fallback_share"]
    champion_share = snapshot["champion_share"]
    leaderboard_share = snapshot["leaderboard_bonus_share"]
    fulfillment_pool = max(0.0, 1.0 - lab_share - champion_share - leaderboard_share)
    max_sourcing = 1.0 - lab_share - champion_share - fulfillment_pool - leaderboard_share

    rolling_scores = {row["hotkey"]: float(row["score"]) for row in snapshot["rolling_scores"]}
    registered = {hotkey: score for hotkey, score in rolling_scores.items() if hotkey in hotkeys}
    all_total = sum(score for score in rolling_scores.values() if score > 0)
    registered_total = sum(score for score in registered.values() if score > 0)
    deregistered_points = all_total - registered_total
    if snapshot["rolling_lead_count"] >= snapshot["sourcing_floor_threshold"]:
        effective_sourcing = max_sourcing
    else:
        effective_sourcing = (
            snapshot["rolling_lead_count"] / snapshot["sourcing_floor_threshold"]
        ) * max_sourcing
    dereg_burn = (
        effective_sourcing * (deregistered_points / all_total)
        if all_total > 0 and deregistered_points > 0
        else 0.0
    )
    sourcing_to_miners = effective_sourcing - dereg_burn

    ff_enabled = snapshot["ff_enabled"]
    fulfillment_share = snapshot["fulfillment_share"] if ff_enabled and snapshot["fulfillment_fetch_ok"] else 0.0
    fulfillment_rows = snapshot["fulfillment_rows"] if ff_enabled and snapshot["fulfillment_fetch_ok"] else []
    unused_fulfillment = fulfillment_pool - fulfillment_share

    leaderboard_by_uid = {}
    leaderboard_burn = 0.0 if ff_enabled else leaderboard_share
    if ff_enabled:
        if not snapshot["leaderboard_fetch_ok"]:
            leaderboard_burn = leaderboard_share
        else:
            for index, rank_share in enumerate(snapshot["leaderboard_rank_shares"]):
                if index >= len(snapshot["leaderboard_entries"]):
                    leaderboard_burn += rank_share
                    continue
                hotkey = str(snapshot["leaderboard_entries"][index].get("miner_hotkey") or "")
                if hotkey in hotkeys:
                    uid = hotkeys.index(hotkey)
                    leaderboard_by_uid[uid] = leaderboard_by_uid.get(uid, 0.0) + rank_share
                else:
                    leaderboard_burn += rank_share

    lab_by_uid, lab_burn = _legacy_lab_allocation(lab_doc, hotkeys, lab_share)
    total_burn = (
        snapshot["base_burn_share"]
        + (max_sourcing - effective_sourcing)
        + (champion_share - snapshot["effective_champion_share"])
        + dereg_burn
        + unused_fulfillment
        + leaderboard_burn
        + lab_burn
    )
    uid_weights = {burn_uid: total_burn}
    champion_uid = snapshot["champion_uid"]
    if snapshot["effective_champion_share"] > 0 and champion_uid is not None:
        uid_weights[champion_uid] = uid_weights.get(champion_uid, 0.0) + snapshot["effective_champion_share"]
    for row in fulfillment_rows:
        share = float(row["share"])
        if row["hotkey"] in hotkeys:
            uid = hotkeys.index(row["hotkey"])
            uid_weights[uid] = uid_weights.get(uid, 0.0) + share
        else:
            uid_weights[burn_uid] = uid_weights.get(burn_uid, 0.0) + share
    for uid, share in leaderboard_by_uid.items():
        uid_weights[uid] = uid_weights.get(uid, 0.0) + share
    for uid, share in lab_by_uid.items():
        uid_weights[uid] = uid_weights.get(uid, 0.0) + share
    if registered_total < snapshot["min_total_rep_for_distribution"]:
        uid_weights[burn_uid] += sourcing_to_miners
    else:
        for hotkey, score in registered.items():
            if score <= 0:
                continue
            uid = hotkeys.index(hotkey)
            uid_weights[uid] = uid_weights.get(uid, 0.0) + sourcing_to_miners * (score / registered_total)
    uids = list(uid_weights)
    weights = [max(0.0, float(uid_weights[uid])) for uid in uids]
    total = sum(weights)
    return uids, [weight / total for weight in weights]


def _random_snapshot(rng, case):
    size = rng.randint(4, 40)
    hotkeys = ["burn"] + ["hotkey-%s" % index for index in range(1, size)]
    ff_enabled = rng.choice([True, False])
    lab_cap = rng.choice([0.0, 10.0, 20.0])
    leaderboard_share = 0.095
    champion_share = rng.choice([0.0, 0.1])
    pool = max(0.0, 1.0 - lab_cap / 100.0 - champion_share - leaderboard_share)
    fulfillment_share = pool * rng.random() if ff_enabled else 0.0
    ff_rows = []
    if fulfillment_share > 0:
        first = fulfillment_share * rng.random()
        ff_rows = [
            {"hotkey": rng.choice(hotkeys[1:] + ["deregistered-ff"]), "share": first},
            {"hotkey": rng.choice(hotkeys[1:] + ["deregistered-ff-2"]), "share": fulfillment_share - first},
        ]
    allocation_rows = []
    paid_percent = min(lab_cap, lab_cap * rng.random())
    if paid_percent > 0:
        uid = rng.randrange(1, size)
        allocation_rows.append({
            "uid": uid,
            "miner_hotkey": hotkeys[uid] if rng.random() < 0.8 else "wrong-hotkey",
            "paid_alpha_percent": paid_percent,
        })
    allocation = {
        "lab_cap_percent": lab_cap,
        "unallocated_percent": max(0.0, lab_cap - paid_percent),
        "reimbursement_allocations": allocation_rows,
        "champion_allocations": [],
        "queued_champion_allocations": [],
    }
    leaderboard_entries = [
        {"miner_hotkey": rng.choice(hotkeys[1:] + ["deregistered-lb"]), "wins": rng.randint(1, 20)}
        for _ in range(rng.randint(0, 3))
    ]
    rolling_scores = []
    for index in range(rng.randint(0, size + 4)):
        hotkey = hotkeys[rng.randrange(1, size)] if index < size else "deregistered-%s" % index
        if any(row["hotkey"] == hotkey for row in rolling_scores):
            continue
        rolling_scores.append({"hotkey": hotkey, "score": rng.randint(-100000, 1000)})
    champion_uid = rng.randrange(1, size) if champion_share and rng.random() < 0.5 else None
    effective_champion = champion_share if champion_uid is not None else 0.0
    snapshot = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
        "netuid": 71,
        "epoch_id": 1000 + case,
        "block": (1000 + case) * 360 + 350,
        "commit_sha": "%040x" % (case + 1),
        "config_hash": "",
        "parent_receipt_hashes": [],
        "research_lab_allocation_receipt_hash": "",
        "burn_target_uid": 0,
        "expected_burn_target_hotkey": "burn",
        "metagraph_hotkeys": hotkeys,
        "banned_hotkeys": [],
        "banned_lookup_ok": True,
        "ff_enabled": ff_enabled,
        "base_burn_share": 0.0,
        "champion_share": champion_share,
        "champion_uid": champion_uid,
        "effective_champion_share": effective_champion,
        "research_lab_fallback_share": 0.2,
        "research_lab_allocation_doc": allocation,
        "leaderboard_bonus_share": leaderboard_share,
        "leaderboard_rank_shares": [0.05, 0.03, 0.015],
        "leaderboard_entries": leaderboard_entries,
        "leaderboard_fetch_ok": rng.choice([True, True, False]),
        "fulfillment_share": fulfillment_share,
        "fulfillment_rows": ff_rows,
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": rng.randint(0, 200000),
        "rolling_scores": rolling_scores,
        "sourcing_floor_threshold": 125000,
        "min_total_rep_for_distribution": 100,
    }
    snapshot["config_hash"] = weight_config_hash(snapshot)
    return snapshot


def test_enclave_core_matches_current_host_formula_bit_for_bit_for_2000_cases():
    rng = random.Random(71020260710)
    for case in range(2000):
        snapshot = _random_snapshot(rng, case)
        expected_uids, expected_weights = _legacy_current_host_formula(snapshot)
        actual = compute_final_weights(snapshot)
        assert actual["uids"] == expected_uids, case
        assert actual["weight_float_bits"] == _bits(expected_weights), case
