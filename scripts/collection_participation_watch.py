"""Read private collection counters and compare three public validator vectors. Never dispatch work."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from scripts.lab_arena_restart_claim_guard import _read_scoped_environment

VALIDATORS = {
    "primary": "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
    "rizzo": "5GeEtY7Ars1FbSKhueoffzLHrYgw3rxn7eF6egicCbWmNs71",
    "yuma": "5Chnr6Y72gdfTFdoZnsCvkndKpMk8jAtt9JAYKaNG3LmU4BW",
}


def validator_snapshot(env, db):
    from lab_arena import chain, signing
    from lab_arena.store import ArenaStore
    from lab_arena.local_weight_signer import HttpsJsonRpcTransport
    from Leadpoet.utils.subnet_epoch import OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT
    from leadpoet_canonical.arena_weights import compare_weights_hash, derive_arena_weights, verify_accepted_weight_state_signature
    from leadpoet_canonical.chain_source_v2 import (
        CHAIN_RPC_METHOD, decode_selective_metagraph_result, decode_weights_storage,
        encode_selective_metagraph_params, weights_storage_key,
    )

    config = chain.ArenaChainConfig(env["LAB_ARENA_CHAIN_ENDPOINT"], 71, "finney", 15)
    reader = chain.ArenaChain(config, chain.connect_substrate(config))
    try:
        snap = chain.finalized_epoch_snapshot(reader)
    finally:
        reader.close()
    epoch = snap.settlement_epoch_id(chain.load_arena_cutover(env))
    state = ArenaStore(db).get_weight_state("finney", 71, epoch)
    if not state:
        return {"block": snap.current_block, "epoch": epoch, "signed_state": False}
    key = signing.KmsSigner(env["LAB_ARENA_SIGNING_KEY_ID"], region_name="us-east-1")
    doc = state["state_doc"]
    verify_accepted_weight_state_signature(doc, public_key_der=key.public_key_der, expected_public_key_hash=key.public_key_hash)
    rpc = HttpsJsonRpcTransport(OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT, timeout_seconds=15)
    block = snap.block_hash
    meta = rpc.call(method="state_call", params=[CHAIN_RPC_METHOD, encode_selective_metagraph_params(netuid=71), block], request_id=1)
    hotkeys = decode_selective_metagraph_result(meta)["hotkeys"]
    expected = derive_arena_weights(doc, hotkeys)
    matches = {}
    for name, hotkey in VALIDATORS.items():
        if hotkey not in hotkeys:
            matches[name] = "unregistered"
            continue
        storage = weights_storage_key(netuid=71, validator_uid=hotkeys.index(hotkey))
        vector = decode_weights_storage(rpc.call(method="state_getStorage", params=[storage, block], request_id=2))
        matches[name] = "matches" if compare_weights_hash(71, epoch, vector) == expected["weights_hash"] else "differs"
    basis = doc["reward_basis"]
    proof = {"basis_schema": basis["schema_version"], "quoted_jobs_match": None, "unweighted_funders": None}
    if "collection_share_ppb" in expected:
        quotes = db.rpc("collection_reward_quotes_v1", {"p_epoch": epoch})
        proof["quoted_jobs_match"] = {k: v for k, v in basis.items() if k != "valuation"} == quotes
        weighted = {hotkeys[i] for i in expected["sparse_uids"]}
        proof["unweighted_funders"] = len(set(quotes["quotes_microusd"]) - weighted)
    return {"block": snap.current_block, "block_hash": block, "epoch": epoch, "signed_state": True, **proof,
            "validators": matches, "champion_ppb": expected["champion_share_ppb"],
            "collection_ppb": expected.get("collection_share_ppb", 0), "burn_ppb": expected["burned_residual_ppb"]}


def worker_snapshot():
    timer = "leadpoet-collection-pilot.timer"
    service = "leadpoet-collection-pilot.service"
    states = {}
    for unit, fields in [(timer, ("ActiveState", "UnitFileState")), (service, ("ActiveState", "Result"))]:
        output = subprocess.check_output(["systemctl", "show", unit, *["-p" + k for k in fields]], text=True, timeout=10)
        states[unit] = dict(line.split("=", 1) for line in output.splitlines())
    logs = subprocess.check_output(["journalctl", "-u", service, "--since", "20 min ago",
                                    "--no-pager", "-n", "300", "-o", "json"], text=True, timeout=10)
    arms = {}
    recent_failure = False
    for line in logs.splitlines():
        try:
            entry = json.loads(line)
            if entry.get("UNIT_RESULT") not in (None, "success") or entry.get("MESSAGE_ID") == "be02cf6855d2428ba40df7e9d022f03d":
                recent_failure = True
            row = json.loads(entry.get("MESSAGE", ""))
        except (TypeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        status = row.get("status")
        if status in ("correction_held", "comparison_held"):
            arms[status.removesuffix("_held")] = "held"
        elif status in ("idle", "completed") and isinstance(row.get("batch"), str):
            arms["correction" if row["batch"].startswith("xp-") else "comparison"] = status
    timer_ok = states[timer] == {"ActiveState": "active", "UnitFileState": "enabled"}
    return {"timer_enabled_active": timer_ok, "worker": states[service]["ActiveState"],
            "result": states[service]["Result"], "recent_arms": arms, "unit_failed_recently": recent_failure}


def notices(data):
    """Milestones are monotone where possible. Keep idle polls and ordinary blocks quiet."""
    signals = {}
    for name in ("collection", "worker", "weights"):
        if name not in data:
            signals[name + " read unavailable"] = 1
    collection = data.get("collection", {})
    for row in collection.get("enrollments", []):
        label = f"{row['mode']} {row['arm']}"
        signals[label + " enrolled"] = bool(row["registrations"])
        signals[label + " active enrollments"] = row["active"]
    for row in collection.get("work", []):
        label = f"{row['mode']} {row['arm']}"
        signals[label + " first assignment"] = bool(row["assigned"])
        signals[label + " first artifact; inspect stored evidence"] = bool(row["artifacts"])
        for name in ("expired", "failed", "rejected", "ready"):
            signals[label + " " + name + "; inspect job"] = row[name]
        if row["mode"] == "live":
            signals[label + " accepted; verify settlement"] = bool(row["accepted"])
    for row in collection.get("billing", []):
        label = f"{row['mode']} {row['arm']}"
        signals[label + " first billed execution"] = bool(row["funded_runs"])
        signals[label + " bills unresolved after two hours"] = row["unresolved"]
    signals["completed candidate packs; verify ordinary runs and controls for blind review"] = collection.get("completed_review_candidate_packs")
    for name in ("staged", "first_epoch", "correction_quotes"):
        signals["collection " + name] = collection.get(name)
    worker = data.get("worker")
    if worker:
        if not worker["timer_enabled_active"] or worker["result"] != "success" or worker["unit_failed_recently"]:
            signals["inspect pilot unit and timer"] = 1
        for arm in ("comparison", "correction"):
            recent = worker["recent_arms"].get(arm)
            if recent == "held":
                signals["inspect " + arm + " admission hold"] = 1
        if not worker["recent_arms"] and worker["worker"] not in ("activating", "active"):
            signals["no queue readback in 20 minutes"] = 1
    weights = data.get("weights")
    if weights:
        if not weights["signed_state"]:
            signals["current signed state pending; inspect validator publication"] = 1
        for validator, state in weights.get("validators", {}).items():
            if state != "matches":
                signals[validator + " finalized vector " + state] = 1
        signals["collection allocation is nonzero; verify recipients"] = bool(weights.get("collection_ppb"))
        cutover = collection.get("first_epoch")
        if cutover is not None and weights["signed_state"] and weights["epoch"] >= cutover:
            matches = weights.get("validators", {})
            verified = (weights.get("basis_schema") == "leadpoet.collection.fixed_run_quotes.v1"
                        and weights.get("quoted_jobs_match") is True and weights.get("unweighted_funders") == 0
                        and weights.get("champion_ppb") == 0
                        and all(matches.get(name) == "matches" for name in VALIDATORS))
            signals["post-cutover collection basis and zero champion component verified on all three validators"] = verified
            signals["inspect post-cutover reward evidence"] = not verified
    return {key: value for key, value in signals.items() if value}


def save(directory, data):
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / "status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    # An unread source means unknown. Keep its last signals until it can be read again.
    previous_observations = previous.get("last_read", {})
    last_read = {name: data.get(name, previous_observations.get(name)) for name in ("collection", "worker", "weights")}
    current = notices({name: value for name, value in last_read.items() if value is not None})
    for name in last_read:
        if name not in data:
            current[name + " read unavailable"] = 1
    previous_signals = previous.get("signals", {})
    differences = sorted(k for k in current.keys() | previous_signals.keys() if current.get(k) != previous_signals.get(k))
    now = datetime.now(timezone.utc).isoformat()
    event = previous.get("event")
    if differences or not previous:
        event = {"number": (event or {}).get("number", 0) + 1, "at": now,
                 "changes": {key: current.get(key, 0) for key in differences}}
        print(json.dumps(event), flush=True)
    result = dict(data, observed_at=now, last_read=last_read, signals=current, event=event)
    temporary = directory / ".status.new"
    with temporary.open("w") as stream:
        json.dump(result, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    from lab_arena.store import PostgrestTransport

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-file", type=Path, required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    data = {"read_errors": {}}
    try:
        env = _read_scoped_environment(args.environment_file)
        db = PostgrestTransport(env["LAB_ARENA_SUPABASE_URL"], service_key=env["LAB_ARENA_SERVICE_KEY"], timeout_seconds=15)
        data["collection"] = db.rpc("collection_participation_v1", {})
    except Exception as exc:
        data["read_errors"]["collection"] = type(exc).__name__
    try:
        data["worker"] = worker_snapshot()
    except Exception as exc:
        data["read_errors"]["worker"] = type(exc).__name__
    try:
        if "collection" in data:
            data["weights"] = validator_snapshot(env, db)
    except Exception as exc:
        data["read_errors"]["weights"] = type(exc).__name__
    save(args.state_directory, data)
    return bool(data["read_errors"])


if __name__ == "__main__":
    raise SystemExit(main())
