"""Read collection counters, finalized validator weights and closed-epoch miner emissions."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import signal
import time

from lab_arena import collection_distributions, collection_settlement
from scripts.lab_arena_restart_claim_guard import _read_scoped_environment

VALIDATORS = {
    "primary": "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9",
    "rizzo": "5GeEtY7Ars1FbSKhueoffzLHrYgw3rxn7eF6egicCbWmNs71",
    "yuma": "5Chnr6Y72gdfTFdoZnsCvkndKpMk8jAtt9JAYKaNG3LmU4BW",
}


def pending_reveals(rpc, store, signing_key, cutover, snapshot, current, hotkeys, matches):
    """A new encrypted commitment can coexist with the preceding epoch's revealed weights."""
    from leadpoet_canonical.arena_weights import derive_arena_weights, verify_accepted_weight_state_signature
    from leadpoet_canonical.chain_source_v2 import (
        decode_timelocked_weight_commits, ss58_encode_account_id, timelocked_weight_commits_storage_key,
    )

    epoch = current["epoch"]
    previous = store.get_weight_state("finney", 71, epoch - 1)
    if not previous:
        return matches
    previous = previous["state_doc"]
    verify_accepted_weight_state_signature(previous, public_key_der=signing_key.public_key_der,
                                           expected_public_key_hash=signing_key.public_key_hash)
    if (previous["epoch"] != epoch - 1 or previous["valid_until_block"] + 1 != current["valid_from_block"]
            or not current["valid_from_block"] <= snapshot.current_block <= current["valid_until_block"]
            or any(previous[k] != current[k] for k in ("network", "netuid", "genesis_hash", "burn_hotkey"))):
        raise ValueError("preceding signed state does not cover the prior epoch")
    number = previous["valid_until_block"]
    parent = rpc.call(method="chain_getBlockHash", params=[number], request_id=3)
    old_registry = collection_distributions.registry_at(rpc, number, parent)
    expected = derive_arena_weights(previous, old_registry)
    if (expected["champion_share_ppb"] != 0 or
            any(i >= len(hotkeys) or hotkeys[i] != old_registry[i] for i in expected["sparse_uids"])):
        return matches
    applied = collection_distributions.validator_vectors(
        rpc, snapshot.block_hash, hotkeys, epoch - 1, expected["weights_hash"], VALIDATORS)
    key = timelocked_weight_commits_storage_key(netuid=71, subnet_epoch_index=cutover.subnet_epoch_index(epoch))
    queue = decode_timelocked_weight_commits(rpc.call(method="state_getStorage", params=[key, snapshot.block_hash],
                                                      request_id=4))
    committed = Counter(ss58_encode_account_id(bytes.fromhex(entry["hotkey_public_key"])) for entry in queue
                        if current["valid_from_block"] <= entry["submitted_at"] <= snapshot.current_block)
    result = dict(matches)
    for name, status in matches.items():
        if status == "differs" and applied[name] == "matches" and committed[VALIDATORS[name]] == 1:
            result[name] = "pending_reveal"
    return result


def validator_snapshot(env, db):
    from lab_arena import chain, signing
    from lab_arena.store import ArenaStore
    from lab_arena.local_weight_signer import HttpsJsonRpcTransport
    from Leadpoet.utils.subnet_epoch import OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT
    from leadpoet_canonical.arena_weights import COLLECTION_BASIS_SCHEMA, derive_arena_weights, verify_accepted_weight_state_signature

    config = chain.ArenaChainConfig(env["LAB_ARENA_CHAIN_ENDPOINT"], 71, "finney", 15)
    reader = chain.ArenaChain(config, chain.connect_substrate(config))
    try:
        snap = chain.finalized_epoch_snapshot(reader)
    finally:
        reader.close()
    cutover = chain.load_arena_cutover(env)
    epoch = snap.settlement_epoch_id(cutover)
    store = ArenaStore(db)
    weight_start = store.collection_reward_start_epoch()
    state = store.get_weight_state("finney", 71, epoch)
    if not state:
        return {"block": snap.current_block, "epoch": epoch, "last_epoch_block": snap.last_epoch_block, "weight_start_epoch": weight_start, "signed_state": False}
    key = signing.KmsSigner(env["LAB_ARENA_SIGNING_KEY_ID"], region_name="us-east-1")
    doc = state["state_doc"]
    verify_accepted_weight_state_signature(doc, public_key_der=key.public_key_der, expected_public_key_hash=key.public_key_hash)
    scope = ("finney", 71, snap.network_genesis_hash.removeprefix("0x"))
    if (doc["network"], doc["netuid"], doc["genesis_hash"]) != scope or doc["epoch"] != epoch:
        raise ValueError("signed state does not belong to this finalized epoch")
    rpc = HttpsJsonRpcTransport(OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT, timeout_seconds=15)
    block = snap.block_hash
    hotkeys = collection_distributions.registry_at(rpc, snap.current_block, block)
    expected = derive_arena_weights(doc, hotkeys)
    matches = collection_distributions.validator_vectors(rpc, block, hotkeys, epoch, expected["weights_hash"], VALIDATORS)
    basis = doc["reward_basis"]
    if (basis["schema_version"] == COLLECTION_BASIS_SCHEMA and weight_start is not None
            and epoch > weight_start and "differs" in matches.values()):
        matches = pending_reveals(rpc, store, key, cutover, snap, doc, hotkeys, matches)
    proof = {"basis_schema": basis["schema_version"], "quoted_jobs_match": None, "unweighted_funders": None}
    if "collection_share_ppb" in expected:
        quotes = db.rpc("collection_reward_quotes_v1", {"p_epoch": epoch})
        proof["quoted_jobs_match"] = {k: v for k, v in basis.items() if k != "valuation"} == quotes
        weighted = {hotkeys[i] for i in expected["sparse_uids"]}
        proof["unweighted_funders"] = len(set(quotes["quotes_microusd"]) - weighted)
        if weight_start is not None and epoch >= weight_start and basis["job_count"] == 0:
            prior = store.get_weight_state("finney", 71, weight_start - 1)
            proof["signed_burn_only"] = False
            if prior:
                previous = prior["state_doc"]
                verify_accepted_weight_state_signature(previous, public_key_der=key.public_key_der,
                                                       expected_public_key_hash=key.public_key_hash)
                proof["signed_burn_only"] = (
                    previous["epoch"] == weight_start - 1
                    and (previous["network"], previous["netuid"], previous["genesis_hash"]) == scope
                    and doc["burn_hotkey"] == previous["burn_hotkey"] and weighted == {doc["burn_hotkey"]}
                    and expected["sparse_weights_u16"] == [65535] and expected["burned_residual_ppb"] == 1_000_000_000)
    return {"block": snap.current_block, "block_hash": block, "epoch": epoch, "last_epoch_block": snap.last_epoch_block, "weight_start_epoch": weight_start,
            "signed_state": True, **proof,
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
        if isinstance(row.get("batch"), str) and row["batch"].startswith("pool-"):
            if status in ("accepted", "rejected", "failed", "idle", "pool_unavailable"):
                arms["pool"] = status
        elif status in ("correction_held", "comparison_held"):
            arms[status.removesuffix("_held")] = "held"
        elif status in ("idle", "completed") and isinstance(row.get("batch"), str):
            arms["correction" if row["batch"].startswith("xp-") else "comparison"] = status
    timer_ok = states[timer] == {"ActiveState": "active", "UnitFileState": "enabled"}
    return {"timer_enabled_active": timer_ok, "worker": states[service]["ActiveState"],
            "result": states[service]["Result"], "recent_arms": arms, "unit_failed_recently": recent_failure}


_EDGE_REQUEST = re.compile(rb'\b(GET|POST) /arena/v1/collection/(terms|register|revoke|statement)(?:\?[^\s"]*)? HTTP/[\d.]+"\s+([1-5]\d\d)\b')


def edge_snapshot(directory, prior):
    """Keep bounded forward cursors. Public requests can include operator probes."""
    counts = dict(prior.get("http", {}))
    cursors = {}
    gaps = prior.get("windows_with_gap", 0)
    for slot in (0, 1):
        name = f"arena-edge-slot{slot}.log"
        try:
            stream = (directory / name).open("rb")
        except FileNotFoundError:
            continue
        with stream:
            st = os.fstat(stream.fileno())
            previous = prior.get("cursors", {}).get(name)
            # The baseline begins at installation. Historical reads require a separate scope.
            offset = st.st_size if not previous else previous[1]
            if previous and (previous[0] != st.st_ino or offset > st.st_size):
                gaps += 1
                offset = 0
            if st.st_size - offset > 4 * 1024 * 1024:
                gaps += 1
                offset = st.st_size - 4 * 1024 * 1024
            stream.seek(offset)
            chunk = stream.read(st.st_size - offset)
            complete = chunk.rfind(b"\n") + 1
            for verb, op, status in _EDGE_REQUEST.findall(chunk[:complete]):
                if (verb == b"GET") == (op == b"terms"):
                    key = b" ".join((verb, op, status)).decode("ascii")
                    counts[key] = counts.get(key, 0) + 1
            cursors[name] = [st.st_ino, offset + complete]
    if not cursors:
        raise FileNotFoundError("no edge log")
    return {"http": counts, "cursors": cursors, "windows_with_gap": gaps}


def notices(data):
    """Milestones are monotone where possible. Keep idle polls and ordinary blocks quiet."""
    signals = {}
    for name in ("collection", "worker", "weights", "edge", "distribution", "settlement"):
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
    edge = data.get("edge", {})
    for name, count in edge.get("http", {}).items():
        if name.startswith("POST register ") and int(name[-3:]) >= 400:
            signals["public registration " + name[-3:] + " responses; check intake"] = count
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
            if state not in ("matches", "pending_reveal"):
                signals[validator + " finalized vector " + state] = 1
        signals["collection allocation is nonzero; verify recipients"] = bool(weights.get("collection_ppb"))
        paid_start = collection.get("first_epoch")
        weight_start = weights.get("weight_start_epoch") or paid_start
        signals["first collection-weight epoch"] = weight_start
        if weight_start is not None and weights["signed_state"] and weights["epoch"] >= weight_start:
            matches = weights.get("validators", {})
            valid_basis = (weights.get("basis_schema") == "leadpoet.collection.fixed_run_quotes.v1"
                           and weights.get("quoted_jobs_match") is True and weights.get("unweighted_funders") == 0
                           and weights.get("champion_ppb") == 0)
            verified = valid_basis and all(matches.get(name) == "matches" for name in VALIDATORS)
            if paid_start is None or weights["epoch"] < paid_start:
                verified = verified and weights.get("signed_burn_only") is True
                signals["retired champion weights verified at the designated burn hotkey on all three validators"] = verified
                signals["inspect burn-only retirement weights"] = not verified
            else:
                signals["post-cutover collection basis and zero champion component verified on all three validators"] = verified
                pending = ("pending_reveal" in matches.values()
                           and all(matches.get(name) in ("pending_reveal", "matches") for name in VALIDATORS))
                signals["validator commitments pending reveal"] = pending
                signals["inspect post-cutover reward evidence"] = not (verified or valid_basis and pending)
    closed = data.get("distribution", {})
    settlement = (data.get("settlement") or {}).get("last_paid") or {}
    settled_here = (settlement.get("result") == "collection_recipients_observed"
                    and settlement.get("distribution_epoch") == closed.get("epoch"))
    signals["closed miner distribution requires inspection"] = closed.get("result") == "inspect" and not settled_here
    signals["unreconciled paid collection distribution"] = settlement.get("result") == "inspect"
    signals["finalized miner distribution paid only the designated burn hotkey"] = closed.get("result") == "burn_only"
    signals["finalized collection recipients received miner emissions"] = settled_here or closed.get("result") == "collection_recipients_observed"
    signals["catching up on unobserved closed distributions"] = bool(closed.get("epochs_to_catch_up"))
    return {key: value for key, value in signals.items() if value}


def save(directory, data):
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / "status.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    # An unread source means unknown. Keep its last signals until it can be read again.
    previous_observations = previous.get("last_read", {})
    last_read = {name: data.get(name, previous_observations.get(name)) for name in ("collection", "worker", "weights", "edge", "distribution", "settlement")}
    if "exports" in data or "exports" in previous_observations:
        last_read["exports"] = data.get("exports", previous_observations.get("exports"))
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
    started = time.monotonic()
    data = {"read_errors": {}}
    db = None
    try:
        env = _read_scoped_environment(args.environment_file)
        db = PostgrestTransport(env["LAB_ARENA_SUPABASE_URL"], service_key=env["LAB_ARENA_SERVICE_KEY"], timeout_seconds=15)
        data["collection"] = db.rpc("collection_participation_v1", {})
    except Exception as exc:
        data["read_errors"]["collection"] = type(exc).__name__
    prior = {}
    try:
        status = args.state_directory / "status.json"
        prior = json.loads(status.read_text()).get("last_read", {}) if status.exists() else {}
        data["edge"] = edge_snapshot(Path("/home/ec2-user/gateway"), prior.get("edge") or {})
    except Exception as exc:
        data["read_errors"]["edge"] = type(exc).__name__
    try:
        data["worker"] = worker_snapshot()
    except Exception as exc:
        data["read_errors"]["worker"] = type(exc).__name__
    try:
        if db is not None:
            data["weights"] = validator_snapshot(env, db)
        else:
            data["read_errors"]["weights"] = "StoreUnavailable"
    except Exception as exc:
        data["read_errors"]["weights"] = type(exc).__name__
    try:
        if db is not None:
            data["distribution"] = collection_distributions.observe(
                env, db, args.state_directory, prior.get("distribution") or {}, data.get("weights"), VALIDATORS)
        else:
            data["read_errors"]["distribution"] = "StoreUnavailable"
    except Exception as exc:
        data["read_errors"]["distribution"] = type(exc).__name__
    try:
        if db is not None and "distribution" in data and "collection" in data:
            closed = data["distribution"].get("epoch")
            previous_settlement = prior.get("settlement") or {}
            # Keep the two expensive archive proofs in separate timer invocations.
            if closed != (prior.get("distribution") or {}).get("epoch"):
                data["settlement"] = previous_settlement
            else:
                data["settlement"] = collection_settlement.observe(
                    env, db, args.state_directory, previous_settlement, closed,
                    data["collection"].get("first_epoch"), VALIDATORS,
                )
    except Exception as exc:
        data["read_errors"]["settlement"] = type(exc).__name__
    private_root = args.state_directory.parent / "pilot"
    # Leave time for billing reconciliation in the same three-minute systemd unit.
    window = min(45, 120 - int(time.monotonic() - started))
    if db is not None and window > 0 and (private_root / "paid" / "batch.json").is_file():
        try:
            from lab_arena.service import S3ObjectStore
            from scripts.collection_export_snapshots import ExportWindowExpired, observe

            def timed_out(*_):
                raise ExportWindowExpired("private export tick exceeded its time budget")

            prior_alarm = signal.signal(signal.SIGALRM, timed_out)
            try:
                deadline = time.monotonic() + window
                signal.alarm(window)
                objects = S3ObjectStore(env["LAB_ARENA_BUCKET"], prefix=env.get("LAB_ARENA_OBJECT_PREFIX", ""))
                data["exports"] = observe(private_root, args.state_directory / "exports", db, objects,
                                          prior.get("exports"), deadline=deadline)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prior_alarm)
        except Exception as exc:
            data["read_errors"]["exports"] = type(exc).__name__
    save(args.state_directory, data)
    # Archive or training publication retries must not skip billing reconciliation.
    return bool(data["read_errors"].keys() - {"settlement", "exports"})


if __name__ == "__main__":
    raise SystemExit(main())
