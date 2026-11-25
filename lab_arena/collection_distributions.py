"""Verify the finalized miner distribution for an epoch that used collection weights."""

from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from Leadpoet.utils.subnet_epoch import OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT, read_subnet_epoch_snapshot
from lab_arena import chain, signing
from lab_arena.local_weight_signer import HttpsJsonRpcTransport
from lab_arena.store import ArenaStore
from leadpoet_canonical.arena_weights import (
    COLLECTION_BASIS_SCHEMA, compare_weights_hash, derive_arena_weights, verify_accepted_weight_state_signature,
)
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_RPC_METHOD, decode_selective_metagraph_result, decode_weights_storage,
    encode_selective_metagraph_params, weights_storage_key,
)


def registry_at(rpc, number, block_hash):
    meta = rpc.call(method="state_call", params=[CHAIN_RPC_METHOD, encode_selective_metagraph_params(netuid=71), block_hash],
                    request_id=1)
    registry = decode_selective_metagraph_result(meta)
    hotkeys = registry["hotkeys"]
    if registry["netuid"] != 71 or registry["block"] != number or not hotkeys or len(set(hotkeys)) != len(hotkeys):
        raise ValueError("invalid finalized SN71 recipient registry")
    return hotkeys


def validator_vectors(rpc, block_hash, hotkeys, epoch, expected, validators):
    matches = {}
    for name, hotkey in validators.items():
        if hotkey not in hotkeys:
            matches[name] = "unregistered"
            continue
        storage = weights_storage_key(netuid=71, validator_uid=hotkeys.index(hotkey))
        raw = rpc.call(method="state_getStorage", params=[storage, block_hash], request_id=2)
        vector = decode_weights_storage(raw)
        matches[name] = "matches" if compare_weights_hash(71, epoch, vector) == expected else "differs"
    return matches


def miner_events(events, width):
    distributions = {}
    for index, record in enumerate(events):
        event = record["event"]
        if (event["module_id"], event["event_id"]) != ("SubtensorModule", "IncentiveAlphaEmittedToMiners"):
            continue
        storage_netuid = event["attributes"]["netuid"]
        if type(storage_netuid) is not int or not 0 <= storage_netuid < 2**16:
            raise ValueError("invalid miner distribution network")
        if storage_netuid % 4096 != 71:
            continue
        mechanism = storage_netuid // 4096
        values = event["attributes"]["emissions"]
        if (record.get("phase") != "Initialization" or mechanism in distributions or not isinstance(values, list)
                or len(values) != width or any(type(v) is not int or not 0 <= v < 2**64 for v in values)):
            raise ValueError("ambiguous or invalid SN71 miner emission event")
        distributions[mechanism] = index, values
    if 0 not in distributions:
        raise ValueError("missing SN71 miner emission event")
    return distributions


def receipt_at(graph, rpc, db, signer, cutover, boundary, epoch, validators):
    """Read the parent and the boundary to bind identities, signed weights, and actual miner alpha."""
    parent_hash, block_hash = (graph.finalized_block_hash(n) for n in (boundary - 1, boundary))
    parent, closed = (read_subnet_epoch_snapshot(graph.epoch_subtensor(), netuid=71, block_hash=h)
                      for h in (parent_hash, block_hash))
    if (parent.settlement_epoch_id(cutover) != epoch or closed.settlement_epoch_id(cutover) != epoch + 1
            or closed.last_epoch_block != boundary or parent.current_block != boundary - 1 or closed.current_block != boundary):
        raise ValueError("block does not close the requested settlement epoch")
    store = ArenaStore(db)
    first = store.collection_reward_start_epoch()
    if first is None or epoch < first:
        raise ValueError("epoch predates the retirement of champion rewards")
    state = store.get_weight_state("finney", 71, epoch)["state_doc"]
    predecessor = store.get_weight_state("finney", 71, first - 1)["state_doc"]
    for doc, expected_epoch in ((state, epoch), (predecessor, first - 1)):
        verify_accepted_weight_state_signature(doc, public_key_der=signer.public_key_der,
                                               expected_public_key_hash=signer.public_key_hash)
        if (doc["network"], doc["netuid"], doc["epoch"], doc["genesis_hash"]) != (
                "finney", 71, expected_epoch, parent.network_genesis_hash.removeprefix("0x")):
            raise ValueError("signed distribution state belongs to another epoch")
    basis = state["reward_basis"]
    issued_at = datetime.fromisoformat(state["issued_at"].replace("Z", "+00:00"))
    parent_time = datetime.fromisoformat(parent.observed_at.replace("Z", "+00:00"))
    if (not state["valid_from_block"] <= boundary - 1 <= state["valid_until_block"]
            or issued_at > parent_time
            or basis["schema_version"] != COLLECTION_BASIS_SCHEMA or state["burn_hotkey"] != predecessor["burn_hotkey"]):
        raise ValueError("inapplicable signed collection state")
    ledger = db.rpc("collection_reward_quotes_v1", {"p_epoch": epoch})
    if ledger != {k: v for k, v in basis.items() if k != "valuation"}:
        raise ValueError("closed signed quotes differ from accepted jobs")
    parent_keys = registry_at(rpc, boundary - 1, parent_hash)
    keys = registry_at(rpc, boundary, block_hash)
    if keys != parent_keys:
        raise ValueError("UID registry changed at the distribution boundary")
    expected = derive_arena_weights(state, keys)
    checks = {where: validator_vectors(rpc, h, keys, epoch, expected["weights_hash"], validators)
              for where, h in (("parent", parent_hash), ("boundary", block_hash))}
    events = miner_events(graph.client.get_events(block_hash), len(keys))
    event_index, amounts = events[0]
    actual = {keys[i]: amount for i, amount in enumerate(amounts) if amount}
    target = {keys[i]: w for i, w in zip(expected["sparse_uids"], expected["sparse_weights_u16"])}
    quotes = basis["quotes_microusd"]
    burn = state["burn_hotkey"]
    other_mechanisms = {str(m): sum(values) for m, (_, values) in events.items() if m}
    every_vector_matches = all(v == "matches" for side in checks.values() for v in side.values())
    recipients_match = (bool(actual) and set(actual) == set(target) and set(quotes) <= set(actual)
                        and not any(other_mechanisms.values()) and expected["champion_share_ppb"] == 0)
    if not quotes:
        recipients_match = (recipients_match and actual.keys() == {burn} and target == {burn: 65535}
                            and basis["job_count"] == 0 and basis["job_set_sha256"] == sha256(b"").hexdigest())
    result = "inspect"
    if every_vector_matches and recipients_match:
        result = "collection_recipients_observed" if quotes else "burn_only"
    return {"epoch": epoch, "boundary": boundary, "parent_hash": parent_hash, "block_hash": block_hash,
            "state_hash": state["state_hash"], "closed_at": closed.observed_at, "event_index": event_index,
            "job_count": basis["job_count"], "job_set_sha256": basis["job_set_sha256"], "quotes_microusd": quotes,
            "burn_hotkey": burn, "expected_weights_u16": target, "validators": checks,
            "miner_emissions_rao": actual, "other_mechanism_miner_rao": other_mechanisms, "result": result}


def persist_receipt(directory, epoch, receipt):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{epoch}.json"
    data = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with NamedTemporaryFile(dir=directory, delete=False) as stream:
        scratch = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(scratch, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError("closed distribution receipt changed")
        finally:
            scratch.unlink()
    return sha256(data).hexdigest()


def observe(env, db, directory, previous, weights, validators):
    """Record at most one closed epoch per poll, including catch-up after a missed poll."""
    cutover = chain.load_arena_cutover(env)
    store = ArenaStore(db)
    first = weights["weight_start_epoch"] if weights and "weight_start_epoch" in weights else store.collection_reward_start_epoch()
    if first is None:
        if previous.get("epoch") is not None:
            raise ValueError("recorded retirement epoch is unavailable")
        return {"result": "before_retirement"}
    archive = None

    def archive_chain():
        nonlocal archive
        if archive is None:
            config = chain.ArenaChainConfig(OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT, 71, "finney", 20)
            archive = chain.ArenaChain(config, chain.connect_substrate(config))
        return archive

    try:
        if weights:
            latest = weights["epoch"] - 1
            latest_boundary = weights["last_epoch_block"]
        else:
            latest_snapshot = chain.finalized_epoch_snapshot(archive_chain())
            latest = latest_snapshot.settlement_epoch_id(cutover) - 1
            latest_boundary = latest_snapshot.last_epoch_block
        if latest < first:
            return {"result": "before_retirement"}

        receipt_dir = directory / "distributions"
        last = previous.get("epoch")
        if last is not None and latest < last:
            raise ValueError("finalized settlement epoch moved backwards")
        if latest == last:
            raw = (receipt_dir / f"{last}.json").read_bytes()
            if previous["boundary"] != latest_boundary or sha256(raw).hexdigest() != previous["receipt_sha256"]:
                raise ValueError("stored closed distribution differs")
            return previous

        epoch = latest if last is None else last + 1
        boundary = latest_boundary
        if epoch < latest:
            # The following signed state supplies a candidate boundary. receipt_at
            # verifies both adjacent subnet epochs independently on the archive.
            following = store.get_weight_state("finney", 71, epoch + 1)
            if not following:
                raise ValueError("missing following signed epoch during catch-up")
            boundary = following["state_doc"]["valid_from_block"]
        signer = signing.KmsSigner(env["LAB_ARENA_SIGNING_KEY_ID"], region_name="us-east-1")
        rpc = HttpsJsonRpcTransport(OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT, timeout_seconds=18)
        receipt = receipt_at(archive_chain(), rpc, db, signer, cutover, boundary, epoch, validators)
        digest = persist_receipt(receipt_dir, epoch, receipt)
        return {"epoch": epoch, "boundary": boundary, "result": receipt["result"], "receipt_sha256": digest,
                "epochs_to_catch_up": latest - epoch}
    finally:
        if archive is not None:
            archive.close()
